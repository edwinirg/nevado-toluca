import os
import json
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box, shape, mapping
from shapely.ops import transform, unary_union
import pyproj
import geopandas as gpd
import openeo

# ========= PARÁMETROS =========
geojson_files = {
    "ZonaCraterRestringida": r"/Users/edwin/Desktop/Article/nevado-toluca/GeoJSONs/ZonaCraterRestringida.json",
}

out_base = r"s2_ndsi_celdas50m"
os.makedirs(out_base, exist_ok=True)

TARGET_CRS = "EPSG:32614"
TARGET_RES = 10  # Procesamiento base a 10m para luego agrupar en celdas de 50m

fecha_inicio = date(2018, 12, 1)
fecha_final  = date(2019, 3, 30)

MAX_RETRY = 1
RETRY_SHIFT_DAYS = 4

# ========= UTILIDADES =========

def bbox_from_geojsons(paths_dict, buffer_deg=0.01):
    geoms = []
    for path in paths_dict.values():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") == "FeatureCollection":
            for feat in data["features"]:
                geoms.append(shape(feat["geometry"]))
        else:
            geoms.append(shape(data))
    uni = unary_union(geoms)
    minx, miny, maxx, maxy = uni.bounds
    return {"west": minx - buffer_deg, "south": miny - buffer_deg, 
            "east": maxx + buffer_deg, "north": maxy + buffer_deg}

def load_polygons_and_fishnet(paths_dict, cell_size=50.0):
    zonas_proj = {}
    fishnets = {}
    for nombre, path in paths_dict.items():
        gdf = gpd.read_file(path).to_crs(TARGET_CRS)
        geom_proj = gdf.unary_union
        zonas_proj[nombre] = geom_proj
        
        # Generar Fishnet
        minx, miny, maxx, maxy = geom_proj.bounds
        xs = np.arange(np.floor(minx/cell_size)*cell_size, maxx, cell_size)
        ys = np.arange(np.floor(miny/cell_size)*cell_size, maxy, cell_size)
        cells = []
        cid = 0
        for x in xs:
            for y in ys:
                cell = box(x, y, x + cell_size, y + cell_size)
                if cell.intersects(geom_proj):
                    cells.append((cid, cell.intersection(geom_proj)))
                    cid += 1
        fishnets[nombre] = cells
    return fishnets

def normalize_s2(arr):
    a = arr.astype("float32")
    # Sentinel-2 L2A suele venir escalado por 10000
    if np.nanmedian(a) > 100: 
        a /= 10000.0
    a[a < 0] = np.nan
    a[a > 1.2] = np.nan # Margen para glint/nubes
    return a

def build_month_samples(año, mes):
    # Definición de 3 ventanas por mes
    import calendar
    last_day = calendar.monthrange(año, mes)[1]
    return [
        (date(año, mes, 1),  date(año, mes, 10)),
        (date(año, mes, 11), date(año, mes, 20)),
        (date(año, mes, 21), date(año, mes, last_day)),
    ]

# ========= PROCESAMIENTO PRINCIPAL =========

conexion = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
bbox_total = bbox_from_geojsons(geojson_files)
fishnets = load_polygons_and_fishnet(geojson_files)

fecha_actual = date(fecha_inicio.year, fecha_inicio.month, 1)

while fecha_actual <= fecha_final:
    año, mes = fecha_actual.year, fecha_actual.month
    nombre_mes = fecha_actual.strftime("%m_%B").lower()
    month_out = os.path.join(out_base, str(año), nombre_mes)
    os.makedirs(month_out, exist_ok=True)

    muestras = build_month_samples(año, mes)
    muestras_validas_paths = []

    for idx, (ini, fin) in enumerate(muestras):
        print(f"--- Procesando {año}-{mes:02d} Muestra {idx+1} ---")
        sub_out = os.path.join(month_out, f"m{idx+1}")
        os.makedirs(sub_out, exist_ok=True)
        
        b03_path = os.path.join(sub_out, "B03.tif")
        b11_path = os.path.join(sub_out, "B11.tif")
        ndsi_path = os.path.join(sub_out, "NDSI.tif")

        # 1. Descarga vía openEO
        try:
            cube = conexion.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=bbox_total,
                temporal_extent=[ini.isoformat(), fin.isoformat()],
                bands=["B03", "B11"]
            )
            # Reducción temporal por mediana para limpiar nubes en el rango de 10 días
            composite = cube.reduce_dimension(dimension="t", reducer="median")
            composite = composite.resample_spatial(resolution=TARGET_RES, projection=TARGET_CRS)
            
            composite.save_result(format="GTiff").download(b03_path) # Nota: Descarga bandas juntas o separado según versión
            # Si descarga un stack, hay que separarlo. Aquí asumimos descarga por banda para claridad:
            # (En algunas versiones de openeo-python-client necesitas save_result por banda)
        except Exception as e:
            print(f" Error en descarga M{idx+1}: {e}")
            continue

        # 2. Cálculo de NDSI
        with rasterio.open(b03_path) as src:
            # openEO a veces descarga todas las bandas en un solo archivo si no se especifica
            if src.count >= 2:
                green = normalize_s2(src.read(1))
                swir = normalize_s2(src.read(2))
            else:
                green = normalize_s2(src.read(1))
                with rasterio.open(b11_path) as src2:
                    swir = normalize_s2(src2.read(1))
            
            profile = src.profile
            # Formula NDSI
            ndsi = (green - swir) / (green + swir + 1e-10)
            ndsi[(ndsi < -1) | (ndsi > 1)] = np.nan
            
            profile.update(dtype="float32", count=1, nodata=np.nan)
            with rasterio.open(ndsi_path, "w", **profile) as dst:
                dst.write(ndsi, 1)
            
            muestras_validas_paths.append(ndsi_path)
            print(f" NDSI M{idx+1} generado.")

    # 3. Combinación Mensual (EL CORAZÓN DEL PROBLEMA)
    if muestras_validas_paths:
        print(f" Combinando {len(muestras_validas_paths)} muestras para el promedio mensual...")
        
        arrays_mes = []
        meta_ref = None
        
        for p in muestras_validas_paths:
            with rasterio.open(p) as s:
                arrays_mes.append(s.read(1))
                if meta_ref is None: meta_ref = s.profile

        # Stack y Promedio real
        stack = np.stack(arrays_mes, axis=0)
        # nanmean calcula el promedio ignorando los NaNs por cada píxel
        # Si un píxel tiene [0.5, NaN, 0.7], el resultado será 0.6
        combined_arr = np.nanmean(stack, axis=0)
        
        combined_tif = os.path.join(month_out, "NDSI_Mensual_Final.tif")
        with rasterio.open(combined_tif, "w", **meta_ref) as dst:
            dst.write(combined_arr.astype("float32"), 1)
        
        print(f" ✅ Archivo combinado creado: {combined_tif}")
    
    fecha_actual += relativedelta(months=1)

print("\n🚀 Proceso finalizado.")