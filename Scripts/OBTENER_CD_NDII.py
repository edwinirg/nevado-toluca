import os
import time
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
from sentinelhub import SHConfig
import scipy.signal


# config = SHConfig()
# config.instance_id = 'NevadoToluca'
# config.sh_client_id = 'sh-9f06507f-ee3e-4c4e-863e-685b1c9b7f99'
# config.sh_client_secret = 'm63nuotACS2LuuoHcftd5mxa2TvhT61I'

# Conexión OpenEO
conexion = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc_client_credentials(
    client_id="TU_CLIENT_ID",
    client_secret="TU_CLIENT_SECRET"
)

# Rutas GeoJSON
geojson_files = {
    "Bosques_densos_pino_oyamel": "C:\\Users\samsu\Escritorio\Bosques densos de pino y oyamel.json",
    "Mariposa_monarca": "C:\\Users\samsu\Escritorio\Mariposa monarca.json",
    "Parque_venados": "C:\\Users\samsu\Escritorio\Parque venados.json",
    "Toluca": "C:\\Users\samsu\Escritorio\Toluca.json",
    "Bosques_con_muerdago_2": "C:\\Users\samsu\Escritorio\Bosques con muerdago 2.json",
    "Bosques_con_muerdago_3": "C:\\Users\samsu\Escritorio\Bosques con muerdago 3.json",
}

output_base = "ndvi_total_y_zonas_indices_opt"
os.makedirs(output_base, exist_ok=True)

# Fija CRS/Resolución objetivo

TARGET_CRS = "EPSG:32614"
TARGET_RES = 10  # metros
RGB_ENABLE = False  #  NDVI_RGB

# funciones

def bbox_from_geojsons(paths_dict, buffer_deg=0.01):
    geoms = []
    for path in paths_dict.values():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") == "FeatureCollection":
            for feat in data["features"]:
                geoms.append(shape(feat["geometry"]))
        elif data.get("type") == "Feature":
            geoms.append(shape(data["geometry"]))
        else:
            geoms.append(shape(data))
    uni = unary_union(geoms)
    minx, miny, maxx, maxy = uni.bounds
    return {"west": float(minx - buffer_deg),
            "south": float(miny - buffer_deg),
            "east": float(maxx + buffer_deg),
            "north": float(maxy + buffer_deg)}

def load_polygons_from_geojson(paths_dict):
    zonas = {}
    for nombre, path in paths_dict.items():
        gdf = gpd.read_file(path)
        gdf = gdf.explode(index_parts=False)
        geom = gdf.unary_union
        zonas[nombre] = geom
    return zonas

def transform_geom(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)

def make_fishnet_in_geom(geom_proj, cell_size=50.0):
    minx, miny, maxx, maxy = geom_proj.bounds
    def align_down(v, s): return np.floor(v / s) * s
    def align_up(v, s):   return np.ceil(v / s) * s
    minx = align_down(minx, cell_size)
    miny = align_down(miny, cell_size)
    maxx = align_up(maxx, cell_size)
    maxy = align_up(maxy, cell_size)
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    cells = []
    cid = 0
    for x in xs:
        for y in ys:
            cell = box(x, y, x + cell_size, y + cell_size)
            if cell.intersects(geom_proj):
                inter = cell.intersection(geom_proj)
                if not inter.is_empty and inter.area > 0:
                    cells.append((cid, inter))
                    cid += 1
    return cells

def stats_from_array_grouped_by_labels(ndvi, labels):

    valid = ~np.isnan(ndvi) & (labels >= 0)
    if not np.any(valid):
        return pd.DataFrame(columns=["Celda_ID","NDVI_mean","NDVI_std","Pixeles_validos","Pct_validos"])

    v = ndvi[valid].astype(np.float64)
    lab = labels[valid].astype(np.int64)

    # Sumas y conteos por label
    max_label = lab.max()
    counts = np.bincount(lab, minlength=max_label+1)
    sums = np.bincount(lab, weights=v, minlength=max_label+1)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64), where=counts>0)

    # Varianza: E[x^2] - (E[x])^2
    sums2 = np.bincount(lab, weights=v*v, minlength=max_label+1)
    ex2 = np.divide(sums2, counts, out=np.full_like(sums2, np.nan, dtype=np.float64), where=counts>0)
    vars_ = ex2 - means*means
    vars_[vars_ < 0] = 0  # numérico
    stds = np.sqrt(vars_)

    # Porcentaje válidos por label sobre tamaño total de cada celda en píxeles:
    # Nota: como las celdas pueden recortarse en bordes, usamos "counts" como denom.
    pct_valid = np.full_like(means, np.nan, dtype=np.float64)
    pct_valid[counts > 0] = 100.0  # en la máscara valid ya excluimos NaN/nodata

    df = pd.DataFrame({
        "Celda_ID": np.arange(max_label+1, dtype=np.int64),
        "NDVI_mean": means,
        "NDVI_std": stds,
        "Pixeles_validos": counts,
        "Pct_validos": pct_valid
    })
    # Filtra celdas sin píxeles válidos
    df = df[df["Pixeles_validos"] > 0].reset_index(drop=True)
    return df

def ndvi_to_rgb(ndvi, p_low=2, p_high=98):
    ndvi = ndvi.copy()
    mask_nan = np.isnan(ndvi)
    ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan
    vals = ndvi[~np.isnan(ndvi)]
    if vals.size == 0:
        return (np.zeros_like(ndvi, dtype=np.uint8),)*3
    vmin = np.nanpercentile(vals, p_low)
    vmax = np.nanpercentile(vals, p_high)
    if not np.isfinite(vmin): vmin = -0.2
    if not np.isfinite(vmax): vmax = 0.8
    if vmax <= vmin: vmax = vmin + 1e-6

    R = np.zeros_like(ndvi, dtype="float32")
    G = np.zeros_like(ndvi, dtype="float32")
    B = np.zeros_like(ndvi, dtype="float32")

    low = ndvi < -0.2
    R[low], G[low], B[low] = 0.1, 0.2, 0.8
    mid1 = (ndvi >= -0.2) & (ndvi < 0.2)
    R[mid1], G[mid1], B[mid1] = 1.0, 1.0, 0.1
    mid2 = (ndvi >= 0.2) & (ndvi < 0.6)
    R[mid2], G[mid2], B[mid2] = 0.4, 0.9, 0.3
    high = ndvi >= 0.6
    R[high], G[high], B[high] = 0.1, 0.5, 0.1

    R = np.clip(R * 255, 0, 255).astype(np.uint8)
    G = np.clip(G * 255, 0, 255).astype(np.uint8)
    B = np.clip(B * 255, 0, 255).astype(np.uint8)
    R[mask_nan] = G[mask_nan] = B[mask_nan] = 0
    return R, G, B

def save_rgb_geotiff(reference_src, ndvi_array, out_path):
    profile = reference_src.profile.copy()
    profile.update({"count": 3, "dtype": "uint8", "nodata": 0})
    R, G, B = ndvi_to_rgb(ndvi_array)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(R, 1); dst.write(G, 2); dst.write(B, 3)

# ===========================
# 2) PREPARACIÓN DE GEOMETRÍAS
# ===========================

bbox_total = bbox_from_geojsons(geojson_files, buffer_deg=0.01)
print("BBox total:", bbox_total)

zonas_wgs84 = load_polygons_from_geojson(geojson_files)

# ⚡ Preconstruye fishnets por subzona (en CRS objetivo) para reutilizar
zonas_proj = {}
fishnets = {}
for zona_nombre, geom_wgs84 in zonas_wgs84.items():
    geom_proj = transform_geom(geom_wgs84, "EPSG:4326", TARGET_CRS)
    zonas_proj[zona_nombre] = geom_proj
    cells = make_fishnet_in_geom(geom_proj, cell_size=50.0)
    fishnets[zona_nombre] = cells
    print(f"   ▫ {zona_nombre}: {len(cells)} celdas de 50×50 m")

# ===========================
# 3) PROCESAMIENTO POR MES/MUESTRA
# ===========================

fecha_inicio = date(2019, 1, 31)
fecha_final  = date(2019, 12, 31)

# Para minimizar reintentos
MAX_RETRY = 1           # ⚡ 0 o 1 suele bastar si usas mediana
RETRY_SHIFT_DAYS = 4    # ⚡ Pequeño corrimiento temporal

def build_month_samples(año, mes):
    dmax = 28 if mes == 2 else 30
    return [
        (date(año, mes, 1),  date(año, mes, 10)),
        (date(año, mes, 11), date(año, mes, 20)),
        (date(año, mes, 21), date(año, mes, dmax)),
    ]

# ⚡ Kernel gaussiano para nube (si usas BAP adicional en backend)
g = scipy.signal.windows.gaussian(11, std=1.6)
kernel = (np.outer(g, g) / np.outer(g, g).sum()).tolist()

fecha_actual = date(fecha_inicio.year, fecha_inicio.month, 1)

while fecha_actual <= fecha_final:
    año = fecha_actual.year
    mes = fecha_actual.month
    nombre_mes = fecha_actual.strftime("%m_%B").lower()

    muestras = build_month_samples(año, mes)

    for idx, (ini, fin) in enumerate(muestras):
        subcarpeta = os.path.join(output_base, f"{año}", nombre_mes, f"muestra_{idx+1}")
        os.makedirs(subcarpeta, exist_ok=True)
        tif_path = os.path.join(subcarpeta, "NDII_BAP.tif")
        print(f"\n🔄 {año}-{mes:02d} M{idx+1}: {ini} → {fin}")

        # ======================
        # 3.1 OpenEO BACKEND ⚡
        # ======================
        attempt = 0
        downloaded = False
        ini_t, fin_t = ini, fin

        while attempt <= MAX_RETRY and not downloaded:
            try:
                cube = conexion.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=bbox_total,
                    temporal_extent=[ini_t.isoformat(), fin_t.isoformat()],
                    bands=["B04","B08","B11","SCL"],
                )

                # ================= Calcular NDII =================
                # NDII = (NIR - SWIR1) / (NIR + SWIR1) = (B08 - B11) / (B08 + B11)
                nir  = cube.band("B08")
                swir = cube.band("B11")
                ndii = (nir - swir) / (nir + swir)
                
                # Máscara de nubes basada en SCL (3 sombra, 8/9 nubes, 10 cirros)
                scl = cube.band("SCL")
                cloud_mask = ((scl == 3) | (scl == 8) | (scl == 9) | (scl == 10))
                cloud_mask = cloud_mask.apply_kernel(kernel=kernel)
                cloud_mask = cloud_mask > 0.1
                ndii = ndii.mask(cloud_mask)
                
                # Best Available Pixel (mediana temporal)
                ndii_bap = ndii.reduce_dimension(dimension="t", reducer="median")
                
                # Alinea proyección/resolución
                ndii_bap = ndii_bap.resample_spatial(
                    resolution=TARGET_RES,
                    projection=TARGET_CRS,
                    method="near"
                )
                
                # Descargar GeoTIFF
                task = ndii_bap.save_result(format="GTiff", options={"tiled": True, "compress": "LZW", "bigtiff": "YES"})
                task.download(tif_path)  # ojo: más abajo cambiamos el nombre a NDII_BAP.tif
                downloaded = True

            except Exception as e:
                print(f"❌ OpenEO intento {attempt+1}: {e}")
                attempt += 1
                if attempt <= MAX_RETRY:
                    ini_t += relativedelta(days=RETRY_SHIFT_DAYS)
                    fin_t += relativedelta(days=RETRY_SHIFT_DAYS)

        if not downloaded:
            print("⏭️ Saltando muestra (sin NDVI utilizable).")
            continue

        # ======================
        # 3.2 STATS POR CELDA ⚡
        # ======================
        with rasterio.open(tif_path) as src:
            ndii_arr = src.read(1).astype("float32")
            # saneo valores
            ndii_arr[(ndii_arr < -1) | (ndii_arr > 1)] = np.nan
            nd = src.nodata
            if nd is not None:
                ndii_arr[ndii_arr == nd] = np.nan

            # Perfil para rasterizar celdas
            transform_affine = src.transform
            out_shape = (src.height, src.width)

            # Recolecta todas las celdas de todas las subzonas con IDs **globales** por muestra
            # ⚡ Construimos una sola rasterización de etiquetas
            shapes = []
            meta_rows = []
            global_id = 0

            for zona_nombre, cells in fishnets.items():
                # Intersecta con bounds del raster para reducir shapes
                rbounds = box(*src.bounds)
                for cid, geom in cells:
                    inter = geom.intersection(rbounds)
                    if inter.is_empty:
                        continue
                    shapes.append((mapping(inter), global_id))
                    # guardamos metadatos para luego
                    cx, cy = inter.centroid.x, inter.centroid.y
                    meta_rows.append((global_id, zona_nombre, cid, cx, cy))
                    global_id += 1

            if len(shapes) == 0:
                print("⚠️ No hay celdas que intersecten el raster en esta muestra.")
                continue

            # ⚡ Rasterizamos todas las celdas en un solo paso
            labels = rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=transform_affine,
                fill=-1,
                dtype="int32",
                all_touched=False,  # si quieres maximizar cobertura, cambia a True
            )

            # ⚡ Estadísticas vectorizadas por label
            df_stats = stats_from_array_grouped_by_labels(ndii_arr, labels)
            df_stats.rename(columns={"NDVI_mean":"NDII_mean", "NDVI_std":"NDII_std"}, inplace=True)
            # Une con metadatos de cada celda
            meta_df = pd.DataFrame(meta_rows, columns=["Label","Subzona","Celda_ID_local","Centroide_X","Centroide_Y"])
            out_df = df_stats.merge(meta_df, left_on="Celda_ID", right_on="Label", how="left")
            out_df.drop(columns=["Label"], inplace=True)
            out_df.rename(columns={"Celda_ID":"Celda_ID_global"}, inplace=True)
             
            
            # Añade columnas de contexto
            out_df.insert(0, "Año", año)
            out_df.insert(1, "Mes", mes)
            out_df.insert(2, "Muestra", idx+1)

            # Guarda CSV
            out_csv = os.path.join(subcarpeta, f"ndii_celdas50m_muestra{idx+1}.csv")  # ← NDII en el nombre
            out_df.to_csv(out_csv, index=False, encoding="utf-8")
            print(f"📄 Guardado: {out_csv}  (celdas: {len(out_df)})")


            # Opcional: RGB
            if RGB_ENABLE:
                rgb_path = os.path.join(subcarpeta, "NDVI_RGB.tif")
                save_rgb_geotiff(src, ndii_arr, rgb_path)
                print(f"🖼️ Guardado RGB: {rgb_path}")

    fecha_actual += relativedelta(months=1)

print("\n✅ Proceso completo finalizado (optimizado).")
