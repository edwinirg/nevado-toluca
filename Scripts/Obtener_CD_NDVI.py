# -*- coding: utf-8 -*-
"""
Comparación NDVI (Sentinel-2) vs Sentinel-1 (VV/VH/RVI) por celdas 50x50 m,
alineando CRS, resolución y ventanas temporales.

Estrategia:
- openEO: descarga VV/VH en unidades lineales con mediana temporal (sin log allí).
- Python: convierte a dB (= 10*log10) y calcula RVI.
- Reusa fishnet 50x50, rasterize y estadísticas por celda como en tu pipeline.

Requiere: openeo, rasterio, numpy, pandas, shapely, geopandas, pyproj, scipy
"""

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
from scipy.stats import pearsonr, spearmanr

# ========= PARÁMETROS =========
geojson_files = {
    "Bosques_densos_pino_oyamel": r"C:\Users\samsu\Escritorio\Bosques densos de pino y oyamel.json",
    "Toluca": r"C:\Users\samsu\Escritorio\Toluca.json",
    "Bosques_con_muerdago_2": r"C:\Users\samsu\Escritorio\Bosques con muerdago 2.json",
    "Bosques_con_muerdago_3": r"C:\Users\samsu\Escritorio\Bosques con muerdago 3.json",
}

# Carpeta base donde YA tienes NDVI por muestra (estructura de tu script)
# ndvi_base/{YYYY}/{mm_Month}/muestra_{1..3}/NDVI_BAP.tif
ndvi_base   = r"ndvi_total_y_zonas_indices"  # :contentReference[oaicite:3]{index=3}

# Carpeta salida para S1 y comparaciones
out_base    = r"s1_vs_ndvi_comparacion"
os.makedirs(out_base, exist_ok=True)

# CRS/Resolución igual a tu pipeline
TARGET_CRS  = "EPSG:32614"  # :contentReference[oaicite:4]{index=4}
TARGET_RES  = 10            # :contentReference[oaicite:5]{index=5}

# Periodo (usa el mismo rango que NDVI para comparar)
fecha_inicio = date(2019, 1, 1)
fecha_final  = date(2019, 1, 31)

# Reintentos (por si no hay adquisiciones exactas)
MAX_RETRY = 1
RETRY_SHIFT_DAYS = 4

# ========= UTILIDADES GEOM / CELDAS =========
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
    return zonas  # :contentReference[oaicite:6]{index=6}

def transform_geom(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)  # :contentReference[oaicite:7]{index=7}

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
    return cells  # :contentReference[oaicite:8]{index=8}

def stats_from_array_grouped_by_labels(arr, labels, prefix):
    """Media/std/píxeles válidos por etiqueta; `prefix` nombra columnas (NDVI, VV_dB, etc.)."""
    a = arr.copy()
    valid = ~np.isnan(a) & (labels >= 0)
    if not np.any(valid):
        return pd.DataFrame(columns=["Celda_ID", f"{prefix}_mean", f"{prefix}_std", "Pixeles_validos", "Pct_validos"])

    v = a[valid].astype(np.float64)
    lab = labels[valid].astype(np.int64)
    max_label = lab.max()

    counts = np.bincount(lab, minlength=max_label+1)
    sums   = np.bincount(lab, weights=v, minlength=max_label+1)
    means  = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64), where=counts>0)

    sums2 = np.bincount(lab, weights=v*v, minlength=max_label+1)
    ex2   = np.divide(sums2, counts, out=np.full_like(sums2, np.nan, dtype=np.float64), where=counts>0)
    vars_ = ex2 - means*means
    vars_[vars_ < 0] = 0
    stds  = np.sqrt(vars_)

    pct_valid = np.full_like(means, np.nan, dtype=np.float64)
    pct_valid[counts > 0] = 100.0

    df = pd.DataFrame({
        "Celda_ID": np.arange(max_label+1, dtype=np.int64),
        f"{prefix}_mean": means,
        f"{prefix}_std": stds,
        "Pixeles_validos": counts,
        "Pct_validos": pct_valid
    })
    df = df[df["Pixeles_validos"] > 0].reset_index(drop=True)
    return df  # (deriva de tu función NDVI) :contentReference[oaicite:9]{index=9}

def build_month_samples(año, mes):
    dmax = 28 if mes == 2 else 30
    return [
        (date(año, mes, 1),  date(año, mes, 10)),
        (date(año, mes, 11), date(año, mes, 20)),
        (date(año, mes, 21), date(año, mes, dmax)),
    ]  # :contentReference[oaicite:10]{index=10}

# ========= CONEXIÓN OPENE0 Y PREPARACIÓN =========
conexion = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

bbox_total = bbox_from_geojsons(geojson_files, buffer_deg=0.01)
zonas_wgs84 = load_polygons_from_geojson(geojson_files)

# Construye fishnets por subzona (en CRS objetivo) para reutilizar
zonas_proj = {}
fishnets = {}
for zona_nombre, geom_wgs84 in zonas_wgs84.items():
    geom_proj = transform_geom(geom_wgs84, "EPSG:4326", TARGET_CRS)
    zonas_proj[zona_nombre] = geom_proj
    cells = make_fishnet_in_geom(geom_proj, cell_size=50.0)
    fishnets[zona_nombre] = cells
    print(f"   ▫ {zona_nombre}: {len(cells)} celdas de 50×50 m")  # :contentReference[oaicite:11]{index=11}

# ========= BUCLE TEMPORAL =========
fecha_actual = date(fecha_inicio.year, fecha_inicio.month, 1)

all_corr_rows = []

while fecha_actual <= fecha_final:
    año = fecha_actual.year
    mes = fecha_actual.month
    nombre_mes = fecha_actual.strftime("%m_%B").lower()

    muestras = build_month_samples(año, mes)
    for idx, (ini, fin) in enumerate(muestras):
        print(f"\n🔄 {año}-{mes:02d} M{idx+1}: {ini} → {fin}")

        # Rutas NDVI/S1 por muestra
        sub_ndvi   = os.path.join(ndvi_base, f"{año}", nombre_mes, f"muestra_{idx+1}")
        ndvi_tif   = os.path.join(sub_ndvi, "NDVI_BAP.tif")  # :contentReference[oaicite:12]{index=12}

        sub_out    = os.path.join(out_base, f"{año}", nombre_mes, f"muestra_{idx+1}")
        os.makedirs(sub_out, exist_ok=True)

        # Salidas S1 (lineal descargado) y comparativos
        s1_vv_tif_lin  = os.path.join(sub_out, "S1_VV_lin_BAP.tif")
        s1_vh_tif_lin  = os.path.join(sub_out, "S1_VH_lin_BAP.tif")
        out_cells_csv  = os.path.join(sub_out, "comparacion_ndvi_s1_celdas50m.csv")
        out_corr_csv   = os.path.join(sub_out, "correlaciones_por_subzona.csv")

        # A) Descarga S1 en unidades lineales (sin log en openEO)
        attempt = 0
        downloaded = False
        ini_t, fin_t = ini, fin

        while attempt <= MAX_RETRY and not downloaded:
            try:
                cube = conexion.load_collection(
                    "SENTINEL1_GRD",
                    spatial_extent=bbox_total,
                    temporal_extent=[ini_t.isoformat(), fin_t.isoformat()],
                    bands=["VV", "VH"],
                )
                vv = cube.band("VV")
                vh = cube.band("VH")

                # Mediana temporal en lineal
                vv_med = vv.reduce_dimension(dimension="t", reducer="median")
                vh_med = vh.reduce_dimension(dimension="t", reducer="median")

                # Reproyecta/alianea a EPSG:32614 / 10 m (igual que NDVI) :contentReference[oaicite:13]{index=13}
                vv_med = vv_med.resample_spatial(resolution=TARGET_RES, projection=TARGET_CRS, method="near")
                vh_med = vh_med.resample_spatial(resolution=TARGET_RES, projection=TARGET_CRS, method="near")

                # Descargar GeoTIFFs lineales
                vv_task = vv_med.save_result(format="GTiff", options={"tiled": True, "compress": "LZW", "bigtiff": "YES"})
                vh_task = vh_med.save_result(format="GTiff", options={"tiled": True, "compress": "LZW", "bigtiff": "YES"})
                vv_task.download(s1_vv_tif_lin)
                vh_task.download(s1_vh_tif_lin)
                downloaded = True

            except Exception as e:
                print(f" OpenEO intento {attempt+1}: {e} :(")
                attempt += 1
                if attempt <= MAX_RETRY:
                    ini_t += relativedelta(days=RETRY_SHIFT_DAYS)
                    fin_t += relativedelta(days=RETRY_SHIFT_DAYS)

        if not downloaded:
            print("⚠️ S1 inservible para esta muestra; se omite comparación.")
            continue

        # B) Lee NDVI y arma etiquetas por celdas 50x50 (reutiliza tu flujo) :contentReference[oaicite:14]{index=14}
        if not os.path.exists(ndvi_tif):
            print(f"⚠️ No se encontró {ndvi_tif}. Se omite esta muestra.")
            continue

        with rasterio.open(ndvi_tif) as ndsrc:
            ndvi_arr = ndsrc.read(1).astype("float32")
            nd = ndsrc.nodata
            if nd is not None:
                ndvi_arr[ndvi_arr == nd] = np.nan
            ndvi_arr[(ndvi_arr < -1) | (ndvi_arr > 1)] = np.nan
            transform_affine = ndsrc.transform
            out_shape = (ndsrc.height, ndsrc.width)
            rbounds = box(*ndsrc.bounds)

            shapes = []
            meta_rows = []
            global_id = 0
            for zona_nombre, cells in fishnets.items():
                for cid, geom in cells:
                    inter = geom.intersection(rbounds)
                    if inter.is_empty:
                        continue
                    shapes.append((mapping(inter), global_id))
                    cx, cy = inter.centroid.x, inter.centroid.y
                    meta_rows.append((global_id, zona_nombre, cid, cx, cy))
                    global_id += 1

            if len(shapes) == 0:
                print("⚠️ No hay celdas que intersecten el NDVI en esta muestra.")
                continue

            labels = rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=transform_affine,
                fill=-1,
                dtype="int32",
                all_touched=False,
            )

        # C) Leer S1 lineal, convertir a dB y calcular RVI en Python
        def read_float(path):
            with rasterio.open(path) as src:
                a = src.read(1).astype("float32")
                nd = src.nodata
                if nd is not None:
                    a[a == nd] = np.nan
                return a

        vv_lin_arr = read_float(s1_vv_tif_lin)
        vh_lin_arr = read_float(s1_vh_tif_lin)

        eps = 1e-10
        vv_db_arr = 10.0 * np.log10(np.clip(vv_lin_arr, eps, None))
        vh_db_arr = 10.0 * np.log10(np.clip(vh_lin_arr, eps, None))
        rvi_arr   = (4.0 * vh_lin_arr) / (vv_lin_arr + vh_lin_arr + eps)

        # D) Estadísticas por celda (NDVI/SAR)
        df_ndvi = stats_from_array_grouped_by_labels(ndvi_arr, labels, prefix="NDVI")
        df_vv   = stats_from_array_grouped_by_labels(vv_db_arr, labels, prefix="VV_dB")
        df_vh   = stats_from_array_grouped_by_labels(vh_db_arr, labels, prefix="VH_dB")
        df_rvi  = stats_from_array_grouped_by_labels(rvi_arr,   labels, prefix="RVI")

        meta_df = pd.DataFrame(meta_rows, columns=["Celda_ID","Subzona","Celda_ID_local","Centroide_X","Centroide_Y"])
        out_df  = meta_df.merge(df_ndvi[["Celda_ID","NDVI_mean"]], on="Celda_ID", how="left") \
                         .merge(df_vv[["Celda_ID","VV_dB_mean"]], on="Celda_ID", how="left") \
                         .merge(df_vh[["Celda_ID","VH_dB_mean"]], on="Celda_ID", how="left") \
                         .merge(df_rvi[["Celda_ID","RVI_mean"]],   on="Celda_ID", how="left")

        # Contexto muestra
        out_df.insert(0, "Año", año)
        out_df.insert(1, "Mes", mes)
        out_df.insert(2, "Muestra", idx+1)

        # Guarda comparativo por celdas
        out_df.to_csv(out_cells_csv, index=False, encoding="utf-8")
        print(f"📄 Guardado: {out_cells_csv}  (celdas: {len(out_df)})")

        # E) Correlaciones por subzona y global
        def corr_pair(x, y):
            x_ = x.astype(float)
            y_ = y.astype(float)
            m = np.isfinite(x_) & np.isfinite(y_)
            if m.sum() < 3:
                return np.nan, np.nan, m.sum()
            r_p, _ = pearsonr(x_[m], y_[m])
            r_s, _ = spearmanr(x_[m], y_[m])
            return r_p, r_s, m.sum()

        rows = []
        for sub in sorted(out_df["Subzona"].unique()):
            t = out_df[out_df["Subzona"] == sub]
            rp_vv, rs_vv, n1 = corr_pair(t["NDVI_mean"].values, t["VV_dB_mean"].values)
            rp_vh, rs_vh, n2 = corr_pair(t["NDVI_mean"].values, t["VH_dB_mean"].values)
            rp_rv, rs_rv, n3 = corr_pair(t["NDVI_mean"].values, t["RVI_mean"].values)
            rows.append([año, mes, idx+1, sub, n1, rp_vv, rs_vv, n2, rp_vh, rs_vh, n3, rp_rv, rs_rv])

        t = out_df
        rp_vv, rs_vv, n1 = corr_pair(t["NDVI_mean"].values, t["VV_dB_mean"].values)
        rp_vh, rs_vh, n2 = corr_pair(t["NDVI_mean"].values, t["VH_dB_mean"].values)
        rp_rv, rs_rv, n3 = corr_pair(t["NDVI_mean"].values, t["RVI_mean"].values)
        rows.append([año, mes, idx+1, "GLOBAL", n1, rp_vv, rs_vv, n2, rp_vh, rs_vh, n3, rp_rv, rs_rv])

        cols = ["Año","Mes","Muestra","Subzona",
                "N_VV","Pearson_NDVI_VVdB","Spearman_NDVI_VVdB",
                "N_VH","Pearson_NDVI_VHdB","Spearman_NDVI_VHdB",
                "N_RVI","Pearson_NDVI_RVI","Spearman_NDVI_RVI"]
        corr_df = pd.DataFrame(rows, columns=cols)
        corr_df.to_csv(out_corr_csv, index=False, encoding="utf-8")
        print(f"📊 Guardado: {out_corr_csv}")

        # Para resumen global de todo el periodo
        corr_df2 = corr_df[corr_df["Subzona"] == "GLOBAL"].copy()
        corr_df2["Etiqueta_muestra"] = f"{año}-{mes:02d}-M{idx+1}"
        all_corr_rows.append(corr_df2)

    fecha_actual += relativedelta(months=1)

if len(all_corr_rows) > 0:
    resumen_global = pd.concat(all_corr_rows, ignore_index=True)
    resumen_global.to_csv(os.path.join(out_base, "correlaciones_globales_todas_muestras.csv"), index=False, encoding="utf-8")
    print("✅ Resumen global guardado.")
else:
    print("No hubo correlaciones para resumir.")
