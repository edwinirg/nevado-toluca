# -*- coding: utf-8 -*-
"""
Obtiene SOLO el indice RVI (Sentinel-1 VV/VH) por muestras temporales.

Flujo:
1) Descarga VV y VH lineales (mediana temporal) desde openEO.
2) Calcula RVI = 4*VH/(VV+VH).
3) Guarda RVI raster por muestra y estadisticas por celdas de 50x50 m.

Requiere: openeo, rasterio, numpy, pandas, shapely, geopandas, pyproj
"""

import os
import json
from datetime import date

import geopandas as gpd
import numpy as np
import openeo
import pandas as pd
import pyproj
import rasterio
from dateutil.relativedelta import relativedelta
from rasterio.features import rasterize
from shapely.geometry import box, mapping, shape
from shapely.ops import transform, unary_union

# ========= PARÁMETROS =========
geojson_files = {
    "Bosques_densos_pino_oyamel": "/Users/edwin/Desktop/Article/nevado-toluca/GeoJSONs/Bosques densos de pino y oyamel.json",
    "Toluca": "/Users/edwin/Desktop/Article/nevado-toluca/GeoJSONs/Toluca.json",
    "Bosques_con_muerdago_2": "/Users/edwin/Desktop/Article/nevado-toluca/GeoJSONs/Bosques con muerdago 2.json",
    "Bosques_con_muerdago_3": "/Users/edwin/Desktop/Article/nevado-toluca/GeoJSONs/Bosques con muerdago 3.json",
}

out_base = "salidas_rvi"
os.makedirs(out_base, exist_ok=True)

TARGET_CRS = "EPSG:32614"
TARGET_RES = 10

fecha_inicio = date(2019, 3, 1)
fecha_final = date(2019, 3, 30)

MAX_RETRY = 1
RETRY_SHIFT_DAYS = 4


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
    return {
        "west": float(minx - buffer_deg),
        "south": float(miny - buffer_deg),
        "east": float(maxx + buffer_deg),
        "north": float(maxy + buffer_deg),
    }


def load_polygons_from_geojson(paths_dict):
    zonas = {}
    for nombre, path in paths_dict.items():
        gdf = gpd.read_file(path)
        gdf = gdf.explode(index_parts=False)
        zonas[nombre] = gdf.unary_union
    return zonas


def transform_geom(geom, src_crs, dst_crs):
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)


def make_fishnet_in_geom(geom_proj, cell_size=50.0):
    minx, miny, maxx, maxy = geom_proj.bounds

    def align_down(v, s):
        return np.floor(v / s) * s

    def align_up(v, s):
        return np.ceil(v / s) * s

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


def stats_from_array_grouped_by_labels(arr, labels, prefix):
    valid = np.isfinite(arr) & (labels >= 0)
    if not np.any(valid):
        return pd.DataFrame(
            columns=[
                "Celda_ID",
                f"{prefix}_mean",
                f"{prefix}_std",
                "Pixeles_validos",
                "Pct_validos",
            ]
        )

    v = arr[valid].astype(np.float64)
    lab = labels[valid].astype(np.int64)
    max_label = lab.max()

    counts = np.bincount(lab, minlength=max_label + 1)
    sums = np.bincount(lab, weights=v, minlength=max_label + 1)
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts > 0,
    )

    sums2 = np.bincount(lab, weights=v * v, minlength=max_label + 1)
    ex2 = np.divide(
        sums2,
        counts,
        out=np.full_like(sums2, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    vars_ = ex2 - means * means
    vars_[vars_ < 0] = 0
    stds = np.sqrt(vars_)

    pct_valid = np.full_like(means, np.nan, dtype=np.float64)
    pct_valid[counts > 0] = 100.0

    df = pd.DataFrame(
        {
            "Celda_ID": np.arange(max_label + 1, dtype=np.int64),
            f"{prefix}_mean": means,
            f"{prefix}_std": stds,
            "Pixeles_validos": counts,
            "Pct_validos": pct_valid,
        }
    )
    return df[df["Pixeles_validos"] > 0].reset_index(drop=True)


def build_month_samples(anio, mes):
    dmax = 28 if mes == 2 else 30
    return [
        (date(anio, mes, 1), date(anio, mes, 10)),
        (date(anio, mes, 11), date(anio, mes, 20)),
        (date(anio, mes, 21), date(anio, mes, dmax)),
    ]


def read_float(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        profile = src.profile.copy()
        transform_affine = src.transform
        out_shape = (src.height, src.width)
        bounds = box(*src.bounds)
    return arr, profile, transform_affine, out_shape, bounds


def save_raster(path, arr, profile):
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype("float32"), 1)


conexion = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

bbox_total = bbox_from_geojsons(geojson_files, buffer_deg=0.01)
zonas_wgs84 = load_polygons_from_geojson(geojson_files)

fishnets = {}
for zona_nombre, geom_wgs84 in zonas_wgs84.items():
    geom_proj = transform_geom(geom_wgs84, "EPSG:4326", TARGET_CRS)
    cells = make_fishnet_in_geom(geom_proj, cell_size=50.0)
    fishnets[zona_nombre] = cells
    print(f"{zona_nombre}: {len(cells)} celdas de 50x50 m")


fecha_actual = date(fecha_inicio.year, fecha_inicio.month, 1)
resumen_rows = []

while fecha_actual <= fecha_final:
    anio = fecha_actual.year
    mes = fecha_actual.month
    nombre_mes = fecha_actual.strftime("%m_%B").lower()

    for idx, (ini, fin) in enumerate(build_month_samples(anio, mes), start=1):
        print(f"Procesando {anio}-{mes:02d} muestra {idx}: {ini} a {fin}")

        sub_out = os.path.join(out_base, f"{anio}", nombre_mes, f"muestra_{idx}")
        os.makedirs(sub_out, exist_ok=True)

        s1_vv_tif_lin = os.path.join(sub_out, "S1_VV_lin_BAP.tif")
        s1_vh_tif_lin = os.path.join(sub_out, "S1_VH_lin_BAP.tif")
        out_rvi_tif = os.path.join(sub_out, "RVI_BAP.tif")
        out_rvi_cells_csv = os.path.join(sub_out, "rvi_celdas50m.csv")

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

                vv_med = cube.band("VV").reduce_dimension(dimension="t", reducer="median")
                vh_med = cube.band("VH").reduce_dimension(dimension="t", reducer="median")

                vv_med = vv_med.resample_spatial(
                    resolution=TARGET_RES,
                    projection=TARGET_CRS,
                    method="near",
                )
                vh_med = vh_med.resample_spatial(
                    resolution=TARGET_RES,
                    projection=TARGET_CRS,
                    method="near",
                )

                save_opts = {"tiled": True, "compress": "LZW", "bigtiff": "YES"}
                vv_med.save_result(format="GTiff", options=save_opts).download(s1_vv_tif_lin)
                vh_med.save_result(format="GTiff", options=save_opts).download(s1_vh_tif_lin)
                downloaded = True
            except Exception as e:
                print(f"OpenEO intento {attempt + 1} fallo: {e}")
                attempt += 1
                if attempt <= MAX_RETRY:
                    ini_t += relativedelta(days=RETRY_SHIFT_DAYS)
                    fin_t += relativedelta(days=RETRY_SHIFT_DAYS)

        if not downloaded:
            print("No se pudo descargar S1 para esta muestra; se omite.")
            continue

        vv_lin_arr, profile, transform_affine, out_shape, rbounds = read_float(s1_vv_tif_lin)
        vh_lin_arr, _, _, _, _ = read_float(s1_vh_tif_lin)

        eps = 1e-10
        rvi_arr = (4.0 * vh_lin_arr) / (vv_lin_arr + vh_lin_arr + eps)
        rvi_arr[~np.isfinite(rvi_arr)] = np.nan
        save_raster(out_rvi_tif, rvi_arr, profile)

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

        if not shapes:
            print("No hay celdas que intersecten el raster RVI de esta muestra.")
            continue

        labels = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform_affine,
            fill=-1,
            dtype="int32",
            all_touched=False,
        )

        df_rvi = stats_from_array_grouped_by_labels(rvi_arr, labels, prefix="RVI")
        meta_df = pd.DataFrame(
            meta_rows,
            columns=["Celda_ID", "Subzona", "Celda_ID_local", "Centroide_X", "Centroide_Y"],
        )

        out_df = meta_df.merge(df_rvi, on="Celda_ID", how="left")
        out_df.insert(0, "Año", anio)
        out_df.insert(1, "Mes", mes)
        out_df.insert(2, "Muestra", idx)
        out_df.to_csv(out_rvi_cells_csv, index=False, encoding="utf-8")

        resumen_rows.append(
            {
                "Año": anio,
                "Mes": mes,
                "Muestra": idx,
                "Pixeles_RVI_validos": int(np.isfinite(rvi_arr).sum()),
                "RVI_media_raster": float(np.nanmean(rvi_arr)),
                "RVI_std_raster": float(np.nanstd(rvi_arr)),
                "Salida_RVI_TIF": out_rvi_tif,
                "Salida_RVI_CSV": out_rvi_cells_csv,
            }
        )

        print(f"Guardado RVI raster: {out_rvi_tif}")
        print(f"Guardado RVI por celda: {out_rvi_cells_csv}")

    fecha_actual += relativedelta(months=1)


if resumen_rows:
    resumen_df = pd.DataFrame(resumen_rows)
    resumen_path = os.path.join(out_base, "resumen_rvi_muestras.csv")
    resumen_df.to_csv(resumen_path, index=False, encoding="utf-8")
    print(f"Resumen guardado: {resumen_path}")
else:
    print("No se generaron salidas RVI.")
