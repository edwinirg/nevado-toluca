# -*- coding: utf-8 -*-
"""
Obtener_CD_EVI.py
Calcula EVI mensual (S2 L2A) para un AOI (GeoJSON), enmascara con SCL, exporta:
 - GeoTIFF mensual (mediana) por fecha
 - CSV de estadísticas por celdas 50×50 m (mean, std, count)
Requisitos: openeo, numpy, rasterio, xarray, pandas
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, List
from openeo.processes import median as pmedian

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

import openeo
from openeo.processes import median as pmedian

# ============== CONFIG ==============
SERVICE_URL = "openeo.dataspace.copernicus.eu"
COLLECTION = "SENTINEL2_L2A"
# Fechas (ajusta a tu caso)
TEMPORAL_EXTENT = ["2019-01-01", "2019-03-30"]

# AOI GeoJSON (Feature/FeatureCollection/Geometry)
# Ejemplos:
# AOI_PATH = r"C:\Users\samsu\Escritorio\Final script\Toluca.json"
# AOI_PATH = r"C:\Users\samsu\Escritorio\Final script\Bosques con muerdago 2.json"
AOI_PATH = r"C:\Users\samsu\Escritorio\Final script\GEOJSONs\Toluca.json"

# Resolución objetivo y proyección (S2 10 m; 50 m = 5 px)
TARGET_RES = 10  # metros
TARGET_CRS = None  # None => nativo del backend. Si quieres forzar: "EPSG:32614" (UTM zona 14N)

# Carpeta de salida
OUT_DIR = Path(r"C:\Users\samsu\Escritorio\Final script\outputs_evi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====================================


def read_geojson(path: str) -> Tuple[Dict[str, Any], str]:
    """Lee GeoJSON y regresa (geometry, nombre_sugerido)."""
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    props = {}
    if gj.get("type") == "Feature":
        geom = gj["geometry"]
        props = gj.get("properties", {})
    elif gj.get("type") == "FeatureCollection":
        if not gj.get("features"):
            raise ValueError("FeatureCollection vacío.")
        feat = gj["features"][0]
        geom = feat["geometry"]
        props = feat.get("properties", {})
    else:
        geom = gj

    name = props.get("NOMBRE") or props.get("name") or Path(path).stem
    return geom, name


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()


def build_cloud_mask(scl_cube) -> Any:
    """
    Máscara: Sombra(3), Nubes(8,9), Nieve/Hielo(10) con ligera dilatación.
    """
    # (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    m = (scl_cube == 3) | (scl_cube == 8) | (scl_cube == 9) | (scl_cube == 10)

    # dilatación con kernel 3x3 (apertura ligera)
    kernel = [
        [0.05, 0.1, 0.05],
        [0.1,  0.4, 0.1 ],
        [0.05, 0.1, 0.05],
    ]
    # m = m.apply_kernel(kernel=kernel)
    # Umbral para binarizar
    # m = m > 0.1
    return m


def compute_evi_cube(cube) -> Any:
    """
    EVI (MODIS-like): G=2.5, L=1, C1=6, C2=7.5
    Requiere B02 (blue), B04 (red), B08 (nir)
    """
    nir = cube.band("B08")
    red = cube.band("B04")
    blue = cube.band("B02")
    G, L, C1, C2 = 2.5, 1.0, 6.0, 7.5
    evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-6)
    # Renombrar banda a 'EVI' si procede
    try:
        evi = evi.rename_labels(dimension="bands", target=["EVI"])
    except Exception:
        pass
    return evi


def month_range(ini: str, fin: str) -> List[str]:
    """Genera 'YYYY-MM' entre dos fechas ISO (incluye bordes)."""
    d0 = datetime.fromisoformat(ini[:10].replace("/", "-"))
    d1 = datetime.fromisoformat(fin[:10].replace("/", "-"))
    months = []
    y, m = d0.year, d0.month
    while (y < d1.year) or (y == d1.year and m <= d1.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def save_monthly_evi_tiff(evi_cube, month: str, out_folder: Path) -> Path:
    """
    Filtra un mes (YYYY-MM), reduce temporalmente por mediana (BAP mensual),
    guarda GeoTIFF, retorna path.
    """
    y_str, m_str = month.split("-")
    y = int(y_str); m = int(m_str)

    # Inicio del mes (YYYY-MM-01)
    first = f"{y:04d}-{m:02d}-01"

    # Primer día del mes siguiente (para usar como límite abierto)
    if m == 12:
        last = f"{y+1:04d}-01-01"
    else:
        last = f"{y:04d}-{m+1:02d}-01"

    # 1) Filtra por el mes
    monthly = evi_cube.filter_temporal([first, last])

    # 2) Reduce el eje temporal con mediana (equivalente a BAP mensual)
    #    (Evita aggregate_temporal si tu cliente exige 'intervals')
    monthly = monthly.reduce_dimension(dimension="t", reducer=pmedian)

    # 3) Exporta
    tiff_path = out_folder / f"EVI_{y:04d}-{m:02d}.tif"
    export = monthly.save_result(
        format="GTiff",
        options={"tiled": True, "compress": "LZW", "bigtiff": "YES"}
    )
    export.download(str(tiff_path))
    return tiff_path



def stats_50m_from_tif(tif_path: Path, cell_size_m: int = 50) -> pd.DataFrame:
    """
    Calcula estadísticas por celdas 50×50 m a partir del GeoTIFF de EVI.
    Asume resolución 10 m → celda = 5×5 píxeles.
    """
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan

        # Clamp suave a rango razonable EVI
        arr[~np.isfinite(arr)] = np.nan
        arr[(arr < -1.5) | (arr > 2.5)] = np.nan

        # Checar resolución ~10 m (si no, ajustar factor dinámicamente)
        # ancho píxel ~ scale derivada de transform (metros/píxel en UTM)
        transform: Affine = src.transform
        px_w = abs(transform.a)
        px_h = abs(transform.e)

        # factor por dimensión (redondeo a entero >= 1)
        fx = max(1, int(round(cell_size_m / px_w)))
        fy = max(1, int(round(cell_size_m / px_h)))

        # cortar array a múltiplos de fx, fy
        nrows = (arr.shape[0] // fy) * fy
        ncols = (arr.shape[1] // fx) * fx
        arr = arr[:nrows, :ncols]

        # reshape por bloques y promedios/STD/COUNT
        # Estructura: (n_bloques_y, fy, n_bloques_x, fx)
        by = nrows // fy
        bx = ncols // fx
        reshaped = arr.reshape(by, fy, bx, fx)

        mean = np.nanmean(reshaped, axis=(1, 3))
        std = np.nanstd(reshaped, axis=(1, 3))
        cnt = np.sum(np.isfinite(reshaped), axis=(1, 3))

        # Coordenadas de centroide aproximado de cada celda
        # fila/columna del píxel superior-izquierdo de cada bloque
        rows = (np.arange(by) * fy) + fy / 2.0
        cols = (np.arange(bx) * fx) + fx / 2.0
        grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
        # Convertir a coords reales con transform
        xs, ys = rasterio.transform.xy(transform, grid_r, grid_c)

        df = pd.DataFrame({
            "cell_row": np.repeat(np.arange(by), bx),
            "cell_col": np.tile(np.arange(bx), by),
            "x_center": np.array(xs).ravel(),
            "y_center": np.array(ys).ravel(),
            "EVI_mean": mean.ravel(),
            "EVI_std": std.ravel(),
            "valid_count": cnt.ravel().astype(int),
            "cell_w_px": fx,
            "cell_h_px": fy,
            "cell_w_m": fx * px_w,
            "cell_h_m": fy * px_h,
        })
        return df


def main():
    # 1) AOI
    geom, aoi_name = read_geojson(AOI_PATH)
    aoi_safe = sanitize(aoi_name)
    out_aoi = OUT_DIR / aoi_safe
    out_aoi.mkdir(parents=True, exist_ok=True)

    print(f"AOI: {aoi_name} ({AOI_PATH})")

    # 2) Conexión
    print("Autenticando con openEO…")
    try:
        conn = openeo.connect(SERVICE_URL).authenticate_oidc()
    except Exception:
        conn = openeo.connect(SERVICE_URL)
        conn.authenticate_oidc_client_credentials(
            client_id="cdse-public", client_secret="cdse-public"
        )
    print("Conectado.")

    # 3) Carga colección + máscara nubes + EVI
    cube = conn.load_collection(
        COLLECTION,
        temporal_extent=TEMPORAL_EXTENT,
        bands=["B02", "B04", "B08", "SCL"],  # Blue, Red, NIR, SCL
    ).filter_spatial(geometries=geom)

    scl = cube.band("SCL")
    cloud_mask = build_cloud_mask(scl)

    # Aplicar máscara
    cube_masked = cube.mask(cloud_mask)

    # EVI
    evi = compute_evi_cube(cube_masked)

    # reproyección (opcional)
    if TARGET_CRS or TARGET_RES:
        evi = evi.resample_spatial(
            resolution=TARGET_RES,
            projection=TARGET_CRS if TARGET_CRS else None,
            method="near"
        )

    # 4) Meses dentro del rango
    months = month_range(TEMPORAL_EXTENT[0], TEMPORAL_EXTENT[1])
    print(f"Meses a procesar: {months}")

    # 5) Loop mensual: mediana y export GeoTIFF + CSV de celdas 50 m
    summary_rows = []
    for mm in months:
        print(f" - Procesando {mm} …")
        try:
            tif_path = save_monthly_evi_tiff(evi, mm, out_aoi)
            print(f"   GeoTIFF: {tif_path}")

            df_cells = stats_50m_from_tif(tif_path, cell_size_m=50)
            csv_cells = out_aoi / f"EVI_50m_{mm}.csv"
            df_cells.to_csv(csv_cells, index=False, encoding="utf-8")
            print(f"   CSV 50m: {csv_cells} (celdas: {len(df_cells)})")

            # para resumen general
            summary_rows.append({
                "month": mm,
                "tif_path": str(tif_path),
                "csv_50m_path": str(csv_cells),
                "valid_cells": int((df_cells["valid_count"] > 0).sum()),
                "EVI_mean_overall": float(df_cells["EVI_mean"].mean(skipna=True)),
            })
        except Exception as e:
            print(f"   [WARN] Mes {mm} falló: {e}")

    # 6) Resumen por mes
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        sum_path = out_aoi / "EVI_resumen_mensual.csv"
        df_sum.to_csv(sum_path, index=False, encoding="utf-8")
        print(f"\nResumen mensual: {sum_path}")

    print("\n¡Listo!")


if __name__ == "__main__":
    main()
