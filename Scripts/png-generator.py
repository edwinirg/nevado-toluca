# -*- coding: utf-8 -*-
"""
Panel (enero 2019–2024) con:
- Tira de thumbnails: PNG generados desde NDVI_RGB.tif (o NDVI_BAP.tif -> coloreado)
- Gráfica de NDVI (con LOESS y líneas guía)

Recorre:
ndvi_total_y_zonas_indices/{YYYY}/01_january/muestra_1/(NDVI_RGB.tif|NDVI_BAP.tif)

Entradas:
- CSV_NDVI: pivot mensual con columnas: year, month, <SUBZONAS...>
- SUBZONA: columna a graficar del CSV_NDVI

Salidas:
- PNGs de thumbnails en THUMB_OUT_DIR
- Figura final OUT_PNG
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess
from PIL import Image
import rasterio

# ===================== CONFIG =====================
# (1) Carpeta con los TIF de enero (por años y muestra_1)
INPUT_BASE   = r"C:\Users\samsu\Escritorio\Final script\Scripts\ndvi_total_y_zonas_indices"
YEARS        = range(2019, 2025)          # 2019..2024
MONTH_DIR    = "01_january"
MUESTRA_DIR  = "muestra_1"
TIF_COLOR    = "NDVI_RGB.tif"             # preferido
TIF_BW       = "NDVI_BAP.tif"             # fallback si no hay NDVI_RGB

# (2) Carpeta donde guardar los PNG para los thumbnails
THUMB_OUT_DIR = Path(r".\png_thumbs_enero_muestra1")
THUMB_OUT_DIR.mkdir(parents=True, exist_ok=True)

# (3) CSV de NDVI pivotado (para gráfica de NDVI)
CSV_NDVI = Path(r"C:\Users\samsu\Escritorio\Final script\Scripts\salidas_cluster_serie\ndvi_mensual_pivot_subzonas_muestra_elegida.csv")

# Subzona exacta a graficar (debe ser una columna del CSV)
SUBZONA = "Bosques_densos_pino_oyamel"

# Rango temporal de la gráfica (puedes limitar a 2019–2024 si quieres)
Y_MIN, Y_MAX = 2016, 2024

# Parámetros visuales
ROLL = 3
HAMPEL_WINDOW = 3
HAMPEL_K = 3.0
LOESS_FRAC = 0.12  # suavizado de la tendencia

# Líneas guía NDVI (texto en español)
NDVI_GUIDES = [(0.6, "Forest"), (0.5, "Shrubland"), (0.4, "Grassland")]

# Gráfico final
OUT_PNG = Path(r".\panel_ndvi_con_thumbs_enero.png")
DPI = 300
# ==================================================


# ---------- utilidades de limpieza/suavizado ----------
def hampel_filter(series: pd.Series, window=3, k=3.0) -> pd.Series:
    x = series.copy()
    med = x.rolling(window=window, center=True, min_periods=1).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
    mad = mad.replace(0, mad[mad != 0].min() if (mad != 0).any() else 1e-9)
    z = (x - med).abs() / (1.4826 * mad)
    x[z > k] = np.nan
    return x

def mk_loess(y: pd.Series, frac=0.12) -> pd.Series | None:
    if y is None or y.dropna().empty:
        return None
    x = y.index.map(pd.Timestamp.toordinal).astype(float).values
    vals = y.values.astype(float)
    m = np.isfinite(vals)
    if m.sum() < 4:
        return None
    lo = lowess(vals[m], x[m], frac=frac, return_sorted=False)
    yhat = pd.Series(index=y.index[m], data=lo)
    yhat = yhat.reindex(y.index).interpolate(limit_direction="both")
    return yhat
# -----------------------------------------------------


# ---------- colorización (fallback si solo hay B/N) ----------
def detect_and_normalize_ndvi(arr: np.ndarray) -> np.ndarray:
    a = arr.astype("float32")
    a[~np.isfinite(a)] = np.nan
    vmin = np.nanpercentile(a, 1)
    vmax = np.nanpercentile(a, 99)
    if vmax <= 1.5 and vmin >= -1.5:
        ndvi = a
    elif (a.dtype == np.uint8) or (vmin >= 0 and vmax <= 255):
        ndvi = (a / 127.5) - 1.0
    elif vmax > 1.5 and vmax <= 10000 + 1:
        ndvi = (a / 10000.0) * 2.0 - 1.0
    else:
        ndvi = (a - vmin) / (vmax - vmin + 1e-6) * 2.0 - 1.0
    ndvi[(ndvi < -1.2) | (ndvi > 1.2)] = np.nan
    return ndvi

def ndvi_to_rgb(ndvi: np.ndarray, p_low=2, p_high=98):
    x = ndvi.copy()
    mask_nan = ~np.isfinite(x)
    x[(x < -1) | (x > 1)] = np.nan

    vals = x[~np.isnan(x)]
    if vals.size == 0:
        return (np.zeros_like(x, dtype=np.uint8),) * 3

    # cortes de color por rangos de NDVI
    R = np.zeros_like(x, dtype="float32")
    G = np.zeros_like(x, dtype="float32")
    B = np.zeros_like(x, dtype="float32")

    low  = x < -0.2
    mid1 = (x >= -0.2) & (x < 0.2)
    mid2 = (x >= 0.2) & (x < 0.6)
    high = x >= 0.6

    R[low],  G[low],  B[low]  = 0.1, 1.0, 1.0   # cian tenue
    R[mid1], G[mid1], B[mid1] = 1.0, 1.0, 0.1   # amarillento
    R[mid2], G[mid2], B[mid2] = 0.4, 0.9, 0.3   # verde
    R[high], G[high], B[high] = 0.1, 0.5, 0.1   # verde oscuro

    R = np.clip(R * 255, 0, 255).astype(np.uint8)
    G = np.clip(G * 255, 0, 255).astype(np.uint8)
    B = np.clip(B * 255, 0, 255).astype(np.uint8)

    R[mask_nan] = 0; G[mask_nan] = 0; B[mask_nan] = 0
    return R, G, B
# -------------------------------------------------------------


# ---------- carga NDVI (CSV pivot) ----------
def load_pivot(csv_path: Path, col: str, y_min: int, y_max: int) -> pd.Series | None:
    if not csv_path or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df = df.set_index("date").sort_index()
    s = df[col].astype(float)
    s = s[(s.index.year >= y_min) & (s.index.year <= y_max)]
    s = hampel_filter(s, window=HAMPEL_WINDOW, k=HAMPEL_K)
    s = s.interpolate(limit_direction="both")
    if ROLL and ROLL > 1:
        s = s.rolling(ROLL, center=True, min_periods=1).mean()
    return s
# ---------------------------------------------


# ---------- generar PNGs de enero (muestra_1) ----------
def export_january_thumbs() -> list[Path]:
    """
    Recorre 2019..2024, toma NDVI_RGB.tif (o NDVI_BAP.tif->colorea) de muestra_1,
    exporta PNG por año en THUMB_OUT_DIR y devuelve lista de rutas generadas (ordenadas por año).
    """
    pngs: list[Path] = []
    for y in YEARS:
        mdir = Path(INPUT_BASE) / str(y) / MONTH_DIR / MUESTRA_DIR
        if not mdir.exists():
            print(f"⚠️ No existe: {mdir}")
            continue

        tif_color = mdir / TIF_COLOR
        tif_bw    = mdir / TIF_BW
        out_png   = THUMB_OUT_DIR / f"{y}_enero_muestra1.png"

        try:
            if tif_color.exists():
                with rasterio.open(tif_color) as src:
                    if src.count >= 3:
                        R = src.read(1).astype(np.uint8)
                        G = src.read(2).astype(np.uint8)
                        B = src.read(3).astype(np.uint8)
                        rgb = np.dstack([R, G, B])
                    else:
                        # raro: un solo canal en NDVI_RGB; lo coloreamos
                        a = src.read(1)
                        ndvi = detect_and_normalize_ndvi(a)
                        R, G, B = ndvi_to_rgb(ndvi)
                        rgb = np.dstack([R, G, B])
            elif tif_bw.exists():
                with rasterio.open(tif_bw) as src:
                    a = src.read(1)
                    nd = src.nodata
                    if nd is not None:
                        a = a.astype("float32")
                        a[a == nd] = np.nan
                    ndvi = detect_and_normalize_ndvi(a)
                    R, G, B = ndvi_to_rgb(ndvi)
                    rgb = np.dstack([R, G, B])
            else:
                print(f"⚠️ No hay {TIF_COLOR} ni {TIF_BW} en {mdir}")
                continue

            Image.fromarray(rgb, mode="RGB").save(out_png)
            pngs.append(out_png)
            print(f"🖼️ PNG: {out_png}")
        except Exception as e:
            print(f"❌ Error en {mdir}: {e}")

    # orden por año
    pngs = sorted(pngs, key=lambda p: p.name)
    return pngs
# ---------------------------------------------------------


def load_thumbnails_from_list(paths: list[Path], max_imgs=8) -> list[Image.Image]:
    """Carga hasta max_imgs imágenes desde rutas dadas y devuelve PIL Images."""
    thumbs: list[Image.Image] = []
    for p in paths[:max_imgs]:
        try:
            im = Image.open(p).convert("RGB")
            thumbs.append(im)
        except Exception as e:
            print(f"⚠️ No se pudo abrir {p}: {e}")
    return thumbs


def plot_panel(ndvi: pd.Series | None, thumbs: list[Image.Image], title_note: str = ""):
    import matplotlib as mpl
    mpl.rcParams['image.interpolation'] = 'nearest'
    mpl.rcParams['image.resample'] = False

    # ---- Layout: 2 filas (thumbnails arriba, NDVI abajo) ----
    n_thumbs = len(thumbs)
    rows = 2 if n_thumbs > 0 else 1
    fig = plt.figure(figsize=(10, 6.5 if n_thumbs > 0 else 4.2), dpi=DPI)
    gs = gridspec.GridSpec(rows, 1, height_ratios=([1.05, 1.9] if n_thumbs > 0 else [1]),
                           hspace=0.20)

    # ---- Fila 0: thumbnails en una subgrilla 1xN ----
    if n_thumbs > 0:
        gs_top = gs[0].subgridspec(1, n_thumbs, wspace=0.06)
        for i in range(n_thumbs):
            ax_img = fig.add_subplot(gs_top[0, i])
            ax_img.imshow(thumbs[i], interpolation='nearest')
            ax_img.set_axis_off()
            # Etiqueta de año arriba de cada miniatura (2019+i)
            ax_img.set_title(str(2019 + i), fontsize=10, pad=2)

    # ---- Fila 1: NDVI ----
    ax = fig.add_subplot(gs[1 if n_thumbs > 0 else 0, 0])
    if ndvi is not None and ndvi.notna().any():
        ax.plot(ndvi.index, ndvi.values, color="green", lw=1.2, marker=None)
        # LOESS de tendencia
        ndvi_loess = mk_loess(ndvi, frac=LOESS_FRAC)
        if ndvi_loess is not None:
            ax.plot(ndvi_loess.index, ndvi_loess.values, color="darkgreen", lw=2.0)

        # Guías y etiquetas
        for y, lbl in NDVI_GUIDES:
            ax.axhline(y, color="gray", linestyle="--", lw=0.8)
            ax.text(ndvi.index.min(), y + 0.01, lbl, color="gray", fontsize=9, va="bottom")

        ax.set_ylabel("NDVI")
        ax.set_ylim(0.2, 0.8)
        ax.set_xlim(ndvi.index.min(), ndvi.index.max())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, "Sin NDVI", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle(f"AOI (January 2019–2024)  {title_note}", y=0.98, fontsize=12)
    # Reservar un poco de margen para que los thumbs no choquen con el título
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Panel guardado en: {OUT_PNG}")



def main():
    # 1) Generar PNGs de enero (muestra_1) desde tus TIF
    png_paths = export_january_thumbs()
    thumbs = load_thumbnails_from_list(png_paths, max_imgs=8)

    # 2) Cargar serie NDVI de tu CSV pivot (para la subzona elegida)
    ndvi = load_pivot(CSV_NDVI, SUBZONA, Y_MIN, Y_MAX)

    # 3) Dibujar panel
    plot_panel(ndvi, thumbs, title_note="")

if __name__ == "__main__":
    main()
