# -*- coding: utf-8 -*-
"""
Fig. 11 — Validación cuantitativa (sin datos externos)
Confusión 2x2 entre NDVI (óptico) y RVI (radar) para ENERO de un año dado.

Requisitos: numpy, rasterio, matplotlib, pandas, scikit-learn
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score

# ------------------- CONFIG -------------------
YEAR = 2019  # <-- cambia a 2024 si quieres
# NDVI (enero YEAR)
NDVI_TIF = r"C:\Users\samsu\Escritorio\Final script\Scripts\ndvi_total_y_zonas_indices\2019\01_january\muestra_1\NDVI_BAP.tif"

# Sentinel-1 (enero YEAR) en LINEAL (no dB)
S1_VV = r"C:\Users\samsu\Escritorio\Final script\Scripts\s1_vs_ndvi_comparacion\2019\01_january\muestra_1\S1_VV_lin_BAP.tif"
S1_VH = r"C:\Users\samsu\Escritorio\Final script\Scripts\s1_vs_ndvi_comparacion\2019\01_january\muestra_1\S1_VH_lin_BAP.tif"

# Umbrales de clasificación
TH_NDVI = 0.60  # Vegetación óptica
TH_RVI  = 0.50  # Vegetación estructural (radar)

# Salida
OUT_PNG = Path(fr".\Fig11_validacion_metrics_NDVIvsRVI_{YEAR}_01.png")
DPI = 300
# ----------------------------------------------


def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        prof = src.profile
        transform = src.transform
        crs = src.crs
    return arr, prof, transform, crs


def normalize_ndvi(a):
    """Devuelve NDVI en [-1,1]. Detecta escala común ([-1,1], 0..255, 0..10000)."""
    x = a.astype("float32")
    x[~np.isfinite(x)] = np.nan
    p1, p99 = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if p99 <= 1.5 and p1 >= -1.5:
        ndvi = x
    elif p99 > 1.5 and p99 <= 10000 + 1:
        ndvi = (x / 10000.0) * 2.0 - 1.0
    elif p1 >= 0 and p99 <= 255:
        ndvi = (x / 127.5) - 1.0
    else:
        ndvi = (x - p1) / (p99 - p1 + 1e-6) * 2.0 - 1.0
    ndvi[(ndvi < -1.2) | (ndvi > 1.2)] = np.nan
    return ndvi


def reproject_to_grid(src_path, dst_profile, dst_transform, dst_crs, dst_shape):
    """Reproyecta un raster (1 banda) al grid destino (NDVI)."""
    with rasterio.open(src_path) as src:
        dst = np.full(dst_shape, np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst


def main():
    # 1) NDVI base (define grid destino)
    ndvi_raw, prof_ndvi, tr_ndvi, crs_ndvi = read_band(NDVI_TIF)
    ndvi = normalize_ndvi(ndvi_raw)
    ndvi01 = (ndvi + 1.0) / 2.0  # 0..1 para umbral 0.60

    # 2) Reproyectar VV/VH a la malla del NDVI (si es necesario)
    with rasterio.open(NDVI_TIF) as ref:
        height, width = ref.height, ref.width
    vv = reproject_to_grid(S1_VV, prof_ndvi, tr_ndvi, crs_ndvi, (height, width))
    vh = reproject_to_grid(S1_VH, prof_ndvi, tr_ndvi, crs_ndvi, (height, width))

    # 3) Calcular RVI en 0..1 y enmascarar valores no finitos
    eps = 1e-10
    rvi = (4.0 * vh) / (vv + vh + eps)
    rvi = np.clip(rvi, 0.0, 1.0)

    # 4) Máscara de validez común
    m = np.isfinite(ndvi01) & np.isfinite(rvi)
    if not np.any(m):
        raise RuntimeError("No hay intersección de datos válidos entre NDVI y RVI.")

    # 5) Clasificación binaria por umbrales
    ndvi_cls = (ndvi01 >= TH_NDVI).astype(np.uint8)[m]  # 1=Vegetación óptica
    rvi_cls  = (rvi >= TH_RVI).astype(np.uint8)[m]      # 1=Vegetación estructural

    # 6) Métricas
    cm = confusion_matrix(rvi_cls, ndvi_cls, labels=[0, 1])  # filas=RVI ref, cols=NDVI pred
    tn, fp, fn, tp = cm.ravel()
    oa = (tp + tn) / cm.sum()
    precision, recall, f1, _ = precision_recall_fscore_support(rvi_cls, ndvi_cls, average='binary', zero_division=0)
    kappa = cohen_kappa_score(rvi_cls, ndvi_cls)

    # 7) Figura: heatmap + barras de métricas
    fig = plt.figure(figsize=(11, 5.2), dpi=DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.35)  # ← más espacio

    # --- Heatmap ---
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax0.text(j, i, f"{v:,}", ha='center', va='center', fontsize=10)
    ax0.set_title(f"Confusion Matrix (Jan {YEAR})")
    ax0.set_xlabel(f"NDVI ≥ {TH_NDVI:.2f} (Pred)")
    ax0.set_ylabel(f"RVI ≥ {TH_RVI:.2f} (Ref)")
    ax0.set_xticks([0, 1]); ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["No veg.", "Veg."])
    ax0.set_yticklabels(["No veg.", "Veg."])
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Pixel count")

    # --- Barras de métricas ---
    ax1 = fig.add_subplot(gs[0, 1])
    names = ["OA", "Precision", "Recall", "F1", "Kappa"]
    vals = [oa, precision, recall, f1, kappa]
    bars = ax1.bar(names, vals, color="#1f77b4")
    ax1.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}", ha="center", fontsize=10)
    ax1.set_title("Validation Metrics (NDVI vs RVI)")
    ax1.set_ylabel("Score (0–1)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"NDVI vs RVI validation (Jan {YEAR})", fontsize=13, y=0.97)
    os.makedirs(OUT_PNG.parent, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
