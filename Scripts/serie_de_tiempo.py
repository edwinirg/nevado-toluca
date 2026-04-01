# -*- coding: utf-8 -*-
"""
Panel estilo artículo:
- (Opcional) Tira de thumbnails arriba (quicklooks mensuales o anuales).
- Tmax (rojo)
- NDVI con LOESS + líneas guía (verde)
- NDMI con LOESS (azul) + precipitación (barras grises, eje derecho)

Entrada:
- CSV NDVI pivot mensual por subzonas: columnas: year, month, <subzonas...>
- CSV NDMI pivot mensual por subzonas (opcional): mismas columnas
- CSV clima diario/mensual (date, tmax, precip). Si diario, se agrega a mensual.
- Elegir SUBZONA a graficar.

Autor: adaptado a partir de tu script de series. (Hampel + rolling + LOESS)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statsmodels.nonparametric.smoothers_lowess import lowess
from PIL import Image

# ===================== CONFIG =====================
# Archivos de entrada (ajusta rutas)
CSV_NDVI =  Path('C:\\Users\samsu\Escritorio\Final script\Scripts\salidas_cluster_serie\ndvi_mensual_pivot_subzonas_muestra_elegida.csv')
CSV_NDMI = Path(r"/ruta/a/ndmi_mensual_pivot_subzonas.csv")   # Opcional (si no existe, se omite NDMI)
CSV_CLIMA = Path(r"/ruta/a/clima.csv")                         # Opcional (date,tmax,precip)

# Nombre exacto de la subzona a graficar (usa uno de tus campos de columna)
SUBZONA = "Bosques_densos_pino_oyamel"

# (Opcional) Carpeta con thumbnails (png/jpg) para la primera fila.
# Si no tienes, deja en None y no se dibuja.
THUMB_DIR = None  # por ejemplo r"C:\quicklooks\2023_10\"

# Rango temporal (si quieres recortar)
Y_MIN, Y_MAX = 2016, 2024

# Parámetros visuales/analíticos
ROLL = 3                   # media móvil leve tras interpolación
HAMPEL_WINDOW = 3          # ventana hampel (meses)
HAMPEL_K = 3.0
LOESS_FRAC = 0.12          # ~12% del rango como span para tendencia suave

# Guías NDVI y etiquetas (como en el paper)
NDVI_GUIDES = [(0.6, "Bosque"), (0.5, "Matorrales"), (0.4, "Pradera")]

# Salida
OUT_PNG = Path(r"./panel_estilo_articulo.png")
DPI = 300
# ==================================================


def hampel_filter(series: pd.Series, window=3, k=3.0) -> pd.Series:
    """Filtro Hampel simple (serie mensual). Reemplaza atípicos por NaN."""
    x = series.copy()
    med = x.rolling(window=window, center=True, min_periods=1).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=1).median()
    mad = mad.replace(0, mad[mad != 0].min() if (mad != 0).any() else 1e-9)
    z = (x - med).abs() / (1.4826 * mad)
    x[z > k] = np.nan
    return x


def load_pivot(csv_path: Path, col: str) -> pd.Series | None:
    """Lee pivot mensual (year, month, ...subzonas) y devuelve serie mensual para `col`."""
    if not csv_path or not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # Construir índice mensual
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df = df.set_index("date").sort_index()
    s = df[col].astype(float)
    # Recorte temporal
    s = s[(s.index.year >= Y_MIN) & (s.index.year <= Y_MAX)]
    # Limpieza: hampel + interp + rolling
    s = hampel_filter(s, window=HAMPEL_WINDOW, k=HAMPEL_K)
    s = s.interpolate(limit_direction="both")
    if ROLL and ROLL > 1:
        s = s.rolling(ROLL, center=True, min_periods=1).mean()
    return s


def mk_loess(y: pd.Series, frac=0.12) -> pd.Series | None:
    """Ajuste LOESS sobre índice temporal (convertido a ordinal)."""
    if y is None or y.dropna().empty:
        return None
    x = y.index.map(pd.Timestamp.toordinal).astype(float).values
    vals = y.values.astype(float)
    m = np.isfinite(vals)
    if m.sum() < 4:
        return None
    lo = lowess(vals[m], x[m], frac=frac, return_sorted=False)
    yhat = pd.Series(index=y.index[m], data=lo)
    # reindex para cubrir todo el eje temporal
    yhat = yhat.reindex(y.index).interpolate(limit_direction="both")
    return yhat


def load_climate(csv_path: Path):
    """Devuelve DF mensual con columnas (tmax, precip). Acepta diario o mensual."""
    if not csv_path or not csv_path.exists():
        return None
    c = pd.read_csv(csv_path)
    if "date" not in c.columns:
        raise ValueError("El CSV de clima debe tener columna 'date' (YYYY-MM-DD).")
    c["date"] = pd.to_datetime(c["date"])
    c = c.set_index("date").sort_index()
    # recorta
    c = c[(c.index.year >= Y_MIN) & (c.index.year <= Y_MAX)]
    # Si es diario, agrega a mensual
    if c.index.freq is None or c.index.freqstr not in ("MS", "M"):
        c_m = pd.DataFrame(index=pd.date_range(c.index.min().to_period('M').to_timestamp(),
                                               c.index.max().to_period('M').to_timestamp(),
                                               freq="MS"))
        if "tmax" in c.columns:
            c_m["tmax"] = c["tmax"].resample("MS").mean()
        if "precip" in c.columns:
            c_m["precip"] = c["precip"].resample("MS").sum()
        c = c_m
    return c


def load_thumbnails(folder: str | None, ncols=4, max_imgs=8):
    """Carga hasta `max_imgs` thumbnails (orden alfabético) y devuelve lista PIL Images."""
    if not folder:
        return []
    p = Path(folder)
    if not p.exists():
        return []
    imgs = [im for im in sorted(p.iterdir()) if im.suffix.lower() in (".png", ".jpg", ".jpeg")]
    imgs = imgs[:max_imgs]
    out = []
    for im in imgs:
        try:
            out.append(Image.open(im).convert("RGB"))
        except Exception:
            pass
    return out


def plot_panel(ndvi: pd.Series | None, ndmi: pd.Series | None, clima: pd.DataFrame | None,
               thumbs: list[Image], title_note: str = ""):
    """
    Dibuja el panel con 3 o 4 filas:
      [0] thumbnails (opcional)
      [1] Tmax
      [2] NDVI (+ LOESS y guías)
      [3] NDMI (+ LOESS) + precip eje derecho
    """
    # Construir layout
    rows = 3 + (1 if thumbs else 0)
    height_ratios = ([1] if thumbs else []) + [1.2, 1.4, 1.6]
    fig = plt.figure(figsize=(8.0, 10.5), dpi=DPI)
    gs = gridspec.GridSpec(rows, 1, height_ratios=height_ratios, hspace=0.35)

    ax_idx = 0

    # --- Thumbnails (opcional) ---
    if thumbs:
        ax0 = fig.add_subplot(gs[ax_idx, 0])
        ax0.axis("off")
        # grid simple de miniaturas
        n = len(thumbs)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        # colocar como “mosaico”
        x0, y0, w, h = 0.0, 0.05, 1.0, 0.9  # márgenes relativos
        cell_w, cell_h = w / ncols, h / nrows
        for i, im in enumerate(thumbs):
            r = i // ncols
            c = i % ncols
            left = x0 + c * cell_w
            bottom = y0 + (nrows - 1 - r) * cell_h
            ax0_in = fig.add_axes([left + 0.02 * cell_w, bottom + 0.08 * cell_h,
                                   0.96 * cell_w, 0.84 * cell_h])
            ax0_in.imshow(im)
            ax0_in.set_axis_off()
        ax_idx += 1

    # --- Tmax ---
    ax1 = fig.add_subplot(gs[ax_idx, 0]); ax_idx += 1
    if clima is not None and "tmax" in clima.columns and clima["tmax"].notna().any():
        tmax = clima["tmax"].copy().interpolate(limit_direction="both")
        ax1.plot(tmax.index, tmax.values, color="red", lw=1.2)
        ax1.set_ylabel("Air Tmax (°C)")
        ax1.set_xlim(tmax.index.min(), tmax.index.max())
    else:
        ax1.text(0.5, 0.5, "Sin datos de Tmax", ha="center", va="center")
        ax1.set_axis_off()

    # --- NDVI ---
    ax2 = fig.add_subplot(gs[ax_idx, 0]); ax_idx += 1
    if ndvi is not None and ndvi.notna().any():
        ax2.plot(ndvi.index, ndvi.values, color="green", lw=0.9)
        # LOESS
        ndvi_loess = mk_loess(ndvi, frac=LOESS_FRAC)
        if ndvi_loess is not None:
            ax2.plot(ndvi_loess.index, ndvi_loess.values, color="darkgreen", lw=2.0)
        # Guías + etiquetas
        for y, lbl in NDVI_GUIDES:
            ax2.axhline(y, color="gray", linestyle="--", lw=0.8)
            ax2.text(ndvi.index.min(), y + 0.01, lbl, color="gray", fontsize=9, va="bottom")
        ax2.set_ylabel("NDVI")
        ax2.set_ylim(0.2, 0.8)
        ax2.set_xlim(ndvi.index.min(), ndvi.index.max())
    else:
        ax2.text(0.5, 0.5, "Sin NDVI", ha="center", va="center")
        ax2.set_axis_off()

    # --- NDMI + Precip ---
    ax3 = fig.add_subplot(gs[ax_idx, 0]); ax_idx += 1
    if ndmi is not None and ndmi.notna().any():
        ax3.plot(ndmi.index, ndmi.values, color="blue", lw=0.9)
        ndmi_loess = mk_loess(ndmi, frac=LOESS_FRAC)
        if ndmi_loess is not None:
            ax3.plot(ndmi_loess.index, ndmi_loess.values, color="navy", lw=2.0, label="NDMI (LOESS)")
        ax3.set_ylabel("NDMI")
        ax3.set_xlim(ndmi.index.min(), ndmi.index.max())
        # Eje derecho: precip mensual en barras grises
        if clima is not None and "precip" in clima.columns and clima["precip"].notna().any():
            precip = clima["precip"].copy()
            # asegúrate de que sea mensual
            if precip.index.freq is None or precip.index.freqstr not in ("MS", "M"):
                precip = precip.resample("MS").sum()
            ax3b = ax3.twinx()
            ax3b.bar(precip.index, precip.values, width=20, color="lightgray", edgecolor="none")
            ax3b.set_ylabel("Precipitations (mm)")
    else:
        ax3.text(0.5, 0.5, "Sin NDMI", ha="center", va="center")
        ax3.set_axis_off()

    # Etiqueta general / título
    fig.suptitle(f"{SUBZONA}  {title_note}", y=0.99, fontsize=12)
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Panel guardado en: {OUT_PNG}")


def main():
    # Cargar series
    ndvi = load_pivot(CSV_NDVI, SUBZONA)
    ndmi = load_pivot(CSV_NDMI, SUBZONA) if CSV_NDMI and CSV_NDMI.exists() else None
    clima = load_climate(CSV_CLIMA)

    # Ajustar rangos para coincidir
    # Usa el rango máximo común entre series disponibles
    idxs = []
    for s in [ndvi, ndmi]:
        if s is not None and not s.dropna().empty:
            idxs.append((s.index.min(), s.index.max()))
    if clima is not None and not clima.dropna(how="all").empty:
        idxs.append((clima.index.min(), clima.index.max()))
    if idxs:
        start = max(t[0] for t in idxs)
        end   = min(t[1] for t in idxs)
        if ndvi is not None: ndvi = ndvi.loc[(ndvi.index >= start) & (ndvi.index <= end)]
        if ndmi is not None: ndmi = ndmi.loc[(ndmi.index >= start) & (ndmi.index <= end)]
        if clima is not None: clima = clima.loc[(clima.index >= start) & (clima.index <= end)]

    # Thumbnails (opcional)
    thumbs = load_thumbnails(THUMB_DIR) if THUMB_DIR else []

    # Dibujar
    plot_panel(ndvi, ndmi, clima, thumbs, title_note="(2016–2024)")

if __name__ == "__main__":
    main()
