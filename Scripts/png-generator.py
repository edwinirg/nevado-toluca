import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import rasterio

# ===================== CONFIG ACTUALIZADA =====================
INPUT_BASE    = r"C:\Users\samsu\Escritorio\Final script\Scripts\ndvi_total_y_zonas_indices"
YEARS         = range(2019, 2025)
CSV_NDVI      = Path(r"C:\Users\samsu\Escritorio\Final script\Scripts\salidas_cluster_serie\ndvi_mensual_pivot_subzonas_muestra_elegida.csv")
SUBZONA       = "Bosques_densos_pino_oyamel"
OUT_PNG_FULL  = Path(r".\panel_ndvi_completo_estacional.png")
OUT_PNG_ZOOM  = Path(r".\panel_ndvi_zoom_marzo_mayo.png")
DPI           = 300

# Definición de temporadas (Meses en México)
RAINY_SEASON = [6, 7, 8, 9, 10] # Junio a Octubre
DRY_SEASON   = [11, 12, 1, 2, 3, 4, 5]

# ==============================================================

def load_raw_data(csv_path: Path, col: str) -> pd.Series:
    """Carga datos sin suavizado (Raw)"""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df = df.set_index("date").sort_index()
    return df[col].astype(float)

def add_seasonal_spans(ax, dates):
    """Añade sombreado de fondo para distinguir temporadas"""
    start_date = dates.min()
    end_date = dates.max()
    
    # Iterar por meses para sombrear lluvias
    curr = start_date
    while curr <= end_date:
        if curr.month in RAINY_SEASON:
            ax.axvspan(curr, curr + pd.DateOffset(months=1), 
                       color='blue', alpha=0.1, label='Rainy Season' if curr == start_date else "")
        else:
            ax.axvspan(curr, curr + pd.DateOffset(months=1), 
                       color='orange', alpha=0.05, label='Dry Season' if curr == start_date else "")
        curr += pd.DateOffset(months=1)

def plot_custom_panel(ndvi_series, output_path, is_zoom=False):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)
    
    # 1. Graficar la línea cruda (Sin suavizado)
    ax.plot(ndvi_series.index, ndvi_series.values, color="#2d5a27", lw=1.5, 
            marker='o', markersize=4, label="Raw NDVI")
    
    # 2. Añadir capas de contexto
    if not is_zoom:
        add_seasonal_spans(ax, ndvi_series.index)
        ax.set_title(f"PGROI Analysis: {SUBZONA}", fontsize=14, pad=15)
    else:
        ax.set_title(f"Zoom (Marzo-Mayo) - {SUBZONA}", fontsize=14, pad=15)
        ax.grid(True, linestyle='--', alpha=0.6)

    # 3. Estética
    ax.set_ylabel("NDVI Value")
    ax.set_xlabel("Year/Month")
    ax.set_ylim(ndvi_series.min() - 0.05, ndvi_series.max() + 0.05)
    
    # Eliminar spines innecesarios
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✓ Guardado: {output_path}")
    plt.close()

def main():
    # 1. Cargar datos base
    data = load_raw_data(CSV_NDVI, SUBZONA)
    
    # 2. Generar Gráfico Original (Sin suavizado + Temporadas)
    # Filtramos por los años de interés para el reporte
    full_series = data[(data.index.year >= 2019) & (data.index.year <= 2024)]
    plot_custom_panel(full_series, OUT_PNG_FULL, is_zoom=False)
    
    # 3. Generar Gráfico Zoom (Marzo a Mayo)
    # Filtramos solo los meses de "humedad" primaveral
    zoom_data = full_series[full_series.index.month.isin([3, 4, 5])]
    plot_custom_panel(zoom_data, OUT_PNG_ZOOM, is_zoom=True)

if __name__ == "__main__":
    main()