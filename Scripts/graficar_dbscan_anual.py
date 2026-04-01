#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graficar_tendencias.py
----------------------
Lee 'preprocessed_timeseries.csv' y grafica las curvas de NDVI promedio anual
para cada subzona (líneas suaves, sin guardar archivos).
"""

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================
CSV_PREPRO = "/Users/samsu/Escritorio/Final script/salidas_dbscan/preprocessed_timeseries.csv"

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(CSV_PREPRO, parse_dates=["date"])

# Identificar columnas de subzonas
subzonas = [c for c in df.columns if c not in ("date", "muestra_elegida")]

# Agregar columna de año
df["year"] = df["date"].dt.year

# Promedio anual por subzona
yearly = df.groupby("year")[subzonas].mean()

# =========================
# GRAFICAR
# =========================
plt.figure(figsize=(11,6))

for s in subzonas:
    plt.plot(yearly.index, yearly[s], marker="o", linewidth=2, label=s)

plt.title("NDVI promedio anual por subzona (2019–2024)")
plt.xlabel("Año")
plt.ylabel("NDVI promedio (preprocesado)")
plt.grid(True, alpha=0.3)
plt.legend(loc="best", ncol=2)
plt.tight_layout()
plt.show()
