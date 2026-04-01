#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
     Hampel = marca outliers como NaN
     Interpolación temporal
   Suavizado con rolling
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

# Ruta al pivot
CSV_PIVOT = Path("/Users/samsu/Escritorio/Final script/salidas_cluster_serie/ndvi_mensual_pivot_subzonas_muestra_elegida.csv")

OUTDIR = Path("/Users/samsu/Escritorio/Final script/salidas_dbscan")
OUTDIR.mkdir(parents=True, exist_ok=True)

SUBZONAS = ['Bosques_con_muerdago_2','Bosques_con_muerdago_3','Bosques_densos_pino_oyamel','Toluca']   # Array de subzonas, omite mariposa y parque venados

HAMPEL_WINDOW = 3
HAMPEL_K = 3.0
ROLLING_WINDOW = 5
ZSCORE = False

EPS = 0.10
MIN_SAMPLES = 1
METRIC = "euclidean" 

def hampel_filter(series: pd.Series, window_size: int = 3, n_sigmas: float = 3.0) -> pd.Series:
    """
    Filtro de Hampel: marca outliers como NaN (para luego interpolar).
    """
    if series.isna().all():
        return series.copy()
    x = series.values.astype(float)
    y = x.copy()
    k = int(max(1, window_size))
    n = len(x)
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        w = x[lo:hi]
        med = np.nanmedian(w)
        mad = np.nanmedian(np.abs(w - med))
        if mad == 0 or np.isnan(mad):
            continue
        thresh = n_sigmas * 1.4826 * mad
        if not np.isnan(x[i]) and abs(x[i] - med) > thresh:
            y[i] = np.nan
    return pd.Series(y, index=series.index)


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score por columna: (x - media)/std. Si std=0 o NaN -> 0.
    """
    out = df.copy()
    for c in out.columns:
        col = out[c]
        mu, sd = col.mean(skipna=True), col.std(skipna=True)
        out[c] = (col - mu) / sd if (sd and not np.isnan(sd) and sd > 0) else 0.0
    return out


def preprocess_pivot(pivot_ndvi: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesado estilo 'serie_de_tiempo.py':
      1) Hampel (marca NaN)
      2) Interpolación temporal
      3) Suavizado rolling(5, center=True)
      4) (Opcional) Z-score
    """
    work = pivot_ndvi.copy()
    # Asegurar índice datetime para interpolate(method="time")
    if not isinstance(work.index, pd.DatetimeIndex):
        raise ValueError("pivot_ndvi debe estar indexado por fechas (DatetimeIndex).")

    for c in work.columns:
        s = work[c].astype(float)
        s = hampel_filter(s, window_size=HAMPEL_WINDOW, n_sigmas=HAMPEL_K)
        s = s.interpolate(method="time")
        s = s.rolling(window=ROLLING_WINDOW, min_periods=1, center=True).mean()
        work[c] = s

    if ZSCORE:
        work = zscore_df(work)

    return work

# =========================
# Carga del pivot
# =========================

def load_pivot(csv_path: Path, wanted_cols: list[str]) -> pd.DataFrame:
    """
    Devuelve DF indexado por fecha (primer día del mes) con SOLO las columnas solicitadas.
    Requiere 'year' y 'month' en el CSV pivot.
    """
    df = pd.read_csv(csv_path)
    if not {"year", "month"}.issubset(df.columns):
        raise ValueError("El CSV pivot debe contener columnas 'year' y 'month'.")
    dt = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1), errors="coerce")
    df = df.drop(columns=["year", "month"]).assign(date=dt).set_index("date").sort_index()
    keep = [c for c in wanted_cols if c in df.columns]
    if not keep:
        raise ValueError("Ninguna de las columnas solicitadas existe en el CSV pivot.")
    return df[keep].copy()

# =========================
# DBSCAN
# =========================

def run_dbscan(cleaned_df: pd.DataFrame) -> pd.Series:
    """
    DBSCAN sobre columnas=subzonas, filas=tiempo (se transpone).
    Devuelve etiquetas por subzona.
    """
    X = cleaned_df.T.values  # cada subzona = vector temporal
    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=METRIC)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=cleaned_df.columns, name="cluster")


def run_dbscan_yearly(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta DBSCAN por año. Para cada year:
      - vectoriza (≈12 meses) por subzona
      - asigna cluster
    Devuelve DataFrame (year, subzona, cluster).
    """
    if not isinstance(cleaned_df.index, pd.DatetimeIndex):
        raise ValueError("cleaned_df debe estar indexado por fechas.")
    rows = []
    for year, df_year in cleaned_df.groupby(cleaned_df.index.year):
        if df_year.shape[0] < 3:
            for subzona in df_year.columns:
                rows.append({"year": int(year), "subzona": subzona, "cluster": -1})
            continue
        labels = run_dbscan(df_year)
        for subzona, label in labels.items():
            rows.append({"year": int(year), "subzona": subzona, "cluster": int(label)})
    return pd.DataFrame(rows)

# =========================
# Exports
# =========================

def export_preprocessed_with_sample(cleaned: pd.DataFrame, pivot_all: pd.DataFrame):
    """
    Guarda preprocessed_timeseries.csv con 'muestra_elegida' al final.
    Acepta 'chosen_sample' (pivot) o 'muestra_elegida' ya presente.
    """
    out_df = cleaned.copy()
    if "chosen_sample" in pivot_all.columns:
        out_df["muestra_elegida"] = pd.to_numeric(pivot_all["chosen_sample"], errors="coerce").astype("Int64")
    elif "muestra_elegida" in pivot_all.columns:
        out_df["muestra_elegida"] = pd.to_numeric(pivot_all["muestra_elegida"], errors="coerce").astype("Int64")
    out_df = out_df.reset_index().rename(columns={"index": "date"})
    cols = [c for c in out_df.columns if c != "muestra_elegida"]
    if "muestra_elegida" in out_df.columns:
        cols.append("muestra_elegida")
    out_df = out_df[cols]
    out_df.to_csv(OUTDIR / "preprocessed_timeseries.csv", index=False)


def export_yearly_clusters(df_yearly: pd.DataFrame):
    df_yearly.sort_values(["year", "cluster", "subzona"]).to_csv(OUTDIR / "clusters_por_anio.csv", index=False)


def export_global_clusters(cleaned: pd.DataFrame):
    labels = run_dbscan(cleaned)
    pd.DataFrame({"entity": cleaned.columns, "cluster": labels}) \
      .sort_values(["cluster", "entity"]) \
      .to_csv(OUTDIR / "clusters_labels_global.csv", index=False)

# =========================
# MAIN
# =========================

def main():
    # Columnas que queremos: NDVI por subzona + UNA columna global 'chosen_sample'
    wanted_cols = SUBZONAS + ["chosen_sample"]

    # 1) Cargar pivot
    pivot_all = load_pivot(CSV_PIVOT, wanted_cols=wanted_cols)

    # 2) NDVI para clustering
    pivot_ndvi = pivot_all[[c for c in SUBZONAS if c in pivot_all.columns]]

    # 3) Preprocesar NDVI con la receta de 'serie_de_tiempo.py'
    cleaned = preprocess_pivot(pivot_ndvi)

    # 4) Exportar series preprocesadas + muestra_elegida
    export_preprocessed_with_sample(cleaned, pivot_all)

    # 5) DBSCAN por año
    df_clusters_yearly = run_dbscan_yearly(cleaned)
    export_yearly_clusters(df_clusters_yearly)

    # 6) DBSCAN global (opcional)
    export_global_clusters(cleaned)

    print("Listo. Archivos generados en:", OUTDIR)


if __name__ == "__main__":
    main()
