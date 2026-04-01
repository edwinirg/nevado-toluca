import os
import re
import glob
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Carpeta que contiene los años, es la misma que el dataset que te envie (el .zip), solo descomprimela y pega el path o ruta
BASE_DIR = Path('/home/edwinirg/Desktop/ndvi_total_y_zonas_indices')

# Carpeta de salida para CSVs, aqui se guardará todo lo que arroje el script, incluso los gifs
OUTPUT_DIR = Path('./salidas_ndvi')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo de meses, misma estructura que el dataset
MONTH_FOLDERS = [
    (1,  '01_january'),
    (2,  '02_february'),
    (3,  '03_march'),
    (4,  '04_april'),
    (5,  '05_may'),
    (6,  '06_june'),
    (7,  '07_july'),
    (8,  '08_august'),
    (9,  '09_september'),
    (10, '10_october'),
    (11, '11_november'),
    (12, '12_december'),
]

# Patrón de nombre de archivo CSV, busca que tenga laestructura base del archivo mediange REGEX
CSV_NAME_REGEX = re.compile(
    r"(?i)(ndvi).*?(celdas\s*?50m|celdas_?50m).*?(muestra\s*?([123])|muestra([123]))\.csv$"
)

# FUNCIONES AUXILIARES

def identificar_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Devuelve las filas outlier según media±threshold*std de la columna dada."""
    mean = df[column].mean()
    std = df[column].std(ddof=0)
    if pd.isna(std) or std == 0:
        return df.iloc[0:0]  # sin outliers si no hay variabilidad
    return df[np.abs(df[column] - mean) > threshold * std]


def remove_outliers_by_group(df: pd.DataFrame, group_col: str, value_col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Elimina outliers dentro de cada grupo (e.g., Subzona)."""
    outliers = df.groupby(group_col, group_keys=False).apply(
        lambda x: identificar_outliers(x, value_col, threshold)
    )
    if outliers.empty:
        return df.copy()
    return df.loc[~df.index.isin(outliers.index)].copy()


def normalize_ndvi(ndvi_series: pd.Series) -> pd.Series:
    """Normaliza NDVI de [-1, 1] a [0, 1]."""
    return (ndvi_series + 1.0) / 2.0

#NOTA: Esta estructura es la que definí yo en el script para obtener el NDVI por cada año, mes y muestra
#Si se cambia la estructura de carpetas o nombres de archivos, tambien necesitamos modificar un poco el codigo
#Dejé la cantidad de carpetas un poco más abierta, si quisieramos procesar más años o muestras, este script lo encontrará

def find_sample_csvs(month_dir: Path, sample_num: int) -> list[Path]:
    """Busca CSVs del mes para una muestra N dentro de carpeta muestra_N y nombres típicos."""
    # Primero buscamos dentro de carpeta muestra_N si existe
    candidates = []
    subdir = month_dir / f"muestra_{sample_num}"
    patterns = [
        str(subdir / "**/*.csv"),                # cualquier CSV dentro de la carpeta
        str(month_dir / f"**/*muestra{sample_num}*.csv"),  # CSVs con muestran el numero de muestra o sample en este caso
        str(month_dir / f"**/*muestra_{sample_num}*.csv"),
    ]
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            path = Path(p)
            if CSV_NAME_REGEX.search(path.name):
                candidates.append(path)
    # Devolver únicos manteniendo orden, SET genera un arreglo sin valores repeditos. por eso se valida 
    seen = set()
    uniq = []
    for c in candidates:
        if c.resolve() not in seen:
            seen.add(c.resolve())
            uniq.append(c)
    return uniq


#Funciona con la estructura de los CSV del dataset, no podemos quitar columnas
def load_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        # Normalizamos posibles nombres de columnas
        cols = {c.lower().strip(): c for c in df.columns}
        # Buscamos columnas clave con variantes comunes
        colmap = {}
        # Subzona
        for k in list(cols.keys()):
            if k in ("subzona", "sub_zone", "sub-zona"):
                colmap[cols[k]] = "Subzona"
                break
        # NDVI
        for k in list(cols.keys()):
            if k in ("ndvi_mean", "ndvi", "ndvi_media"):
                colmap[cols[k]] = "NDVI_mean"
                break
        # Centroide X/Y 
        for k in list(cols.keys()):
            if k in ("centroide_x", "centroid_x", "x"):
                colmap[cols[k]] = "Centroide_X"
                break
        for k in list(cols.keys()):
            if k in ("centroide_y", "centroid_y", "y"):
                colmap[cols[k]] = "Centroide_Y"
                break
        if "NDVI_mean" not in colmap.values() or "Subzona" not in colmap.values():
            return None
        df = df.rename(columns=colmap)
        return df
    except Exception as e:
        warnings.warn(f"No se pudo leer {path}: {e}")
        return None


# PROCESAMIENTO PRINCIPAL

def process_all(base_dir: Path = BASE_DIR) -> None:
    records_per_sample = []  # filas por muestra

    # Recorre carpetas por años, verifica que sea un number y que sea un directorio válido
    year_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if not year_dirs:
        warnings.warn("No se encontraron carpetas de años en BASE_DIR.")

    for ydir in year_dirs:
        year = int(ydir.name)
        for month_num, month_folder in MONTH_FOLDERS:
            mdir = ydir / month_folder
            if not mdir.exists():
                # Si no existe se omite
                continue

            for sample in (1, 2, 3):
                csv_paths = find_sample_csvs(mdir, sample)
                if not csv_paths:
                    # Si no hay CSV para esa muestra se salta ese sample
                    continue

                # Tomamos el primer match por muestra 
                csv_file = csv_paths[0]
                df = load_csv_safe(csv_file)
                if df is None or df.empty:
                    continue

                # Eliminar outliers por Subzona
                df_f = remove_outliers_by_group(df, group_col="Subzona", value_col="NDVI_mean", threshold=3.0)

                # Agregar métricas por Subzona para esta muestra
                g_raw = df.groupby("Subzona", as_index=False).agg(
                    NDVI_mean_raw=("NDVI_mean", "mean"),
                    n_raw=("NDVI_mean", "size"),
                )
                g_fil = df_f.groupby("Subzona", as_index=False).agg(
                    NDVI_mean_filtered=("NDVI_mean", "mean"),
                    n_filtered=("NDVI_mean", "size"),
                )
                merged = pd.merge(g_raw, g_fil, on="Subzona", how="outer")
                merged["year"] = year
                merged["month"] = month_num
                merged["sample"] = sample
                records_per_sample.append(merged)
    #Si no se generaron registros, quiere decir que no encontró nada, no retorna nada. 
    if not records_per_sample:
        return

    df_samples = pd.concat(records_per_sample, ignore_index=True)

    # Normalización [0,1] para evitar datos erroneos
    for col in ("NDVI_mean_raw", "NDVI_mean_filtered"):
        if col in df_samples.columns:
            df_samples[col + "_norm"] = ((df_samples[col] + 1.0) / 2.0).clip(lower=0, upper=1)

    # Guardar por muestra
    path_samples = OUTPUT_DIR / "ndvi_por_subzona_por_muestra.csv"
    df_samples.to_csv(path_samples, index=False)

    # Agregación mensual, 3 muestras por mes
    agg_cols = {
        "NDVI_mean_raw": "mean",
        "NDVI_mean_filtered": "mean",
        "n_raw": "sum",
        "n_filtered": "sum",
    }
    monthly = df_samples.groupby(["year", "month", "Subzona"], as_index=False).agg(agg_cols)

    # Normalizar agregados también
    for col in ("NDVI_mean_raw", "NDVI_mean_filtered"):
        monthly[col + "_norm"] = ((monthly[col] + 1.0) / 2.0).clip(lower=0, upper=1)

    # Pivot opcional para series de tiempo (una columna por Subzona)
    monthly_pivot = monthly.pivot_table(
        index=["year", "month"],
        columns="Subzona",
        values="NDVI_mean_filtered",
        aggfunc="mean"
    ).sort_index()

    # Guardar agregados en la carpeta que definimos al principio
    path_monthly = OUTPUT_DIR / "ndvi_mensual_por_subzona.csv"
    monthly.to_csv(path_monthly, index=False)

    path_pivot = OUTPUT_DIR / "ndvi_mensual_pivot_subzonas.csv"
    monthly_pivot.to_csv(path_pivot)

    print(f"L- {path_samples}\n- {path_monthly}\n- {path_pivot}")


# GENERACIÓN DE GIFs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _has_xy(df: pd.DataFrame) -> bool:
    return set(["Centroide_X", "Centroide_Y"]).issubset(df.columns)


def _make_scatter(ax, df: pd.DataFrame, title: str):
    ax.clear()
    x = df.get("Centroide_X")
    y = df.get("Centroide_Y")
    c = ((df["NDVI_mean"] + 1.0) / 2.0).clip(0, 1)
    sc = ax.scatter(x, y, c=c, s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Centroide X")
    ax.set_ylabel("Centroide Y")
    return sc


def _make_bars(ax, df: pd.DataFrame, title: str):
    ax.clear()
    g = df.groupby("Subzona", as_index=False)["NDVI_mean"].mean().sort_values("NDVI_mean")
    ax.bar(g["Subzona"].astype(str), g["NDVI_mean"])
    ax.set_title(title)
    ax.set_xlabel("Subzona")
    ax.set_ylabel("NDVI_mean")
    for label in ax.get_xticklabels():
        label.set_rotation(90)

#Crea un GIF que alterna las 3 muestras del mes (si hay XY) o barras por subzona por defecto.
def save_month_comparison_gif(dfs_by_sample: dict, out_path: Path, year: int, month_num: int):
    if not dfs_by_sample:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    samples = sorted(dfs_by_sample.keys())
    frames = []

    def update(i):
        sample = samples[i]
        dfi = dfs_by_sample[sample]
        title = f"Año {year} - Mes {month_num:02d} - Muestra {sample}"
        if _has_xy(dfi):
            _make_scatter(ax, dfi, title)
        else:
            _make_bars(ax, dfi, title)
        return []

#Cambié ImageMagick que usabas tú para el gid por FuncAnimation, esta solo genera una secuencia de oimagenes
    ani = FuncAnimation(fig, update, frames=len(samples), interval=1200, blit=False, repeat=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=1))
    plt.close(fig)


# Extendemos process_all para guardar GIFs por año y mes
old_process_all = process_all

def process_all(base_dir: Path = BASE_DIR) -> None:
    records_per_sample = []
    dfs_for_gifs = {}

    # Necesita carpetas de años como enteros (2019, 2020 etc)
    year_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    for ydir in year_dirs:
        year = int(ydir.name)
        for month_num, month_folder in MONTH_FOLDERS:
            mdir = ydir / month_folder
            if not mdir.exists():
                continue
            for sample in (1, 2, 3):
                csv_paths = find_sample_csvs(mdir, sample)
                if not csv_paths:
                    continue
                csv_file = csv_paths[0]
                df = load_csv_safe(csv_file)
                if df is None or df.empty:
                    continue
                # Mantener copia cruda para visual
                df_f = remove_outliers_by_group(df, group_col="Subzona", value_col="NDVI_mean", threshold=3.0)
                # Guardar para GIF (preferimos filtrado)
                dfs_for_gifs.setdefault((year, month_num), {})[sample] = df_f.copy()

                # Métricas por Subzona
                g_raw = df.groupby("Subzona", as_index=False).agg(
                    NDVI_mean_raw=("NDVI_mean", "mean"),
                    n_raw=("NDVI_mean", "size"),
                )
                g_fil = df_f.groupby("Subzona", as_index=False).agg(
                    NDVI_mean_filtered=("NDVI_mean", "mean"),
                    n_filtered=("NDVI_mean", "size"),
                )
                merged = pd.merge(g_raw, g_fil, on="Subzona", how="outer")
                merged["year"] = year
                merged["month"] = month_num
                merged["sample"] = sample
                records_per_sample.append(merged)

    if not records_per_sample:
        return

    df_samples = pd.concat(records_per_sample, ignore_index=True)

    # Normalización [0,1]
    for col in ("NDVI_mean_raw", "NDVI_mean_filtered"):
        if col in df_samples.columns:
            df_samples[col + "_norm"] = ((df_samples[col] + 1.0) / 2.0).clip(lower=0, upper=1)

    # Guardar por muestra
    path_samples = OUTPUT_DIR / "ndvi_por_subzona_por_muestra.csv"
    df_samples.to_csv(path_samples, index=False)

    # Agregación mensual (promedio de 3 muestras)
    agg_cols = {
        "NDVI_mean_raw": "mean",
        "NDVI_mean_filtered": "mean",
        "n_raw": "sum",
        "n_filtered": "sum",
    }
    monthly = df_samples.groupby(["year", "month", "Subzona"], as_index=False).agg(agg_cols)

    for col in ("NDVI_mean_raw", "NDVI_mean_filtered"):
        monthly[col + "_norm"] = ((monthly[col] + 1.0) / 2.0).clip(lower=0, upper=1)

    monthly_pivot = monthly.pivot_table(
        index=["year", "month"],
        columns="Subzona",
        values="NDVI_mean_filtered",
        aggfunc="mean"
    ).sort_index()

    path_monthly = OUTPUT_DIR / "ndvi_mensual_por_subzona.csv"
    monthly.to_csv(path_monthly, index=False)

    path_pivot = OUTPUT_DIR / "ndvi_mensual_pivot_subzonas.csv"
    monthly_pivot.to_csv(path_pivot)

    # GIFs por cada mes de las 3 muestras 
    for (year, month_num), d in dfs_for_gifs.items():
        if not d:
            continue
        out_gif = OUTPUT_DIR / f"gifs/{year}/{month_num:02d}_comparacion_muestras.gif"
        try:
            save_month_comparison_gif(d, out_gif, year, month_num)
        except Exception as e:
            warnings.warn(f"No se pudo crear GIF {out_gif}: {e}")

if __name__ == "__main__":
    process_all(BASE_DIR)

