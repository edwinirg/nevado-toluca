import os
import glob
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = Path('/Users/samsu/Escritorio/Final script/Scripts/ndvi_total_y_zonas_indices')
OUTPUT_DIR = Path('/Users/samsu/Escritorio/Final script/Scripts/salidas_cluster_serie')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# celdas 50x50 m:
#  Columnas por celda:
#     Pixeles_validos: total de píxeles de la celda 
#     Pct_validos: píxeles válidos de la celda)
#   Si ambas columnas son 0 entonces es celda totalmente nublada.
#  Datos de salida en archivo CSV:
#    Mayor suma de píxeles válidos   (sum_valid_pixels)
#    Mayor suma de píxeles totales   (sum_total_pixels)
#    Menor número de celdas nubladas (cloud_cells)

# NECESITA SOLO ALGUNAS COLUMNAS DE LOS CSV DE SALIDA QUE SE GENERA CON 
#EL SCRIPT DE OBTENER_CD_NDVI, ESTA FUNCION MAPEA LAS COLUMNAS

def load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Lee CSV y normaliza columnas relevantes.
    Requiere poder mapear Pixeles_validos y Pct_validos. NDVI_* y Subzona son opcionales.
    """
    try:
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        colmap = {}
        # Subzona (opcional)
        for k in cols:
            if k in ("subzona", "sub_zone", "sub-zona", "sub zone"):
                colmap[cols[k]] = "Subzona"; break
        # NDVI_mean / std (opcionales)
        for k in cols:
            if k in ("ndvi_mean", "ndvi", "ndvi_media", "mean_ndvi", "ndvi_avg", "ndvi-mean"):
                colmap[cols[k]] = "NDVI_mean"; break
        for k in cols:
            if k in ("ndvi_std", "ndvi-std", "std_ndvi", "ndvi_desv", "ndvi_stddev", "ndvi stdev"):
                colmap[cols[k]] = "NDVI_std"; break
        # Centroide X/Y (opcionales)
        for k in cols:
            if k in ("centroide_x", "centroid_x", "x", "centroid x"):
                colmap[cols[k]] = "Centroide_X"; break
        for k in cols:
            if k in ("centroide_y", "centroid_y", "y", "centroid y"):
                colmap[cols[k]] = "Centroide_Y"; break
        # Pixeles totales por celda
        for k in cols:
            if k in ("pixeles_validos", "pixeles_totales", "pixels_total", "num_pixels", "pixeles", "total_pixeles"):
                colmap[cols[k]] = "Pixeles_validos"; break
        # Pixeles válidos por celda
        for k in cols:
            if k in ("pct_validos", "pix_validos", "pixels_validos", "valid_pixels", "pixeles_validos_ok", "pix_valid"):
                colmap[cols[k]] = "Pct_validos"; break
        # Requisitos para esta lógica
        if "Pixeles_validos" not in colmap.values() or "Pct_validos" not in colmap.values():
            return None
        return df.rename(columns=colmap)
    except Exception as e:
        warnings.warn(f"No se pudo leer {path}: {e}")
        return None


def find_sample_csvs(month_dir: Path, sample_num: int, *, verbose: bool = False) -> list[Path]:
    candidates: list[Path] = []
    subdir = month_dir / f"muestra_{sample_num}"

    if subdir.exists():
        candidates.extend(subdir.rglob("*.csv"))
    if not candidates:
        for pat in (f"*muestra{sample_num}*.csv", f"*muestra_{sample_num}*.csv"):
            candidates.extend(month_dir.rglob(pat))
    if not candidates:
        candidates.extend(month_dir.rglob("*.csv"))

    seen, uniq = set(), []
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(c)
    return uniq

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _has_xy(df: pd.DataFrame) -> bool:
    return set(["Centroide_X", "Centroide_Y"]).issubset(df.columns)


def _make_scatter(ax, df: pd.DataFrame, title: str):
    ax.clear()
    x = df.get("Centroide_X")
    y = df.get("Centroide_Y")
    c = ((df["NDVI_mean"].fillna(0) + 1.0) / 2.0).clip(0, 1)
    ax.scatter(x, y, c=c, s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Centroide X")
    ax.set_ylabel("Centroide Y")


def _make_bars(ax, df: pd.DataFrame, title: str):
    ax.clear()
    g = df.groupby("Subzona", as_index=False)["NDVI_mean"].mean().sort_values("NDVI_mean")
    ax.bar(g["Subzona"].astype(str), g["NDVI_mean"])
    ax.set_title(title)
    ax.set_xlabel("Subzona")
    ax.set_ylabel("NDVI_mean")
    for label in ax.get_xticklabels():
        label.set_rotation(90)


def save_months_gif_for_year(best_month_dfs: dict[int, pd.DataFrame], year: int, out_path: Path):
    if not best_month_dfs:
        return
    months_sorted = sorted(best_month_dfs.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    def update(i):
        month_num = months_sorted[i]
        dfi = best_month_dfs[month_num]
        title = f"Año {year} - Mes {month_num:02d} (mejor muestra)"
        if _has_xy(dfi):
            _make_scatter(ax, dfi, title)
        else:
            _make_bars(ax, dfi, title)
        return []

    ani = FuncAnimation(fig, update, frames=len(months_sorted), interval=1200, blit=False, repeat=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=1))
    plt.close(fig)


def _choose_best_sample(cands: list[dict]) -> dict:
    if not cands:
        return {}
    return sorted(
        cands,
        key=lambda c: (-c["sum_valid_pixels"], -c["sum_total_pixels"], c["cloud_cells"], c["sample"])  # criterios
    )[0]


def process_all(base_dir: Path = BASE_DIR, *, verbose: bool = True) -> None:
    chosen_records = []
    chosen_index_rows = []
    best_for_gif_by_year: dict[int, dict[int, pd.DataFrame]] = {}
    if not base_dir.exists():
        warnings.warn(f"PATH no existe: {base_dir}")
        return

    year_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    if not year_dirs:
        return

    for ydir in year_dirs:
        year = int(ydir.name)
        best_for_gif_by_year.setdefault(year, {})
        for month_num, month_folder in MONTH_FOLDERS:
            mdir = ydir / month_folder
            if not mdir.exists():
                if verbose:
                    print(f"Falta carpeta de mes: {mdir}")
                continue

            candidates = []

            for sample in (1, 2, 3):
                csv_paths = find_sample_csvs(mdir, sample, verbose=verbose)
                if not csv_paths:
                    if verbose:
                        print(f"{mdir} muestra {sample}: sin CSVs")
                    continue
                csv_file = csv_paths[0]
                df = load_csv_safe(csv_file)
                if df is None or df.empty:
                    if verbose:
                        print(f"{csv_file}: inválido o sin columnas de pixeles")
                    continue

                # Castear columnas de pixeles numéricas
                for ccol in ("Pixeles_validos", "Pct_validos"):
                    df[ccol] = pd.to_numeric(df[ccol], errors='coerce').fillna(0)

                sum_total_pixels = float(df["Pixeles_validos"].sum())
                sum_valid_pixels = float(df["Pct_validos"].sum())
                cloud_cells = int(((df["Pixeles_validos"] == 0) & (df["Pct_validos"] == 0)).sum())

                candidates.append({
                    "sample": sample,
                    "csv_file": str(csv_file),
                    "df_for_outputs": df.copy(),
                    "sum_total_pixels": sum_total_pixels,
                    "sum_valid_pixels": sum_valid_pixels,
                    "cloud_cells": cloud_cells,
                })

            if not candidates:
                if verbose:
                    print(f"[WARN] {year}-{month_num:02d}: sin candidatos")
                continue

            best = _choose_best_sample(candidates)
            if not best or best.get("df_for_outputs") is None or best["df_for_outputs"].empty:
                if verbose:
                    print(f"[WARN] {year}-{month_num:02d}: no se pudo elegir muestra")
                continue

            best_sample = best["sample"]
            df_best = best["df_for_outputs"]

            if verbose:
                print(
                    f"todo good"
                    # f" {year}-{month_num:02d}: muestra -> {best_sample} | "
                    # f"sum_valid_pixels={best['sum_valid_pixels']:.0f} "
                    # f"sum_total_pixels={best['sum_total_pixels']:.0f} "
                    # f"cloud_cells={best['cloud_cells']}"
                )

            # Para GIF
            df_best_for_gif = df_best.copy()
            if "NDVI_mean" not in df_best_for_gif.columns:
                df_best_for_gif["NDVI_mean"] = np.nan
            best_for_gif_by_year[year][month_num] = df_best_for_gif.copy()

            # Verifica Subzona
            if "Subzona" not in df_best.columns:
                df_best["Subzona"] = "_sin_subzona_"

            g_sub = df_best.groupby("Subzona", as_index=False).agg(
                NDVI_mean_filtered=("NDVI_mean", "mean"),
                n_filtered=("Subzona", "size"),
                sum_total_pixels=("Pixeles_validos", "sum"),
                sum_valid_pixels=("Pct_validos", "sum"),
            )
            cloud_by_sub = (
                df_best.assign(_cloud=((df_best["Pixeles_validos"]==0) & (df_best["Pct_validos"]==0)).astype(int))
                      .groupby("Subzona")['_cloud'].sum().reset_index(name='cloud_cells')
            )
            g_sub = g_sub.merge(cloud_by_sub, on="Subzona", how="left")

            g_sub["year"] = year
            g_sub["month"] = month_num
            g_sub["sample"] = best_sample
            chosen_records.append(g_sub)

            chosen_index_rows.append({
                "year": year,
                "month": month_num,
                "chosen_sample": best_sample,
                "csv_file": best.get("csv_file", ""),
                "sum_valid_pixels": float(best["sum_valid_pixels"]),
                "sum_total_pixels": float(best["sum_total_pixels"]),
                "cloud_cells": int(best["cloud_cells"]),
            })

    if not chosen_records:
        if verbose:
            print("[END] No se generaron registros elegidos.")
        return

    df_chosen = pd.concat(chosen_records, ignore_index=True)
    # Normalización solo si NDVI_mean existe
    if "NDVI_mean_filtered" in df_chosen.columns:
        df_chosen["NDVI_mean_filtered_norm"] = ((df_chosen["NDVI_mean_filtered"].fillna(0) + 1.0) / 2.0).clip(lower=0, upper=1)
    path_chosen = OUTPUT_DIR / "ndvi_por_subzona_muestra_elegida.csv"
    df_chosen.to_csv(path_chosen, index=False)

    monthly = df_chosen.groupby(["year", "month", "Subzona"], as_index=False).agg(
        NDVI_mean_filtered=("NDVI_mean_filtered", "mean"),
        n_filtered=("n_filtered", "sum"),
        sum_total_pixels=("sum_total_pixels", "sum"),
        sum_valid_pixels=("sum_valid_pixels", "sum"),
        cloud_cells=("cloud_cells", "sum"),
        chosen_sample=("sample", "first"),
    )
    if "NDVI_mean_filtered" in monthly.columns:
        monthly["NDVI_mean_filtered_norm"] = ((monthly["NDVI_mean_filtered"].fillna(0) + 1.0) / 2.0).clip(lower=0, upper=1)
    path_monthly = OUTPUT_DIR / "ndvi_mensual_muestra_elegida_por_subzona.csv"
    monthly.to_csv(path_monthly, index=False)

    # Pivot solo si hay NDVI
    if "NDVI_mean_filtered" in monthly.columns:
        pivot_ndvi = monthly.pivot_table(
            index=["year", "month"],
            columns="Subzona",
            values="NDVI_mean_filtered",
            aggfunc="mean"
        ).sort_index()

    # Obtener muestra elegida global por mes/año
    chosen_index = pd.DataFrame(chosen_index_rows).sort_values(["year", "month"]).reset_index(drop=True)
    chosen_index = chosen_index.set_index(["year", "month"])[["chosen_sample"]]
    
    # Combinar NDVI + columna única de muestra
    combined = pivot_ndvi.join(chosen_index, how="left")
    
    combined.to_csv(OUTPUT_DIR / "ndvi_mensual_pivot_subzonas_muestra_elegida.csv")

    chosen_index = pd.DataFrame(chosen_index_rows).sort_values(["year", "month"]).reset_index(drop=True)
    path_index = OUTPUT_DIR / "indice_muestra_elegida_por_mes.csv"
    chosen_index.to_csv(path_index, index=False)

    total_gifs = 0
    for year, best_month_dfs in best_for_gif_by_year.items():
        if not best_month_dfs:
            if verbose:
                print(f" Año {year}: sin meses válidos para GIF")
            continue
        out_gif = OUTPUT_DIR / f"gifs_best/anio_{year}_comparacion_12meses.gif"
        try:
            save_months_gif_for_year(best_month_dfs, year, out_gif)
            total_gifs += 1
            if verbose:
                print(f" Generado: {out_gif}")
        except Exception as e:
            warnings.warn(f"No se pudo crear GIF {out_gif}: {e}")


if __name__ == "__main__":
    process_all(BASE_DIR)
