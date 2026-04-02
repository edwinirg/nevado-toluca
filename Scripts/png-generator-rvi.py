from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ===================== CONFIG RVI =====================
RVI_BASE = Path("./salidas_rvi")
SUBZONA = "Bosques_densos_pino_oyamel"
START_DATE = "2018-12-01"
END_DATE = "2019-03-31"

OUT_PNG_FULL = Path("./panel_rvi_completo_estacional.png")
OUT_PNG_ZOOM = Path("./panel_rvi_zoom_marzo_mayo.png")
DPI = 300

RAINY_SEASON = [6, 7, 8, 9, 10]
# ======================================================


def add_seasonal_spans(ax, dates):
    start_date = dates.min()
    end_date = dates.max()

    curr = start_date
    while curr <= end_date:
        if curr.month in RAINY_SEASON:
            ax.axvspan(
                curr,
                curr + pd.DateOffset(months=1),
                color="blue",
                alpha=0.1,
                label="Rainy Season" if curr == start_date else "",
            )
        else:
            ax.axvspan(
                curr,
                curr + pd.DateOffset(months=1),
                color="orange",
                alpha=0.05,
                label="Dry Season" if curr == start_date else "",
            )
        curr += pd.DateOffset(months=1)


def load_rvi_monthly_series(base_dir: Path, subzona: str) -> pd.Series:
    csv_files = sorted(base_dir.glob("*/**/muestra_*/rvi_celdas50m.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {base_dir}")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        required_cols = {"Año", "Mes", "Subzona", "RVI_mean"}
        if not required_cols.issubset(df.columns):
            continue

        df = df[df["Subzona"] == subzona].copy()
        if df.empty:
            continue

        df["RVI_mean"] = pd.to_numeric(df["RVI_mean"], errors="coerce")
        df = df.dropna(subset=["RVI_mean"])
        if df.empty:
            continue

        frames.append(df[["Año", "Mes", "RVI_mean"]])

    if not frames:
        raise ValueError(f"No hay datos validos de RVI para subzona: {subzona}")

    all_data = pd.concat(frames, ignore_index=True)

    # Combina todas las celdas y todas las muestras por mes (promedio mensual)
    monthly = all_data.groupby(["Año", "Mes"], as_index=False)["RVI_mean"].mean()
    monthly["date"] = pd.to_datetime(
        monthly["Año"].astype(int).astype(str)
        + "-"
        + monthly["Mes"].astype(int).astype(str)
        + "-01"
    )
    monthly = monthly.sort_values("date").set_index("date")

    return monthly["RVI_mean"]


def plot_custom_panel(rvi_series: pd.Series, output_path: Path, is_zoom: bool = False):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI)

    ax.plot(
        rvi_series.index,
        rvi_series.values,
        color="#4c2a85",
        lw=1.5,
        marker="o",
        markersize=4,
        label="Raw RVI",
    )

    if not is_zoom:
        add_seasonal_spans(ax, rvi_series.index)
        ax.set_title(f"PGROI Analysis (RVI): {SUBZONA}", fontsize=14, pad=15)
    else:
        ax.set_title(f"Zoom (Marzo-Mayo) RVI - {SUBZONA}", fontsize=14, pad=15)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(rvi_series.index)
        ax.set_xticklabels(rvi_series.index.strftime("%Y-%m"), rotation=45, ha="right")

    ax.set_ylabel("RVI Value")
    ax.set_xlabel("Year/Month")

    y_min = rvi_series.min()
    y_max = rvi_series.max()
    pad = max((y_max - y_min) * 0.15, 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✓ Guardado: {output_path}")
    plt.close()


def main():
    rvi_series = load_rvi_monthly_series(RVI_BASE, SUBZONA)

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    full_series = rvi_series[(rvi_series.index >= start) & (rvi_series.index <= end)]

    if full_series.empty:
        raise ValueError("No hay datos en el rango temporal solicitado.")

    plot_custom_panel(full_series, OUT_PNG_FULL, is_zoom=False)

    zoom_data = full_series[full_series.index.month.isin([3, 4, 5])]
    if not zoom_data.empty:
        plot_custom_panel(zoom_data, OUT_PNG_ZOOM, is_zoom=True)
    else:
        print("⚠️ No hay datos de marzo-mayo para generar zoom.")


if __name__ == "__main__":
    main()
