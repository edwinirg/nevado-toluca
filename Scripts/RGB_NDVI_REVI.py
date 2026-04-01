import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

# ---------- RUTAS (ajusta) ----------
NDVI_TIF = r"C:\Users\samsu\Escritorio\Final script\Scripts\ndvi_total_y_zonas_indices\2019\01_january\muestra_1\NDVI_BAP.tif"   # NDVI en B/N [-1..1] o 0..10000
NDVI_RGB = r"C:\Users\samsu\Escritorio\Final script\Scripts\ndvi_total_y_zonas_indices\2019\01_january\muestra_1\NDVI_RGB.tif"   # si ya lo tienes coloreado (opcional)

S1_VV = r"C:\Users\samsu\Escritorio\Final script\Scripts\s1_vs_ndvi_comparacion\2019\01_january\muestra_1\S1_VV_lin_BAP.tif"   # lineal
S1_VH = r"C:\Users\samsu\Escritorio\Final script\Scripts\s1_vs_ndvi_comparacion\2019\01_january\muestra_1\S1_VH_lin_BAP.tif"   # lineal
OUT_RVI = r"C:\Users\samsu\Escritorio\Final script\Scripts\OUTPUT_COMP\RVI_2019_01.tif"
OUT_FIG = r"C:\Users\samsu\Escritorio\Final script\Scripts\OUTPUT_COMP\Fig_NDVI_vs_RVI_2019_01.png"
os.makedirs(os.path.dirname(OUT_RVI), exist_ok=True)
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

# ---------- utilidades ----------
def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        prof = src.profile
    return arr, prof

def normalize_ndvi(a):
    a = a.astype("float32")
    a[~np.isfinite(a)] = np.nan
    p1, p99 = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
    # intenta detectar escala
    if p99 <= 1.5 and p1 >= -1.5:  # ya [-1..1]
        ndvi = a
    elif p99 > 1.5 and p99 <= 10000 + 1:  # 0..10000
        ndvi = (a/10000.0)*2 - 1
    elif p1 >= 0 and p99 <= 255:  # 0..255
        ndvi = (a/127.5) - 1
    else:
        # estirar a [-1..1]
        ndvi = (a - p1) / (p99 - p1 + 1e-6) * 2 - 1
    ndvi[(ndvi < -1.2) | (ndvi > 1.2)] = np.nan
    return ndvi

# ---------- calcula RVI desde VV/VH lineal ----------
vv, prof = read_band(S1_VV)
vh, _    = read_band(S1_VH)
eps = 1e-10
rvi = (4.0 * vh) / (vv + vh + eps)
rvi = np.clip(rvi, 0, 1)

# guarda RVI como GeoTIFF (float32)
prof_out = prof.copy()
prof_out.update({"count": 1, "dtype": "float32", "nodata": np.nan})
with rasterio.open(OUT_RVI, "w", **prof_out) as dst:
    dst.write(rvi.astype("float32"), 1)

# ---------- NDVI para comparar (rango 0..1 para colormap) ----------
if NDVI_RGB and NDVI_RGB.lower().endswith(".tif"):
    # si ya tienes NDVI en RGB, sólo lo mostramos como está
    use_ndvi_rgb = True
else:
    use_ndvi_rgb = False

if not use_ndvi_rgb:
    ndvi_raw, _ = read_band(NDVI_TIF)
    ndvi = normalize_ndvi(ndvi_raw)
    ndvi01 = (ndvi + 1) / 2.0  # 0..1
else:
    ndvi01 = None

# ---------- Figura: mapas lado a lado + dispersión ----------
plt.rcParams["image.interpolation"] = "nearest"
fig = plt.figure(figsize=(10, 4.8))

# mapas
ax1 = fig.add_subplot(2, 2, 1)
if use_ndvi_rgb:
    with rasterio.open(NDVI_RGB) as src:
        rgb = np.dstack([src.read(1), src.read(2), src.read(3)]).astype(np.uint8)
    ax1.imshow(rgb)
else:
    im1 = ax1.imshow(ndvi01, cmap="YlGn", vmin=0, vmax=1)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="NDVI (0–1)")
ax1.set_title("NDVI (Jan 2019)")
ax1.set_axis_off()

ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(rvi, cmap="YlGn", vmin=0, vmax=1)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="RVI (0–1)")
ax2.set_title("RVI (Jan 2019)")
ax2.set_axis_off()

# dispersión NDVI vs RVI (muestra aleatoria para claridad)
ax3 = fig.add_subplot(2, 1, 2)
if use_ndvi_rgb:
    # si NDVI_RGB: no tenemos NDVI en 0..1 numérico; léelo del NDVI_TIF
    ndvi_raw, _ = read_band(NDVI_TIF)
    ndvi = normalize_ndvi(ndvi_raw)
    ndvi01 = (ndvi + 1) / 2.0

m = np.isfinite(ndvi01) & np.isfinite(rvi)
y, x = rvi[m], ndvi01[m]
if x.size > 150000:
    idx = np.random.choice(x.size, 150000, replace=False)
    x, y = x[idx], y[idx]

ax3.scatter(x, y, s=2, alpha=0.25)
# recta de regresión simple
A = np.vstack([x, np.ones_like(x)]).T
coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
xr = np.linspace(0, 1, 100)
ax3.plot(xr, coef[0]*xr + coef[1], color="black", lw=2)

# correlaciones
from scipy.stats import pearsonr, spearmanr
rp = pearsonr(x, y)[0]
rs = spearmanr(x, y)[0]
ax3.set_xlabel("NDVI (0–1)")
ax3.set_ylabel("RVI (0–1)")
ax3.set_title(f"NDVI vs RVI (Jan 2019)  |  Pearson={rp:.2f}, Spearman={rs:.2f}")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()
print("Guardados:\n", OUT_RVI, "\n", OUT_FIG)
