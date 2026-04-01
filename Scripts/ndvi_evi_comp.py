import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix

def classify_ndvi(x):
    if x >= 0.60: return 1  # Vegetación densa
    elif x >= 0.35: return 2
    elif x >= 0.10: return 3
    else: return 4         # Agua/Nieve o suelo muy ralo

# Rutas a los TIFF elegidos (BAP) para enero 2019 y enero 2024
tif_2019 = ROOT / "2019" / "01_january" / f"muestra_{int(bap_log.query('year==\"2019\" and month==\"01_january\"')['chosen_sample'].iloc[0])}" / "NDVI_BAP.tif"
tif_2024 = ROOT / "2024" / "01_january" / f"muestra_{int(bap_log.query('year==\"2024\" and month==\"01_january\"')['chosen_sample'].iloc[0])}" / "NDVI_BAP.tif"

with rasterio.open(tif_2019) as s1, rasterio.open(tif_2024) as s2:
    a = s1.read(1).astype("float32")
    b = s2.read(1).astype("float32")

mask = np.isfinite(a) & np.isfinite(b)
a_cls = np.vectorize(classify_ndvi)(a[mask])
b_cls = np.vectorize(classify_ndvi)(b[mask])

cm = confusion_matrix(a_cls, b_cls, labels=[1,2,3,4])
print(cm)
