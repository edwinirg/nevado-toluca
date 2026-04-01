import io
import re
import zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from PIL import Image

ROOT_DIR = Path(r"C:\Users\samsu\Escritorio\Final script\Scripts\ndvi_total_y_zonas_indices")
YEAR_START, YEAR_END = 2019, 2024
SAMPLE_DAY = {"muestra_1": "05", "muestra_2": "15", "muestra_3": "25"}

RX_MM_PREFIX = re.compile(r"^(?P<mm>\d{2})_") #Regex pal directorio del mes

def month_from_dirname(name: str) -> str | None:
    m = RX_MM_PREFIX.match(name)
    return m.group("mm") if m else None

def tiff_to_png_bytes_with_bbox(tif_path: Path, max_size_px: int = 0):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        if crs is not None and crs.to_string().upper() != "EPSG:4326":
            w, s, e, n = transform_bounds(
                crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21
            )
        else:
            w, s, e, n = bounds.left, bounds.bottom, bounds.right, bounds.top

        arr = src.read(1).astype("float32")
        nd = src.nodata
        mask = ~np.isfinite(arr) | ((arr == nd) if nd is not None else False)

        valid = arr[~mask]
        if valid.size == 0:
            scaled = np.zeros_like(arr, dtype=np.uint8)
        else:
            lo, hi = np.nanpercentile(valid, [2, 98])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = np.nanmin(valid), np.nanmax(valid)
                if lo == hi:
                    hi = lo + 1e-6
            clipped = np.clip(arr, lo, hi)
            scaled = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
            scaled[mask] = 0

        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = scaled
        rgba[..., 1] = scaled
        rgba[..., 2] = scaled
        rgba[..., 3] = np.where(mask, 0, 255)

        img = Image.fromarray(rgba, mode="RGBA")
        if max_size_px and max(img.size) > max_size_px:
            if img.width >= img.height:
                new_w = max_size_px
                new_h = int(img.height * (max_size_px / img.width))
            else:
                new_h = max_size_px
                new_w = int(img.width * (max_size_px / img.height))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue(), (w, s, e, n)

def kml_header(title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "  <Document>",
        f"    <name>{title}</name>",
    ]

def kml_footer() -> list[str]:
    return ["  </Document>", "</kml>"]

def ground_overlay(name_png: str, date_iso: str, bbox: tuple[float,float,float,float]) -> list[str]:
    w, s, e, n = bbox
    lines = []
    lines.append("    <GroundOverlay>")
    lines.append(f"      <name>{name_png}</name>")
    if date_iso:
        lines.append(f"      <TimeStamp><when>{date_iso}</when></TimeStamp>")
    lines.append("      <Icon>")
    lines.append(f"        <href>{name_png}</href>")
    lines.append("      </Icon>")
    lines.append("      <LatLonBox>")
    lines.append(f"        <north>{n}</north>")
    lines.append(f"        <south>{s}</south>")
    lines.append(f"        <east>{e}</east>")
    lines.append(f"        <west>{w}</west>")
    lines.append("      </LatLonBox>")
    lines.append("      <altitude>0</altitude>")
    lines.append("      <altitudeMode>clampToGround</altitudeMode>")
    lines.append("    </GroundOverlay>")
    return lines

def main():
    kmz_path = ROOT_DIR / "NDVI_BAP_2019_2024.kmz"
    kml_lines = kml_header("NDVI (2019–2024) CD Original")
    overlays_written = 0

    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as kmz:
        for year in range(YEAR_START, YEAR_END + 1):
            year_dir = ROOT_DIR / str(year)
            if not year_dir.is_dir():
                continue

            for month_dir in sorted([p for p in year_dir.iterdir() if p.is_dir()]):
                mm = month_from_dirname(month_dir.name)
                if not mm:
                    continue
                for muestra_name, day in SAMPLE_DAY.items():
                    sample_dir = month_dir / muestra_name
                    if not sample_dir.is_dir():
                        continue

                    tif_path = sample_dir / "NDVI_BAP.tif"
                    if not tif_path.exists():
                        candidates = list(sample_dir.glob("*.tif"))
                        if not candidates:
                            continue
                        tif_path = candidates[0]

                    try:
                        dt = datetime(year, int(mm), int(day))
                        date_iso = dt.strftime("%Y-%m-%dT00:00:00Z")
                    except Exception:
                        date_iso = None

                    try:
                        png_bytes, bbox = tiff_to_png_bytes_with_bbox(tif_path, 0)
                    except Exception as e:
                        print(f"[WARN] Falló {tif_path}: {e}")
                        continue

                    png_name = f"{year}_{mm}_{muestra_name}.png"
                    kmz.writestr(png_name, png_bytes)

                    kml_lines.extend(ground_overlay(png_name, date_iso, bbox))
                    overlays_written += 1

    kml_lines.extend(kml_footer())
    with zipfile.ZipFile(kmz_path, "a", compression=zipfile.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", "\n".join(kml_lines))

if __name__ == "__main__":
    main()
