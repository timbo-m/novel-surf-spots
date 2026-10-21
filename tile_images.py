"""Cut big coastline images into square chips for labelling and scoring.

Works on plain PNG/JPG screenshots or on GeoTIFFs (Sentinel-2, aerial
orthophotos). For GeoTIFFs each chip's centre lat/lon goes in the index so
scores can be mapped back to the coast and exported to KML.

  python tile_images.py images/*.tif --out chips --size 256 --stride 192
"""
import argparse, csv, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(path):
    """Return (HxWx3 uint8 array, transform-fn pixel->lonlat or None)."""
    if path.lower().endswith((".tif", ".tiff", ".jp2")):
        try:
            import rasterio
            from rasterio.warp import transform as warp
        except ImportError:
            rasterio = None
        if rasterio:
            with rasterio.open(path) as src:
                bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
                arr = src.read(bands)
                tf, crs = src.transform, src.crs
            # 8-bit true-colour products (Sentinel-2 TCI, aerial photos) are left as-is;
            # raw 12/16-bit reflectance gets stretched to 8-bit.
            arr = arr if arr.dtype == np.uint8 else stretch(arr.astype(np.float32))
            def to_lonlat(col, row):
                x, y = tf * (col, row)
                if crs and not crs.is_geographic:
                    xs, ys = warp(crs, "EPSG:4326", [x], [y])
                    return xs[0], ys[0]
                return x, y
            return np.transpose(arr, (1, 2, 0)), to_lonlat
    return np.array(Image.open(path).convert("RGB")), None


def stretch(arr):
    """2-98% percentile stretch per band so 12/16-bit satellite data looks normal."""
    out = np.empty_like(arr, dtype=np.uint8)
    for i, b in enumerate(arr):
        valid = b[b > 0]
        lo, hi = np.percentile(valid, (2, 98)) if valid.size else (0, 1)
        out[i] = np.clip((b - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return out


def is_mostly_blank(chip, thresh=0.5):
    """Skip nodata borders (pure black/white)."""
    g = chip.mean(axis=2)
    return ((g < 3) | (g > 252)).mean() > thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="chips")
    ap.add_argument("--size", type=int, default=256, help="chip size in pixels")
    ap.add_argument("--stride", type=int, default=192, help="step between chips (overlap = size-stride)")
    ap.add_argument("--date", default="", help="capture date to record, e.g. 2024-01-12")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.csv")
    new = not os.path.exists(index_path)
    with open(index_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["chip", "source", "date", "x", "y", "size", "lon", "lat"])
        for path in args.images:
            arr, to_lonlat = load(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            H, W = arr.shape[:2]
            n = 0
            for y in range(0, max(H - args.size, 0) + 1, args.stride):
                for x in range(0, max(W - args.size, 0) + 1, args.stride):
                    chip = arr[y:y + args.size, x:x + args.size]
                    if chip.shape[0] < args.size or chip.shape[1] < args.size or is_mostly_blank(chip):
                        continue
                    name = f"{stem}_{x}_{y}.png"
                    Image.fromarray(chip).save(os.path.join(args.out, name))
                    lon = lat = ""
                    if to_lonlat:
                        lon, lat = to_lonlat(x + args.size / 2, y + args.size / 2)
                        lon, lat = f"{lon:.6f}", f"{lat:.6f}"
                    w.writerow([name, path, args.date, x, y, args.size, lon, lat])
                    n += 1
            print(f"{path}: {W}x{H} -> {n} chips")


if __name__ == "__main__":
    main()
