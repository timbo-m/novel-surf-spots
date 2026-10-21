"""Estimate shallow seabed depth (about 0-15 m) from the colour of the sea in
Sentinel-2 images: satellite-derived bathymetry.

Blue light gets through water further than green, so as the water gets deeper
the blue/green ratio of the reflected light changes in a predictable way
(Stumpf et al. 2003). The model learns that relationship from places where
depth is already measured, then fills in the rest of the coast.

Three steps:

  # 1. Find calm, clear, low-swell Sentinel-2 passes and download the bands
  python seabed_depth.py fetch --area penwith --start 2023-04-01 --end 2023-09-30 --out calm/

  # 1b. Make a known-depth CSV from a surveyed depth raster
  python seabed_depth.py sample emodnet_tile.tif --elevation --out known_depths.csv

  # 2. Train on known depths (CSV of lon,lat,depth_m, e.g. sampled from
  #    EMODnet / UKHO / Environment Agency LiDAR) and map the depth
  python seabed_depth.py train --scenes calm/*.tif --depths known_depths.csv --out depth/

  # 3. (optional) apply a saved model to more scenes
  python seabed_depth.py predict --model depth/sdb_model.pkl --scenes more/*.tif --out depth2/

Output: depth/depth.tif (metres below the water level at image time, NaN where
it can't see the bottom), depth/depth.png preview and depth/report.txt.

Limits: it only works in clear water. Big-swell days stir up sand and
whitewater, so use calm summer days for depth. Pixels where the bottom isn't
visible (optically deep, foam, glint, cloud) are masked rather than guessed.
"""
import argparse, csv, glob, os, pickle, sys
import numpy as np

BANDS = ["blue", "green", "red", "nir", "scl"]          # Earth Search asset names
SCL_BAD = {0, 1, 3, 8, 9, 10, 11}                        # nodata, saturated, shadow, cloud, cirrus, snow


# ---------------------------------------------------------------- fetch
def cmd_fetch(a):
    import requests, rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
    from rasterio.vrt import WarpedVRT
    from find_swell_scenes import AREAS, STAC, conditions

    bbox, (lat, lon) = AREAS[a.area]
    cond = conditions(lat, lon, a.start, a.end)
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox),
            "datetime": f"{a.start}T00:00:00Z/{a.end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": a.max_cloud}}, "limit": 200}
    items = requests.post(STAC, json=body).json().get("features", [])
    os.makedirs(a.out, exist_ok=True)
    epsg = 32600 + int(((bbox[0] + bbox[2]) / 2 + 180) // 6) + 1      # UTM zone of the area (north)
    bb = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
    x0, y1 = np.floor(bb[0] / 10) * 10, np.ceil(bb[3] / 10) * 10
    ref = dict(crs=rasterio.crs.CRS.from_epsg(epsg),
               transform=rasterio.transform.from_origin(x0, y1, 10, 10),
               width=int(np.ceil((bb[2] - x0) / 10)), height=int(np.ceil((y1 - bb[1]) / 10)))
    # Overlapping tiles (and reprocessed copies) of one pass are mosaicked into one
    # file per day, so a day isn't counted several times in the median.
    daycloud = {}
    for it in items:
        d = it["properties"]["datetime"][:10]
        daycloud[d] = min(daycloud.get(d, 100), it["properties"]["eo:cloud_cover"])
    mosaics, cover = {}, {}
    for it in sorted(items, key=lambda i: (daycloud[i["properties"]["datetime"][:10]],
                                           i["properties"]["datetime"][:10], i["properties"]["eo:cloud_cover"])):
        day = it["properties"]["datetime"][:10]
        c = cond.get(day)
        if not c or c["wave_m"] is None or c["wave_m"] > a.max_wave or (c["wind_kn"] or 0) > a.max_wind:
            continue
        if day not in mosaics and sum(v >= a.min_cover for v in cover.values()) >= a.max_scenes:
            break
        path = os.path.join(a.out, f"S2_{day}.tif")
        # Cornwall straddles UTM zones 29/30 and several MGRS tiles: warp every scene
        # onto one fixed grid so composite() can stack them pixel for pixel.
        def band(b):
            with rasterio.open(it["assets"][b]["href"]) as src, WarpedVRT(
                    src, crs=ref["crs"], transform=ref["transform"], width=ref["width"],
                    height=ref["height"], nodata=0,
                    resampling=Resampling.nearest if b == "scl" else Resampling.bilinear) as vrt:
                return vrt.read(1).astype(np.float32)
        scl = band("scl")
        usable = (scl > 0) & ~np.isin(scl.astype(int), list(SCL_BAD))
        # Judge coverage on the day's mosaic, not per tile: a pass is often split over tiles.
        if usable.mean() < 0.15:
            print(f"{day}  skip {it['id']}: only {usable.mean()*100:.0f}% of the area clear")
            continue
        layers = [scl if b == "scl" else band(b) for b in BANDS]
        if day in mosaics:
            hole = ~mosaics[day][4].astype(bool) | np.isin(mosaics[day][4].astype(int), list(SCL_BAD))
            new = (scl > 0) & hole
            for o, n in zip(mosaics[day], layers):
                o[new] = n[new]
            layers = mosaics[day]
        mosaics[day] = layers
        cover[day] = float(((layers[4] > 0) & ~np.isin(layers[4].astype(int), list(SCL_BAD))).mean())
        prof = dict(driver="GTiff", dtype="float32", count=len(BANDS), compress="deflate", **ref)
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(np.stack(layers))
            dst.update_tags(date=day, wave_m=c["wave_m"], wind_kn=c["wind_kn"])
        print(f"{day}  waves {c['wave_m']} m  wind {c['wind_kn']} kn  cloud "
              f"{it['properties']['eo:cloud_cover']:.0f}%  -> {path}")
    for day, v in cover.items():
        if v < a.min_cover:
            os.remove(os.path.join(a.out, f"S2_{day}.tif"))
            print(f"{day}  dropped: mosaic only {v*100:.0f}% clear")
    if not any(v >= a.min_cover for v in cover.values()):
        print("no calm clear scenes found; widen the dates or relax --max-wave/--max-wind")


# ---------------------------------------------------------------- features
def read_scene(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        return arr, src.profile, src.transform, src.crs


def features(arr):
    """Per-pixel features + a mask of pixels where the bottom might be visible.

    arr: (5, H, W) blue, green, red, nir, scl. L2A reflectance is scaled by 10000
    (with a +1000 offset since processing baseline 04.00, handled by --offset).
    """
    blue, green, red, nir, scl = arr
    water = (nir < 0.05 * 10000) & (green > nir) & (blue > 0)
    bad = np.isin(scl.astype(int), list(SCL_BAD))
    foam = (blue > 0.12 * 10000) & (nir > 0.03 * 10000)        # whitewater / glint
    ok = water & ~bad & ~foam
    n = 1000.0
    eps = 1.0
    lb, lg, lr = (np.log(n * np.maximum(x / 10000, 1e-4)) for x in (blue, green, red))
    psdb_bg = lb / np.where(np.abs(lg) < 1e-3, 1e-3, lg)        # Stumpf blue/green
    psdb_br = lb / np.where(np.abs(lr) < 1e-3, 1e-3, lr)        # blue/red, better in very shallow water
    F = np.stack([psdb_bg, psdb_br, np.log(blue + eps), np.log(green + eps), np.log(red + eps),
                  (green - red) / (green + red + eps)])
    return F, ok


def composite(paths, offset):
    """Median of features over several calm scenes: averages out waves, glint and murk."""
    Fs, oks, prof = [], [], None
    for p in paths:
        arr, prof_p, tf, crs = read_scene(p)
        if prof is None:
            prof, shape = prof_p, arr.shape
        if arr.shape != shape:
            print(f"skip {p}: grid differs from first scene"); continue
        arr[:4] -= offset
        F, ok = features(arr)
        F[:, ~ok] = np.nan
        Fs.append(F); oks.append(ok)
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore", RuntimeWarning)
        F = np.nanmedian(np.stack(Fs), axis=0)
    nvalid = np.sum(np.stack(oks), axis=0)
    return F, nvalid, prof


def sample(F, prof, depths_csv):
    import rasterio
    from rasterio.warp import transform as warp
    with open(depths_csv) as f:
        rows = [r for r in csv.DictReader(f)]
    lon = [float(r["lon"]) for r in rows]; lat = [float(r["lat"]) for r in rows]
    z = np.array([abs(float(r["depth_m"])) for r in rows])
    xs, ys = warp("EPSG:4326", prof["crs"], lon, lat)
    rc = [rasterio.transform.rowcol(prof["transform"], x, y) for x, y in zip(xs, ys)]
    H, W = F.shape[1:]
    X, Y, P = [], [], []
    for (r, c), d in zip(rc, z):
        if 0 <= r < H and 0 <= c < W and np.all(np.isfinite(F[:, r, c])):
            X.append(F[:, r, c]); Y.append(d); P.append((r, c))
    return np.array(X), np.array(Y), np.array(P)


def cmd_sample(a):
    """Turn a surveyed depth raster (EMODnet, UKHO, EA LiDAR) into a known-depth CSV.

    Elevations are usually negative below datum, so depth = -value. Keeps only
    0..max-depth m and thins to --n random points."""
    import rasterio
    from rasterio.warp import transform as warp
    with rasterio.open(a.raster) as src:
        v = src.read(1).astype(np.float64); nod = src.nodata; tf, crs = src.transform, src.crs
    d = -v if a.elevation else v
    ok = np.isfinite(d) & (d >= 0) & (d <= a.max_depth)
    if nod is not None:
        ok &= v != nod
    rr, cc = np.nonzero(ok)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rr), min(a.n, len(rr)), replace=False)
    xs, ys = rasterio.transform.xy(tf, rr[pick], cc[pick])
    lon, lat = warp(crs, "EPSG:4326", xs, ys)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lon", "lat", "depth_m"])
        for lo, la, z in zip(lon, lat, d[rr[pick], cc[pick]]):
            w.writerow([f"{lo:.6f}", f"{la:.6f}", f"{z:.2f}"])
    print(f"wrote {len(pick)} known depths -> {a.out}")


# ---------------------------------------------------------------- train / predict
def cmd_train(a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    paths = sorted(p for s in a.scenes for p in glob.glob(s))
    F, nvalid, prof = composite(paths, a.offset)
    X, y, pos = sample(F, prof, a.depths)
    keep = y <= a.max_depth
    X, y, pos = X[keep], y[keep], pos[keep]
    if len(y) < 30:
        sys.exit(f"only {len(y)} known depths fall on usable water pixels; need at least 30")

    # Spatial block hold-out: whole 1 km blocks go to test, so the score
    # reflects new stretches of coast rather than neighbouring pixels.
    block = (pos[:, 0] // 100) * 10007 + pos[:, 1] // 100
    rng = np.random.default_rng(0)
    ub = np.unique(block); test_blocks = set(rng.choice(ub, max(1, len(ub) // 5), replace=False))
    te = np.array([b in test_blocks for b in block]); tr = ~te
    if te.sum() < 5 or tr.sum() < 20:
        te = rng.random(len(y)) < 0.2; tr = ~te

    lin = LinearRegression().fit(X[tr, :1], y[tr])                       # classic Stumpf
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0).fit(X[tr], y[tr])
    def stats(pred):
        err = pred - y[te]
        return np.sqrt(np.mean(err ** 2)), np.median(np.abs(err)), np.corrcoef(pred, y[te])[0, 1] ** 2
    s_lin, s_gbm = stats(lin.predict(X[te, :1])), stats(gbm.predict(X[te]))
    best = gbm if s_gbm[0] <= s_lin[0] else lin
    nfeat = X.shape[1] if best is gbm else 1
    # Refit the winner on everything.
    best = best.fit(X if nfeat > 1 else X[:, :1], y)

    os.makedirs(a.out, exist_ok=True)
    model = dict(model=best, nfeat=nfeat, max_depth=a.max_depth, offset=a.offset)
    with open(os.path.join(a.out, "sdb_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    write_depth(F, nvalid, prof, model, a.out)
    rep = (f"scenes used: {len(paths)}\nknown depths on usable pixels: {len(y)} "
           f"(train {tr.sum()}, held-out {te.sum()})\n"
           f"held-out  RMSE / median abs err / R2\n"
           f"  Stumpf linear   {s_lin[0]:.2f} m / {s_lin[1]:.2f} m / {s_lin[2]:.2f}\n"
           f"  gradient boost  {s_gbm[0]:.2f} m / {s_gbm[1]:.2f} m / {s_gbm[2]:.2f}\n"
           f"chosen: {'gradient boost' if nfeat > 1 else 'Stumpf linear'}\n")
    open(os.path.join(a.out, "report.txt"), "w").write(rep)
    print(rep)


def cmd_predict(a):
    with open(a.model, "rb") as f:
        model = pickle.load(f)
    paths = sorted(p for s in a.scenes for p in glob.glob(s))
    F, nvalid, prof = composite(paths, model["offset"])
    write_depth(F, nvalid, prof, model, a.out)


def write_depth(F, nvalid, prof, model, out):
    import rasterio
    from PIL import Image
    H, W = F.shape[1:]
    flat = F.reshape(F.shape[0], -1).T
    good = np.all(np.isfinite(flat), axis=1)
    z = np.full(H * W, np.nan, np.float32)
    z[good] = model["model"].predict(flat[good][:, :model["nfeat"]])
    z = z.reshape(H, W)
    # Beyond the trained range the colour stops changing with depth: mask, don't guess.
    z[(z < 0) | (z > model["max_depth"])] = np.nan
    os.makedirs(out, exist_ok=True)
    p = dict(prof, count=2, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(os.path.join(out, "depth.tif"), "w", **p) as dst:
        dst.write(z, 1); dst.write(nvalid.astype(np.float32), 2)
        dst.set_band_description(1, "depth_m"); dst.set_band_description(2, "n_clear_scenes")
    # Preview: shallow = light, deep = dark blue, land/unknown = grey.
    t = np.clip(np.nan_to_num(z, nan=-1) / model["max_depth"], 0, 1)
    rgb = np.stack([(1 - t) * 230 + 10, (1 - t) * 200 + 50, (1 - t) * 90 + 140], -1)
    rgb[~np.isfinite(z)] = 90
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(out, "depth.png"))
    print(f"wrote {out}/depth.tif and depth.png ({np.isfinite(z).mean()*100:.0f}% of pixels mapped)")


def main():
    from find_swell_scenes import AREAS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--area", default="penwith", choices=AREAS)
    f.add_argument("--start", required=True); f.add_argument("--end", required=True)
    f.add_argument("--max-cloud", type=float, default=10)
    f.add_argument("--max-wave", type=float, default=1.0, help="total wave height, m")
    f.add_argument("--max-wind", type=float, default=12, help="knots")
    f.add_argument("--max-scenes", type=int, default=8)
    f.add_argument("--min-cover", type=float, default=0.5, help="min fraction of the area clear and imaged")
    f.add_argument("--out", default="calm")
    t = sub.add_parser("train")
    t.add_argument("--scenes", nargs="+", required=True)
    t.add_argument("--depths", required=True, help="CSV with lon,lat,depth_m")
    t.add_argument("--max-depth", type=float, default=15)
    t.add_argument("--offset", type=float, default=0,
                   help="L2A offset to subtract; Earth Search already removes it (boa_offset_applied), so 0")
    t.add_argument("--out", default="depth")
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True); p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--out", default="depth")
    k = sub.add_parser("sample", help="surveyed depth raster -> known-depth CSV")
    k.add_argument("raster"); k.add_argument("--out", default="known_depths.csv")
    k.add_argument("--n", type=int, default=5000)
    k.add_argument("--max-depth", type=float, default=15)
    k.add_argument("--elevation", action="store_true", help="raster holds elevation (negative = below water)")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "train": cmd_train, "predict": cmd_predict, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
