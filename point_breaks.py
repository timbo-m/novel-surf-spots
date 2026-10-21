"""Find stretches of coast whose seabed shape could make a point or reef break
come alive on big swell days.

Two halves:

  1. Physics. Big swell is traced as wave rays across the seabed. Rays bend
     towards shallow water (refraction), grow as they shoal and break when
     height reaches about 0.78 of the depth. At each break point we get the
     breaker height and the *peel angle*: the angle between the wave crest and
     the line of breaking. Under about 25 degrees the wave closes out; about
     30-70 degrees is a surfable peel; above that it is too fast or crumbles.
     A long run of surfable peel next to a headland is what a point break is.

  2. Learning. Every coastal point gets a set of features: how far it sticks
     out, the seabed slope at different distances offshore, and the physics
     results for each swell direction. A small model learns which of those
     patterns match known Cornish reefs and points (known_breaks.csv). It is
     tested by hiding one known break at a time and checking where it ranks.

  # Combine colour-derived shallow depth with surveyed depth for deeper water
  python point_breaks.py prepare --sdb depth/depth.tif --fill emodnet.tif --out seabed.tif

  # Scan the coast for several big-swell directions
  python point_breaks.py scan --seabed seabed.tif --known known_breaks.csv --out breaks/ \
      --swell-dirs 250 270 290 310 --period 14 --height 3

Outputs in breaks/: candidates.csv (ranked), candidates.kml (open in Google
Earth), map.png (seabed, breaking points coloured by peel angle, top
candidates) and report.txt (the held-out test).

The seabed raster is depth in metres, positive below the water; land is <= 0
or NaN. It must reach into deep water (about 40 m+) on the swell side, or the
rays start in water that is already shallow.
"""
import argparse, csv, html, os, sys
import numpy as np
from scipy import ndimage

G = 9.81


# ------------------------------------------------------------------ waves
def wave_speeds(depth, T):
    """Phase speed c and group speed cg for period T over a depth grid (Fenton & McKee)."""
    L0 = G * T ** 2 / (2 * np.pi)
    d = np.maximum(depth, 0.05)
    L = L0 * np.tanh((2 * np.pi * d / L0) ** 0.75) ** (2 / 3)
    k = 2 * np.pi / L
    kd = np.minimum(k * d, 30)
    n = 0.5 * (1 + 2 * kd / np.sinh(2 * kd))
    c = L / T
    return c, c * n, L0 / T, 0.5 * L0 / T


def trace(depth, res, from_deg, T, H0, spacing=None, dt=None, gamma=0.78):
    """Trace swell rays across the grid.

    depth: 2-D array, positive = water depth (m); land <= 0 or NaN.
    res: pixel size (m). from_deg: compass direction the swell comes FROM.
    Returns a dict of arrays for every ray that broke: row, col, Hb, db, peel (deg),
    plus all ray paths (for drawing).
    """
    H, W = depth.shape
    land = ~np.isfinite(depth) | (depth <= 0)
    dwork = np.where(land, 0.05, depth)
    c, cg, c0, cg0 = wave_speeds(dwork, T)
    # Speeds outside the grid / in very deep water are deep-water speeds.
    gy, gx = np.gradient(c, res)                         # dc/drow, dc/dcol per metre
    dgy, dgx = np.gradient(ndimage.gaussian_filter(dwork, 1.5), res)

    spacing = spacing or 2 * res
    dt = dt or 0.4 * res / c0
    b = np.deg2rad((from_deg + 180) % 360)              # propagation bearing
    k = np.array([-np.cos(b), np.sin(b)])               # (drow, dcol): north = -row
    # Seed a line of rays perpendicular to k, upstream of the grid.
    centre = np.array([H / 2, W / 2]) * res
    diag = np.hypot(H, W) * res
    perp = np.array([k[1], -k[0]])
    offs = np.arange(-diag / 2, diag / 2, spacing)
    pos = centre - k * (diag / 2 + 10 * res) + offs[:, None] * perp   # metres (row, col)
    dirv = np.tile(k, (len(offs), 1))
    alive = np.ones(len(offs), bool)
    broke = np.zeros(len(offs), bool)
    out = {f: np.full(len(offs), np.nan) for f in ("row", "col", "Hb", "db", "peel", "t")}
    paths = []
    width0 = spacing

    def sample(arr, p, fill):
        r, cc = p[:, 0] / res, p[:, 1] / res
        inside = (r >= 0) & (r <= H - 1) & (cc >= 0) & (cc <= W - 1)
        v = np.full(len(p), fill, float)
        if inside.any():
            v[inside] = ndimage.map_coordinates(arr, [r[inside], cc[inside]], order=1)
        return v, inside

    max_steps = int(3 * diag / (c0 * dt)) + 10
    was_inside = np.zeros(len(offs), bool)
    for step in range(max_steps):
        if not alive.any():
            break
        idx = np.nonzero(alive)[0]
        p = pos[idx]
        cv, inside = sample(c, p, c0)
        was_inside[idx] |= inside
        gyv, _ = sample(gy, p, 0.0); gxv, _ = sample(gx, p, 0.0)
        grad = np.stack([gyv, gxv], 1)
        kv = dirv[idx]
        # Ray equation: direction turns towards lower phase speed.
        gperp = grad - (grad * kv).sum(1, keepdims=True) * kv
        kv = kv - gperp * dt                              # dk/dt = -grad_perp(c)
        kv /= np.linalg.norm(kv, axis=1, keepdims=True)
        pos[idx] = p + kv * cv[:, None] * dt
        dirv[idx] = kv

        # Leaving the grid after having been inside ends a ray.
        r, cc = pos[idx, 0] / res, pos[idx, 1] / res
        outside = (r < 0) | (r > H - 1) | (cc < 0) | (cc > W - 1)
        gone = outside & was_inside[idx]
        alive[idx[gone]] = False
        live = idx[~outside]
        if len(live) == 0:
            continue
        p = pos[live]
        dv, _ = sample(dwork, p, 1000)
        landv, _ = sample(land.astype(float), p, 0)
        cgv, _ = sample(cg, p, cg0)
        # Ray-tube width from neighbours on the same wavefront (same travel time).
        gap = np.linalg.norm(np.diff(pos, axis=0), axis=1)           # i to i+1
        pair = alive[:-1] & alive[1:]
        left = np.where(live > 0, np.where(pair[np.maximum(live - 1, 0)], gap[np.maximum(live - 1, 0)], np.nan), np.nan)
        right = np.where(live < len(offs) - 1,
                         np.where(pair[np.minimum(live, len(gap) - 1)], gap[np.minimum(live, len(gap) - 1)], np.nan), np.nan)
        both = np.stack([left, right]); cnt = np.isfinite(both).sum(0)
        nb = np.where(cnt > 0, np.nansum(both, 0) / np.maximum(cnt, 1), np.nan)
        nb = np.where(np.isfinite(nb), np.minimum(nb, 20 * width0), width0)
        Hs = H0 * np.sqrt(cg0 / np.maximum(cgv, 1e-3)) * np.sqrt(width0 / np.maximum(nb, 1e-3))
        brk = (Hs >= gamma * dv) | (landv > 0.5)
        if brk.any():
            bi = live[brk]
            dgr, _ = sample(dgy, pos[bi], 0.0); dgc, _ = sample(dgx, pos[bi], 0.0)
            gvec = np.stack([dgr, dgc], 1)
            gnorm = np.linalg.norm(gvec, axis=1)
            cosang = np.abs((gvec * dirv[bi]).sum(1)) / np.maximum(gnorm, 1e-9)
            peel = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
            real = landv[brk] < 0.5          # broke in water, not ran into land
            out["row"][bi] = pos[bi, 0] / res; out["col"][bi] = pos[bi, 1] / res
            out["Hb"][bi] = np.where(real, Hs[brk], np.nan)
            out["db"][bi] = dv[brk]
            out["t"][bi] = step * dt         # break time: tells which way the break line peels
            out["peel"][bi] = np.where(real & (gnorm > 1e-4), peel, np.nan)
            alive[bi] = False; broke[bi] = True
        if step % 5 == 0:
            paths.append(pos.copy())
    keep = broke & np.isfinite(out["Hb"])
    res_ = {f: v[keep] for f, v in out.items()}
    res_["paths"] = np.stack(paths) if paths else np.zeros((0, len(offs), 2))
    res_["spacing"] = spacing
    return res_


# ------------------------------------------------------------------ coast features
def coast_points(depth, res, every_m=100):
    land = ~np.isfinite(depth) | (depth <= 0)
    water = ~land
    edge = water & ndimage.binary_dilation(land)
    rr, cc = np.nonzero(edge)
    # Thin to roughly one point per every_m along the coast.
    cell = max(1, int(every_m / res))
    key = (rr // cell) * 100003 + cc // cell
    _, first = np.unique(key, return_index=True)
    return rr[first], cc[first], land


def static_features(depth, land, rr, cc, res):
    """Shape of the land and seabed around each coastal point."""
    feats = {}
    for R in (300, 800, 2000):
        rad = max(1, int(R / res))
        kern = disk(rad)
        frac = conv(land.astype(np.float32), kern / kern.sum(), pad=1.0)
        feats[f"land_frac_{R}m"] = frac[rr, cc]           # low = sticks out (headland)
    d = np.where(land, 0, np.nan_to_num(depth))
    for R in (100, 300, 600, 1200):
        rad = max(1, int(R / res))
        kern = ring(rad)
        m = conv((~land).astype(np.float32), kern)
        s = conv(d.astype(np.float32), kern)
        feats[f"depth_{R}m"] = (s / np.maximum(m, 1e-6))[rr, cc]   # mean water depth on a ring
    feats["slope_near"] = feats["depth_300m"] / 300
    feats["slope_far"] = (feats["depth_1200m"] - feats["depth_300m"]) / 900
    size = max(3, int(200 / res) | 1)
    m1 = ndimage.uniform_filter(d, size); m2 = ndimage.uniform_filter(d * d, size)
    rough = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    feats["seabed_roughness"] = rough[rr, cc]                 # reefs/rock vs smooth sand
    return feats


def conv(a, kern, pad=0.0):
    """Large-kernel convolution via FFT (fast for 2 km radii on 10 m grids)."""
    from scipy.signal import fftconvolve
    r = kern.shape[0] // 2
    ap = np.pad(a, r, constant_values=pad)
    return fftconvolve(ap, kern, mode="same")[r:-r, r:-r]


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.float32)


def ring(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(x, y)
    return ((d <= r) & (d > r - max(1.5, r * 0.15))).astype(np.float32)


def physics_features(trc, rr, cc, res, H0, radius_m=400, peel_lo=30, peel_hi=70):
    """Summarise the breaking waves near each coastal point."""
    n = len(rr)
    out = dict(n_break=np.zeros(n), surf_frac=np.zeros(n), peel_med=np.full(n, np.nan),
               hb_max=np.zeros(n), ride_m=np.zeros(n))
    if len(trc["row"]) == 0:
        return out
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([trc["row"], trc["col"]], 1))
    rad = radius_m / res
    for i, nb in enumerate(tree.query_ball_point(np.stack([rr, cc], 1), rad)):
        if not nb:
            continue
        peel = trc["peel"][nb]; hb = trc["Hb"][nb]
        ok = np.isfinite(peel)
        good = ok & (peel >= peel_lo) & (peel <= peel_hi) & (hb >= 0.6 * H0)
        out["n_break"][i] = len(nb)
        out["surf_frac"][i] = good.sum() / max(ok.sum(), 1)
        out["peel_med"][i] = np.median(peel[ok]) if ok.any() else np.nan
        out["hb_max"][i] = hb.max() / H0
        if good.sum() >= 2:
            pts = np.stack([trc["row"][nb][good], trc["col"][nb][good]], 1) * res
            # Ride length: how far the surfable break line runs.
            span = pts.max(0) - pts.min(0)
            out["ride_m"][i] = float(np.hypot(*span))
    return out


# ------------------------------------------------------------------ commands
def cmd_prepare(a):
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(a.sdb) as src:
        sdb = src.read(1).astype(np.float32); prof = src.profile
    with rasterio.open(a.fill) as f:
        fill = np.full(sdb.shape, np.nan, np.float32)
        reproject(f.read(1).astype(np.float32), fill, src_transform=f.transform, src_crs=f.crs,
                  src_nodata=f.nodata, dst_transform=prof["transform"], dst_crs=prof["crs"],
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
    filld = fill if a.fill_is_depth else -fill
    # Colour depth saturates in deep water and would read 10-15 m there: only trust
    # it where the survey also says the water is fairly shallow.
    sdb = np.where(~np.isfinite(filld) | (filld > a.sdb_max_survey), np.nan, sdb)
    out = np.where(np.isfinite(sdb), sdb, filld)
    # Smooth the seam where colour-derived depth meets survey depth.
    seam = ndimage.binary_dilation(np.isfinite(sdb)) ^ ndimage.binary_erosion(np.isfinite(sdb))
    sm = ndimage.gaussian_filter(np.nan_to_num(out), 2)
    out = np.where(seam & np.isfinite(out) & (out > 0), sm, out)
    if a.resample:
        f = a.resample / abs(prof["transform"].a)
        h, w = int(out.shape[0] / f), int(out.shape[1] / f)
        tf = prof["transform"] * prof["transform"].scale(f, f)
        small = np.full((h, w), np.nan, np.float32)
        reproject(out.astype(np.float32), small, src_transform=prof["transform"], src_crs=prof["crs"],
                  src_nodata=np.nan, dst_transform=tf, dst_crs=prof["crs"], dst_nodata=np.nan,
                  resampling=Resampling.average)
        out = small; prof.update(transform=tf, width=w, height=h)
    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(a.out, "w", **prof) as dst:
        dst.write(out.astype(np.float32), 1)
    print(f"wrote {a.out}  ({np.isfinite(sdb).mean()*100:.0f}% colour-derived, rest surveyed)")


def cmd_scan(a):
    import rasterio
    from rasterio.warp import transform as warp
    with rasterio.open(a.seabed) as src:
        depth = src.read(1).astype(np.float64); tf, crs = src.transform, src.crs
        res = abs(tf.a)
    if a.tide:
        depth = depth + a.tide
    rr, cc, land = coast_points(depth, res, a.every)
    print(f"{len(rr)} coastal points every ~{a.every} m; tracing {len(a.swell_dirs)} swell directions")
    feats = static_features(depth, land, rr, cc, res)
    traces = {}
    best_score = np.zeros(len(rr)); best_dir = np.zeros(len(rr)); best = {}
    for sd in a.swell_dirs:
        trc = trace(depth, res, sd, a.period, a.height)
        traces[sd] = trc
        ph = physics_features(trc, rr, cc, res, a.height)
        for kf, v in ph.items():
            feats[f"{kf}_{sd}"] = v
        score = ph["surf_frac"] * np.minimum(ph["ride_m"] / 300, 1) * np.minimum(ph["hb_max"], 2)
        better = score > best_score
        best_score[better] = score[better]; best_dir[better] = sd
        for kf, v in ph.items():
            best.setdefault(kf, np.zeros(len(rr)) if kf != "peel_med" else np.full(len(rr), np.nan))
            best[kf][better] = v[better]
        print(f"  swell from {sd}: {len(trc['row'])} breaking rays")
    for kf, v in best.items():
        feats[f"best_{kf}"] = v
    feats["physics_score"] = best_score

    xs, ys = rasterio.transform.xy(tf, rr, cc)
    lon, lat = warp(crs, "EPSG:4326", xs, ys)
    lon, lat = np.array(lon), np.array(lat)
    names = sorted(feats)
    X = np.nan_to_num(np.stack([feats[n] for n in names], 1), nan=0.0)

    learned = np.full(len(rr), np.nan)
    report = []
    if a.known:
        learned, report = learn(X, lon, lat, a.known, a.pos_types)
    phys_rank = rank01(best_score)
    final = phys_rank if np.all(np.isnan(learned)) else 0.5 * phys_rank + 0.5 * rank01(learned)

    os.makedirs(a.out, exist_ok=True)
    order = np.argsort(-final)
    # Keep one candidate per ~500 m so a single headland doesn't fill the list.
    picked, taken = [], []
    for i in order:
        p = np.array([rr[i], cc[i]]) * res
        if all(np.hypot(*(p - q)) > a.min_sep for q in taken):
            picked.append(i); taken.append(p)
        if len(picked) >= a.top:
            break
    cols = ["rank", "lat", "lon", "score", "physics_score", "learned_prob", "best_swell_from",
            "best_peel_med", "best_ride_m", "best_hb_max", "land_frac_800m", "depth_300m", "seabed_roughness"]
    with open(os.path.join(a.out, "candidates.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for n, i in enumerate(picked, 1):
            w.writerow([n, f"{lat[i]:.5f}", f"{lon[i]:.5f}", f"{final[i]:.3f}", f"{best_score[i]:.3f}",
                        "" if np.isnan(learned[i]) else f"{learned[i]:.3f}",
                        int(best_dir[i]) if best_score[i] > 0 else "",
                        "" if np.isnan(feats['best_peel_med'][i]) else f"{feats['best_peel_med'][i]:.0f}",
                        f"{feats['best_ride_m'][i]:.0f}", f"{feats['best_hb_max'][i]:.2f}",
                        f"{feats['land_frac_800m'][i]:.2f}", f"{feats['depth_300m'][i]:.1f}",
                        f"{feats['seabed_roughness'][i]:.2f}"])
    write_kml(os.path.join(a.out, "candidates.kml"), picked, lat, lon, final, best_dir, feats)
    draw_map(os.path.join(a.out, "map.png"), depth, land, traces, rr, cc, picked, res)
    rep = [f"coastal points scored: {len(rr)}", f"swell: {a.height} m @ {a.period} s from {a.swell_dirs}"] + report
    open(os.path.join(a.out, "report.txt"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"top candidates -> {a.out}/candidates.csv, candidates.kml, map.png")
    for n, i in enumerate(picked[:10], 1):
        phys = (f"swell from {int(best_dir[i])}  peel {feats['best_peel_med'][i]:.0f} deg  "
                f"ride {feats['best_ride_m'][i]:.0f} m") if best_score[i] > 0 else "no surfable peel traced (learned score only)"
        print(f"  {n:2d}. {lat[i]:.5f}, {lon[i]:.5f}  score {final[i]:.2f}  {phys}")


def rank01(v):
    v = np.nan_to_num(v, nan=-1)
    r = np.argsort(np.argsort(v))
    return r / max(len(v) - 1, 1)


def learn(X, lon, lat, known_csv, pos_types):
    """Positive-unlabelled learning: known reefs/points vs coast far from any known break."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    with open(known_csv) as f:
        known = [r for r in csv.DictReader(f)]
    kl = np.array([[float(r["lat"]), float(r["lon"])] for r in known])
    is_pos = np.array([r["type"].strip().lower() in pos_types for r in known])
    # Only beaches count as known non-examples; mixed/unsure types are left out of training.
    is_neg = np.array([r["type"].strip().lower() == "beach" for r in known])

    def dist_m(la1, lo1, la2, lo2):
        return np.hypot((la1 - la2) * 111_000, (lo1 - lo2) * 111_000 * np.cos(np.radians(la1)))

    D = np.stack([dist_m(lat, lon, k[0], k[1]) for k in kl], 1)       # (points, known)
    near = D < 300
    pos_of = [np.nonzero(near[:, j])[0] for j in range(len(kl))]
    far = D.min(1) > 1500           # away from every listed spot, positive or not
    rng = np.random.default_rng(0)

    def fit(exclude):
        pos_idx = np.concatenate([pos_of[j] for j in range(len(kl)) if is_pos[j] and j != exclude] or [[]]).astype(int)
        neg_known = np.concatenate([pos_of[j] for j in range(len(kl)) if is_neg[j] and j != exclude] or [[]]).astype(int)
        neg = np.nonzero(far)[0]
        neg = rng.choice(neg, min(len(neg), 20 * max(len(pos_idx), 1)), replace=False) if len(neg) else neg
        neg = np.concatenate([neg, neg_known]).astype(int)
        if len(pos_idx) < 3 or len(neg) < 3:
            return None
        Xtr = np.concatenate([X[pos_idx], X[neg]]); ytr = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg))]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))
        return m.fit(Xtr, ytr)

    report = []
    usable = [j for j in range(len(kl)) if is_pos[j] and len(pos_of[j])]
    report.append(f"known breaks in this area: {sum(len(p) > 0 for p in pos_of)} "
                  f"({len(usable)} reefs/points used as positives)")
    pct = []
    for j in usable:
        m = fit(j)
        if m is None:
            continue
        p = m.predict_proba(X)[:, 1]
        best_near = p[pos_of[j]].max()
        pct.append((p < best_near).mean() * 100)
        report.append(f"  hidden: {known[j]['name']:<28s} ranked in top {100 - pct[-1]:.0f}% of the coast")
    if pct:
        report.append(f"median hidden-break rank: top {100 - np.median(pct):.0f}% "
                      f"(random guessing would be top 50%)")
    else:
        report.append("not enough known reefs/points in this area to test (need 2+); using physics only")
    m = fit(-1)
    if m is None:
        return np.full(len(X), np.nan), report
    return m.predict_proba(X)[:, 1], report


def write_kml(path, picked, lat, lon, score, best_dir, feats):
    pm = []
    for n, i in enumerate(picked, 1):
        peel = feats["best_peel_med"][i]
        desc = (f"score {score[i]:.2f}; best swell from {int(best_dir[i]) or 'n/a'} deg; "
                f"peel {'n/a' if np.isnan(peel) else f'{peel:.0f} deg'}; ride {feats['best_ride_m'][i]:.0f} m")
        pm.append(f"<Placemark><name>#{n}</name><description>{html.escape(desc)}</description>"
                  f"<Point><coordinates>{lon[i]:.6f},{lat[i]:.6f},0</coordinates></Point></Placemark>")
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                f"<Document><name>Point break candidates</name>{''.join(pm)}</Document></kml>")


def draw_map(path, depth, land, traces, rr, cc, picked, res):
    """Seabed shading, rays for the first swell direction, break points coloured
    by peel angle (red = closeout, green = surfable, yellow = too steep), top candidates circled."""
    from PIL import Image, ImageDraw
    H, W = depth.shape
    t = np.clip(np.nan_to_num(depth) / 40, 0, 1)
    img = np.stack([(1 - t) * 170 + 20, (1 - t) * 170 + 60, (1 - t) * 110 + 120], -1)
    img[land] = [70, 90, 60]
    im = Image.fromarray(img.astype(np.uint8)); dr = ImageDraw.Draw(im)
    for trc in list(traces.values())[:1]:
        paths = trc["paths"]
        for j in range(0, paths.shape[1], 3):
            pts = [(q[1] / res, q[0] / res) for q in paths[:, j]
                   if 0 <= q[0] / res < H and 0 <= q[1] / res < W]
            if len(pts) > 1:
                dr.line(pts, fill=(225, 225, 225), width=1)
    for trc in traces.values():
        for r, c, pe in zip(trc["row"], trc["col"], trc["peel"]):
            if np.isnan(pe):
                continue
            col = (230, 60, 60) if pe < 25 else (60, 220, 90) if pe <= 70 else (240, 200, 60)
            dr.ellipse((c - 1, r - 1, c + 1, r + 1), fill=col)
    for n, i in enumerate(picked[:15], 1):
        x, y = cc[i], rr[i]
        dr.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(255, 0, 255), width=2)
        dr.text((x + 9, y - 6), str(n), fill=(255, 255, 255))
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="merge colour-derived and surveyed depth")
    p.add_argument("--sdb", required=True); p.add_argument("--fill", required=True)
    p.add_argument("--fill-is-depth", action="store_true",
                   help="fill raster already stores positive depth (default: elevation, negative under water, as EMODnet)")
    p.add_argument("--sdb-max-survey", type=float, default=20,
                   help="ignore colour depth where the survey says deeper than this (m)")
    p.add_argument("--resample", type=float, help="output pixel size in m (e.g. 20 to speed up scan)")
    p.add_argument("--out", default="seabed.tif")
    s = sub.add_parser("scan", help="trace swell and rank the coast")
    s.add_argument("--seabed", required=True)
    s.add_argument("--known", help="CSV name,lat,lon,type of known breaks (see known_breaks.csv)")
    s.add_argument("--pos-types", nargs="+", default=["point", "reef"],
                   help="types in --known that count as what you're looking for")
    s.add_argument("--swell-dirs", nargs="+", type=float, default=[250, 270, 290, 310])
    s.add_argument("--period", type=float, default=14)
    s.add_argument("--height", type=float, default=3.0, help="deep-water swell height, m")
    s.add_argument("--tide", type=float, default=0.0, help="metres to add to every depth (tide level)")
    s.add_argument("--every", type=float, default=100, help="spacing of coastal points, m")
    s.add_argument("--min-sep", type=float, default=500, help="min distance between listed candidates, m")
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--out", default="breaks")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "scan": cmd_scan}[a.cmd](a)


if __name__ == "__main__":
    main()
