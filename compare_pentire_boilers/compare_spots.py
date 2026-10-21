"""Compare two surf spots (Pentire Point, Cornwall vs Boilers, Morocco) and give a
0-100 similarity score built from five components, each scored 0-1:

  a) geometry  how far the point sticks out, and the headland axis vs the dominant swell
  b) seabed    depth-vs-distance-offshore profile and seabed roughness near the take-off
  c) physics   peel angle and breaker height vs swell direction (aligned on each spot's
               best direction) and the width of the working swell window
  d) imagery   big-swell Sentinel-2 days: crest wrap around the point, crest spacing,
               whitewater near the take-off
  e) climate   how often swell comes from the working window, and with offshore wind

  python compare_pentire_boilers/compare_spots.py            (run from surf-finder)

Intermediate results are cached in compare_pentire_boilers/cache/.
"""
import csv, json, os, sys
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from point_breaks import trace, coast_points, static_features, physics_features  # noqa: E402

SPOTS = {
    "Pentire Point": dict(lat=50.5852, lon=-4.9319, seabed="data/seabed_trevose.tif",
                          offshore=(50.62, -5.00), min_swell=2.0),
    "Boilers": dict(lat=30.62456, lon=-9.87758, seabed="compare_pentire_boilers/data/seabed_boilers.tif",
                    offshore=(30.62, -10.00), min_swell=2.0),
}
DIRS = list(range(230, 341, 10))
PERIOD, HEIGHT = 14, 3.0
WEIGHTS = dict(geometry=0.15, seabed=0.20, physics=0.30, imagery=0.20, climate=0.15)
STAC = "https://earth-search.aws.element84.com/v1/search"
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def sim_diff(a, b, scale):
    return float(max(0.0, 1 - abs(a - b) / scale))


def sim_ratio(a, b):
    a, b = abs(a), abs(b)
    return 1.0 if max(a, b) == 0 else float(min(a, b) / max(a, b))


def angdiff(a, b):
    return (a - b + 180) % 360 - 180


def bearing(drow, dcol):
    """Compass bearing of a (row, col) step, north = -row."""
    return float(np.degrees(np.arctan2(dcol, -drow)) % 360)


# ------------------------------------------------------------------ physics + geometry
def load_seabed(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64), src.transform, src.crs


def spot_physics(name, s):
    cache = os.path.join(CACHE, f"physics_{name.replace(' ', '_')}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["result"].item()
    import rasterio
    from rasterio.warp import transform as warp
    depth, tf, crs = load_seabed(s["seabed"])
    res = abs(tf.a)
    rr, cc, land = coast_points(depth, res, 60)
    x, y = warp("EPSG:4326", crs, [s["lon"]], [s["lat"]])
    r0, c0 = rasterio.transform.rowcol(tf, x[0], y[0])
    d = np.hypot(rr - r0, cc - c0) * res
    near = np.nonzero(d < 500)[0]
    if not len(near):
        sys.exit(f"{name}: no coastline within 500 m of the spot coordinates")
    feats = static_features(depth, land, rr[near], cc[near], res)
    per_dir, traces = {}, {}
    for sd in DIRS:
        trc = trace(depth, res, sd, PERIOD, HEIGHT)
        ph = physics_features(trc, rr[near], cc[near], res, HEIGHT)
        ph["score"] = ph["surf_frac"] * np.minimum(ph["ride_m"] / 300, 1) * np.minimum(ph["hb_max"], 2)
        per_dir[sd] = ph
        # keep rays and break points within ~6 km of the spot for the figure
        keep = np.hypot(trc["row"] - r0, trc["col"] - c0) * res < 6000
        traces[sd] = dict(row=trc["row"][keep], col=trc["col"][keep], peel=trc["peel"][keep],
                          paths=trc["paths"][:, ::2])
        print(f"  {name}: swell from {sd}: {len(trc['row'])} breaking rays")
    # Take-off = the best-scoring coastal point within 200 m of the listed coordinates
    # (wider and it drifts onto neighbouring beaches).
    mean_score = np.mean([per_dir[sd]["score"] for sd in DIRS], 0)
    cand = np.nonzero(d[near] < 200)[0]
    if not len(cand):
        cand = np.array([int(np.argmin(d[near]))])
    k = int(cand[np.argmax(mean_score[cand])])
    R, C = rr[near][k], cc[near][k]
    # Facing (seaward normal) from the land centroid within 1 km; headland axis within 2.5 km.
    def facing(radius_m):
        rad = int(radius_m / res)
        r1, r2, c1, c2 = max(R - rad, 0), R + rad + 1, max(C - rad, 0), C + rad + 1
        lr, lc = np.nonzero(land[r1:r2, c1:c2])
        inside = np.hypot(lr + r1 - R, lc + c1 - C) <= rad
        return bearing(R - (lr[inside] + r1).mean(), C - (lc[inside] + c1).mean())
    # Depth profile along the facing direction, 0-2 km.
    from scipy import ndimage
    fac = facing(1000)
    dist = np.arange(0, 2001, 50)
    b = np.radians(fac)
    pr = R - np.cos(b) * dist / res; pc = C + np.sin(b) * dist / res
    prof = ndimage.map_coordinates(np.nan_to_num(depth), [pr, pc], order=1)
    result = dict(
        row=int(R), col=int(C), res=res, dist_from_coord_m=float(d[near][k]),
        facing=fac, headland_axis=facing(2500),
        land_frac_300=float(feats["land_frac_300m"][k]), land_frac_800=float(feats["land_frac_800m"][k]),
        land_frac_2000=float(feats["land_frac_2000m"][k]), roughness=float(feats["seabed_roughness"][k]),
        profile_dist=dist, profile=prof,
        per_dir={sd: {f: float(v[k]) for f, v in per_dir[sd].items()} for sd in DIRS},
        traces=traces, crs=str(crs), transform=tuple(tf)[:6],
    )
    np.savez(cache, result=np.array(result, dtype=object))
    return result


# ------------------------------------------------------------------ climate
def climate(name, s):
    cache = os.path.join(CACHE, f"climate_{name.replace(' ', '_')}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    lat, lon = s["offshore"]
    out = dict(time=[], hs=[], tp=[], dirn=[], ws=[], wd=[])
    for y in range(2022, 2026):
        m = requests.get("https://marine-api.open-meteo.com/v1/marine", params=dict(
            latitude=lat, longitude=lon, start_date=f"{y}-01-01", end_date=f"{y}-12-31", timezone="UTC",
            hourly="swell_wave_height,swell_wave_period,swell_wave_direction")).json()["hourly"]
        w = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
            latitude=s["lat"], longitude=s["lon"], start_date=f"{y}-01-01", end_date=f"{y}-12-31",
            timezone="UTC", hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="kn")).json()["hourly"]
        wind = dict(zip(w["time"], zip(w["wind_speed_10m"], w["wind_direction_10m"])))
        for t, hs, tp, dr in zip(m["time"], m["swell_wave_height"], m["swell_wave_period"], m["swell_wave_direction"]):
            if t[11:13] in ("06", "09", "12", "15", "18") and hs is not None and t in wind and wind[t][0] is not None:
                out["time"].append(t); out["hs"].append(hs); out["tp"].append(tp); out["dirn"].append(dr)
                out["ws"].append(wind[t][0]); out["wd"].append(wind[t][1])
    json.dump(out, open(cache, "w"))
    return out


def climate_stats(c, window, facing):
    hs, tp, dr = np.array(c["hs"]), np.array(c["tp"]), np.array(c["dirn"])
    ws, wd = np.array(c["ws"]), np.array(c["wd"])
    big = (hs >= 1.5) & (tp >= 9)
    inwin = np.zeros(len(hs), bool)
    for sd in window:
        inwin |= np.abs(angdiff(dr, sd)) <= 5
    offshore = (np.abs(angdiff(wd, (facing + 180) % 360)) <= 60) | (ws < 5)
    e = hs ** 2 * tp * big
    dom = float(np.degrees(np.arctan2((e * np.sin(np.radians(dr))).sum(), (e * np.cos(np.radians(dr))).sum())) % 360)
    return dict(p_working=float((big & inwin).mean()), p_working_offshore=float((big & inwin & offshore).mean()),
                dominant_swell=dom, n=len(hs))


# ------------------------------------------------------------------ imagery
def conditions_daily(lat, lon, start, end):
    m = requests.get("https://marine-api.open-meteo.com/v1/marine", params=dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end, timezone="UTC",
        hourly="swell_wave_height,swell_wave_period,swell_wave_direction")).json()["hourly"]
    return {t[:10]: (h, p, d) for t, h, p, d in zip(m["time"], m["swell_wave_height"], m["swell_wave_period"],
                                                      m["swell_wave_direction"]) if t.endswith("T11:00") and h is not None}


def imagery_days(name, s, phys, max_days=6):
    """Download small 10 m chips (blue, red, nir, scl) around the spot on big-swell days."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    cache = os.path.join(CACHE, f"imagery_{name.replace(' ', '_')}")
    os.makedirs(cache, exist_ok=True)
    index = os.path.join(cache, "days.json")
    if os.path.exists(index):
        return json.load(open(index))
    tf = rasterio.Affine(*phys["transform"])
    x0, y0 = tf * (phys["col"] + 0.5, phys["row"] + 0.5)
    half = 2500
    grid = dict(crs=rasterio.crs.CRS.from_string(phys["crs"]), width=500, height=500,
                transform=rasterio.transform.from_origin(x0 - half, y0 + half, 10, 10))
    cond = {}
    for y in range(2022, 2026):
        cond.update(conditions_daily(*s["offshore"], f"{y}-01-01", f"{y}-12-31"))
    bbox = [s["lon"] - 0.03, s["lat"] - 0.03, s["lon"] + 0.03, s["lat"] + 0.03]
    body = {"collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
            "datetime": "2022-01-01T00:00:00Z/2025-12-31T23:59:59Z", "query": {"eo:cloud_cover": {"lt": 40}}, "limit": 100}
    items, url = [], STAC
    while url:
        r = requests.post(url, json=body).json()
        items += r.get("features", [])
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    cands = []
    for it in items:
        day = it["properties"]["datetime"][:10]
        c = cond.get(day)
        if c and c[0] >= s["min_swell"] and (c[1] or 0) >= 10:
            cands.append((-c[0], day, it))
    cands.sort(key=lambda t: (t[0], t[2]["properties"]["eo:cloud_cover"]))
    days, seen = [], set()
    for _, day, it in cands:
        if day in seen:
            continue
        layers = []
        try:
            for b in ("scl", "blue", "red", "nir"):
                with rasterio.open(it["assets"][b]["href"]) as src, WarpedVRT(
                        src, nodata=0, resampling=Resampling.nearest if b == "scl" else Resampling.bilinear,
                        **grid) as vrt:
                    layers.append(vrt.read(1).astype(np.float32))
                if b == "scl":
                    scl = layers[0]
                    if (scl == 0).mean() > 0.05 or np.isin(scl, [3, 8, 9, 10]).mean() > 0.15:
                        break
        except Exception as e:
            print(f"  {day}: read failed ({e})"); continue
        if len(layers) < 4:
            continue
        seen.add(day)
        path = os.path.join(cache, f"{day}.npy")
        np.save(path, np.stack(layers))
        h, p, dr = cond[day]
        days.append(dict(day=day, path=path, swell_m=h, period_s=p, swell_dir=dr))
        print(f"  {name} {day}: swell {h} m @ {p} s from {dr}")
        if len(days) >= max_days:
            break
    json.dump(days, open(index, "w"))
    return days


def crest_fft(nir, mask, centre, size, prop_hint, L0):
    """Dominant swell wavelength (m) and propagation bearing (deg) in a square window.
    Returns None unless the spectrum has a clear peak at a physical wavelength."""
    from scipy import ndimage
    r, c = centre
    h = size // 2
    if r - h < 0 or c - h < 0 or r + h > nir.shape[0] or c + h > nir.shape[1]:
        return None
    win = nir[r - h:r + h, c - h:c + h].astype(np.float64)
    m = mask[r - h:r + h, c - h:c + h]
    if m.mean() < 0.6:
        return None
    win = np.where(m, win, np.median(win[m]))          # fill land/foam/cloud so it adds no edges
    hp = win - ndimage.gaussian_filter(win, 12)
    hp = np.where(m, hp, 0) * np.outer(np.hanning(size), np.hanning(size))
    P = np.abs(np.fft.fftshift(np.fft.fft2(hp))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(size, d=10.0))
    FY, FX = np.meshgrid(f, f, indexing="ij")
    k = np.hypot(FX, FY)
    band = (k > 1 / (1.1 * L0)) & (k < 1 / max(0.25 * L0, 40)) & (FX != 0) & (FY != 0)
    P[~band] = 0
    i = np.unravel_index(np.argmax(P), P.shape)
    peak_ratio = P[i] / max(P[band].mean(), 1e-12)
    if peak_ratio < 8:
        return None
    wl = 1 / k[i]
    ax = bearing(FY[i], FX[i])          # propagation axis (180-deg ambiguous)
    if abs(angdiff(ax, prop_hint)) > 90:
        ax = (ax + 180) % 360
    return dict(wavelength=float(wl), prop=float(ax), from_dir=float((ax + 180) % 360), peak=float(peak_ratio))


def imagery_metrics(phys, days, seabed_path):
    """Per day: crest direction offshore and next to the point, the wrap between them,
    crest spacing relative to deep-water wavelength, whitewater near the take-off."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    depth, tf, crs = load_seabed(seabed_path)
    gtf = rasterio.Affine(*phys["transform"])
    x0, y0 = gtf * (phys["col"] + 0.5, phys["row"] + 0.5)
    ctf = rasterio.transform.from_origin(x0 - 2500, y0 + 2500, 10, 10)
    dch = np.full((500, 500), np.nan, np.float32)
    reproject(depth.astype(np.float32), dch, src_transform=tf, src_crs=crs, dst_transform=ctf, dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    water = np.isfinite(dch) & (dch > 0.5)
    fac = np.radians(phys["facing"])
    step = np.array([-np.cos(fac), np.sin(fac)])
    centre = np.array([250, 250])
    off_c = tuple((centre + step * 150).astype(int))      # 1.5 km seaward
    near_c = tuple((centre + step * 35).astype(int))      # 350 m seaward
    yy, xx = np.mgrid[:500, :500]
    ring = (np.hypot(yy - 250, xx - 250) < 60) & water     # within 600 m of take-off
    out = []
    for d in days:
        scl, blue, red, nir = np.load(d["path"])
        cloud = np.isin(scl, [3, 8, 9, 10])
        foam = water & (red > 1100) & (blue > 1100) & ~cloud
        ok = water & ~cloud & ~foam
        L0 = 9.81 * d["period_s"] ** 2 / (2 * np.pi)
        prop_hint = (d["swell_dir"] + 180) % 360
        off = crest_fft(nir, ok, off_c, 128, prop_hint, L0)
        near = crest_fft(nir, ok, near_c, 64, prop_hint, L0)
        rec = dict(day=d["day"], swell_m=d["swell_m"], period_s=d["period_s"], swell_dir=d["swell_dir"],
                   whitewater=float(foam[ring].mean()) if ring.any() else np.nan)
        # The offshore crests must roughly match the ERA5 swell, or the peak isn't swell.
        if off and near and abs(angdiff(off["from_dir"], d["swell_dir"])) <= 45:
            rec.update(off_from=off["from_dir"], near_from=near["from_dir"],
                       wrap=float(abs(angdiff(near["from_dir"], off["from_dir"]))),
                       spacing_near=near["wavelength"] / L0, spacing_off=off["wavelength"] / L0)
        out.append(rec)
    return out


# ------------------------------------------------------------------ scoring
def score(P, I, Cl):
    names = list(P)
    A, B = (P[n] for n in names)
    comp, notes = {}, {}

    # a) geometry
    cA, cB = Cl[names[0]], Cl[names[1]]
    domA = climate_stats(cA, [], A["facing"])["dominant_swell"]; domB = climate_stats(cB, [], B["facing"])["dominant_swell"]
    angA = abs(angdiff(A["headland_axis"], domA))
    angB = abs(angdiff(B["headland_axis"], domB))
    comp["geometry"] = np.mean([sim_diff(A["land_frac_800"], B["land_frac_800"], 0.3),
                                sim_diff(A["land_frac_2000"], B["land_frac_2000"], 0.3),
                                sim_diff(angA, angB, 45)])
    notes["geometry"] = (f"land_frac 800 m {A['land_frac_800']:.2f} vs {B['land_frac_800']:.2f}, 2 km "
                         f"{A['land_frac_2000']:.2f} vs {B['land_frac_2000']:.2f}; headland axis vs dominant swell "
                         f"{angA:.0f} vs {angB:.0f} deg")

    # b) seabed
    rms = float(np.sqrt(np.mean((A["profile"] - B["profile"]) ** 2)))
    comp["seabed"] = 0.6 * max(0, 1 - rms / 10) + 0.4 * sim_ratio(A["roughness"], B["roughness"])
    notes["seabed"] = (f"depth 0-2 km offshore differs by {rms:.1f} m RMS (at 500 m: {A['profile'][10]:.1f} vs "
                       f"{B['profile'][10]:.1f} m; at 2 km: {A['profile'][-1]:.1f} vs {B['profile'][-1]:.1f} m); "
                       f"roughness {A['roughness']:.2f} vs {B['roughness']:.2f}")

    # c) physics, curves aligned on each spot's best direction
    def curves(p):
        sc = np.array([p["per_dir"][d]["score"] for d in DIRS])
        best = DIRS[int(np.argmax(sc))]
        win = [d for d, v in zip(DIRS, sc) if v >= 0.5 * sc.max()] if sc.max() > 0 else []
        return sc, best, win
    scA, bestA, winA = curves(A); scB, bestB, winB = curves(B)
    rel = range(-60, 61, 10)
    dpeel, dhb = [], []
    for o in rel:
        da, db = bestA + o, bestB + o
        if da in A["per_dir"] and db in B["per_dir"]:
            pa, pb = A["per_dir"][da], B["per_dir"][db]
            if np.isfinite(pa["peel_med"]) and np.isfinite(pb["peel_med"]):
                dpeel.append(pa["peel_med"] - pb["peel_med"])
            if pa["n_break"] > 0 and pb["n_break"] > 0:      # no breaking is not a height of 0
                dhb.append(pa["hb_max"] - pb["hb_max"])
    # Median absolute difference: each direction rests on only a few breaking rays, so one
    # spiky direction shouldn't decide the whole curve.
    peel_sim = max(0, 1 - np.median(np.abs(dpeel)) / 25) if len(dpeel) >= 3 else 0.0
    hb_sim = max(0, 1 - np.median(np.abs(dhb)) / 0.5) if len(dhb) >= 3 else 0.0
    win_sim = sim_ratio(len(winA), len(winB))
    comp["physics"] = np.mean([peel_sim, hb_sim, win_sim])
    notes["physics"] = (f"best swell from {bestA} vs {bestB}; working window {len(winA)*10} vs {len(winB)*10} deg; "
                        f"peel at best {A['per_dir'][bestA]['peel_med']:.0f} vs {B['per_dir'][bestB]['peel_med']:.0f} deg; "
                        f"peel-curve sim {peel_sim:.2f} ({len(dpeel)} shared directions), breaker-height sim {hb_sim:.2f}")

    # d) imagery
    def agg(recs):
        w = [r for r in recs if "wrap" in r]
        return dict(n=len(recs), n_fft=len(w),
                    wrap=float(np.median([r["wrap"] for r in w])) if w else np.nan,
                    spacing=float(np.median([r["spacing_near"] for r in w])) if w else np.nan,
                    white=float(np.nanmedian([r["whitewater"] for r in recs])) if recs else np.nan)
    gA, gB = agg(I[names[0]]), agg(I[names[1]])
    parts = []
    if np.isfinite(gA["wrap"]) and np.isfinite(gB["wrap"]):
        # crest spacing is reported but not scored: at 10 m it is not reliable enough
        parts.append(sim_diff(gA["wrap"], gB["wrap"], 30))
    if np.isfinite(gA["white"]) and np.isfinite(gB["white"]):
        parts.append(sim_ratio(gA["white"] + 0.02, gB["white"] + 0.02))
    comp["imagery"] = float(np.mean(parts)) if parts else np.nan
    notes["imagery"] = (f"{gA['n']} vs {gB['n']} big-swell days ({gA['n_fft']} vs {gB['n_fft']} with readable crests); "
                        f"crest wrap at the point {gA['wrap']:.0f} vs {gB['wrap']:.0f} deg; nearshore crest spacing "
                        f"{gA['spacing']:.2f} vs {gB['spacing']:.2f} x deep-water wavelength; whitewater near take-off "
                        f"{gA['white']*100:.0f}% vs {gB['white']*100:.0f}%")

    # e) climate
    sA = climate_stats(cA, winA, A["facing"]); sB = climate_stats(cB, winB, B["facing"])
    comp["climate"] = np.mean([sim_ratio(sA["p_working"], sB["p_working"]),
                               sim_ratio(sA["p_working_offshore"], sB["p_working_offshore"])])
    notes["climate"] = (f"swell >=1.5 m, >=9 s from the working window {sA['p_working']*100:.0f}% vs "
                        f"{sB['p_working']*100:.0f}% of daytime hours; with offshore/light wind "
                        f"{sA['p_working_offshore']*100:.0f}% vs {sB['p_working_offshore']*100:.0f}%; dominant swell "
                        f"{domA:.0f} vs {domB:.0f} deg")

    used = {k: w for k, w in WEIGHTS.items() if np.isfinite(comp[k])}
    total = 100 * sum(comp[k] * w for k, w in used.items()) / sum(used.values())
    extra = dict(best=(bestA, bestB), window=(winA, winB), curves=(scA, scB), img=(gA, gB), clim=(sA, sB))
    return comp, notes, total, extra


def figure(P, I, extra, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(P)
    fig, ax = plt.subplots(2, 3, figsize=(17, 11))
    for j, n in enumerate(names):
        p = P[n]; best = extra["best"][j]
        depth, tf, crs = load_seabed(SPOTS[n]["seabed"])
        R, C, res = p["row"], p["col"], p["res"]
        h = int(5000 / res)
        r1, c1 = max(R - h, 0), max(C - h, 0)
        sub = depth[r1:R + h, c1:C + h]
        a = ax[0, j]
        a.imshow(np.where(sub > 0, sub, np.nan), cmap="Blues", vmin=0, vmax=60,
                 extent=[c1, c1 + sub.shape[1], r1 + sub.shape[0], r1])
        a.imshow(np.where(sub > 0, np.nan, 1), cmap="Greens", vmin=0, vmax=2,
                 extent=[c1, c1 + sub.shape[1], r1 + sub.shape[0], r1])
        t = p["traces"][best]
        paths = t["paths"] / res
        for k in range(0, paths.shape[1], 2):
            q = paths[:, k]
            inside = (q[:, 0] >= r1) & (q[:, 0] < R + h) & (q[:, 1] >= c1) & (q[:, 1] < C + h)
            if inside.sum() > 1:
                a.plot(q[inside, 1], q[inside, 0], color="white", lw=0.4, alpha=0.7)
        col = np.where(t["peel"] < 25, "red", np.where(t["peel"] <= 70, "lime", "gold"))
        okp = np.isfinite(t["peel"])
        a.scatter(t["col"][okp], t["row"][okp], c=col[okp], s=6)
        a.plot(C, R, "m*", ms=16, mec="k")
        a.set_xlim(c1, C + h); a.set_ylim(R + h, r1)
        a.set_title(f"{n}: seabed (0-60 m) and rays, swell from {best}°\n"
                    "break points: green surfable peel, red closeout, yellow too steep")
        a.set_xticks([]); a.set_yticks([])
    a = ax[0, 2]
    rel = np.arange(-60, 61, 10)
    for j, n in enumerate(names):
        best = extra["best"][j]
        pe = [P[n]["per_dir"].get(best + o, {}).get("peel_med", np.nan) for o in rel]
        a.plot(rel, pe, "o-", label=f"{n} (best {best}°)")
    a.axhspan(30, 70, color="green", alpha=0.1)
    a.set_xlabel("swell direction relative to each spot's best (deg)"); a.set_ylabel("median peel angle (deg)")
    a.set_title("Peel angle vs swell direction (green band = surfable)"); a.legend()
    a = ax[1, 0]
    for j, n in enumerate(names):
        a.plot(DIRS, extra["curves"][j], "o-", label=n)
    a.set_xlabel("swell from (deg)"); a.set_ylabel("physics score"); a.set_title("Physics score vs absolute swell direction"); a.legend()
    a = ax[1, 1]
    for j, n in enumerate(names):
        best = extra["best"][j]
        hb = [P[n]["per_dir"].get(best + o, {}).get("hb_max", np.nan) for o in rel]
        a.plot(rel, hb, "o-", label=n)
    a.set_xlabel("swell direction relative to best (deg)"); a.set_ylabel("max breaker height / deep-water height")
    a.set_title("Breaking-height curve"); a.legend()
    a = ax[1, 2]
    for j, n in enumerate(names):
        pr = P[n]
        a.plot(pr["profile_dist"], -pr["profile"], label=f"{n} (faces {pr['facing']:.0f}°)")
    a.set_xlabel("distance offshore from take-off (m)"); a.set_ylabel("seabed (m)"); a.set_title("Depth profile"); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def imagery_figure(I, days, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(days)
    n = max(len(days[k]) for k in names)
    if n == 0:
        return
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6.8), squeeze=False)
    for j, k in enumerate(names):
        for i in range(n):
            a = ax[j, i]; a.axis("off")
            if i >= len(days[k]):
                continue
            d = days[k][i]
            scl, blue, red, nir = np.load(d["path"])
            g = np.clip(red / 2500, 0, 1)
            a.imshow(g, cmap="gray"); a.plot(250, 250, "m+", ms=10)
            a.set_title(f"{k}\n{d['day']} {d['swell_m']} m {d['period_s']:.0f} s {d['swell_dir']}°", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    P, I, Cl, D = {}, {}, {}, {}
    for n, s in SPOTS.items():
        print(f"== {n}")
        P[n] = spot_physics(n, s)
        Cl[n] = climate(n, s)
        D[n] = imagery_days(n, s, P[n])
        I[n] = imagery_metrics(P[n], D[n], s["seabed"])
    comp, notes, total, extra = score(P, I, Cl)
    figure(P, I, extra, os.path.join(HERE, "side_by_side.png"))
    imagery_figure(I, D, os.path.join(HERE, "big_swell_days.png"))
    with open(os.path.join(HERE, "component_scores.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["component", "weight", "score_0_1", "detail"])
        for k, wt in WEIGHTS.items():
            w.writerow([k, wt, "" if not np.isfinite(comp[k]) else f"{comp[k]:.3f}", notes[k]])
        w.writerow(["overall_0_100", "", f"{total:.1f}", ""])
    with open(os.path.join(HERE, "imagery_days.csv"), "w", newline="") as f:
        keys = ["spot", "day", "swell_m", "period_s", "swell_dir", "off_from", "near_from", "wrap",
                "spacing_off", "spacing_near", "whitewater"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for n in I:
            for r in I[n]:
                w.writerow(dict(spot=n, **r))
    names = list(P)
    lines = [f"Similarity {names[0]} vs {names[1]}: {total:.0f} / 100", ""]
    for k, wt in WEIGHTS.items():
        v = "n/a" if not np.isfinite(comp[k]) else f"{comp[k]*100:.0f}"
        lines.append(f"{k:<9s} weight {wt:.2f}  score {v}: {notes[k]}")
    lines += ["", f"take-off points used: " + "; ".join(
        f"{n} {P[n]['dist_from_coord_m']:.0f} m from the listed coordinates, facing {P[n]['facing']:.0f} deg, "
        f"headland axis {P[n]['headland_axis']:.0f} deg" for n in names)]
    txt = "\n".join(lines)
    print(txt)
    json.dump(dict(total=total, comp={k: (None if not np.isfinite(v) else v) for k, v in comp.items()},
                   notes=notes, best=extra["best"], window=extra["window"]),
              open(os.path.join(HERE, "result.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
