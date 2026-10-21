"""Score every candidate break against a set of proven point/reef breaks, and call
each spot a left or a right from the traced wave physics.

Same components as compare_pentire_boilers/compare_spots.py, minus the satellite
imagery (too slow and too weak at 10 m to run for 40+ spots), so the weights are
renormalised:  geometry 0.1875, seabed 0.25, physics 0.375, climate 0.1875.

Left/right: for each swell direction, take the rays that broke within 400 m of the
take-off and fit break time against position along the coast. If the break line
runs towards a surfer's right as they ride in (facing the beach), it's a right.

  python compare_candidates/compare_many.py          (run from surf-finder)
"""
import csv, html, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "compare_pentire_boilers"))
from point_breaks import trace, coast_points, static_features, physics_features  # noqa: E402
from compare_spots import (load_seabed, climate, climate_stats, sim_diff, sim_ratio,  # noqa: E402
                           angdiff, bearing, PERIOD, HEIGHT)

W = dict(geometry=0.1875, seabed=0.25, physics=0.375, climate=0.1875)
D230 = list(range(230, 341, 10))
AREAS = {
    "trevose": dict(seabed="data/seabed_trevose.tif", dirs=D230, offshore=(50.62, -5.00)),
    "bude": dict(seabed="data/seabed_bude.tif", dirs=D230, offshore=(50.83, -4.80)),
    "boilers": dict(seabed="compare_pentire_boilers/data/seabed_boilers.tif", dirs=D230, offshore=(30.62, -10.00)),
    "anchor": dict(seabed="compare_candidates/data/seabed_anchor.tif", dirs=D230, offshore=(30.55, -9.86)),
    "porthleven": dict(seabed="compare_candidates/data/seabed_porthleven.tif", dirs=list(range(190, 301, 10)),
                       offshore=(50.03, -5.40)),
}
BBOX = {"trevose": (-5.25, 50.38, -4.86, 50.65), "bude": (-4.85, 50.70, -4.48, 50.95)}
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def spot_list():
    """Candidates from the Google Maps CSV and proven references (with their known direction)."""
    cands = []
    with open(os.path.join(ROOT, "google_maps", "cornwall_surf_candidates.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            area = "trevose" if r["name"].startswith("Trevose") else "bude"
            cands.append(dict(name=r["name"], lat=float(r["latitude"]), lon=float(r["longitude"]), area=area))
    refs = [dict(name="Boilers", lat=30.62456, lon=-9.87758, area="boilers", known="right"),
            dict(name="Anchor Point", lat=30.544045, lon=-9.727768, area="anchor", known="right")]
    with open(os.path.join(ROOT, "known_breaks.csv")) as f:
        for r in csv.DictReader(f):
            if r["type"] not in ("reef", "point"):
                continue
            la, lo = float(r["lat"]), float(r["lon"])
            area = "porthleven" if r["name"] == "Porthleven" else next(
                (a for a, b in BBOX.items() if b[0] <= lo <= b[2] and b[1] <= la <= b[3]), None)
            if area:
                refs.append(dict(name=r["name"], lat=la, lon=lo, area=area, known=r["direction"] or "?"))
    return cands, refs


def handedness(br, R, C, res, facing):
    """Sign > 0: break line runs to the surfer's right (a right-hander)."""
    near = np.hypot(br["row"] - R, br["col"] - C) * res < 400
    if near.sum() < 4:
        return 0.0, 0.0
    b = np.radians(facing - 90)                         # surfer's right, riding towards the beach
    u = np.array([-np.cos(b), np.sin(b)])
    s = ((br["row"][near] - R) * u[0] + (br["col"][near] - C) * u[1]) * res
    t = br["t"][near]
    if np.std(s) < 1e-6 or np.std(t) < 1e-6:
        return 0.0, 0.0
    r = float(np.corrcoef(s, t)[0, 1])
    return float(np.sign(r)), abs(r)


def area_physics(area, spots):
    cache = os.path.join(CACHE, f"physics_{area}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import rasterio
    from rasterio.warp import transform as warp
    from scipy import ndimage
    cfg = AREAS[area]
    depth, tf, crs = load_seabed(os.path.join(ROOT, cfg["seabed"]))
    res = abs(tf.a)
    rr, cc, land = coast_points(depth, res, 60)
    xs, ys = warp("EPSG:4326", crs, [s["lon"] for s in spots], [s["lat"] for s in spots])
    sp_rc = [rasterio.transform.rowcol(tf, x, y) for x, y in zip(xs, ys)]
    near = {}
    for s, (r0, c0) in zip(spots, sp_rc):
        d = np.hypot(rr - r0, cc - c0) * res
        idx = np.nonzero(d < 500)[0]
        near[s["name"]] = (idx, d[idx])
    allidx = np.unique(np.concatenate([v[0] for v in near.values() if len(v[0])] or [np.zeros(0, int)]))
    feats = static_features(depth, land, rr[allidx], cc[allidx], res)
    pos = {g: j for j, g in enumerate(allidx)}
    per_dir, breaks = {}, {}
    for sd in cfg["dirs"]:
        trc = trace(depth, res, sd, PERIOD, HEIGHT)
        ph = physics_features(trc, rr[allidx], cc[allidx], res, HEIGHT)
        ph["score"] = ph["surf_frac"] * np.minimum(ph["ride_m"] / 300, 1) * np.minimum(ph["hb_max"], 2)
        per_dir[sd] = ph
        breaks[sd] = {k: trc[k] for k in ("row", "col", "t", "peel", "Hb")}
        print(f"  {area}: swell from {sd}: {len(trc['row'])} breaking rays", flush=True)
    out = {}
    for s in spots:
        idx, d = near[s["name"]]
        if not len(idx):
            print(f"  {s['name']}: no coastline within 500 m, skipped"); continue
        j = np.array([pos[g] for g in idx])
        mean = np.mean([per_dir[sd]["score"][j] for sd in cfg["dirs"]], 0)
        cand = np.nonzero(d < 200)[0]
        if not len(cand):
            cand = np.array([int(np.argmin(d))])
        k = int(cand[np.argmax(mean[cand])])
        R, C, jk = int(rr[idx[k]]), int(cc[idx[k]]), int(j[k])

        def facing(radius_m):
            rad = int(radius_m / res)
            r1, c1 = max(R - rad, 0), max(C - rad, 0)
            lr, lc = np.nonzero(land[r1:R + rad + 1, c1:C + rad + 1])
            inside = np.hypot(lr + r1 - R, lc + c1 - C) <= rad
            return bearing(R - (lr[inside] + r1).mean(), C - (lc[inside] + c1).mean())
        fac = facing(1000)
        dist = np.arange(0, 2001, 50)
        b = np.radians(fac)
        prof = ndimage.map_coordinates(np.nan_to_num(depth), [R - np.cos(b) * dist / res, C + np.sin(b) * dist / res], order=1)
        pdir, hand = {}, {}
        for sd in cfg["dirs"]:
            pdir[sd] = {f: float(v[jk]) for f, v in per_dir[sd].items()}
            hand[sd] = handedness(breaks[sd], R, C, res, fac)
        out[s["name"]] = dict(
            area=area, row=R, col=C, dist_from_coord_m=float(d[k]), facing=fac, headland_axis=facing(2500),
            land_frac_800=float(feats["land_frac_800m"][jk]), land_frac_2000=float(feats["land_frac_2000m"][jk]),
            roughness=float(feats["seabed_roughness"][jk]), profile=prof.tolist(),
            per_dir={str(k2): v for k2, v in pdir.items()}, hand={str(k2): v for k2, v in hand.items()})
    json.dump(out, open(cache, "w"))
    return out


def summarise(p):
    dirs = sorted(int(k) for k in p["per_dir"])
    sc = np.array([p["per_dir"][str(d)]["score"] for d in dirs])
    best = dirs[int(np.argmax(sc))] if sc.max() > 0 else None
    win = [d for d, v in zip(dirs, sc) if sc.max() > 0 and v >= 0.5 * sc.max()]
    # Hand: votes from the working directions, weighted by physics score and fit strength.
    vote = sum(p["per_dir"][str(d)]["score"] * p["hand"][str(d)][0] * p["hand"][str(d)][1] for d in win)
    tot = sum(p["per_dir"][str(d)]["score"] * p["hand"][str(d)][1] for d in win)
    conf = abs(vote) / tot if tot > 0 else 0.0
    call = "unclear" if tot == 0 else ("right" if vote > 0 else "left") if conf >= 0.4 else "peak (both ways)"
    return dict(dirs=dirs, best=best, window=win, hand=call, hand_conf=conf)


def score_pair(A, B, sA, sB, cA, cB):
    comp = {}
    domA = climate_stats(cA, [], A["facing"])["dominant_swell"]
    domB = climate_stats(cB, [], B["facing"])["dominant_swell"]
    comp["geometry"] = np.mean([sim_diff(A["land_frac_800"], B["land_frac_800"], 0.3),
                                sim_diff(A["land_frac_2000"], B["land_frac_2000"], 0.3),
                                sim_diff(abs(angdiff(A["headland_axis"], domA)), abs(angdiff(B["headland_axis"], domB)), 45)])
    rms = float(np.sqrt(np.mean((np.array(A["profile"]) - np.array(B["profile"])) ** 2)))
    comp["seabed"] = 0.6 * max(0, 1 - rms / 10) + 0.4 * sim_ratio(A["roughness"], B["roughness"])
    if sA["best"] is None or sB["best"] is None:
        comp["physics"] = 0.0
    else:
        dpeel, dhb = [], []
        for o in range(-60, 61, 10):
            pa, pb = A["per_dir"].get(str(sA["best"] + o)), B["per_dir"].get(str(sB["best"] + o))
            if pa and pb and pa["n_break"] > 0 and pb["n_break"] > 0:
                if np.isfinite(pa["peel_med"]) and np.isfinite(pb["peel_med"]):
                    dpeel.append(pa["peel_med"] - pb["peel_med"])
                dhb.append(pa["hb_max"] - pb["hb_max"])
        peel_sim = max(0, 1 - np.median(np.abs(dpeel)) / 25) if len(dpeel) >= 3 else 0.0
        hb_sim = max(0, 1 - np.median(np.abs(dhb)) / 0.5) if len(dhb) >= 3 else 0.0
        comp["physics"] = np.mean([peel_sim, hb_sim, sim_ratio(len(sA["window"]), len(sB["window"]))])
    kA = climate_stats(cA, sA["window"], A["facing"]); kB = climate_stats(cB, sB["window"], B["facing"])
    comp["climate"] = np.mean([sim_ratio(kA["p_working"], kB["p_working"]),
                               sim_ratio(kA["p_working_offshore"], kB["p_working_offshore"])])
    total = 100 * sum(comp[k] * w for k, w in W.items())
    return total, comp


def km(a, b):
    return np.hypot((a["lat"] - b["lat"]) * 111.0, (a["lon"] - b["lon"]) * 111.0 * np.cos(np.radians(a["lat"])))


def main():
    cands, refs = spot_list()
    only = sys.argv[1:]                      # optionally just build the physics cache for some areas
    P, CL = {}, {}
    for area in AREAS:
        if only and area not in only:
            continue
        spots = [s for s in cands + refs if s["area"] == area]
        print(f"== {area}: {len(spots)} spots", flush=True)
        P.update(area_physics(area, spots))
        lat, lon = AREAS[area]["offshore"]
        CL[area] = climate(f"area_{area}", dict(lat=lat, lon=lon, offshore=(lat, lon)))
    if only:
        return
    S = {n: summarise(p) for n, p in P.items()}
    refs = [r for r in refs if r["name"] in P]
    cands = [c for c in cands if c["name"] in P]

    def sc(a, b):
        return score_pair(P[a["name"]], P[b["name"]], S[a["name"]], S[b["name"]], CL[a["area"]], CL[b["area"]])

    # Validation: references against each other, and hand calls against known directions.
    with open(os.path.join(HERE, "references_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["reference", "known_direction", "called", "hand_conf", "best_swell_from"]
                                      + [r["name"] for r in refs])
        for a in refs:
            w.writerow([a["name"], a["known"], S[a["name"]]["hand"], f"{S[a['name']]['hand_conf']:.2f}",
                        S[a["name"]]["best"]] + [f"{sc(a, b)[0]:.0f}" if a is not b else "" for b in refs])
    rows = []
    for c in cands:
        scores = {r["name"]: sc(c, r) for r in refs}
        # A candidate sitting on a known spot would just match itself: skip references within 1 km.
        far = {k: v for k, v in scores.items() if km(c, next(r for r in refs if r["name"] == k)) > 1.0}
        bm = max(far, key=lambda k: far[k][0])
        own = [r["name"] for r in refs if km(c, r) <= 1.0]
        rows.append(dict(name=c["name"], lat=c["lat"], lon=c["lon"], area=c["area"], hand=S[c["name"]]["hand"],
                         hand_conf=round(S[c["name"]]["hand_conf"], 2), best_swell_from=S[c["name"]]["best"],
                         working_window=" ".join(map(str, S[c["name"]]["window"])),
                         closest_proven=bm, closest_score=round(far[bm][0], 1),
                         vs_boilers=round(scores["Boilers"][0], 1), vs_anchor=round(scores["Anchor Point"][0], 1),
                         on_known_spot=" / ".join(own),
                         **{f"best_{k}": round(v * 100) for k, v in far[bm][1].items()},
                         **{f"vs_{r['name']}": round(scores[r["name"]][0], 1) for r in refs}))
    rows.sort(key=lambda r: -r["closest_score"])
    with open(os.path.join(HERE, "ranked_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_kml(rows)
    json.dump(dict(summary={n: {k: v for k, v in s.items()} for n, s in S.items()}), open(os.path.join(HERE, "spots.json"), "w"),
              indent=1, default=float)
    print("\nTop candidates by closest proven match")
    for r in rows[:12]:
        print(f"  {r['name']:<45s} {r['lat']:.5f},{r['lon']:.5f} {r['hand']:<17s} best {r['best_swell_from']}  "
              f"-> {r['closest_proven']} {r['closest_score']:.0f}   vs Boilers {r['vs_boilers']:.0f}")
    print("\nReferences:")
    for a in refs:
        print(f"  {a['name']:<18s} known {a['known']:<15s} called {S[a['name']]['hand']:<17s} "
              f"({S[a['name']]['hand_conf']:.2f})  best {S[a['name']]['best']}")


def write_kml(rows):
    pm = []
    for r in rows:
        style = "#right" if r["hand"] == "right" else "#left" if r["hand"] == "left" else "#peak"
        desc = (f"{r['hand']} (confidence {r['hand_conf']:.2f}); best swell from {r['best_swell_from']}°. "
                f"Closest proven break: {r['closest_proven']} ({r['closest_score']:.0f}/100). "
                f"Similarity to Boilers {r['vs_boilers']:.0f}, Anchor Point {r['vs_anchor']:.0f}."
                + (f" Sits on known spot: {r['on_known_spot']}." if r["on_known_spot"] else "")
                + " Model estimate only; check before you go.")
        pm.append(f"<Placemark><name>{html.escape(r['name'])}</name><description>{html.escape(desc)}</description>"
                  f"<styleUrl>{style}</styleUrl><Point><coordinates>{r['lon']:.6f},{r['lat']:.6f},0</coordinates></Point></Placemark>")
    st = lambda i, c: (f'<Style id="{i}"><IconStyle><color>{c}</color><scale>1.1</scale><Icon>'
                       '<href>http://maps.google.com/mapfiles/kml/shapes/star.png</href></Icon></IconStyle></Style>')
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           '<name>Cornwall candidates matched to proven breaks</name>'
           + st("right", "ff0000ff") + st("left", "ffff8800") + st("peak", "ff00ffff") + "".join(pm) + "</Document></kml>")
    open(os.path.join(HERE, "candidates_matched.kml"), "w", encoding="utf-8").write(kml)


if __name__ == "__main__":
    main()
