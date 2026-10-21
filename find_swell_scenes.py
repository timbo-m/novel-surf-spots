"""Find cloud-free Sentinel-2 passes over Cornwall on big-swell days, and
optionally download a crop of each one as a GeoTIFF ready for tile_images.py.

Swell + wind come from Open-Meteo (free, no key: ERA5 reanalysis), scenes from
the open Earth Search STAC catalogue (Sentinel-2 L2A, free, no key).

  python find_swell_scenes.py --start 2023-10-01 --end 2024-03-31 --min-swell 2.5
  python find_swell_scenes.py ... --download scenes/     # needs rasterio

Sentinel-2 passes over Cornwall at about 11:00 UTC, so conditions are taken at 11:00.
"""
import argparse, csv, math, os
import requests

# Rough bounding boxes (lon_min, lat_min, lon_max, lat_max) and an offshore point
# for each to read the swell from. Add your own.
AREAS = {
    "north_coast": ((-5.75, 50.05, -4.40, 50.75), (50.45, -5.25)),   # Sennen -> Bude
    "south_coast": ((-5.75, 49.95, -4.20, 50.35), (50.00, -5.00)),   # Lands End -> Rame
    "penwith":     ((-5.75, 50.02, -5.40, 50.22), (50.10, -5.85)),
    "newquay":     ((-5.20, 50.35, -4.90, 50.48), (50.48, -5.15)),
    # widened west to about 40 m depth (Bude Bay is shallow) and to Boscastle/Morwenstow
    # Boilers, Cap Ghir, Morocco (30.6246, -9.8776 per mondo.surf), for comparison runs
    "boilers":     ((-10.00, 30.52, -9.76, 30.72), (30.62, -10.00)),
    # Anchor Point, Taghazout (30.5440, -9.7278 per mondo.surf), for comparison runs
    "anchor":      ((-9.86, 30.47, -9.64, 30.62), (30.55, -9.86)),
    # Porthleven reef (50.0820, -5.3212), Mount's Bay
    "porthleven":  ((-5.45, 49.98, -5.20, 50.13), (50.03, -5.40)),
    "bude":        ((-4.85, 50.70, -4.48, 50.95), (50.83, -4.70)),
    # Newquay -> Trevose -> Padstow; coast is -5.16..-4.86, widened W/N to reach ~40 m depth
    "trevose":     ((-5.25, 50.38, -4.86, 50.65), (50.55, -5.15)),
}
STAC = "https://earth-search.aws.element84.com/v1/search"


def conditions(lat, lon, start, end):
    """Daily 11:00 UTC swell height/period/direction and wind speed/direction."""
    m = requests.get("https://marine-api.open-meteo.com/v1/marine", params=dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end, timezone="UTC",
        hourly="swell_wave_height,swell_wave_period,swell_wave_direction,wave_height")).json()
    w = requests.get("https://archive-api.open-meteo.com/v1/archive", params=dict(
        latitude=lat, longitude=lon, start_date=start, end_date=end, timezone="UTC",
        hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="kn")).json()
    out = {}
    wind = {t: (s, d) for t, s, d in zip(w["hourly"]["time"], w["hourly"]["wind_speed_10m"],
                                          w["hourly"]["wind_direction_10m"])}
    h = m["hourly"]
    for t, hs, per, dr, wh in zip(h["time"], h["swell_wave_height"], h["swell_wave_period"],
                                  h["swell_wave_direction"], h["wave_height"]):
        if t.endswith("T11:00") and hs is not None:
            ws, wd = wind.get(t, (None, None))
            out[t[:10]] = dict(swell_m=hs, period_s=per, swell_dir=dr, wave_m=wh, wind_kn=ws, wind_dir=wd)
    return out


def scenes(bbox, start, end, max_cloud):
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": 200}
    items, url = [], STAC
    while url:
        r = requests.post(url, json=body).json()
        items += r.get("features", [])
        nxt = [l for l in r.get("links", []) if l.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return items


def offshore_ish(wind_dir, swell_dir):
    """True when the wind blows roughly from land to sea, i.e. opposite the swell (+-60 deg)."""
    if wind_dir is None or swell_dir is None:
        return ""
    diff = abs(((wind_dir - swell_dir) + 180) % 360 - 180)
    return "yes" if diff > 120 else "no"


def download_crop(item, bbox, out_dir):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    href = item["assets"]["visual"]["href"]  # 10 m true-colour COG
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*b, src.transform).intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(window=win)
        prof = src.profile | dict(width=win.width, height=win.height,
                                  transform=src.window_transform(win), driver="GTiff")
    path = os.path.join(out_dir, f'{item["id"]}.tif')
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", default="north_coast", choices=AREAS)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--min-swell", type=float, default=2.0, help="metres")
    ap.add_argument("--max-cloud", type=float, default=25)
    ap.add_argument("--download", metavar="DIR", help="save a GeoTIFF crop of each match here")
    ap.add_argument("--out", default="swell_scenes.csv")
    args = ap.parse_args()

    bbox, (lat, lon) = AREAS[args.area]
    cond = conditions(lat, lon, args.start, args.end)
    found = scenes(bbox, args.start, args.end, args.max_cloud)
    rows = []
    for it in found:
        day = it["properties"]["datetime"][:10]
        c = cond.get(day)
        if not c or c["swell_m"] < args.min_swell:
            continue
        rows.append(dict(date=day, scene=it["id"], cloud=round(it["properties"]["eo:cloud_cover"], 1),
                         **c, offshore=offshore_ish(c["wind_dir"], c["swell_dir"]),
                         visual=it["assets"]["visual"]["href"]))
    rows.sort(key=lambda r: -r["swell_m"])
    if not rows:
        print("no matching scenes; try a lower --min-swell or higher --max-cloud"); return
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} scenes with swell >= {args.min_swell} m and cloud < {args.max_cloud}% -> {args.out}")
    for r in rows[:15]:
        print(f'  {r["date"]}  swell {r["swell_m"]} m @ {r["period_s"]} s from {r["swell_dir"]}  '
              f'wind {r["wind_kn"]} kn from {r["wind_dir"]} (offshore: {r["offshore"]})  cloud {r["cloud"]}%')
    if args.download:
        os.makedirs(args.download, exist_ok=True)
        for it in found:
            if any(r["scene"] == it["id"] for r in rows):
                print("downloading", download_crop(it, bbox, args.download))


if __name__ == "__main__":
    main()
