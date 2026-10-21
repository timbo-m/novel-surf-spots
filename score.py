"""Score every chip with a trained model and rank candidate surf spots.

  python score.py --model model.pt --chips swellday_chips --out results

Writes results/scores.csv (ranked by P(clean)), results/top.html (gallery of
the best chips) and, if chips came from GeoTIFFs, results/candidates.kml which
you can open in Google Earth to eyeball each spot.
"""
import argparse, csv, html, os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from common import build_model, device, eval_transform
from tile_images import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--chips", default="chips")
    ap.add_argument("--out", default="results")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--min-p", type=float, default=0.5, help="P(clean) needed to go into the KML")
    ap.add_argument("--heatmap", action="store_true", help="also write a P(clean) overlay per source image")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.model, map_location="cpu")
    classes = ck["classes"]
    dev = device()
    model = build_model(ck["arch"], pretrained=False, num_classes=len(classes))
    model.load_state_dict(ck["state"]); model.to(dev).eval()
    tf = eval_transform()

    index = {}
    ip = os.path.join(args.chips, "index.csv")
    if os.path.exists(ip):
        with open(ip) as f:
            index = {r["chip"]: r for r in csv.DictReader(f)}
    chips = sorted(c for c in os.listdir(args.chips) if c.endswith(".png"))

    rows = []
    with torch.no_grad():
        for i in range(0, len(chips), 64):
            batch = chips[i:i + 64]
            x = torch.stack([tf(Image.open(os.path.join(args.chips, c)).convert("RGB")) for c in batch])
            # Average over 4 rotations (test-time augmentation) for steadier scores.
            p = sum(torch.softmax(model(torch.rot90(x, k, (2, 3)).to(dev)), 1) for k in range(4)) / 4
            for c, pr in zip(batch, p.cpu().numpy()):
                meta = index.get(c, {})
                rows.append({"chip": c, **{f"p_{k}": round(float(v), 4) for k, v in zip(classes, pr)},
                             "lon": meta.get("lon", ""), "lat": meta.get("lat", ""),
                             "date": meta.get("date", ""), "source": meta.get("source", ""),
                             "x": meta.get("x", ""), "y": meta.get("y", ""), "size": meta.get("size", "")})
    rows.sort(key=lambda r: -r["p_clean"])

    with open(os.path.join(args.out, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    chips_rel = os.path.relpath(args.chips, args.out)
    cards = "".join(
        f'<figure><img src="{html.escape(os.path.join(chips_rel, r["chip"]))}"><figcaption>'
        f'{r["p_clean"]:.2f} clean<br>{html.escape(r["lat"])} {html.escape(r["lon"])}</figcaption></figure>'
        for r in rows[:args.top])
    with open(os.path.join(args.out, "top.html"), "w") as f:
        f.write("<!doctype html><meta charset=utf-8><title>Top surf candidates</title><style>"
                "body{font:13px system-ui;background:#111;color:#eee}figure{display:inline-block;margin:6px}"
                "img{width:200px;height:200px}</style>" + cards)

    geo = [r for r in rows if r["lat"] and r["p_clean"] >= args.min_p]
    if geo:
        pm = "".join(
            f'<Placemark><name>{r["p_clean"]:.2f}</name><description>{html.escape(r["chip"])} {html.escape(r["date"])}'
            f'</description><Point><coordinates>{r["lon"]},{r["lat"]},0</coordinates></Point></Placemark>'
            for r in geo)
        with open(os.path.join(args.out, "candidates.kml"), "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                    f"<Document><name>Surf candidates</name>{pm}</Document></kml>")

    if args.heatmap:
        by_src = defaultdict(list)
        for r in rows:
            if r["source"] and os.path.exists(r["source"]):
                by_src[r["source"]].append(r)
        for src, rs in by_src.items():
            img = load(src)[0].astype(np.float32)
            heat = np.zeros(img.shape[:2], np.float32); cnt = np.zeros_like(heat)
            for r in rs:
                x, y, s = int(r["x"]), int(r["y"]), int(r["size"])
                heat[y:y + s, x:x + s] += r["p_clean"]; cnt[y:y + s, x:x + s] += 1
            heat = np.where(cnt > 0, heat / np.maximum(cnt, 1), 0)[..., None]
            red = np.array([255, 40, 40], np.float32)
            out = img * (1 - 0.55 * heat) + red * 0.55 * heat
            name = os.path.splitext(os.path.basename(src))[0] + "_heat.jpg"
            Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(os.path.join(args.out, name), quality=85)

    print(f"scored {len(rows)} chips -> {args.out}/scores.csv, top.html"
          + (f", candidates.kml ({len(geo)} spots)" if geo else ""))
    for r in rows[:10]:
        print(f'  {r["p_clean"]:.2f}  {r["chip"]}  {r["lat"]} {r["lon"]}')


if __name__ == "__main__":
    main()
