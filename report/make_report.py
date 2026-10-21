"""Build an easy-to-read PDF of the surf-finder results.

  python report/make_report.py        (run from surf-finder)
"""
import csv, json, os
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report")
P = lambda *a: os.path.join(ROOT, *a)

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontSize=11, leading=15, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=19, leading=23, spaceBefore=4, spaceAfter=8,
                    textColor=colors.HexColor("#0b3d5c"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13.5, leading=17, spaceBefore=10, spaceAfter=5,
                    textColor=colors.HexColor("#0b3d5c"))
TITLE = ParagraphStyle("title", parent=H1, fontSize=28, leading=33, alignment=TA_CENTER)
SUB = ParagraphStyle("sub", parent=BODY, fontSize=13, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
BOX = ParagraphStyle("box", parent=BODY, fontSize=11, leading=15, spaceAfter=3)
HEAD_BG = colors.HexColor("#0b3d5c")
ALT_BG = colors.HexColor("#eef4f8")
HAND_COL = {"right": "#c0392b", "left": "#1f6fb2", "peak (both ways)": "#b8860b", "unclear": "#777777"}


def compass(d):
    if d in (None, ""):
        return "-"
    d = float(d)
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return f"{int(d)}° ({pts[int((d + 11.25) // 22.5) % 16]})"


def hand_txt(h):
    label = {"peak (both ways)": "both ways"}.get(h, h)
    return f'<font color="{HAND_COL.get(h, "#000000")}"><b>{label}</b></font>'


def table(rows, widths, header=True, font=8.5):
    data = [[Paragraph(str(c), CELL) if not isinstance(c, Paragraph) else c for c in r] for r in rows]
    if header:
        data[0] = [Paragraph(f'<font color="white"><b>{c}</b></font>', CELL) for c in rows[0]]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8d3dc")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    for i in range(1 if header else 0, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), ALT_BG))
    t.setStyle(TableStyle(st))
    return t


def callout(paras, bg="#fff6e0", border="#e0a800"):
    t = Table([[[Paragraph(p, BOX) for p in paras]]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                           ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(border)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def picture(path, width_cm, max_h_cm=22, crop=None):
    im = Image.open(path)
    if crop:
        im = im.crop(crop)
    im = im.convert("RGB")
    im.thumbnail((1800, 1800))
    tmp = os.path.join(OUT, "_img_%d.jpg" % len([f for f in os.listdir(OUT) if f.startswith("_img_")]))
    im.save(tmp, quality=88)
    w, h = im.size
    width = width_cm * cm
    height = width * h / w
    if height > max_h_cm * cm:
        height = max_h_cm * cm; width = height * w / h
    return RLImage(tmp, width=width, height=height)


def place(row, known):
    name = row["name"]
    if name.startswith("Trevose / Newquay #6 "):
        return "Pentire Point (Polzeath)"
    if "(" in name and "possible new spot" not in name:
        return name[name.index("(") + 1:name.index(")")]
    if row.get("on_known_spot"):
        return "at " + row["on_known_spot"]
    la, lo = float(row["lat"]), float(row["lon"])
    d, k = min((np.hypot((la - float(k["lat"])) * 111, (lo - float(k["lon"])) * 111 * np.cos(np.radians(la))), k["name"])
               for k in known)
    return f"{d:.1f} km from {k}"


def short(name):
    return name.replace("Trevose / Newquay", "Trevose").replace(" (possible new spot)", " *")


def main():
    known = list(csv.DictReader(open(P("known_breaks.csv"))))
    R = list(csv.DictReader(open(P("compare_candidates", "ranked_candidates.csv"), encoding="utf-8")))
    pb = json.load(open(P("compare_pentire_boilers", "result.json")))
    for r in R:
        r["where"] = place(r, known)
    rights = sorted([r for r in R if r["hand"] == "right"], key=lambda r: -float(r["vs_boilers"]))[:5]
    new = [r for r in R if "possible new spot" in r["name"]]
    unknown_top = [r for r in R if not r["on_known_spot"] and "(" not in r["name"]][:5]

    doc = SimpleDocTemplate(os.path.join(OUT, "Cornwall_surf_spot_finder_report.pdf"), pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Cornwall surf spot finder report", author="surf-finder")
    s = []

    # ---------------------------------------------------------------- cover + summary
    s += [Spacer(1, 2.5 * cm), Paragraph("Secret Cornwall Surf Spots", TITLE),
          Paragraph("What the surf-finder model found on the Newquay to Padstow and Bude coasts", SUB),
          Spacer(1, 0.4 * cm), Paragraph("October 2026", SUB), Spacer(1, 1.2 * cm)]
    best = rights[0]
    s.append(callout([
        "<b>The short version</b>",
        f"&bull; <b>Most Boilers-like right:</b> {short(best['name'])}, {best['where']} "
        f"({float(best['lat']):.5f}, {float(best['lon']):.5f}). It's a right on {compass(best['best_swell_from'])} swell "
        f"and scores {float(best['vs_boilers']):.0f}/100 against Boilers.",
        f"&bull; <b>Pentire Point vs Boilers:</b> {pb['total']:.0f}/100. The headland and wave size are similar, "
        "but the seabed is very different. Pentire comes out as a right on NW swell.",
        "&bull; <b>Best unknown spots to check:</b> " + "; ".join(
            f"{short(r['name'])} ({r['where']}, a {r['hand'].replace('peak (both ways)', 'peak')} like {r['closest_proven']})"
            for r in unknown_top[:3]) + ".",
        "&bull; The model got the left/right calls right on the known spots it could test: Boilers and Anchor Point "
        "are rights, and six Cornish reefs and points are lefts.",
        "&bull; Everything here is a computer model. Use it as a shortlist to check in Google Earth and in person, "
        "not a guarantee.",
    ]))
    s += [Spacer(1, 0.8 * cm), Paragraph("How to read this report", H2),
          Paragraph("<b>Similarity score (0-100).</b> How alike two spots are in shape and setting: the headland, "
                    "the seabed, how waves break across different swell directions, and how often the right swell "
                    "and wind turn up. A high score means \"built like Boilers\", not \"as good as Boilers\". "
                    "For scale, proven Cornish reefs score between 6 and 52 against Boilers.", BODY),
          Paragraph("<b>Left / right.</b> Which way the wave peels for a surfer riding it towards the beach. "
                    f"{hand_txt('right')} = breaks to your right, like Boilers and Anchor Point. "
                    f"{hand_txt('left')} = breaks to your left. "
                    f"{hand_txt('peak (both ways)')} = a peak with both a left and a right.", BODY),
          Paragraph("<b>Best swell.</b> The direction the swell comes <i>from</i> that the model says suits the spot "
                    "best, for example 300° (WNW). A star (*) after a name means a possible new spot, "
                    "at least 1 km from any break in the list.", BODY),
          PageBreak()]

    # ---------------------------------------------------------------- best bets
    s.append(Paragraph("1. Rights most like Boilers", H1))
    s.append(Paragraph("These are the candidates the model calls rights, ranked by how closely they resemble Boilers. "
                       "Anchor Point, the other famous Moroccan right point, is shown for comparison. "
                       "Boilers and Anchor Point score 66 against each other.", BODY))
    rows = [["#", "Spot", "Where", "Lat, lon", "Best swell", "vs Boilers", "vs Anchor"]]
    for i, r in enumerate(rights, 1):
        rows.append([i, short(r["name"]), r["where"], f"{float(r['lat']):.5f}, {float(r['lon']):.5f}",
                     compass(r["best_swell_from"]), f"<b>{float(r['vs_boilers']):.0f}</b>", f"{float(r['vs_anchor']):.0f}"])
    s.append(table(rows, [0.7 * cm, 2.8 * cm, 3.6 * cm, 3.4 * cm, 2.6 * cm, 1.9 * cm, 1.9 * cm]))
    s.append(Spacer(1, 0.3 * cm))
    s.append(Paragraph("All five want north-west swell. The top two score as high as the most Boilers-like proven "
                       "Cornish reef (Treyarnon, 52).", BODY))

    s.append(Paragraph("2. Possible new spots", H1))
    s.append(Paragraph("These candidates are at least 1 km from any break in the list. Each one is matched to the "
                       "proven break it is most built like.", BODY))
    rows = [["Spot", "Lat, lon", "Left / right", "Best swell", "Most like", "Score", "vs Boilers"]]
    for r in sorted(new, key=lambda r: -float(r["closest_score"])):
        rows.append([short(r["name"]), f"{float(r['lat']):.5f}, {float(r['lon']):.5f}", Paragraph(hand_txt(r["hand"]), CELL),
                     compass(r["best_swell_from"]), r["closest_proven"], f"<b>{float(r['closest_score']):.0f}</b>",
                     f"{float(r['vs_boilers']):.0f}"])
    s.append(table(rows, [2.6 * cm, 3.4 * cm, 2.2 * cm, 2.6 * cm, 2.9 * cm, 1.5 * cm, 1.8 * cm]))
    s.append(PageBreak())

    # ---------------------------------------------------------------- full list
    s.append(Paragraph("3. All 30 candidates", H1))
    s.append(Paragraph("Ranked by how closely each one matches a proven break. Proven breaks within 1 km don't "
                       "count, so a candidate sitting on a known spot can't just match itself. Spots marked "
                       "\"at ...\" are already known breaks, which shows the model rediscovering real waves.", BODY))
    rows = [["Spot", "Where", "Lat, lon", "L / R", "Best swell", "Most like (score)", "vs Boilers"]]
    for r in R:
        rows.append([short(r["name"]), r["where"], f"{float(r['lat']):.5f}, {float(r['lon']):.5f}",
                     Paragraph(hand_txt(r["hand"]), CELL), compass(r["best_swell_from"]),
                     f"{r['closest_proven']} ({float(r['closest_score']):.0f})", f"{float(r['vs_boilers']):.0f}"])
    s.append(table(rows, [2.3 * cm, 3.2 * cm, 3.1 * cm, 1.7 * cm, 2.2 * cm, 3.0 * cm, 1.5 * cm]))
    s.append(PageBreak())

    # ---------------------------------------------------------------- Pentire vs Boilers
    s.append(Paragraph(f"4. Pentire Point vs Boilers: {pb['total']:.0f} / 100", H1))
    s.append(Paragraph("A detailed head-to-head, including satellite images of big-swell days. Pentire's best "
                       f"modelled swell is {compass(pb['best'][0])}; Boilers' is {compass(pb['best'][1])}.", BODY))
    names = {"geometry": "Headland shape", "seabed": "Seabed", "physics": "Wave physics",
             "imagery": "Satellite images", "climate": "Swell and wind climate"}
    weights = dict(geometry=15, seabed=20, physics=30, imagery=20, climate=15)
    rows = [["Part", "Weight", "Score", "What it found"]]
    plain = dict(
        geometry="Both headlands stick out about the same amount. But Boilers sits round the corner from its main "
                 "swell (in the lee of Cap Ghir), while Pentire faces its swell more directly.",
        seabed="Both are about 9 m deep 500 m out. Then Boilers plunges to 46 m by 2 km, while Pentire stays "
               "10-13 m deep across Padstow Bay. This is the biggest difference.",
        physics="Wave size and peel are fairly similar once lined up on each spot's best swell. Pentire peels at "
                "about 55°, Boilers at about 34°. Each works across a swell window about 30-40° wide.",
        imagery="Big-swell satellite photos: waves bend more around Pentire (about 53°) than at Boilers (about 23°). "
                "There's a similar amount of whitewater at the take-off (15% vs 12%).",
        climate="Pentire gets the right swell a bit more often (11% vs 7% of daylight hours), but rarely with "
                "offshore wind (2% vs 5% at Boilers).")
    for k in names:
        rows.append([names[k], f"{weights[k]}%", f"<b>{pb['comp'][k]*100:.0f}</b>", plain[k]])
    s.append(table(rows, [3.2 * cm, 1.5 * cm, 1.3 * cm, 11 * cm]))
    s.append(Spacer(1, 0.3 * cm))
    s.append(Paragraph("<b>In plain words:</b> both headlands stick out about the same amount and make similar-sized "
                       "waves. But Boilers drops into deep water within 1 km, while Pentire sits behind the shallow "
                       "Padstow Bay. Boilers also wraps swell around Cap Ghir, where Pentire faces the swell more "
                       "directly, and Boilers gets good wind with its swell more than twice as often.", BODY))
    s.append(picture(P("compare_pentire_boilers", "side_by_side.png"), 17, 11))
    s.append(Paragraph("Top: the seabed (darker blue = deeper) with incoming swell lines. The purple star is the "
                       "take-off point measured at each spot. Green dots are surfable peeling waves, red dots are "
                       "close-outs and yellow dots are waves too steep to ride. Bottom: how peel angle, physics score, "
                       "breaking height and depth change with swell direction and distance offshore.", SMALL))
    s.append(PageBreak())

    # ---------------------------------------------------------------- maps
    for title, path, crop in [("5. Map: Newquay to Padstow (Trevose)", P("breaks_trevose", "map.png"), (280, 300, 1424, 1540)),
                              ("6. Map: Bude coast", P("breaks_bude", "map.png"), (500, 0, 1334, 1420))]:
        s.append(Paragraph(title, H1))
        s.append(picture(path, 17, 22, crop))
        s.append(Paragraph("Purple circles with numbers are the candidates, numbered as in the tables (Trevose #1 to "
                           "#15, Bude #1 to #15). White lines show swell from 250°. Coloured dots show where waves "
                           "break: green = surfable peel, red = close-out, yellow = too steep.", SMALL))
        s.append(PageBreak())

    # ---------------------------------------------------------------- trust + how to use
    s.append(Paragraph("7. How much to trust this", H1))
    s.append(Paragraph("<b>Checks that passed</b>", H2))
    for t in ["Hiding each known reef in turn, the model still ranked it in the top 2% of the Trevose coast "
              "(Cribbar 2%, Constantine 1%, Treyarnon 3%) and the top 17% of the Bude coast.",
              "Left/right calls: Boilers and Anchor Point came out as rights, which is correct. All six known lefts "
              "the model finds a wave at came out as lefts: Treyarnon, Millook, Tremoutha Haven, Foxhole, Marsland "
              "Left and Compass Point.",
              "The two Moroccan right points, Boilers and Anchor Point, are each other's closest match (66)."]:
        s.append(Paragraph("&bull; " + t, BODY))
    s.append(Paragraph("<b>Limits to keep in mind</b>", H2))
    for t in ["It's a computer model of one swell (3 m at 14 s) with no tide, wind-chop or wave diffraction. "
              "Many reefs only work at one stage of the tide.",
              "Seabed depth near the shore comes from the colour of the sea in satellite photos. It's accurate to "
              "about 1-2 m in Cornwall, but rougher at Boilers and Porthleven.",
              "Daymer Point and Constantine Reef didn't show up as waves in the model, so it can miss real spots.",
              "\"Built like Boilers\" isn't \"as good as Boilers\". Check every spot in Google Earth on a big-swell "
              "day and in person before relying on it."]:
        s.append(Paragraph("&bull; " + t, BODY))
    s.append(Paragraph("8. See the spots on a map", H1))
    s.append(Paragraph("&bull; <b>Google Earth (browser):</b> go to earth.google.com/web, then Projects, New project, "
                       "Import KML file, and pick <i>candidates_matched.kml</i>. Red stars are rights, blue stars "
                       "lefts and yellow stars peaks.", BODY))
    s.append(Paragraph("&bull; <b>Google Earth Pro (desktop):</b> File, then Open, and pick the same file. Turn on the "
                       "clock (historical imagery) to look for big-swell days.", BODY))
    s.append(Paragraph("&bull; <b>Google Maps on your phone:</b> import the file at mymaps.google.com. It appears "
                       "under You, Saved, Maps.", BODY))
    s.append(Paragraph("The file is in C:\\Users\\thoma\\Surf Forecast\\surf-finder\\compare_candidates\\. The "
                       "spreadsheets with every score are in the same folder.", SMALL))

    def footer(canvas, d):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Cornwall surf spot finder  |  page {d.page}")
        canvas.restoreState()
    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)
    for f in os.listdir(OUT):
        if f.startswith("_img_"):
            os.remove(os.path.join(OUT, f))
    print("wrote", os.path.join(OUT, "Cornwall_surf_spot_finder_report.pdf"))


if __name__ == "__main__":
    main()
