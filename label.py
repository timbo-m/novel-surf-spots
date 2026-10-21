"""Tiny browser labeller. Opens http://localhost:8765 showing one chip at a time.

Keys: 1 = clean (lined-up, peeling, directional breaking waves)
      2 = messy (whitewater / closeouts / wind chop, no clean shape)
      3 = no surf (flat water, land, sand, rocks)
      s = skip,  b = back

  python label.py chips            # writes chips/labels.csv as you go
"""
import csv, json, os, sys, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, unquote

from common import CLASSES

CHIPS = sys.argv[1] if len(sys.argv) > 1 else "chips"
LABELS = os.path.join(CHIPS, "labels.csv")


def load_labels():
    if not os.path.exists(LABELS):
        return {}
    with open(LABELS) as f:
        return {r["chip"]: r["label"] for r in csv.DictReader(f)}


def save_labels(d):
    tmp = LABELS + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["chip", "label"])
        for k, v in d.items():
            w.writerow([k, v])
    os.replace(tmp, LABELS)


PAGE = """<!doctype html><meta charset=utf-8><title>Surf labeller</title>
<style>body{font:15px system-ui;background:#111;color:#eee;text-align:center}
img{width:512px;height:512px;image-rendering:pixelated;border:2px solid #444;margin:12px}
.k{display:inline-block;margin:0 10px;padding:4px 10px;border:1px solid #666;border-radius:6px}</style>
<div id=info></div><img id=im><div>
<span class=k>1 clean</span><span class=k>2 messy</span><span class=k>3 no surf</span>
<span class=k>s skip</span><span class=k>b back</span></div><div id=counts></div>
<script>
let chips=[],labels={},i=0;const C=%s;
async function init(){const r=await (await fetch('/state')).json();chips=r.chips;labels=r.labels;
 i=chips.findIndex(c=>!(c in labels));if(i<0)i=0;show()}
function show(){const c=chips[i];im.src='/img/'+encodeURIComponent(c);
 info.textContent=`${i+1}/${chips.length}  ${c}  ${labels[c]?'['+labels[c]+']':''}`;
 const n={};Object.values(labels).forEach(v=>n[v]=(n[v]||0)+1);counts.textContent=JSON.stringify(n)}
async function set(l){const c=chips[i];labels[c]=l;
 await fetch('/label',{method:'POST',body:JSON.stringify({chip:c,label:l})});next()}
function next(){if(i<chips.length-1)i++;show()}
onkeydown=e=>{if(e.key>='1'&&e.key<='3')set(C[+e.key-1]);else if(e.key=='s')next();
 else if(e.key=='b'&&i>0){i--;show()}};init()
</script>""" % json.dumps(CLASSES)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype):
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            self.send(PAGE.encode(), "text/html")
        elif p == "/state":
            chips = sorted(f for f in os.listdir(CHIPS) if f.endswith(".png"))
            self.send(json.dumps({"chips": chips, "labels": load_labels()}).encode(), "application/json")
        elif p.startswith("/img/"):
            name = os.path.basename(unquote(p[5:]))
            with open(os.path.join(CHIPS, name), "rb") as f:
                self.send(f.read(), "image/png")
        else:
            self.send_error(404)

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if d["label"] in CLASSES:
            labels = load_labels(); labels[d["chip"]] = d["label"]; save_labels(labels)
        self.send(b"ok", "text/plain")


if __name__ == "__main__":
    print(f"Labelling {CHIPS} -> {LABELS}\nOpen http://localhost:8765  (Ctrl+C to stop)")
    webbrowser.open("http://localhost:8765")
    HTTPServer(("127.0.0.1", 8765), H).serve_forever()
