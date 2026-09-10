"""Play a trained latent space in a browser, live.

Any `.onnx` the `Decoder` node runs — a generator from `fastgan.py`, the linear space from
`pixel_pca.py` — served by a small local server that runs the model per request. Every knob is a real
axis, so what you find here is what the same numbers will do wired into a patch.

    python play.py --model noema.onnx
    python play.py --model pca100.onnx --range unit --port 8011

Nothing is precomputed and nothing leaves the machine. It binds to localhost only.
"""

import argparse
import io
import json
import pathlib
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime
from PIL import Image

PROVIDERS = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__NAME__ — latent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@350;400;500&display=swap">
<style>
:root{--ground:#0b0e11;--surface:#12171b;--edge:#212b32;--edge2:#2c3941;--ink:#dfe7ec;
--dim:#8b98a4;--faint:#5f6c78;--ice:#7fb5d4;--amber:#cf9155;
--sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;
line-height:1.55;font-weight:350;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:26px 22px 70px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap;
margin-bottom:22px}
h1{font-size:17px;font-weight:500;margin:0;letter-spacing:-.005em}
h1 span{font-family:var(--mono);color:var(--faint);font-weight:400;font-size:13px;margin-left:10px}
.stat{font-family:var(--mono);font-size:12px;color:var(--faint);font-variant-numeric:tabular-nums}
.stat b{color:var(--ice);font-weight:500}
.stage{display:grid;gap:22px;grid-template-columns:minmax(0,1.5fr) minmax(280px,1fr);align-items:start}
@media(max-width:860px){.stage{grid-template-columns:1fr}}
.view{background:var(--surface);border:1px solid var(--edge);border-radius:6px;overflow:hidden;
aspect-ratio:1;position:relative}
.view img{width:100%;height:100%;display:block;object-fit:cover}
.view.busy::after{content:'';position:absolute;inset:0;box-shadow:inset 0 0 0 2px var(--ice);
opacity:.35;pointer-events:none}
.side{display:flex;flex-direction:column;gap:16px}
.padbox{position:relative;background:var(--surface);border:1px solid var(--edge2);border-radius:6px;
aspect-ratio:1;cursor:crosshair;touch-action:none}
.dot{position:absolute;width:14px;height:14px;margin:-8px 0 0 -8px;border-radius:50%;
border:2px solid var(--ink);background:rgba(127,181,212,.3);pointer-events:none}
.cross{position:absolute;inset:0;pointer-events:none;
background:linear-gradient(var(--edge),var(--edge)) center/1px 100% no-repeat,
linear-gradient(var(--edge),var(--edge)) center/100% 1px no-repeat}
.lab{display:flex;justify-content:space-between;font-family:var(--mono);font-size:10.5px;
color:var(--faint);letter-spacing:.09em;text-transform:uppercase;margin-bottom:6px}
.row{display:flex;gap:7px;flex-wrap:wrap}
button,select{font-family:var(--mono);font-size:12px;background:var(--surface);color:var(--ink);
border:1px solid var(--edge2);border-radius:4px;padding:7px 11px;cursor:pointer}
button:hover,select:hover{border-color:var(--ice);color:var(--ice)}
button[aria-pressed=true]{background:var(--ice);color:#0b0e11;border-color:var(--ice)}
button:focus-visible,select:focus-visible,.padbox:focus-visible{outline:2px solid var(--ice);outline-offset:2px}
.knob{display:grid;grid-template-columns:52px 1fr 54px;gap:9px;align-items:center;padding:3px 0}
.knob span{font-family:var(--mono);font-size:11px;color:var(--faint)}
.knob output{font-family:var(--mono);font-size:11px;color:var(--dim);text-align:right;
font-variant-numeric:tabular-nums}
.knob.hot span{color:var(--ice)}
input[type=range]{width:100%;accent-color:var(--ice);height:20px;margin:0}
.knobs{border:1px solid var(--edge);border-radius:6px;padding:10px 14px;background:var(--surface);
max-height:340px;overflow-y:auto}
h2{font-size:11px;font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;
color:var(--faint);margin:0 0 8px;font-weight:500}
.hint{color:var(--faint);font-size:12.5px;margin:0}
</style></head><body><div class="wrap">
<header>
  <h1>__NAME__<span>__WIDTH__ axes &middot; __SIZE__ px &middot; __PROVIDER__</span></h1>
  <p class="stat" id="stat">&nbsp;</p>
</header>
<div class="stage">
  <div class="view" id="view"><img id="img" alt="generated frame"></div>
  <div class="side">
    <div>
      <div class="lab"><span id="labx">axis 0 &rarr;</span><span id="laby">&darr; axis 1</span></div>
      <div class="padbox" id="pad" tabindex="0" role="application" aria-label="drag two axes">
        <div class="cross"></div><div class="dot" id="dot"></div>
      </div>
    </div>
    <div class="row">
      <select id="selx" aria-label="horizontal axis"></select>
      <select id="sely" aria-label="vertical axis"></select>
    </div>
    <div class="row">
      <button id="rand">random</button>
      <button id="zero">centre</button>
      <button id="walk" aria-pressed="false">walk</button>
      <button id="save">save png</button>
    </div>
    <div>
      <h2>morph</h2>
      <div class="row">
        <button id="pinA">pin A</button>
        <button id="pinB">pin B</button>
        <button id="loop" aria-pressed="false">loop A&harr;B</button>
      </div>
      <div class="knob"><span>A &rarr; B</span><input type="range" id="mix" min="0" max="1" step="0.004" value="0"><output id="mixv">0.00</output></div>
    </div>
    <div>
      <h2>global</h2>
      <div class="knob"><span>spread</span><input type="range" id="spread" min="0" max="2" step="0.02" value="1"><output id="spreadv">1.00</output></div>
      <div class="knob"><span>step</span><input type="range" id="rate" min="0.002" max="0.12" step="0.002" value="0.03"><output id="ratev">0.030</output></div>
    </div>
    <p class="hint">Drag the pad, or move any axis below. The walk drifts through the whole space at
    the step size above &mdash; the closest thing here to a slow signal driving it.</p>
  </div>
</div>
<h2 style="margin-top:26px">axes &mdash; ordered by how much the model varies along each</h2>
<div class="knobs" id="knobs"></div>
</div>
<script>
const WIDTH = __WIDTH__, SHOWN = Math.min(WIDTH, 24);
const z = new Float32Array(WIDTH);
let xa = 0, ya = 1, busy = false, pending = false, walking = false, t0 = 0;
const img = document.getElementById('img'), view = document.getElementById('view');
const stat = document.getElementById('stat'), dot = document.getElementById('dot');
const pad = document.getElementById('pad');
const SPAN = 2.6;

function render() {
  if (busy) { pending = true; return; }
  busy = true; pending = false; view.classList.add('busy');
  t0 = performance.now();
  const q = '/frame?spread=' + document.getElementById('spread').value +
            '&z=' + Array.from(z, v => v.toFixed(3)).join(',');
  const next = new Image();
  next.onload = () => {
    img.src = next.src; busy = false; view.classList.remove('busy');
    stat.innerHTML = '<b>' + Math.round(performance.now() - t0) + '</b> ms &middot; norm <b>' +
      Math.hypot(...z).toFixed(1) + '</b>';
    if (pending) render();
  };
  next.onerror = () => { busy = false; view.classList.remove('busy'); };
  next.src = q;
}
function sync() {
  for (let k = 0; k < SHOWN; k++) {
    const s = document.getElementById('k' + k);
    if (s) { s.value = z[k]; document.getElementById('o' + k).textContent = z[k].toFixed(2); }
  }
  dot.style.left = ((z[xa] / SPAN + 1) / 2 * 100).toFixed(1) + '%';
  dot.style.top = ((z[ya] / SPAN + 1) / 2 * 100).toFixed(1) + '%';
}
function at(e) {
  const r = pad.getBoundingClientRect();
  z[xa] = Math.max(-SPAN, Math.min(SPAN, ((e.clientX - r.left) / r.width * 2 - 1) * SPAN));
  z[ya] = Math.max(-SPAN, Math.min(SPAN, ((e.clientY - r.top) / r.height * 2 - 1) * SPAN));
  sync(); render();
}
pad.addEventListener('pointerdown', e => { pad.setPointerCapture(e.pointerId); at(e); });
pad.addEventListener('pointermove', e => { if (e.buttons) at(e); });
pad.addEventListener('keydown', e => {
  const d = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]}[e.key];
  if (!d) return;
  e.preventDefault();
  z[xa] = Math.max(-SPAN, Math.min(SPAN, z[xa] + d[0] * 0.15));
  z[ya] = Math.max(-SPAN, Math.min(SPAN, z[ya] + d[1] * 0.15));
  sync(); render();
});

const knobs = document.getElementById('knobs');
for (let k = 0; k < SHOWN; k++) {
  const d = document.createElement('div');
  d.className = 'knob'; d.id = 'row' + k;
  d.innerHTML = '<span>axis ' + k + '</span><input type="range" id="k' + k +
    '" min="-3" max="3" step="0.02" value="0"><output id="o' + k + '">0.00</output>';
  knobs.appendChild(d);
  d.querySelector('input').addEventListener('input', e => {
    z[k] = +e.target.value;
    document.getElementById('o' + k).textContent = z[k].toFixed(2);
    dot.style.left = ((z[xa] / SPAN + 1) / 2 * 100).toFixed(1) + '%';
    dot.style.top = ((z[ya] / SPAN + 1) / 2 * 100).toFixed(1) + '%';
    render();
  });
}
for (const [sel, get, set] of [['selx', () => xa, v => xa = v], ['sely', () => ya, v => ya = v]]) {
  const el = document.getElementById(sel);
  for (let k = 0; k < SHOWN; k++) el.add(new Option('axis ' + k, k));
  el.value = get();
  el.onchange = e => {
    set(+e.target.value);
    document.getElementById('labx').innerHTML = 'axis ' + xa + ' &rarr;';
    document.getElementById('laby').innerHTML = '&darr; axis ' + ya;
    sync();
  };
}
document.getElementById('rand').onclick = () => {
  for (let k = 0; k < WIDTH; k++) z[k] = (Math.random() * 2 - 1) * 1.6;
  sync(); render();
};
document.getElementById('zero').onclick = () => { z.fill(0); sync(); render(); };
document.getElementById('save').onclick = () => {
  const a = document.createElement('a');
  a.href = img.src; a.download = 'latent.png'; a.click();
};
const walkBtn = document.getElementById('walk');
walkBtn.onclick = () => {
  walking = !walking;
  walkBtn.setAttribute('aria-pressed', walking);
  if (walking) drift();
};
const vel = new Float32Array(WIDTH);
function drift() {
  if (!walking) return;
  const r = +document.getElementById('rate').value;
  for (let k = 0; k < WIDTH; k++) {
    vel[k] = vel[k] * 0.94 + (Math.random() * 2 - 1) * r;
    z[k] = Math.max(-3, Math.min(3, z[k] + vel[k]));
  }
  sync(); render();
  setTimeout(drift, 40);
}
for (const [id, out] of [['spread', 'spreadv'], ['rate', 'ratev']]) {
  const el = document.getElementById(id);
  el.addEventListener('input', e => {
    document.getElementById(out).textContent = (+e.target.value).toFixed(id === 'rate' ? 3 : 2);
    if (id === 'spread') render();
  });
}

const A = new Float32Array(WIDTH), B = new Float32Array(WIDTH);
for (let k = 0; k < WIDTH; k++) { A[k] = (Math.random() * 2 - 1) * 1.6; B[k] = (Math.random() * 2 - 1) * 1.6; }

// Direction on the sphere, magnitude straight: a plain lerp between two points passes near the
// origin, which is the model's mean, so the middle of every morph washes out.
function tween(a, b, t) {
  const na = Math.hypot(...a) || 1e-6, nb = Math.hypot(...b) || 1e-6;
  let dot = 0;
  for (let k = 0; k < WIDTH; k++) dot += (a[k] / na) * (b[k] / nb);
  const om = Math.acos(Math.max(-1, Math.min(1, dot))), len = (1 - t) * na + t * nb;
  const out = new Float32Array(WIDTH);
  if (om < 1e-4) { for (let k = 0; k < WIDTH; k++) out[k] = (1 - t) * a[k] + t * b[k]; return out; }
  const s = Math.sin(om), c1 = Math.sin((1 - t) * om) / s, c2 = Math.sin(t * om) / s;
  for (let k = 0; k < WIDTH; k++) out[k] = (c1 * a[k] / na + c2 * b[k] / nb) * len;
  return out;
}
const mix = document.getElementById('mix'), mixv = document.getElementById('mixv');
function morphTo(t) {
  const v = tween(A, B, t);
  for (let k = 0; k < WIDTH; k++) z[k] = v[k];
  mixv.textContent = (+t).toFixed(2);
  sync(); render();
}
mix.addEventListener('input', e => morphTo(+e.target.value));
document.getElementById('pinA').onclick = () => { A.set(z); mix.value = 0; morphTo(0); };
document.getElementById('pinB').onclick = () => { B.set(z); mix.value = 1; morphTo(1); };
let looping = false, phase = 0;
const loopBtn = document.getElementById('loop');
loopBtn.onclick = () => {
  looping = !looping;
  loopBtn.setAttribute('aria-pressed', looping);
  if (looping) { walking = false; walkBtn.setAttribute('aria-pressed', false); cycle(); }
};
function cycle() {
  if (!looping) return;
  phase += +document.getElementById('rate').value * 0.6;
  const t = (1 - Math.cos(phase)) / 2;
  mix.value = t;
  morphTo(t);
  setTimeout(cycle, 40);
}

sync(); render();
</script></body></html>
"""


class Model:
    """One session, one lock — onnxruntime is not re-entrant across threads for a single session."""

    def __init__(self, path, span):
        ready = onnxruntime.get_available_providers()
        self.session = onnxruntime.InferenceSession(str(path), providers=[q for q in PROVIDERS if q in ready])
        self.name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.width = next(int(d) for d in reversed(shape) if isinstance(d, int) and d > 1)
        self.span = span
        self.lock = threading.Lock()
        self.size = self.frame(np.zeros(self.width, np.float32))[1]

    def frame(self, z):
        with self.lock:
            got = self.session.run(None, {self.name: z[None]})[0]
        a = np.asarray(got, dtype=np.float32)
        a = a[0] if a.ndim == 4 else a
        if a.shape[0] <= 4:
            a = a.transpose(1, 2, 0)
        if self.span == "signed":
            a = a * 0.5 + 0.5
        return np.clip(a, 0, 1), a.shape[0]


def serve(model, port, quality):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, kind, cache="no-store"):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/":
                page = (
                    PAGE.replace("__NAME__", pathlib.Path(model.path).name)
                    .replace("__WIDTH__", str(model.width))
                    .replace("__SIZE__", str(model.size))
                    .replace("__PROVIDER__", model.session.get_providers()[0].replace("ExecutionProvider", ""))
                )
                return self.send(page.encode(), "text/html; charset=utf-8")
            if url.path != "/frame":
                self.send_response(404)
                self.end_headers()
                return
            q = urllib.parse.parse_qs(url.query)
            z = np.zeros(model.width, np.float32)
            given = [float(v) for v in q.get("z", [""])[0].split(",") if v]
            z[: min(len(given), model.width)] = given[: model.width]
            z *= float(q.get("spread", ["1"])[0])
            frame, _ = model.frame(z)
            buf = io.BytesIO()
            Image.fromarray((frame * 255).astype(np.uint8)).save(buf, format="JPEG", quality=quality)
            self.send(buf.getvalue(), "image/jpeg")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"  {model.path}")
    print(f"  {model.width} axes, {model.size} square, {model.session.get_providers()[0]}")
    print(f"\n  http://127.0.0.1:{port}\n\n  ctrl-c to stop")
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="An .onnx the Decoder node would run.")
    ap.add_argument("--range", default="signed", choices=["signed", "unit"], help="What the model's output spans.")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--quality", type=int, default=88)
    args = ap.parse_args()
    model = Model(args.model, args.range)
    model.path = args.model
    try:
        serve(model, args.port, args.quality)
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
