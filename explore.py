"""Every training-free instrument in one local page: composite, the IP space, the structure dial.

    python explore.py --space texture-ip-sd15.npz
    python explore.py --space texture-ip-sd15.npz --sources ~/pictures/people --port 8014

Three tabs, all live, nothing precomputed, nothing leaving the machine:

- COMPOSITE — the deterministic layer (`composite.py`'s five modes) at drag rate on CPU. Pick a
  picture and a texture, move the knob, watch the pixels answer.
- SPACE — the IP space drawn per request: a preset or a term mix for the material, one slider per
  fitted concept axis (`concepts.json`), seed and adapter scale. `draft` renders with the LCM LoRA
  in ~0.3 s for searching; `full` turns it off for the real picture.
- TRANSFORM — the two-dial materializer: structure (depth ControlNet) against strength, material
  from the same presets and concepts.

Below the controls sits THE ARCHIVE: every photograph the space was built from, searched by its
own caption, sorted along any fitted concept axis, thumbnailed. A click hands the photograph to
whichever tab is open — as the texture for composite, or as the MATERIAL for space and transform,
because in an IP space every photograph is a prompt. The seven curated picks are only the starting
dropdown; the whole set is the palette.

Models load lazily on the first use of their tab, so the composite tab is up in a second.
"""

import argparse
import io
import math
import json
import pathlib
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

import collections
import contextlib
import re

import composite as cx
from ip_space import PRESETS, slerp
from terms import FIELDS, term_vector
from transform import material_vec, region_masks

from loadlock import LOAD

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>latent textures — explore</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@350;400;500&display=swap">
<style>
:root{--ground:#0b0e11;--surface:#12171b;--edge:#212b32;--edge2:#2c3941;--ink:#dfe7ec;
--dim:#8b98a4;--faint:#5f6c78;--ice:#7fb5d4;--amber:#cf9155;
--sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px;
line-height:1.55;font-weight:350;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:24px 22px 70px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap;margin-bottom:16px}
h1{font-size:17px;font-weight:500;margin:0}
h1 span{font-family:var(--mono);color:var(--faint);font-weight:400;font-size:13px;margin-left:10px}
.stat{font-family:var(--mono);font-size:12px;color:var(--faint);font-variant-numeric:tabular-nums}
.stat b{color:var(--ice);font-weight:500}
.tabs{display:flex;gap:7px;margin-bottom:18px}
.tabs button{font-family:var(--mono);font-size:12px;background:var(--surface);color:var(--ink);
border:1px solid var(--edge2);border-radius:4px;padding:8px 14px;cursor:pointer}
.tabs button[aria-pressed=true]{background:var(--ice);color:#0b0e11;border-color:var(--ice)}
.stage{display:grid;gap:22px;grid-template-columns:minmax(0,1.5fr) minmax(300px,1fr);align-items:start}
@media(max-width:880px){.stage{grid-template-columns:1fr}}
.view{background:var(--surface);border:1px solid var(--edge);border-radius:6px;overflow:hidden;
aspect-ratio:1;position:relative}
.view img{width:100%;height:100%;display:block;object-fit:cover}
.view.busy::after{content:'';position:absolute;inset:0;box-shadow:inset 0 0 0 2px var(--ice);
opacity:.35;pointer-events:none}
.side{display:flex;flex-direction:column;gap:13px}
select,button,input[type=text],input[type=number]{font-family:var(--mono);font-size:12px;
background:var(--surface);color:var(--ink);border:1px solid var(--edge2);border-radius:4px;
padding:7px 10px;max-width:100%}
button{cursor:pointer}
button:hover,select:hover{border-color:var(--ice);color:var(--ice)}
button[aria-pressed=true]{background:var(--ice);color:#0b0e11;border-color:var(--ice)}
.knob{display:grid;grid-template-columns:86px 1fr 56px;gap:9px;align-items:center;padding:2px 0}
.knob span{font-family:var(--mono);font-size:11px;color:var(--faint)}
.knob output{font-family:var(--mono);font-size:11px;color:var(--dim);text-align:right;
font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:var(--ice);height:20px;margin:0}
h2{font-size:11px;font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;
color:var(--faint);margin:10px 0 4px;font-weight:500}
.row{display:flex;gap:7px;flex-wrap:wrap;align-items:center}
.hint{color:var(--faint);font-size:12.5px;margin:4px 0 0}
.lab{font-family:var(--mono);font-size:10.5px;color:var(--faint);letter-spacing:.08em;
text-transform:uppercase;margin-bottom:3px}
.archive{margin-top:26px;border:1px solid var(--edge);border-radius:6px;background:var(--surface);
padding:14px}
.archive .bar2{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.archive input[type=text]{flex:1;min-width:220px}
.thumbs{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:6px}
.thumbs img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px;cursor:pointer;
border:1px solid transparent;display:block}
.thumbs img:hover{border-color:var(--ice)}
.chip{display:inline-flex;gap:7px;align-items:center;font-family:var(--mono);font-size:11.5px;
color:var(--ice);border:1px solid var(--edge2);border-radius:99px;padding:4px 11px;
margin:0 6px 6px 0;cursor:pointer}
.chip b{font-weight:400;color:var(--faint)}
</style></head><body><div class="wrap">
<header>
  <h1>latent textures<span>live — nothing precomputed</span></h1>
  <p class="stat" id="stat">&nbsp;</p>
</header>
<div class="tabs">
  <button id="tab-composite" aria-pressed="true">composite</button>
  <button id="tab-space" aria-pressed="false">space</button>
  <button id="tab-transform" aria-pressed="false">transform</button>
  <button id="tab-walks" aria-pressed="false">walks</button>
</div>
<div class="stage">
  <div class="view" id="view"><img id="img" alt="live frame"></div>
  <div class="side">

    <div id="chips"></div>

    <div data-tab="composite,space,transform,walks">
      <div class="lab">your own images</div>
      <div class="row">
        <button id="upC">+ picture</button>
        <button id="upT">+ texture as material</button>
        <input type="file" id="upCf" accept="image/*" style="display:none">
        <input type="file" id="upTf" accept="image/*" style="display:none">
      </div>
    </div>

    <div data-tab="composite,transform">
      <div class="lab">picture</div>
      <select id="content"></select>
    </div>
    <div data-tab="composite">
      <div class="lab">texture</div>
      <select id="texture"></select>
    </div>
    <div data-tab="composite">
      <div class="lab">mode</div>
      <div class="row" id="modes"></div>
    </div>
    <div data-tab="composite" id="cknobs"></div>

    <div data-tab="walks">
      <h2>the walk — pin two places, slide between</h2>
      <div class="row">
        <button id="pinA">pin A = material</button>
        <button id="pinB">pin B = material</button>
        <button id="aCentre">A &larr; centre</button>
        <button id="bAnti">B &larr; antipode</button>
      </div>
      <div class="knob"><span>A &rarr; B</span><input type="range" id="walkt" min="0" max="1" step="0.02" value="0.5"><output id="walktv">0.50</output></div>
      <p class="hint">set the material below (a preset, terms, or an archive click), pin it as an
      end, then slide. the centre is the photograph the archive keeps trying to take; the antipode
      is everything it refuses. concept sliders bend the whole walk.</p>
    </div>
    <div data-tab="space,transform,walks">
      <div class="lab">material</div>
      <select id="preset"></select>
      <input type="text" id="termmix" placeholder="or terms: encrusted:1, glassy:0.5" style="width:100%;margin-top:6px">
      <div class="knob"><span>push</span><input type="range" id="push" min="0" max="12" step="0.5" value="8"><output id="pushv">8.0</output></div>
      <div class="knob"><span>scale</span><input type="range" id="scale" min="0.2" max="1.4" step="0.05" value="1"><output id="scalev">1.00</output></div>
    </div>
    <div data-tab="space,transform,walks" id="conceptbox">
      <h2>concept axes</h2>
      <div id="concepts"></div>
    </div>
    <div data-tab="space,transform,walks">
      <div class="lab">against &mdash; push away from (optional)</div>
      <input type="text" id="against" placeholder="still water + glassy" style="width:100%">
    </div>
    <div data-tab="transform">
      <h2>the two dials</h2>
      <div class="knob"><span>structure</span><input type="range" id="structure" min="0" max="1.6" step="0.05" value="1"><output id="structurev">1.00</output></div>
      <div class="knob"><span>strength</span><input type="range" id="strength" min="0.2" max="0.98" step="0.02" value="0.7"><output id="strengthv">0.70</output></div>
    </div>
    <div data-tab="transform">
      <h2>regions &mdash; two materials, one pass</h2>
      <div class="row">
        <select id="region">
          <option value="">off</option>
          <option value="depth">by depth: near | far</option>
          <option value="lum">by light: bright | dark</option>
        </select>
        <select id="bpreset"></select>
        <button id="pickB" aria-pressed="false" title="the next archive click sets material B">B from archive</button>
      </div>
      <div class="knob"><span>split at</span><input type="range" id="rthresh" min="0.1" max="0.9" step="0.02" value="0.5"><output id="rthreshv">0.50</output></div>
    </div>
    <div data-tab="transform">
      <h2>then &mdash; a second material finishes the surface</h2>
      <div class="row"><select id="thenpreset"><option value="">off</option></select></div>
      <div class="knob"><span>strength</span><input type="range" id="thens" min="0.2" max="0.7" step="0.05" value="0.45"><output id="thensv">0.45</output></div>
    </div>
    <div data-tab="space,transform,walks">
      <div class="row">
        <span class="lab" style="margin:0">seed</span>
        <input type="number" id="seed" value="0" style="width:84px">
        <button id="reseed">random</button>
        <button id="quality" aria-pressed="false">full quality</button>
        <button id="go">draw</button>
      </div>
      <p class="hint">draft is the LCM LoRA at 5 steps — a sketch in a blink. full turns it off:
      16 steps, the real picture. sliders redraw as drafts; press draw for full.</p>
    </div>
    <div data-tab="composite">
      <p class="hint">exact pixels, CPU, every move live. the picture is carried by the texture's
      own photograph — provenance intact.</p>
    </div>
  </div>
</div>

<section class="archive">
  <div class="bar2">
    <span class="lab" id="acount" style="margin:0">the archive</span>
    <input type="text" id="aq" placeholder="search captions: glassy, oxidised copper, lichen on stone&hellip;">
    <select id="asort"><option value="">file order</option></select>
    <button id="aprev" aria-label="previous page">&lsaquo;</button>
    <span class="lab" id="apage" style="margin:0">&nbsp;</span>
    <button id="anext" aria-label="next page">&rsaquo;</button>
  </div>
  <div class="thumbs" id="thumbs"></div>
  <p class="hint">click a photograph &mdash; in composite it becomes the texture; in space and
  transform it becomes the material itself, because every photograph is a point in the space.</p>
</section>
</div>
<script>
const CKNOBS = {
  bands:    [['split', 0.5, 24, 0.25, 6, 'where the handover sits (blur px)'],
             ['grain', 0, 2, 0.05, 1, 'how loud the texture bands come in']],
  displace: [['amp', 0, 0.12, 0.002, 0.04, 'how far pixels travel'],
             ['reach', 1, 10, 0.25, 3, 'blur before the gradient is read'],
             ['shade', 0, 1.5, 0.05, 0.7, 'how hard the light presses']],
  blend:    [['wmix', 0, 1, 0.02, 0.6, 'bias toward the picture\\u2019s gradients']],
  chroma:   [['share', 0, 1, 0.02, 0.2, 'texture luminance let back in']],
  grain:    [['share', 0, 1, 0.02, 0.2, 'content luminance let back in']],
  mask:     [['level', 0, 1, 0.02, 0.5, 'the waterline'],
             ['feather', 0.01, 0.3, 0.01, 0.08, 'softness of the edge']],
  relief:   [['angle', 0, 6.28, 0.05, 1.0, 'where the light comes from'],
             ['ambient', 0.15, 1.1, 0.05, 0.55, 'light that needs no slope']],
  veins:    [['depth', 0, 1.8, 0.05, 0.9, 'how deep the texture\\u2019s cracks engrave'],
             ['vreach', 1, 8, 0.25, 3, 'the crack detector\\u2019s surround']],
  flow:     [['length', 0, 0.12, 0.002, 0.05, 'how far the comb pulls'],
             ['freach', 1, 8, 0.25, 3, 'blur before the contours are read'],
             ['smear', 0, 1, 0.02, 0.35, '0 combs sharply, 1 smears softly']],
  quilt:    [['alpha', 0.2, 0.98, 0.02, 0.8, 'how hard patches must match the picture\\u2019s light'],
             ['block', 16, 96, 4, 48, 'the mosaic\\u2019s grain, pixels']],
  analogy:  [['patch', 3, 15, 2, 7, 'what a neighbourhood sees, pixels'],
             ['melt', 0, 3, 0.25, 1, '0 keeps verbatim pixels, more melts to a warp']],
  guided:   [['radius', 2, 60, 2, 18, 'the local fit\\u2019s neighbourhood'],
             ['detail', 0, 1, 0.05, 0.35, 'the texture\\u2019s grain returned on top'],
             ['geps', 0.002, 0.08, 0.002, 0.015, 'how easily an edge counts']],
};
let tab = 'composite', mode = 'displace', busy = false, pending = false;
let texArch = null, photoArch = null, photoB = null;  // archive picks: {i, name}
let pinA = {point: 'centre', label: 'the centre'};
let pinB = {point: 'antipode', label: 'the antipode'};
const S = {};  // current knob values per name
const img = document.getElementById('img'), view = document.getElementById('view');
const stat = document.getElementById('stat');

function chips() {
  const box = document.getElementById('chips');
  box.innerHTML = '';
  function chip(kind, name, clear) {
    const s = document.createElement('span');
    s.className = 'chip';
    s.innerHTML = '<b>' + kind + '</b> ' + name + ' ✕';
    s.title = 'back to the dropdown / preset';
    s.onclick = () => { clear(); chips(); render(false); };
    box.appendChild(s);
  }
  if (tab === 'composite' && texArch) chip('texture', texArch.name, () => texArch = null);
  if (tab !== 'composite' && photoArch) chip('material', photoArch.name, () => photoArch = null);
  if (tab === 'transform' && photoB) chip('material B', photoB.name, () => photoB = null);
  if (tab === 'walks') {
    chip('A', pinA.label, () => pinA = {point: 'centre', label: 'the centre'});
    chip('B', pinB.label, () => pinB = {point: 'antipode', label: 'the antipode'});
  }
}

function snapshot() {
  if (photoArch) return photoArch.u ? {upmat: photoArch.u, label: photoArch.name}
                                    : {photo: photoArch.i, label: photoArch.name};
  const terms = document.getElementById('termmix').value.trim();
  if (terms) return {terms: terms, label: terms.slice(0, 28)};
  const preset = document.getElementById('preset').value;
  return preset ? {preset: preset, label: preset} : {point: 'centre', label: 'the centre'};
}

function visible() {
  document.querySelectorAll('[data-tab]').forEach(el => {
    el.style.display = el.dataset.tab.split(',').includes(tab) ? '' : 'none';
  });
}
for (const t of ['composite', 'space', 'transform', 'walks']) {
  document.getElementById('tab-' + t).onclick = () => {
    tab = t;
    for (const u of ['composite', 'space', 'transform', 'walks'])
      document.getElementById('tab-' + u).setAttribute('aria-pressed', u === t);
    visible(); chips(); render(t !== 'composite');
  };
}

function knobRow(box, name, lo, hi, step, val, hint) {
  const d = document.createElement('div');
  d.className = 'knob'; d.title = hint || '';
  d.innerHTML = '<span>' + name + '</span><input type="range" min="' + lo + '" max="' + hi +
    '" step="' + step + '" value="' + val + '"><output>' + (+val).toFixed(2) + '</output>';
  box.appendChild(d);
  S[name] = +val;
  d.querySelector('input').addEventListener('input', e => {
    S[name] = +e.target.value;
    d.querySelector('output').textContent = (+e.target.value).toFixed(2);
    render(false);
  });
}
function buildModeKnobs() {
  const box = document.getElementById('cknobs');
  box.innerHTML = '';
  for (const [n, lo, hi, st, v, hint] of CKNOBS[mode]) knobRow(box, n, lo, hi, st, S[n] ?? v, hint);
}
const modesBox = document.getElementById('modes');
for (const m of Object.keys(CKNOBS)) {
  const b = document.createElement('button');
  b.textContent = m; b.setAttribute('aria-pressed', m === mode);
  b.onclick = () => {
    mode = m;
    modesBox.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x.textContent === m));
    buildModeKnobs(); render(false);
  };
  modesBox.appendChild(b);
}
buildModeKnobs();

for (const [id, out, digits] of [['push','pushv',1],['scale','scalev',2],
    ['structure','structurev',2],['strength','strengthv',2],['rthresh','rthreshv',2],['thens','thensv',2]]) {
  const el = document.getElementById(id);
  el.addEventListener('input', e => {
    document.getElementById(out).textContent = (+e.target.value).toFixed(digits);
    render(false);
  });
}
document.getElementById('termmix').addEventListener('change', () => { photoArch = null; chips(); render(false); });
document.getElementById('against').addEventListener('change', () => render(false));
document.getElementById('region').addEventListener('change', () => render(false));
document.getElementById('bpreset').addEventListener('change', () => { photoB = null; chips(); render(false); });
document.getElementById('thenpreset').addEventListener('change', () => render(false));
async function pushUp(kind, file) {
  const r = await fetch('/upload?name=' + encodeURIComponent(file.name), {method: 'POST', body: file});
  if (!r.ok) { stat.textContent = 'upload refused'; return; }
  const got = await r.json();
  const texSel = document.getElementById('texture');
  if (![...texSel.options].some(o => o.value === 'u:' + got.name))
    texSel.add(new Option('⬆ ' + got.name, 'u:' + got.name));
  if (kind === 'content') {
    const el = document.getElementById('content');
    if (![...el.options].some(o => o.value === 'u:' + got.name))
      el.add(new Option('⬆ ' + got.name, 'u:' + got.name));
    el.value = 'u:' + got.name;
  } else if (tab === 'composite') {
    texSel.value = 'u:' + got.name; texArch = null;
  } else {
    photoArch = {u: got.name, name: '⬆ ' + got.name.slice(0, 24)};
  }
  chips(); render(false);
}
document.getElementById('upC').onclick = () => document.getElementById('upCf').click();
document.getElementById('upT').onclick = () => document.getElementById('upTf').click();
document.getElementById('upCf').addEventListener('change', e => e.target.files[0] && pushUp('content', e.target.files[0]));
document.getElementById('upTf').addEventListener('change', e => e.target.files[0] && pushUp('material', e.target.files[0]));

const pickB = document.getElementById('pickB');
pickB.onclick = () => pickB.setAttribute('aria-pressed', pickB.getAttribute('aria-pressed') !== 'true');
document.getElementById('pinA').onclick = () => { pinA = snapshot(); chips(); render(false); };
document.getElementById('pinB').onclick = () => { pinB = snapshot(); chips(); render(false); };
document.getElementById('aCentre').onclick = () => { pinA = {point: 'centre', label: 'the centre'}; chips(); render(false); };
document.getElementById('bAnti').onclick = () => { pinB = {point: 'antipode', label: 'the antipode'}; chips(); render(false); };
const walkt = document.getElementById('walkt');
walkt.addEventListener('input', e => {
  document.getElementById('walktv').textContent = (+e.target.value).toFixed(2);
  render(false);
});
document.getElementById('preset').addEventListener('change', () => { photoArch = null; chips(); render(false); });
document.getElementById('content').addEventListener('change', () => render(false));
document.getElementById('texture').addEventListener('change', () => { texArch = null; chips(); render(false); });
document.getElementById('seed').addEventListener('change', () => render(false));
document.getElementById('reseed').onclick = () => {
  document.getElementById('seed').value = Math.floor(Math.random() * 99999);
  render(false);
};
const qbtn = document.getElementById('quality');
qbtn.onclick = () => {
  qbtn.setAttribute('aria-pressed', qbtn.getAttribute('aria-pressed') !== 'true');
  render(true);
};
document.getElementById('go').onclick = () => render(true);

function query(finalDraw) {
  const q = new URLSearchParams();
  if (tab === 'composite') {
    q.set('content', document.getElementById('content').value);
    q.set('texture', texArch ? 'i:' + texArch.i : document.getElementById('texture').value);
    q.set('mode', mode);
    for (const [n] of CKNOBS[mode]) q.set(n, S[n]);
    return '/composite?' + q;
  }
  if (photoArch) { if (photoArch.u) q.set('upmat', photoArch.u); else q.set('photo', photoArch.i); }
  q.set('preset', document.getElementById('preset').value);
  q.set('terms', document.getElementById('termmix').value);
  q.set('against', document.getElementById('against').value);
  q.set('push', document.getElementById('push').value);
  q.set('scale', document.getElementById('scale').value);
  q.set('seed', document.getElementById('seed').value);
  const full = qbtn.getAttribute('aria-pressed') === 'true';
  q.set('q', (full && finalDraw) ? 'full' : 'draft');
  document.querySelectorAll('#concepts input').forEach(el => {
    if (+el.value) q.set('c_' + el.dataset.name, el.value);
  });
  if (tab === 'walks') {
    for (const [pre, pin] of [['a_', pinA], ['b_', pinB]])
      for (const k of ['point', 'photo', 'terms', 'preset'])
        if (pin[k] !== undefined) q.set(pre + k, pin[k]);
    q.set('t', walkt.value);
    return '/walk?' + q;
  }
  if (tab === 'space') return '/space?' + q;
  q.set('content', document.getElementById('content').value);
  q.set('structure', document.getElementById('structure').value);
  q.set('strength', document.getElementById('strength').value);
  const region = document.getElementById('region').value;
  if (region) {
    q.set('region', region);
    q.set('rthresh', document.getElementById('rthresh').value);
    if (photoB) q.set('b_photo', photoB.i);
    else q.set('b_preset', document.getElementById('bpreset').value);
  }
  const then = document.getElementById('thenpreset').value;
  if (then) {
    q.set('then', then);
    q.set('then_s', document.getElementById('thens').value);
  }
  return '/transform?' + q;
}
let t0 = 0;
function render(finalDraw) {
  if (busy) { pending = true; return; }
  busy = true; pending = false; view.classList.add('busy');
  t0 = performance.now();
  const next = new Image();
  next.onload = () => {
    img.src = next.src; busy = false; view.classList.remove('busy');
    stat.innerHTML = '<b>' + Math.round(performance.now() - t0) + '</b> ms';
    if (pending) render(false);
  };
  next.onerror = () => { busy = false; view.classList.remove('busy'); stat.textContent = 'error — see terminal'; };
  next.src = query(finalDraw) + '&_=' + Date.now();
}

// ---- the archive panel ----
let apage = 0, atimer = null;
function loadArchive() {
  const q = new URLSearchParams();
  q.set('term', document.getElementById('aq').value);
  q.set('sort', document.getElementById('asort').value);
  q.set('page', apage);
  fetch('/archive?' + q).then(r => r.json()).then(a => {
    const pages = Math.max(1, Math.ceil(a.total / a.per));
    if (apage >= pages) { apage = pages - 1; return loadArchive(); }
    document.getElementById('acount').textContent = 'the archive — ' + a.total + ' photographs';
    document.getElementById('apage').textContent = (a.page + 1) + ' / ' + pages;
    const grid = document.getElementById('thumbs');
    grid.innerHTML = '';
    for (const row of a.rows) {
      const im = document.createElement('img');
      im.src = '/thumb?i=' + row.i;
      im.loading = 'lazy';
      im.title = row.cap || row.name;
      im.alt = row.name;
      im.onclick = () => {
        if (tab === 'composite') texArch = {i: row.i, name: row.name};
        else if (tab === 'transform' && pickB.getAttribute('aria-pressed') === 'true') {
          photoB = {i: row.i, name: row.name};
          pickB.setAttribute('aria-pressed', 'false');
          if (!document.getElementById('region').value) document.getElementById('region').value = 'depth';
        } else photoArch = {i: row.i, name: row.name};
        chips(); render(false);
        window.scrollTo({top: 0, behavior: 'smooth'});
      };
      grid.appendChild(im);
    }
  });
}
document.getElementById('aq').addEventListener('input', () => {
  clearTimeout(atimer);
  atimer = setTimeout(() => { apage = 0; loadArchive(); }, 350);
});
document.getElementById('asort').addEventListener('change', () => { apage = 0; loadArchive(); });
document.getElementById('aprev').onclick = () => { apage = Math.max(0, apage - 1); loadArchive(); };
document.getElementById('anext').onclick = () => { apage += 1; loadArchive(); };

fetch('/menu').then(r => r.json()).then(m => {
  for (const [id, list] of [['content', m.contents], ['texture', m.textures]]) {
    const el = document.getElementById(id);
    for (const f of list) el.add(new Option(f, f));
    for (const f of m.uploads) el.add(new Option('⬆ ' + f, 'u:' + f));
  }
  const pr = document.getElementById('preset');
  pr.add(new Option('— the centre of the space —', ''));
  for (const p of m.presets) pr.add(new Option(p, p));
  if (m.presets.length) pr.value = m.presets[0];
  const bp = document.getElementById('bpreset'), tp = document.getElementById('thenpreset');
  for (const p of m.presets) { bp.add(new Option('B: ' + p, p)); tp.add(new Option('then: ' + p, p)); }
  if (m.presets.includes('water reflections')) bp.value = 'water reflections';
  const cb = document.getElementById('concepts');
  for (const c of m.concepts) {
    const d = document.createElement('div');
    d.className = 'knob';
    d.innerHTML = '<span>' + c + '</span><input type="range" min="-8" max="8" step="0.25" value="0" data-name="' +
      c + '"><output>0.00</output>';
    cb.appendChild(d);
    d.querySelector('input').addEventListener('input', e => {
      d.querySelector('output').textContent = (+e.target.value).toFixed(2);
      render(false);
    });
  }
  if (!m.concepts.length) document.getElementById('conceptbox').style.display = 'none';
  if (!m.transform) document.getElementById('tab-transform').style.display = 'none';
  const srt = document.getElementById('asort');
  for (const c of m.concepts) {
    srt.add(new Option(c + ' ▾ high first', c + ' -'));
    srt.add(new Option(c + ' ▴ low first', c + ' +'));
  }
  visible(); chips(); render(false); loadArchive();
});
</script></body></html>
"""


@contextlib.contextmanager
def wrapped(*models):
    """Every convolution in these models reading past the edge wraps round to the other side, so
    what is drawn repeats without a seam (a tileable material)."""
    convs = [m for model in models for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    before = [c.padding_mode for c in convs]
    for c in convs:
        c.padding_mode = "circular"
    try:
        yield
    finally:
        for c, mode in zip(convs, before):
            c.padding_mode = mode


def once(load):
    """A getter whose first call loads a model, held under the engine's loading lock: a live draft
    and a job asking at the same moment load it once, instead of twice at once and both failing.
    What the first call adds to `pipes` is registered with the memory manager (models.py), so it
    can be let go when memory runs short; the next call loads it again."""
    name = "pipe: " + load.__name__

    def get(self, *args, **kwargs):
        from models import MODELS

        with self.loading:
            owned = self.__dict__.setdefault("_owned", {}).get(name)
            if owned and all(k in self.pipes for k in owned):
                out = load(self, *args, **kwargs)
            else:
                keys = []
                drop = lambda: [self.pipes.pop(k, None) for k in keys]  # noqa: E731
                with MODELS.loading(name, drop, self.lock, guess=4.0):
                    before = set(self.pipes)
                    out = load(self, *args, **kwargs)
                    keys += sorted(set(self.pipes) - before)
                self._owned[name] = keys
        MODELS.used(name)
        return out
    get.__doc__ = load.__doc__
    return get


class Engine:
    """Lazy pipelines behind one GPU lock; the composite tab needs none of them."""

    def __init__(self, args):
        self.args = args
        self.p = dict(PRESETS[args.preset])
        if args.size is None:
            args.size = self.p["size"]
        self.device = torch.device(args.device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lock = threading.Lock()
        self.loading = LOAD  # first loads, one at a time, with every helper's (loadlock.py)
        self.space = dict(np.load(args.space, allow_pickle=True))
        self.norm = float(self.space["norm"])
        expect = {"sd15": 1024, "sdxl": 1280, "sdxl-turbo": 1280}[args.preset]
        if self.space["emb"].shape[1] != expect:
            raise SystemExit(
                f"this space holds {self.space['emb'].shape[1]}-wide vectors and the {args.preset} "
                f"adapter reads {expect}: another encoder's vectors are another space")
        self.presets = {k: v for k, v in json.loads(pathlib.Path(args.presets).read_text(encoding="utf-8")).items()
                        if not k.startswith("_")}
        book = pathlib.Path(args.concepts)
        self.concepts = json.loads(book.read_text(encoding="utf-8")) if book.is_file() else {}
        self.contents = self.listing(args.sources)
        self.textures = self.listing(args.textures)
        self.pipes = {}
        self.depths = {}
        self.updir = pathlib.Path(args.space).parent / "poetics" / "uploads"
        self.updir.mkdir(parents=True, exist_ok=True)
        self.uploads = {f.name: {"path": f, "vec": None} for f in sorted(self.updir.glob("*"))
                        if f.suffix.lower() in SUFFIXES}
        self.files = [str(f) for f in self.space["files"]]
        self.per_image = None  # one mean vector per photograph, built lazily
        self.scores = {}
        self.thumbs = {}
        # Thousands of caption sidecars live on a slow drive, so they are read behind the server's
        # back: the page is up at once, and the first search or term render joins the reader.
        self.captions = None
        self.found = None
        self._reader = threading.Thread(target=self._read_captions, daemon=True)
        self._reader.start()

    def _read_captions(self):
        # The captions live beside the photographs, on a drive that is not always there. Each
        # successful read is copied beside the space, and the copy stands in when the drive is
        # away, so the vocabulary (and every preset built on it) never depends on a cable.
        cache = pathlib.Path(self.args.space).with_suffix(".captions.json")
        caps = []
        for f in self.files:
            beside = pathlib.Path(f).with_suffix(".txt")
            try:
                caps.append(beside.read_text(encoding="utf-8").strip() if beside.is_file() else "")
            except OSError:
                caps.append("")
        have = sum(1 for c in caps if c)
        if have:
            if not cache.is_file() or have >= len(json.loads(cache.read_text(encoding="utf-8"))):
                cache.write_text(json.dumps(caps), encoding="utf-8")
        elif cache.is_file():
            caps = json.loads(cache.read_text(encoding="utf-8"))
            print(f"  photo drive unreachable: captions from {cache.name}")
        found = collections.defaultdict(list)
        for i, text in enumerate(caps):
            for field, pattern in FIELDS.items():
                if m := re.search(pattern, text):
                    for part in m.groups():
                        for word in (part.split(" and ") if field == "surface" else [part]):
                            found[word.strip()].append(i)
        self.captions = caps
        self.found = found
        print(f"  captions read: {sum(1 for c in caps if c)} of {len(caps)}")

    def caption_list(self):
        self._reader.join()
        return self.captions

    def vocab(self):
        self._reader.join()
        return self.found

    def image_means(self):
        if self.per_image is None:
            emb, owner = self.space["emb"], self.space["owner"]
            sums = np.zeros((len(self.files), emb.shape[1]), np.float32)
            np.add.at(sums, owner, emb)
            self.per_image = sums / np.bincount(owner, minlength=len(self.files))[:, None]
        return self.per_image

    def concept_scores(self, name):
        if name not in self.scores:
            c = self.concepts[name]
            self.scores[name] = self.image_means() @ np.asarray(c["direction"], np.float32)
        return self.scores[name]

    def archive(self, q):
        """Names and captions for one page of the archive, filtered and ordered."""
        want = q.get("term", [""])[0].strip().lower()
        order = np.arange(len(self.files))
        sort = q.get("sort", [""])[0]
        if sort:
            name, _, direction = sort.rpartition(" ")
            if name in self.concepts:
                order = np.argsort(self.concept_scores(name))
                if direction == "-":
                    order = order[::-1]
        caps = self.caption_list()
        if want:
            words = [w.strip() for w in want.split(",") if w.strip()]
            order = [i for i in order if all(w in caps[i].lower() for w in words)]
        page, per = int(q.get("page", ["0"])[0]), 24
        start = max(0, page) * per
        rows = [{"i": int(i), "name": pathlib.Path(self.files[i]).name, "cap": caps[i]}
                for i in order[start : start + per]]
        return {"total": len(order), "page": page, "per": per, "rows": rows}

    def thumb(self, i):
        if i not in self.thumbs:
            with Image.open(self.files[i]) as im:
                im.draft("RGB", (im.width // 8, im.height // 8))
                im = im.convert("RGB")
                side = min(im.size)
                im = im.crop(((im.width - side) // 2, (im.height - side) // 2,
                              (im.width + side) // 2, (im.height + side) // 2)).resize((144, 144), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=82)
            self.thumbs[i] = buf.getvalue()
        return self.thumbs[i]

    def photo_vector(self, i):
        """One photograph as a place in the space: its crops' mean, at real-crop length."""
        mine = self.space["owner"] == i
        v = self.space["emb"][mine].mean(0)
        return v * (self.norm / np.linalg.norm(v))

    def original_size(self, path):
        """A picture's own pixel size, read from its header once."""
        memo = self.__dict__.setdefault("_sizes", {})
        key = str(path)
        if key not in memo:
            with Image.open(path) as im:
                memo[key] = im.size
        return memo[key]

    def frame(self, path, long_side, multiple=1):
        """The frame a picture keeps its shape in, at this long side."""
        return cx.frame_dims(*self.original_size(path), int(long_side), multiple)

    def load_image(self, path, width, height):
        """The picture cropped to cover this frame and resized to it (JPEGs decode at reduced scale)."""
        return cx.cover(cx.open_rgb(path, width, height), width, height)

    def content_path(self, name):
        """A content by name: the sources folder, or an upload (`u:` prefix)."""
        if name.startswith("u:"):
            return self.uploads[name[2:]]["path"]
        return pathlib.Path(self.args.sources) / name

    def upload_vec(self, name):
        """An uploaded photograph as a material."""
        u = self.uploads[name]
        if u["vec"] is None:
            u["vec"] = self.file_vec(u["path"])
        return u["vec"]

    @once
    def clip(self):
        """The adapter's own image encoder, with its normalisation."""
        if "clip" not in self.pipes:
            for attempt in range(5):  # transformers' lazy import can fail while another thread runs it
                try:
                    from transformers import CLIPVisionModelWithProjection
                    break
                except ImportError:
                    if attempt == 4:
                        raise
                    import time
                    time.sleep(0.5)

            from ip_space import CLIP_MEAN, CLIP_STD

            net = CLIPVisionModelWithProjection.from_pretrained(
                self.p["repo"], subfolder=self.p["encoder"]).to(self.device).eval()
            mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
            self.pipes["clip"] = (net, mean, std)
            print("  material encoder up")
        return self.pipes["clip"]

    def texfile(self, name):
        """A texture by its name under the textures folder, subfolders kept (an archive photograph sits
        in its category's); a name that would climb out of the folder is read by its file name alone."""
        root = pathlib.Path(self.args.textures)
        p = root / name
        return p if p.resolve().is_relative_to(root.resolve()) else root / pathlib.Path(name).name

    def file_vec(self, path):
        """Any photograph file as a material: four crops through the adapter's own encoder."""
        memo = self.__dict__.setdefault("_file_vecs", {})
        key = str(path)
        if key not in memo:
            net, mean, std = self.clip()
            with Image.open(path) as im:
                im.draft("RGB", (1024, 1024))  # four 224px crops do not need 24 megapixels
                im = im.convert("RGB")
            half = max(1, min(im.size) // 2)
            crops = []
            for cy, cxx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                left, top = cxx * (im.width - half), cy * (im.height - half)
                crops.append(im.crop((left, top, left + half, top + half)).resize((224, 224), Image.LANCZOS))
            x = torch.stack([torch.tensor(np.asarray(c, np.float32) / 255.0).permute(2, 0, 1) for c in crops]).to(self.device)
            with self.lock, torch.no_grad():
                v = net(pixel_values=(x - mean) / std).image_embeds.float().mean(0).cpu().numpy()
            memo[key] = (v * (self.norm / np.linalg.norm(v))).astype(np.float32)
        return memo[key]

    def add_upload(self, name, data):
        safe = pathlib.Path(name).name
        if pathlib.Path(safe).suffix.lower() not in SUFFIXES:
            safe += ".png"
        where = self.updir / safe
        where.write_bytes(data)
        Image.open(where).verify()  # not an image -> raises, and the entry is never registered
        self.uploads[safe] = {"path": where, "vec": None}
        return safe

    def listing(self, folder):
        root = pathlib.Path(folder).expanduser()
        return sorted(str(f.relative_to(root)) for f in root.rglob("*") if f.suffix.lower() in SUFFIXES) if root.is_dir() else []

    def schedulers(self, pipe):
        from diffusers import DPMSolverMultistepScheduler, LCMScheduler

        full = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        return {"draft": LCMScheduler.from_config(pipe.scheduler.config), "full": full}

    @once
    def space_pipe(self):
        if "space" not in self.pipes:
            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(
                self.p["model"], torch_dtype=self.dtype,
                safety_checker=None, requires_safety_checker=False).to(self.device)
            pipe.load_ip_adapter(self.p["repo"], subfolder=self.p["subfolder"],
                                 weight_name=self.p["weight"], image_encoder_folder=None)
            if self.p["fast"]:
                pipe.load_lora_weights(self.p["fast"], adapter_name="fast")
            pipe.set_progress_bar_config(disable=True)
            self.pipes["space"] = (pipe, self.schedulers(pipe))
            print("  space pipeline up")
        return self.pipes["space"]

    @once
    def transform_pipe(self):
        if "transform" not in self.pipes:
            if self.args.preset == "sd15":
                from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

                net = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=self.dtype)
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=net, torch_dtype=self.dtype,
                    safety_checker=None, requires_safety_checker=False).to(self.device)
                pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                                     weight_name="ip-adapter_sd15.safetensors", image_encoder_folder=None)
                pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="fast")
            else:
                # XL holds structure by EDGES (the cached canny controlnet) rather than depth, and
                # borrows the space pipeline's own base so only the controlnet costs new memory.
                from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

                net = ControlNetModel.from_pretrained("xinsir/controlnet-canny-sdxl-1.0",
                                                      torch_dtype=self.dtype).to(self.device)
                base, _ = self.space_pipe()
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(base, controlnet=net)
            pipe.set_progress_bar_config(disable=True)
            self.pipes["transform"] = (pipe, self.schedulers(pipe))
            print("  transform pipeline up")
        return self.pipes["transform"]

    @once
    def illusion_net(self):
        """QR Code Monster v2 (monster-labs, OpenRAIL++): a ControlNet trained to bend a drawing toward
        a pattern of light and dark, the usual way to hide one picture inside another. Only the
        ControlNet is new: it is swapped into the transform pipeline for the illusion's frames."""
        if "illusion" not in self.pipes:
            from diffusers import ControlNetModel

            self.pipes["illusion"] = ControlNetModel.from_pretrained(
                "monster-labs/control_v1p_sd15_qrcode_monster", subfolder="v2",
                revision="560fb7b15d0badb409f8cd578a2bfe63bd4b8046", use_safetensors=True,
                torch_dtype=self.dtype).to(self.device)
            print("  illusion controlnet up")
        return self.pipes["illusion"]

    def draw_illusion(self, q, source, hint, illusion, let_go):
        """The texture photograph `source` redrawn in its own material while the ControlNet bends its
        light and dark toward `hint` (carve.control): a picture hidden in a texture."""
        pipe, sched = self.transform_pipe()
        net = self.illusion_net()
        draft = q.get("q", ["draft"])[0] == "draft"
        seed = int(float(q.get("seed", ["0"])[0]))
        scale = float(q.get("scale", ["1"])[0])
        strength = float(q.get("strength", ["0.9"])[0])
        vec = self.material(q)  # before the lock: reading a photograph's place in the space takes it too
        with self.lock, torch.no_grad():
            steps, guidance = self.set_quality(pipe, sched, draft, self.look_of(pipe, q))
            steps = max(steps, math.ceil(2 / max(strength, 0.05)))
            held, pipe.controlnet = pipe.controlnet, net
            try:
                pipe.set_ip_adapter_scale(scale)
                done = pipe(prompt=q.get("prompt", [""])[0], image=source, strength=strength, control_image=hint,
                            controlnet_conditioning_scale=float(illusion), control_guidance_end=float(let_go),
                            ip_adapter_image_embeds=[self.embeds([vec], guidance)],
                            num_inference_steps=steps, guidance_scale=guidance,
                            generator=torch.Generator(self.args.device).manual_seed(seed)).images
            finally:
                pipe.controlnet = held
        return done[0]

    def edges(self, name, width, height=None):
        """The structure signal for XL: a canny-style edge map, numpy only."""
        height = height or width
        key = ("edge", name, width, height)
        if key not in self.depths:
            im = self.load_image(self.content_path(name), width, height)
            g = cx.blur(np.asarray(im.convert("L"), np.float32) / 255.0, 1.2)
            gy, gx = np.gradient(g)
            mag = np.hypot(gx, gy)
            hi = np.quantile(mag, 0.92)
            strong = mag >= hi
            weak = mag >= 0.4 * hi
            grown = strong
            for _ in range(8):  # hysteresis, approximately: weak edges joined to strong ones
                near = grown | np.roll(grown, 1, 0) | np.roll(grown, -1, 0) | np.roll(grown, 1, 1) | np.roll(grown, -1, 1)
                next_grown = near & weak
                if (next_grown == grown).all():
                    break
                grown = next_grown
            self.depths[key] = Image.fromarray((grown * 255).astype(np.uint8)).convert("RGB")
        return self.depths[key]

    def depth(self, name, width, height=None):
        height = height or width
        key = (name, width, height)
        if key not in self.depths:
            from transformers import DPTForDepthEstimation, DPTImageProcessor

            from models import MODELS

            with self.loading:
                got = self.pipes.get("dpt")
                if got is None:
                    with MODELS.loading("depth: DPT", lambda: self.pipes.pop("dpt", None), self.lock, guess=1.0):
                        got = self.pipes["dpt"] = (
                            DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas"),
                            DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device).eval())
            MODELS.used("depth: DPT")
            proc, net = got
            image = self.load_image(self.content_path(name), width, height)
            with torch.no_grad():
                out = net(**proc(images=image, return_tensors="pt").to(self.device)).predicted_depth
            d = torch.nn.functional.interpolate(out[None], size=(height, width), mode="bicubic")[0, 0]
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            self.depths[key] = Image.fromarray((d.cpu().numpy() * 255).astype(np.uint8)).convert("RGB")
            # The same depth unrounded, for whatever takes its slopes (surface.flatten).
            self.__dict__.setdefault("depth_floats", {})[key] = d.cpu().numpy().astype(np.float32)
        return self.depths[key]

    def base_material(self, q):
        """One place in the space: a named point, a photograph, a preset or a term mix.

        Remembered by what defines it: a term mix averages thousands of crops, and a live slider
        should not pay for that on every frame when only a concept offset moved.
        """
        key = tuple(q.get(k, [""])[0] for k in ("point", "upmat", "texfile", "photo", "terms", "preset", "push"))
        memo = self.__dict__.setdefault("_places", {})
        if key not in memo:
            if len(memo) > 256:
                memo.clear()
            memo[key] = self._base_material(q)
        return memo[key]

    def _base_material(self, q):
        point = q.get("point", [""])[0]
        raw = q.get("terms", [""])[0].strip()
        push = float(q.get("push", ["8"])[0])
        if point == "antipode":
            v = -self.space["centre"]
        elif point == "centre":
            v = self.space["centre"]
        elif q.get("upmat", [""])[0] in self.uploads:
            v = self.upload_vec(q["upmat"][0])
        elif q.get("texfile", [""])[0]:
            v = self.file_vec(self.texfile(q["texfile"][0]))
        elif q.get("photo", [""])[0]:
            v = self.photo_vector(int(q["photo"][0]))
        elif raw:
            mix = {}
            for part in raw.split(","):
                word, _, w = part.partition(":")
                if word.strip():
                    mix[word.strip()] = float(w or 1.0)
            v = term_vector(self.space, mix, self.vocab(), push)
        elif q.get("preset", [""])[0]:
            v = term_vector(self.space, dict(self.presets[q["preset"][0]]), self.vocab(), push)
        else:
            v = self.space["centre"]
        return (v * (self.norm / np.linalg.norm(v))).astype(np.float32)

    def with_concepts(self, v, q):
        """Every turned axis added, and the point put back at real-crop length."""
        for key, vals in q.items():
            if key.startswith("c_") and key[2:] in self.concepts:
                c = self.concepts[key[2:]]
                v = v + float(vals[0]) * c["sigma"] * np.asarray(c["direction"], np.float32)
        return (v * (self.norm / np.linalg.norm(v))).astype(np.float32)

    def material(self, q):
        """The conditioning vector: a place, plus every turned concept axis."""
        return self.with_concepts(self.base_material(q), q)

    def spec_vec(self, spec):
        return material_vec(self.space, self.vocab(), self.presets, spec, 8.0)

    def embeds(self, vectors, guidance, against=None):
        """One positive per region; the against-material, when given, is the negative half."""
        if not isinstance(vectors, list):
            vectors = [vectors]
        pos = torch.tensor(np.stack(vectors), dtype=self.dtype, device=self.device)[None]
        if guidance <= 1.0:
            return pos
        if against is None:
            return torch.cat([torch.zeros_like(pos), pos])
        neg = torch.tensor(np.stack([against] * len(vectors)), dtype=self.dtype, device=self.device)[None]
        return torch.cat([neg, pos])

    def look_of(self, pipe, q):
        """The archive's own LoRA under this render, loaded once per pipe and run under its name.

        `lora` is a checkpoint folder relative to the models folder (sd15-lora-yellow/step-002000),
        `lora_w` how hard it pulls. Several stay loaded; set_adapters picks which one speaks.
        """
        name = q.get("lora", ["none"])[0]
        if not name or name == "none" or self.args.preset != "sd15":
            return None
        key = "look_" + re.sub(r"[^0-9a-zA-Z]", "_", name)
        loaded = pipe.__dict__.setdefault("_looks", set())
        if key not in loaded:
            if pipe.__dict__.get("_fused"):
                pipe.unfuse_lora()  # a LoRA is loaded into clean weights; the next draft re-fuses
                pipe.__dict__.update(_fused=None, _adapters=None)
            where = pathlib.Path(self.args.space).parent / name
            pipe.load_lora_weights(str(where), weight_name="pytorch_lora_weights.safetensors", adapter_name=key)
            loaded.add(key)
            print(f"  lora up: {name}")
        return key, float(q.get("lora_w", ["1.4"])[0])

    def set_quality(self, pipe, sched, draft, look=None, drag=False):
        """Steps and guidance for this frame, with the pipe's adapters arranged to match.

        Drafts run on the consistency LoRA FUSED into the UNet's weights, the look's LoRA fused in
        with it: left as live adapters they double every step's cost. Both must be fused, and
        nothing may be disabled afterwards, because of two PEFT rules: a merged layer skips every
        adapter that is not merged, and disabling LoRA unmerges what was fused. (Fusing the
        consistency LoRA alone and then disabling the rest left drafts with neither — grey mush
        with no material, at every size.) A new look or weight re-fuses (a quarter second).
        Full quality unfuses and runs the look as a live adapter beside a silenced consistency LoRA.
        """
        if self.p["fast"]:
            pipe.scheduler = sched["draft" if draft else "full"]
            fused = pipe.__dict__.get("_fused")
            if draft:
                want = (("fast", look[0]), (1.0, look[1])) if look else (("fast",), (1.0,))
                if fused != want:
                    if fused:
                        pipe.unfuse_lora()
                    pipe.enable_lora()
                    pipe.set_adapters(list(want[0]), list(want[1]))  # fuse merges at the CURRENT weights
                    pipe.fuse_lora(adapter_names=list(want[0]), lora_scale=1.0)
                    pipe.__dict__.update(_fused=want, _adapters=None)
                # A little guidance: without it the material's pull is the bare adapter's, and a
                # draft shows the picture re-photographed, not the material (measured: 2.5 brings
                # it in, at 1.1 s for 1024 against 0.9 s).
                return (3 if drag else 5), 2.5
            if fused:
                pipe.unfuse_lora()  # before any weight changes: unmerging subtracts at the current weights
                pipe.__dict__.update(_fused=None, _adapters=None)
            names, weights = ["fast"], [0.0]
            if look:
                names.append(look[0])
                weights.append(look[1])
            want = (tuple(names), tuple(weights))
            if pipe.__dict__.get("_adapters") != want:
                pipe.enable_lora()
                pipe.set_adapters(names, weights)
                pipe.__dict__["_adapters"] = want
            return 20, 7.0
        # No consistency LoRA for this base: a draft is simply fewer solver steps.
        pipe.scheduler = sched["full"]
        if drag:
            return 6, 4.0
        return (10, 5.0) if draft else (self.p["steps"], self.p["guidance"])

    @once
    def tiny_vae(self):
        """The tiny autoencoder (TAESD): decodes a drag frame in a few ms instead of tens."""
        if "taesd" not in self.pipes:
            from diffusers import AutoencoderTiny

            rid = "madebyollin/taesd" if self.args.preset == "sd15" else "madebyollin/taesdxl"
            self.pipes["taesd"] = AutoencoderTiny.from_pretrained(rid, torch_dtype=self.dtype).to(self.device)
        return self.pipes["taesd"]

    def against_of(self, q):
        spec = q.get("against", [""])[0].strip()
        return self.spec_vec(spec) if spec else None

    def render_vec(self, vec, q, wrap=False):
        pipe, sched = self.space_pipe()
        draft = q.get("q", ["draft"])[0] == "draft"
        drag = q.get("drag", [""])[0] == "1"
        against = self.against_of(q)
        seed = int(float(q.get("seed", ["0"])[0]))
        long_side = int(q.get("size", [self.args.size])[0])
        aw, ah = (int(v) for v in q.get("aspect", ["1:1"])[0].split(":"))
        width, height = cx.frame_dims(aw, ah, long_side, 8)
        with self.lock, torch.no_grad():
            steps, guidance = self.set_quality(pipe, sched, draft or drag, self.look_of(pipe, q), drag)
            if (draft or drag) and against is not None:
                guidance = max(guidance, 1.5)  # the negative half only speaks through CFG
            pipe.set_ip_adapter_scale(float(q.get("scale", ["1"])[0]))
            full_vae = pipe.vae
            if drag:
                pipe.vae = self.tiny_vae()
            try:
                with wrapped(pipe.unet, pipe.vae) if wrap else contextlib.nullcontext():
                    return pipe(prompt=q.get("prompt", [""])[0],
                            ip_adapter_image_embeds=[self.embeds(vec, guidance, against)],
                            num_inference_steps=steps, guidance_scale=guidance, height=height, width=width,
                            generator=torch.Generator(self.args.device).manual_seed(seed)).images[0]
            finally:
                pipe.vae = full_vae

    def draw_space(self, q):
        return self.render_vec(self.material(q), q)

    def draw_walk(self, q):
        """The great circle between two pinned places, the slider somewhere along it."""
        ends = []
        for prefix in ("a_", "b_"):
            sub = {}
            for k in ("point", "photo", "terms", "preset", "upmat"):
                v = q.get(prefix + k, [""])[0]
                if v:
                    sub[k] = [v]
            sub["push"] = q.get("push", ["8"])
            ends.append(self.base_material(sub))
        t = float(q.get("t", ["0.5"])[0])
        vec = self.with_concepts(slerp(ends[0], ends[1], t).astype(np.float32), q)
        return self.render_vec(vec, q)

    def draw_transform(self, q):
        pipe, sched = self.transform_pipe()
        draft = q.get("q", ["draft"])[0] == "draft"
        drag = q.get("drag", [""])[0] == "1"
        against = self.against_of(q)
        vectors = [self.material(q)]
        scale = float(q.get("scale", ["1"])[0])
        name = q.get("content", [self.contents[0]])[0]
        width, height = self.frame(self.content_path(name), int(q.get("size", [self.args.size])[0]), 8)
        source = self.load_image(self.content_path(name), width, height)
        control = self.depth(name, width, height) if self.args.preset == "sd15" else self.edges(name, width, height)
        seed = int(float(q.get("seed", ["0"])[0]))
        region = q.get("region", [""])[0]
        first_scale = scale
        if region:
            if q.get("b_upmat", [""])[0] in self.uploads:
                vectors.append(self.upload_vec(q["b_upmat"][0]))
            elif q.get("b_texfile", [""])[0]:
                vectors.append(self.file_vec(self.texfile(q["b_texfile"][0])))
            elif q.get("b_photo", [""])[0]:
                vectors.append(self.photo_vector(int(q["b_photo"][0])))
            elif q.get("b_terms", [""])[0]:
                vectors.append(self.spec_vec(q["b_terms"][0]))
            else:
                vectors.append(self.spec_vec(q.get("b_preset", [""])[0] or "water reflections"))
            first_scale = [[scale] * 2]

        def split(w, h, picture):
            """The region masks at this size, as attention kwargs; none without a region."""
            if not region:
                return {}
            splitter = self.depth(name, w, h) if region == "depth" else picture
            masks, _ = region_masks(picture, splitter, region, float(q.get("rthresh", ["0.5"])[0]), w, h)
            return {"cross_attention_kwargs": {"ip_adapter_masks": masks}}

        structure = float(q.get("structure", ["1"])[0])
        strength = float(q.get("strength", ["0.7"])[0])
        then = q.get("then", [""])[0].strip()
        with self.lock, torch.no_grad():
            steps, guidance = self.set_quality(pipe, sched, draft or drag, self.look_of(pipe, q), drag)
            if (draft or drag) and against is not None:
                guidance = max(guidance, 1.5)
            # img2img denoises only steps × strength of them: a few draft steps at a gentle
            # strength would round to none, so the count is raised until two remain.
            steps_1 = max(steps, math.ceil(2 / max(strength, 0.05)))
            prompt = q.get("prompt", [""])[0]
            main = [self.embeds(vectors, guidance, against)]
            gen = lambda: torch.Generator(self.args.device).manual_seed(seed)  # noqa: E731
            full_vae = pipe.vae
            if drag:
                pipe.vae = self.tiny_vae()
            try:
                # Latents between passes: a drawn texture re-encoded from pixels is adversarial to
                # img2img and confettis; handed over inside the VAE's space it redraws clean.
                pipe.set_ip_adapter_scale(first_scale)
                done = pipe(prompt=prompt, image=source, strength=strength, control_image=control,
                            controlnet_conditioning_scale=structure, ip_adapter_image_embeds=main,
                            num_inference_steps=steps_1, guidance_scale=guidance, generator=gen(),
                            output_type="latent" if then else "pil", **split(width, height, source)).images
                if then:
                    then_s = float(q.get("then_s", ["0.45"])[0])
                    pipe.set_ip_adapter_scale(scale)
                    done = pipe(prompt=prompt, image=done, strength=then_s, control_image=control,
                                controlnet_conditioning_scale=structure,
                                ip_adapter_image_embeds=[self.embeds(self.spec_vec(then), guidance, against)],
                                num_inference_steps=max(steps, math.ceil(2 / then_s)), guidance_scale=guidance,
                                generator=gen()).images
            finally:
                pipe.vae = full_vae
        return done[0]

    def draw_composite(self, q):
        content = cx.arr(cx.fit(self.content_path(q["content"][0]), self.args.csize))
        given = q["texture"][0]
        if given.startswith("i:"):
            where = self.files[int(given[2:])]
        elif given.startswith("u:"):
            where = self.uploads[given[2:]]["path"]
        else:
            where = pathlib.Path(self.args.textures) / given
        texture = cx.arr(cx.fit(where, self.args.csize))
        knob = lambda name, fallback: float(q.get(name, [fallback])[0])
        mode = q.get("mode", ["displace"])[0]
        if mode == "bands":
            out = cx.bands(content, texture, knob("split", 6), knob("grain", 1))
        elif mode == "displace":
            out = cx.displace(content, texture, knob("amp", 0.04), knob("reach", 3), knob("shade", 0.7))
        elif mode == "chroma":
            out = cx.chroma(content, texture, knob("share", 0.2))
        elif mode == "grain":
            out = cx.chroma(texture, content, knob("share", 0.2))
        elif mode == "mask":
            out = cx.mask(content, texture, knob("level", 0.5), knob("feather", 0.08))
        elif mode == "relief":
            out = cx.relief(content, texture, knob("angle", 1.0), knob("ambient", 0.55))
        elif mode == "veins":
            out = cx.veins(content, texture, knob("depth", 0.9), knob("vreach", 3.0))
        elif mode == "flow":
            out = cx.flow(content, texture, knob("length", 0.05), knob("freach", 3.0), knob("smear", 0.35))
        elif mode == "quilt":
            out = cx.quilt(content, texture, knob("alpha", 0.8), int(knob("block", 48)))
        elif mode == "analogy":
            out = cx.analogy(content, texture, int(knob("patch", 7)), knob("melt", 1.0))
        elif mode == "guided":
            out = cx.guided(content, texture, knob("radius", 18), knob("geps", 0.015), knob("detail", 0.35))
        else:
            out = cx.blend(content, texture, knob("wmix", 0.6))
        return cx.img(out)


def serve(engine, port):
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
            q = urllib.parse.parse_qs(url.query)
            try:
                if url.path == "/":
                    return self.send(PAGE.encode(), "text/html; charset=utf-8")
                if url.path == "/menu":
                    menu = dict(contents=engine.contents, textures=engine.textures,
                                presets=list(engine.presets), concepts=list(engine.concepts),
                                archive=len(engine.files), transform=True,
                                uploads=sorted(engine.uploads))
                    return self.send(json.dumps(menu).encode(), "application/json")
                if url.path == "/upfile":
                    return self.send(engine.uploads[q["name"][0]]["path"].read_bytes(), "image/jpeg", "max-age=3600")
                if url.path == "/archive":
                    return self.send(json.dumps(engine.archive(q)).encode(), "application/json")
                if url.path == "/thumb":
                    return self.send(engine.thumb(int(q["i"][0])), "image/jpeg", "max-age=86400")
                if url.path == "/composite":
                    frame = engine.draw_composite(q)
                elif url.path == "/space":
                    frame = engine.draw_space(q)
                elif url.path == "/walk":
                    frame = engine.draw_walk(q)
                elif url.path == "/transform":
                    frame = engine.draw_transform(q)
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                buf = io.BytesIO()
                frame.save(buf, format="JPEG", quality=90)
                self.send(buf.getvalue(), "image/jpeg")
            except Exception as e:  # surfaced in the terminal, a 500 in the page
                print(f"  ! {url.path}: {e}", flush=True)
                self.send_response(500)
                self.end_headers()

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            try:
                if url.path != "/upload":
                    self.send_response(404)
                    self.end_headers()
                    return
                data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                name = engine.add_upload(q.get("name", ["upload.png"])[0], data)
                self.send(json.dumps({"name": name}).encode(), "application/json")
            except Exception as e:
                print(f"  ! upload: {e}", flush=True)
                self.send_response(500)
                self.end_headers()

    class OnePerPort(ThreadingHTTPServer):
        # On Windows the default reuse flag lets a second copy bind the SAME port and shadow the
        # first — which is how six forgotten servers once held the whole card. Fail loudly instead.
        allow_reuse_address = False

    try:
        server = OnePerPort(("127.0.0.1", port), Handler)
    except OSError:
        raise SystemExit(f"port {port} is already serving — stop that one first (one server per port)")
    print(f"\n  http://127.0.0.1:{port}\n\n  composite is instant; space and transform load their"
          "\n  models on first use (half a minute, once). ctrl-c to stop")
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--sources", default=None, help="Folder of pictures to carry; default: restyle-sources beside the space.")
    ap.add_argument("--textures", default=None, help="Folder of texture photographs; default: poetics/textures beside the space.")
    ap.add_argument("--presets", default="presets.json")
    ap.add_argument("--concepts", default="concepts.json")
    ap.add_argument("--preset", default="sd15", choices=sorted(PRESETS),
                    help="Which base draws: sd15 at 512 with an LCM draft, or sdxl at 1024.")
    ap.add_argument("--size", type=int, default=None, help="Diffusion tabs' square; default is the preset's.")
    ap.add_argument("--csize", type=int, default=384, help="Composite tab's square — smaller is draggier.")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    beside = pathlib.Path(args.space).parent
    args.sources = args.sources or str(beside / "restyle-sources")
    args.textures = args.textures or str(beside / "poetics" / "textures")
    try:
        serve(Engine(args), args.port)
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
