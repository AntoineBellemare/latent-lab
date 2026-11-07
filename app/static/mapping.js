// Texture mapping on a page of its own: the picture made out of the texture, live. Fourteen ways to
// do it, each shown as a tile drawn from your own pair so the choice is made by looking; the chosen
// one's few knobs; where it goes (the same card as the editor's); how firmly; and, folded, the
// picture's light and forms laid over it. Every change draws a frame over the live socket — a quick
// one mid-drag, the full one on release — and nothing is kept until you keep it.
import {$, el, menu, MENU, stored, store, arrivals, findJob, madeIn, status, runJob, working, Live, MaskPreview, maskKeys, tabOfJob,
        Stage, History, dropCard, dropZone, upload, onArrive, saveRecipe, nameOf} from "./studio.js";
import {WhereCard, slider, control, applies, param} from "./controls.js";
import {Vary} from "./vary.js";

// The modes, in three kinds.
const KINDS = [["lit from the picture", ["chroma", "relit", "guided", "bands", "blend"]],
               ["shaped by the picture", ["displace", "relief", "carve", "veins", "flow", "grain", "mask"]],
               ["rebuilt from the texture's pieces", ["quilt", "analogy", "synth"]]];
const QUICK = [["none", "whole picture"], ["subject", "main subject"], ["person", "person"], ["body part", "body part"],
               ["object (words)", "named thing"], ["object (click)", "click on it"]];
const STRENGTH = [["amount", "material strength"], ["mask_soft", "soft edge"], ["follow", "follow contours"],
                  ["mask_noise", "ragged edge"], ["grow", "grow / shrink"], ["harmonise", "settle colours"]];
const TILE = 160;

let MODES = [];
let mode = "displace";
const st = {};                 // what every mode shares: the mask, the surface, the size, the material size
const own = {};                // each mode's own knobs: mode -> {key: value}
const inputs = {content: null, texture: null};
let shown = null;              // a render from the history on show, instead of the live frame
let frameUrl = null;           // the newest live frame
let stage, where, hist, vary, masks, live, tiles, drops, surfaceMore = false;

const spec = (m = mode) => MENU.methods[m];
const ownKeys = m => spec(m).params.filter(p => p.group === null).map(p => p.key);
const params = (m = mode) => ({...st, ...own[m]});

// ---------- boot ----------
menu().then(m => {
  MODES = KINDS.flatMap(k => k[1]).filter(k => m.methods[k]);
  for (const k of MODES) {
    own[k] = {};
    for (const p of spec(k).params) {
      if (p.group === null) own[k][p.key] = p.default;
      else if (!(p.key in st)) st[p.key] = p.default;
    }
  }
  st.size = 512;
  inputs.content = m.contents[0];
  inputs.texture = m.textures[0];
  const got = stored("mapping");
  if (got) {
    for (const k in got.st || {}) if (k in st) st[k] = got.st[k];
    for (const k in got.own || {}) if (own[k]) Object.assign(own[k], got.own[k]);
    if (got.mode && own[got.mode]) mode = got.mode;
    Object.assign(inputs, got.inputs || {});
  }
  build();
  hist.load(m.history);
  const sent = arrivals();
  if (sent.content) inputs.content = sent.content;
  if (sent.texture) inputs.texture = sent.texture;
  const job = sent.job && findJob(sent.job);
  if (job) reuse(job);
  else { sync(); draw("settle"); refreshTiles(); }
});

const save = () => store("mapping", {st, own, mode, inputs});

// ---------- the controls ----------
function build() {
  stage = new Stage($("view"), {views: [["result", "result"], ["mask", "mask"], ["picture", "picture"]],
                                onView: setView, onPoint: (x, y, shift) => where.click(x, y, shift)});
  stage.mark("result");
  hist = new History({key: "mapping.history", here: j => tabOfJob(j) === "mapping",
                      describe, onShow: show, onReuse: reuse});
  live = new Live({
    onFrame: (blob, meta) => {
      if (frameUrl) URL.revokeObjectURL(frameUrl);
      frameUrl = URL.createObjectURL(blob);
      if (!shown && stage.view === "result") stage.compare(frameUrl, before(), meta.method);
      if (!shown) status(`${meta.method} · ${meta.w}×${meta.h} · ${Math.round(meta.ms)} ms` + (meta.phase === "drag" ? " · quick" : ""));
    },
    onError: msg => status(msg.note, true),
    onKept: job => { hist.add(job); show(job); status("kept · in the history, and ready to send or download"); },
  });
  tiles = new Live({
    onFrame: (blob, meta) => {
      const img = $("tiles").querySelector(`[data-mode="${meta.method}"] img`);
      if (img && tileSeq[meta.method] === meta.seq) {
        if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
        img.src = URL.createObjectURL(blob);
        img.parentElement.classList.remove("waiting");
      }
      nextTile();
    },
    onError: () => nextTile(),
  });
  masks = new MaskPreview(url => {
    if (stage.view === "mask") stage.still(url);
    status(st.mask === "none" ? "no mask: the material covers the whole picture" : "the mask, in gold");
  }, note => status("mask: " + note, true));
  vary = new Vary(stage, {
    method: () => mode, params: () => params(), inputs: () => ({...inputs}), material: "texture",
    seeded: () => "seed" in own[mode] || st.mask_noise > 0, get seedKey() { return "seed" in own[mode] ? "seed" : "mask_seed"; },
    settings, done: job => hist.add(job), show,
  });
  onArrive(choose);
  drops = {content: dropCard("content", {label: "a picture", get: () => inputs.content, choose: v => choose("content", v)}),
           texture: dropCard("texture", {label: "a texture", get: () => inputs.texture, choose: v => choose("texture", v)})};
  $("inputsCard").append(drops.content, drops.texture);
  // the tiles
  for (const [kind, keys] of KINDS) {
    $("tiles").appendChild(el("div", "kind", kind));
    const row = el("div", "row3");
    for (const k of keys.filter(x => MODES.includes(x))) {
      const b = el("button", "mtile waiting");
      b.dataset.mode = k;
      b.title = spec(k).doc;
      const img = el("img");
      img.alt = "";
      b.append(img, el("span", "", k));
      b.onclick = () => pickMode(k);
      row.appendChild(b);
    }
    $("tiles").appendChild(row);
  }
  where = new WhereCard({root: $("where"), spec: spec(), st, quick: QUICK, onChange: moved => changed(moved ? "mask" : "settle"),
                         onArm: key => { stage.setPicking(!!key); if (key) setView("mask"); }});
  for (const [key, label] of STRENGTH) {
    const row = slider(param(spec(), key), st, {label, doc: param(spec(), key).doc,
                                                onInput: () => changed("drag"), onChange: () => changed("settle")});
    $("sliders").appendChild(row);
  }
  for (const s of param(spec(), "size").options) {
    const b = el("button", "", String(s));
    b.dataset.size = s;
    b.onclick = () => { st.size = s; changed("settle"); };
    $("sizeSeg").appendChild(b);
  }
  $("surfaceBox").ontoggle = () => store("mapping.surface", $("surfaceBox").open);
  $("surfaceBox").open = !!stored("mapping.surface");
  $("keep").onclick = keep;
  $("export").onclick = exportFull;
  $("vary").onclick = () => vary.open();
  $("recipe").onclick = recipe;
  $("flip").onclick = () => location.assign("/hidden.html#content=" + encodeURIComponent(inputs.content) +
                                            "&texture=" + encodeURIComponent(inputs.texture));
  dropZone($("stageBox"), async f => choose("content", await upload(f)));
  addEventListener("keydown", e => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "k" || e.key === "K") keep();
  });
}

function pickMode(k) {
  mode = k;
  where.spec = spec();
  changed("settle");
}

// What changed decides what is drawn: a knob mid-drag a quick frame, a release the full one, a
// change to where the mask goes the mask itself too.
function changed(phase, tilesToo) {
  shown = null;
  if (phase !== "drag") sync();  // sync rebuilds the panel: mid-drag it would replace the slider being dragged
  save();
  draw(phase === "drag" ? "drag" : "settle");
  if (phase === "mask") setView("mask");
  else if (stage.view !== "result" && phase !== "drag") setView("result");
  if (tilesToo) refreshTiles(tilesToo === true ? null : tilesToo);
}

function draw(phase) {
  live.send({method: mode, phase, params: params(), inputs: {...inputs}});
}

function sync() {
  drops.content.refresh();
  drops.texture.refresh();
  $("tiles").querySelectorAll(".mtile").forEach(b => b.setAttribute("aria-pressed", b.dataset.mode === mode));
  $("modeDoc").textContent = spec().doc;
  // the mode's own knobs, and the material's size
  const knobs = $("knobs");
  knobs.innerHTML = "";
  for (const p of spec().params.filter(q => q.group === null))
    knobs.appendChild(control(p, own[mode], ph => changed(ph, ph === "settle" ? mode : null)));
  knobs.appendChild(slider(param(spec(), "tex_scale"), st, {onInput: () => changed("drag"), onChange: () => changed("settle", true)}));
  // where, and how firmly
  where.sync();
  const masked = st.mask !== "none";
  $("strengthCard").hidden = !masked;
  for (const row of $("sliders").children) row._set(st[row.dataset.key]);
  // on the forms: what this mode offers, the seldom-touched behind "more"
  const box = $("surface");
  box.innerHTML = "";
  const sp = spec().params.filter(p => p.group === "surface" && applies(p, st));
  for (const p of sp.filter(q => !q.more || surfaceMore)) box.appendChild(control(p, st, ph => changed(ph)));
  if (sp.some(q => q.more)) {
    const more = el("button", "morebtn", surfaceMore ? "less" : "more");
    more.onclick = () => { surfaceMore = !surfaceMore; sync(); };
    box.appendChild(more);
  }
  const set = [st.shading > 0 && "shading " + st.shading, st.wrap > 0 && "wrap " + st.wrap].filter(Boolean);
  $("surfaceBox").querySelector(".sum").textContent = set.length ? "· " + set.join(" · ") : "";
  $("sizeSeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.size === st.size));
  stage.marks(where.marks());
}

// The settings vary can sweep: this mode's own, the material's size, and the strength and surface
// ones in use.
function settings() {
  const out = spec().params.filter(p => p.group === null && ["float", "int", "log", "angle"].includes(p.kind)).map(p => [p]);
  out.push([param(spec(), "tex_scale")]);
  if (st.mask !== "none") {
    for (const [k, label] of STRENGTH.slice(0, 3)) out.push([param(spec(), k), label]);
    const cov = param(spec(), "coverage");
    if (applies(cov, st)) out.push([cov]);
  }
  for (const k of ["shading", "wrap"]) { const p = param(spec(), k); if (p) out.push([p]); }
  // vary sweeps values on the shared state or the mode's: both go into params(), so either works
  return out;
}

// ---------- the tiles: every mode drawn small from the pair, one after another ----------
let tileQueue = [];
const tileSeq = {};
function refreshTiles(only) {
  const keys = only ? [only] : [mode, ...MODES.filter(k => k !== mode)];
  for (const k of keys) $("tiles").querySelector(`[data-mode="${k}"]`).classList.add("waiting");
  tileQueue = [...keys, ...tileQueue.filter(k => !keys.includes(k))];
  nextTile();
}
function nextTile() {
  if (!tiles.idle) return;
  const k = tileQueue.shift();
  if (!k) return;
  tileSeq[k] = tiles.send({method: k, phase: "settle", inputs: {...inputs},
                           params: {...own[k], tex_scale: st.tex_scale, size: TILE, mask: "none", shading: 0, wrap: 0}});
}

// ---------- the view ----------
const before = (job) => {
  const c = job ? job.inputs && job.inputs.content : inputs.content;
  return c ? "/api/picture?name=" + encodeURIComponent(c) + "&size=" + ((job ? job.params.size : st.size) || 512) : null;
};

function setView(v) {
  stage.mark(v);
  if (v === "grid") return vary.showGrid();
  if (v === "result") {
    if (shown) return stage.compare(shown.url, before(shown), shown.method);
    if (frameUrl) return stage.compare(frameUrl, before(), mode);
    return;
  }
  if (v === "mask") {
    if (masks.url) stage.still(masks.url);
    const p = {size: Math.min(st.size, 768), tex_scale: st.tex_scale};
    for (const k of maskKeys(spec())) p[k] = st[k];
    status("finding the mask…");
    return masks.request({params: p, content: inputs.content, texture: inputs.texture});
  }
  stage.still("/api/picture?name=" + encodeURIComponent(inputs.content) + "&size=" + st.size);
}

function show(job) {
  shown = job;
  $("compare").style.setProperty("--x", "50%");
  stage.download(job);
  hist.select(job);
  setView("result");
  status(hist.here(job) ? describe(job) : job.method + " · " + madeIn(job));
}

function describe(job) {
  const p = job.params, i = job.inputs;
  const at = !p.mask || p.mask === "none" ? "" : " · " + (p.mask === "body part" ? p.part : p.mask === "object (words)"
    ? "“" + p.words + "”" : p.mask);
  return `${job.method} · ${nameOf(i.content)} × ${nameOf(i.texture)}${at}` + (job.took ? ` · ${job.took} s` : "");
}

function reuse(job) {
  if (!own[job.method]) return;
  mode = job.method;
  where.spec = spec();
  for (const k in job.params) {
    if (k === "_export") continue;
    if (k in own[mode]) own[mode][k] = job.params[k];
    else if (k in st) st[k] = job.params[k];
  }
  if (job.inputs.content) inputs.content = job.inputs.content;
  if (job.inputs.texture) inputs.texture = job.inputs.texture;
  changed("settle", true);
  show(job);
  status("settings taken from that render · change anything to draw live again");
}

// ---------- keep, export, recipe ----------
function keep() {
  if (shown) return status("that one is already in the history");
  live.keep();
}

async function exportFull() {
  stage.busy("exporting · 0 s");
  working(true);
  try {
    const job = await runJob(mode, {...params(), _export: true}, {...inputs}, s => stage.busy("exporting · " + s + " s"));
    hist.add(job);
    show(job);
    status("exported at full size · " + describe(job) + " · in the lab's exports folder");
  } catch (e) {
    status("the export failed: " + e.message, true);
  } finally {
    stage.busy(null);
    working(false);
  }
}

async function recipe() {
  const name = prompt("name this recipe (it appears in the lab's recipe list):",
                      "mapping · " + mode + " · " + nameOf(inputs.content) + " × " + nameOf(inputs.texture));
  if (name) await saveRecipe(name, {method: mode, params: params(), inputs: {...inputs}});
}

// ---------- pictures in ----------
function choose(target, name) {
  inputs[target] = name;
  if (target === "content") { st.points = []; st.points_fade = []; }
  changed("settle", true);
  status((target === "content" ? "picture: " : "texture: ") + nameOf(name));
}
