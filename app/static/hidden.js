// Hidden: the texture leads, and a face from another picture hides in it — the other end of the
// scale the editor (the picture leads) and mapping (the two meet) sit on. Six ways, from instant to
// drawn, each a tile drawn from your own pair; one slider from hidden to plain that moves each way's
// own setting; and a stage made for finding a hidden face: drag to bring the face back, peek, squint.
// Every change draws over the live socket (the drawn ways on release only); keep puts it in the
// history. The stage, history, vary and send-to are the tabs' shared parts.
import {$, el, menu, MENU, stored, store, arrivals, findJob, madeIn, status, runJob, working, Live, tabOfJob,
        Stage, History, dropCard, dropZone, upload, onArrive, saveRecipe, nameOf} from "./studio.js";
import {control, param} from "./controls.js";
import {Vary} from "./vary.js";

// The ways. `dial` is each one's own setting that the slider moves, through five anchors measured on
// four face and texture pairs (from hidden to plain); `set` is what it starts from; `knobs` are the
// rest of its settings, under fine-tune. `flipped`: the editor redraws the texture itself, shown the
// face as its reference, so the two inputs swap places.
const WAYS = [
  {key: "carve", method: "carve", name: "carved", instant: true, dial: "depth", at: [0.15, 0.4, 0.8, 1.5, 2.5],
   knobs: ["angle", "elevation", "ambient", "hollow", "push", "reach", "carve_on", "tex_scale"],
   doc: "The face's real surfaces, read by a geometry model, cut into the texture as light and hollows. Every pixel stays the texture photograph."},
  {key: "blend", method: "blend", name: "traded edges", instant: true, dial: "wmix", at: [0.2, 0.35, 0.5, 0.65, 0.8],
   knobs: ["tex_scale"],
   doc: "At each pixel the stronger edge wins, the face's or the texture's: cracks become wrinkles, ripples an outline."},
  {key: "bands", method: "bands", name: "hybrid image", instant: true, dial: "split", at: [24, 14, 8, 4.5, 2.5], log: true,
   knobs: ["grain", "tex_scale"], set: {grain: 1.2},
   doc: "Texture up close, the face from a distance: the face's broad light under the texture's fine grain. Best with a face lighter than its surroundings."},
  {key: "displace", method: "displace", name: "pushed", instant: true, dial: "shade", at: [0, 0.2, 0.4, 0.65, 1.0],
   knobs: ["amp", "reach", "tex_scale"], set: {amp: 0.05, reach: 4},
   doc: "The texture bent like liquid along the face's slopes, and shaded by its light. Most at home on water."},
  {key: "illusion", method: "illusion", name: "illusion", instant: false, dial: "illusion", at: [0.6, 0.85, 1.1, 1.4, 1.8],
   knobs: ["signal", "hide", "soften", "let_go", "strength", "scale", "seed"], set: {quality: "full"},
   doc: "The texture redrawn in its own material by a ControlNet made for hidden pictures, bending its light and dark toward the face's form. Texture up close, the face when you squint. A few seconds."},
  {key: "asked", method: "edit", name: "asked for", instant: false, flipped: true, dial: "strength", at: [0.6, 0.7, 0.8, 0.9, 1.0],
   knobs: ["seed"], set: {swatch: "texture", mask: "none", quality: "full", focus: "whole picture"},
   doc: "The editor asked, in words, for a face made of the texture's own material, with your face as its reference. Low: an eye or two shows through; high: the whole face. A face after yours, not your likeness. A few seconds."},
];
const KINDS = [["instant", WAYS.filter(w => w.instant)], ["drawn · a few seconds", WAYS.filter(w => !w.instant)]];
const SIZES = [512, 768, 1024];
const TILE = 160;
const VIS = {key: "_vis", kind: "float", lo: 0, hi: 1, step: 0.05, default: 0.5, label: "hidden → plain",
             doc: "from hidden (0) to plain (1), each way's own setting moved with it"};

let wayKey = "carve";
let vis = 0.5;
const inputs = {content: null, texture: null};   // content: the face; texture: what it hides in
const own = {};                                   // each way's settings: key -> {param: value}
const st = {size: 768, words: ""};
let shown = null, frameUrl = null, squint = false;
let stage, hist, vary, live, tiles, hintLive, drops;

const way = (k = wayKey) => WAYS.find(w => w.key === k);
const spec = (w = way()) => MENU.methods[w.method];

// Where the slider sits, as the way's own setting: piecewise between the five anchors (in log for a
// setting that works in scales).
function valueAt(w, t) {
  const x = Math.max(0, Math.min(1, t)) * (w.at.length - 1), i = Math.min(Math.floor(x), w.at.length - 2), f = x - i;
  const a = w.at[i], b = w.at[i + 1];
  const v = w.log ? Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * f) : a + (b - a) * f;
  return +v.toFixed(3);
}

function instruction() {
  const what = st.words.trim();
  return `Reshape this ${what || "texture"} so that a large human face, the face in the reference image, appears in it, ` +
    `made entirely of its own material: the eyes, nose, mouth and jaw formed by its own cracks, ridges, colours and ` +
    `shadows. It stays a close-up photograph of ${what || "the same material"}.`;
}

// Everything a way is drawn with: its settings, the slider's position as its own setting, the tab's
// size, and the tab's mark so the history knows where it was made.
function params(w = way(), t = vis) {
  const p = {...own[w.key], size: w.flipped ? Math.max(768, st.size) : st.size, _tab: "hidden", _vis: +t.toFixed(3)};
  p[w.dial] = valueAt(w, t);
  if (w.key === "illusion") p.prompt = `a close-up photograph of ${st.words.trim() || "a weathered natural surface"}, richly detailed`;
  if (w.flipped) p.prompt = instruction();
  return p;
}
const inputsFor = (w = way()) => w.flipped ? {content: inputs.texture, texture: inputs.content}
                                           : {content: inputs.content, texture: inputs.texture};

// ---------- boot ----------
menu().then(m => {
  for (const w of WAYS) {
    own[w.key] = {...(w.set || {})};
    for (const k of w.knobs) if (!(k in own[w.key])) own[w.key][k] = param(spec(w), k).default;
  }
  inputs.content = m.uploads.includes("hf-face-sombrero.jpg") ? "u:hf-face-sombrero.jpg" : m.contents[0];
  inputs.texture = m.textures.find(t => t.includes("DSC07301")) || m.textures[0];
  const got = stored("hidden");
  if (got) {
    for (const k in got.own || {}) if (own[k]) Object.assign(own[k], got.own[k]);
    if (got.way && way(got.way)) wayKey = got.way;
    if (typeof got.vis === "number") vis = got.vis;
    Object.assign(st, got.st || {});
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

const save = () => store("hidden", {own, way: wayKey, vis, st, inputs});

// ---------- the controls ----------
function build() {
  stage = new Stage($("view"), {views: views(), onView: setView, onPoint: () => {}});
  stage.mark("result");
  $("compare").style.setProperty("--x", "0%");
  // the stage's own: peek (hold to see the face) and squint
  const peek = el("button", "quiet", "peek");
  peek.id = "peek";
  peek.title = "hold to see the face the texture hides";
  const hold = on => { if (stage.view === "result") $("compare").style.setProperty("--x", on ? "100%" : "0%"); };
  peek.onpointerdown = () => hold(true);
  ["pointerup", "pointerleave", "pointercancel"].forEach(t => peek.addEventListener(t, () => hold(false)));
  const sq = el("button", "quiet toggle", "squint");
  sq.title = "blur every picture, the way a hidden face is found from across a room (S)";
  sq.onclick = () => setSquint(!squint);
  $("status").after(peek, sq);
  hist = new History({key: "hidden.history", here: j => tabOfJob(j) === "hidden", describe, onShow: show, onReuse: reuse});
  live = new Live({
    onFrame: (blob, meta) => {
      if (frameUrl) URL.revokeObjectURL(frameUrl);
      frameUrl = URL.createObjectURL(blob);
      if (meta.seq === live.seq) stage.busy(null);
      if (!shown && stage.view === "result") setView("result");
      if (!shown) status(`${way().name} · ${meta.w}×${meta.h} · ${Math.round(meta.ms)} ms` + (meta.phase === "drag" ? " · quick" : ""));
    },
    onError: msg => { stage.busy(null); status(msg.note, true); },
    onKept: job => { hist.add(job); show(job); status("kept · in the history, and ready to send or download"); },
  });
  tiles = new Live({
    onFrame: (blob, meta) => {
      const k = tileOf[meta.seq];
      const img = k && $("tiles").querySelector(`[data-way="${k}"] img`);
      if (img) {
        if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
        img.src = URL.createObjectURL(blob);
        img.parentElement.classList.remove("waiting");
      }
      nextTile();
    },
    onError: () => nextTile(),
  });
  hintLive = new Live({
    onFrame: blob => { if (stage.view === "hint") stage.still(URL.createObjectURL(blob)); status("what the illusion hides: light and dark it bends the texture toward; grey is left free"); },
    onError: msg => status("hint: " + msg.note, true),
  });
  vary = new Vary(stage, {
    method: () => way().method, params: () => params(), inputs: () => inputsFor(),
    get material() { return way().flipped ? "content" : "texture"; },
    seeded: () => "seed" in own[wayKey], seedKey: "seed",
    settings: () => [[VIS, "hidden → plain"], ...way().knobs.map(k => param(spec(), k))
      .filter(p => p && ["float", "int", "log", "angle"].includes(p.kind)).map(p => [p])],
    // the one slider becomes the way's own setting, cell by cell
    adjust: p => ({...p, [way().dial]: valueAt(way(), p._vis)}),
    done: job => hist.add(job), show,
  });
  onArrive(choose);
  drops = {content: dropCard("content", {label: "a face", get: () => inputs.content, choose: v => choose("content", v)}),
           texture: dropCard("texture", {label: "a texture", get: () => inputs.texture, choose: v => choose("texture", v)})};
  $("pairLinks").before(drops.content, drops.texture);
  const pair = () => "#content=" + encodeURIComponent(inputs.content) + "&texture=" + encodeURIComponent(inputs.texture);
  $("toMapping").onclick = () => location.assign("/mapping.html" + pair());
  $("toEditor").onclick = () => location.assign("/editor.html" + pair());
  // the slider
  const v = $("vis");
  v.value = vis;
  v.oninput = () => { vis = +v.value; changed("drag"); };
  v.onchange = () => changed("settle", "instant");
  v.ondblclick = () => { vis = 0.5; v.value = vis; changed("settle", "instant"); };
  // the tiles
  for (const [kind, ways] of KINDS) {
    $("tiles").appendChild(el("div", "kind", kind));
    const row = el("div", "row3");
    for (const w of ways) {
      const b = el("button", "mtile waiting");
      b.dataset.way = w.key;
      b.title = w.doc;
      const img = el("img");
      img.alt = "";
      b.append(img, el("span", "", w.name));
      b.onclick = () => { wayKey = w.key; stage.views = views(); stage.drawViews(); changed("settle"); };
      row.appendChild(b);
    }
    $("tiles").appendChild(row);
  }
  $("words").value = st.words;
  $("words").onchange = () => { st.words = $("words").value.trim(); changed("settle", "drawn"); };
  $("tuneBox").ontoggle = () => store("hidden.tune", $("tuneBox").open);
  $("tuneBox").open = !!stored("hidden.tune");
  for (const s of SIZES) {
    const b = el("button", "", String(s));
    b.dataset.size = s;
    b.onclick = () => { st.size = s; changed("settle"); };
    $("sizeSeg").appendChild(b);
  }
  $("keep").onclick = keep;
  $("export").onclick = exportFull;
  $("vary").onclick = () => vary.open();
  $("recipe").onclick = recipe;
  dropZone($("stageBox"), async f => choose("texture", await upload(f)));
  addEventListener("keydown", e => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "k" || e.key === "K") keep();
    if (e.key === "s" || e.key === "S") setSquint(!squint);
  });
}

function views() {
  const out = [["result", "result"], ["face", "face"], ["texture", "texture"]];
  if (wayKey === "illusion") out.push(["hint", "what is hidden"]);
  return out;
}

function setSquint(on) {
  squint = on;
  $("stageBox").classList.toggle("squint", on);
  $("tiles").classList.toggle("squint", on);
  document.querySelector("#viewbar .toggle").setAttribute("aria-pressed", on);
}

// What changed decides what is drawn: mid-drag a quick frame (the instant ways only), on release the
// full one; `tilesToo` redraws the tiles it names ("instant", "drawn", a way's key, or all).
function changed(phase, tilesToo) {
  shown = null;
  sync();
  save();
  draw(phase === "drag" ? "drag" : "settle");
  if (stage.view !== "result" && phase !== "drag") setView(stage.view === "hint" && wayKey === "illusion" ? "hint" : "result");
  if (tilesToo) refreshTiles(tilesToo);
}

function draw(phase) {
  const w = way();
  if (!w.instant && phase === "drag") return;
  if (!w.instant) stage.busy("drawing · " + w.name);
  live.send({method: w.method, phase, params: params(), inputs: inputsFor()});
}

function sync() {
  drops.content.refresh();
  drops.texture.refresh();
  const w = way();
  $("tiles").querySelectorAll(".mtile").forEach(b => b.setAttribute("aria-pressed", b.dataset.way === wayKey));
  $("wayDoc").textContent = w.doc;
  const p = param(spec(), w.dial);
  $("visNote").textContent = `${w.name}: ${(p && p.label) || w.dial} ${valueAt(w, vis)}` +
    (w.instant ? "" : " · drawn when you let go");
  $("wordsRow").hidden = w.instant;
  const knobs = $("knobs");
  knobs.innerHTML = "";
  for (const k of w.knobs) {
    const q = param(spec(), k);
    if (q) knobs.appendChild(control(q, own[wayKey], ph => changed(ph, ph === "settle" && w.instant ? wayKey : null)));
  }
  $("tuneBox").querySelector(".sum").textContent = "· " + w.name;
  $("sizeSeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.size === st.size));
}

// ---------- the tiles: every way drawn from the pair, the instant ones first ----------
let tileQueue = [];
const tileOf = {};
function refreshTiles(which) {
  const keys = WAYS.filter(w => which === "instant" ? w.instant : which === "drawn" ? !w.instant
    : typeof which === "string" && way(which) ? w.key === which : true).map(w => w.key)
    .sort((a, b) => way(b).instant - way(a).instant);
  for (const k of keys) $("tiles").querySelector(`[data-way="${k}"]`).classList.add("waiting");
  tileQueue = [...keys, ...tileQueue.filter(k => !keys.includes(k))];
  nextTile();
}
function nextTile() {
  if (!tiles.idle) return;
  const k = tileQueue.shift();
  if (!k) return;
  const w = way(k);
  // instant ways small; drawn ways at their smallest real size, full quality (a draft misleads)
  const seq = tiles.send({method: w.method, phase: "settle", inputs: inputsFor(w),
                          params: {...params(w), size: w.instant ? TILE : w.flipped ? 768 : 512}});
  tileOf[seq] = k;
}

// ---------- the view ----------
// The result's "before" is what it hides: the face, laid over the result for the drag and the peek;
// the editor's flipped result is in the texture's own frame, so its before is the texture.
function before(job) {
  const w = job ? WAYS.find(x => x.method === job.method) || way() : way();
  const i = job ? job.inputs : inputsFor(w);
  const size = (job ? job.params.size : params(w).size) || 768;
  return i && i.content ? "/api/picture?name=" + encodeURIComponent(i.content) + "&size=" + size : null;
}

function setView(v) {
  stage.mark(v);
  if (v === "grid") return vary.showGrid();
  if (v === "result") {
    // the left of the drag is what the result hides: the face, or for the flipped editor the texture
    const flipped = shown ? shown.method === "edit" : way().flipped;
    document.querySelector("#compare .tag.l").textContent = flipped ? "texture" : "face";
    if (shown) return stage.compare(shown.url, before(shown), shown.method);
    if (frameUrl) return stage.compare(frameUrl, before(), way().name);
    return;
  }
  if (v === "hint") {
    status("reading what is hidden…");
    return hintLive.send({method: "illusion", phase: "settle", params: {...params(way("illusion")), show_hint: true},
                         inputs: inputsFor(way("illusion"))});
  }
  const name = v === "face" ? inputs.content : inputs.texture;
  stage.still("/api/picture?name=" + encodeURIComponent(name) + "&size=" + st.size + (v === "texture" ? "&kind=texture" : ""));
  status(v === "face" ? "the face to hide: " + nameOf(inputs.content) : "the texture it hides in: " + nameOf(inputs.texture));
}

function show(job) {
  shown = job;
  $("compare").style.setProperty("--x", "0%");
  stage.download(job);
  hist.select(job);
  setView("result");
  status(hist.here(job) ? describe(job) : job.method + " · " + madeIn(job));
}

function describe(job) {
  const w = WAYS.find(x => x.method === job.method);
  const flipped = w && w.flipped;
  const face = flipped ? job.inputs.texture : job.inputs.content, tex = flipped ? job.inputs.content : job.inputs.texture;
  const t = job.params._vis;
  return `${w ? w.name : job.method} · ${nameOf(face)} in ${nameOf(tex)}` +
    (typeof t === "number" ? ` · ${Math.round(t * 100)}% plain` : "") + (job.took ? ` · ${job.took} s` : "");
}

// A render's settings back in the panel, to go on from it.
function reuse(job) {
  const w = WAYS.find(x => x.method === job.method);
  if (!w) return show(job);
  wayKey = w.key;
  for (const k in job.params) if (k in own[w.key]) own[w.key][k] = job.params[k];
  if (typeof job.params._vis === "number") vis = job.params._vis;
  $("vis").value = vis;
  const i = job.inputs;
  inputs.content = w.flipped ? i.texture : i.content;
  inputs.texture = w.flipped ? i.content : i.texture;
  stage.views = views();
  stage.drawViews();
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
    const job = await runJob(way().method, {...params(), _export: true}, inputsFor(), s => stage.busy("exporting · " + s + " s"));
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
                      "hidden · " + way().name + " · " + nameOf(inputs.content) + " in " + nameOf(inputs.texture));
  if (name) await saveRecipe(name, {method: way().method, params: params(), inputs: inputsFor()});
}

// ---------- pictures in ----------
function choose(target, name) {
  inputs[target] = name;
  changed("settle", true);
  status((target === "content" ? "face: " : "texture: ") + nameOf(name));
}
