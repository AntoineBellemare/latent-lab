// Tokens: RAE's token grid as a studio. A picture and up to three material photographs become
// DINOv2's grids of tokens, and everything here happens to those grids before RAE's decoder draws
// them back: materials laid into the picture, tokens moved out of their places (moshing), or new
// textures made from the materials alone. Each way is a tile drawn from your own inputs; one dial
// moves its main setting and the rest sit under fine-tune; the lab's mask decides where, for the ways
// that work on a picture; and "surprise" lays out nine random recipes to go on from. Frames come over
// the live socket, enlarged four times by Real-ESRGAN on release; nothing is kept until you keep it.
import {$, el, menu, MENU, stored, store, arrivals, findJob, madeIn, status, runJob, working, Live, MaskPreview, maskKeys,
        tabOfJob, Stage, History, dropCard, dropZone, upload, onArrive, saveRecipe, nameOf} from "./studio.js";
import {WhereCard, slider, control, applies, param, tick, chips} from "./controls.js";
import {Vary} from "./vary.js";

// The ways: a method and the settings that make it this way (`set`, fixed, never shown as knobs),
// what its other settings start from (`start`), the one setting the dial moves, and whether it works
// on the picture or on the materials alone.
const WAYS = [
  {key: "laid", method: "compose", name: "laid in", on: "picture", dial: "picture", set: {},
   doc: "The materials laid into the picture where the mask says: each in its own region along the picture's light (or depth, or a side), all blended at every place, or each place taking the material it already resembles. The weights set how much of each, and under fine-tune each material can bring its own tokens, only its statistics, or a push along its direction. The decoder draws every seam. The dial mixes the picture's own tokens back in."},
  {key: "rebuilt", method: "mosh", name: "rebuilt", on: "picture", dial: "hold", set: {way: "rebuild", source: "material"},
   doc: "The picture rebuilt out of the materials' tokens, each piece the closest match to the one it replaces: every token drawn out of its own neighbourhood, the materials in the picture's layout. With two or three materials all are searched at once, and the weights tip which one a piece comes from. Richly textured materials show it best. The dial holds the picture."},
  {key: "self", method: "mosh", name: "self-moshed", on: "picture", dial: "pick", set: {way: "rebuild", source: "picture"}, start: {blocks: 2, pick: 4},
   doc: "The picture rebuilt out of its own tokens, every piece swapped for a look-alike from elsewhere in it: a face grows extra eyes, a dog more heads. The dial chooses among more look-alikes at random."},
  {key: "smeared", method: "mosh", name: "smeared", on: "picture", dial: "length", set: {way: "drag", source: "picture"}, start: {by: "noise", side: Math.PI, length: 10},
   doc: "Tokens pulled toward one side, each as far as a field says: the grid tears where neighbours part and streaks where many take the same token, as a datamosh blooms. The dial is how far."},
  {key: "sorted", method: "mosh", name: "sorted", on: "picture", dial: "span", set: {way: "sort", source: "picture"}, start: {along: "columns", span: 0.6},
   doc: "Along the columns (or rows), every run of the picture's brightest places has its tokens put in order of brightness: pixel sorting, a token at a time. The dial is how much of the picture is sorted."},
  {key: "blended", method: "hybrid", name: "blended", on: "materials", dial: "share", set: {pattern: "blend"},
   doc: "The materials mixed token by token: one surface at every point, never a double exposure. The dial walks from the first material through the second to the third."},
  {key: "patchwork", method: "hybrid", name: "patchwork", on: "materials", dial: "balance", set: {pattern: "patchwork"},
   doc: "Patches of each material in a grown pattern, every seam between them drawn by the decoder. The dial moves the boundaries; seam width and patch size are under fine-tune."},
  {key: "layout", method: "hybrid", name: "layout × grain", on: "materials", dial: "hold", set: {pattern: "layout"}, start: {blocks: 2},
   doc: "The first material's arrangement rebuilt out of the second's (and third's) tokens: one's layout in another's grain. The dial holds the first."},
  {key: "remixed", method: "hybrid", name: "remixed", on: "materials", dial: "reach", set: {pattern: "remix"},
   doc: "The first material's own pieces scattered over the frame: its grain in a new layout. The dial is how far a piece may travel."},
  {key: "drawn", method: "draw", name: "drawn new", on: "drawn", dial: "share", set: {},
   doc: "New material drawn from noise by the generator trained on your archive's crops, steered toward the materials' places in the space: not any photograph's tokens, its own. The dial walks between the materials' places, the noise held, so the drawing stays and its material changes; guidance and the seed are under fine-tune. Any frame: it draws in the 16 x 16 windows it learned and joins them, at 768 a 3:2 of 32 x 48 tokens, 2048 x 3072 enlarged, in about 3 s; rougher while you drag."},
];
const KINDS = [["into the picture", "picture"], ["new textures · the materials alone", "materials"],
               ["drawn new · the generator, steered by the materials", "drawn"]];
const SLOTS = [["texture", "first material"], ["textureB", "second material"], ["textureC", "third material"]];
const QUICK = [["none", "whole picture"], ["subject", "main subject"], ["person", "person"], ["light", "light"],
               ["shadow", "shadow"], ["object (words)", "named thing"]];
const SIZES = [512, 768, 1024];
const FRAMES = ["3:2", "1:1", "2:3"];
const OUTPUT = new Set(["long", "enlarge", "frame", "encoder"]);
// The mask's edge: shared by every way that works on the picture.
const EDGE = [["amount", "strength"], ["mask_soft", "feather"], ["mask_noise", "ragged edge"], ["grow", "grow / shrink"],
              ["follow", "follow contours"], ["mask_scale", "feature size"]];
const STORE_V = 2;   // 2: the material zoom starts at 1, so a zoom kept from before is let go
// The settings shown beside the dial when two or more materials are chosen: the mix, and each
// material's weight (named by the slot it belongs to, so an empty slot's weight is left out).
const MIXED = {mix: null, weight: "texture", weight_b: "textureB", weight_c: "textureC"};

let wayKey = "laid";
const own = {};               // each way's settings: key -> {param: value}
const st = {long: 768, enlarge: true, frame: "3:2", encoder: "DINOv2"};   // shared: the mask, the size, the frame, the enlargement
const inputs = {content: null, texture: null, textureB: "", textureC: ""};
let shown = null, frameUrl = null, gridOwner = "vary";
let stage, where, hist, vary, masks, live, tiles, slots = {}, surprises = [];

const way = (k = wayKey) => WAYS.find(w => w.key === k);
const spec = (w = way()) => MENU.methods[w.method];
const onPicture = (w = way()) => w.on === "picture";
const maskSpec = () => MENU.methods.compose;
const knobKeys = w => spec(w).params.filter(p => !maskKeys(spec(w)).includes(p.key) && !OUTPUT.has(p.key) &&
                                            !(p.key in w.set) && p.key !== w.dial).map(p => p.key);

function params(w = way()) {
  const p = {...own[w.key], long: st.long, enlarge: st.enlarge, _tab: "tokens"};
  if (w.method === "hybrid" || w.method === "draw") p.frame = st.frame;
  if (param(spec(w), "encoder")) p.encoder = st.encoder;
  if (onPicture(w)) for (const k of maskKeys(spec(w))) p[k] = st[k];
  return p;
}
function inputsFor(w = way()) {
  const mats = Object.fromEntries(SLOTS.map(([k]) => [k, inputs[k]]).filter(([, v]) => v));
  if (w.method === "hybrid" || w.method === "draw") return mats;
  if (w.key === "rebuilt") return {content: inputs.content, ...mats};   // rebuilt searches every material
  if (w.method === "mosh") return {content: inputs.content, texture: inputs.texture};
  return {content: inputs.content, ...mats};
}

// ---------- boot ----------
menu().then(m => {
  if (!m.methods.compose || !m.methods.mosh || !m.methods.hybrid) {
    status("the token methods are not loaded: models.json needs `rae` and `upscale`, and both files", true);
    return;
  }
  // The generator's way only once a trained checkpoint is there (tokendit.py).
  for (let i = WAYS.length - 1; i >= 0; i--) if (!m.methods[WAYS[i].method]) WAYS.splice(i, 1);
  for (const w of WAYS) {
    own[w.key] = {};
    for (const p of spec(w).params) if (!maskKeys(spec(w)).includes(p.key) && !OUTPUT.has(p.key)) own[w.key][p.key] = p.default;
    Object.assign(own[w.key], w.start || {}, w.set);
  }
  for (const k of maskKeys(maskSpec())) st[k] = param(maskSpec(), k).default;
  inputs.content = m.uploads.includes("4s-DSC01155.jpg") ? "u:4s-DSC01155.jpg" : m.contents[0];
  inputs.texture = m.textures.find(t => t.includes("DSC00587")) || m.textures[0];
  inputs.textureB = m.textures.find(t => t.includes("DSC06819")) || "";
  const got = stored("tokens");
  if (got && (got.v || 1) < STORE_V) for (const k in got.own || {}) delete got.own[k].zoom;
  if (got) {
    for (const k in got.own || {}) if (own[k]) Object.assign(own[k], got.own[k], way(k).set);
    for (const k in got.st || {}) if (k in st) st[k] = got.st[k];
    if (got.way && way(got.way)) wayKey = got.way;
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

const save = () => store("tokens", {v: STORE_V, own, st, way: wayKey, inputs});

// ---------- the controls ----------
function views() {
  return onPicture() ? [["result", "result"], ["mask", "mask"], ["picture", "picture"], ["material", "first material"]]
                     : [["result", "result"], ["material", "first material"]];
}

function build() {
  stage = new Stage($("view"), {views: views(), onView: setView, onPoint: (x, y, shift) => where.click(x, y, shift)});
  stage.mark("result");
  hist = new History({key: "tokens.history", here: j => tabOfJob(j) === "tokens", describe, onShow: show, onReuse: reuse});
  live = new Live({
    onFrame: (blob, meta) => {
      if (frameUrl) URL.revokeObjectURL(frameUrl);
      frameUrl = URL.createObjectURL(blob);
      if (!shown && stage.view === "result") stage.compare(frameUrl, before(), way().name);
      if (!shown) status(`${way().name} · ${meta.w}×${meta.h} · ${Math.round(meta.ms)} ms` +
                         (meta.phase === "drag" ? " · quick, enlarged when you let go" : ""));
    },
    onError: msg => status(msg.note, true),
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
  masks = new MaskPreview(url => {
    if (stage.view === "mask") stage.still(url);
    status(st.mask === "none" ? "no mask: the whole picture" : "the mask, in gold");
  }, note => status("mask: " + note, true));
  vary = new Vary(stage, {
    method: () => way().method, params: () => params(), inputs: () => inputsFor(), material: "texture",
    seeded: () => true, seedKey: "seed", settings, done: job => hist.add(job), show,
  });
  onArrive((target, name) => choose(target === "content" ? "content" : "texture", name));
  // the inputs: the picture, and a palette of three materials (the second and third may be left out)
  const pic = dropCard("content", {label: "a picture", get: () => inputs.content, choose: v => choose("content", v)});
  pic.classList.add("picslot");
  slots.content = pic;
  $("slots").appendChild(pic);
  for (const [key, label] of SLOTS) {
    const d = dropCard("texture", {label: "a material", get: () => inputs[key], choose: v => choose(key, v)});
    const base = d.refresh;
    d.refresh = () => {
      d.classList.toggle("empty", !inputs[key]);
      if (inputs[key]) return base();
      d.querySelector("img").removeAttribute("src");
      d.querySelector("b").textContent = "no " + label;
    };
    if (key !== "texture") {
      const x = el("button", "quiet clear", "×");
      x.title = "leave this material out";
      x.onclick = () => choose(key, "");
      d.appendChild(x);
    }
    slots[key] = d;
    $("slots").appendChild(d);
  }
  // the tiles
  for (const [kind, on] of KINDS) {
    if (!WAYS.some(x => x.on === on)) continue;
    $("tiles").appendChild(el("div", "kind", kind));
    const row = el("div", "row3");
    for (const w of WAYS.filter(x => x.on === on)) {
      const b = el("button", "mtile waiting");
      b.dataset.way = w.key;
      b.title = w.doc;
      const img = el("img");
      img.alt = "";
      b.append(img, el("span", "", w.name));
      b.onclick = () => pickWay(w.key);
      row.appendChild(b);
    }
    $("tiles").appendChild(row);
  }
  where = new WhereCard({root: $("where"), spec: maskSpec(), st, quick: QUICK,
                         onChange: moved => changed(moved ? "mask" : "settle", moved ? "picture" : null),
                         onArm: key => { stage.setPicking(!!key); if (key) setView("mask"); }});
  const mixBox = el("div", "args");
  mixBox.id = "mixBox";
  $("dial").after(mixBox);
  for (const [key, label] of EDGE) {
    const row = slider(param(maskSpec(), key), st, {label, doc: param(maskSpec(), key).doc,
                                                    onInput: () => changed("drag"), onChange: () => changed("settle", "picture")});
    row.dataset.edge = key;
    $("edge").appendChild(row);
  }
  $("edge").appendChild(tick("invert", st.invert, v => { st.invert = v; changed("mask", "picture"); }, param(maskSpec(), "invert").doc));
  $("edge").appendChild(control(param(maskSpec(), "mask_seed"), st, ph => changed(ph, ph === "settle" ? "picture" : null)));
  for (const s of SIZES) {
    const b = el("button", "", String(s));
    b.dataset.size = s;
    b.title = `the decoder draws ${s} on the long side; enlarged four times, ${s * 4}`;
    b.onclick = () => { st.long = s; changed("settle"); };
    $("sizeSeg").appendChild(b);
  }
  for (const f of FRAMES) {
    const b = el("button", "", f);
    b.dataset.frame = f;
    b.onclick = () => { st.frame = f; changed("settle", "materials"); refreshTiles("drawn"); };
    $("frameSeg").appendChild(b);
  }
  const encoders = MENU.methods.compose && param(MENU.methods.compose, "encoder");
  for (const e of encoders ? encoders.options : []) {
    const b = el("button", "", e);
    b.dataset.encoder = e;
    b.title = encoders.choices[e];
    b.onclick = () => { st.encoder = e; changed("settle"); refreshTiles(); };
    $("encoderSeg").appendChild(b);
  }
  $("enlargeRow").appendChild(tick("enlarge ×4 when you let go (Real-ESRGAN)", st.enlarge, v => { st.enlarge = v; changed("settle"); }));
  $("tuneBox").ontoggle = () => store("tokens.tune", $("tuneBox").open);
  $("tuneBox").open = !!stored("tokens.tune");
  $("keep").onclick = keep;
  $("export").onclick = exportFull;
  $("vary").onclick = () => { gridOwner = "vary"; vary.open(); };
  $("recipe").onclick = recipe;
  $("surprise").onclick = () => surprise();
  dropZone($("stageBox"), async f => choose(onPicture() ? "content" : "texture", await upload(f)));
  addEventListener("keydown", e => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "k" || e.key === "K") keep();
    if (e.key === "r" || e.key === "R") surprise();
  });
}

function pickWay(k) {
  wayKey = k;
  stage.views = views();
  stage.drawViews();
  changed("settle");
}

// What changed decides what is drawn: mid-drag a quick frame, on release the full one (enlarged),
// a change to the mask the mask itself too; `tilesToo` redraws the tiles it names.
function changed(phase, tilesToo) {
  shown = null;
  if (phase !== "drag") sync();  // sync rebuilds the panel: mid-drag it would replace the slider being dragged
  save();
  draw(phase === "drag" ? "drag" : "settle");
  if (phase === "mask") setView("mask");
  else if (stage.view !== "result" && phase !== "drag") setView("result");
  if (tilesToo) refreshTiles(tilesToo);
}

function draw(phase) {
  live.send({method: way().method, phase, params: params(), inputs: inputsFor()});
}

function sync() {
  const w = way();
  for (const d of Object.values(slots)) d.refresh();
  slots.content.classList.toggle("unused", !onPicture(w));
  $("tiles").querySelectorAll(".mtile").forEach(b => b.setAttribute("aria-pressed", b.dataset.way === wayKey));
  $("wayDoc").textContent = w.doc;
  const dial = $("dial");
  dial.innerHTML = "";
  const dp = param(spec(w), w.dial);
  dial.appendChild(slider(dp, own[w.key], {onInput: () => changed("drag"), onChange: () => changed("settle")}));
  const knobs = $("knobs"), mixBox = $("mixBox");
  knobs.innerHTML = "";
  mixBox.innerHTML = "";
  const used = inputsFor(w);
  const many = SLOTS.filter(([k]) => used[k]).length > 1;
  for (const k of knobKeys(w)) {
    const p = param(spec(w), k);
    if (!p || !applies(p, own[w.key])) continue;
    const slot = MIXED[k];
    if (slot !== undefined && (!many || (slot && !used[slot]))) continue;   // a weight of a material not chosen, or the mix of one
    (slot !== undefined ? mixBox : knobs).appendChild(control(p, own[w.key], ph => changed(ph, ph === "settle" ? w.key : null)));
  }
  $("tuneBox").querySelector(".sum").textContent = "· " + w.name;
  $("whereCard").hidden = !onPicture(w);
  if (onPicture(w)) where.sync();
  $("edgeCard").hidden = !onPicture(w) || st.mask === "none";
  $("edge").querySelectorAll("[data-edge]").forEach(r => r._set(st[r.dataset.edge]));
  $("frameRow").hidden = !["hybrid", "draw"].includes(w.method);
  $("encoderRow").hidden = !param(spec(w), "encoder");
  $("encoderSeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.encoder === st.encoder));
  $("sizeSeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", +b.dataset.size === st.long));
  $("frameSeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.frame === st.frame));
  stage.marks(onPicture(w) ? where.marks() : []);
}

// The settings vary can sweep: the dial and this way's other numbers.
function settings() {
  const w = way();
  return [w.dial, ...knobKeys(w)].map(k => param(spec(w), k))
    .filter(p => p && ["float", "int", "log", "angle"].includes(p.kind) && applies(p, own[w.key])).map(p => [p]);
}

// ---------- the tiles: every way drawn small from your inputs, one after another ----------
let tileQueue = [];
const tileOf = {};
function refreshTiles(which) {
  // "picture": the ways on the picture; "materials": the new textures; a way's key: that one; else all
  const keys = WAYS.filter(w => which === "picture" || which === "materials" ? w.on === which
    : typeof which === "string" ? w.key === which : true).map(w => w.key);
  const order = [wayKey, ...keys.filter(k => k !== wayKey)].filter(k => keys.includes(k));
  for (const k of order) $("tiles").querySelector(`[data-way="${k}"]`).classList.add("waiting");
  tileQueue = [...order, ...tileQueue.filter(k => !order.includes(k))];
  nextTile();
}
function nextTile() {
  if (!tiles.idle) return;
  const k = tileQueue.shift();
  if (!k) return;
  const w = way(k);
  const seq = tiles.send({method: w.method, phase: "drag", inputs: inputsFor(w), params: {...params(w), long: 512, enlarge: false}});
  tileOf[seq] = k;
}

// ---------- the view ----------
const pictureUrl = (name, kind, size) => "/api/picture?name=" + encodeURIComponent(name) + "&size=" + size + (kind ? "&kind=" + kind : "");
function before(job) {
  const w = job ? wayOf(job) : way();
  const i = job ? job.inputs : inputsFor(w);
  if (!w || !onPicture(w) || !i || !i.content) return null;   // a new texture has no before
  return pictureUrl(i.content, null, (job ? job.params.long : st.long) || 768);
}

function setView(v) {
  stage.mark(v);
  if (v === "grid") return gridOwner === "surprise" ? drawSurprises() : vary.showGrid();
  if (v === "result") {
    if (shown) return stage.compare(shown.url, before(shown), wayOf(shown) ? wayOf(shown).name : shown.method);
    if (frameUrl) return stage.compare(frameUrl, before(), way().name);
    return;
  }
  if (v === "mask") {
    if (masks.url) stage.still(masks.url);
    const p = {size: Math.min(st.long, 768)};
    for (const k of maskKeys(maskSpec())) p[k] = st[k];
    status("finding the mask…");
    return masks.request({params: p, content: inputs.content, texture: inputs.texture});
  }
  if (v === "material") {
    status("the first material: " + nameOf(inputs.texture) + " · the tokens are taken from its middle, " + own[wayKey].zoom + "× closer");
    return stage.still(pictureUrl(inputs.texture, "texture", st.long));
  }
  stage.still(pictureUrl(inputs.content, null, st.long));
  status("the picture: " + nameOf(inputs.content));
}

function show(job) {
  shown = job;
  $("compare").style.setProperty("--x", "50%");
  stage.download(job);
  hist.select(job);
  setView("result");
  status(hist.here(job) ? describe(job) : job.method + " · " + madeIn(job));
}

// Which way a render was made in: its method and the settings that make the way.
function wayOf(job) {
  return WAYS.find(w => w.method === job.method && Object.entries(w.set).every(([k, v]) => job.params[k] === v))
    || WAYS.find(w => w.method === job.method);
}

function describe(job) {
  const w = wayOf(job), i = job.inputs || {};
  const mats = SLOTS.map(([k]) => i[k]).filter(Boolean).map(nameOf).join(" + ");
  const on = w && onPicture(w) && i.content ? nameOf(i.content) + " · " : "";
  return `${w ? w.name : job.method} · ${on}${mats}` + (job.took ? ` · ${job.took} s` : "");
}

function reuse(job) {
  const w = wayOf(job);
  if (!w) return show(job);
  wayKey = w.key;
  for (const k in job.params) {
    if (k.startsWith("_")) continue;
    if (k in own[w.key]) own[w.key][k] = job.params[k];
    else if (k in st) st[k] = job.params[k];
  }
  for (const k of ["content", ...SLOTS.map(s => s[0])]) if (job.inputs && k in job.inputs) inputs[k] = job.inputs[k];
  stage.views = views();
  stage.drawViews();
  changed("settle", true);
  show(job);
  status("settings taken from that render · change anything to draw live again");
}

// ---------- surprise: nine random recipes, drawn small, to go on from ----------
let surpriseFrom = "palette", surpriseRun = 0;
const surpriseLive = new Live({
  onFrame: (blob, meta) => {
    const c = surprises.find(x => x.seq === meta.seq);
    if (c) { c.url = URL.createObjectURL(blob); c.state = "done"; }
    if (gridOwner === "surprise" && stage.view === "grid") drawSurprises();
    nextSurprise();
  },
  onError: msg => {
    const c = surprises.find(x => x.state === "drawing");
    if (c) c.state = "failed";
    nextSurprise();
  },
});
const rnd = (lo, hi) => lo + Math.random() * (hi - lo);
function recipeAt() {
  const w = WAYS[Math.floor(Math.random() * WAYS.length)];
  const p = {...own[w.key]};
  const dp = param(spec(w), w.dial);
  const v = rnd(dp.lo, dp.hi);
  p[w.dial] = dp.kind === "int" ? Math.round(v) : +v.toFixed(2);
  if ("seed" in p) p.seed = Math.floor(Math.random() * 99999);
  if ("blocks" in p && !("blocks" in w.set)) p.blocks = [1, 2, 4][Math.floor(Math.random() * 3)];
  if (w.method === "hybrid" && w.set.pattern === "patchwork") { p.patch = +rnd(4, 24).toFixed(1); p.seam = +rnd(0.1, 0.9).toFixed(2); }
  if (w.method === "mosh" && w.set.way === "drag") { p.side = [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2][Math.floor(Math.random() * 4)];
                                                     p.by = ["light", "depth", "noise"][Math.floor(Math.random() * 3)]; }
  if (w.method === "compose") p.order = ["light", "depth", "across"][Math.floor(Math.random() * 3)];
  // Your palette in a random order, or two or three of every texture at random.
  const pool = surpriseFrom === "palette" ? SLOTS.map(([k]) => inputs[k]).filter(Boolean)
    : [...MENU.textures].sort(() => Math.random() - 0.5).slice(0, Math.random() < 0.4 ? 3 : 2);
  const order = [...pool].sort(() => Math.random() - 0.5);
  const mats = {texture: order[0], textureB: order[1] || "", textureC: order[2] || ""};
  return {w, p, mats};
}
function surprise() {
  gridOwner = "surprise";
  surpriseRun += 1;
  for (const c of surprises) if (c.url) URL.revokeObjectURL(c.url);
  surprises = Array.from({length: 9}, recipeAt).map(r => ({...r, state: "waiting"}));
  stage.offerGrid(true);
  setView("grid");
  nextSurprise();
}
function nextSurprise() {
  if (!surpriseLive.idle) return;
  const c = surprises.find(x => x.state === "waiting");
  if (!c) return status("nine surprises · click one to go on from it · R for nine more");
  c.state = "drawing";
  const saved = {...inputs};
  Object.assign(inputs, c.mats);
  const req = {method: c.w.method, phase: "drag", params: {...params(c.w), ...c.p, long: 512, enlarge: false}, inputs: inputsFor(c.w)};
  Object.assign(inputs, saved);
  c.seq = surpriseLive.send(req);
}
function drawSurprises() {
  stage.grid();
  const bar = $("gridBar");
  bar.innerHTML = "";
  const done = surprises.filter(c => c.state === "done").length;
  bar.append(el("span", "lab", `surprise · ${done} of ${surprises.length}`), el("span", "grow"));
  bar.appendChild(chips([["palette", "your materials"], ["all", "any texture"]], surpriseFrom, v => { surpriseFrom = v; surprise(); }, true));
  const again = el("button", "", "nine more");
  again.onclick = () => surprise();
  bar.appendChild(again);
  const cells = $("gridCells");
  cells.style.setProperty("--cols", 3);
  cells.innerHTML = "";
  for (const c of surprises) {
    const b = el("button", "cell");
    b.disabled = c.state !== "done";
    if (c.url) { const img = el("img"); img.src = c.url; img.alt = ""; b.appendChild(img); }
    else b.appendChild(el("div", "wait", c.state));
    const used = c.w.set.source === "picture" ? [nameOf(inputs.content)] : c.w.method === "mosh" ? [nameOf(c.mats.texture)]
      : [c.mats.texture, c.mats.textureB, c.mats.textureC].filter(Boolean).map(nameOf);
    b.appendChild(el("span", "", `${c.w.name} · ${param(spec(c.w), c.w.dial).label} ${c.p[c.w.dial]} · ${used.join(" + ")}`));
    b.title = "go on from this one: its way, settings and materials, drawn live";
    b.onclick = () => adopt(c);
    cells.appendChild(b);
  }
}
function adopt(c) {
  wayKey = c.w.key;
  Object.assign(own[c.w.key], c.p, c.w.set);
  Object.assign(inputs, c.mats);
  stage.views = views();
  stage.drawViews();
  changed("settle", true);
  setView("result");
  status(`${c.w.name} · taken from the surprises · change anything to go on`);
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
    status("exported, enlarged four times · " + describe(job) + " · in the lab's exports folder");
  } catch (e) {
    status("the export failed: " + e.message, true);
  } finally {
    stage.busy(null);
    working(false);
  }
}

async function recipe() {
  const name = prompt("name this recipe (it appears in the lab's recipe list):", "tokens · " + describe({method: way().method, params: params(), inputs: inputsFor()}));
  if (name) await saveRecipe(name, {method: way().method, params: params(), inputs: inputsFor()});
}

// ---------- pictures in ----------
function choose(target, name) {
  inputs[target] = name;
  if (target === "content") { st.points = []; st.points_fade = []; }
  changed("settle", target === "content" ? "picture" : true);
  status((target === "content" ? "picture: " : SLOTS.find(s => s[0] === target)[1] + ": ") + (name ? nameOf(name) : "none"));
}
