// The material editor on a page of its own: a picture and a material dropped in, where it goes
// chosen and seen, how strongly, and each render laid over the picture to compare. It drives the
// lab's own `edit` method through the same job queue, so every render is in the lab's history too;
// the mask is previewed through the live socket. The stage, history, picker, vary and send-to are
// the tabs' shared parts (studio.js, controls.js, vary.js).
import {$, el, menu, MENU, stored, store, arrivals, findJob, madeIn, status, runJob, working, MaskPreview, maskKeys, tabOfJob,
        Stage, History, dropCard, dropZone, upload, onArrive, saveRecipe, nameOf} from "./studio.js";
import {WhereCard, slider, param} from "./controls.js";
import {Vary} from "./vary.js";

let SPEC = null;
const st = {};                                   // the edit's settings, every one of them
const inputs = {content: null, texture: null, preset: ""};
let shown = null;                                // the render on show: a job record
let stage, where, hist, vary, masks, drops;

const QUICK = [["none", "whole picture"], ["body part", "body part"], ["object (words)", "named thing"],
               ["subject", "main subject"], ["person", "person"], ["object (click)", "click on it"]];
// The strength card: the edit's own knobs under plainer names; `mask` ones only count with a mask.
const SLIDERS = [["strength", "redraw", "how much of the masked part is redrawn: lower, the picture's own shows through more"],
                 ["amount", "material strength", "how fully the result replaces the picture inside the mask", true],
                 ["mask_soft", "soft edge", "how gradually the material fades out at the mask's edge", true],
                 ["follow", "follow contours", "snap the mask's edge onto the picture's own outlines", true],
                 ["grow", "grow / shrink", "push the mask's edge outward or inward", true],
                 ["harmonise", "settle colours", "move the material's colour and tone toward the rest of the picture", true]];

// ---------- boot ----------
menu().then(m => {
  SPEC = m.methods.edit;
  for (const p of SPEC.params) st[p.key] = p.default;
  Object.assign(st, {swatch: "texture", mask: "body part", part: "clothing", quality: "draft", seed: 1});
  inputs.content = m.contents[0];
  inputs.texture = m.textures[0];
  inputs.preset = m.presets[0] || "";
  const got = stored("studio");
  if (got) {
    for (const k in got.st || {}) if (k in st) st[k] = got.st[k];
    Object.assign(inputs, got.inputs || {});
  }
  build();
  hist.load(m.history);
  const sent = arrivals();
  if (sent.content) choose("content", sent.content);
  if (sent.texture) choose("texture", sent.texture);
  const job = sent.job && findJob(sent.job);
  if (job) { reuse(job); show(job); }
  else if (!sent.content) {
    const last = m.history.filter(h => h.method === "edit" && h.url).pop();
    if (last) show(last); else setView("picture");
  }
});

const save = () => store("studio", {st, inputs});

// ---------- the controls ----------
function build() {
  stage = new Stage($("view"), {views: [["result", "result"], ["mask", "mask"], ["picture", "picture"]],
                                onView: setView, onPoint: (x, y, shift) => where.click(x, y, shift)});
  hist = new History({key: "studio.history", here: j => tabOfJob(j) === "editor", describe, onShow: show, onReuse: reuse});
  masks = new MaskPreview(url => {
    if (stage.view === "mask") stage.still(url);
    status(st.mask === "none" ? "no mask: the whole picture is redrawn" : "the mask, in gold");
  }, note => status("mask: " + note, true));
  vary = new Vary(stage, {
    method: () => "edit", params: () => ({...st, prompt: (st.prompt || "").trim()}), inputs: jobInputs,
    material: "texture", materialParams: () => ({swatch: "texture"}), seeded: () => true,
    settings: () => SLIDERS.filter(s => !s[3] || st.mask !== "none").map(([k, label]) => [param(SPEC, k), label])
      .concat(st.mask !== "none" && st.fade !== "off" ? [[param(SPEC, "fade_reach"), "gradient reach"]] : []),
    done: job => hist.add(job), show,
  });
  onArrive(choose);
  // the picture and the material
  drops = {content: dropCard("content", {label: "a picture", get: () => inputs.content, choose: v => choose("content", v)}),
           texture: dropCard("texture", {label: "a texture", get: () => inputs.texture, choose: v => choose("texture", v)})};
  $("pictureCard").appendChild(drops.content);
  $("swatchSeg").after(drops.texture);
  $("swatchSeg").querySelectorAll("button").forEach(b => b.onclick = () => { st.swatch = b.dataset.swatch; changed(); });
  const presets = $("preset");
  for (const p of MENU.presets) presets.add(new Option(p, p));
  presets.value = inputs.preset;
  presets.onchange = () => { inputs.preset = presets.value; changed(); };
  // where
  where = new WhereCard({root: $("where"), spec: SPEC, st, quick: QUICK, onChange: changed,
                         onArm: key => { stage.setPicking(!!key); if (key) setView("mask"); }});
  const focus = el("label", "tick");
  focus.title = "when the mask holds only part of the picture, the editor draws just its surroundings: faster, " +
    "and a small thing gets more pixels";
  const box = el("input");
  box.type = "checkbox"; box.id = "focus";
  box.onchange = () => { st.focus = box.checked ? "around the mask" : "whole picture"; changed(); };
  focus.append(box, " work around the mask (faster, finer)");
  where.extra.appendChild(focus);
  // strength
  for (const [key, label, doc, masked] of SLIDERS) {
    const row = slider(param(SPEC, key), st, {label, doc, onChange: () => changed(masked)});
    if (masked) row.dataset.masked = "1";
    $("sliders").appendChild(row);
  }
  // instruction
  $("prompt").value = st.prompt || "";
  $("prompt").oninput = () => { st.prompt = $("prompt").value; save(); };
  // output
  $("qualitySeg").querySelectorAll("button").forEach(b => b.onclick = () => { st.quality = b.dataset.q; changed(); });
  const size = $("size");
  for (const o of param(SPEC, "size").options) size.add(new Option(o + " px", o));
  size.value = st.size;
  size.onchange = () => { st.size = +size.value; changed(); };
  $("seed").value = st.seed;
  $("seed").onchange = () => { st.seed = +$("seed").value; save(); };
  $("dice").onclick = () => { st.seed = Math.floor(Math.random() * 99999); $("seed").value = st.seed; save(); };
  $("render").onclick = () => render(false);
  $("export").onclick = () => render(true);
  $("vary").onclick = () => vary.open();
  $("recipe").onclick = recipe;
  $("flip").onclick = () => location.assign("/hidden.html#content=" + encodeURIComponent(inputs.content) +
                                            "&texture=" + encodeURIComponent(inputs.texture));
  dropZone($("stageBox"), async f => choose("content", await upload(f)));
  addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) render(false); });
  sync();
}

// Everything the panel shows follows the settings; a change to where the mask goes refreshes its
// preview and turns the view to it, so the mask is seen while it is being decided.
function changed(maskMoved) {
  sync();
  save();
  if (maskMoved) setView("mask");
}

function sync() {
  const on = (seg, attr, v) => seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset[attr] === String(v)));
  on($("swatchSeg"), "swatch", st.swatch);
  on($("qualitySeg"), "q", st.quality);
  drops.texture.hidden = st.swatch !== "texture";
  $("spaceBox").hidden = st.swatch !== "space";
  drops.content.refresh();
  drops.texture.refresh();
  where.sync();
  const masked = st.mask !== "none";
  $("focus").checked = st.focus !== "whole picture";
  $("focus").parentElement.hidden = !masked;
  for (const row of $("sliders").children) {
    row.hidden = row.dataset.masked && !masked;
    row._set(st[row.dataset.key]);
  }
  $("prompt").placeholder = SPEC.instruction.replaceAll("{what}", what() || "the masked area");
  stage.marks(where.marks());
}

// What the mask selects, in the words the editor's default instruction uses (registry: `named`).
function what() {
  return {"object (words)": (st.words || "").trim() || null, "body part": "the " + st.part,
          "person": "the person's clothes", "subject": "the main subject",
          "object (click)": "the selected object"}[st.mask] || null;
}

// ---------- the view ----------
function setView(v) {
  stage.mark(v);
  if (v === "grid") return vary.showGrid();
  if (v === "result" && shown) return stage.compare(shown.url, before(shown), shown.method === "edit" ? "edited" : shown.method);
  if (v === "mask") {
    if (masks.url) stage.still(masks.url);
    const params = {size: 768};
    for (const k of maskKeys(SPEC)) params[k] = st[k];
    status("finding the mask…");
    return masks.request({params, content: inputs.content});
  }
  stage.mark(v === "result" ? "picture" : v);
  stage.still("/api/picture?name=" + encodeURIComponent(inputs.content) + "&size=" + st.size);
}

const before = job => job.inputs && job.inputs.content
  ? "/api/picture?name=" + encodeURIComponent(job.inputs.content) + "&size=" + (job.params.size || 1024) : null;

function show(job) {
  shown = job;
  $("compare").style.setProperty("--x", "50%");
  stage.download(job);
  hist.select(job);
  setView("result");
  status(job.method === "edit" ? describe(job) : job.method + " · " + madeIn(job));
}

function describe(job) {
  const p = job.params, i = job.inputs;
  const at = p.mask === "none" ? "whole picture" : p.mask === "body part" ? p.part : p.mask === "object (words)"
    ? "“" + p.words + "”" : p.mask;
  const mat = p.swatch === "space" ? (i.materialA || {}).preset : nameOf(i.texture || "");
  return `${at} → ${mat} · ${p.quality} · seed ${p.seed}` + (job.took ? ` · ${job.took} s` : "");
}

// A render's settings back in the panel, to go on from it.
function reuse(job) {
  for (const k in job.params) if (k in st && k !== "_export") st[k] = job.params[k];
  inputs.content = job.inputs.content;
  if (job.inputs.texture) inputs.texture = job.inputs.texture;
  if (job.inputs.materialA && job.inputs.materialA.preset) inputs.preset = job.inputs.materialA.preset;
  $("preset").value = inputs.preset;
  $("prompt").value = st.prompt || "";
  $("size").value = st.size;
  $("seed").value = st.seed;
  changed(true);
  status("settings taken from that render");
}

// ---------- render ----------
function jobInputs() {
  const i = {content: inputs.content};
  if (st.swatch === "texture") i.texture = inputs.texture;
  else i.materialA = {preset: inputs.preset, terms: "", photo: null, upmat: null, texfile: "", point: "", push: 8,
                      against: "", concepts: {}};
  return i;
}

async function render(exporting) {
  const params = {...st, prompt: (st.prompt || "").trim()};
  if (exporting) params._export = true;
  const word = exporting ? "exporting" : "drawing";
  stage.busy(word + " · 0 s");
  working(true);
  try {
    const job = await runJob("edit", params, jobInputs(), s => stage.busy(word + " · " + s + " s"));
    hist.add(job);
    show(job);
    if (exporting) status("exported at full size · " + describe(job) + " · in the lab's exports folder");
  } catch (e) {
    status("the render failed: " + e.message, true);
  } finally {
    stage.busy(null);
    working(false);
  }
}

async function recipe() {
  const name = prompt("name this recipe (it appears in the lab's recipe list):",
                      "editor · " + (what() || "whole picture") + " → " +
                      (st.swatch === "space" ? inputs.preset : nameOf(inputs.texture)));
  if (name) await saveRecipe(name, {method: "edit", params: {...st}, inputs: jobInputs()});
}

// ---------- pictures in ----------
function choose(target, name) {
  inputs[target] = name;
  if (target === "texture") st.swatch = "texture";
  if (target === "content") { st.points = []; st.points_fade = []; }
  changed(target === "content");
  if (target === "content" && stage.view !== "mask") setView("picture");
  status((target === "content" ? "picture: " : "material: ") + nameOf(name));
}
