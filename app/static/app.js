// The lab client: panels generated from /api/menu; frames over a live socket; jobs only for slow work.
let MENU = null;
let method = null;
const paramState = {};
const inputs = {content: null, texture: null,
                materialA: {preset: "", terms: "", photo: null, upmat: null, texfile: "", point: "", push: 8,
                            against: "", concepts: {}},
                materialB: null};
let drawerTarget = null;
let apage = 0;

const $ = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const fmt = (v, p) => p.kind === "log" ? "1e" + (+v).toFixed(1) : (+v).toFixed(p.step && p.step < 1 ? 2 : 0);
const spec = () => MENU.methods[method];
const isSlow = () => spec().cost === "minutes";

// ---------- explanations ----------
// What a control does is said in ONE line above the render button, for whatever the pointer or the
// keyboard is on, so the panel itself stays short. "explanations" in the header also writes it
// under every control, for a first read through; the choice is remembered.
const explaining = (() => { try { return localStorage.getItem("lab.explain") === "1"; } catch (e) { return false; } })();
document.body.classList.toggle("explain", explaining);
$("explain").setAttribute("aria-pressed", explaining);
$("explain").onclick = () => {
  const now = document.body.classList.toggle("explain");
  $("explain").setAttribute("aria-pressed", now);
  try { localStorage.setItem("lab.explain", now ? "1" : "0"); } catch (e) { /* private window */ }
};
const hint = text => el("div", "khint", text);
const labelOf = p => p.label || p.key;
const IDLE = "point at a setting to see what it does · space renders · K keeps · H holds · " +
             "double-click a slider to reset it";
const say = text => { $("help").textContent = text || IDLE; };
const helpOf = new WeakMap();  // node -> () => its line, read when the pointer arrives
const helps = (node, text) => helpOf.set(node, text);
const docOf = (p, v) => labelOf(p) + " — " + (p.doc || "") +
  (p.choices && p.choices[String(v)] ? " · now: " + p.choices[String(v)] : "");
function sayAt(target) {
  for (let n = target; n && n.id !== "panel"; n = n.parentElement) {
    if (n.id === "foot") return;  // reading the line itself keeps it
    if (helpOf.has(n)) return say(helpOf.get(n)());
  }
  say();
}
$("panel").addEventListener("mouseover", e => sayAt(e.target));
$("panel").addEventListener("focusin", e => sayAt(e.target));
$("panel").addEventListener("mouseleave", () => say());
say();

// ---------- boot ----------
fetch("/api/menu").then(r => r.json()).then(m => {
  MENU = m;
  $("engineBadge").textContent = m.preset + " · " + m.size + "px · " + m.archive + " archive photographs";
  if (m.exports) $("jobsHead").textContent = "jobs · exports land in " + m.exports;
  buildRail();
  inputs.content = m.contents[0];
  inputs.texture = m.textures[0];
  if (m.presets.length) inputs.materialA.preset = m.presets[0];
  for (const h of m.history) addThumb(h);
  const first = Object.values(m.methods).find(x => x.family === "realtime");
  liveConnect();
  jobsConnect();
  pick(first ? first.key : "displace");
  // A render sent from one of the tabs (its history's double-click): its settings, here.
  const sent = new URLSearchParams(location.hash.slice(1)).get("job");
  const job = sent && m.history.find(h => h.id === sent);
  if (location.hash) history.replaceState(null, "", location.pathname);
  if (job && m.methods[job.method]) restore(job);
  loadRecipes();
  const srt = $("asort");
  for (const c of m.concepts) {
    srt.add(new Option(c + " ▾", c + " -"));
    srt.add(new Option(c + " ▴", c + " +"));
  }
});

// A family listed here is one tool with several modes: it takes one rail entry, and its modes are
// chosen in the panel. The rail entry names the mode in use and returns to it.
const TOOLS = {mapping: "texture mapping"};
const lastMode = (() => { try { return JSON.parse(localStorage.getItem("lab.modes") || "{}"); } catch (e) { return {}; } })();
const members = fam => Object.keys(MENU.methods).filter(k => MENU.methods[k].family === fam);

function buildRail() {
  const rail = $("rail");
  const fams = [["realtime", "GANs · realtime"], ["tokens", "tokens · RAE"], ["growth", "growth · the archive's NCAs"],
                ["diffusion", "the space"], ["words", "LoRA & words"],
                ["mapping", "texture mapping"], ["optimise", "optimisation"]];
  for (const [fam, title] of fams) {
    rail.appendChild(el("div", "fam", title));
    if (TOOLS[fam]) {
      const keys = members(fam);
      if (!keys.length) continue;
      const b = el("button");
      b.id = "rail-" + fam;
      b.dataset.fam = fam;
      b.title = "one tool, " + keys.length + " modes: choose the mode at the top of the right panel";
      b.onclick = () => pick(keys.includes(lastMode[fam]) ? lastMode[fam] : keys[0]);
      rail.appendChild(b);
      railLabel(fam);
      continue;
    }
    for (const [key, m] of Object.entries(MENU.methods)) {
      if (m.family !== fam) continue;
      const b = el("button", "", key + '<span class="cost">' + {live: "·", seconds: "~s", minutes: "~min"}[m.cost] + "</span>");
      b.id = "rail-" + key;
      b.title = m.doc;
      b.onclick = () => pick(key);
      rail.appendChild(b);
    }
  }
}

function railLabel(fam) {
  const keys = members(fam);
  $("rail-" + fam).innerHTML = (keys.includes(lastMode[fam]) ? lastMode[fam] : keys[0]) +
    '<span class="cost">' + keys.length + " modes</span>";
}

// The tool's modes, side by side in the panel: one click swaps the mode and keeps the inputs and
// every shared setting (size, mask), so the same picture can be read through each in turn.
function buildModes(m) {
  const box = $("modes");
  box.innerHTML = "";
  box.hidden = !TOOLS[m.family];
  if (box.hidden) return;
  box.appendChild(el("div", "lab", "mode"));
  const grid = el("div", "modes");
  for (const key of members(m.family)) {
    const b = el("button", "", key);
    b.title = MENU.methods[key].doc;
    b.setAttribute("aria-pressed", key === method);
    b.onclick = () => pick(key);
    b.onmouseenter = () => describe(key);
    b.onmouseleave = () => describe(method);
    grid.appendChild(b);
  }
  box.appendChild(grid);
}

// Settings marked `shared` (the working size, quality, aspect, LoRA) hold one value per family, so
// switching from blend to displace keeps the size you chose. They survive a reload too.
const shared = (() => { try { return JSON.parse(localStorage.getItem("lab.shared") || "{}"); } catch (e) { return {}; } })();
function shareKey(m, p) { return m.family + ":" + p.key; }
function remember(p, v) {
  if (!p.shared) return;
  shared[shareKey(spec(), p)] = v;
  try { localStorage.setItem("lab.shared", JSON.stringify(shared)); } catch (e) { /* private window */ }
}

function pick(key) {
  method = key;
  picking = null;
  $("viewer").classList.remove("picking");
  const m = spec();
  if (TOOLS[m.family]) {
    lastMode[m.family] = key;
    try { localStorage.setItem("lab.modes", JSON.stringify(lastMode)); } catch (e) { /* private window */ }
    railLabel(m.family);
  }
  document.querySelectorAll("#rail button").forEach(b =>
    b.setAttribute("aria-pressed", b.id === "rail-" + key || b.dataset.fam === m.family));
  buildModes(m);
  describe(key);
  if (!paramState[key]) {
    paramState[key] = {};
    for (const p of m.params) paramState[key][p.key] = p.default;
  }
  for (const p of m.params) {
    const v = shared[shareKey(m, p)];
    if (!p.shared || v === undefined) continue;
    if (p.options && !p.options.map(String).includes(String(v))) continue;  // not offered here
    paramState[key][p.key] = v;
  }
  buildInputs(m);
  buildParams(m);
  $("render").textContent = isSlow() ? "render (~min)" : "render";
  $("export").hidden = !m.export;
  changed("settle");
}

function describe(key) {
  const d = $("methodDoc");
  d.innerHTML = "";
  d.append(el("b", "", key), " — " + MENU.methods[key].doc);
}
$("methodDoc").onclick = () => $("methodDoc").classList.toggle("full");

// ---------- inputs ----------
// One compact row each: the picture or texture as a thumbnail and a list, with upload (and, for a
// texture, the archive) beside it; a material as its preset, everything else behind its "more".
const INPUT_NAMES = {content: "picture", texture: "texture", materialA: "material", materialB: "second material",
                     textureB: "second texture", textureC: "third texture"};

function buildInputs(m) {
  const box = $("inputs");
  box.innerHTML = "";
  for (const inp of m.inputs) {
    const name = INPUT_NAMES[inp.key] || inp.key;
    const block = el("div", "input");
    block.dataset.input = inp.key;
    const head = el("div", "inhead");
    // An input shown only when a setting asks for it is needed whenever it shows.
    head.appendChild(el("span", "lab", name + (inp.optional && !(inp.when || []).length ? " · optional" : "")));
    block.appendChild(head);
    if (inp.doc) {
      block.appendChild(el("div", "gdoc", inp.doc));
      helps(head, () => name + " — " + inp.doc);
    }
    if (inp.type === "content" || inp.type === "texture") {
      const row = el("div", "inprev");
      const img = el("img");
      const sel = document.createElement("select");
      const list = inp.type === "content" ? MENU.contents : MENU.textures;
      // An optional input that is always shown (compose's second and third materials) can be left out.
      const none = inp.optional && !(inp.when || []).length;
      if (none) sel.add(new Option("— none", ""));
      for (const f of list) sel.add(new Option(f, f));
      for (const u of MENU.uploads) sel.add(new Option("⬆ " + u, "u:" + u));
      if (inputs[inp.key] && inputs[inp.key].startsWith("i:"))
        sel.add(new Option("archive #" + inputs[inp.key].slice(2), inputs[inp.key]));
      sel.value = inputs[inp.key] && [...sel.options].some(o => o.value === inputs[inp.key])
        ? inputs[inp.key] : sel.options[0].value;
      inputs[inp.key] = sel.value;
      const sync = () => {
        img.style.visibility = inputs[inp.key] ? "" : "hidden";
        if (inputs[inp.key]) img.src = "/api/source?kind=" + inp.type + "&name=" + encodeURIComponent(inputs[inp.key]);
      };
      sel.onchange = () => { inputs[inp.key] = sel.value; sync(); changed("settle"); };
      sync();
      const up = el("button", "icon", "⬆");
      helps(up, () => "upload — use a picture of your own");
      const file = document.createElement("input");
      file.type = "file"; file.accept = "image/*"; file.hidden = true;
      up.onclick = () => file.click();
      file.onchange = async () => {
        if (!file.files[0]) return;
        const r = await fetch("/api/upload?name=" + encodeURIComponent(file.files[0].name),
                              {method: "POST", body: file.files[0]});
        const got = await r.json();
        if (!MENU.uploads.includes(got.name)) MENU.uploads.push(got.name);
        sel.add(new Option("⬆ " + got.name, "u:" + got.name));
        sel.value = "u:" + got.name; inputs[inp.key] = sel.value; sync(); changed("settle");
      };
      row.append(img, sel, up);
      if (inp.type === "texture") {
        const arch = el("button", "icon", "▦");
        helps(arch, () => "archive — any of the archive's photographs (needs the photo drive)");
        arch.onclick = () => openDrawer(inp.key);
        row.appendChild(arch);
      }
      block.append(row, file);
    } else if (inp.type === "material") {
      const which = inp.key === "materialA" ? "A" : "B";
      if (which === "B" && !inputs.materialB)
        inputs.materialB = {preset: MENU.presets[2] || "", terms: "", photo: null, upmat: null, texfile: "", point: ""};
      block.appendChild(materialEditor(which, head));
    }
    box.appendChild(block);
    if (inp === m.inputs[0]) for (const p of m.params.filter(q => q.lead)) box.appendChild(leadSwitch(p));
  }
}

// A choice that decides which input is used (`lead`) sits with the inputs as a switch, under the
// first one: the editor's swatch set to texture brings up the texture row, upload and archive.
function leadSwitch(p) {
  const st = paramState[method];
  const row = el("div", "knob lead");
  const seg = el("div", "wide seg");
  for (const o of p.options) {
    const b = el("button", "", String(o));
    b.setAttribute("aria-pressed", String(st[p.key]) === String(o));
    b.onclick = () => {
      st[p.key] = o;
      remember(p, o);
      seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
      say(docOf(p, o));
      changed("settle");
    };
    seg.appendChild(b);
  }
  row.append(el("span", "k", labelOf(p)), seg);
  helps(row, () => docOf(p, paramState[method][p.key]));
  return row;
}

function matSummary(s) {
  if (!s) return "—";
  if (s.point) return "the " + s.point;
  if (s.upmat) return "⬆ " + s.upmat;
  if (s.texfile) return "texture " + s.texfile;
  if (s.photo !== null && s.photo !== undefined && s.photo !== "") return "archive #" + s.photo;
  if (s.terms) return s.terms;
  return s.preset || "the centre";
}

function materialEditor(which, head) {
  const s = which === "A" ? inputs.materialA : inputs.materialB;
  const wrap = el("div", "matbox");
  // The heading says what the material is now, axes included, so the folded part hides nothing.
  const summary = el("span", "insum");
  head.appendChild(summary);
  const show = () => {
    const axes = Object.entries(s.concepts || {}).filter(([, v]) => +v)
      .map(([c, v]) => c + " " + (v > 0 ? "+" : "") + (+v).toFixed(2).replace(/\.?0+$/, ""));
    summary.textContent = [matSummary(s), ...axes, s.against ? "against " + s.against : ""].filter(Boolean).join(" · ");
  };
  const presetSel = document.createElement("select");
  presetSel.add(new Option("preset…", ""));
  for (const p of MENU.presets) presetSel.add(new Option(p, p));
  presetSel.value = s.preset || "";
  if (!MENU.captions) {
    // Presets and term mixes are built from the archive's captions; with the photo drive away
    // they would quietly fall back to the centre. Say so instead.
    presetSel.disabled = true;
    presetSel.options[0].text = "presets need the photo drive (G:)";
  }
  const terms = document.createElement("input");
  terms.type = "text"; terms.placeholder = "terms: encrusted:1, glassy:.5";
  terms.value = s.terms || "";
  const set = patch => {
    Object.assign(s, {preset: "", terms: "", photo: null, upmat: null, texfile: "", point: ""}, patch);
    show();
    changed("settle");
  };
  presetSel.onchange = () => set({preset: presetSel.value});
  // Any texture photograph from the texture folder, straight in as the material.
  const texSel = document.createElement("select");
  texSel.add(new Option("texture photograph…", ""));
  for (const t of MENU.textures) texSel.add(new Option(t, t));
  texSel.value = s.texfile || "";
  texSel.onchange = () => { if (texSel.value) set({texfile: texSel.value}); };
  terms.onchange = () => set({terms: terms.value.trim()});
  const chips = el("div", "chips");
  for (const [label, fn] of [["archive…", () => openDrawer(which)],
                             ["centre", () => set({point: "centre"})],
                             ["antipode", () => set({point: "antipode"})]]) {
    const b = el("button", "", label);
    b.onclick = fn;
    chips.appendChild(b);
  }
  for (const u of MENU.uploads.slice(-4)) {
    const b = el("button", "", "⬆ " + u.slice(0, 14));
    b.title = u;
    b.onclick = () => set({upmat: u});
    chips.appendChild(b);
  }
  helps(presetSel, () => "preset — a named material, made from the captions of your archive photographs");
  helps(texSel, () => "texture photograph — a photograph from the texture folder, taken as the material");
  helps(terms, () => "terms — your own mix of caption words, each with a weight, e.g. encrusted:1, glassy:.5");
  helps(chips, () => "archive: one of your photographs · centre: the archive's average · antipode: its opposite · ⬆: your uploads");
  // Everything but the preset sits behind "more"; it stays open once opened.
  const more = el("div", "more");
  const moreB = el("button", "morebtn");
  const key = "material " + which;
  const showMore = on => { more.hidden = !on; moreB.textContent = on ? "less" : "texture, terms, archive" +
    (which === "A" ? ", push, axes" : ""); moreB.setAttribute("aria-expanded", on); };
  showMore(!!moreOpen[key]);
  moreB.onclick = () => { moreOpen[key] = more.hidden; showMore(more.hidden); keepOpen("lab.more", moreOpen); };
  more.append(texSel, terms, chips);
  wrap.append(presetSel, moreB, more);
  if (which === "A") {
    const push = knobRow({key: "push", kind: "float", lo: 0, hi: 12, step: 0.5, default: 8,
                          doc: "exaggeration off the centre"}, v => s.push = v, s.push);
    helps(push, () => "push — how far the material is pushed past the archive's average: more is more extreme");
    const against = document.createElement("input");
    against.type = "text"; against.placeholder = "against: still water + glassy";
    against.value = s.against || "";
    against.onchange = () => { s.against = against.value.trim(); show(); changed("settle"); };
    helps(against, () => "against — a material to steer away from (terms or a preset)");
    const clab = el("div", "lab", "concept axes");
    helps(clab, () => "concept axes — push the material along your named qualities; minus is the opposite end");
    more.append(push, against, clab);
    for (const c of MENU.concepts) {
      const row = knobRow({key: c, kind: "float", lo: -8, hi: 8, step: 0.25, default: 0},
                          v => { s.concepts[c] = v; show(); }, s.concepts[c] || 0);
      helps(row, () => c + " — push the material toward " + c + "; minus is the opposite end");
      more.appendChild(row);
    }
  }
  show();
  return wrap;
}

// ---------- params ----------
// A slider sends a rough frame on every move and a sharp one when released; a double-click puts it
// back to its default.
function knobRow(p, onset, value = p.default) {
  const row = el("div", "knob");
  const out = el("output", "", fmt(value, p));
  const input = document.createElement("input");
  input.type = "range";
  input.min = p.lo; input.max = p.hi; input.step = p.step || 1; input.value = value;
  input.oninput = () => { out.textContent = fmt(input.value, p); onset(+input.value); changed("drag"); };
  input.onchange = () => changed("settle");
  input.ondblclick = () => {  // double-click a slider: back to its default
    input.value = p.default; out.textContent = fmt(p.default, p); onset(+p.default); changed("settle");
  };
  row.append(el("span", "k", labelOf(p)), input, out);
  if (p.doc) row.title = p.doc;
  row._set = v => { input.value = v; out.textContent = fmt(v, p); };
  return row;
}

const knobs = {};  // param key -> row, so the XY pad can move the sliders it drives

// The settings, one section per section of the registry, laid out by its LAYOUT: the mask right
// under the inputs (its extra layers, gradient and edges nested in it), then the method's own
// settings, then the optional add-ons, the output last beside the render button. A section folds
// on its heading, and a folded heading says what in it is set; the seldom-touched knobs sit behind
// the section's "more", which says how many of them are set. A setting shows only while it applies
// (its `when`), and a section with nothing that applies hides whole.
const remembered = key => { try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch (e) { return {}; } };
const keepOpen = (key, obj) => { try { localStorage.setItem(key, JSON.stringify(obj)); } catch (e) { /* private window */ } };
const opened = remembered("lab.sections");  // section -> true open / false folded, once chosen
const moreOpen = remembered("lab.more");    // section -> true while its "more" is open
const controls = [];  // [declaration, node] for every setting shown, so `when` can be re-read
const sections = new Map();  // section -> its parts
const PLACES = ["where", "own", "addon", "output"];
const layoutOf = g => (MENU.layout || {})[g] || {};
const groupOf = p => p.group || method + " settings";
const isSet = (p, v) => p.kind === "points" ? (v || []).length > 0 : String(v) !== String(p.default);

function short(p, v) {
  if (p.kind === "bool") return labelOf(p);
  if (p.kind === "points") return labelOf(p) + " ×" + v.length;
  if (p.kind === "seed") return "seed " + v;
  if (p.kind === "text") return "“" + (String(v).length > 22 ? String(v).slice(0, 21) + "…" : v) + "”";
  if (p.kind === "choice") return isNaN(+v) ? String(v) : labelOf(p) + " " + v;
  return labelOf(p) + " " + fmt(v, p);
}

function holds(c, st) {
  if (c[0] === "any") return c[1].some(x => holds(x, st));
  const v = st[c[0]];
  switch (c[1]) {
    case "==": return String(v) === String(c[2]);
    case "!=": return String(v) !== String(c[2]);
    case ">": return +v > +c[2];
    case "in": return c[2].map(String).includes(String(v));
    case "set": return v !== undefined && v !== null && String(v).trim() !== "";
  }
  return true;
}
const applies = (item, st) => (item.when || []).every(c => holds(c, st));

function relevance() {
  if (!method) return;
  const st = paramState[method] || {};
  for (const [p, node] of controls) node.hidden = !applies(p, st);
  for (const [name, s] of sections) {
    const mine = controls.filter(([p]) => groupOf(p) === name && applies(p, st)).map(([p]) => p);
    const extra = mine.filter(p => p.more);
    const set = extra.filter(p => isSet(p, st[p.key])).length;
    s.card.hidden = !mine.length;
    s.moreB.hidden = !extra.length;
    s.moreB.textContent = s.more.hidden ? `more (${extra.length})` + (set ? ` · ${set} set` : "") : "less";
    s.sum.textContent = mine.filter(p => isSet(p, st[p.key])).map(p => short(p, st[p.key])).join(" · ");
  }
  for (const block of document.querySelectorAll("#inputs [data-input]")) {
    const inp = spec().inputs.find(i => i.key === block.dataset.input);
    block.hidden = !!inp && !applies(inp, st);
  }
  marks();
}

// ---------- picking on the picture ----------
// A mask layer whose source is a click (an object, a colour, a spot) arms the picture: each click
// places a point in 0–1 of the frame — for an object, points add up and a shift-click or a
// right-click marks a part to leave out; for a colour or a spot, a click moves the one point.
// The points show on the picture while it is armed.
let picking = null;          // the points setting the picture's clicks go to, or null
const pointViews = [];       // the clicks controls' counters, to refresh after a click
const MULTI_CLICK = new Set(["object (click)", "material (click)"]);  // sources that gather clicks

function arm(key) {
  picking = key || null;
  $("viewer").classList.toggle("picking", !!picking);
  document.querySelectorAll("[data-pick]").forEach(b => b.setAttribute("aria-pressed", b.dataset.pick === picking));
  marks();
  if (picking) $("note").textContent = pickHint();
}

function pickHint() {
  const s = picking.slice("points".length);
  return MULTI_CLICK.has((paramState[method] || {})["mask" + s])
    ? "click the picture to select · shift-click or right-click leaves a part out"
    : "click the picture to place it";
}

function pickAt(e, include) {
  if (!picking || !method) return;
  const r = $("mainCanvas").getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  const st = paramState[method];
  const s = picking.slice("points".length);
  const pt = [+x.toFixed(4), +y.toFixed(4), include ? 1 : 0];
  st[picking] = MULTI_CLICK.has(st["mask" + s]) ? [...(st[picking] || []), pt] : [pt];
  remember(spec().params.find(q => q.key === picking), st[picking]);
  pointViews.forEach(f => f());
  marks();
  changed("settle");
}
$("mainCanvas").addEventListener("click", e => pickAt(e, !e.shiftKey));
$("mainCanvas").addEventListener("contextmenu", e => { if (picking) { e.preventDefault(); pickAt(e, false); } });

function marks() {
  const box = $("marks");
  box.innerHTML = "";
  if (!picking || !method) return;
  const st = paramState[method] || {};
  for (const p of spec().params.filter(q => q.kind === "points" && applies(q, st))) {
    const layer = p.key === "points" ? "1" : p.key === "points_fade" ? "◐" : p.key.slice(-1);
    for (const [x, y, inc] of st[p.key] || []) {
      const d = el("div", "mark" + (inc ? "" : " out"), layer);
      d.style.left = (x * 100) + "%";
      d.style.top = (y * 100) + "%";
      box.appendChild(d);
    }
  }
}

function buildParams(m) {
  const box = $("params");
  box.innerHTML = "";
  for (const k in knobs) delete knobs[k];
  controls.length = 0;
  pointViews.length = 0;
  sections.clear();
  const st = paramState[method];
  const listed = m.params.filter(p => !p.lead);  // a lead choice sits with the inputs instead
  const groups = [...new Set(listed.map(groupOf))];
  const place = g => PLACES.indexOf(layoutOf(g).place || "own");
  groups.sort((a, b) => place(a) - place(b));  // stable: the registry's own order within a place
  const lead = groups.find(g => layoutOf(g).place === "where");
  for (const g of groups) {
    const nested = g !== lead && layoutOf(g).place === "where";
    section(g, nested ? sections.get(lead).subs : box, nested);
  }
  for (const p of listed) {
    const s = sections.get(groupOf(p));
    if (m.family === "realtime" && p.key === "axis 0") s.body.appendChild(xyPad(st));
    const ctl = control(p, st);
    (p.more ? s.more : s.body).appendChild(ctl);
    controls.push([p, ctl]);
  }
  relevance();
}

function section(name, parent, nested) {
  const card = el("div", "section" + (nested ? " sub" : ""));
  const head = el("button", "sechead");
  const sum = el("span", "sum");
  head.append(el("span", "t", name), sum);
  const inner = el("div", "secbody");
  const body = el("div", "main");
  const moreB = el("button", "morebtn");
  const more = el("div", "more");
  const subs = el("div", "subs");
  const doc = MENU.groups && MENU.groups[name];
  if (doc) {
    inner.appendChild(el("div", "gdoc", doc));
    helps(head, () => name + " — " + doc);
  }
  inner.append(body, moreB, more, subs);
  card.append(head, inner);
  parent.appendChild(card);
  const fold = shut => { card.classList.toggle("folded", shut); head.setAttribute("aria-expanded", !shut); };
  fold(name in opened ? !opened[name] : !!layoutOf(name).folded);
  head.onclick = () => {
    const shut = !card.classList.contains("folded");
    fold(shut);
    opened[name] = !shut;
    keepOpen("lab.sections", opened);
  };
  more.hidden = !moreOpen[name];
  moreB.onclick = () => {
    more.hidden = !more.hidden;
    moreOpen[name] = !more.hidden;
    keepOpen("lab.more", moreOpen);
    relevance();
  };
  sections.set(name, {card, sum, body, more, moreB, subs});
}

// One setting: its label and control on one row. What it does is said in the help line (and,
// with explanations on, under it).
function control(p, st) {
  const ctl = el("div", "ctl");
  helps(ctl, () => docOf(p, paramState[method][p.key]));
  if (p.kind === "choice") {
    const row = el("div", "knob");
    const sel = document.createElement("select");
    const add = (parent, o) => parent.appendChild(new Option(String(o), String(o)));
    const grouped = new Set((p.optgroups || []).flatMap(g => g[1]).map(String));
    for (const o of p.options) if (!grouped.has(String(o))) add(sel, o);  // "none" and the like, first
    for (const [title, opts] of p.optgroups || []) {
      const g = document.createElement("optgroup");
      g.label = title;
      for (const o of opts) add(g, o);
      sel.appendChild(g);
    }
    sel.value = String(st[p.key]);
    sel.className = "wide";
    sel.onchange = () => {
      st[p.key] = isNaN(+sel.value) ? sel.value : +sel.value;
      remember(p, st[p.key]);
      say(docOf(p, st[p.key]));
      // A mask layer (or the gradient) turned to a click source arms the picture for its clicks
      // straight away.
      const layer = /^mask(_\d)?$/.exec(p.key);
      const ptsKey = layer ? "points" + (layer[1] || "") : p.key === "fade" ? "points_fade" : null;
      const pts = ptsKey && spec().params.find(q => q.key === ptsKey);
      if (pts) arm(applies(pts, st) ? pts.key : (picking === pts.key ? null : picking));
      changed("settle");
    };
    row.append(el("span", "k", labelOf(p)), sel);
    ctl.appendChild(row);
    if (Object.keys(p.choices || {}).length) {  // with explanations on: the chosen option, in full
      const now = hint("");
      const tell = () => { now.textContent = (p.doc ? p.doc + " · now: " : "") + (p.choices[sel.value] || sel.value); };
      tell();
      sel.addEventListener("change", tell);
      ctl.appendChild(now);
    } else if (p.doc) ctl.appendChild(hint(p.doc));
    return ctl;
  }
  if (p.kind === "bool") {
    const row = el("div", "knob");
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = !!st[p.key];
    box.onchange = () => { st[p.key] = box.checked; remember(p, st[p.key]); changed("settle"); };
    row.append(el("span", "k", labelOf(p)), box, el("span"));
    ctl.appendChild(row);
  } else if (p.kind === "seed") {
    const row = el("div", "knob");
    const num = document.createElement("input");
    num.type = "number"; num.value = st[p.key];
    num.onchange = () => { st[p.key] = +num.value; changed("settle"); };
    const dice = el("button", "", "⚄");
    dice.title = "random";
    dice.onclick = () => { num.value = Math.floor(Math.random() * 99999); st[p.key] = +num.value; changed("settle"); };
    row.append(el("span", "k", labelOf(p)), num, dice);
    ctl.appendChild(row);
  } else if (p.kind === "points") {
    const row = el("div", "knob");
    const tools = el("div", "wide pointtools");
    const count = el("span", "pcount");
    const pickB = el("button", "", "click the picture");
    const clearB = el("button", "", "clear");
    pickB.dataset.pick = p.key;
    pickB.onclick = () => arm(picking === p.key ? null : p.key);
    clearB.onclick = () => { st[p.key] = []; remember(p, []); pointViews.forEach(f => f()); marks(); changed("settle"); };
    const tell = () => {
      const n = (paramState[method][p.key] || []).length;
      count.textContent = n ? n + (n === 1 ? " point" : " points") : "no points yet";
    };
    pointViews.push(tell);
    tell();
    tools.append(count, pickB, clearB);
    row.append(el("span", "k", labelOf(p)), tools);
    ctl.appendChild(row);
  } else if (p.kind === "text") {
    const row = el("div", "knob");
    const input = document.createElement("input");
    input.type = "text"; input.value = st[p.key] || ""; input.placeholder = p.doc;
    input.className = "wide";
    input.onchange = () => { st[p.key] = input.value.trim(); remember(p, st[p.key]); changed("settle"); };
    row.append(el("span", "k", labelOf(p)), input);
    ctl.appendChild(row);
  } else {
    const row = knobRow(p, v => { st[p.key] = v; remember(p, v); }, st[p.key]);
    knobs[p.key] = row;
    ctl.appendChild(row);
  }
  if (p.doc) ctl.appendChild(hint(p.doc));
  return ctl;
}

// One finger on the two strongest directions of a GAN's latent.
function xyPad(st) {
  const wrap = el("div");
  const lab = el("div", "lab", "axis 0 →  ·  ↓ axis 1  — drag");
  const pad = el("div", "xypad");
  const dot = el("div", "xydot");
  pad.appendChild(dot);
  const SPAN = 3;
  const place = () => {
    dot.style.left = ((st["axis 0"] / SPAN + 1) / 2 * 100) + "%";
    dot.style.top = ((st["axis 1"] / SPAN + 1) / 2 * 100) + "%";
  };
  const at = (e, phase) => {
    const r = pad.getBoundingClientRect();
    const x = Math.max(-SPAN, Math.min(SPAN, ((e.clientX - r.left) / r.width * 2 - 1) * SPAN));
    const y = Math.max(-SPAN, Math.min(SPAN, ((e.clientY - r.top) / r.height * 2 - 1) * SPAN));
    st["axis 0"] = +x.toFixed(2); st["axis 1"] = +y.toFixed(2);
    knobs["axis 0"] && knobs["axis 0"]._set(st["axis 0"]);
    knobs["axis 1"] && knobs["axis 1"]._set(st["axis 1"]);
    place();
    changed(phase);
  };
  pad.addEventListener("pointerdown", e => { pad.setPointerCapture(e.pointerId); at(e, "drag"); });
  pad.addEventListener("pointermove", e => { if (e.buttons) at(e, "drag"); });
  pad.addEventListener("pointerup", e => at(e, "settle"));
  wrap.append(lab, pad);
  setTimeout(place, 0);
  return wrap;
}

// ---------- the stage: one square, every frame scaled to it ----------
// Display size belongs to the viewer, never to the frame. A 256px draft, a 512px final and a
// 1024px GAN frame all fill the same square, so nothing jumps between drag and release or
// between methods. The canvas holds the last frame until the next is decoded: no flicker.
const stage = {main: {canvas: $("mainCanvas"), host: $("viewer"), bmp: null},
               hold: {canvas: $("holdCanvas"), host: $("holdPane"), bmp: null}};
let showToken = 0, shownToken = 0;

// The stage takes the frame's own shape — the largest rectangle of that ratio the pane holds. A
// drag frame and its release share a ratio, so the stage holds still between them; only a new
// picture (a new shape) moves it.
function fit(s) {
  const r = s.host.getBoundingClientRect();
  if (!r.width || !r.height) return;
  const ratio = s.bmp ? s.bmp.width / s.bmp.height : 1;
  const room = {w: r.width - 8, h: r.height - 8};
  let w = room.w, h = w / ratio;
  if (h > room.h) { h = room.h; w = h * ratio; }
  w = Math.max(64, Math.floor(w)); h = Math.max(64, Math.floor(h));
  s.canvas.style.width = w + "px";
  s.canvas.style.height = h + "px";
  if (s === stage.main) { $("marks").style.width = w + "px"; $("marks").style.height = h + "px"; }
  const dpr = window.devicePixelRatio || 1;
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (s.canvas.width !== pw || s.canvas.height !== ph) { s.canvas.width = pw; s.canvas.height = ph; }
}

function paint(s) {
  fit(s);
  const ctx = s.canvas.getContext("2d");
  ctx.clearRect(0, 0, s.canvas.width, s.canvas.height);
  if (!s.bmp) return;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const k = Math.min(s.canvas.width / s.bmp.width, s.canvas.height / s.bmp.height);
  const w = s.bmp.width * k, h = s.bmp.height * k;
  ctx.drawImage(s.bmp, (s.canvas.width - w) / 2, (s.canvas.height - h) / 2, w, h);
}

// Accepts a Blob or a URL. Decodes off the main thread; a frame that finishes decoding after a
// newer one has been painted is dropped rather than shown out of order.
async function show(src) {
  const token = ++showToken;
  const blob = src instanceof Blob ? src : await (await fetch(src)).blob();
  const bmp = await createImageBitmap(blob);
  if (token < shownToken) { bmp.close(); return; }
  shownToken = token;
  if (stage.main.bmp) stage.main.bmp.close();
  stage.main.bmp = bmp;
  paint(stage.main);
}

const refit = new ResizeObserver(() => { paint(stage.main); paint(stage.hold); });
refit.observe($("viewer"));
refit.observe($("holdPane"));

// ---------- the live socket ----------
let live = null, inflight = false, pending = null, seq = 0;
let frames = 0, fpsT = performance.now(), fps = 0;

function liveConnect() {
  live = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/live");
  live.binaryType = "arraybuffer";
  live.onopen = () => { inflight = false; if (pending) flush(); else if (method && !isSlow()) changed("settle"); };
  live.onmessage = e => {
    if (typeof e.data === "string") {
      const msg = JSON.parse(e.data);
      if (msg.type === "kept") { addThumb(msg.job, true); flash("kept"); }
      if (msg.type === "error") { $("note").textContent = "error · " + msg.note; $("note").classList.add("err"); }
      inflight = msg.type === "kept" ? inflight : false;
      if (msg.type === "error" && pending) flush();
      return;
    }
    const view = new DataView(e.data);
    const n = view.getUint32(0, true);
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(e.data, 4, n)));
    show(new Blob([new Uint8Array(e.data, 4 + n)], {type: "image/jpeg"}));
    frames += 1;
    const now = performance.now();
    if (now - fpsT > 700) { fps = frames * 1000 / (now - fpsT); frames = 0; fpsT = now; }
    $("note").classList.remove("err");
    $("note").textContent = (picking ? pickHint() + " · " : "") + `${meta.method} · ${meta.w}×${meta.h} · ${meta.ms} ms` +
      (meta.phase === "drag" ? " · draft" : "") + (fps ? ` · ${fps.toFixed(1)} fps` : "");
    $("viewer").classList.toggle("busy", meta.phase === "drag");
    inflight = false;
    if (pending) flush();
  };
  live.onclose = () => { inflight = false; setTimeout(liveConnect, 1200); };
}

function request(phase) {
  return {seq: ++seq, phase, method, params: {...paramState[method]}, inputs: pickInputs(spec())};
}

function flush() {
  if (!live || live.readyState !== 1 || inflight || !pending) return;
  inflight = true;
  live.send(JSON.stringify(pending));
  pending = null;
}

function changed(phase) {
  if (!MENU || !method) return;
  relevance();
  if (isSlow() || !$("auto").checked) return;
  pending = request(phase);
  flush();
}

$("render").onclick = () => {
  if (isSlow()) return submitJob();
  pending = request("settle");
  flush();
};
$("auto").onchange = () => changed("settle");
$("keep").onclick = keep;
function keep() {
  if (isSlow()) return;
  if (live && live.readyState === 1) live.send(JSON.stringify({type: "keep"}));
}
function flash(word) {
  const n = $("note");
  const was = n.textContent;
  n.textContent = word + " ✓";
  setTimeout(() => { if (n.textContent === word + " ✓") n.textContent = was; }, 900);
}

function pickInputs(m) {
  const out = {};
  for (const inp of m.inputs) out[inp.key] = inputs[inp.key];
  return out;
}

// ---------- slow methods and exports: tracked until they land ----------
// Every job this page starts is tagged by the page itself (cid) before it is sent, so the very
// first notices — which can arrive before the reply naming the job — are recognised. Each one gets
// a row in the jobs list; unfinished rows are re-checked every few seconds and on every reconnect,
// so a result is never lost to a dropped socket, and starting a second export never orphans the
// first.
const mine = new Map();          // cid -> {cid, id, method, exporting, status, started, job, row}
let focusCid = null;             // the job whose previews and result take over the viewer

$("export").onclick = () => submitJob(true);

async function submitJob(exporting) {
  const cid = Math.random().toString(36).slice(2, 12);
  const params = exporting ? {...paramState[method], _export: true} : {...paramState[method]};
  const entry = {cid, id: null, method, exporting, status: "queued", started: Date.now(), job: null, row: null};
  mine.set(cid, entry);
  focusCid = cid;
  drawRow(entry);
  $("viewer").classList.add("busy");
  $("note").textContent = method + (exporting ? " · exporting at the picture's own size…" : " · queued");
  try {
    const r = await fetch("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"},
                                        body: JSON.stringify({method, params, inputs: pickInputs(spec()), cid})});
    const job = await r.json();
    entry.id = job.id;
    if (entry.status === "queued") update(entry, job);
  } catch (e) {
    entry.status = "error"; entry.note = "the server did not take the job"; drawRow(entry);
  }
}

function update(entry, job) {
  const was = entry.status;
  entry.id = job.id || entry.id;
  entry.status = job.status;
  entry.job = job;
  const focused = entry.cid === focusCid;
  if (focused && job.status === "running" && job.previews && job.previews.length)
    show("/api/out/" + job.previews[job.previews.length - 1]);
  if (focused && job.status !== "done") {
    $("note").textContent = job.method + " · " + (entry.exporting ? "exporting · " : "") + job.status
      + (job.note ? " · " + job.note : "");
    $("note").classList.toggle("err", job.status === "error");
  }
  if (job.status === "done" && was !== "done") {
    addThumb(job, focused);
    if (focused) {
      show(job.url || "/api/out/" + job.result);
      $("viewer").classList.remove("busy");
      $("note").textContent = `${job.method} · ${entry.exporting ? "exported" : "done"} · ${job.took}s`
        + (job.size ? ` · ${job.size[0]}×${job.size[1]}` : "");
    }
  }
  if (focused && (job.status === "error" || job.status === "cancelled")) $("viewer").classList.remove("busy");
  drawRow(entry);
  lightUp();
}

function drawRow(entry) {
  const box = $("jobsList");
  if (!entry.row) {
    entry.row = el("div", "jobrow");
    box.prepend(entry.row);
    $("jobsBox").hidden = false;
  }
  const job = entry.job || {};
  const secs = Math.round(((job.took ? job.took * 1000 : Date.now() - entry.started)) / 1000);
  const what = entry.exporting ? "export" : "render";
  entry.row.innerHTML = "";
  const label = el("span", "jl", `${entry.method} · ${what}`);
  const state = el("span", "js");
  if (entry.status === "done") {
    state.textContent = (job.size ? `${job.size[0]}×${job.size[1]} · ` : "") + `${secs}s`;
    entry.row.append(label, state);
    if (entry.exporting) {
      const a = el("a", "", "↓ save");
      a.href = job.url || "/api/out/" + job.result;
      a.download = job.result || "";
      a.title = job.path || "";
      entry.row.appendChild(a);
    }
    const view = el("button", "", "view");
    view.onclick = () => { focusCid = entry.cid; show(job.url || "/api/out/" + job.result); };
    entry.row.appendChild(view);
  } else if (entry.status === "error" || entry.status === "cancelled") {
    state.textContent = entry.status + (job.note || entry.note ? " · " + (job.note || entry.note) : "");
    state.classList.add("err");
    entry.row.append(label, state);
  } else {
    state.textContent = `${entry.status}… ${secs}s`;
    entry.row.append(label, state);
  }
  if (job.path) entry.row.title = "saved as " + job.path;
}

function lightUp() {
  const busy = [...mine.values()].filter(e => e.status === "queued" || e.status === "running").length;
  $("jobsLight").textContent = busy ? `${busy} working` : "·";
  $("jobsLight").classList.toggle("busy", busy > 0);
}

async function recheck() {
  for (const entry of mine.values()) {
    if (!entry.id || ["done", "error", "cancelled"].includes(entry.status)) {
      if (entry.status === "queued" || entry.status === "running") drawRow(entry);
      continue;
    }
    try {
      const job = await (await fetch("/api/jobs/" + entry.id)).json();
      if (job.status === "unknown") { entry.status = "error"; entry.note = "lost when the server restarted"; drawRow(entry); }
      else update(entry, job);
    } catch (e) { /* the server is away; the next pass tries again */ }
  }
  lightUp();
}
setInterval(recheck, 3000);

function jobsConnect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/ws");
  ws.onopen = recheck;  // whatever happened while the socket was down
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type !== "job") return;
    const entry = msg.job.cid && mine.get(msg.job.cid);
    if (entry) update(entry, msg.job);
  };
  ws.onclose = () => setTimeout(jobsConnect, 1500);
}

// ---------- history: only what was kept or rendered on purpose ----------
function addThumb(job, select) {
  if (!job.thumb) return;
  const img = document.createElement("img");
  img.src = "/api/out/" + job.thumb;
  img.title = job.method + " — click to view, double-click to reopen its settings";
  img.onclick = () => {
    document.querySelectorAll("#strip img").forEach(i => i.classList.remove("sel"));
    img.classList.add("sel");
    show(job.url || "/api/out/" + job.result);
    $("note").textContent = job.method + " · from history";
  };
  img.ondblclick = () => restore(job);
  $("strip").prepend(img);
  if (select) {
    document.querySelectorAll("#strip img").forEach(i => i.classList.remove("sel"));
    img.classList.add("sel");
  }
  while ($("strip").children.length > 120) $("strip").lastChild.remove();
}

function restore(job) {
  paramState[job.method] = {...job.params};
  // Its shared settings (the mask, the quality…) become the remembered ones too, or pick() would
  // lay the last-used ones back over them.
  const m = MENU.methods[job.method];
  for (const p of m.params) if (p.shared && p.key in job.params) shared[shareKey(m, p)] = job.params[p.key];
  try { localStorage.setItem("lab.shared", JSON.stringify(shared)); } catch (e) { /* private window */ }
  if (job.inputs.content) inputs.content = job.inputs.content;
  if (job.inputs.texture) inputs.texture = job.inputs.texture;
  for (const k of ["textureB", "textureC"]) if (k in job.inputs) inputs[k] = job.inputs[k];
  if (job.inputs.materialA) inputs.materialA = JSON.parse(JSON.stringify(job.inputs.materialA));
  if (job.inputs.materialB) inputs.materialB = JSON.parse(JSON.stringify(job.inputs.materialB));
  pick(job.method);
}

// ---------- hold / compare ----------
function hold() {
  if (!stage.main.bmp) return;
  $("holdPane").hidden = false;
  createImageBitmap(stage.main.bmp).then(copy => {
    if (stage.hold.bmp) stage.hold.bmp.close();
    stage.hold.bmp = copy;
    paint(stage.hold);
    paint(stage.main);  // the live pane just lost half its width
  });
}
$("hold").onclick = hold;
$("unhold").onclick = () => { $("holdPane").hidden = true; paint(stage.main); };

// ---------- keys: K keeps, H holds, space renders ----------
addEventListener("keydown", e => {
  if (e.target.matches("input[type=text], input[type=number], select")) return;
  if (e.key === "k" || e.key === "K") keep();
  if (e.key === "h" || e.key === "H") hold();
  if (e.key === " ") { e.preventDefault(); $("render").click(); }
});

// ---------- recipes ----------
async function loadRecipes() {
  const book = await (await fetch("/api/recipes")).json();
  const sel = $("recipeSel");
  sel.innerHTML = "";
  for (const name of Object.keys(book)) sel.add(new Option(name, name));
  sel.dataset.book = JSON.stringify(book);
}
$("recipeSave").onclick = async () => {
  const name = prompt("name this recipe:");
  if (!name) return;
  await fetch("/api/recipes", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name, recipe: {method, params: paramState[method], inputs: pickInputs(spec())}})});
  loadRecipes();
};
$("recipeLoad").onclick = () => {
  const book = JSON.parse($("recipeSel").dataset.book || "{}");
  const r = book[$("recipeSel").value];
  if (r) restore(r);
};

// ---------- archive drawer ----------
function openDrawer(target) {
  drawerTarget = target;
  $("drawer").hidden = false;
  loadArchive();
}
$("aclose").onclick = () => $("drawer").hidden = true;
$("drawer").onclick = e => { if (e.target === $("drawer")) $("drawer").hidden = true; };
let aT = null;
$("aq").oninput = () => { clearTimeout(aT); aT = setTimeout(() => { apage = 0; loadArchive(); }, 300); };
$("asort").onchange = () => { apage = 0; loadArchive(); };
$("aprev").onclick = () => { apage = Math.max(0, apage - 1); loadArchive(); };
$("anext").onclick = () => { apage += 1; loadArchive(); };

async function loadArchive() {
  const q = new URLSearchParams({term: $("aq").value, sort: $("asort").value, page: apage});
  const a = await (await fetch("/api/archive?" + q)).json();
  const pages = Math.max(1, Math.ceil(a.total / a.per));
  if (apage >= pages) { apage = pages - 1; return loadArchive(); }
  $("acount").textContent = a.total + " photographs";
  $("apage").textContent = (a.page + 1) + "/" + pages;
  const grid = $("agrid");
  grid.innerHTML = "";
  for (const row of a.rows) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = "/api/thumb?i=" + row.i;
    img.title = row.cap || row.name;
    img.onclick = () => {
      if (drawerTarget.startsWith("texture")) {
        inputs[drawerTarget] = "i:" + row.i;
      } else {
        const s = drawerTarget === "A" ? inputs.materialA : inputs.materialB;
        Object.assign(s, {preset: "", terms: "", upmat: null, texfile: "", point: "", photo: row.i});
      }
      buildInputs(spec());
      $("drawer").hidden = true;
      changed("settle");
    };
    grid.appendChild(img);
  }
}
