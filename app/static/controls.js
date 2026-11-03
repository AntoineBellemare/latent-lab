// The tabs' controls, made from a method's own declaration (registry.P): a slider, a switch, chips;
// and the "where" card, which decides where the material goes and is the same in every tab.
import {el} from "./studio.js";

// `when`: the clauses a setting needs to hold before it applies (registry.P).
export function holds(c, st) {
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
export const applies = (p, st) => (p.when || []).every(c => holds(c, st));
export const param = (spec, key) => spec.params.find(p => p.key === key);

const fmt = (p, v) => p.kind === "angle" ? Math.round(v * 180 / Math.PI) + "°" : p.kind === "int" ? String(Math.round(v))
  : (+v).toFixed(Math.abs(p.step || 0.01) >= 0.1 ? 1 : 2) + (p.unit === "px" ? " px" : "");

// A slider: dragging calls `onInput` (a live tab draws a quick frame), letting go calls `onChange`;
// a double-click puts the default back.
export function slider(p, st, {label, doc, onInput, onChange} = {}) {
  const row = el("div", "slide");
  row.dataset.key = p.key;
  const name = el("span", "k", label || p.label);
  name.title = doc || p.doc || "";
  const input = el("input");
  input.type = "range"; input.min = p.lo; input.max = p.hi; input.step = p.step || 0.01; input.value = st[p.key];
  const out = el("output", "", fmt(p, st[p.key]));
  const set = v => { st[p.key] = p.kind === "int" ? Math.round(v) : v; out.textContent = fmt(p, v); };
  input.oninput = () => { set(+input.value); if (onInput) onInput(); };
  input.onchange = () => { if (onChange) onChange(); };
  input.ondblclick = () => { input.value = p.default; set(p.default); if (onChange) onChange(); };
  row.append(name, input, out);
  row._set = v => { input.value = v; out.textContent = fmt(p, v); };
  return row;
}

export function chips(options, value, onPick, small) {
  const box = el("div", "chips" + (small ? " small" : ""));
  for (const [v, label, title] of options) {
    const b = el("button", "", label);
    if (title) b.title = title;
    b.setAttribute("aria-pressed", String(v) === String(value));
    b.onclick = () => onPick(v);
    box.appendChild(b);
  }
  return box;
}

export function tick(label, checked, onToggle, title) {
  const t = el("label", "tick");
  const box = el("input");
  box.type = "checkbox"; box.checked = !!checked;
  box.onchange = () => onToggle(box.checked);
  if (title) t.title = title;
  t.append(box, " " + label);
  return t;
}

// Any setting, drawn by its kind. `changed(phase)`: "drag" mid-gesture, "settle" when let go.
export function control(p, st, changed) {
  if (p.kind === "choice") {
    const opts = p.options.map(o => [o, String(o), p.choices[o] || ""]);
    const wrap = el("div", "ctl");
    wrap.appendChild(el("span", "k", p.label));
    wrap.lastChild.title = p.doc || "";
    if (opts.length <= 6) {
      wrap.appendChild(chips(opts, st[p.key], v => { st[p.key] = v; changed("settle"); }, true));
    } else {
      const sel = el("select");
      for (const o of p.options) sel.add(new Option(String(o), String(o)));
      sel.value = String(st[p.key]);
      sel.onchange = () => { st[p.key] = isNaN(+sel.value) ? sel.value : +sel.value; changed("settle"); };
      wrap.appendChild(sel);
    }
    return wrap;
  }
  if (p.kind === "bool") return tick(p.label, st[p.key], v => { st[p.key] = v; changed("settle"); }, p.doc);
  if (p.kind === "seed") {
    const row = el("div", "slide");
    const num = el("input");
    num.type = "number"; num.value = st[p.key];
    num.onchange = () => { st[p.key] = +num.value; changed("settle"); };
    const dice = el("button", "", "⚄");
    dice.title = "a random seed";
    dice.onclick = () => { st[p.key] = Math.floor(Math.random() * 99999); num.value = st[p.key]; changed("settle"); };
    const k = el("span", "k", p.label);
    k.title = p.doc || "";
    row.append(k, num, dice);
    return row;
  }
  if (p.kind === "text") {
    const w = el("input");
    w.type = "text"; w.value = st[p.key] || ""; w.placeholder = p.doc;
    w.onchange = () => { st[p.key] = w.value.trim(); changed("settle"); };
    return w;
  }
  return slider(p, st, {onInput: () => changed("drag"), onChange: () => changed("settle")});
}

// ---------- where the material goes ----------
const FADES = [["off", "none"], ["across", "from a side"], ["from a spot", "from a spot"],
               ["from the edges", "from the edges"], ["with depth", "with depth"]];
const SIDES = [["top", 0], ["right", Math.PI / 2], ["bottom", Math.PI], ["left", 3 * Math.PI / 2]];
const WORDS_HINT = {"object (words)": "what to look for: the dress, the cliff, trees",
                    "quality (words)": "a quality or thing: wet, shiny, rough, sky"};

// The card: the quick sources as chips, every other source behind "other…", then what the chosen
// source needs (a part, words, clicks, a side, how much), fine edges, and the gradient folded
// below. Clicks on the picture land through `click`; `armed` is the point list they go to.
export class WhereCard {
  constructor({root, spec, st, quick, onChange, onArm}) {
    Object.assign(this, {root, spec, st, quick, onChange, onArm});
    this.armed = null;
    root.innerHTML = "";
    this.chipBox = el("div", "chips");
    this.chipBox.setAttribute("role", "group");
    this.other = el("select");
    this.other.setAttribute("aria-label", "every mask source");
    const mp = param(spec, "mask");
    const grouped = new Set((mp.optgroups || []).flatMap(g => g[1]));
    this.other.add(new Option("other…", ""));
    for (const o of mp.options) if (!grouped.has(o) && o !== "none") this.other.add(new Option(o, o));
    for (const [title, opts] of mp.optgroups || []) {
      const g = document.createElement("optgroup");
      g.label = title;
      for (const o of opts) { const opt = new Option(o, o); opt.title = mp.choices[o] || ""; g.appendChild(opt); }
      this.other.appendChild(g);
    }
    this.other.onchange = () => { if (this.other.value) this.source(this.other.value); };
    this.args = el("div", "args");
    this.extra = el("div", "args");        // the page's own, under the source (the editor's focus)
    this.gradient = el("details", "fold");
    this.gradient.innerHTML = `<summary>gradient <span class="sum"></span></summary>`;
    this.fadeBox = el("div", "args");
    this.gradient.appendChild(this.fadeBox);
    root.append(this.chipBox, this.args, this.extra, this.gradient);
  }
  source(v) {
    const st = this.st;
    st.mask = v;
    const pts = param(this.spec, "points");
    const wants = pts && applies(pts, st);
    this.arm(wants ? "points" : this.armed === "points" ? null : this.armed);
    this.onChange(true);
  }
  arm(key) {
    this.armed = key;
    if (this.onArm) this.onArm(key);
  }
  click(x, y, shift) {
    if (!this.armed) return;
    const pt = [x, y, shift ? 0 : 1];
    if (this.armed === "points") this.st.points = [...(this.st.points || []), pt];
    else this.st.points_fade = [pt];
    this.onChange(true);
  }
  marks() {
    const st = this.st, out = [];
    const pts = param(this.spec, "points");
    if (pts && applies(pts, st)) for (const [x, y, inc] of st.points || []) out.push({x, y, out: !inc});
    if (st.mask !== "none" && st.fade === "from a spot") for (const [x, y] of st.points_fade || []) out.push({x, y, label: "◐"});
    return out;
  }
  sync() {
    const st = this.st, spec = this.spec;
    const quickSet = new Set(this.quick.map(q => q[0]));
    this.chipBox.innerHTML = "";
    for (const [v, label] of this.quick) {
      const b = el("button", "", label);
      b.title = param(spec, "mask").choices[v] || "";
      b.setAttribute("aria-pressed", st.mask === v);
      b.onclick = () => this.source(v);
      this.chipBox.appendChild(b);
    }
    this.other.value = quickSet.has(st.mask) ? "" : st.mask;
    this.other.classList.toggle("on", !quickSet.has(st.mask));
    this.chipBox.appendChild(this.other);
    // what the chosen source needs
    const a = this.args;
    a.innerHTML = "";
    const masked = st.mask !== "none";
    const doc = param(spec, "mask").choices[st.mask];
    if (masked && !quickSet.has(st.mask) && doc) a.appendChild(el("div", "hintline", doc));
    const p = k => { const q = param(spec, k); return q && applies(q, st) ? q : null; };
    if (p("part")) a.appendChild(chips(p("part").options.map(o => [o, o]), st.part, v => { st.part = v; this.onChange(true); }, true));
    if (p("words")) {
      const w = el("input");
      w.type = "text"; w.value = st.words || ""; w.placeholder = WORDS_HINT[st.mask] || "";
      w.onchange = () => { st.words = w.value.trim(); this.onChange(true); };
      a.appendChild(w);
    }
    if (p("points")) a.appendChild(this.pointsRow("points", st.mask === "object (click)" || st.mask === "material (click)"
      ? "click the picture to select · shift-click leaves a part out" : "click the picture"));
    if (p("angle")) a.appendChild(chips(SIDES.map(([n, v]) => [v, "from the " + n]), st.angle,
                                        v => { st.angle = v; this.onChange(true); }, true));
    if (p("coverage")) a.appendChild(slider(p("coverage"), st, {onChange: () => this.onChange(true)}));
    if (p("matte")) a.appendChild(tick("fine edges (hair, fur, leaves)", st.matte, v => { st.matte = v; this.onChange(true); },
                                       param(spec, "matte").doc));
    // the gradient
    this.gradient.hidden = !masked;
    this.gradient.querySelector(".sum").textContent = st.fade === "off" ? "" : "· " + FADES.find(f => f[0] === st.fade)[1];
    const fa = this.fadeBox;
    fa.innerHTML = "";
    fa.appendChild(chips(FADES, st.fade, v => {
      st.fade = v;
      this.arm(v === "from a spot" ? "points_fade" : this.armed === "points_fade" ? null : this.armed);
      this.onChange(true);
    }));
    if (st.fade !== "off") {
      if (st.fade === "across") {
        fa.appendChild(chips(SIDES.map(([n, v]) => [v, "from the " + n]), null, v => { st.fade_angle = v; this.onChange(true); }, true));
        fa.lastChild.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", Math.abs(st.fade_angle - SIDES[i][1]) < 0.05));
      } else if (st.fade === "from a spot") {
        fa.appendChild(this.pointsRow("points_fade", "click the picture to place the spot"));
      }
      for (const [key, label, doc] of [["fade_reach", "reach", "how far into the selection the material reaches"],
                                       ["fade_length", "fade length", "how long the fade is: 1 across all of it, 0 a clean line"]])
        fa.appendChild(slider(param(spec, key), st, {label, doc, onChange: () => this.onChange(true)}));
      fa.appendChild(tick("the other way round", st.fade_reverse, v => { st.fade_reverse = v; this.onChange(true); }));
      const by = param(spec, "fade_by");
      if (by) {
        fa.appendChild(chips([["redraw", "grades the redraw", by.choices.redraw], ["blend", "fades the result", by.choices.blend]],
                             st.fade_by, v => { st.fade_by = v; this.onChange(false); }));
        fa.lastChild.className = "seg";
      }
    }
  }
  pointsRow(key, help) {
    const row = el("div", "points");
    const n = (this.st[key] || []).length;
    const arm = el("button", "", this.armed === key ? "clicking…" : "click the picture");
    arm.setAttribute("aria-pressed", this.armed === key);
    arm.onclick = () => { this.arm(this.armed === key ? null : key); this.onChange(false); };
    const clear = el("button", "", "clear");
    clear.onclick = () => { this.st[key] = []; this.onChange(true); };
    row.append(el("span", "pcount", n ? n + (n === 1 ? " point" : " points") : "no points yet"), arm, clear);
    const wrap = el("div");
    wrap.append(row, el("div", "hintline", help));
    return wrap;
  }
}
