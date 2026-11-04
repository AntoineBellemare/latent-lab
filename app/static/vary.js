// Vary: one thing changed across a row of renders, seen side by side on the stage — several
// materials, one setting swept over a range, or several seeds. The renders go through the job
// queue one after another (so "stop" stops, and the queue stays free for others between them); each
// lands in the history with its settings, and the grid can be saved as one sheet.
import {$, el, pick, runJob, cancelJob, nameOf, status, working} from "./studio.js";
import {chips} from "./controls.js";

// The page says what it renders: `method()`, `params()`, `inputs()`, the numeric `settings()` it
// offers ([p, label]), whether seeds matter (`seeded()`), the input its materials go to
// (`material`), and what to do with a finished render (`done(job)`) and a clicked cell (`show(job)`).
export class Vary {
  constructor(stage, page) {
    this.stage = stage;
    this.page = page;
    this.cells = [];
    this.kind = "materials";
    this.materials = [];
    this.count = 9;
  }

  open() {
    document.querySelectorAll(".popcard").forEach(n => n.remove());
    const card = el("div", "popcard");
    const draw = () => {
      card.innerHTML = "";
      const head = el("div", "row");
      head.append(el("span", "lab", "vary"), el("span", "grow"));
      const close = el("button", "", "close");
      close.onclick = () => card.remove();
      head.appendChild(close);
      card.appendChild(head);
      const kinds = [["materials", "materials"], ["setting", "a setting"]];
      if (this.page.seeded()) kinds.push(["seeds", "seeds"]);
      if (!kinds.some(k => k[0] === this.kind)) this.kind = "materials";
      card.appendChild(chips(kinds, this.kind, v => { this.kind = v; draw(); }));
      card.lastChild.className = "seg";
      let n = 0;
      if (this.kind === "materials") {
        const list = el("div", "chips small");
        for (const m of this.materials) {
          const b = el("button", "", nameOf(m) + " ×");
          b.onclick = () => { this.materials = this.materials.filter(x => x !== m); draw(); };
          list.appendChild(b);
        }
        const add = el("button", "", "choose textures…");
        add.onclick = async () => {
          const got = await pick("texture", true);
          if (got) this.materials = [...new Set([...this.materials, ...got])];
          draw();
        };
        card.append(list, add, el("div", "hintline", "the same picture and settings, once with each texture"));
        n = this.materials.length;
      } else if (this.kind === "setting") {
        const settings = this.page.settings();
        if (!settings.some(([p]) => p.key === this.key)) this.key = settings.length ? settings[0][0].key : null;
        const sel = el("select");
        for (const [p, label] of settings) sel.add(new Option(label || p.label, p.key));
        sel.value = this.key;
        const range = el("div", "row");
        const p = (settings.find(([q]) => q.key === this.key) || [])[0];
        if (p) {
          if (this.from === undefined || this.forKey !== p.key) { this.from = p.lo; this.to = p.hi; this.forKey = p.key; }
          const num = (v, set) => {
            const i = el("input");
            i.type = "number"; i.step = p.step || 0.01; i.value = v;
            i.onchange = () => set(+i.value);
            return i;
          };
          range.append(el("span", "k", "from"), num(this.from, v => this.from = v), el("span", "k", "to"),
                       num(this.to, v => this.to = v));
        }
        sel.onchange = () => { this.key = sel.value; draw(); };
        const steps = el("div", "row");
        steps.append(el("span", "k", "steps"),
                     chips([5, 7, 9, 13].map(c => [c, String(c)]), this.count, v => { this.count = v; draw(); }, true));
        card.append(sel, range, steps, el("div", "hintline", "steps across the range, everything else held"));
        n = p ? this.count : 0;
      } else {
        const counts = chips([4, 6, 9, 12].map(c => [c, String(c)]), this.count, v => { this.count = v; draw(); }, true);
        const row = el("div", "row");
        row.append(el("span", "k", "how many"), counts);
        card.append(row, el("div", "hintline", "the same settings drawn with different random seeds"));
        n = this.count;
      }
      const go = el("button", "primary", n ? `render ${n}` : "render");
      go.disabled = !n;
      go.onclick = () => { card.remove(); this.run(); };
      card.appendChild(go);
    };
    draw();
    $("stageBox").appendChild(card);
  }

  // A page may translate a cell's settings before it is drawn (`adjust`): the hidden tab's one
  // "hidden → plain" setting becomes each way's own.
  plan() {
    const cells = this.cells_();
    return this.page.adjust ? cells.map(c => ({...c, params: this.page.adjust(c.params)})) : cells;
  }

  cells_() {
    const base = this.page.params(), inputs = this.page.inputs();
    if (this.kind === "materials")
      return this.materials.map(m => ({label: nameOf(m), params: {...base, ...this.page.materialParams?.()},
                                       inputs: {...inputs, [this.page.material]: m}}));
    if (this.kind === "seeds") {
      const key = this.page.seedKey || "seed";
      const s0 = +base[key] || 0;
      return Array.from({length: this.count}, (_, i) => ({label: "seed " + (s0 + i), params: {...base, [key]: s0 + i}, inputs}));
    }
    const [p, label] = this.page.settings().find(([q]) => q.key === this.key);
    return Array.from({length: this.count}, (_, i) => {
      let v = this.from + (this.to - this.from) * i / (this.count - 1);
      v = p.kind === "int" ? Math.round(v) : +v.toFixed(4);
      return {label: (label || p.label) + " " + (p.kind === "angle" ? Math.round(v * 180 / Math.PI) + "°" : v),
              params: {...base, [p.key]: v}, inputs};
    });
  }

  async run() {
    const plan = this.plan();
    if (!plan.length) return;
    this.stop();
    const runId = this.runId = (this.runId || 0) + 1;
    const method = this.page.method();
    const [p, label] = this.page.settings().find(([q]) => q.key === this.key) || [];
    this.title = this.kind === "materials" ? "materials" : this.kind === "seeds" ? "seeds" : label || (p && p.label) || this.key;
    this.cells = plan.map(c => ({...c, job: null, state: "waiting"}));
    this.stage.offerGrid(true);
    this.showGrid();
    working(true);
    for (const cell of this.cells) {
      if (this.runId !== runId) break;
      cell.state = "drawing";
      this.drawGrid();
      try {
        cell.job = await runJob(method, cell.params, cell.inputs);
        cell.state = "done";
        this.page.done(cell.job);
      } catch (e) {
        cell.state = this.runId === runId ? "failed: " + e.message : "stopped";
      }
      this.drawGrid();
    }
    if (this.runId === runId) {
      status("the grid is done · click a cell to see it against the picture");
      this.runId += 1;
    }
    working(false);
  }

  stop() {
    if (!this.running) return;
    this.runId += 1;
    cancelJob(runJob.current);
    for (const c of this.cells) if (c.state === "waiting" || c.state === "drawing") c.state = "stopped";
    working(false);
    this.drawGrid();
  }
  get running() { return this.cells.some(c => c.state === "waiting" || c.state === "drawing"); }

  showGrid() {
    this.stage.mark("grid");
    this.stage.grid();
    this.drawGrid();
  }

  drawGrid() {
    const bar = $("gridBar");
    bar.innerHTML = "";
    const done = this.cells.filter(c => c.job).length;
    bar.append(el("span", "lab", `vary · ${this.title} · ${done} of ${this.cells.length}`), el("span", "grow"));
    if (this.running) {
      const stop = el("button", "", "stop");
      stop.onclick = () => this.stop();
      bar.appendChild(stop);
    }
    if (done) {
      const sheet = el("button", "", "save sheet");
      sheet.title = "the grid as one picture, each cell labelled";
      sheet.onclick = () => this.sheet();
      bar.appendChild(sheet);
    }
    const again = el("button", "", "vary…");
    again.onclick = () => this.open();
    bar.appendChild(again);
    const box = $("gridCells");
    box.innerHTML = "";
    box.style.setProperty("--cols", Math.ceil(Math.sqrt(this.cells.length * 1.4)));
    for (const c of this.cells) {
      const cell = el("button", "cell");
      if (c.job) {
        const img = el("img");
        img.src = c.job.url;
        img.alt = c.label;
        cell.appendChild(img);
        cell.onclick = () => this.page.show(c.job);
        cell.title = c.label + " · click to see it against the picture";
      } else {
        cell.appendChild(el("div", "wait", c.state === "drawing" ? "drawing…" : c.state));
        cell.disabled = true;
      }
      cell.appendChild(el("span", "", c.label));
      box.appendChild(cell);
    }
  }

  // The grid as one image: the cells in their rows, each labelled underneath.
  async sheet() {
    const done = this.cells.filter(c => c.job);
    const imgs = await Promise.all(done.map(c => new Promise(res => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = () => res(null);
      i.src = c.job.url;
    })));
    const cols = Math.ceil(Math.sqrt(done.length * 1.4)), rows = Math.ceil(done.length / cols);
    const first = imgs.find(Boolean);
    const cw = 480, ch = Math.round(cw * first.naturalHeight / first.naturalWidth), lh = 30, gap = 10;
    const cv = el("canvas");
    cv.width = cols * (cw + gap) + gap;
    cv.height = rows * (ch + lh + gap) + gap + 36;
    const g = cv.getContext("2d");
    g.fillStyle = "#0b0e11";
    g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = "#c9a455";
    g.font = "15px 'IBM Plex Mono', monospace";
    g.fillText(`${this.page.method()} · vary ${this.title} · ${nameOf(this.page.inputs().content || "")}`, gap, 24);
    done.forEach((c, i) => {
      const x = gap + (i % cols) * (cw + gap), y = 36 + gap + Math.floor(i / cols) * (ch + lh + gap);
      if (imgs[i]) g.drawImage(imgs[i], x, y, cw, ch);
      g.fillStyle = "#97a3ab";
      g.font = "13px 'IBM Plex Mono', monospace";
      g.fillText(c.label.slice(0, 60), x, y + ch + 20);
    });
    const a = el("a");
    a.href = cv.toDataURL("image/jpeg", 0.9);
    a.download = `sheet-${this.page.method().replace(/\W+/g, "-")}-${this.title.replace(/\W+/g, "-")}-${Date.now()}.jpg`;
    a.click();
  }
}
