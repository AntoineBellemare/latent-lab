// What every tab is built from: the stage (a picture, a before-and-after, a grid), the history
// strip, the picture picker, the job queue and the live socket, and the way a result travels from
// one tab to another. The lab page keeps its own; the tabs (editor, mapping, …) share these.
export const $ = id => document.getElementById(id);
export const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
export const wait = ms => new Promise(r => setTimeout(r, ms));

export let MENU = null;
export async function menu() {
  MENU = await (await fetch("/api/menu")).json();
  return MENU;
}

// ---------- the tabs ----------
export const PAGES = {lab: "/", editor: "/editor.html", mapping: "/mapping.html", hidden: "/hidden.html", tokens: "/tokens.html"};
// Which tab a render belongs to: the one that says so in its settings (`_tab`), else by its method.
export function tabOf(method, params) {
  if (params && params._tab) return params._tab;
  if (method === "edit") return "editor";
  if (method === "illusion") return "hidden";
  const m = MENU && MENU.methods[method];
  return m && m.family === "mapping" ? "mapping" : "lab";
}
export const tabOfJob = job => tabOf(job.method, job.params);

export function stored(key) {
  try { return JSON.parse(localStorage.getItem(key) || "null"); } catch (e) { return null; }
}
export function store(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private window */ }
}

// What another tab sent here, in the address: #content=…, #texture=…, #job=… (read once, then
// taken out of the address so a reload does not send it again).
export function arrivals() {
  const got = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  if (location.hash) history.replaceState(null, "", location.pathname);
  return got;
}
export const madeIn = job => tabOfJob(job) === "lab" ? "made in the lab" : "made in the " + tabOfJob(job) + " tab";
export const findJob = id => (MENU.history || []).find(h => h.id === id) || null;

export function status(text, bad) {
  $("status").textContent = text;
  $("status").classList.toggle("err", !!bad);
}
export const nameOf = n => !n ? "" : n.startsWith("u:") ? "⬆ " + n.slice(2) : n.startsWith("i:") ? "archive #" + n.slice(2)
  : n.replace(/^[0-9a-f]{8}_/, "");

// ---------- the job queue ----------
export async function runJob(method, params, inputs, onTick) {
  const t0 = Date.now();
  const tick = onTick && setInterval(() => onTick(Math.round((Date.now() - t0) / 1000)), 500);
  try {
    let job = await (await fetch("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"},
                                              body: JSON.stringify({method, params, inputs})})).json();
    runJob.current = job.id;
    while (job.status === "queued" || job.status === "running") {
      await wait(500);
      job = await (await fetch("/api/jobs/" + job.id)).json();
    }
    if (job.status !== "done") throw new Error(job.note || job.status);
    job.params = job.params || params;
    job.inputs = job.inputs || inputs;
    return job;
  } finally {
    if (tick) clearInterval(tick);
  }
}
export const cancelJob = id => id && fetch("/api/jobs/" + id, {method: "DELETE"});

export function working(on) {
  const light = $("jobsLight");
  if (!light) return;
  light.textContent = on ? "working" : "·";
  light.classList.toggle("busy", on);
}

// ---------- the live socket: settings up, frames down, the newest request wins ----------
// One request in flight per socket; a newer one waits in `pending` and replaces any older one
// waiting there, so the picture always answers the newest position and a fast hand builds no
// backlog. `onFrame(blob, meta)`; `meta.seq` says which request the frame answers.
export class Live {
  constructor({onFrame, onError, onKept, onOpen} = {}) {
    Object.assign(this, {onFrame, onError, onKept, onOpen});
    this.seq = 0;
    this.pending = null;
    this.inflight = false;
    this.connect();
  }
  connect() {
    const ws = this.ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/api/live");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { this.inflight = false; if (this.onOpen) this.onOpen(); this.flush(); };
    ws.onmessage = ev => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "kept") { if (this.onKept) this.onKept(msg.job); return; }
        if (msg.type === "error") { this.inflight = false; if (this.onError) this.onError(msg); this.flush(); }
        return;
      }
      const n = new DataView(ev.data).getUint32(0, true);
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(ev.data, 4, n)));
      this.inflight = false;
      if (this.onFrame) this.onFrame(new Blob([ev.data.slice(4 + n)], {type: "image/jpeg"}), meta);
      this.flush();
    };
    ws.onclose = () => { this.inflight = false; setTimeout(() => this.connect(), 1500); };
  }
  send(req) {
    req.seq = ++this.seq;
    this.pending = req;
    this.flush();
    return req.seq;
  }
  get idle() { return !this.inflight && !this.pending; }
  flush() {
    if (!this.ws || this.ws.readyState !== 1 || this.inflight || !this.pending) return;
    this.inflight = true;
    this.ws.send(JSON.stringify(this.pending));
    this.pending = null;
  }
  keep() {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({type: "keep"}));
  }
}

// The mask alone, in gold over the picture: grain's show-mask builds the same mask from the same
// settings in a fraction of a second, and leaves the history alone.
export class MaskPreview {
  constructor(onUrl, onError) {
    this.url = null;
    this.live = new Live({
      onFrame: (blob, meta) => {
        if (meta.seq !== this.live.seq) return;
        if (this.url) URL.revokeObjectURL(this.url);
        this.url = URL.createObjectURL(blob);
        onUrl(this.url);
      },
      onError: msg => onError && onError(msg.note),
      onOpen: () => this.last && this.request(this.last),
    });
  }
  request(req) {
    this.last = req;
    clearTimeout(this.t);
    this.t = setTimeout(() => {
      const {params, content, texture} = req;
      this.live.send({method: "grain", phase: "settle", params: {...params, show_mask: true},
                      inputs: {content, texture: texture || MENU.textures[0]}});
    }, 200);
  }
}
// The settings a mask is made of, read from a method's own declaration.
export const MASK_GROUPS = ["where", "second layer", "third layer", "gradient", "mask edges"];
export const maskKeys = spec => spec.params.filter(p => MASK_GROUPS.includes(p.group)).map(p => p.key);

// ---------- the stage: a picture, a before-and-after, or a grid ----------
// `views` are the page's own ([value, label]); the page says what each shows through `onView`.
export class Stage {
  constructor(root, {views, onView, onPoint}) {
    this.root = root;
    root.innerHTML = `
      <div id="stageBox">
        <div id="compare" hidden><img id="after" alt="the result"><img id="before" alt="the picture">
          <span class="tag l">picture</span><span class="tag r" id="afterTag">result</span>
          <div class="rule"></div><div class="grip">◂ drag ▸</div></div>
        <div id="still"><img id="stillImg" alt=""><div id="marks"></div></div>
        <div id="grid" hidden><div id="gridBar"></div><div id="gridCells"></div></div>
        <div id="busy" hidden><span></span></div>
      </div>
      <div id="viewbar">
        <div class="seg" id="viewSeg" role="group" aria-label="view"></div>
        <span id="status"></span>
        <button id="sendTo" class="quiet" hidden title="use this result as a picture or a texture in a tab">send to…</button>
        <a id="download" hidden download>download</a>
      </div>
      <div id="historyRow">
        <div class="seg small" id="historySeg" role="group" aria-label="history">
          <button data-h="here" title="what was made in this tab">here</button><button data-h="all" title="what was made in every tab">all</button>
        </div>
        <div id="history" aria-label="renders"></div>
      </div>`;
    this.view = null;
    this.onView = onView;
    this.views = views;
    this.drawViews();
    const c = $("compare");
    const at = e => {
      const r = c.getBoundingClientRect();
      c.style.setProperty("--x", Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100)) + "%");
    };
    c.addEventListener("pointerdown", e => { c.setPointerCapture(e.pointerId); at(e); });
    c.addEventListener("pointermove", e => { if (e.buttons) at(e); });
    const point = (e, shift) => {
      if (!this.picking) return;
      const r = $("stillImg").getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      if (x < 0 || x > 1 || y < 0 || y > 1) return;
      onPoint(+x.toFixed(4), +y.toFixed(4), shift);
    };
    $("still").addEventListener("click", e => point(e, e.shiftKey));
    $("still").addEventListener("contextmenu", e => { if (this.picking) { e.preventDefault(); point(e, true); } });
    addEventListener("resize", () => this.fit());
  }
  drawViews() {
    const seg = $("viewSeg");
    seg.innerHTML = "";
    for (const [v, label] of this.views) {
      const b = el("button", "", label);
      b.dataset.view = v;
      b.onclick = () => this.onView(v);
      seg.appendChild(b);
    }
    this.mark();
  }
  // the grid's own button appears once there is a grid
  offerGrid(on) {
    const has = this.views.some(v => v[0] === "grid");
    if (on && !has) this.views = [...this.views, ["grid", "grid"]];
    if (!on && has) this.views = this.views.filter(v => v[0] !== "grid");
    if (on !== has) this.drawViews();
  }
  mark(v = this.view) {
    this.view = v;
    $("viewSeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.view === v));
  }
  still(src) {
    $("compare").hidden = true;
    $("grid").hidden = true;
    $("still").hidden = false;
    const img = $("stillImg");
    img.onload = () => this.fit();
    if (src && img.getAttribute("src") !== src) img.src = src;
    else this.fit();
  }
  compare(after, before, tag = "result") {
    $("still").hidden = true;
    $("grid").hidden = true;
    $("compare").hidden = false;
    $("afterTag").textContent = tag;
    const a = $("after");
    a.onload = () => this.fit();
    if (a.getAttribute("src") !== after) a.src = after;
    if (before && $("before").getAttribute("src") !== before) $("before").src = before;
    $("before").hidden = !before;
    this.fit();
  }
  grid() {
    $("still").hidden = true;
    $("compare").hidden = true;
    $("grid").hidden = false;
  }
  busy(text) {
    $("busy").hidden = !text;
    if (text) $("busy").querySelector("span").textContent = text;
  }
  marks(list) {
    const box = $("marks");
    box.innerHTML = "";
    for (const {x, y, out, label} of list) {
      const d = el("div", "mark" + (out ? " out" : ""), label || "");
      d.style.left = (x * 100) + "%"; d.style.top = (y * 100) + "%";
      box.appendChild(d);
    }
  }
  setPicking(on) {
    this.picking = on;
    $("still").classList.toggle("picking", !!on);
  }
  // The stage holds the frame's own shape at the largest size the box allows.
  fit() {
    const box = $("stageBox").getBoundingClientRect();
    const fitInto = (node, img) => {
      if (!img.naturalWidth) return;
      const k = Math.min((box.width - 16) / img.naturalWidth, (box.height - 16) / img.naturalHeight);
      node.style.width = Math.floor(img.naturalWidth * k) + "px";
      node.style.height = Math.floor(img.naturalHeight * k) + "px";
    };
    fitInto($("still"), $("stillImg"));
    fitInto($("compare"), $("after"));
  }
  download(job) {
    const a = $("download");
    a.hidden = !job;
    $("sendTo").hidden = !job;
    if (!job) return;
    a.href = job.url;
    a.setAttribute("download", job.result || "render.jpg");
    $("sendTo").onclick = e => sendMenu(e.currentTarget, job);
  }
}

// ---------- the history: every render and every kept frame, of this tab or of all of them ----------
// A render made here is shown on a click and gives its settings back on a double-click; one made in
// another tab is shown here and, on a double-click, opened in its own tab with its settings.
export class History {
  constructor({here, describe, onShow, onReuse, key}) {
    Object.assign(this, {here, describe, onShow, onReuse, key});
    this.jobs = [];
    this.scope = (stored(key) || {}).scope || "here";
    $("historySeg").querySelectorAll("button").forEach(b => b.onclick = () => {
      this.scope = b.dataset.h;
      store(key, {scope: this.scope});
      this.draw();
    });
  }
  load(jobs) { this.jobs = jobs.filter(j => j.url); this.draw(); }
  add(job) { this.jobs.push(job); this.draw(); this.select(job); }
  draw() {
    $("historySeg").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.h === this.scope));
    const box = $("history");
    box.innerHTML = "";
    const shown = this.jobs.filter(j => this.scope === "all" || this.here(j)).slice(-160);
    for (const job of shown) {
      const n = el("button", "h");
      n._job = job;
      const img = el("img");
      img.src = job.thumb ? "/api/out/" + job.thumb : job.url;
      img.loading = "lazy";
      const mine = this.here(job);
      n.title = (mine ? this.describe(job) : job.method + " · " + madeIn(job)) +
        (mine ? " · click to show, double-click to take its settings" : " · click to show, double-click to open it there");
      if (!mine) n.classList.add("other");
      n.appendChild(img);
      n.onclick = () => this.onShow(job);
      n.ondblclick = () => mine ? this.onReuse(job) : location.assign(PAGES[tabOfJob(job)] + "#job=" + job.id);
      box.prepend(n);
    }
    if (!shown.length) box.appendChild(el("span", "empty", this.scope === "all" ? "nothing made yet"
      : "nothing made in this tab yet"));
    this.select(this.selected);
  }
  select(job) {
    this.selected = job;
    $("history").querySelectorAll(".h").forEach(n => n.classList.toggle("sel", n._job === job));
  }
}

// ---------- send to: a result becomes a picture or a texture, here or in another tab ----------
const DESTINATIONS = [["editor", "content", "the editor's picture"], ["editor", "texture", "the editor's material"],
                      ["mapping", "content", "mapping's picture"], ["mapping", "texture", "mapping's texture"],
                      ["hidden", "content", "the face to hide"], ["hidden", "texture", "the texture to hide it in"],
                      ["tokens", "content", "the tokens tab's picture"], ["tokens", "texture", "the tokens tab's first material"]];
let arriveHere = null;  // the page's own: (target, name) => void
export const onArrive = fn => { arriveHere = fn; };

function sendMenu(anchor, job) {
  document.querySelectorAll(".popmenu").forEach(n => n.remove());
  const menuBox = el("div", "popmenu");
  const here = location.pathname;
  for (const [tab, target, label] of DESTINATIONS) {
    const b = el("button", "", (PAGES[tab] === here ? "here: " : "") + label);
    b.onclick = async () => {
      menuBox.remove();
      status("sending…");
      const blob = await (await fetch(job.url)).blob();
      const r = await fetch("/api/upload?name=" + encodeURIComponent(job.method.replace(/\W+/g, "-") + "-" + job.id + ".jpg"),
                            {method: "POST", body: blob});
      if (!r.ok) return status("that result could not be sent", true);
      const name = "u:" + (await r.json()).name;
      if (!MENU.uploads.includes(name.slice(2))) MENU.uploads.push(name.slice(2));
      if (PAGES[tab] === here && arriveHere) arriveHere(target, name);
      else location.assign(PAGES[tab] + "#" + target + "=" + encodeURIComponent(name));
    };
    menuBox.appendChild(b);
  }
  const r = anchor.getBoundingClientRect();
  menuBox.style.left = Math.max(8, r.right - 220) + "px";
  menuBox.style.bottom = (innerHeight - r.top + 6) + "px";
  document.body.appendChild(menuBox);
  setTimeout(() => addEventListener("click", function off(e) {
    if (!menuBox.contains(e.target)) { menuBox.remove(); removeEventListener("click", off); }
  }), 0);
}

// ---------- pictures in: a drop card per input, the picker, uploads ----------
export function dropCard(target, {get, choose, label}) {
  const d = el("div", "drop");
  d.dataset.target = target;
  const img = el("img");
  img.alt = "";
  img.title = "click to upload, or drop a picture here";
  const text = el("div");
  const b = el("b");
  const hint = el("span");
  const link = el("button", "link", "choose…");
  link.onclick = async () => { const v = await pick(target); if (v) choose(v); };
  hint.append("drop " + label + ", or ", link);
  text.append(b, hint);
  const file = el("input");
  file.type = "file"; file.accept = "image/*"; file.hidden = true;
  file.onchange = async () => { if (file.files[0]) choose(await upload(file.files[0])); };
  img.onclick = () => file.click();
  d.append(img, text, file);
  dropZone(d, async f => choose(await upload(f)));
  d.refresh = () => {
    const name = get();
    const src = "/api/source?kind=" + target + "&name=" + encodeURIComponent(name);
    if (img.getAttribute("src") !== src) img.src = src;
    b.textContent = nameOf(name);
  };
  return d;
}

export function dropZone(node, onFile) {
  node.addEventListener("dragover", e => { e.preventDefault(); node.classList.add("over"); });
  node.addEventListener("dragleave", () => node.classList.remove("over"));
  node.addEventListener("drop", async e => {
    e.preventDefault();
    node.classList.remove("over");
    const file = [...(e.dataTransfer.files || [])].find(f => f.type.startsWith("image/"));
    if (file) onFile(file);
  });
}

export async function upload(file) {
  status("uploading " + file.name + "…");
  const r = await fetch("/api/upload?name=" + encodeURIComponent(file.name), {method: "POST", body: file});
  if (!r.ok) { status("that file could not be read as a picture", true); throw new Error("upload"); }
  const got = await r.json();
  if (!MENU.uploads.includes(got.name)) MENU.uploads.push(got.name);
  return "u:" + got.name;
}

// The chooser: pictures, uploads, and for textures the archive. With `several`, tiles toggle and
// the choice is a list.
let pickPage = 0;
export function pick(target, several = false) {
  let box = $("picker");
  if (!box) {
    box = el("div");
    box.id = "picker";
    box.hidden = true;
    box.innerHTML = `<div id="pickerCard"><div class="row"><div class="seg" id="pickerTabs" role="group" aria-label="source"></div>
      <span class="lab" id="pickerCount"></span><button id="pickerUse" class="primary" hidden></button><button id="pickerClose">close</button></div>
      <div id="pickerGrid"></div></div>`;
    document.body.appendChild(box);
  }
  return new Promise(resolve => {
    const chosen = [];
    const done = v => { box.hidden = true; removeEventListener("keydown", esc); resolve(v); };
    const esc = e => { if (e.key === "Escape") done(null); };
    addEventListener("keydown", esc);
    $("pickerClose").onclick = () => done(null);
    box.onclick = e => { if (e.target === box) done(null); };
    const use = $("pickerUse");
    use.hidden = !several;
    const tell = () => { use.textContent = "use " + chosen.length; use.disabled = !chosen.length; };
    use.onclick = () => done([...chosen]);
    tell();
    const tabs = $("pickerTabs");
    tabs.innerHTML = "";
    const lists = target === "content"
      ? [["pictures", () => MENU.contents.map(n => [n, n])], ["your uploads", () => MENU.uploads.map(n => ["u:" + n, n])]]
      : [["textures", () => MENU.textures.map(n => [n, n])], ["your uploads", () => MENU.uploads.map(n => ["u:" + n, n])],
         ["the archive", null]];
    const tileFor = (src, label, value) => {
      const b = el("button", "tile");
      const img = el("img");
      img.src = src; img.loading = "lazy"; img.alt = "";
      b.title = label;
      b.append(img, el("span", "", label.replace(/^[0-9a-f]{8}_/, "")));
      b.setAttribute("aria-pressed", chosen.includes(value));
      b.onclick = () => {
        if (!several) return done(value);
        const i = chosen.indexOf(value);
        if (i >= 0) chosen.splice(i, 1); else chosen.push(value);
        b.setAttribute("aria-pressed", chosen.includes(value));
        tell();
      };
      return b;
    };
    const fill = async i => {
      tabs.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-pressed", i === j));
      const grid = $("pickerGrid");
      grid.innerHTML = "";
      const list = lists[i][1];
      if (list) {
        const items = list();
        $("pickerCount").textContent = items.length + " pictures" + (several ? " · pick several" : "");
        for (const [value, label] of items)
          grid.appendChild(tileFor("/api/source?kind=" + target + "&name=" + encodeURIComponent(value), label, value));
        return;
      }
      try {  // the archive's photographs by page (it needs the photo drive)
        const a = await (await fetch("/api/archive?page=" + pickPage)).json();
        $("pickerCount").textContent = a.total + " photographs · page " + (a.page + 1);
        for (const row of a.rows) grid.appendChild(tileFor("/api/thumb?i=" + row.i, row.cap || row.name, "i:" + row.i));
        const next = el("button", "", "next page");
        next.onclick = () => { pickPage += 1; fill(i); };
        grid.appendChild(next);
      } catch (e) {
        $("pickerCount").textContent = "the archive needs the photo drive (G:)";
      }
    };
    lists.forEach(([label], i) => {
      const b = el("button", "", label);
      b.onclick = () => fill(i);
      tabs.appendChild(b);
    });
    box.hidden = false;
    fill(0);
  });
}

export async function saveRecipe(name, recipe) {
  await fetch("/api/recipes", {method: "POST", headers: {"Content-Type": "application/json"},
                               body: JSON.stringify({name, recipe})});
  status("saved as the recipe “" + name + "”");
}
