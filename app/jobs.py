"""One queue for every render, fast or slow, with progress and a remembered history.

Fast methods pass through in milliseconds; the minutes-long ones get stage previews and a cancel.
Every finished render lands in the history with its full method+inputs+params record, so any
earlier picture can be reopened AS its settings — the history is the recipe book you didn't have
to keep.
"""

import json
import pathlib
import queue
import re
import threading
import time
import uuid

from .registry import complete


class Hub:
    """WebSocket fan-out that worker threads can safely post to."""

    def __init__(self):
        self.loop = None
        self.queues = set()

    def attach(self, loop):
        self.loop = loop

    def push(self, message):
        if not self.loop:
            return
        data = json.dumps(message)
        for q in list(self.queues):
            self.loop.call_soon_threadsafe(q.put_nowait, data)


class Jobs:
    """Two lanes: live methods (CPU mapping, GANs) never wait behind a minutes-long optimisation."""

    def __init__(self, engine, outdir, hub, methods, exports=None):
        self.engine = engine
        self.methods = methods
        self.out = pathlib.Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.exports = pathlib.Path(exports) if exports else self.out
        self.exports.mkdir(parents=True, exist_ok=True)
        self.hub = hub
        self.lanes = {"live": queue.Queue(), "slow": queue.Queue()}
        self.jobs = {}
        # The history is written beside the renders, so a restart keeps it: the strip, and the
        # editor's renders, come back as they were (entries whose picture is gone are dropped).
        self.book = self.out / "history.json"
        self.history = self._read_book()
        self.lock = threading.Lock()
        for lane in self.lanes.values():
            threading.Thread(target=self._work, args=(lane,), daemon=True).start()

    def submit(self, method, inputs, params, cid=None):
        """`cid` is the client's own tag for the job, so the page recognises even the first message
        about it — the queued and running notices can arrive before the reply carrying the id."""
        if method not in self.methods:
            raise KeyError(method)
        cost = self.methods[method]["cost"]
        params = complete(self.methods[method], params)
        job = dict(id=uuid.uuid4().hex[:10], cid=cid, method=method, inputs=inputs, params=params,
                   status="queued", note="", previews=[], result=None, t=time.time(), cost=cost)
        with self.lock:
            self.jobs[job["id"]] = job
        self.lanes["live" if cost == "live" else "slow"].put(job["id"])
        self.hub.push(dict(type="job", job=self._public(job)))
        return self._public(job)

    def keep(self, method, inputs, params, data):
        """A live frame the player chose to keep: written once, entered in the history like a job."""
        import io

        from PIL import Image

        jid = uuid.uuid4().hex[:10]
        (self.out / f"{jid}.jpg").write_bytes(data)
        thumb = Image.open(io.BytesIO(data))
        thumb.thumbnail((192, 192))
        thumb.convert("RGB").save(self.out / f"{jid}-t.jpg", quality=85)
        job = dict(id=jid, method=method, inputs=inputs, params=params, status="done", note="kept",
                   previews=[], result=f"{jid}.jpg", url=f"/api/out/{jid}.jpg", thumb=f"{jid}-t.jpg", t=time.time(),
                   cost=self.methods[method]["cost"], took=None)
        self._remember(job)
        return job

    def _read_book(self):
        try:
            got = json.loads(self.book.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        where = lambda j: (self.exports if str(j.get("url", "")).startswith("/api/exports/") else self.out) / j["result"]
        return [j for j in got if j.get("result") and where(j).is_file()]

    def _remember(self, job):
        with self.lock:
            self.history.append(job)
            del self.history[:-400]
            try:
                self.book.write_text(json.dumps(self.history), encoding="utf-8")
            except OSError as e:  # the history in memory still holds
                print(f"  ! history not written: {e}", flush=True)

    def get(self, jid):
        job = self.jobs.get(jid)
        return self._public(job) if job else None

    def _label(self, job):
        """A file name that says what the picture is: method, picture, texture or material."""

        def stem(v):
            if not v:
                return None
            if isinstance(v, dict):
                for k in ("texfile", "upmat", "preset", "terms", "point"):
                    if v.get(k):
                        return pathlib.Path(str(v[k])).stem if k in ("texfile", "upmat") else str(v[k])
                return f"archive{v['photo']}" if v.get("photo") not in (None, "") else None
            return pathlib.Path(str(v).split(":", 1)[-1]).stem

        names = [stem(job["inputs"].get(k)) for k in ("content", "texture", "materialA")]
        return re.sub(r"[^0-9A-Za-z._-]+", "-", "_".join([job["method"]] + [n for n in names if n]))[:120]

    def _export(self, job, img):
        """A full-size export, written where it can be found, with the settings that made it."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{self._label(job)}_{img.width}x{img.height}_{stamp}.jpg"
        img.convert("RGB").save(self.exports / name, quality=95)
        return name

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if job and job["status"] in ("queued", "running"):
                job["status"] = "cancelled"
                self.hub.push(dict(type="job", job=self._public(job)))

    def _save(self, jid, img, kind, quality=90):
        name = f"{jid}-{kind}.jpg" if kind else f"{jid}.jpg"
        img.convert("RGB").save(self.out / name, quality=quality)
        return name

    def _work(self, lane):
        while True:
            jid = lane.get()
            job = self.jobs.get(jid)
            if not job or job["status"] == "cancelled":
                continue
            job["status"] = "running"
            self.hub.push(dict(type="job", job=self._public(job)))

            def watch(note, img, job=job):
                if job["status"] == "cancelled":
                    raise RuntimeError("cancelled")
                name = self._save(job["id"], img, f"p{len(job['previews'])}")
                job["previews"].append(name)
                job["note"] = note
                self.hub.push(dict(type="job", job=self._public(job)))

            try:
                t0 = time.time()
                img = self.methods[job["method"]]["run"](self.engine, job["inputs"], job["params"], watch)
                thumb = img.copy()
                thumb.thumbnail((192, 192))
                export = bool(job["params"].get("_export"))
                if export:
                    job["result"] = self._export(job, img)
                    job["url"] = "/api/exports/" + job["result"]
                    job["path"] = str(self.exports / job["result"])
                else:
                    job["result"] = self._save(jid, img, "")
                    job["url"] = "/api/out/" + job["result"]
                job["size"] = list(img.size)
                job["thumb"] = self._save(jid, thumb, "t")
                job["took"] = round(time.time() - t0, 2)
                job["status"] = "done"
                if export:
                    record = {k: job[k] for k in ("method", "inputs", "size", "took")}
                    record["params"] = {k: v for k, v in job["params"].items() if not k.startswith("_")}
                    record["made"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    (self.exports / (job["result"][:-4] + ".json")).write_text(json.dumps(record, indent=1),
                                                                              encoding="utf-8")
                self._remember(self._public(job))
            except Exception as e:
                if job["status"] != "cancelled":
                    job["status"] = "error"
                    job["note"] = str(e)[:300]
                    print(f"  ! job {job['method']}: {e}", flush=True)
            self.hub.push(dict(type="job", job=self._public(job)))

    def _public(self, job):
        return {k: v for k, v in job.items()}
