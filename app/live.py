"""Frames, not jobs: the realtime path a slider rides on.

The job queue is right for an optimisation that takes a minute — progress, previews, a record. It
is wrong for a slider: every frame paid for a disk write, a thumbnail, a history entry, a broadcast
and a second HTTP fetch, which is most of a GAN frame's cost and all of its stutter.

Here one WebSocket carries settings up and JPEG bytes down, rendered in memory and never written.
Each connection holds a single slot, `latest`: a new request replaces whatever was waiting, so a
fast hand never builds a backlog and the picture always answers the newest position. The client
keeps at most one frame in flight too, which makes the render time itself the frame rate.

`phase` says how the frame will be looked at. `drag` is mid-gesture: mapping drops to 256px and
diffusion to its draft at a smaller square. `settle` is the release: full size, full quality. A
frame is only kept — saved to disk, entered in the history — when the client asks.
"""

import asyncio
import io
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from .registry import complete

DRAG_DIFFUSION = {"sd15": 384, "sdxl": 512}
DRAG_MAPPING = 256


class Live:
    def __init__(self, engine, methods, jobs, preset):
        self.engine = engine
        self.methods = methods
        self.jobs = jobs
        self.preset = preset
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="live")

    def adjust(self, spec, params, phase):
        p = dict(params)
        if phase != "drag":
            return p
        if spec["family"] == "mapping":
            # Half the final size (a quarter of the pixels), never below 256 and never above the
            # final: a 768 render drags at 384, a 256 render drags at 256. The page scales either
            # to the same square, so only sharpness changes on release, not size.
            final = int(p.get("size", DRAG_MAPPING))
            p["size"] = min(final, max(DRAG_MAPPING, final // 2 // 32 * 32))
        elif spec["family"] in ("diffusion", "words"):
            p["quality"] = "draft"
            p["_drag"] = True  # fewer steps, the tiny decoder
            p["_size"] = DRAG_DIFFUSION.get(self.preset, 384)
            p["then"] = ""  # a second pass is a settle-time luxury
        elif spec["family"] == "tokens":
            p["_drag"] = True  # the four-times enlargement waits for the release
        return p

    def render(self, req):
        spec = self.methods[req["method"]]
        phase = req.get("phase", "settle")
        params = self.adjust(spec, complete(spec, req.get("params", {})), phase)
        t0 = time.perf_counter()
        img = spec["run"](self.engine, req.get("inputs", {}), params)
        took = (time.perf_counter() - t0) * 1000
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90 if phase == "settle" else 78)
        meta = dict(seq=req.get("seq"), ms=round(took, 1), w=img.width, h=img.height, phase=phase,
                    method=req["method"])
        return buf.getvalue(), meta

    async def serve(self, socket: WebSocket):
        await socket.accept()
        loop = asyncio.get_running_loop()
        state = {"latest": None, "kept": None}
        wake = asyncio.Event()

        async def worker():
            while True:
                await wake.wait()
                wake.clear()
                req, state["latest"] = state["latest"], None
                if req is None:
                    continue
                try:
                    data, meta = await loop.run_in_executor(self.pool, self.render, req)
                except Exception as e:  # the frame fails, the socket lives
                    await socket.send_text(json.dumps(dict(type="error", seq=req.get("seq"), note=str(e)[:300])))
                    print(f"  ! live {req.get('method')}: {e}", flush=True)
                    continue
                if meta["phase"] == "settle" or state["kept"] is None:
                    state["kept"] = (req, data)
                head = json.dumps(meta).encode()
                await socket.send_bytes(struct.pack("<I", len(head)) + head + data)
                if state["latest"] is not None:
                    wake.set()

        task = asyncio.create_task(worker())
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                if msg.get("type") == "keep":
                    if state["kept"]:
                        req, data = state["kept"]
                        job = self.jobs.keep(req["method"], req.get("inputs", {}), req.get("params", {}), data)
                        await socket.send_text(json.dumps(dict(type="kept", job=job)))
                    continue
                if msg.get("method") not in self.methods:
                    continue
                state["latest"] = msg
                wake.set()
        except WebSocketDisconnect:
            pass
        finally:
            task.cancel()
