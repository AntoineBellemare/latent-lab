"""The lab as one app: a FastAPI backend generated from the registry, over the explorer's engine.

    python lab.py --space texture-ip-sd15.npz
    python lab.py --space texture-ip-yellow3-xl.npz --preset sdxl --concepts concepts-yellow3-xl.json

Everything the old explorer knew — the archive, uploads, materials, concepts, the pipelines — is
reused as-is through its Engine; what is new is the shape around it: one method registry, one job
queue with progress over a WebSocket, a history that remembers every render with its settings, and
recipes. The client at `/` is a single page generated from `/api/menu`.
"""

import asyncio
import collections
import json
import mimetypes
import pathlib

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import threading

import numpy as np

from .jobs import Hub, Jobs
from .live import Live
from .registry import build, public, GROUPS, LAYOUT

LAB = pathlib.Path(__file__).resolve().parent.parent
FRAMES_BYTES = 1_200_000_000  # the photographs kept decoded, as floats, for the sliders

# Windows reads these from the registry, where .js is often text/plain — and a browser refuses to
# run a module served as text/plain. Say it explicitly.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class LabEngine:
    """explore.Engine plus the few views the registry's run functions need."""

    def __new__(cls, args):
        import explore

        engine = explore.Engine(args)
        engine.__class__ = type("LabEngine", (explore.Engine,), dict(
            texture_path=cls.texture_path, content_path=cls.content_path, vgg=cls.vgg, query=cls.query,
            gan_session=cls.gan_session, draw_gan=cls.draw_gan, load_frame=cls.load_frame,
            depth_arr=cls.depth_arr, geometry_arr=cls.geometry_arr, intrinsics_arr=cls.intrinsics_arr,
            intrinsics_of=cls.intrinsics_of, rae=cls.rae, enlarged=cls.enlarged, tokendit=cls.tokendit,
            segment=cls.segment, transfer=cls.transfer, editor=cls.editor, make_room=cls.make_room))
        engine._vgg = None
        engine._gans = {}
        engine._gan_lock = threading.Lock()
        engine._arrs = collections.OrderedDict()
        engine._arr_lock = threading.Lock()
        engine._segmenter = None
        engine._geometry = None
        engine._intrinsics = None
        engine._transfer = None
        engine._editor = None
        engine._raes = {}
        engine._enlarger = None
        engine._tokendit = None
        engine.rae_lock = threading.RLock()  # one frame at a time on the decoder, from any socket
        from models import MODELS

        MODELS.add("photo frames", engine._arrs.clear, engine._arr_lock)
        return engine

    def rae(self, decoder):
        """RAE's decoder and DINOv2's encoder (rae.py), or RAEv2's and DINOv3's (raev2.py), loaded on first
        use: one to two gigabytes in half precision, one per decoder (the 512 composes, the 256 draws the
        generator's grids, RAEv2's composes in DINOv3). FLUX.2's VAE (fluxvae.py) is the token space
        named by `fluxvae.KEY` in place of a path."""
        import fluxvae
        import rae
        from models import MODELS

        name = "tokens: FLUX.2 VAE" if decoder == fluxvae.KEY else "tokens: " + pathlib.Path(decoder).parent.name
        with self.loading:
            got = self._raes.get(decoder)
            if got is None:
                with MODELS.loading(name, lambda: self._raes.pop(decoder, None), self.rae_lock, guess=2.5):
                    got = fluxvae.FluxVAE(str(self.device)) if decoder == fluxvae.KEY else rae.load(decoder, str(self.device))
                self._raes[decoder] = got
                print("  " + name + " up")
        MODELS.used(name)
        return got

    def tokendit(self, folder):
        """The token generator (tokendit.py) at its newest checkpoint in the folder (or its published
        .safetensors), read again when a training run still going writes a newer one: (checkpoint, model,
        mean, std, step)."""
        import tokendit

        from models import MODELS

        newest = max(pathlib.Path(folder).glob("ema-*.pt"), default=None) or max(pathlib.Path(folder).glob("*.safetensors"))
        with self.loading:
            got = self._tokendit
            if got is None or got[0] != newest:
                with MODELS.loading("token generator", lambda: setattr(self, "_tokendit", None), self.rae_lock, guess=1.0):
                    model, mean, std, step = tokendit.load(newest, self.device)
                got = self._tokendit = (newest, model, mean, std, step)
                print(f"  tokendit up: step {step}")
        MODELS.used("token generator")
        return got

    def enlarged(self, weights, image, keep=False):
        """A picture made four times larger by Real-ESRGAN (upscale.py), loaded on first use. At 3072 x 2048
        its activations take a few gigabytes, so the editor's image model steps off the card first.
        `keep`: the result stays on the card as a tensor, for a caller that finishes it there."""
        import upscale
        from models import MODELS

        with self.loading:
            net = self._enlarger
            if net is None:
                with MODELS.loading("enlarge: Real-ESRGAN", lambda: setattr(self, "_enlarger", None), self.rae_lock, guess=1.0):
                    net = self._enlarger = upscale.load(weights, self.device.type)
                print("  real-esrgan up")
        MODELS.used("enlarge: Real-ESRGAN")
        self.make_room()
        with self.rae_lock:
            return upscale.enlarge(net, image, keep)

    def segment(self, name, kind, arg, width, height):
        """An AI mask (segment.py) for this picture, in this frame. The models see the picture at
        1024px at most, once per request: the answer is kept and resized to whatever frame asks,
        so a drag, a release and an export share it."""
        from PIL import Image

        from composite import _resize
        from segment import Segmenter

        if kind.startswith("intrinsic:"):
            # The picture's colours, light or shine (intrinsics.py), for the mask's own fields.
            return self.intrinsics_arr(name, width, height)[kind.split(":", 1)[1]]
        with self.loading:
            if self._segmenter is None:
                self._segmenter = Segmenter(self.device)
        cpath = self.content_path(name)
        w, h = self.frame(cpath, min(1024, max(self.original_size(cpath))))
        got = self._segmenter(f"{cpath}@{w}x{h}", lambda: self.load_image(cpath, w, h), kind, arg)
        return got if (w, h) == (width, height) else _resize(got, int(width), int(height), Image.BILINEAR)

    def depth_arr(self, name, width, height):
        """The picture's depth as floats in this frame, nearer is higher. Estimated at 768px at most
        and resized: a depth model has nothing to say at 24 megapixels that it can't say at 0.6."""
        from PIL import Image

        from composite import _resize

        cpath = self.content_path(name)
        w0, h0 = self.frame(cpath, min(max(width, height), 768))
        shown = self.depth(name, w0, h0)  # estimated on the first ask, kept after
        d = self.__dict__.get("depth_floats", {}).get((name, w0, h0))
        if d is None:
            d = np.asarray(shown.convert("L"), np.float32) / 255.0
        return d if (w0, h0) == (width, height) else _resize(d, width, height, Image.BICUBIC)

    def geometry_arr(self, name, width, height):
        """The picture's surfaces (geometry.py: normals, depth in metres, focal length) in this
        frame. Read once per picture at 768 px at most, so a drag, a release and an export share
        one reading."""
        import geometry

        with self.loading:
            if self._geometry is None:
                self._geometry = geometry.Geometry(self.device)
        cpath = self.content_path(name)
        w0, h0 = self.frame(cpath, min(geometry.SIDE, max(self.original_size(cpath))))
        got = self._geometry(f"{cpath}@{w0}x{h0}", lambda: self.load_image(cpath, w0, h0))
        return geometry.resized(got, int(width), int(height))

    def transfer(self):
        """The SDXL material-transfer pipeline (zest.py), loaded on first use."""
        import zest

        with self.loading:  # one instance, however many threads ask first
            if self._transfer is None:
                self._transfer = zest.Transfer(self.device, pathlib.Path(self.args.space).parent)
        return self._transfer

    def make_room(self):
        """A big pipeline is about to run: the editor's image model steps off the card, which would
        otherwise overflow into shared memory and slow everything to a crawl (the driver pages rather
        than failing)."""
        if self._editor is not None:
            self._editor.park()

    def editor(self):
        """The FLUX.2 klein editor (editor.py), loaded on first use."""
        import editor

        with self.loading:
            if self._editor is None:
                self._editor = editor.Editor(self.device)
        return self._editor

    def intrinsics_arr(self, name, width, height):
        """The picture's albedo, shading and shine (intrinsics.py) in this frame, read once per
        picture at 768 px at most."""
        import intrinsics

        return self.intrinsics_of(self.content_path(name), width, height)

    def intrinsics_of(self, path, width, height):
        """Any photograph's albedo, shading and shine (a picture's, or a texture's)."""
        import intrinsics

        with self.loading:
            if self._intrinsics is None:
                self._intrinsics = intrinsics.Intrinsics(self.device)
        w0, h0 = self.frame(path, min(intrinsics.SIDE, max(self.original_size(path))))
        got = self._intrinsics(f"{path}@{w0}x{h0}", lambda: self.load_image(path, w0, h0))
        return intrinsics.resized(got, int(width), int(height))

    def load_frame(self, path, width, height):
        """A photograph covering this frame, as floats — decoded once, then served from memory.

        Decoding a 6000px JPEG off the photo drive costs more than most mapping modes themselves;
        a slider that redraws twenty times a second must not pay it twenty times. The frames kept
        are bounded by their bytes (FRAMES_BYTES), up to the 3072 px a release lays the picture's own
        pixels at; export-sized frames are not kept. The memory manager can let go of all of them.
        """
        from composite import arr
        from models import MODELS

        key = (str(path), int(width), int(height))
        MODELS.used("photo frames")
        with self._arr_lock:
            if key in self._arrs:
                self._arrs.move_to_end(key)
                return self._arrs[key]
        got = arr(self.load_image(path, int(width), int(height)))
        got.setflags(write=False)
        if width * height <= 7_000_000:
            with self._arr_lock:
                self._arrs[key] = got
                while len(self._arrs) > 1 and sum(a.nbytes for a in self._arrs.values()) > FRAMES_BYTES:
                    self._arrs.popitem(last=False)
        return got

    def gan_session(self, rel):
        """One onnxruntime session per GAN, loaded on first use; CPU is fast enough to drag."""
        with self._gan_lock:
            if rel not in self._gans:
                import onnxruntime

                path = pathlib.Path(self.args.space).parent / rel
                ready = onnxruntime.get_available_providers()
                order = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ready]
                sess = onnxruntime.InferenceSession(str(path), providers=order)
                self._gans[rel] = (sess, threading.Lock())
                print(f"  gan up: {rel} ({sess.get_providers()[0]})")
            return self._gans[rel]

    def draw_gan(self, spec, params, look=None):
        from PIL import Image

        sess, lock = self.gan_session(spec["file"])
        zin = sess.get_inputs()[0]
        width = int(zin.shape[-1])

        def base(seed):
            return np.random.default_rng(int(seed)).standard_normal(width).astype(np.float32)

        za, zb = base(params["seed"]), base(params["seedB"])
        t = float(params.get("t", 0))
        if t > 0:
            # Direction on the sphere, magnitude straight, as play.py morphs: a plain lerp passes
            # near the origin, which is the model's mean, and the middle of the walk washes out.
            na, nb = np.linalg.norm(za), np.linalg.norm(zb)
            om = np.arccos(np.clip((za / na) @ (zb / nb), -1, 1))
            if om > 1e-4:
                z = (np.sin((1 - t) * om) * za / na + np.sin(t * om) * zb / nb) / np.sin(om) * ((1 - t) * na + t * nb)
            else:
                z = (1 - t) * za + t * zb
        else:
            z = za
        z = z * float(params["spread"])
        for k in range(min(width, 64)):
            key = f"axis {k}"
            if key in params:
                z[k] += float(params[key])
        feed = {zin.name: z[None].astype(np.float32)}
        if look is not None:
            feed[sess.get_inputs()[1].name] = np.asarray(look, np.float32)[None]
        elif len(sess.get_inputs()) > 1:
            cin = sess.get_inputs()[1]
            cat = np.array([float(params.get(name, 0)) for name in spec["categories"]], np.float32)
            if cat.sum() <= 0:
                cat[:] = 1.0 / len(cat)
            feed[cin.name] = cat[None]
        with lock:
            out = sess.run(None, feed)[0]
        a = np.asarray(out, np.float32)[0].transpose(1, 2, 0) * 0.5 + 0.5
        return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))

    def texture_path(self, name):
        if name.startswith("i:"):
            return self.files[int(name[2:])]
        if name.startswith("u:"):
            return self.uploads[name[2:]]["path"]
        p = pathlib.Path(self.args.textures) / name
        if not p.is_file() and (pathlib.Path(self.args.sources) / name).is_file():
            return pathlib.Path(self.args.sources) / name  # a picture shown as the swatch (the hidden tab)
        return p

    def content_path(self, name):
        """A picture by name — or, for the hidden tab, where the editor redraws the texture itself, a
        texture or an archive photograph standing in as the picture."""
        if name.startswith("i:"):
            return self.files[int(name[2:])]
        if name.startswith("u:"):
            return self.uploads[name[2:]]["path"]
        p = pathlib.Path(self.args.sources) / name
        if not p.is_file() and (pathlib.Path(self.args.textures) / name).is_file():
            return pathlib.Path(self.args.textures) / name
        return p

    def vgg(self):
        from models import MODELS

        with self.loading:
            net = self._vgg
            if net is None:
                from torchvision.models import VGG19_Weights, vgg19

                with MODELS.loading("stylize: VGG-19", lambda: setattr(self, "_vgg", None), guess=1.0):
                    net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
                    for p in net.parameters():
                        p.requires_grad_(False)
                self._vgg = net
                print("  vgg up")
        MODELS.used("stylize: VGG-19")
        return net

    def query(self, inputs, params, kind):
        """The registry's typed request, folded into the parse_qs shape the engine speaks."""
        q = {}

        def put(key, value):
            if value not in (None, "", []):
                q[key] = [str(value)]

        def material(spec, prefix=""):
            spec = spec or {}
            for k in ("point", "photo", "upmat", "texfile", "terms", "preset"):
                put(prefix + k, spec.get(k))

        a = inputs.get("materialA") or {}
        material(a)
        material(a, "a_")  # walks read prefixed ends; harmless elsewhere
        material(inputs.get("materialB"), "b_")
        put("push", a.get("push", 8))
        put("against", a.get("against"))
        for name, t in (a.get("concepts") or {}).items():
            if float(t):
                put("c_" + name, t)
        put("content", inputs.get("content"))
        put("q", params.get("quality"))
        for k in ("scale", "seed", "t", "structure", "strength", "rthresh", "then_s", "then",
                  "prompt", "lora", "lora_w", "aspect"):
            put(k, params.get(k))
        put("size", params.get("_size") or params.get("size"))  # _size: smaller while a slider drags
        if params.get("_drag"):
            put("drag", "1")
        if params.get("region") and params["region"] != "off":
            put("region", params["region"])
        return q


def warm_imports():
    """The libraries' lazy parts resolved once, in one thread, before any request: transformers and
    diffusers build their classes on first touch, and two threads touching at once (a job and a
    page's live socket reconnecting after a restart) can each meet a half-built module — an
    ImportError for a class that exists."""
    import diffusers
    import transformers

    for name in ("CLIPVisionModelWithProjection", "CLIPTextModel", "CLIPTextModelWithProjection", "CLIPTokenizer",
                 "DPTForDepthEstimation", "DPTImageProcessor", "Qwen2TokenizerFast", "Qwen3ForCausalLM",
                 "AutoModelForImageSegmentation", "AutoImageProcessor", "AutoModel", "AutoProcessor"):
        getattr(transformers, name, None)
    for name in ("AutoPipelineForText2Image", "StableDiffusionControlNetImg2ImgPipeline",
                 "StableDiffusionXLControlNetInpaintPipeline", "StableDiffusionXLControlNetImg2ImgPipeline",
                 "Flux2KleinInpaintPipeline", "ControlNetModel", "AutoencoderKL", "AutoencoderTiny"):
        try:
            getattr(diffusers, name, None)
        except Exception:  # a pipeline this diffusers lacks is simply never asked for
            pass


def create_app(args):
    import torch

    from models import MODELS

    # TensorFloat-32 for float32 matrix products and convolutions (MoGe, the stylize VGG, the VAE's
    # float32 statistics): several times faster on this card, a precision these images never show.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    warm_imports()
    app = FastAPI(title="Latent Lab")
    engine = LabEngine(args)
    hub = Hub()
    root = pathlib.Path(args.space).parent
    manifest_file = LAB / "models.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
    manifest["gans"] = [g for g in manifest.get("gans", []) if (root / g["file"]).is_file()]
    methods = build(args.preset, root, manifest)
    outdir = root / "poetics" / "app"
    jobs = Jobs(engine, outdir, hub, methods, exports=root / "exports")
    live = Live(engine, methods, jobs, args.preset)
    recipes_file = LAB / "recipes.json"

    @app.middleware("http")
    async def fresh_page(request, call_next):
        # The page is edited while the app runs; a cached app.js is how a fixed bug stays visible.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.websocket("/api/live")
    async def live_socket(socket: WebSocket):
        await live.serve(socket)

    @app.on_event("startup")
    async def startup():
        hub.attach(asyncio.get_running_loop())
        MODELS.start()
        # The first frame of the tokens tab and of the lab's GAN should not wait for a load: their
        # models come up in the background while the page opens (a few gigabytes, let go when idle).
        threading.Thread(target=warm, name="warm", daemon=True).start()

    def warm():
        try:
            if manifest.get("rae") and (root / manifest["rae"]).is_file():
                engine.rae(str(root / manifest["rae"]))
            if manifest.get("gans"):
                engine.gan_session(manifest["gans"][0]["file"])
        except Exception as e:
            print("  warm-up:", e, flush=True)

    @app.get("/api/memory")
    def memory():
        """What the lab holds: each model's size and idle time, the free commit and card memory, and
        what was let go lately (models.py)."""
        return MODELS.report()

    @app.get("/api/menu")
    def menu():
        return dict(methods=public(methods), presets=list(engine.presets), concepts=list(engine.concepts),
                    contents=engine.contents, textures=engine.textures, uploads=sorted(engine.uploads),
                    archive=len(engine.files), preset=args.preset, size=args.size,
                    history=jobs.history[-120:], captions=sum(1 for c in engine.caption_list() if c),
                    exports=str(root / "exports"), groups=GROUPS, layout=LAYOUT)

    @app.get("/api/archive")
    def archive(term: str = "", sort: str = "", page: int = 0):
        return engine.archive({"term": [term], "sort": [sort], "page": [str(page)]})

    @app.get("/api/thumb")
    def thumb(i: int):
        return Response(engine.thumb(i), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/upfile")
    def upfile(name: str):
        return Response(engine.uploads[name]["path"].read_bytes(), media_type="image/jpeg")

    @app.get("/api/source")
    def source(kind: str, name: str):
        where = engine.content_path(name) if kind == "content" else engine.texture_path(name)
        from composite import fit
        import io

        buf = io.BytesIO()
        fit(where, 192).save(buf, format="JPEG", quality=85)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/picture")
    def picture(name: str, size: int = 1024, kind: str = "content"):
        """A picture in its own shape, framed as a result of that size is (the editor's before)."""
        import io

        where = engine.content_path(name) if kind == "content" else engine.texture_path(name)
        im = engine.load_image(where, *engine.frame(where, max(64, min(int(size), 2048)), 16))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/upload")
    async def upload(request: Request, name: str = "upload.png"):
        stored = engine.add_upload(name, await request.body())
        return {"name": stored}

    @app.post("/api/jobs")
    async def submit(request: Request):
        body = await request.json()
        return jobs.submit(body["method"], body.get("inputs", {}), body.get("params", {}), body.get("cid"))

    @app.get("/api/jobs/{jid}")
    def job_state(jid: str):
        return jobs.get(jid) or {"id": jid, "status": "unknown"}

    @app.get("/api/exports/{name}")
    def exported(name: str):
        return FileResponse(jobs.exports / pathlib.Path(name).name)

    @app.delete("/api/jobs/{jid}")
    def cancel(jid: str):
        jobs.cancel(jid)
        return {"ok": True}

    @app.get("/api/out/{name}")
    def out(name: str):
        return FileResponse(outdir / pathlib.Path(name).name)

    @app.get("/api/recipes")
    def recipes():
        return json.loads(recipes_file.read_text(encoding="utf-8")) if recipes_file.is_file() else {}

    @app.post("/api/recipes")
    async def save_recipe(request: Request):
        body = await request.json()
        book = json.loads(recipes_file.read_text(encoding="utf-8")) if recipes_file.is_file() else {}
        book[body["name"]] = body["recipe"]
        recipes_file.write_text(json.dumps(book, indent=1), encoding="utf-8")
        return {"ok": True}

    @app.websocket("/api/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        q = asyncio.Queue()
        hub.queues.add(q)
        try:
            while True:
                await socket.send_text(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            hub.queues.discard(q)

    app.mount("/", StaticFiles(directory=LAB / "app" / "static", html=True), name="static")
    return app
