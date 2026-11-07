"""AI masks: regions found by models, where masks.py otherwise reads light, depth or texture.

Five ways to find a region, each returning a map in [0, 1] the size of the picture handed in:

- person          a person-cutout model (u2net_human_seg, the ONNX file rembg keeps in ~/.u2net)
- subject         the picture's main subject, whatever it is, as a soft matte (BiRefNet matting, MIT)
- body part       a part of the people in it — their clothing, hair, skin, face, hands — from Sapiens2's
                  29-class body-part segmentation (Meta, Sapiens2 Licence: no surveillance or
                  biometric use)
- object (words)  Grounding DINO finds boxes for the words, SAM 2 cuts each box out sharply
- quality (words) CLIPSeg's soft map of how much each place matches the words — it knows moods and
                  surfaces ("wet", "shiny", "rough") as well as things, and is ranked by coverage
- object (click)  SAM 2 from points clicked on the picture, to include or to exclude
- material (click) how alike each place is to the places clicked, in DINOv2's patch features
                  (Meta, Apache 2.0; the registers model, without the stray high-norm patches
                  that punch holes in a similarity map): the same moss, rust or fabric wherever it is, whatever
                  it's on. A field, ranked by coverage like the picture's own sources

Any of the sharp cut-outs can be MATTED: its edge turned into a band of unknowns that ViTMatte
(hustvl, Apache 2.0) resolves pixel by pixel from the picture, so hair, fur and leaves come out
as soft, partial coverage instead of a smooth outline. Ask for `kind + MATTE`.

Models load on first use. A result is kept per picture and request, so dragging a mask knob over
an AI mask costs nothing after the first frame; SAM keeps each picture's embedding, so a further
click costs only its small decoder.
"""

import collections
import pathlib
import threading

import numpy as np
import torch
from PIL import Image

from loadlock import LOAD
from models import MODELS

SAM = "facebook/sam2.1-hiera-small"
DINO = "IDEA-Research/grounding-dino-tiny"
CLIPSEG = "CIDAS/clipseg-rd64-refined"
SUBJECT, SUBJECT_REV = "ZhengPeng7/BiRefNet_dynamic-matting", "074df545be"  # remote code: pinned
VITMATTE = "hustvl/vitmatte-small-composition-1k"
BODY = "facebook/sapiens2-seg-0.4b"
PARTS = {  # Sapiens2's classes, grouped as one asks for them
    "clothing": [1, 13, 23], "upper clothing": [23], "lower clothing": [13], "hair": [4],
    "face": [3, 24, 25], "skin": [3, 6, 7, 8, 11, 12, 15, 16, 17, 20, 21, 22], "hands": [6, 15],
    "arms": [7, 11, 16, 20], "legs": [8, 12, 17, 21], "shoes": [5, 9, 10, 14, 18, 19],
}
DINO_FEATURES, FEATURE_LAYERS = "facebook/dinov2-with-registers-base", (9, 12)  # a texture layer, a "what is it" layer
MATTE = " +matte"  # appended to a kind: that cut-out, matted
PERSON = pathlib.Path.home() / ".u2net" / "u2net_human_seg.onnx"
KEEP = 64  # results remembered


def words_of(text):
    """'the dress, trees' -> ['the dress', 'trees']: each phrase is looked for on its own."""
    return [w.strip() for w in str(text or "").replace(";", ",").split(",") if w.strip()]


class Segmenter:
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.lock = threading.Lock()
        self.memo = collections.OrderedDict()
        self.embeddings = collections.OrderedDict()

    def __call__(self, key, picture, kind, arg):
        """The map for `kind` on this picture. `key` names the picture (its path and size) and
        `picture` makes it when needed — a PIL image, called only on a miss."""
        with self.lock:
            return self._get(key, picture, kind, arg)

    def _get(self, key, picture, kind, arg):
        k = (key, kind, arg)
        if k in self.memo:
            self.memo.move_to_end(k)
            return self.memo[k]
        with torch.no_grad():
            if kind.endswith(MATTE):
                base = self._get(key, picture, kind[:-len(MATTE)], arg)
                out = self.matte(picture(), base)
            else:
                image = picture()
                if kind == "person":
                    out = self.person(image)
                elif kind == "subject":
                    out = self.subject(image)
                elif kind == "object (words)":
                    out = self.object_words(key, image, arg)
                elif kind == "quality (words)":
                    out = self.quality_words(image, arg)
                elif kind == "object (click)":
                    out = self.object_click(key, image, arg)
                elif kind == "material (click)":
                    out = self.material_click(key, image, arg)
                elif kind == "body part":
                    out = self.body_part(key, image, arg)
                else:
                    raise ValueError(f"no AI mask {kind!r}")
        self.memo[k] = out.astype(np.float32)
        while len(self.memo) > KEEP:
            self.memo.popitem(last=False)
        return self.memo[k]

    # ---- the models, each loaded once ----
    def model(self, name):
        with LOAD:  # loadlock.py: never beside another model's load
            got = self._model(name)
        MODELS.used("mask: " + name)
        return got

    def _model(self, name):
        if name not in self.models:
            with MODELS.loading("mask: " + name, lambda: self.models.pop(name, None), self.lock, guess=1.0):
                if name == "person":
                    import onnxruntime as ort

                    self.models[name] = ort.InferenceSession(
                        str(PERSON), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                elif name == "sam":
                    from transformers import Sam2Model, Sam2Processor

                    self.models[name] = (Sam2Processor.from_pretrained(SAM),
                                         Sam2Model.from_pretrained(SAM).to(self.device).eval())
                elif name == "dino":
                    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                    self.models[name] = (AutoProcessor.from_pretrained(DINO),
                                         AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(self.device).eval())
                elif name == "clipseg":
                    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

                    self.models[name] = (CLIPSegProcessor.from_pretrained(CLIPSEG),
                                         CLIPSegForImageSegmentation.from_pretrained(CLIPSEG).to(self.device).eval())
                elif name == "subject":
                    from transformers import AutoModelForImageSegmentation

                    self.models[name] = AutoModelForImageSegmentation.from_pretrained(
                        SUBJECT, revision=SUBJECT_REV, trust_remote_code=True).to(self.device).eval().half()
                elif name == "body":
                    from huggingface_hub import snapshot_download
                    from transformers import Sapiens2ForSemanticSegmentation, Sapiens2ImageProcessor

                    where = snapshot_download(BODY, allow_patterns=["config.json", "preprocessor_config.json",
                                                                    "model.safetensors"])
                    self.models[name] = (Sapiens2ImageProcessor.from_pretrained(where),
                                         Sapiens2ForSemanticSegmentation.from_pretrained(
                                             where, torch_dtype=torch.float16).to(self.device).eval())
                elif name == "features":
                    from transformers import AutoModel

                    self.models[name] = AutoModel.from_pretrained(DINO_FEATURES).to(self.device).eval()
                elif name == "vitmatte":
                    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

                    self.models[name] = (VitMatteImageProcessor.from_pretrained(VITMATTE),
                                         VitMatteForImageMatting.from_pretrained(VITMATTE).to(self.device).eval())
            print(f"  segmentation: {name} up")
        return self.models[name]

    # ---- the four kinds ----
    def person(self, image):
        """u2net at its 320px square, normalised as rembg does, stretched back to the picture."""
        session = self.model("person")
        x = np.asarray(image.convert("RGB").resize((320, 320), Image.LANCZOS), np.float32)
        x = x / max(float(x.max()), 1e-6)
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        got = session.run(None, {session.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
        got = (got - got.min()) / max(float(got.max() - got.min()), 1e-6)
        return np.asarray(Image.fromarray(got.astype(np.float32), "F").resize(image.size, Image.BILINEAR))

    def subject(self, image, side=1024):
        """BiRefNet's matte of the main subject, read at about 1024 px in the picture's shape."""
        model = self.model("subject")
        w, h = image.size
        k = side / max(w, h)
        sw, sh = max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)
        x = torch.from_numpy(np.asarray(image.convert("RGB").resize((sw, sh), Image.BILINEAR), np.float32) / 255.0)
        x = (x - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        got = model(x.permute(2, 0, 1)[None].to(self.device).half())[-1].sigmoid().float()
        return torch.nn.functional.interpolate(got, size=(h, w), mode="bilinear")[0, 0].cpu().numpy()

    def matte(self, image, region):
        """A cut-out's edge resolved by ViTMatte: sure inside, sure outside, and between them a
        band about 1% of the frame wide where the picture decides how much is the thing."""
        from masks import grow

        if not (region > 0.5).any():
            return region
        processor, model = self.model("vitmatte")
        band = max(4.0, 0.012 * max(image.size))
        hard = (region > 0.5).astype(np.float32)
        trimap = np.full(region.shape, 128, np.uint8)
        trimap[grow(hard, -band) > 0.5] = 255
        trimap[grow(hard, band) < 0.5] = 0
        inputs = processor(images=image.convert("RGB"), trimaps=Image.fromarray(trimap), return_tensors="pt").to(self.device)
        h, w = region.shape
        return model(**inputs).alphas[0, 0, :h, :w].float().cpu().numpy()

    def features(self, key, image, side=896):
        """DINOv2's patch features of the picture (14 px patches at about 896 px), unit length,
        one grid per layer read; computed once per picture, like SAM's embedding."""
        k = ("features", key)
        if k not in self.embeddings:
            model = self.model("features")
            w, h = image.size
            f = side / max(w, h)
            sw, sh = max(14, round(w * f / 14) * 14), max(14, round(h * f / 14) * 14)
            x = torch.from_numpy(np.asarray(image.convert("RGB").resize((sw, sh), Image.BICUBIC), np.float32) / 255.0)
            x = (x - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
            states = model(pixel_values=x.permute(2, 0, 1)[None].to(self.device), output_hidden_states=True).hidden_states
            first = 1 + getattr(model.config, "num_register_tokens", 0)  # the class token, then the registers
            self.embeddings[k] = [torch.nn.functional.normalize(states[n][0, first:].float(), dim=-1)
                                  .reshape(sh // 14, sw // 14, -1) for n in FEATURE_LAYERS]
            while len(self.embeddings) > 8:
                self.embeddings.popitem(last=False)
        self.embeddings.move_to_end(k)
        return self.embeddings[k]

    def body_part(self, key, image, part):
        """How likely each place is to be this part of a person: Sapiens2's class probabilities
        (read once per picture, at its 1024×768) summed over the part's classes."""
        k = ("body", key)
        if k not in self.embeddings:
            processor, model = self.model("body")
            inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
            logits = model(pixel_values=inputs["pixel_values"].half()).logits.float()
            self.embeddings[k] = logits.softmax(1)[0].half()
            while len(self.embeddings) > 8:
                self.embeddings.popitem(last=False)
        self.embeddings.move_to_end(k)
        prob = self.embeddings[k][PARTS.get(part, PARTS["clothing"])].float().sum(0)
        w, h = image.size
        return torch.nn.functional.interpolate(prob[None, None], size=(h, w), mode="bilinear")[0, 0].cpu().numpy()

    def material_click(self, key, image, points):
        """How alike each place is to the clicked places (a 3×3-patch neighbourhood each), both
        layers counted: the best match among the clicks to include, less the best among those to
        leave out. Higher is more alike; coverage decides how much is taken."""
        pts = [(float(x), float(y), int(lab)) for x, y, lab in (points or ())]
        w, h = image.size
        if not any(lab for _, _, lab in pts):
            return np.zeros((h, w), np.float32)
        grids = self.features(key, image)
        gh, gw = grids[0].shape[:2]

        def alike(x, y):
            i, j = min(int(y * gh), gh - 1), min(int(x * gw), gw - 1)
            return sum(g @ torch.nn.functional.normalize(g[max(0, i - 1):i + 2, max(0, j - 1):j + 2].reshape(-1, g.shape[-1])
                                                         .mean(0), dim=0) for g in grids) / len(grids)

        got = torch.stack([alike(x, y) for x, y, lab in pts if lab]).max(0).values
        out = [alike(x, y) for x, y, lab in pts if not lab]
        if out:
            got = got - torch.stack(out).max(0).values.clamp(min=0)
        got = got.cpu().numpy().astype(np.float32)
        return np.asarray(Image.fromarray(got, "F").resize((w, h), Image.BICUBIC))

    def sam_embedding(self, key, image):
        """SAM's view of the picture, computed once per picture: every prompt after it is cheap."""
        processor, model = self.model("sam")
        if key not in self.embeddings:
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            self.embeddings[key] = model.get_image_embeddings(inputs["pixel_values"])
            while len(self.embeddings) > 8:
                self.embeddings.popitem(last=False)
        self.embeddings.move_to_end(key)
        return self.embeddings[key]

    def sam(self, key, image, multimask, **prompts):
        """SAM 2's masks for these prompts, each brought to the picture's size, as probabilities."""
        processor, model = self.model("sam")
        inputs = processor(images=image, return_tensors="pt", **prompts).to(self.device)
        out = model(image_embeddings=self.sam_embedding(key, image), multimask_output=multimask,
                    **{k: inputs[k] for k in ("input_points", "input_labels", "input_boxes") if k in inputs})
        logits = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                              binarize=False)[0]  # (objects, masks, H, W)
        best = out.iou_scores[0].argmax(-1).cpu()  # the mask SAM itself rates best, per object
        chosen = logits[torch.arange(logits.shape[0]), best]
        return torch.sigmoid(chosen.float()).max(0).values.numpy()

    def object_click(self, key, image, points):
        """SAM 2 from clicks: (x, y, include) in 0–1 of the frame. Nothing to include, nothing found."""
        pts = [(float(x), float(y), int(lab)) for x, y, lab in (points or ())]
        w, h = image.size
        if not any(lab for _, _, lab in pts):
            return np.zeros((h, w), np.float32)
        return self.sam(key, image, multimask=len(pts) == 1,
                        input_points=[[[[x * w, y * h] for x, y, _ in pts]]],
                        input_labels=[[[lab for _, _, lab in pts]]])

    def object_words(self, key, image, text):
        """Grounding DINO's boxes for each phrase, each cut out by SAM 2, all of them together."""
        labels = words_of(text)
        w, h = image.size
        if not labels:
            return np.zeros((h, w), np.float32)
        processor, model = self.model("dino")
        inputs = processor(images=image, text=[labels], return_tensors="pt").to(self.device)
        found = processor.post_process_grounded_object_detection(
            model(**inputs), inputs["input_ids"], threshold=0.3, text_threshold=0.25, target_sizes=[(h, w)])[0]
        boxes = found["boxes"].cpu().tolist()
        if not boxes:
            return np.zeros((h, w), np.float32)
        return self.sam(key, image, multimask=False, input_boxes=[boxes])

    def quality_words(self, image, text):
        """CLIPSeg's map for each phrase (it sees a 352px square), the strongest phrase kept."""
        labels = words_of(text)
        w, h = image.size
        if not labels:
            return np.zeros((h, w), np.float32)
        processor, model = self.model("clipseg")
        inputs = processor(text=labels, images=[image] * len(labels), padding=True, return_tensors="pt").to(self.device)
        logits = model(**inputs).logits
        logits = logits.reshape(len(labels), *logits.shape[-2:])
        got = torch.sigmoid(logits).max(0).values.float().cpu().numpy()
        return np.asarray(Image.fromarray(got, "F").resize((w, h), Image.BILINEAR))
