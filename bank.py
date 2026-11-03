"""Put named materials in front of a space-conditioned GAN, so a patch drives it like the category model.

    python bank.py --model looks.onnx --space texture-ip-sd15.npz --out looks-presets.onnx
    python bank.py --model looks.onnx --space texture-ip-sd15.npz --textures dataset/textures --no-presets --out looks-photos.onnx

`fastgan.py --look` exports a generator that takes `z` and a 1024-wide place in the lab's space. A
patch rarely holds such a vector. This folds a bank of named places into the front of the model, so
it takes `z` and `category` — one weight per named material — which is the contract the category
models already have: the lab's GAN code, goofi's `Onnx` node, `morph.py` and `sheet.py` all read it
unchanged. The weights are shares, as there, and the mix is made the way the lab makes one: a
weighted sum of places, of which only the direction is read.

The places are computed the way the lab computes them, so a name here means what it means there: a
preset is `terms.term_vector` at the lab's push of 8, a photograph is four crops of half its short
side through the adapter's encoder. `<out>.json` lists the names in input order, and `--places`
writes each place as its own `.npy`, for a patch that would rather hold the vectors itself and feed
the bare model's `look`: a `Weights` node per file and a sum, as the NCA rules are blended.
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from PIL import Image
from torch import nn

import terms
from lora import images

PUSH = 8.0  # the lab's push for a preset (explore.Engine.base_material)


class Front(nn.Module):
    def __init__(self, bank):
        super().__init__()
        self.register_buffer("bank", bank)

    def forward(self, category):
        share = category.clamp_min(0.0)
        total = share.sum(1, keepdim=True)
        # Nothing wired yet sends zeros: every material at once, as the category models read it.
        share = torch.where(total > 0, share / total.clamp_min(1e-6), torch.full_like(share, 1.0 / share.shape[1]))
        return share @ self.bank


def photo_place(eye, path):
    """explore.Engine.file_vec: four crops of half the short side, their mean."""
    with Image.open(path) as im:
        im.draft("RGB", (1024, 1024))
        im = im.convert("RGB")
    half = max(1, min(im.size) // 2)
    crops = []
    for cy, cx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        left, top = cx * (im.width - half), cy * (im.height - half)
        crops.append(np.asarray(im.crop((left, top, left + half, top + half)).resize((224, 224), Image.LANCZOS), np.float32) / 255)
    return eye(crops).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="An .onnx from `fastgan.py --look`.")
    ap.add_argument("--space", required=True, help="The lab's space, for the presets' terms.")
    ap.add_argument("--presets", default="presets.json")
    ap.add_argument("--no-presets", action="store_true", help="Photographs only.")
    ap.add_argument("--textures", help="A folder of photographs to add, each one material, named by its file.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--places", help="A folder to write each material's place into, as <name>.npy.")
    args = ap.parse_args()

    import onnx
    import onnx.compose

    space = dict(np.load(args.space, allow_pickle=True))
    names, places = [], []
    if not args.no_presets:
        found = terms.index(space)
        for name, mix in json.loads(pathlib.Path(args.presets).read_text(encoding="utf-8")).items():
            if not name.startswith("_"):
                names.append(name)
                places.append(terms.term_vector(space, dict(mix), found, PUSH))
    if args.textures:
        from follows import Eye

        eye = Eye(str(space["preset"]), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for f in images(args.textures):
            names.append(f.stem)
            places.append(photo_place(eye, f))
    bank = np.stack(places).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    print(f"{len(names)} materials: {', '.join(names)}")
    if args.places:
        where = pathlib.Path(args.places)
        where.mkdir(parents=True, exist_ok=True)
        for n, v in zip(names, bank):
            np.save(where / f"{n.replace(' ', '-')}.npy", v.astype(np.float32))

    gan = onnx.load(args.model)
    z, look = gan.graph.input[0], gan.graph.input[1]
    out = pathlib.Path(args.out)
    front_path = out.with_name(out.stem + "-front.onnx")
    torch.onnx.export(Front(torch.from_numpy(bank)), (torch.zeros(1, len(names)),), str(front_path),
                      input_names=["category"], output_names=["mixed"], opset_version=17, dynamo=False)
    # Both graphs were traced by torch and name their edges alike; the front's go under a prefix.
    front = onnx.compose.add_prefix(onnx.load(str(front_path)), "bank/", rename_inputs=False, rename_outputs=False)
    front_path.unlink()
    front.ir_version = gan.ir_version
    merged = onnx.compose.merge_models(front, gan, io_map=[("mixed", look.name)])
    # `z` first and `category` second, the order the category models have and their readers assume.
    order = sorted(merged.graph.input, key=lambda i: i.name != z.name)
    del merged.graph.input[:]
    merged.graph.input.extend(order)
    onnx.save(merged, str(out))
    out.with_suffix(".json").write_text(json.dumps({"materials": names}, indent=1), encoding="utf-8")

    import onnxruntime

    a, b = onnxruntime.InferenceSession(args.model), onnxruntime.InferenceSession(str(out))
    zz = np.random.default_rng(0).standard_normal((1, z.type.tensor_type.shape.dim[1].dim_value)).astype(np.float32)
    w = np.zeros((1, len(names)), np.float32)
    w[0, 0], w[0, -1] = 0.7, 0.3
    want = a.run(None, {z.name: zz, look.name: (0.7 * bank[0] + 0.3 * bank[-1])[None]})[0]
    got = b.run(None, {z.name: zz, "category": w})[0]
    print(f"wrote {out} and {out.with_suffix('.json').name}; matches the bare model to {np.abs(got - want).max():.1e}")


if __name__ == "__main__":
    main()
