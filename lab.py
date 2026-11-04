"""Launch Latent Lab: every method, every knob, one page.

    python lab.py --space texture-ip-sd15.npz
    python lab.py --space texture-ip-yellow3-xl.npz --preset sdxl --concepts concepts-yellow3-xl.json --port 8021
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ip_space import PRESETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space", required=True)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--textures", default=None)
    ap.add_argument("--presets", default="presets.json")
    ap.add_argument("--concepts", default="concepts.json")
    ap.add_argument("--preset", default="sd15", choices=sorted(PRESETS))
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--csize", type=int, default=384)
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    here = pathlib.Path(__file__).resolve().parent
    for key in ("presets", "concepts"):
        val = pathlib.Path(getattr(args, key))
        if not val.is_absolute():
            setattr(args, key, str(here / val))
    beside = pathlib.Path(args.space).parent
    dataset = here / "dataset"
    args.sources = args.sources or str(dataset / "content" if (dataset / "content").is_dir()
                                       else beside / "restyle-sources")
    args.textures = args.textures or str(dataset / "textures" if (dataset / "textures").is_dir()
                                         else beside / "poetics" / "textures")

    import uvicorn

    from app.server import create_app

    print(f"\n  Latent Lab · http://127.0.0.1:{args.port}\n")
    uvicorn.run(create_app(args), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
