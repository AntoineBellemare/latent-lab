"""Hold on to the snapshot that sits closest to the data, because the trainer overwrites its export.

    python keep_best.py --watch C:/Users/Antoine/goofi-models/textures-raw.onnx

Scores each new export on three colour statistics against the real set and keeps the best pair of
files seen. A GAN wanders; without this the good pass is gone by the time you have measured it.
"""

import argparse
import pathlib
import shutil
import time

import numpy as np

from diversity import drawn, open_model, real, sharpness

def look(frames):
    """Colour AND detail. Scored on colour alone, a run picks a snapshot that matches the palette
    exactly and carries half the data's detail - measured 0.47 against 0.82 later in the same run."""
    sat = (frames.max(axis=3) - frames.min(axis=3)).mean(axis=(1, 2)).mean()
    hue = frames.mean(axis=(1, 2))
    return (sat, np.linalg.norm(hue - hue.mean(0), axis=1).mean(),
            frames.mean(axis=(1, 2, 3)).std(), sharpness(frames))


def lattice(frames, ring=4):
    """How far the stem's own period stands above its spectral ring."""
    out = []
    for f in frames:
        g = f.mean(axis=2)
        P = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
        mid = np.array(P.shape) // 2
        y, x = np.indices(P.shape)
        b = P[np.hypot(y - mid[0], x - mid[1]).astype(int) == ring]
        out.append(float(b.max() / (np.median(b) + 1e-12)))
    return float(np.median(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--watch", required=True)
    ap.add_argument("--tag", default="best", help="Suffix for the kept copies.")
    ap.add_argument("--beat", type=float, help="Only keep a snapshot scoring below this.")
    ap.add_argument("--images", default="G:/Antoine/PHOTOGRAPHIE/goofi-training-set")
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--ceiling", type=float, default=2.0, help="Reject a snapshot tiling this many times the data's own.")
    args = ap.parse_args()

    watch = pathlib.Path(args.watch)
    best, seen = args.beat, None
    truth, floor = None, None
    while True:
        stamp = watch.stat().st_mtime if watch.is_file() else None
        if stamp and stamp != seen:
            # The size settling is what says the export finished; there is no other signal.
            size = watch.stat().st_size
            time.sleep(20)
            if watch.stat().st_size != size:
                continue
            seen = stamp
            try:
                session, width = open_model(watch)
                fake = drawn(session, width, args.draws, seed=11)
                if truth is None:
                    seen_real = real(args.images, fake.shape[1], 96, seed=11)
                    truth, floor = look(seen_real), lattice(seen_real)
                now, grid = look(fake), lattice(fake) / floor
                score = sum(abs(a / b - 1.0) for a, b in zip(now, truth))
                mark = "rejected: tiling" if grid > args.ceiling else ""
                if not mark and (best is None or score < best):
                    for who in (watch, watch.with_name(watch.name.replace("-raw", ""))):
                        if who.is_file():
                            shutil.copy2(who, who.with_name(who.stem + "-" + args.tag + who.suffix))
                    best, mark = score, "KEPT"
                print(f"  score {score:.3f} lattice {grid:.1f}x  sat {now[0]:.4f} colour {now[1]:.4f} "
                      f"bright {now[2]:.4f} sharp {now[3]:.4f}  {mark}", flush=True)
            except Exception as e:
                print(f"  could not read the export: {e}", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
