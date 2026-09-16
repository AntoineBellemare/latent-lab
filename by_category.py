"""Fold a `categories.csv` into one folder per category, as hardlinks: no copy, no second set.

    python by_category.py --images ~/pictures/set --csv set-categories/categories.csv --out ~/pictures/set-by-category

A trainer then reads one category with `--images .../cat3`, and every category at once with
`--images .../`, which takes each image's class from the folder that holds it. Every tool here that
wants a reference set — `diversity.py`, `keep_best.py` — takes one of these folders as it is.

Hardlinks because the set is gigabytes and a second copy of it buys nothing. Beside the set and
never inside it: a trainer walks its root with `rglob` and would read the links back in as extra
training data, which is a thing that has already happened once.
"""

import argparse
import collections
import csv
import os
import pathlib
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="The set the CSV names.")
    ap.add_argument("--csv", required=True, help="`categories.csv` from categories.py.")
    ap.add_argument("--out", required=True, help="A folder BESIDE the set, never inside it.")
    ap.add_argument(
        "--trim",
        type=float,
        default=0.0,
        help="Drop this share of the least typical of each category, which is where its strays sit.",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.images).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    if out == root or root in out.parents:
        raise SystemExit(f"--out {out} is inside --images {root}; put it beside the set")

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    mine = collections.defaultdict(list)
    for row in rows:
        mine[int(row["category"])].append((row["file"], float(row["typicality"])))

    if out.exists():
        shutil.rmtree(out)
    linked = copied = 0
    for cat in sorted(mine):
        files = sorted(mine[cat], key=lambda q: -q[1])
        keep = files[: max(1, round(len(files) * (1.0 - args.trim)))]
        folder = out / f"cat{cat}"
        folder.mkdir(parents=True)
        for name, _ in keep:
            src, dst = root / name, folder / pathlib.Path(name).name
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                # A different volume, or a filesystem without links: a copy still trains.
                shutil.copy2(src, dst)
                copied += 1
        print(f"  cat{cat}: {len(keep)} images" + (f" ({len(files) - len(keep)} least typical left out)" if args.trim else ""))
    print(f"{linked} hardlinked" + (f", {copied} copied" if copied else "") + f" under {out}")


if __name__ == "__main__":
    main()
