"""Drop NEON VST crops whose review images the annotator deleted.

A crop is kept when ``<review_dir>/<Label>/<crop id>.jpg`` still exists. Writes
``manifests/<manifest>_reviewed.csv`` and ``manifests/<manifest>_culled.csv``.

Usage:
    python apply_vst_review.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead \
        --review-dir /path/to/returned/neon_vst_review_v1
"""

import argparse
from pathlib import Path

import pandas as pd


def main(args: argparse.Namespace) -> None:
    man = pd.read_csv(args.root / "manifests" / f"{args.manifest}.csv", low_memory=False)
    kept = {(f.parent.name, f.stem) for f in args.review_dir.rglob("*.jpg")}
    key = list(zip(man.label, man.crop_path.map(lambda p: Path(p).stem), strict=True))
    keep = pd.Series([k in kept for k in key], index=man.index)
    unknown = kept - set(key)
    assert not unknown, (
        f"{len(unknown)} review images don't match the manifest, e.g. {list(unknown)[:3]}"
    )

    out = args.root / "manifests"
    man[keep].to_csv(out / f"{args.manifest}_reviewed.csv", index=False)
    man[~keep].to_csv(out / f"{args.manifest}_culled.csv", index=False)
    summary = pd.crosstab(man.label, keep.map({True: "kept", False: "culled"}))
    print(summary.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--review-dir",
        type=Path,
        required=True,
        help="returned folder with Alive/ and Dead/",
    )
    p.add_argument("--manifest", default="neon_vst")
    main(p.parse_args())
