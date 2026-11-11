"""Build Alive/Dead crown crops from deadtrees.earth (DTE-aerial) masks.

DTE masks label pixels as background (0), tree cover (1), mortality (2) and
ignored (255). DeepForest detects crowns on each 0.1 m patch, and each crown box
is labelled from the mask pixels inside it:

- Dead:  mortality fraction >= ``--dead-frac``
- Alive: no mortality pixel and tree-cover fraction >= ``--alive-tree-frac``
- crowns in between, or touching ignored pixels, are skipped.

Two sources:
- ``train``: the CC BY train/val WebDataset shards (model-assisted labels, weak),
  subsampled per orthophoto. Group = orthophoto id.
- ``bench``: the 25-site expert benchmark, 10 cm patches only, for evaluation.

Writes crops to ``crops/dte_<source>/{Alive,Dead}/`` and ``manifests/dte_<source>.csv``.

Usage:
    python build_dte_crops.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead --source bench
    python build_dte_crops.py --root ... --source train
"""

import argparse
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from build_treefinder_crops import box_sum, detect, int_boxes
from PIL import Image

from deepforest import main as df_main


def label_crowns(
    rgb: np.ndarray, mask: np.ndarray, model, args, device, rng
) -> list[dict]:
    """Detect crowns on an HxWx3 patch and label them from the mask."""
    img = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255
    det = detect(model, img)
    det = det[det[:, 4] >= args.min_score]
    h, w = mask.shape
    boxes = int_boxes(det[:, :4], 0, w, h)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1)
    mort = box_sum(mask == 2, boxes) / area
    tree = box_sum((mask == 1) | (mask == 2), boxes) / area
    ignored = box_sum(mask == 255, boxes) > 0

    dead = (mort >= args.dead_frac) & ~ignored
    alive = (mort == 0) & (tree >= args.alive_tree_frac) & ~ignored
    alive_idx = np.flatnonzero(alive)
    n_alive = max(args.alive_per_dead * dead.sum(), args.min_alive_per_patch)
    alive_idx = rng.permutation(alive_idx)[:n_alive]

    out = []
    for label, idx in (("Dead", np.flatnonzero(dead)), ("Alive", alive_idx)):
        for i in idx:
            x0, y0, x1, y1 = int_boxes(boxes[i : i + 1], args.pad, w, h)[0]
            out.append(
                {
                    "label": label,
                    "crop": rgb[y0:y1, x0:x1],
                    "xmin": x0,
                    "ymin": y0,
                    "xmax": x1,
                    "ymax": y1,
                    "det_score": float(det[i, 4]),
                    "mort_frac": float(mort[i]),
                    "tree_frac": float(tree[i]),
                }
            )
    return out


def iter_train(root: Path, args, rng):
    """Yield (key, rgb, mask, meta) for a per-orthophoto subsample of train/val shards."""
    kept = defaultdict(int)
    shards = sorted((root / "sources/dte_aerial_train_ccby/data").glob("*/*/0.1/*.tar"))
    for s, shard in enumerate(shards):
        print(f"shard {s}/{len(shards)} {shard.relative_to(root)}", flush=True)
        files = {}
        with tarfile.open(shard) as tar:
            for member in tar:
                key, _, ext = member.name.partition(".")
                files.setdefault(key, {})[ext] = tar.extractfile(member).read()
                if len(files[key]) < 3:
                    continue
                parts = files.pop(key)
                meta = json.loads(parts["json"])
                has_mort = meta["frac_mortality"] > args.min_patch_mortality
                cap = args.patches_per_ortho if has_mort else args.empty_patches_per_ortho
                ortho = (meta["id"], has_mort)
                if kept[ortho] >= cap or (not has_mort and rng.random() > 0.1):
                    continue
                kept[ortho] += 1
                with rasterio.open(io.BytesIO(parts["tif"])) as src:
                    rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
                with rasterio.open(io.BytesIO(parts["mask.tif"])) as src:
                    mask = src.read(1)
                meta["split_dte"] = shard.parts[-4]
                yield key, rgb, mask, meta


def iter_bench(root: Path, args, rng):
    """Yield (key, rgb, mask, meta) for the 10 cm benchmark patches."""
    base = root / "sources/dte_aerial_bench"
    meta = pd.read_csv(base / "DTE-aerial-bench-meta.csv")
    for row in meta[meta.resolution == "10cm"].itertuples():
        rgb = np.array(Image.open(base / row.tile_path).convert("RGB"))
        mask = np.array(Image.open(base / row.mask_path))
        yield Path(row.tile_path).stem, rgb, mask, row._asdict()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = df_main.deepforest()
    detector.load_model("weecology/deepforest-tree")
    model = detector.model.to(device).eval()
    rng = np.random.default_rng(args.seed)

    name = f"dte_{args.source}"
    for lab in ("Alive", "Dead"):
        (args.root / "crops" / name / lab).mkdir(parents=True, exist_ok=True)
    patches = (
        iter_train(args.root, args, rng)
        if args.source == "train"
        else iter_bench(args.root, args, rng)
    )

    records = []
    for n, (key, rgb, mask, meta) in enumerate(patches):
        for k, r in enumerate(label_crowns(rgb, mask, model, args, device, rng)):
            rel = Path("crops") / name / r["label"] / f"{key}_{r['label'][0]}{k}.png"
            Image.fromarray(r.pop("crop")).save(args.root / rel)
            records.append(
                {
                    "crop_path": str(rel),
                    **r,
                    "patch": key,
                    "group": f"ortho{meta['id']}",
                    "site": meta.get("biome_group", meta.get("biome")),
                    "year": meta.get("acquisition_date_year"),
                    "lat": meta.get("lat"),
                    "lon": meta.get("lon"),
                    "split_dte": meta.get("split_dte", "bench"),
                }
            )
        if n % 200 == 0:
            print(f"{n} patches, {len(records)} crops", flush=True)
        if args.limit and n + 1 >= args.limit:
            break

    df = pd.DataFrame(records).assign(
        source=name,
        label_type="weak" if args.source == "train" else "expert_mask",
        gsd_m=0.1,
    )
    df["width"] = df.xmax - df.xmin
    df["height"] = df.ymax - df.ymin
    suffix = "_test" if args.limit else ""
    df.to_csv(args.root / "manifests" / f"{name}{suffix}.csv", index=False)
    print(df.label.value_counts().to_dict(), "groups", df.group.nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", choices=["train", "bench"], required=True)
    parser.add_argument(
        "--min-score", type=float, default=0.3, help="detector score threshold"
    )
    parser.add_argument(
        "--dead-frac", type=float, default=0.5, help="mortality fraction for Dead"
    )
    parser.add_argument(
        "--alive-tree-frac", type=float, default=0.7, help="tree-cover fraction for Alive"
    )
    parser.add_argument(
        "--pad", type=int, default=10, help="px added around each crown (10 px = 1 m)"
    )
    parser.add_argument("--alive-per-dead", type=int, default=3)
    parser.add_argument("--min-alive-per-patch", type=int, default=5)
    parser.add_argument(
        "--patches-per-ortho", type=int, default=8, help="train: patches with mortality"
    )
    parser.add_argument(
        "--empty-patches-per-ortho", type=int, default=2, help="train: patches without"
    )
    parser.add_argument("--min-patch-mortality", type=float, default=0.005)
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N patches (testing)"
    )
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
