"""Build weak Alive/Dead pretraining crops from TreeFinder (NAIP 0.6 m).

TreeFinder labels dead-tree pixels only. Dead crops are the bounding boxes of
labelled dead blobs. Alive crops are DeepForest tree detections that contain no
dead pixel (within a margin) and no nodata. Unlabelled dead trees will leak into
the Alive class, so these labels are weak and used for pretraining only.

Tiles are upsampled by ``--factor`` (3 -> ~0.2 m) before detection and cropping.
Only the official ``trainval`` tiles are used, so the TreeFinder test split stays
clean for benchmarking.

Writes crops to ``crops/treefinder/{Alive,Dead}/`` and ``manifests/treefinder.csv``.
The group is the source NAIP image, so neighbouring tiles share a group.

Usage:
    python build_treefinder_crops.py --root /blue/ewhite/b.weinstein/DeepForest/alive_dead
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision.ops import nms

from deepforest import main as df_main


def window_starts(size: int, patch: int, overlap: float) -> list[int]:
    """Window origins along one axis, last window flush with the edge."""
    if size <= patch:
        return [0]
    stride = int(patch * (1 - overlap))
    return sorted(set(range(0, size - patch, stride)) | {size - patch})


@torch.no_grad()
def detect(
    model, img: torch.Tensor, patch: int = 400, overlap: float = 0.05
) -> np.ndarray:
    """Tree boxes (xmin, ymin, xmax, ymax, score) on a 3xHxW 0-1 image."""
    _, h, w = img.shape
    padded = F.pad(img, (0, max(0, patch - w), 0, max(0, patch - h)))
    origins = [
        (x, y)
        for y in window_starts(h, patch, overlap)
        for x in window_starts(w, patch, overlap)
    ]
    outputs = model([padded[:, y : y + patch, x : x + patch] for x, y in origins])
    boxes = torch.cat(
        [
            o["boxes"] + img.new_tensor([x, y, x, y])
            for o, (x, y) in zip(outputs, origins, strict=True)
        ]
    )
    scores = torch.cat([o["scores"] for o in outputs])
    keep = nms(boxes, scores, 0.15)
    boxes = boxes[keep]
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    return torch.cat([boxes, scores[keep, None]], 1).cpu().numpy()


def box_sum(mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Number of True pixels of mask inside each integer box."""
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    x0, y0, x1, y1 = boxes.T
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def int_boxes(boxes: np.ndarray, pad: int, w: int, h: int) -> np.ndarray:
    """Pad boxes and round outward to integer pixel bounds inside the image."""
    b = np.column_stack([np.floor(boxes[:, :2]) - pad, np.ceil(boxes[:, 2:4]) + pad])
    return b.astype(int).clip(0, [w, h, w, h])


def tile_crops(
    name: str, row, model, args, device, rng, dead_sides: np.ndarray
) -> list[dict]:
    """Return crop records for one tile and write the crop PNGs."""
    with rasterio.open(args.tile_dir / name) as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        label = src.read(5)
    h, w = label.shape
    dead, nodata = label == 1, label == 255
    blobs, n = ndimage.label(dead, structure=np.ones((3, 3), int))

    # Dead: blob boxes, area >= min_area, not cut by the tile edge or nodata
    objs = ndimage.find_objects(blobs)
    area = np.bincount(blobs.ravel(), minlength=n + 1)[1:]
    pos = np.array(
        [[o[1].start, o[0].start, o[1].stop, o[0].stop] for o in objs], dtype=float
    ).reshape(-1, 4)
    pos = int_boxes(pos, args.pad, w, h)
    touches = (pos[:, 0] == 0) | (pos[:, 1] == 0) | (pos[:, 2] == w) | (pos[:, 3] == h)
    pos = pos[(area >= args.min_area) & ~touches & (box_sum(nodata, pos) == 0)]

    # Alive: confident detections with no dead pixel within `margin` and no nodata
    img = F.interpolate(
        torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255,
        scale_factor=args.factor,
        mode="bilinear",
        align_corners=False,
    )[0].clamp(0, 1)
    det = detect(model, img)
    det = det[det[:, 4] >= args.min_score]
    # Resize Alive boxes to squares drawn from the Dead box sizes, so crop size
    # and upsampling blur do not give away the label
    centers = (det[:, :2] + det[:, 2:4]) / 2 / args.factor
    half = rng.choice(dead_sides, size=len(det)) / 2
    neg = (
        np.column_stack([centers - half[:, None], centers + half[:, None]])
        .round()
        .astype(int)
    )
    inside = (neg[:, :2] >= 0).all(1) & (neg[:, 2] <= w) & (neg[:, 3] <= h)
    neg = neg[inside]
    clear = box_sum(dead, int_boxes(neg, args.margin, w, h)) == 0
    neg = neg[clear & (box_sum(nodata, neg) == 0)]
    n_neg = max(args.neg_per_pos * len(pos), args.neg_per_empty_tile)
    neg = neg[rng.permutation(len(neg))[:n_neg]]

    up = (img * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    records = []
    for label_name, boxes in (("Dead", pos), ("Alive", neg)):
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            f = args.factor
            crop = up[int(y0 * f) : int(y1 * f), int(x0 * f) : int(x1 * f)]
            rel = (
                Path("crops/treefinder")
                / label_name
                / f"{Path(name).stem}_{label_name[0]}{k}.png"
            )
            Image.fromarray(crop).save(args.root / rel)
            records.append(
                {
                    "crop_path": str(rel),
                    "label": label_name,
                    "tile": name,
                    "xmin": x0,
                    "ymin": y0,
                    "xmax": x1,
                    "ymax": y1,
                    "group": row.ImageRawID,
                    "site": row.State,
                    "climate": row.ClimateType,
                    "tree_types": row.TreeTypes,
                    "width": crop.shape[1],
                    "height": crop.shape[0],
                }
            )
    return records


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = df_main.deepforest()
    detector.load_model("weecology/deepforest-tree")
    model = detector.model.to(device).eval()

    tiles = pd.read_csv(args.root / "sources/treefinder/tiles.csv")
    tiles = tiles[tiles.split_random == "trainval"].set_index("FileName")
    if args.limit:
        tiles = tiles.head(args.limit)
    for lab in ("Alive", "Dead"):
        (args.root / "crops/treefinder" / lab).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    gt = pd.read_csv(args.root / "sources/treefinder/gt_instances.csv")
    gt = gt[
        (gt.split_random == "trainval")
        & (gt.area_px >= args.min_area)
        & ~gt.touches_border
    ]
    dead_sides = (
        np.maximum(gt.xmax - gt.xmin, gt.ymax - gt.ymin).to_numpy() + 2 * args.pad
    )
    records = []
    for i, (name, row) in enumerate(tiles.iterrows()):
        records += tile_crops(name, row, model, args, device, rng, dead_sides)
        if i % 500 == 0:
            print(f"{i}/{len(tiles)} tiles, {len(records)} crops", flush=True)

    df = pd.DataFrame(records).assign(
        source="treefinder", label_type="weak", gsd_m=0.6 / args.factor
    )
    out = (
        args.root
        / "manifests"
        / ("treefinder.csv" if not args.limit else "treefinder_test.csv")
    )
    df.to_csv(out, index=False)
    print(df.label.value_counts().to_dict(), "groups", df.group.nunique())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--tile-dir",
        type=Path,
        default=None,
        help="default: <root>/sources/treefinder/data/tiles224_v3/tiles224_v3",
    )
    parser.add_argument(
        "--factor", type=float, default=3, help="upsampling factor before detection"
    )
    parser.add_argument(
        "--pad", type=int, default=2, help="native px added around dead blobs"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=3,
        help="native px around Alive boxes that must be free of dead pixels",
    )
    parser.add_argument(
        "--min-area", type=int, default=10, help="smallest dead blob in native px"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.3, help="detector score for Alive boxes"
    )
    parser.add_argument("--neg-per-pos", type=int, default=3)
    parser.add_argument("--neg-per-empty-tile", type=int, default=2)
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N tiles (testing)"
    )
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    a.tile_dir = a.tile_dir or a.root / "sources/treefinder/data/tiles224_v3/tiles224_v3"
    main(a)
