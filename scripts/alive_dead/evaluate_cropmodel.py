"""Evaluate an Alive/Dead CropModel on one split of an ImageFolder tree.

Writes per-crop predictions and summary metrics (overall and per site).

Usage:
    python evaluate_cropmodel.py --split-dir <root>/splits/neon_hand_v1 --split test \
        --model weecology/cropmodel-deadtrees --out results/hf_current
    python evaluate_cropmodel.py ... --model runs/baseline/best.ckpt --out runs/baseline/eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchmetrics.functional.classification import binary_auroc, binary_average_precision

from deepforest.model import CropModel


def load(model: str) -> CropModel:
    """Load a CropModel from a Lightning checkpoint or a Hugging Face repo id."""
    if model.endswith(".ckpt"):
        m = CropModel.load_from_checkpoint(model, map_location="cpu")
    else:
        m = CropModel.load_model(model)
    return m.eval()


@torch.no_grad()
def predict(m: CropModel, paths: list[str], batch_size: int = 256) -> np.ndarray:
    """Return P(Dead) for each image path."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(device)
    tf = m.get_transform(augmentations=None)
    dead = m.label_dict["Dead"]
    out = []
    for i in range(0, len(paths), batch_size):
        x = torch.stack(
            [tf(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]]
        )
        out.append(torch.softmax(m(x.to(device)), 1)[:, dead].cpu().numpy())
    return np.concatenate(out)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary metrics with Dead as the positive class."""
    y = y.astype(bool)
    pred = p >= threshold
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    alive_rec = (~pred[~y]).mean() if (~y).any() else np.nan
    both = y.any() and (~y).any()
    pt, yt = torch.as_tensor(p, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    return {
        "n": int(len(y)),
        "n_dead": int(y.sum()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.nanmean([rec, alive_rec])),
        "dead_precision": float(prec),
        "dead_recall": float(rec),
        "dead_f1": float(f1),
        "alive_recall": float(alive_rec),
        "auroc": float(binary_auroc(pt, yt)) if both else np.nan,
        "average_precision": float(binary_average_precision(pt, yt)) if both else np.nan,
    }


def main(split_dir: Path, split: str, model: str, out: Path) -> None:
    df = pd.read_csv(split_dir / "split.csv")
    df = df[df.split == split].copy()
    df["path"] = [
        str(split_dir / split / r.label / f"{r.source}__{Path(r.crop_path).name}")
        for r in df.itertuples()
    ]
    df["y"] = (df.label == "Dead").astype(int)
    df["p_dead"] = predict(load(model), df.path.tolist())

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"predictions_{split}.csv", index=False)
    summary = {"model": model, "split_dir": str(split_dir), "split": split}
    summary.update(metrics(df.y.to_numpy(), df.p_dead.to_numpy()))
    # Threshold that maximizes Dead F1, for reference only (chosen on this split)
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [metrics(df.y.to_numpy(), df.p_dead.to_numpy(), t)["dead_f1"] for t in grid]
    summary["best_f1_threshold"] = float(grid[int(np.argmax(f1s))])
    summary["best_f1"] = float(max(f1s))
    (out / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2))

    rows = []
    for key in ("source", "site"):
        for g, x in df.groupby(key):
            rows.append(
                {"by": key, "group": g, **metrics(x.y.to_numpy(), x.p_dead.to_numpy())}
            )
    pd.DataFrame(rows).to_csv(out / f"metrics_{split}_by_group.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model", required=True, help=".ckpt path or Hugging Face repo id"
    )
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    main(a.split_dir, a.split, a.model, a.out)
