"""Evaluate bird detection models on an annotated test set.

Predicts once per model at a low score threshold, then reports box precision,
recall and F1 at each IoU and score threshold, so models can be compared and an
operating threshold chosen. Models are checkpoints from train_birds.py and/or
HuggingFace repositories (optionally with a revision, e.g. a PR ref).

Example usage:
    python evaluate_birds.py --csv_file /path/to/test.csv --root_dir /path/to/images \\
        --checkpoint /path/to/bird.ckpt --hf_model weecology/deepforest-bird \\
        --output_dir /path/to/results
"""

import argparse
import os

import pandas as pd

from deepforest import main
from deepforest.evaluate import evaluate_boxes


def load(model: str, score_thresh: float, nms_thresh: float) -> main.deepforest:
    """Load a checkpoint path or a HuggingFace 'org/repo[@revision]' model."""
    if os.path.exists(model):
        m = main.deepforest.load_from_checkpoint(model)
    else:
        repo, _, revision = model.partition("@")
        m = main.deepforest()
        m.load_model(repo, revision=revision or None)
    m.config.score_thresh = m.model.score_thresh = score_thresh
    m.config.nms_thresh = m.model.nms_thresh = nms_thresh
    m.config.devices = 1
    m.create_trainer(logger=False)
    return m


def predict(
    m: main.deepforest, csv_file: str, root_dir: str, patch_size: int | None
) -> pd.DataFrame:
    """Predict whole images, or tile each image when patch_size is given."""
    if patch_size is None:
        pred = m.predict_file(csv_file, root_dir)
    else:
        images = pd.read_csv(csv_file).image_path.unique()
        tiles = [
            m.predict_tile(
                path=os.path.join(root_dir, p), patch_size=patch_size, patch_overlap=0.1
            ).assign(image_path=p)
            for p in images
        ]
        pred = pd.concat([t for t in tiles if t is not None], ignore_index=True)
    return pred.assign(label="Bird")


def score(
    pred: pd.DataFrame,
    ground: pd.DataFrame,
    root_dir: str,
    iou_thresholds: list[float],
    score_thresholds: list[float],
) -> pd.DataFrame:
    """Box precision, recall and F1 for every IoU x score threshold pair."""
    rows = []
    for iou in iou_thresholds:
        for thr in score_thresholds:
            p = pred[pred.score >= thr].copy()
            p.root_dir = ground.root_dir = root_dir
            r = evaluate_boxes(p, ground, iou_threshold=iou)
            precision, recall = r["box_precision"], r["box_recall"]
            f1 = (
                2 * precision * recall / (precision + recall) if precision + recall else 0
            )
            rows.append(
                {
                    "iou_threshold": iou,
                    "score_threshold": thr,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "n_pred": len(p),
                    "n_ground": len(ground),
                }
            )
    return pd.DataFrame(rows)


def run():
    """Evaluate each model and write per-model predictions and one metrics
    table."""
    parser = argparse.ArgumentParser(description="Evaluate bird detection models")
    parser.add_argument("--csv_file", required=True, help="Ground truth annotations")
    parser.add_argument(
        "--root_dir", default=None, help="Image directory (default: the CSV's directory)"
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="Checkpoint (repeatable)"
    )
    parser.add_argument(
        "--hf_model",
        action="append",
        default=[],
        help="HuggingFace model 'org/repo[@revision]' (repeatable)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Tile images with predict_tile at this size (default: predict whole images)",
    )
    parser.add_argument("--nms_thresh", type=float, default=0.05)
    parser.add_argument(
        "--iou_thresholds", type=float, nargs="+", default=[0.2, 0.4], help="IoU levels"
    )
    parser.add_argument(
        "--score_thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        help="Score thresholds; models predict at the lowest one",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    models = args.checkpoint + args.hf_model
    if not models:
        parser.error("Give at least one --checkpoint or --hf_model")
    root_dir = args.root_dir or os.path.dirname(args.csv_file)
    os.makedirs(args.output_dir, exist_ok=True)

    cols = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    ground = pd.read_csv(args.csv_file)[cols].assign(label="Bird")

    results = []
    for model in models:
        print(f"Evaluating {model}")
        m = load(model, min(args.score_thresholds), args.nms_thresh)
        pred = predict(m, args.csv_file, root_dir, args.patch_size)
        name = os.path.basename(model).replace("@", "_")
        pred.drop(columns="geometry", errors="ignore").to_csv(
            os.path.join(args.output_dir, f"{name}_predictions.csv"), index=False
        )
        metrics = score(
            pred, ground, root_dir, args.iou_thresholds, args.score_thresholds
        )
        results.append(metrics.assign(model=model))

    results = pd.concat(results, ignore_index=True)
    results.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    best = results.loc[results.groupby(["model", "iou_threshold"]).f1.idxmax()]
    print("\nBest F1 per model and IoU threshold:")
    print(best.to_string(index=False))


if __name__ == "__main__":
    run()
