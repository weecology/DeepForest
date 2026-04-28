"""Create quick visuals for marine mammal detector blog/reporting use."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from deepforest.main import deepforest
from deepforest.visualize import plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training data and prediction visualizations."
    )
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--score-thresh", type=float, default=0.2)
    return parser.parse_args()


def sample_non_empty_images(annotations: pd.DataFrame, n: int) -> list[str]:
    non_empty = annotations[
        ~(
            (annotations["xmin"] == 0)
            & (annotations["xmax"] == 0)
            & (annotations["ymin"] == 0)
            & (annotations["ymax"] == 0)
        )
    ]
    images = sorted(non_empty["image_path"].unique())
    return images[:n]


def save_annotation_plot(
    image_path: str,
    image_annotations: pd.DataFrame,
    root_dir: Path,
    output_path: Path,
) -> None:
    image = Image.open(root_dir / image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for _, row in image_annotations.iterrows():
        width = row["xmax"] - row["xmin"]
        height = row["ymax"] - row["ymin"]
        rect = plt.Rectangle(
            (row["xmin"], row["ymin"]),
            width,
            height,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
        )
        ax.add_patch(rect)
    ax.set_title(f"Ground truth: {image_path}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_plot(
    model,
    image_path: str,
    root_dir: Path,
    output_path: Path,
    score_thresh: float,
) -> None:
    image_fp = root_dir / image_path
    predictions = model.predict_image(path=str(image_fp))
    predictions = predictions[predictions["score"] >= score_thresh].copy()
    if predictions.empty:
        Image.open(image_fp).convert("RGB").save(output_path)
        return
    plot_results(
        results=predictions,
        savedir=str(output_path.parent),
        basename=output_path.stem,
        image=str(image_fp),
        show=False,
        results_color=[0, 255, 255],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = pd.read_csv(args.annotations_csv, low_memory=False)
    sampled = sample_non_empty_images(annotations, args.num_images)
    if not sampled:
        raise ValueError("No non-empty images found in annotations CSV.")

    model = deepforest.load_from_checkpoint(str(args.checkpoint))
    model.label_dict = {"Object": 0}
    model.numeric_to_label_dict = {0: "Object"}

    for image_path in sampled:
        image_ann = annotations[annotations["image_path"] == image_path]
        stem = Path(image_path).stem
        save_annotation_plot(
            image_path=image_path,
            image_annotations=image_ann,
            root_dir=args.root_dir,
            output_path=args.output_dir / f"{stem}_ground_truth.png",
        )
        save_prediction_plot(
            model=model,
            image_path=image_path,
            root_dir=args.root_dir,
            output_path=args.output_dir / f"{stem}_prediction.png",
            score_thresh=args.score_thresh,
        )

    print(f"Wrote {len(sampled) * 2} files to {args.output_dir}")


if __name__ == "__main__":
    main()
