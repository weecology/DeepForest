import os
from pathlib import Path

import numpy as np
import pandas as pd

from deepforest.main import deepforest
from deepforest.visualize import plot_results

# Hardcoded settings for short-term reproduction
MODEL_NAME = "weecology/deepforest-bird"
REVISION = "main"
WIDTH = 2048
HEIGHT = 2048
PATCH_SIZE = 128
PATCH_OVERLAP = 0.0
OUTPUT_IMAGE = "white_predict_tile.png"
OUTPUT_CSV = "white_predict_tile.csv"
SHOW = True
NMS_THRESH: float | None = None  # Leave None to use config default


def run() -> pd.DataFrame | None:
    """Generate a 255-white image, run predict_tile, and optionally plot/save
    outputs."""
    # Create all-white 8-bit image (RGB). For an all-white image, RGB/BGR is equivalent.
    img = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)

    # Initialize model with default config
    m = deepforest()

    # Load bird model from Hugging Face
    m.load_model(model_name=MODEL_NAME, revision=REVISION)

    # Optionally override NMS threshold (used internally by predict_tile)
    if NMS_THRESH is not None:
        m.config.nms_thresh = float(NMS_THRESH)

    # Run tiled prediction directly on the in-memory image
    results = m.predict_tile(
        image=img,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        iou_threshold=m.config.nms_thresh,
    )

    # Normalize return type for downstream handling
    if isinstance(results, tuple):
        results = results[0]

    # Print quick summary for inspection
    if results is None:
        print("No predictions returned (None).")
    elif isinstance(results, pd.DataFrame) and results.empty:
        print("Predictions dataframe is empty (0 boxes).")
    else:
        assert isinstance(results, pd.DataFrame)
        num_boxes = len(results)
        labels = ", ".join(sorted(map(str, results.label.unique())))
        max_score = results["score"].max() if "score" in results.columns else None
        print(f"Predictions: {num_boxes} boxes")
        print(f"Unique labels: {labels}")
        if max_score is not None:
            print(f"Max score: {max_score:.4f}")
        print(results.head(10))

    # Save CSV if requested and available
    if OUTPUT_CSV and isinstance(results, pd.DataFrame):
        os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
        results.to_csv(OUTPUT_CSV, index=False)
        print(f"Wrote predictions CSV to: {OUTPUT_CSV}")

    # Plot results on the white image and optionally save
    if isinstance(results, pd.DataFrame) and not results.empty:
        savedir = None
        basename = None
        if OUTPUT_IMAGE:
            output_path = Path(OUTPUT_IMAGE)
            savedir = str(output_path.parent)
            basename = output_path.stem
            os.makedirs(savedir or ".", exist_ok=True)

        # plot_results can take a raw image array; pass basename to avoid reliance on df.image_path
        plot_results(
            results=results,
            image=img,
            savedir=savedir,
            basename=basename,
            show=SHOW,
        )
        if OUTPUT_IMAGE:
            print(f"Wrote visualization to: {OUTPUT_IMAGE}")
    else:
        print("Skipping plot: no predictions to visualize.")

    return results


if __name__ == "__main__":
    run()
