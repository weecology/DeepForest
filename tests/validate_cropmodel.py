#!/usr/bin/env python3
"""
CropModel Validation Script

A standalone tool for contributors to validate their CropModel before submission.
Runs DeepForest detection on a tile, classifies each crown with the contributor's
CropModel, and performs sanity checks on the results.

Usage:
    # Sanity check with bundled OSBS tile
    python tests/validate_cropmodel.py --model weecology/cropmodel-neon-resnet18-species

    # Sanity check with a custom tile
    python tests/validate_cropmodel.py --model weecology/cropmodel-neon-resnet18-species --tile path/to/tile.tif
"""

import argparse
import os
import sys


def load_detector():
    """Load the DeepForest tree detection model."""
    from deepforest import main

    detector = main.deepforest()
    detector.load_model("weecology/deepforest-tree")
    detector.create_trainer()
    return detector


def load_crop_model(model_id: str):
    """Load a CropModel from HuggingFace or local path."""
    from deepforest.model import CropModel

    print(f"Loading CropModel: {model_id}")
    crop_model = CropModel.load_model(model_id)
    print(
        f"  Classes: {len(crop_model.label_dict)} ({', '.join(list(crop_model.label_dict)[:5])}...)"
    )
    return crop_model


def run_classification(detector, tile_path: str, crop_model):
    """Run detection + CropModel classification on a tile."""
    print(f"\nRunning detection + classification on: {tile_path}")
    results = detector.predict_tile(
        path=tile_path,
        patch_size=400,
        patch_overlap=0.05,
        iou_threshold=0.15,
        crop_model=crop_model,
    )
    print(f"  Detected and classified {len(results)} crowns")
    return results


def sanity_checks(results) -> list[str]:
    """Run basic sanity checks on classification results.

    Returns:
        List of failure messages (empty if all checks pass).
    """
    failures = []

    if len(results) == 0:
        failures.append("FAIL: No crowns detected")
        return failures

    # Check required columns exist
    if "cropmodel_label" not in results.columns:
        failures.append("FAIL: 'cropmodel_label' column missing from results")
        return failures
    if "cropmodel_score" not in results.columns:
        failures.append("FAIL: 'cropmodel_score' column missing from results")
        return failures

    labels = results["cropmodel_label"]
    scores = results["cropmodel_score"]

    # Not all same class
    n_unique = labels.nunique()
    if n_unique == 1:
        failures.append(
            f"FAIL: All {len(results)} predictions are the same class '{labels.iloc[0]}'"
        )

    # Confidence sanity
    if scores.max() < 0.3:
        failures.append(
            f"FAIL: Max confidence is only {scores.max():.3f} — model may not be producing meaningful predictions"
        )

    # No NaN labels
    n_nan = labels.isna().sum()
    if n_nan > 0:
        failures.append(f"FAIL: {n_nan} predictions have NaN labels")

    return failures


def print_summary(results):
    """Print a summary of classification results."""
    labels = results["cropmodel_label"]
    scores = results["cropmodel_score"]

    print(f"\n{'=' * 50}")
    print("Classification Summary")
    print(f"{'=' * 50}")
    print(f"Total crowns: {len(results)}")
    print(f"Unique classes predicted: {labels.nunique()}")
    print(
        f"Confidence: min={scores.min():.3f}, mean={scores.mean():.3f}, max={scores.max():.3f}"
    )
    print("\nSpecies distribution:")
    counts = labels.value_counts()
    for species, count in counts.items():
        pct = 100 * count / len(results)
        print(f"  {species:>10s}: {count:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CropModel for DeepForest integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path to CropModel checkpoint",
    )
    parser.add_argument(
        "--tile",
        default=None,
        help="Path to a georeferenced tile. Defaults to bundled HARV test tile.",
    )
    args = parser.parse_args()

    if args.tile:
        tile_path = args.tile
    else:
        from deepforest import get_data

        tile_path = get_data("OSBS_029.tif")

    if not os.path.exists(tile_path):
        print(f"ERROR: Tile not found: {tile_path}")
        sys.exit(1)

    # Load models
    print("Loading DeepForest detector...")
    detector = load_detector()
    crop_model = load_crop_model(args.model)

    # Run detection + classification
    results = run_classification(detector, tile_path, crop_model)

    if results is None or len(results) == 0:
        print("\nERROR: No crowns detected. Check your tile or detection parameters.")
        sys.exit(1)

    # Sanity checks
    failures = sanity_checks(results)
    print_summary(results)

    if failures:
        print(f"\n{'=' * 50}")
        print("Sanity Check Failures")
        print(f"{'=' * 50}")
        for f in failures:
            print(f"  {f}")
        print(f"\n{len(failures)} check(s) failed.")
    else:
        print("\nAll sanity checks passed.")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
