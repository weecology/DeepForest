"""Evaluate sensitivity of box_recall and box_precision to patch_size.

This script wraps evaluate_deepwater_horizon.py to evaluate multiple patch sizes
for both checkpoint and pretrained models, and generate a sensitivity plot showing
how metrics vary with patch size for comparison.
"""

import os
import importlib.util
import argparse
import matplotlib.pyplot as plt
import pandas as pd

# Import from evaluate_deepwater_horizon in the same directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_eval_module_path = os.path.join(_script_dir, "evaluate_deepwater_horizon.py")
spec = importlib.util.spec_from_file_location("evaluate_deepwater_horizon", _eval_module_path)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

load_shapefiles_and_create_test_csv = eval_module.load_shapefiles_and_create_test_csv
split_test_images_for_evaluation = eval_module.split_test_images_for_evaluation
evaluate_models = eval_module.evaluate_models


def evaluate_patch_size_sensitivity(
    data_dir,
    checkpoint_path,
    patch_sizes,
    iou_threshold=0.4,
    patch_overlap=0.0,
):
    """Evaluate both checkpoint and pretrained models across multiple patch sizes and collect results.

    Args:
        data_dir: Directory containing shapefiles and images
        checkpoint_path: Path to checkpoint file
        patch_sizes: List of patch sizes to evaluate
        iou_threshold: IoU threshold for evaluation
        patch_overlap: Overlap between patches

    Returns:
        DataFrame with patch_size and metrics for both checkpoint and pretrained models
    """
    # Step 1: Load shapefiles and create test.csv (only once)
    print("=" * 80)
    print("Step 1: Loading shapefiles and creating test.csv")
    print("=" * 80)
    test_csv = load_shapefiles_and_create_test_csv(data_dir)

    results = []

    for patch_size in patch_sizes:
        print("\n" + "=" * 80)
        print(f"Evaluating patch size: {patch_size}")
        print("=" * 80)

        # Create patch-size-specific output directory
        base_output_dir = os.path.join(os.path.dirname(test_csv), "test_splits")
        patch_output_dir = os.path.join(base_output_dir, f"patch_{patch_size}")
        split_csv_name = f"test_split_patch_{patch_size}.csv"

        # Check if split CSV already exists
        split_csv_path = os.path.join(patch_output_dir, split_csv_name)
        if os.path.exists(split_csv_path):
            print(f"Found existing split CSV at {split_csv_path}, skipping splitting...")
            split_dir = patch_output_dir
        else:
            # Step 2: Split test images for this patch size
            print(f"\nSplitting test images for patch size {patch_size}...")
            split_csv_path, split_dir = split_test_images_for_evaluation(
                test_csv=test_csv,
                data_dir=data_dir,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                output_dir=patch_output_dir,
                split_csv_name=split_csv_name,
            )

        # Step 3: Evaluate both models
        print(f"\nEvaluating both models for patch size {patch_size}...")
        eval_results, _, _ = evaluate_models(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            test_csv=split_csv_path,
            split_dir=split_dir,
            iou_threshold=iou_threshold,
        )

        # Extract results for both models
        checkpoint_results = eval_results.get("checkpoint", {})
        pretrained_results = eval_results.get("pretrained", {})

        checkpoint_precision = checkpoint_results.get("box_precision", None)
        checkpoint_recall = checkpoint_results.get("box_recall", None)
        pretrained_precision = pretrained_results.get("box_precision", None)
        pretrained_recall = pretrained_results.get("box_recall", None)

        results.append(
            {
                "patch_size": patch_size,
                "checkpoint_precision": checkpoint_precision,
                "checkpoint_recall": checkpoint_recall,
                "pretrained_precision": pretrained_precision,
                "pretrained_recall": pretrained_recall,
            }
        )

        print(f"Patch size {patch_size}:")
        print(f"  Checkpoint - Precision={checkpoint_precision}, Recall={checkpoint_recall}")
        print(f"  Pretrained - Precision={pretrained_precision}, Recall={pretrained_recall}")

    return pd.DataFrame(results)


def plot_sensitivity(results_df, output_path):
    """Create a plot showing sensitivity of metrics to patch size for both models.

    Args:
        results_df: DataFrame with patch_size and metrics for both checkpoint and pretrained models
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot checkpoint model (solid lines)
    ax.plot(
        results_df["patch_size"],
        results_df["checkpoint_precision"],
        marker="o",
        label="Checkpoint Precision",
        linewidth=2,
        markersize=8,
        linestyle="-",
        color="C0",
    )
    ax.plot(
        results_df["patch_size"],
        results_df["checkpoint_recall"],
        marker="s",
        label="Checkpoint Recall",
        linewidth=2,
        markersize=8,
        linestyle="-",
        color="C1",
    )

    # Plot pretrained model (dashed lines)
    ax.plot(
        results_df["patch_size"],
        results_df["pretrained_precision"],
        marker="o",
        label="Pretrained Precision",
        linewidth=2,
        markersize=8,
        linestyle="--",
        color="C0",
        alpha=0.7,
    )
    ax.plot(
        results_df["patch_size"],
        results_df["pretrained_recall"],
        marker="s",
        label="Pretrained Recall",
        linewidth=2,
        markersize=8,
        linestyle="--",
        color="C1",
        alpha=0.7,
    )

    ax.set_xlabel("Patch Size (pixels)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Sensitivity of Box Precision and Recall to Patch Size\n(Checkpoint vs Pretrained Model)", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Ensure y-axis shows full range
    all_metrics = pd.concat([
        results_df["checkpoint_precision"],
        results_df["checkpoint_recall"],
        results_df["pretrained_precision"],
        results_df["pretrained_recall"],
    ])
    y_min = all_metrics.min()
    y_max = all_metrics.max()
    y_range = y_max - y_min
    ax.set_ylim(
        max(0, y_min - 0.1 * y_range),
        min(1.0, y_max + 0.1 * y_range),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved sensitivity plot to: {output_path}")
    plt.close(fig)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate sensitivity of metrics to patch size"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/blue/ewhite/b.weinstein/bird_detector_retrain/zero_shot/avian_images_annotated",
        help="Directory containing shapefiles and images",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/blue/ewhite/b.weinstein/bird_detector_retrain/data/checkpoints/6181df1ab7ac40f291b863a2a9b86024.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.4,
        help="IoU threshold for evaluation",
    )
    parser.add_argument(
        "--patch_overlap",
        type=float,
        default=0.0,
        help="Patch overlap for splitting images (default: 0.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: data_dir/plots)",
    )
    parser.add_argument(
        "--patch_sizes",
        type=int,
        nargs="+",
        default=[200, 400, 600, 800, 1000, 1500, 2000],
        help="List of patch sizes to evaluate (default: 200 400 600 800 1000 1500 2000)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "plots")
    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate across patch sizes (both models)
    results_df = evaluate_patch_size_sensitivity(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        patch_sizes=args.patch_sizes,
        iou_threshold=args.iou_threshold,
        patch_overlap=args.patch_overlap,
    )

    # Save results to CSV
    results_csv = os.path.join(args.output_dir, "patch_size_sensitivity_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved results to: {results_csv}")

    # Create and save plot
    plot_path = os.path.join(args.output_dir, "patch_size_sensitivity.png")
    plot_sensitivity(results_df, plot_path)

    # Print summary
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
