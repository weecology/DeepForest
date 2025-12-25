"""Evaluate bird detection model at multiple score thresholds.

This script evaluates a checkpoint model at multiple score thresholds and
generates a precision-recall curve.

Example usage:
    python evaluate_thresholds.py --checkpoint_path /path/to/checkpoint.ckpt --data_dir /path/to/data
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from deepforest import main


def evaluate_thresholds(
    checkpoint_path, data_dir, iou_threshold=0.4, thresholds=None, output_path=None
):
    """Evaluate checkpoint model at multiple score thresholds.

    Args:
        checkpoint_path: Path to the checkpoint file
        data_dir: Directory containing test.csv and images
        iou_threshold: IoU threshold for evaluation (default: 0.4)
        thresholds: List of score thresholds to evaluate (default: 0.1 to 0.5 in 0.05 steps)
        output_path: Path to save the plot (default: data_dir/precision_recall_curve.png)

    Returns:
        dict: Dictionary with thresholds, precision, and recall arrays
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.55, 0.05).round(2).tolist()

    test_csv = os.path.join(data_dir, "test.csv")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("Evaluating Checkpoint Model at Multiple Score Thresholds")
    print("=" * 80)
    print(f"\nTest dataset: {test_csv}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Score thresholds: {thresholds}\n")

    # Load model once
    print("Loading checkpoint model...")
    model = main.deepforest.load_from_checkpoint(checkpoint_path)

    precision_scores = []
    recall_scores = []

    print("\nEvaluating at each threshold:")
    print("-" * 80)
    for i, threshold in enumerate(thresholds):
        print(f"\n[{i+1}/{len(thresholds)}] Evaluating at score threshold: {threshold:.2f}")
        model.config.score_thresh = threshold
        model.model.score_thresh = threshold

        results = model.evaluate(
            csv_file=test_csv,
            root_dir=data_dir,
            iou_threshold=iou_threshold,
        )

        precision = results["box_precision"]
        recall = results["box_recall"]

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

    # Create results dictionary
    threshold_results = {
        "thresholds": thresholds,
        "precision": precision_scores,
        "recall": recall_scores,
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 40)
    for thresh, prec, rec in zip(thresholds, precision_scores, recall_scores):
        print(f"{thresh:<12.2f} {prec:<12.4f} {rec:<12.4f}")

    # Generate plot
    if output_path is None:
        output_path = os.path.join(data_dir, "precision_recall_curve.png")

    print(f"\nGenerating plot: {output_path}")
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_scores, "o-", label="Precision", linewidth=2, markersize=8)
    plt.plot(thresholds, recall_scores, "s-", label="Recall", linewidth=2, markersize=8)
    plt.xlabel("Score Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Precision and Recall vs Score Threshold\n(Retrained Bird Detection Model)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    plt.ylim(0, max(max(precision_scores), max(recall_scores)) * 1.1)

    # Add value labels on points
    for thresh, prec, rec in zip(thresholds, precision_scores, recall_scores):
        plt.annotate(
            f"{prec:.3f}",
            (thresh, prec),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )
        plt.annotate(
            f"{rec:.3f}",
            (thresh, rec),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return threshold_results


def run():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint model at multiple score thresholds"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing test.csv and images",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.4,
        help="IoU threshold for evaluation (default: 0.4)",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Path to save the precision-recall plot (default: data_dir/precision_recall_curve.png)",
    )

    args = parser.parse_args()

    evaluate_thresholds(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        iou_threshold=args.iou_threshold,
        output_path=args.plot_output,
    )


if __name__ == "__main__":
    run()

