"""Compare retrained bird model checkpoint with pretrained weecology/deepforest-bird model.

This script evaluates both models on the same test dataset and prints a comparison
of performance metrics. It also evaluates the checkpoint model at multiple score
thresholds and generates a precision-recall curve.

Example usage:
    python compare_bird_models.py --checkpoint_path /path/to/checkpoint.ckpt --data_dir /path/to/data
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepforest import main


def compare_models(checkpoint_path, data_dir, iou_threshold=0.4):
    """Compare checkpoint model with pretrained weecology/deepforest-bird model.

    Args:
        checkpoint_path: Path to the checkpoint file
        data_dir: Directory containing test.csv and images
        iou_threshold: IoU threshold for evaluation (default: 0.4)

    Returns:
        dict: Dictionary containing results for both models
    """
    test_csv = os.path.join(data_dir, "test.csv")

    # Read test set and make a tiny subset of 100 images
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df.image_path.str.startswith("BDA")].head(10)
    test_csv = os.path.join(data_dir, "test_subset.csv")
    test_df.to_csv(test_csv, index=False)

    print("=" * 80)
    print("Bird Detection Model Comparison")
    print("=" * 80)
    print(f"\nTest dataset: {test_csv}")
    print(f"IoU threshold: {iou_threshold}\n")

    results = {}

    # Evaluate checkpoint model
    print("-" * 80)
    print("Evaluating retrained checkpoint model...")
    print(f"Checkpoint: {checkpoint_path}")
    print("-" * 80)
    checkpoint_model = main.deepforest.load_from_checkpoint(checkpoint_path)
    checkpoint_model.config.score_thresh = 0.25
    checkpoint_model.model.score_thresh = 0.25

    # Set up validation configuration
    checkpoint_model.config.validation.csv_file = test_csv
    checkpoint_model.config.validation.root_dir = data_dir
    checkpoint_model.config.validation.iou_threshold = iou_threshold
    checkpoint_model.config.validation.val_accuracy_interval = 1
    checkpoint_model.create_trainer()

    # Evaluate using trainer.validate()
    print("\n1. trainer.validate() results:")
    validation_results = checkpoint_model.trainer.validate(checkpoint_model)
    checkpoint_validate = validation_results[0] if validation_results else {}
    results["checkpoint_validate"] = checkpoint_validate

    checkpoint_precision_validate = checkpoint_validate.get('box_precision')
    checkpoint_recall_validate = checkpoint_validate.get('box_recall')
    if checkpoint_precision_validate is not None:
        print(f"  Box Precision: {checkpoint_precision_validate:.4f}")
    else:
        print("  Box Precision: N/A")
    if checkpoint_recall_validate is not None:
        print(f"  Box Recall: {checkpoint_recall_validate:.4f}")
    else:
        print("  Box Recall: N/A")
    print(f"  Empty Frame Accuracy: {checkpoint_validate.get('empty_frame_accuracy', 'N/A')}")

    # Evaluate using main.evaluate()
    print("\n2. main.evaluate() results:")
    checkpoint_evaluate = checkpoint_model.evaluate(
        csv_file=test_csv,
        root_dir=data_dir,
        iou_threshold=iou_threshold,
    )
    results["checkpoint_evaluate"] = checkpoint_evaluate

    checkpoint_precision_evaluate = checkpoint_evaluate.get('box_precision')
    checkpoint_recall_evaluate = checkpoint_evaluate.get('box_recall')
    if checkpoint_precision_evaluate is not None:
        print(f"  Box Precision: {checkpoint_precision_evaluate:.4f}")
    else:
        print("  Box Precision: N/A")
    if checkpoint_recall_evaluate is not None:
        print(f"  Box Recall: {checkpoint_recall_evaluate:.4f}")
    else:
        print("  Box Recall: N/A")
    print(f"  Empty Frame Accuracy: {checkpoint_evaluate.get('empty_frame_accuracy', 'N/A')}")

    # Store both for backward compatibility
    results["checkpoint"] = checkpoint_validate

    # Evaluate pretrained model
    print("\n" + "-" * 80)
    print("Evaluating pretrained weecology/deepforest-bird model...")
    print("-" * 80)
    pretrained_model = main.deepforest()
    pretrained_model.load_model("weecology/deepforest-bird")
    pretrained_model.config.score_thresh = 0.25
    pretrained_model.model.score_thresh = 0.25

    # Set label dictionaries to match
    pretrained_model.label_dict = {"Bird": 0}
    pretrained_model.numeric_to_label_dict = {0: "Bird"}
    pretrained_model.config.label_dict = {"Bird": 0}
    pretrained_model.config.num_classes = 1

    # Set up validation configuration
    pretrained_model.config.validation.csv_file = test_csv
    pretrained_model.config.validation.root_dir = data_dir
    pretrained_model.config.validation.iou_threshold = iou_threshold
    pretrained_model.config.validation.val_accuracy_interval = 1
    pretrained_model.create_trainer()

    # Evaluate using trainer.validate()
    print("\n1. trainer.validate() results:")
    validation_results = pretrained_model.trainer.validate(pretrained_model)
    pretrained_validate = validation_results[0] if validation_results else {}
    results["pretrained_validate"] = pretrained_validate

    pretrained_precision_validate = pretrained_validate.get('box_precision')
    pretrained_recall_validate = pretrained_validate.get('box_recall')
    if pretrained_precision_validate is not None:
        print(f"  Box Precision: {pretrained_precision_validate:.4f}")
    else:
        print("  Box Precision: N/A")
    if pretrained_recall_validate is not None:
        print(f"  Box Recall: {pretrained_recall_validate:.4f}")
    else:
        print("  Box Recall: N/A")
    print(f"  Empty Frame Accuracy: {pretrained_validate.get('empty_frame_accuracy', 'N/A')}")

    # Evaluate using main.evaluate()
    print("\n2. main.evaluate() results:")
    pretrained_evaluate = pretrained_model.evaluate(
        csv_file=test_csv,
        root_dir=data_dir,
        iou_threshold=iou_threshold,
    )
    results["pretrained_evaluate"] = pretrained_evaluate

    pretrained_precision_evaluate = pretrained_evaluate.get('box_precision')
    pretrained_recall_evaluate = pretrained_evaluate.get('box_recall')
    if pretrained_precision_evaluate is not None:
        print(f"  Box Precision: {pretrained_precision_evaluate:.4f}")
    else:
        print("  Box Precision: N/A")
    if pretrained_recall_evaluate is not None:
        print(f"  Box Recall: {pretrained_recall_evaluate:.4f}")
    else:
        print("  Box Recall: N/A")
    print(f"  Empty Frame Accuracy: {pretrained_evaluate.get('empty_frame_accuracy', 'N/A')}")

    # Store both for backward compatibility
    results["pretrained"] = pretrained_validate

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Comparison using trainer.validate() results
    print("\n" + "-" * 80)
    print("Using trainer.validate() results:")
    print("-" * 80)

    checkpoint_precision_validate = checkpoint_validate.get("box_precision")
    checkpoint_recall_validate = checkpoint_validate.get("box_recall")
    pretrained_precision_validate = pretrained_validate.get("box_precision")
    pretrained_recall_validate = pretrained_validate.get("box_recall")

    if checkpoint_precision_validate is not None and pretrained_precision_validate is not None:
        precision_diff = checkpoint_precision_validate - pretrained_precision_validate
        print(f"\nBox Precision:")
        print(f"  Checkpoint:  {checkpoint_precision_validate:.4f}")
        print(f"  Pretrained:  {pretrained_precision_validate:.4f}")
        if pretrained_precision_validate != 0:
            print(f"  Difference:  {precision_diff:+.4f} ({precision_diff/pretrained_precision_validate*100:+.2f}%)")
        else:
            print(f"  Difference:  {precision_diff:+.4f} (N/A%)")
    else:
        print(f"\nBox Precision: Unable to compute (missing values)")

    if checkpoint_recall_validate is not None and pretrained_recall_validate is not None:
        recall_diff = checkpoint_recall_validate - pretrained_recall_validate
        print(f"\nBox Recall:")
        print(f"  Checkpoint:  {checkpoint_recall_validate:.4f}")
        print(f"  Pretrained:  {pretrained_recall_validate:.4f}")
        if pretrained_recall_validate != 0:
            print(f"  Difference:  {recall_diff:+.4f} ({recall_diff/pretrained_recall_validate*100:+.2f}%)")
        else:
            print(f"  Difference:  {recall_diff:+.4f} (N/A%)")
    else:
        print(f"\nBox Recall: Unable to compute (missing values)")

    if "empty_frame_accuracy" in checkpoint_validate and "empty_frame_accuracy" in pretrained_validate:
        checkpoint_empty = checkpoint_validate["empty_frame_accuracy"]
        pretrained_empty = pretrained_validate["empty_frame_accuracy"]
        if checkpoint_empty is not None and pretrained_empty is not None:
            empty_diff = checkpoint_empty - pretrained_empty
            print(f"\nEmpty Frame Accuracy:")
            print(f"  Checkpoint:  {checkpoint_empty:.4f}")
            print(f"  Pretrained:  {pretrained_empty:.4f}")
            print(f"  Difference:  {empty_diff:+.4f}")
        else:
            print(f"\nEmpty Frame Accuracy: Unable to compute (missing values)")
            print(f"  Checkpoint:  {checkpoint_empty}")
            print(f"  Pretrained:  {pretrained_empty}")

    # Comparison using main.evaluate() results
    print("\n" + "-" * 80)
    print("Using main.evaluate() results:")
    print("-" * 80)

    checkpoint_precision_evaluate = checkpoint_evaluate.get("box_precision")
    checkpoint_recall_evaluate = checkpoint_evaluate.get("box_recall")
    pretrained_precision_evaluate = pretrained_evaluate.get("box_precision")
    pretrained_recall_evaluate = pretrained_evaluate.get("box_recall")

    if checkpoint_precision_evaluate is not None and pretrained_precision_evaluate is not None:
        precision_diff = checkpoint_precision_evaluate - pretrained_precision_evaluate
        print(f"\nBox Precision:")
        print(f"  Checkpoint:  {checkpoint_precision_evaluate:.4f}")
        print(f"  Pretrained:  {pretrained_precision_evaluate:.4f}")
        if pretrained_precision_evaluate != 0:
            print(f"  Difference:  {precision_diff:+.4f} ({precision_diff/pretrained_precision_evaluate*100:+.2f}%)")
        else:
            print(f"  Difference:  {precision_diff:+.4f} (N/A%)")
    else:
        print(f"\nBox Precision: Unable to compute (missing values)")

    if checkpoint_recall_evaluate is not None and pretrained_recall_evaluate is not None:
        recall_diff = checkpoint_recall_evaluate - pretrained_recall_evaluate
        print(f"\nBox Recall:")
        print(f"  Checkpoint:  {checkpoint_recall_evaluate:.4f}")
        print(f"  Pretrained:  {pretrained_recall_evaluate:.4f}")
        if pretrained_recall_evaluate != 0:
            print(f"  Difference:  {recall_diff:+.4f} ({recall_diff/pretrained_recall_evaluate*100:+.2f}%)")
        else:
            print(f"  Difference:  {recall_diff:+.4f} (N/A%)")
    else:
        print(f"\nBox Recall: Unable to compute (missing values)")

    if "empty_frame_accuracy" in checkpoint_evaluate and "empty_frame_accuracy" in pretrained_evaluate:
        checkpoint_empty = checkpoint_evaluate["empty_frame_accuracy"]
        pretrained_empty = pretrained_evaluate["empty_frame_accuracy"]
        if checkpoint_empty is not None and pretrained_empty is not None:
            empty_diff = checkpoint_empty - pretrained_empty
            print(f"\nEmpty Frame Accuracy:")
            print(f"  Checkpoint:  {checkpoint_empty:.4f}")
            print(f"  Pretrained:  {pretrained_empty:.4f}")
            print(f"  Difference:  {empty_diff:+.4f}")
        else:
            print(f"\nEmpty Frame Accuracy: Unable to compute (missing values)")
            print(f"  Checkpoint:  {checkpoint_empty}")
            print(f"  Pretrained:  {pretrained_empty}")

    print("\n" + "=" * 80)

    return results


def evaluate_multiple_thresholds(
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

    precision_scores_validate = []
    recall_scores_validate = []
    precision_scores_evaluate = []
    recall_scores_evaluate = []

    # Set up validation configuration once
    model.config.validation.csv_file = test_csv
    model.config.validation.root_dir = data_dir
    model.config.validation.iou_threshold = iou_threshold
    model.config.validation.val_accuracy_interval = 1

    print("\nEvaluating at each threshold:")
    print("-" * 80)
    for i, threshold in enumerate(thresholds):
        print(f"\n[{i+1}/{len(thresholds)}] Evaluating at score threshold: {threshold:.2f}")
        model.config.score_thresh = threshold
        model.model.score_thresh = threshold

        # Evaluate using trainer.validate()
        model.create_trainer()
        validation_results = model.trainer.validate(model)
        validate_results = validation_results[0] if validation_results else {}

        precision_validate = validate_results.get("box_precision", 0.0)
        recall_validate = validate_results.get("box_recall", 0.0)

        precision_scores_validate.append(precision_validate)
        recall_scores_validate.append(recall_validate)

        print(f"  trainer.validate() - Precision: {precision_validate:.4f}, Recall: {recall_validate:.4f}")

        # Evaluate using main.evaluate()
        evaluate_results = model.evaluate(
            csv_file=test_csv,
            root_dir=data_dir,
            iou_threshold=iou_threshold,
        )

        precision_evaluate = evaluate_results.get("box_precision", 0.0)
        recall_evaluate = evaluate_results.get("box_recall", 0.0)

        precision_scores_evaluate.append(precision_evaluate)
        recall_scores_evaluate.append(recall_evaluate)

        print(f"  main.evaluate() - Precision: {precision_evaluate:.4f}, Recall: {recall_evaluate:.4f}")

    # Create results dictionary
    threshold_results = {
        "thresholds": thresholds,
        "precision_validate": precision_scores_validate,
        "recall_validate": recall_scores_validate,
        "precision_evaluate": precision_scores_evaluate,
        "recall_evaluate": recall_scores_evaluate,
    }

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - trainer.validate()")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 40)
    for thresh, prec, rec in zip(thresholds, precision_scores_validate, recall_scores_validate):
        print(f"{thresh:<12.2f} {prec:<12.4f} {rec:<12.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE - main.evaluate()")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 40)
    for thresh, prec, rec in zip(thresholds, precision_scores_evaluate, recall_scores_evaluate):
        print(f"{thresh:<12.2f} {prec:<12.4f} {rec:<12.4f}")

    # Generate plot
    if output_path is None:
        output_path = os.path.join(data_dir, "precision_recall_curve.png")

    print(f"\nGenerating plot: {output_path}")
    plt.figure(figsize=(14, 8))
    
    # Plot trainer.validate() results
    plt.plot(thresholds, precision_scores_validate, "o-", label="Precision (trainer.validate())", linewidth=2, markersize=8, color='blue')
    plt.plot(thresholds, recall_scores_validate, "s-", label="Recall (trainer.validate())", linewidth=2, markersize=8, color='blue', linestyle='--')
    
    # Plot main.evaluate() results
    plt.plot(thresholds, precision_scores_evaluate, "o-", label="Precision (main.evaluate())", linewidth=2, markersize=8, color='red')
    plt.plot(thresholds, recall_scores_evaluate, "s-", label="Recall (main.evaluate())", linewidth=2, markersize=8, color='red', linestyle='--')
    
    plt.xlabel("Score Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Precision and Recall vs Score Threshold\n(Retrained Bird Detection Model - Both Methods)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    max_score = max(
        max(precision_scores_validate) if precision_scores_validate else 0,
        max(recall_scores_validate) if recall_scores_validate else 0,
        max(precision_scores_evaluate) if precision_scores_evaluate else 0,
        max(recall_scores_evaluate) if recall_scores_evaluate else 0,
    )
    plt.ylim(0, max_score * 1.1 if max_score > 0 else 1.0)

    # Add value labels on points for trainer.validate()
    for thresh, prec, rec in zip(thresholds, precision_scores_validate, recall_scores_validate):
        plt.annotate(
            f"{prec:.3f}",
            (thresh, prec),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color='blue',
        )
        plt.annotate(
            f"{rec:.3f}",
            (thresh, rec),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=7,
            color='blue',
        )
    
    # Add value labels on points for main.evaluate()
    for thresh, prec, rec in zip(thresholds, precision_scores_evaluate, recall_scores_evaluate):
        plt.annotate(
            f"{prec:.3f}",
            (thresh, prec),
            textcoords="offset points",
            xytext=(0, 20),
            ha="center",
            fontsize=7,
            color='red',
        )
        plt.annotate(
            f"{rec:.3f}",
            (thresh, rec),
            textcoords="offset points",
            xytext=(0, -25),
            ha="center",
            fontsize=7,
            color='red',
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return threshold_results


def run():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare retrained bird model with pretrained model"
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
        "--evaluate_thresholds",
        action="store_true",
        help="Evaluate checkpoint model at multiple score thresholds (0.1-0.5) and generate plot",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="Path to save the precision-recall plot (default: data_dir/precision_recall_curve.png)",
    )

    args = parser.parse_args()

    # Run comparison
    compare_models(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        iou_threshold=args.iou_threshold,
    )

    # Evaluate at multiple thresholds if requested
    if args.evaluate_thresholds:
        evaluate_multiple_thresholds(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            iou_threshold=args.iou_threshold,
            output_path=args.plot_output,
        )


if __name__ == "__main__":
    run()

