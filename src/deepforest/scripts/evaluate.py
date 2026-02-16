import os
from warnings import warn

import pandas as pd
from omegaconf import DictConfig

from deepforest.main import deepforest


def evaluate(
    config: DictConfig,
    ground_truth: str | None = None,
    root_dir: str | None = None,
    predictions: str | None = None,
    output_path: str | None = None,
    save_predictions: str | None = None,
) -> None:
    """Run evaluation on ground truth annotations, optionally logging to Comet.

    This function evaluates model predictions against ground truth annotations:

    1. Provide existing predictions via --predictions:
       deepforest evaluate ground_truth.csv --predictions predictions.csv

    2. Generate predictions during evaluation:
       deepforest evaluate ground_truth.csv --root-dir /path/to/images

       Optionally save generated predictions with --save-predictions:
       deepforest evaluate ground_truth.csv --root-dir /path/to/images \\
           --save-predictions predictions.csv -o eval_results.csv

    Args:
        config (DictConfig): DeepForest configuration
        ground_truth (Optional[str]): Path to ground truth CSV file with annotations. If None, uses config.validation.csv_file.
        root_dir (Optional[str]): Root directory containing images. If None, uses config value or directory of csv_file.
        predictions (Optional[str]): Path to existing predictions CSV file. If None, generates predictions.
        output_path (Optional[str]): Path to save evaluation metrics summary CSV.
        save_predictions (Optional[str]): Path to save generated predictions CSV. Only used when predictions is None.

    Returns:
        None
    """
    m = deepforest(config=config)

    if ground_truth is None:
        if config.validation.csv_file is None:
            raise ValueError(
                "No CSV file provided and config.validation.csv_file is not set"
            )
        ground_truth = config.validation.csv_file
        m.print(f"Using validation CSV from config: {ground_truth}")

    if root_dir is None:
        root_dir = config.validation.root_dir

    # TODO: Use trainer.validate in future?
    if predictions:
        predictions = pd.read_csv(predictions)

    results = m.__evaluate__(
        csv_file=ground_truth,
        root_dir=root_dir,
        predictions=predictions,
    )

    # Save generated predictions if requested and they were generated (not loaded from file)
    if save_predictions is not None and predictions is None:
        predictions_df = results.get("predictions")
        if predictions_df is not None and not predictions_df.empty:
            if os.path.dirname(save_predictions):
                os.makedirs(os.path.dirname(save_predictions), exist_ok=True)
            predictions_df.to_csv(save_predictions, index=False)
            m.print(f"Generated predictions saved to: {save_predictions}")
        else:
            warn(
                "Warning: No predictions to save (predictions dataframe is empty)",
                stacklevel=2,
            )
    elif save_predictions is not None and predictions is not None:
        warn(
            "--save-predictions is ignored when --predictions is provided (predictions already exist)",
            stacklevel=2,
        )

    # Print results to console
    m.print("Evaluation Results:")
    for key, value in results.items():
        if key not in ["predictions", "results", "ground_df", "class_recall"]:
            if value is not None:
                m.print(f"{key}: {value}")

    # Print class-specific results if available
    if results.get("class_recall") is not None:
        m.print("Class-specific Results:")
        for _, row in results["class_recall"].iterrows():
            label_name = row["label"]
            m.print(
                f"{label_name} - Recall: {row['recall']:.4f}, Precision: {row['precision']:.4f}"
            )

    # Save results to CSV if output path provided
    if output_path is not None:
        # Create a summary dataframe with evaluation metrics
        summary_data = []
        for key, value in results.items():
            if key not in ["predictions", "results", "ground_df", "class_recall"]:
                if value is not None:
                    summary_data.append({"metric": key, "value": value})

        # Add class-specific results if available
        if results.get("class_recall") is not None:
            for _, row in results["class_recall"].iterrows():
                label_name = row["label"]
                summary_data.append(
                    {"metric": f"{label_name}_Recall", "value": row["recall"]}
                )
                summary_data.append(
                    {"metric": f"{label_name}_Precision", "value": row["precision"]}
                )

        summary_df = pd.DataFrame(summary_data)
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        m.print(f"Evaluation results saved to: {output_path}")
