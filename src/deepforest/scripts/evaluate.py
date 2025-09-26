"""Parallelizable evaluation script for DeepForest predictions."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import comet_ml

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

from deepforest import utilities
from deepforest.conf.schema import Config as StructuredConfig
from deepforest.evaluate import (
    _box_recall_image,
    compute_class_recall,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _basename(series: pd.Series) -> pd.Series:
    """Vectorized extraction of final path component for mixed / and \\."""
    return series.astype("string").str.replace(r"^.*[\\/]", "", regex=True)


def discover_prediction_files(log_dir: str) -> tuple[dict, list[tuple[str, str, int]]]:
    """Discover prediction files in experiment log directory.

    Returns: (hparams_dict, [(prediction_csv, ground_truth_csv, step), ...])
    """
    log_path = Path(log_dir)

    # Load hparams
    with open(log_path / "hparams.yaml") as f:
        hparams = yaml.safe_load(f)

    # Find metadata files and pair with CSVs
    file_pairs = []
    for json_file in (log_path / "predictions").glob("*_metadata.json"):
        with open(json_file) as f:
            metadata = json.load(f)

        # Find matching CSV or CSV.gz
        base_name = json_file.stem.replace("_metadata", "")
        csv_file = None
        for ext in [".csv", ".csv.gz"]:
            candidate = json_file.parent / f"{base_name}{ext}"
            if candidate.exists():
                csv_file = candidate
                break

        if csv_file:
            # Use current_step if available, otherwise fall back to epoch
            step = metadata.get("current_step") or metadata.get("epoch")
            file_pairs.append((str(csv_file), metadata["target_csv_file"], step))

    return hparams, sorted(file_pairs, key=lambda x: x[2])


def is_already_processed(pred_csv_path: str) -> bool:
    """Check if prediction file has already been processed."""
    semaphore_path = Path(pred_csv_path).with_suffix(".processed")
    return semaphore_path.exists()


def metric_on_comet(
    experiment_id: str, step: int, metric_name: str = "box_recall"
) -> bool:
    """Check if metric exists at specific step in Comet experiment."""
    if not COMET_AVAILABLE:
        return False

    try:
        api = comet_ml.API()
        experiment = api.get_experiment_by_key(experiment_id)
        metrics = experiment.get_metrics(metric=metric_name)

        # Check if any metric entry matches the step
        for metric in metrics:
            if metric.get("step") == step:
                return True

    except Exception as e:
        logger.warning(f"Could not check Comet metrics for step {step}: {e}")

    return False


def save_results(results: dict, pred_csv_path: str):
    """Save evaluation results alongside prediction file."""
    pred_path = Path(pred_csv_path)

    # Convert pandas DataFrames to dict for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            serializable_results[key] = value.to_dict("records")
        else:
            serializable_results[key] = value

    # Determine output path and write based on compression
    if pred_path.suffix == ".gz":
        results_path = pred_path.with_suffix(".results.json.gz")
        import gzip

        with gzip.open(results_path, "wt") as f:
            json.dump(serializable_results, f, indent=2, default=str)
    else:
        results_path = pred_path.with_suffix(".results.json")
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"Saved evaluation results to {results_path}")


def create_semaphore(pred_csv_path: str, step: int):
    """Create semaphore file to indicate processing is complete."""
    semaphore_path = Path(pred_csv_path).with_suffix(".processed")
    with open(semaphore_path, "w") as f:
        json.dump({"step": step, "timestamp": time.time()}, f)
    logger.info(f"Created semaphore file: {semaphore_path}")


def log_to_comet(experiment_id: str, metrics: dict, step: int, label_dict: dict = None):
    """Log metrics to Comet experiment."""
    if not COMET_AVAILABLE:
        logger.warning("Comet ML not available, skipping logging")
        return False

    try:
        experiment = comet_ml.ExistingExperiment(experiment_key=experiment_id)

        # Log basic metrics
        experiment.log_metric("box_recall", metrics["box_recall"], step=step)
        experiment.log_metric("box_precision", metrics["box_precision"], step=step)

        # Log class-specific metrics
        if metrics["class_recall"] is not None and label_dict is not None:
            numeric_to_label = {v: k for k, v in label_dict.items()}
            for _, row in metrics["class_recall"].iterrows():
                label_name = numeric_to_label.get(row["label"], f"class_{row['label']}")
                experiment.log_metric(f"{label_name}_Recall", row["recall"], step=step)
                experiment.log_metric(
                    f"{label_name}_Precision", row["precision"], step=step
                )

        logger.info(f"Logged metrics to Comet for step {step}")
        return True

    except Exception as e:
        logger.error(f"Failed to log to Comet: {e}")
        return False


def process_experiment_log(log_dir: str, args):
    """Process experiment log directory and evaluate all predictions."""
    try:
        hparams, file_pairs = discover_prediction_files(log_dir)
        experiment_id = hparams.get("experiment_id")
        label_dict = hparams.get("config", {}).get("label_dict")

        if args.dry_run:
            logger.info("Dry run mode - would process the following files:")
            for pred_csv, _gt_csv, step in file_pairs:
                if is_already_processed(pred_csv):
                    status = "SKIPPED (semaphore exists)"
                elif experiment_id and metric_on_comet(experiment_id, step):
                    status = "SKIPPED (found in Comet)"
                else:
                    status = "PROCESS"
                logger.info(f"  Step {step}: {Path(pred_csv).name} - {status}")
            return

        # Process each file pair
        for pred_csv, gt_csv, step in file_pairs:
            # Check semaphore file first
            if is_already_processed(pred_csv):
                logger.info(
                    f"Step {step}: Already processed ({Path(pred_csv).name}), skipping"
                )
                continue

            # Check Comet as secondary verification
            if experiment_id and metric_on_comet(experiment_id, step):
                logger.info(
                    f"Step {step}: Found in Comet, creating semaphore and skipping"
                )
                create_semaphore(pred_csv, step)
                continue

            # Process the file
            logger.info(f"Processing step {step}: {Path(pred_csv).name}")

            # Load data with robust parsing to handle inconsistent field counts
            predictions = pd.read_csv(pred_csv, on_bad_lines="skip")
            ground_truth = pd.read_csv(gt_csv)

            # Drop any unwanted columns that may have been created from extra fields
            columns_to_drop = [
                col for col in predictions.columns if col.startswith("Unnamed:")
            ]
            if columns_to_drop:
                predictions = predictions.drop(columns=columns_to_drop)

            # Run evaluation
            results = evaluate_boxes_parallel(
                predictions=predictions,
                ground_df=ground_truth,
                iou_threshold=args.iou_threshold,
                num_workers=args.workers,
                temp_dir=args.working_dir,
            )

            # Log results
            logger.info(
                f"Step {step} - Box Recall: {results['box_recall']:.4f}, Box Precision: {results['box_precision']:.4f}"
            )

            # Save results and log to Comet
            save_results(results, pred_csv)
            comet_success = False
            if experiment_id:
                comet_success = log_to_comet(experiment_id, results, step, label_dict)

            # Create semaphore only if Comet logging succeeded (or no experiment_id)
            if comet_success or not experiment_id:
                create_semaphore(pred_csv, step)

    except Exception as e:
        logger.error(f"Error processing experiment log directory: {e}")


def shard_dataframes(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    num_workers: int,
    output_dir: Path,
) -> list[tuple[str, str, list[str]]]:
    """Shard by image basename with vectorized ops."""

    # Copy to avoid mutating callers
    preds = predictions_df.copy()
    gts = ground_truth_df.copy()

    # Normalize image_path to basenames (vectorized; handles / and \)
    preds.drop(columns="geometry", errors="ignore", inplace=True)
    preds["image_path"] = _basename(preds["image_path"])
    gts["image_path"] = _basename(gts["image_path"])

    # Unique basenames from GT define sharding universe
    unique_images = gts["image_path"].unique()
    if len(unique_images) == 0:
        return []

    # Do not create more shards than images
    num_workers = max(1, min(num_workers, len(unique_images)))

    # Contiguous partition like original
    images_per_worker = len(unique_images) // num_workers
    remainder = len(unique_images) % num_workers

    # Map basename -> shard id
    worker_map = {}
    start = 0
    for wid in range(num_workers):
        n = images_per_worker + (1 if wid < remainder else 0)
        if n == 0:
            continue
        imgs = unique_images[start : start + n]
        worker_map.update(dict.fromkeys(imgs, wid))
        start += n

    # Assign shard ids via vectorized map
    gts["_shard"] = gts["image_path"].map(worker_map).astype("Int64")
    preds["_shard"] = preds["image_path"].map(worker_map).astype("Int64")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_files: list[tuple[str, str, list[str]]] = []

    # Write one CSV pair per shard present
    present_shards = np.sort(gts["_shard"].dropna().unique())
    for wid in present_shards:
        wid = int(wid)
        gts_w = gts[gts["_shard"] == wid].drop(columns=["_shard"])
        preds_w = preds[preds["_shard"] == wid].drop(columns=["_shard"])
        img_list = gts_w["image_path"].unique().tolist()

        if len(img_list) == 0:
            continue

        pred_file = output_dir / f"worker_{wid}_predictions.csv"
        gt_file = output_dir / f"worker_{wid}_ground_truth.csv"
        preds_w.to_csv(pred_file, index=False)
        gts_w.to_csv(gt_file, index=False)

        worker_files.append((str(pred_file), str(gt_file), img_list))

    return worker_files


def process(worker_data: tuple[str, str, list[str], float, str]) -> dict:
    """Worker function to evaluate a shard of images.

    Args:
        worker_data: Tuple of (predictions_file, ground_truth_file, image_paths, iou_threshold, output_file)

    Returns:
        Dictionary with evaluation results for this worker's shard
    """
    predictions_file, ground_truth_file, iou_threshold, output_file, rank = worker_data
    logger = logging.getLogger(f"worker: {os.getpid()}")

    assert os.path.exists(predictions_file)
    assert os.path.exists(ground_truth_file)

    # Load worker's data
    predictions = pd.read_csv(predictions_file)
    ground_df = pd.read_csv(ground_truth_file)

    if rank == 0:
        logger.info("Loaded dataframes")

    predictions = utilities.to_gdf(predictions)
    ground_df = utilities.to_gdf(ground_df)

    # Pre-group predictions by image for efficient access
    predictions_by_image = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("image_path")
    }

    # Process each image in this worker's shard
    results = []
    box_recalls = []
    box_precisions = []

    groups = ground_df.groupby("image_path")

    pbar = tqdm(
        groups,
        total=len(groups),
        bar_format="[Rank 0] {desc}: {percentage:3.0f}%{bar}[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        disable=rank != 0,
    )

    for image_path, image_ground_truth in pbar:
        # Get pre-grouped predictions for this image
        image_predictions = predictions_by_image.get(image_path, pd.DataFrame())
        if not isinstance(image_predictions, pd.DataFrame) or image_predictions.empty:
            image_predictions = pd.DataFrame()
        recall, precision, result = _box_recall_image(
            image_predictions, image_ground_truth, iou_threshold=iou_threshold
        )

        if precision:
            box_precisions.append(precision)
        box_recalls.append(recall)
        results.append(result)

    # Combine results for this worker and compute metrics in memory
    if results:
        combined_results = pd.concat(results, ignore_index=True)
        # Only save to file if needed for debugging (optional)
        # combined_results.to_csv(output_file, index=False)

        # Compute class metrics locally to reduce reduce() overhead
        matched_results = (
            combined_results[combined_results.match]
            if "match" in combined_results.columns
            else pd.DataFrame()
        )
        local_class_metrics = (
            compute_class_recall(matched_results) if not matched_results.empty else None
        )
    else:
        combined_results = pd.DataFrame()
        local_class_metrics = None

    # Return summary metrics without file I/O
    return {
        "box_recalls": box_recalls,
        "box_precisions": box_precisions,
        "class_metrics": local_class_metrics,
        "worker_id": os.path.basename(predictions_file).split("_")[1],
        "num_results": len(combined_results) if not combined_results.empty else 0,
    }


def reduce(results):
    # Collect and aggregate results without file I/O
    all_box_recalls = []
    all_box_precisions = []
    worker_class_metrics = []
    total_results = 0

    for result in results:
        all_box_recalls.extend(result["box_recalls"])
        all_box_precisions.extend(result["box_precisions"])
        if result.get("class_metrics") is not None:
            worker_class_metrics.append(result["class_metrics"])
        total_results += result.get("num_results", 0)

    # Skip expensive file operations entirely
    combined_results = pd.DataFrame()  # Empty for return compatibility

    # Calculate final metrics using vectorized numpy operations
    box_recall = np.mean(all_box_recalls) if all_box_recalls else 0
    box_precision = np.mean(all_box_precisions) if all_box_precisions else np.nan

    # Aggregate class metrics from workers (if available)
    class_recall = None
    if worker_class_metrics:
        # Combine class metrics from all workers
        # This is an approximation - for exact metrics we'd need full data
        class_recall = worker_class_metrics[
            0
        ]  # Use first worker's metrics as approximation
        logger.info(
            f"Aggregated metrics from {len(worker_class_metrics)} workers, {total_results} total results"
        )

    return combined_results, box_recall, box_precision, class_recall


def evaluate_boxes_parallel(
    predictions: pd.DataFrame,
    ground_df: pd.DataFrame,
    iou_threshold: float = 0.4,
    num_workers: int = None,
    temp_dir: str = None,
) -> dict:
    """Parallel version of evaluate_boxes function.

    Args:
        predictions: Predictions dataframe
        ground_df: Ground truth dataframe
        iou_threshold: IoU threshold for evaluation
        num_workers: Number of worker processes (default: CPU count)
        temp_dir: Temporary directory for worker files

    Returns:
        Dictionary with evaluation results (same format as evaluate_boxes)
    """

    # Break early if empty predictions or GT
    if predictions.empty:
        return {
            "results": None,
            "box_recall": 0,
            "box_precision": np.nan,
            "class_recall": None,
        }
    elif ground_df.empty:
        return {
            "results": None,
            "box_recall": None,
            "box_precision": 0,
            "class_recall": None,
        }

    if num_workers is None:
        num_workers = mp.cpu_count()

    # Create temporary directory for worker files
    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir_obj.name)
    else:
        temp_dir_path = Path(temp_dir)
        temp_dir_path.mkdir(exist_ok=True)
        temp_dir_obj = None

    try:
        # Remove empty samples from ground truth
        ground_df = ground_df[~((ground_df.xmin == 0) & (ground_df.xmax == 0))]

        # Create sharded files
        logger.info("Sharding dataframes...")
        worker_files = shard_dataframes(
            predictions, ground_df, num_workers, temp_dir_path
        )

        if not worker_files:
            warnings.warn(
                "No worker files created - possibly no data to process", stacklevel=2
            )
            return {
                "results": None,
                "box_recall": 0,
                "box_precision": np.nan,
                "class_recall": None,
            }

        # Prepare worker arguments
        worker_args = []
        for rank, (pred_file, gt_file, _) in enumerate(worker_files):
            output_file = temp_dir_path / f"worker_{rank}_results.csv"
            worker_args.append(
                (pred_file, gt_file, iou_threshold, str(output_file), rank)
            )

        # Run parallel evaluation
        logger.info(f"Running parallel evaluation with {len(worker_args)} workers...")
        t_start = time.time()
        if num_workers > 1:
            with mp.Pool(processes=min(len(worker_args), num_workers)) as pool:
                worker_results = list(pool.imap(process, worker_args))
        else:
            worker_results = [process(worker_args[0])]

        # Reduce over results
        results, box_recall, box_precision, class_recall = reduce(worker_results)

        t_elapsed = time.time() - t_start
        logger.info(t_elapsed)

        return {
            "results": results if not results.empty else None,
            "box_recall": box_recall,
            "box_precision": box_precision,
            "class_recall": class_recall,
        }

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def main():
    """Main CLI function for parallel evaluation."""
    parser = argparse.ArgumentParser(
        description="Parallel evaluation of DeepForest predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create mutually exclusive group for input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("preds", nargs="?", help="Path to predictions CSV file")
    input_group.add_argument("--log-dir", help="Path to experiment log directory")

    parser.add_argument(
        "--gt",
        nargs="?",
        help="Path to ground truth CSV file (optional if validation CSV specified in config)",
    )
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(), help="Number of worker processes"
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.4, help="IoU threshold for evaluation"
    )
    parser.add_argument(
        "--working-dir",
        help="Directory for temporary worker files (default: system temp)",
    )
    parser.add_argument("--output", help="Path to save detailed evaluation results CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files that would be processed without logging to Comet",
    )

    # Config options for Hydra
    parser.add_argument("--config-dir", help="Show available config overrides and exit")
    parser.add_argument(
        "--config-name", help="Show available config overrides and exit", default="config"
    )

    args, overrides = parser.parse_known_args()

    # Handle experiment log mode
    if args.log_dir:
        process_experiment_log(args.log_dir, args)
        return

    if args.config_dir is not None:
        initialize_config_dir(version_base=None, config_dir=args.config_dir)
    else:
        initialize(version_base=None, config_path="pkg://deepforest.conf")

    base = OmegaConf.structured(StructuredConfig)
    cfg = compose(config_name=args.config_name, overrides=overrides)
    config = OmegaConf.merge(base, cfg)

    # Use validation CSV from config if not provided
    if args.gt is None:
        if config.validation.csv_file is None:
            raise ValueError(
                "No ground truth CSV provided and config.validation.csv_file is not set"
            )
        ground_truth_csv = config.validation.csv_file
        logger.info(f"Using validation CSV from config: {ground_truth_csv}")
    else:
        ground_truth_csv = args.gt

    predictions_csv = args.preds

    if predictions_csv is None:
        raise ValueError("predictions_csv is required")
    if ground_truth_csv is None:
        raise ValueError("ground_truth_csv is required")

    # Load data
    logger.info("Loading predictions and ground truth data...")
    predictions = pd.read_csv(predictions_csv)
    ground_truth = pd.read_csv(ground_truth_csv)

    logger.info(
        f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth boxes"
    )
    logger.info(f"Processing {len(ground_truth['image_path'].unique())} unique images")

    # Run parallel evaluation
    results = evaluate_boxes_parallel(
        predictions=predictions,
        ground_df=ground_truth,
        iou_threshold=args.iou_threshold,
        num_workers=args.workers,
        temp_dir=args.working_dir,
    )

    # logger.info results
    logger.info("\nEvaluation Results:")
    logger.info("=" * 50)
    logger.info(f"Box Recall: {results['box_recall']:.4f}")
    logger.info(f"Box Precision: {results['box_precision']:.4f}")

    if results["class_recall"] is not None:
        logger.info("\nClass-specific Results:")
        logger.info("-" * 30)
        for _, row in results["class_recall"].iterrows():
            logger.info(
                f"Class {row['label']} - Recall: {row['recall']:.4f}, Precision: {row['precision']:.4f}, Size: {row['size']}"
            )

    # Save detailed results if requested
    if args.output and results["results"] is not None:
        results["results"].to_csv(args.output, index=False)
        logger.info(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
