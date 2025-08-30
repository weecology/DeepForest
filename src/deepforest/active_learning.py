"""This module provides active learning utilities for the weecology/deepforest
library.

It includes a Config dataclass for experiment configuration, an
ActiveLearner class that wraps DeepForest's model and training routines,
and an entropy-based acquisition function for selecting unlabeled
images. Training and validation CSV files are expected to follow the
DeepForest format, containing columns for image_path, xmin, ymin, xmax,
ymax, and label.
"""

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from deepforest import main as df_main


@dataclass(frozen=True)
class Config:
    """Configuration for active learning experiments.

    Attributes:
        workdir: Working directory for logs, checkpoints, and acquisitions.
        images_dir: Directory containing all image files.
        train_csv: CSV with labeled training data (DeepForest format).
        val_csv: CSV with labeled validation data.
        classes: List of class labels.

        epochs_per_round: Training epochs per active learning round.
        batch_size: Training batch size.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        precision: Mixed precision setting (int or str, depending on PL version).
        device: Device spec, e.g., "auto", "cpu", "cuda:0".
        num_workers: Dataloader workers.
        seed: Random seed for reproducibility.
        use_release_weights: Whether to warm start from NEON release weights.

        iou_eval: IoU threshold for evaluation.

        k_per_round: Number of images to acquire per round.
        score_threshold_pred: Score threshold when generating predictions.
    """

    workdir: str
    images_dir: str
    train_csv: str
    val_csv: str
    classes: list

    # Training
    epochs_per_round: int = 10
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    precision: int = 32
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42
    use_release_weights: bool = False

    # Evaluation
    iou_eval: float = 0.5

    # Acquisition
    k_per_round: int = 50
    score_threshold_pred: float = 0.2


def _seed_everything(seed):
    """Seed Python, NumPy, and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        pl.seed_everything(seed, workers=True)
    except Exception:
        pass


def _resolve_device(device):
    """Return (accelerator, devices) tuple understood by PyTorch Lightning."""
    if device == "auto":
        return ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logging.warning("CUDA requested but not available; falling back to CPU.")
            return ("cpu", 1)
        return ("gpu", 1)
    return ("cpu", 1)


def _ensure_dir(p):
    """Create directory p (and parents) if it does not exist."""
    Path(p).mkdir(parents=True, exist_ok=True)


def _read_paths_file(path_or_list):
    """Read a file of image paths, or accept a list directly."""
    if isinstance(path_or_list, (list, tuple)):
        return [str(Path(p)) for p in path_or_list]
    p = Path(path_or_list)
    lines = p.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _image_entropy_from_predictions(pred_df, classes):
    """Compute Shannon entropy from per-class aggregated detection scores.

    Args:
        pred_df: DataFrame of predictions (DeepForest format).
        classes: List of known class labels.

    Returns:
        (entropy, n_preds, mean_score)
        - entropy: Shannon entropy of class distribution.
        - n_preds: Number of predicted boxes.
        - mean_score: Mean detection score.

    Empty predictions receive maximum uncertainty (log(C)).
    """
    if pred_df is None or len(pred_df) == 0:
        c = max(1, len(classes))
        return (math.log(c), 0, 0.0)

    # Aggregate score mass per class
    class_mass = {c: 0.0 for c in classes}
    for _, row in pred_df.iterrows():
        label = str(row.get("label", ""))
        score = float(row.get("score", 0.0))
        if label in class_mass:
            class_mass[label] += max(0.0, min(1.0, score))

    total = sum(class_mass.values())
    if total <= 0:
        c = max(1, len(classes))
        return (math.log(c), len(pred_df), 0.0)

    probs = np.array([class_mass[c] / total for c in classes], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return (entropy, int(len(pred_df)), float(pred_df["score"].mean()))


class ActiveLearner:
    """High-level wrapper for DeepForest active learning.

    Methods:
        fit_one_round() -> Path: Train for one round, return checkpoint path.
        evaluate() -> dict: Evaluate model on validation set.
        predict_images(paths) -> dict[str, DataFrame]: Run predictions on images.
        select_for_labeling(unlabeled_paths, k) -> DataFrame: Rank images by entropy.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.workdir = Path(cfg.workdir)
        self.images_dir = Path(cfg.images_dir)
        self.train_csv = Path(cfg.train_csv)
        self.val_csv = Path(cfg.val_csv)
        self.classes = list(cfg.classes)

        _ensure_dir(self.workdir)
        _ensure_dir(self.workdir / "logs")
        _ensure_dir(self.workdir / "acquisition")
        _seed_everything(cfg.seed)

        self.model = self._build_model(cfg)
        self.trainer, self._ckpt_cb = self._create_trainer(cfg, self.workdir / "logs")

        self._attach_training_data()

    def _build_model(self, cfg):
        """Initialize DeepForest model with correct class count."""
        model = df_main.deepforest()
        if cfg.use_release_weights:
            model.use_release()
        model.config["num_classes"] = len(cfg.classes)
        model.config["batch_size"] = cfg.batch_size
        return model

    def _create_trainer(self, cfg, log_dir):
        """Create PyTorch Lightning Trainer with checkpointing and early
        stopping."""
        accelerator, devices = _resolve_device(cfg.device)
        ckpt_dir = Path(log_dir) / "checkpoints"
        _ensure_dir(ckpt_dir)

        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:02d}-val_map",
            monitor="val_map",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            auto_insert_metric_name=False,
        )
        es_cb = EarlyStopping(monitor="val_map", mode="max", patience=3)

        trainer = pl.Trainer(
            max_epochs=cfg.epochs_per_round,
            accelerator=accelerator,
            devices=devices,
            precision=cfg.precision,
            default_root_dir=str(log_dir),
            callbacks=[ckpt_cb, es_cb],
            deterministic=True,
            log_every_n_steps=10,
            enable_checkpointing=True,
        )
        return trainer, ckpt_cb

    def _attach_training_data(self):
        """Attach train/val CSVs and root dirs to model config."""
        self.model.config["train"] = self.model.config.get("train", {})
        self.model.config["val"] = self.model.config.get("val", {})
        self.model.config["train"]["csv_file"] = str(self.train_csv)
        self.model.config["train"]["root_dir"] = str(self.images_dir)
        self.model.config["val"]["csv_file"] = str(self.val_csv)
        self.model.config["val"]["root_dir"] = str(self.images_dir)

    def fit_one_round(self):
        """Train for one active learning round and return best checkpoint
        path."""
        self.model.create_trainer(trainer=self.trainer)
        self.trainer.fit(self.model)
        ckpt_path = Path(
            self._ckpt_cb.best_model_path) if self._ckpt_cb else (self.workdir / "logs" /
                                                                  "checkpoints")
        logging.info("Training finished. Best checkpoint: %s", ckpt_path)
        return ckpt_path

    def evaluate(self):
        """Run evaluation on validation CSV and return results dict."""
        try:
            results = self.model.evaluate(csv_file=str(self.val_csv),
                                          root_dir=str(self.images_dir),
                                          iou_threshold=self.cfg.iou_eval,
                                          predictions=None)
            log_summary = {k: v for k, v in results.items() if not hasattr(v, "head")}
            logging.info("Evaluation: %s", log_summary)
            return dict(results)
        except Exception as e:
            logging.warning("Evaluation failed: %s", e)
            return {}

    def predict_images(self, paths):
        """Run predictions for a list of image paths, returning dict[path ->
        DataFrame]."""
        self.model.eval()
        out = {}
        for p in paths:
            p_str = str(p)
            try:
                with torch.no_grad():
                    df = self.model.predict_image(
                        image_path=p_str,
                        return_plot=False,
                        score_threshold=self.cfg.score_threshold_pred)
                if df is None:
                    df = pd.DataFrame(columns=[
                        "xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"
                    ])
            except Exception as e:
                logging.warning("Prediction error for %s: %s", p_str, e)
                df = pd.DataFrame(columns=[
                    "xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"
                ])
            out[p_str] = df
        return out

    def select_for_labeling(self, unlabeled_paths, k=None):
        """Rank unlabeled images by entropy and return top-k for labeling.

        Args:
            unlabeled_paths: List of image paths or file containing paths.
            k: Number of images to return (defaults to cfg.k_per_round).

        Returns:
            DataFrame with ranked images and entropy scores.
        """
        if k is None:
            k = self.cfg.k_per_round

        paths = _read_paths_file(unlabeled_paths)
        if not paths:
            raise ValueError("No unlabeled paths provided")

        logging.info("Acquisition over %d images", len(paths))
        preds = self.predict_images(paths)

        rows = []
        for img_path, df in preds.items():
            ent, n_preds, mean_score = _image_entropy_from_predictions(df, self.classes)
            rows.append({
                "image_path": img_path,
                "entropy": ent,
                "n_preds": n_preds,
                "mean_score": mean_score
            })

        manifest = pd.DataFrame(rows).sort_values("entropy",
                                                  ascending=False).reset_index(drop=True)
        out_path = self.workdir / "acquisition" / "selection_round.csv"
        manifest.to_csv(out_path, index=False)
        logging.info("Wrote acquisition manifest: %s", out_path)

        return manifest.head(k).copy()
