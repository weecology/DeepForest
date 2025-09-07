"""This submodule provides active learning utilities for the
weecology/deepforest library.

Features:
- Configuration management via YAML files for active learning experiments.
- ActiveLearner class: wraps DeepForest model training, evaluation, prediction, and acquisition routines.
- Entropy-based acquisition function for selecting unlabeled images to label next.
- Utilities for reproducibility, device management, and data handling.
- Training and validation CSVs must follow DeepForest format: image_path, xmin, ymin, xmax, ymax, label.
- Supports iterative active learning workflows: model training, evaluation, prediction, selection, and retraining with new labels.

Intended for use in tree detection and similar object detection tasks with DeepForest.
library.
"""

from __future__ import annotations
import logging
import math
import random
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from deepforest import main as df_main


def load_config(yaml_path: str = "active_learning.yml") -> dict:
    """Load and validate configuration from YAML.

    No defaults are applied.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Required keys (must all be present in YAML)
    required = [
        # Paths & labels
        "workdir",
        "images_dir",
        "train_csv",
        "val_csv",
        "classes",
        # Training
        "epochs_per_round",
        "batch_size",
        "lr",
        "weight_decay",
        "precision",
        "device",
        "num_workers",
        "seed",
        "use_release_weights",
        # Evaluation
        "iou_eval",
        # Acquisition
        "k_per_round",
        "score_threshold_pred",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys in {yaml_path}: {missing}")

    if not isinstance(cfg["classes"], (list, tuple)) or not cfg["classes"]:
        raise ValueError("Config 'classes' must be a non-empty list.")

    return cfg


def learner_from_yaml(yaml_path: str = "active_learning.yml") -> "ActiveLearner":
    """Create an ActiveLearner by loading configuration from a YAML file."""
    cfg = load_config(yaml_path)
    return ActiveLearner(cfg)


def _seed_everything(seed: int):
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


def _resolve_device(device: str):
    """Return (accelerator, devices) tuple understood by PyTorch Lightning."""
    dev = str(device).lower()
    if dev == "auto":
        return ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
    if dev.startswith("cuda"):
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

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.workdir = Path(cfg["workdir"])
        self.images_dir = Path(cfg["images_dir"])
        self.train_csv = Path(cfg["train_csv"])
        self.val_csv = Path(cfg["val_csv"])
        self.classes = list(cfg["classes"])

        _ensure_dir(self.workdir)
        _ensure_dir(self.workdir / "logs")
        _ensure_dir(self.workdir / "acquisition")
        _seed_everything(int(cfg["seed"]))

        self.model = self._build_model(cfg)
        self.trainer, self._ckpt_cb = self._create_trainer(cfg, self.workdir / "logs")

        self._attach_training_data()

    def _build_model(self, cfg):
        """Initialize DeepForest model with correct class count."""
        model = df_main.deepforest()
        if cfg["use_release_weights"]:
            model.use_release()
        model.config["num_classes"] = len(cfg["classes"])
        model.config["batch_size"] = int(cfg["batch_size"])
        # If desired and supported by your DeepForest version, you may also
        # set optimizer hyperparameters here using cfg["lr"] / cfg["weight_decay"].
        return model

    def _create_trainer(self, cfg, log_dir):
        """Create PyTorch Lightning Trainer with checkpointing and early
        stopping."""
        accelerator, devices = _resolve_device(cfg["device"])
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
            max_epochs=int(cfg["epochs_per_round"]),
            accelerator=accelerator,
            devices=devices,
            precision=cfg["precision"],  # int 16/32 or string "bf16"
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
            results = self.model.evaluate(
                csv_file=str(self.val_csv),
                root_dir=str(self.images_dir),
                iou_threshold=float(self.cfg["iou_eval"]),
                predictions=None,
            )
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
                        score_threshold=float(self.cfg["score_threshold_pred"]),
                    )
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
            k: Number of images to return. If None, uses cfg['k_per_round'].

        Returns:
            DataFrame with ranked images and entropy scores.
        """
        if k is None:
            k = int(self.cfg["k_per_round"])

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
                "mean_score": mean_score,
            })

        manifest = pd.DataFrame(rows).sort_values("entropy",
                                                  ascending=False).reset_index(drop=True)
        out_path = self.workdir / "acquisition" / "selection_round.csv"
        manifest.to_csv(out_path, index=False)
        logging.info("Wrote acquisition manifest: %s", out_path)

        return manifest.head(k).copy()

    def append_and_retrain(self, new_labels_csv: str, round_id=None) -> dict:
        """Append new DeepForest-format labels to train_csv and retrain.

        new_labels_csv must have columns:
          image_path, xmin, ymin, xmax, ymax, label
        It may optionally include a 'round' column; if missing, round_id is used.

        Returns:
          dict with counts and checkpoint path.
        """
        new_path = Path(new_labels_csv)
        if not new_path.exists():
            raise FileNotFoundError(f"New labels CSV not found: {new_labels_csv}")

        new_df = pd.read_csv(new_path)
        required_cols = {"image_path", "xmin", "ymin", "xmax", "ymax", "label"}
        if not required_cols.issubset(set(new_df.columns)):
            raise ValueError(
                f"{new_labels_csv} must contain columns {sorted(required_cols)}")

        # Ensure labels are in cfg.classes
        before = len(new_df)
        new_df = new_df[new_df["label"].astype(str).isin(self.classes)].copy()
        filtered_out = before - len(new_df)

        # Add round column if needed
        if "round" not in new_df.columns:
            new_df["round"] = round_id if round_id is not None else 0

        # Clamp to valid bounds if any stray values slipped in
        def _clamp_row(r):
            W = None  # optional: could verify against actual image size here
            r["xmin"] = max(0, int(r["xmin"]))
            r["ymin"] = max(0, int(r["ymin"]))
            r["xmax"] = max(int(r["xmin"]) + 1, int(r["xmax"]))
            r["ymax"] = max(int(r["ymin"]) + 1, int(r["ymax"]))
            return r

        new_df = new_df.apply(_clamp_row, axis=1)

        # Load existing training CSV if present
        if Path(self.train_csv).exists():
            old_df = pd.read_csv(self.train_csv)
        else:
            old_df = pd.DataFrame(columns=list(required_cols) + ["round"])

        # Deduplicate on exact geometry and label
        key_cols = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
        merged = pd.concat([old_df, new_df], ignore_index=True)
        deduped = merged.drop_duplicates(subset=key_cols, keep="first")

        added_boxes = len(deduped) - len(old_df)
        added_images = deduped.tail(
            added_boxes)["image_path"].nunique() if added_boxes > 0 else 0

        deduped.to_csv(self.train_csv, index=False)

        logging.info(
            "Appended labels: %d boxes (%d images). Filtered-out labels not in classes: %d. New train_csv size: %d",
            added_boxes, added_images, filtered_out, len(deduped))

        ckpt = self.fit_one_round()
        return {
            "added_boxes": int(added_boxes),
            "added_images": int(added_images),
            "filtered_out": int(filtered_out),
            "checkpoint": str(ckpt),
            "train_csv_size": int(len(deduped)),
        }
