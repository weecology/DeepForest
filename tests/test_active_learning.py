from pathlib import Path
import math
import os
import pandas as pd
import pytest
import yaml
import numpy as np

from deepforest import active_learning as al

# two standard files in src/deepforest/data/
SRC_CSV = Path("src/deepforest/data/2018_SJER_3_252000_4107000_image_477.csv")
SRC_IMG = Path("src/deepforest/data/2018_SJER_3_252000_4107000_image_477.tif")

pytestmark = pytest.mark.skipif(
    not (SRC_CSV.exists() and SRC_IMG.exists()),
    reason="Expected CSV/TIF not found"
)

def _stage_single_asset(tmp_path: Path):
    """Copy CSV and image into a temp dataset and fix image_path to the copied TIF."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # copy image
    img = images_dir / SRC_IMG.name
    img.write_bytes(SRC_IMG.read_bytes())

    # rewrite CSV to point to the copied image
    df = pd.read_csv(SRC_CSV)
    assert "image_path" in df.columns, "CSV must include 'image_path'"
    assert "label" in df.columns, "CSV must include 'label'"
    df = df.copy()
    df["image_path"] = str(img)  # full path to the staged image

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    df.to_csv(train_csv, index=False)
    df.to_csv(val_csv, index=False)
    return train_csv, val_csv, images_dir, img

def _make_cfg(tmp_path: Path, train_csv: Path, val_csv: Path, images_dir: Path):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    return {
        "workdir": str(workdir),
        "images_dir": str(images_dir),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        # CSV has a single class '0'
        "classes": ["0"],
        "epochs_per_round": 1,
        "batch_size": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "precision": 32,
        "device": "cpu",
        "num_workers": 0,
        "seed": 123,
        "use_release_weights": False,  # keep offline & fast by default
        "iou_eval": 0.5,
        "k_per_round": 1,
        "score_threshold_pred": 0.2,
    }

def test_predict_and_acquisition_with_single_image(tmp_path):
    train_csv, val_csv, images_dir, img = _stage_single_asset(tmp_path)
    cfg = _make_cfg(tmp_path, train_csv, val_csv, images_dir)
    learner = al.ActiveLearner(cfg)

    # Predict on the single image
    preds = learner.predict_images([str(img)])
    assert set(preds.keys()) == {str(img)}
    df = preds[str(img)]
    # Even if model returns nothing, we expect these columns
    assert list(df.columns) == ["xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"]

    # Acquisition over the single image
    unlabeled = tmp_path / "unlabeled.txt"
    unlabeled.write_text(str(img) + "\n", encoding="utf-8")
    topk = learner.select_for_labeling(unlabeled, k=1)

    # Manifest exists with expected schema and bounded entropy
    manifest = Path(cfg["workdir"]) / "acquisition" / "selection_round.csv"
    assert manifest.exists()
    mdf = pd.read_csv(manifest)
    for col in ["image_path", "entropy", "n_preds", "mean_score"]:
        assert col in mdf.columns

    # With one class, entropy ∈ [0, ln(1)=0] so it must be 0
    assert pytest.approx(float(mdf["entropy"].iloc[0])) == 0.0
    assert len(topk) == 1
    assert topk["image_path"].iloc[0] == str(img)


def test_load_config_validates_and_loads(tmp_path):
    # Happy path: write a minimal valid YAML from your helper cfg
    train_csv, val_csv, images_dir, _ = _stage_single_asset(tmp_path)
    cfg = _make_cfg(tmp_path, train_csv, val_csv, images_dir)

    yml = tmp_path / "cfg.yml"
    yml.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loaded = al.load_config(str(yml))
    # Ensure all required keys survived and a couple of core values match
    for k in [
        "workdir", "images_dir", "train_csv", "val_csv", "classes",
        "epochs_per_round", "batch_size", "precision", "device",
        "iou_eval", "k_per_round", "score_threshold_pred"
    ]:
        assert k in loaded
    assert loaded["classes"] == ["0"]
    assert int(loaded["epochs_per_round"]) == 1

    # Missing required keys -> KeyError
    bad_yml = tmp_path / "bad.yml"
    bad_cfg = {k: v for k, v in cfg.items() if k not in {"classes", "workdir"}}
    bad_yml.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
    with pytest.raises(KeyError):
        al.load_config(str(bad_yml))

    # Invalid classes -> ValueError
    empty_classes_yml = tmp_path / "empty_classes.yml"
    bad_cfg2 = cfg.copy()
    bad_cfg2["classes"] = []
    empty_classes_yml.write_text(yaml.safe_dump(bad_cfg2), encoding="utf-8")
    with pytest.raises(ValueError):
        al.load_config(str(empty_classes_yml))


def test_image_entropy_from_predictions_multiclass_and_empty():
    # Two classes; score mass: class "0" gets 1.0, class "1" gets 0.5
    classes = ["0", "1"]
    df = pd.DataFrame(
        [
            {"label": "0", "score": 0.2},
            {"label": "0", "score": 0.8},
            {"label": "1", "score": 0.5},
        ]
    )
    entropy, n_preds, mean_score = al._image_entropy_from_predictions(df, classes)

    # Expected probabilities: [2/3, 1/3]
    p = np.array([2/3, 1/3], dtype=float)
    expected_entropy = float(-(p * np.log(p)).sum())
    assert pytest.approx(entropy, rel=1e-6) == expected_entropy
    assert n_preds == 3
    assert pytest.approx(mean_score, rel=1e-6) == np.mean([0.2, 0.8, 0.5])

    # Empty predictions -> maximum uncertainty log(C)
    empty_df = pd.DataFrame(columns=["label", "score"])
    entropy2, n_preds2, mean_score2 = al._image_entropy_from_predictions(empty_df, classes)
    assert pytest.approx(entropy2, rel=1e-9) == math.log(len(classes))
    assert n_preds2 == 0
    assert mean_score2 == 0.0
