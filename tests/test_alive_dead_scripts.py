"""Tests for the alive/dead CropModel scripts in scripts/alive_dead."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "alive_dead"
sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("sklearn")

from build_treefinder_crops import box_sum, int_boxes, window_starts  # noqa: E402
from evaluate_cropmodel import metrics  # noqa: E402
from make_split import assign_splits  # noqa: E402
from make_split import main as make_split  # noqa: E402


@pytest.fixture
def manifest(tmp_path):
    """Tiny data root: 20 groups of 5 crops, one Dead crop per group."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(20):
        for i in range(5):
            label = "Dead" if i == 0 else "Alive"
            rel = Path("crops/test") / label / f"g{g}_{i}.png"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(
                tmp_path / rel
            )
            rows.append(
                {
                    "crop_path": str(rel),
                    "label": label,
                    "source": "test",
                    "group": f"g{g}",
                    "site": f"s{g % 4}",
                    "md5": f"{g}_{i}",
                }
            )
    (tmp_path / "manifests").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "manifests" / "test.csv", index=False)
    return tmp_path


def test_assign_splits_keeps_groups_together(manifest):
    df = pd.read_csv(manifest / "manifests" / "test.csv")
    split = assign_splits(df, n_folds=5, seed=0)
    assert set(split) == {"train", "val", "test"}
    assert df.assign(split=split).groupby("group").split.nunique().max() == 1


def test_make_split_writes_imagefolder(manifest):
    make_split(manifest, ["test.csv"], "v1", n_folds=5, seed=0, all_test=False)
    out = manifest / "splits" / "v1"
    df = pd.read_csv(out / "split.csv")
    assert len(df) == 100
    for row in df.itertuples():
        link = out / row.split / row.label / f"test__{Path(row.crop_path).name}"
        assert (
            link.is_symlink() and link.resolve() == (manifest / row.crop_path).resolve()
        )
    with pytest.raises(FileExistsError):
        make_split(manifest, ["test.csv"], "v1", n_folds=5, seed=0, all_test=False)


def test_make_split_all_test(manifest):
    make_split(manifest, ["test.csv"], "bench", n_folds=5, seed=0, all_test=True)
    df = pd.read_csv(manifest / "splits" / "bench" / "split.csv")
    assert (df.split == "test").all()


def test_metrics():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.6, 0.1, 0.0])
    m = metrics(y, p)
    assert m["dead_precision"] == pytest.approx(0.5)
    assert m["dead_recall"] == pytest.approx(0.5)
    assert m["alive_recall"] == pytest.approx(2 / 3)
    assert m["auroc"] == pytest.approx(5 / 6)


def test_box_helpers():
    mask = np.zeros((10, 10), bool)
    mask[2:4, 2:4] = True
    boxes = np.array([[0, 0, 5, 5], [5, 5, 10, 10]])
    assert box_sum(mask, boxes).tolist() == [4, 0]
    assert int_boxes(np.array([[1.5, 1.5, 3.2, 3.2]]), 2, 10, 10).tolist() == [
        [0, 0, 6, 6]
    ]
    assert window_starts(1000, 400, 0.05) == [0, 380, 600]
    assert window_starts(300, 400, 0.05) == [0]


def test_flag_crowns():
    import argparse

    from build_dte_crops import flag_crowns

    args = argparse.Namespace(
        pad=10, patch_px=1024, min_side_m=2.0, max_aspect=2.0, max_nested=0.5
    )
    crowns = pd.DataFrame(
        {
            "patch": "p",
            "gsd_m": 0.1,
            "label": "Dead",
            # whole crown, small crown, edge crown, sliver, crown nested in the first
            "crown_xmin": [100, 400, 0, 600, 110],
            "crown_ymin": [100, 400, 500, 600, 110],
            "crown_xmax": [140, 410, 30, 660, 135],
            "crown_ymax": [140, 410, 530, 610, 135],
        }
    )
    crowns[["xmin", "ymin"]] = crowns[["crown_xmin", "crown_ymin"]].to_numpy() - 10
    crowns[["xmax", "ymax"]] = crowns[["crown_xmax", "crown_ymax"]].to_numpy() + 10
    crowns["width"], crowns["height"] = (
        crowns.xmax - crowns.xmin,
        crowns.ymax - crowns.ymin,
    )
    flags = flag_crowns(crowns, args).filter(like="flag_")
    assert not flags.iloc[0].any()
    assert flags.flag_small.tolist() == [False, True, False, False, False]
    assert flags.flag_edge.tolist() == [False, False, True, False, False]
    assert flags.flag_aspect.tolist() == [False, False, False, True, False]
    assert flags.flag_nested.tolist() == [False, False, False, False, True]
