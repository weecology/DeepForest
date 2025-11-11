import os
import json
import tempfile
import pandas as pd

from deepforest.visualize.spotlight_adapter import view_with_spotlight


def make_sample_df():
    data = [
        {"image_path": "images/img_0001.png", "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1, "label": "tree", "score": 0.9, "width": 1, "height": 1},
        {"image_path": "images/img_0002.png", "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1, "label": "tree", "score": 0.8, "width": 1, "height": 1},
    ]
    return pd.DataFrame(data)


def test_view_with_spotlight_lightly(tmp_path):
    df = make_sample_df()
    out = view_with_spotlight(df, format="lightly")

    assert isinstance(out, dict)
    assert "samples" in out
    assert len(out["samples"]) == 2

    # write to disk
    out_dir = tmp_path / "pkg"
    out_dir.mkdir()
    res = view_with_spotlight(df, format="lightly", out_dir=str(out_dir))
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf8") as fh:
        loaded = json.load(fh)
    assert "samples" in loaded


def test_dataframe_spotlight_accessor(tmp_path):
    """Smoke test for the DataFrame accessor `df.spotlight`.

    Verifies the accessor returns the same dict as the direct helper and
    writes a `manifest.json` when `out_dir` is provided.
    """
    df = make_sample_df()

    out_direct = view_with_spotlight(df, format="lightly")
    out_accessor = df.spotlight(format="lightly")
    assert out_direct == out_accessor

    out_dir = tmp_path / "pkg_accessor"
    out_dir.mkdir()
    # write via accessor
    _ = df.spotlight(format="lightly", out_dir=str(out_dir))
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf8") as fh:
        loaded = json.load(fh)
    assert loaded == out_direct
import pytest
from pathlib import Path
from PIL import Image
