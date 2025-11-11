import json
from pathlib import Path

from PIL import Image

from deepforest.visualize import export_to_gallery


def test_prepare_spotlight_package(tmp_path: Path):
    """Test preparing a Spotlight package from gallery directory."""
    # create a small test image
    img_path = tmp_path / "img1.png"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(img_path)

    # build dataframe-like input via pandas DataFrame
    import pandas as pd

    df = pd.DataFrame([
        {"image_path": img_path.name, "xmin": 1, "ymin": 1, "xmax": 20, "ymax": 20, "label": "Tree", "score": 0.9}
    ])
    df.root_dir = str(tmp_path)

    gallery_dir = tmp_path / "gallery"
    export_to_gallery(df, str(gallery_dir), root_dir=None, max_crops=10)

    from deepforest.visualize.spotlight_export import prepare_spotlight_package

    out = tmp_path / "spot_pkg"
    res = prepare_spotlight_package(gallery_dir, out_dir=out)
    assert Path(res["manifest"]).exists()
    # check images dir populated
    images_dir = out / "images"
    assert images_dir.exists()
    assert any(images_dir.iterdir())

    # verify manifest content
    with open(res["manifest"]) as f:
        manifest = json.load(f)
    assert "version" in manifest
    assert "bbox_format" in manifest
    assert "images" in manifest
    assert len(manifest["images"]) > 0
