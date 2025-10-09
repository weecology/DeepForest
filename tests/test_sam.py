"""Tests for SAM polygon generation CLI tool."""
import os
import subprocess
import sys
from importlib.resources import files

import pandas as pd
from shapely import wkt
from shapely.geometry import box

from deepforest import get_data
from deepforest.scripts.sam import (
    convert_boxes_to_polygons,
    load_sam2_model,
    process_image_group,
)

SAM_SCRIPT = files("deepforest.scripts").joinpath("sam.py")


def test_load_sam2_model():
    """Test SAM2 model sucessfully loads."""
    from transformers import Sam2Model, Sam2Processor

    model, processor = load_sam2_model("facebook/sam2.1-hiera-small", device="cpu")

    assert isinstance(model, Sam2Model)
    assert isinstance(processor, Sam2Processor)


def test_process_image_group():
    """Test processing a single image with detections."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))

    df = pd.read_csv(test_csv)

    model, processor = load_sam2_model("facebook/sam2.1-hiera-small", device="cpu")

    polygons = process_image_group(
        image_path="OSBS_029.tif",
        detections=df,
        model=model,
        processor=processor,
        device="cpu",
        image_root=test_image_dir,
        box_batch_size=2
    )

    assert len(polygons) == len(df)

    # Verify all are valid WKT strings
    for poly_wkt in polygons:
        poly = wkt.loads(poly_wkt)
        assert poly is not None


def test_convert_boxes_to_polygons(tmp_path):
    """Test the main conversion function directly."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))
    output_csv = tmp_path / "polygons.csv"
    viz_dir = tmp_path / "viz"

    input_df = pd.read_csv(test_csv)

    convert_boxes_to_polygons(
        input_csv=test_csv,
        output_csv=str(output_csv),
        image_root=test_image_dir,
        box_batch_size=2,
        visualize=True,
        viz_output_dir=viz_dir
    )

    assert output_csv.exists()

    result_df = pd.read_csv(output_csv)
    assert "polygon_geometry" in result_df.columns
    assert len(result_df) == len(input_df)

    # Validate all polygons are valid WKT
    for poly_wkt in result_df["polygon_geometry"]:
        poly = wkt.loads(poly_wkt)
        assert poly is not None

    assert viz_dir.exists()
    viz_files = list(viz_dir.glob("*.png"))
    assert len(viz_files) > 0, "No visualization files were created"


def test_polygon_box_overlap():
    """Test that output polygons overlap with input bounding boxes."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))

    df = pd.read_csv(test_csv)

    model, processor = load_sam2_model("facebook/sam2.1-hiera-small", device="cpu")

    polygons = process_image_group(
        image_path="OSBS_029.tif",
        detections=df,
        model=model,
        processor=processor,
        device="cpu",
        image_root=test_image_dir,
        box_batch_size=2
    )

    # Check that each polygon overlaps with its corresponding box
    for idx, (_, row) in enumerate(df.iterrows()):
        poly = wkt.loads(polygons[idx])

        # Skip empty polygons
        if poly.is_empty:
            continue

        # Create box polygon from detection
        bbox = box(row["xmin"], row["ymin"], row["xmax"], row["ymax"])

        # Calculate intersection
        intersection = poly.intersection(bbox)

        # Assert there is overlap
        assert intersection.area > 0, f"Polygon {idx} has no overlap with its bounding box"


def test_sam_cli_end_to_end(tmp_path):
    """Test complete CLI workflow with visualization."""
    test_csv = get_data("OSBS_029.csv")
    test_image_dir = os.path.dirname(get_data("OSBS_029.tif"))

    df = pd.read_csv(test_csv)

    output_csv = tmp_path / "polygons.csv"
    viz_dir = tmp_path / "viz"

    args = [
        sys.executable,
        str(SAM_SCRIPT),
        test_csv,
        "-o", str(output_csv),
        "--image-root", test_image_dir,
        "--box-batch", "2",
        "--device", "cpu",
        "--mask-threshold", "0.5",
        "--iou-threshold", "0.5",
        "--visualize",
        "--viz-output-dir", str(viz_dir)
    ]

    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300  # 5 minute timeout for model loading
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert output_csv.exists(), f"Expected output file not found: {output_csv}"

    # Check output CSV
    df_out = pd.read_csv(output_csv)
    assert "polygon_geometry" in df_out.columns
    assert len(df_out) == len(df)

    # Verify polygons are valid WKT
    valid_polygons = 0
    for poly_wkt in df_out["polygon_geometry"]:
        poly = wkt.loads(poly_wkt)
        assert poly is not None
        if not poly.is_empty:
            valid_polygons += 1

    # At least one polygon should be non-empty
    assert valid_polygons > 0, "All polygons are empty"

    # Check visualization created
    assert viz_dir.exists()
    viz_files = list(viz_dir.glob("*.png"))
    assert len(viz_files) > 0, "No visualization files were created"
