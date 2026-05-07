"""Spotlight / Lightly adapter helpers.

Converts DeepForest DataFrames (read_file output or prediction tables) into formats
compatible with Renumics Spotlight and Lightly data visualization tools.

The adapter supports two output formats:
1. "objects" - Canonical format matching Spotlight's expected schema
2. "lightly" - Format compatible with Lightly's object detection conventions

Public API:
- `view_with_spotlight(df, format="lightly", out_dir=None)` - Main conversion function
- `df_to_objects_manifest(df)` - Convert DataFrame to canonical objects format
- `objects_to_lightly(manifest)` - Convert objects format to Lightly format
- `prepare_spotlight_package(gallery_dir, out_dir)` - Package gallery for Spotlight

Usage:
    # Direct conversion
    manifest = view_with_spotlight(df, format="objects")

    # Using DataFrame accessor
    lightly_data = df.spotlight(format="lightly", out_dir="export")

    # Launch Spotlight viewer (via accessor)
    df.spotlight(launch=True)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Constants
MANIFEST_VERSION = "1.0"
BBOX_FORMAT = "pixels"


def df_to_objects_manifest(df: pd.DataFrame) -> dict:
    """Convert a DeepForest-style DataFrame into the canonical objects
    manifest.

    Expected input columns: one of ['image_path','file_name','source_image'] for
    image reference, and bbox columns ['xmin','ymin','xmax','ymax'], plus
    optional 'label' and 'score'. The function is permissive and will group
    annotations by image reference.
    """
    # Choose the column that references image files
    image_col = None
    for name in ("image_path", "file_name", "source_image", "image"):
        if name in df.columns:
            image_col = name
            break
    if image_col is None:
        raise ValueError("DataFrame must contain an image reference column")

    # Required bbox columns
    for c in ("xmin", "ymin", "xmax", "ymax"):
        if c not in df.columns:
            raise ValueError(f"Missing required bbox column: {c}")

    images: list[dict] = []
    grouped_by_image = df.groupby(image_col)
    for image_name, group in grouped_by_image:
        annotations: list[dict] = []
        for _, row in group.iterrows():
            bbox = [
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            ]
            annotation = {"bbox": bbox}
            if "label" in row.index and not pd.isna(row["label"]):
                annotation["label"] = row["label"]
            if "score" in row.index and not pd.isna(row["score"]):
                annotation["score"] = float(row["score"])
            annotations.append(annotation)

        # Width/height optional if present in any row
        width = None
        height = None
        if "width" in group.columns and not group["width"].isnull().all():
            width = int(group["width"].dropna().iloc[0])
        if "height" in group.columns and not group["height"].isnull().all():
            height = int(group["height"].dropna().iloc[0])

        image_entry = {
            "file_name": str(image_name),
            "annotations": annotations,
        }

        # Only include width/height if they have valid values
        if width is not None:
            image_entry["width"] = width
        if height is not None:
            image_entry["height"] = height

        images.append(image_entry)

    manifest = {"version": MANIFEST_VERSION, "bbox_format": BBOX_FORMAT, "images": images}
    return manifest


def objects_to_lightly(manifest: dict) -> dict:
    """Map the canonical objects manifest to a Lightly-compatible format.

    This produces a dict compatible with Lightly's expected format for object detection.
    The format follows Lightly's conventions for image datasets with bounding box annotations.

    Note: This is a minimal implementation. For production use, validate against
    the official Lightly schema and adjust field names/structure as needed.
    """
    samples = []
    for image in manifest.get("images", []):
        sample = {
            "file_name": image.get("file_name"),
            "metadata": {"bbox_format": manifest.get("bbox_format", BBOX_FORMAT)},
        }

        # Add image dimensions to metadata if available
        if image.get("width") is not None:
            sample["metadata"]["width"] = image.get("width")
        if image.get("height") is not None:
            sample["metadata"]["height"] = image.get("height")

        # Format annotations for Lightly
        annotations = image.get("annotations", [])
        if annotations:
            sample["annotations"] = []
            for annotation in annotations:
                annotation_entry = {
                    "bbox": annotation.get("bbox"),
                    "category_id": annotation.get("label"),
                    "label": annotation.get("label"),
                }
                if annotation.get("score") is not None:
                    annotation_entry["score"] = annotation.get("score")
                sample["annotations"].append(annotation_entry)

        samples.append(sample)

    return {
        "samples": samples,
        "version": manifest.get("version", MANIFEST_VERSION),
        "bbox_format": manifest.get("bbox_format", BBOX_FORMAT),
    }


def view_with_spotlight(
    df: pd.DataFrame,
    *,
    format: str = "lightly",
    out_dir: str | None = None,
) -> dict:
    """Convert a DataFrame to the requested format.

    Args:
        df: DataFrame with detection results (must have image reference and bbox columns)
        format: 'objects' (canonical Spotlight format) or 'lightly' (Lightly-compatible format)
        out_dir: Optional directory to write manifest.json file

    Returns:
        Dict in the requested format

    Raises:
        ValueError: If format is unsupported or DataFrame is missing required columns
    """
    if format not in ("objects", "lightly"):
        raise ValueError(f"Unsupported format: {format}. Use 'objects' or 'lightly'")

    if df.empty:
        raise ValueError("DataFrame is empty")

    manifest = df_to_objects_manifest(df)

    if format == "objects":
        result = manifest
    elif format == "lightly":
        result = objects_to_lightly(manifest)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

    return result


def _launch_spotlight_from_manifest(
    manifest: dict, *, port: int = 8000, host: str = "localhost"
) -> None:
    """Launch Spotlight viewer from a manifest dict.

    Args:
        manifest: Manifest dict in objects or lightly format
        port: Port for Spotlight server
        host: Host for Spotlight server

    Raises:
        ImportError: If renumics-spotlight is not installed
    """
    try:
        import renumics.spotlight as spotlight
    except ImportError as e:
        raise ImportError(
            "renumics-spotlight is required for launching Spotlight viewer. "
            "Install it with: pip install 'deepforest[spotlight]' or pip install renumics-spotlight"
        ) from e

    rows = []

    # Extract data based on manifest format
    if "images" in manifest:
        data_items = [
            (image, image.get("annotations", [])) for image in manifest["images"]
        ]
        file_key = "file_name"
    elif "samples" in manifest:
        data_items = [
            (sample, sample.get("annotations", [])) for sample in manifest["samples"]
        ]
        file_key = "file_name"
    else:
        raise ValueError("Manifest must contain 'images' or 'samples' key")

    # Convert to Spotlight DataFrame format
    for item, item_annotations in data_items:
        for annotation in item_annotations:
            row = {
                "file_name": str(item[file_key]),
                "bbox_xmin": float(annotation["bbox"][0]),
                "bbox_ymin": float(annotation["bbox"][1]),
                "bbox_xmax": float(annotation["bbox"][2]),
                "bbox_ymax": float(annotation["bbox"][3]),
                "bbox_width": float(annotation["bbox"][2] - annotation["bbox"][0]),
                "bbox_height": float(annotation["bbox"][3] - annotation["bbox"][1]),
                "label": str(annotation.get("label", "unknown")),
                "score": float(annotation.get("score", 1.0)),
            }
            rows.append(row)

    if not rows:
        raise ValueError("No annotations found in manifest")

    spotlight_df = pd.DataFrame(rows)
    spotlight.show(spotlight_df, port=port, host=host)


# Provide a small DataFrame accessor so users can call `df.spotlight.view(...)`
# or `df.spotlight(format="lightly", out_dir=...)` as a convenience wrapper.
@pd.api.extensions.register_dataframe_accessor("spotlight")
class SpotlightAccessor:
    """DataFrame accessor for Spotlight/Lightly convenience helpers.

    Usage:
        df.spotlight(format="lightly", out_dir=None)

    This forwards to `view_with_spotlight` using the DataFrame as input.
    """

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._df = pandas_obj

    def __call__(self, *args, **kwargs) -> dict:
        return self.view(*args, **kwargs)

    def view(
        self,
        *,
        format: str = "lightly",
        out_dir: str | None = None,
        launch: bool = False,
        port: int = 8000,
        host: str = "localhost",
    ) -> dict:
        """Convert DataFrame to requested format and optionally launch
        Spotlight viewer.

        This is a convenience wrapper around `view_with_spotlight()` that adds
        optional viewer launch capability for interactive use.

        Args:
            format: 'objects' or 'lightly'
            out_dir: Optional directory to write manifest.json file
            launch: If True, launch Spotlight viewer in browser
            port: Port for Spotlight server (only used if launch=True)
            host: Host for Spotlight server (only used if launch=True)

        Returns:
            Dict in the requested format
        """
        result = view_with_spotlight(self._df, format=format, out_dir=out_dir)

        if launch:
            _launch_spotlight_from_manifest(result, port=port, host=host)

        return result


def prepare_spotlight_package(
    gallery_dir: str | Path, *, out_dir: str | Path
) -> dict[str, Any]:
    """Prepare a gallery directory for Spotlight visualization.

    Args:
        gallery_dir: Path to the gallery directory containing images and metadata
        out_dir: Output directory for the Spotlight package

    Returns:
        Dict containing package information and file paths

    Raises:
        FileNotFoundError: If gallery directory doesn't exist
    """
    gallery_path = Path(gallery_dir)
    out_path = Path(out_dir)

    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    metadata_files = list(gallery_path.glob("*.csv")) + list(gallery_path.glob("*.json"))

    if not metadata_files:
        raise FileNotFoundError(f"No metadata files (CSV/JSON) found in {gallery_path}")

    metadata_file = metadata_files[0]

    if metadata_file.suffix == ".csv":
        df = pd.read_csv(metadata_file)
    else:
        # Handle JSON metadata
        with open(metadata_file) as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    # Convert to Spotlight format
    spotlight_data = view_with_spotlight(df, format="lightly", out_dir=str(out_path))

    result = {
        "gallery_dir": str(gallery_path),
        "out_dir": str(out_path),
        "metadata_file": str(metadata_file),
        "num_images": len(spotlight_data.get("samples", [])),
        "manifest_path": str(out_path / "manifest.json"),
        "format": "lightly",
    }

    return result
