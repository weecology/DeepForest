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

Usage:
    # Direct conversion
    manifest = `view_with_spotlight(df, format="objects")`

    # Using DataFrame accessor
    lightly_data = df.spotlight(format="lightly", out_dir="export")

    # Export to file
    result = `view_with_spotlight(df, format="lightly", out_dir="spotlight_export")`
"""

from __future__ import annotations

import json
import os

import pandas as pd


def df_to_objects_manifest(df: pd.DataFrame) -> dict:
    """Convert a DeepForest-style DataFrame into the canonical objects
    manifest.

    Expected input columns: one of ['image_path','file_name','source_image'] for
    image reference, and bbox columns ['xmin','ymin','xmax','ymax'], plus
    optional 'label' and 'score'. The function is permissive and will group
    annotations by image reference.
    """
    # choose the column that references image files
    image_col = None
    for name in ("image_path", "file_name", "source_image", "image"):
        if name in df.columns:
            image_col = name
            break
    if image_col is None:
        raise ValueError("DataFrame must contain an image reference column")

    # required bbox columns
    for c in ("xmin", "ymin", "xmax", "ymax"):
        if c not in df.columns:
            raise ValueError(f"Missing required bbox column: {c}")

    images: list[dict] = []
    grouped = df.groupby(image_col)
    for img, group in grouped:
        anns: list[dict] = []
        for _, row in group.iterrows():
            bbox = [
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            ]
            ann = {"bbox": bbox}
            if "label" in row.index and not pd.isna(row["label"]):
                ann["label"] = row["label"]
            if "score" in row.index and not pd.isna(row["score"]):
                ann["score"] = float(row["score"])
            anns.append(ann)

        # width/height optional if present in any row
        width = None
        height = None
        if "width" in group.columns and not group["width"].isnull().all():
            width = int(group["width"].dropna().iloc[0])
        if "height" in group.columns and not group["height"].isnull().all():
            height = int(group["height"].dropna().iloc[0])

        image_entry = {
            "file_name": str(img),
            "annotations": anns,
        }

        # Only include width/height if they have valid values (schema requires integers)
        if width is not None:
            image_entry["width"] = width
        if height is not None:
            image_entry["height"] = height

        images.append(image_entry)

    manifest = {"version": "1.0", "bbox_format": "pixels", "images": images}
    return manifest


def objects_to_lightly(manifest: dict) -> dict:
    """Map the canonical objects manifest to a Lightly-compatible format.

    This produces a dict compatible with Lightly's expected format for object detection.
    The format follows Lightly's conventions for image datasets with bounding box annotations.

    Note: This is a minimal implementation. For production use, validate against
    the official Lightly schema and adjust field names/structure as needed.
    """
    samples = []
    for img in manifest.get("images", []):
        # Use 'file_name' to match Lightly conventions (not 'filepath')
        sample = {
            "file_name": img.get("file_name"),
            "metadata": {"bbox_format": manifest.get("bbox_format", "pixels")},
        }

        # Add image dimensions to metadata if available
        if img.get("width") is not None:
            sample["metadata"]["width"] = img.get("width")
        if img.get("height") is not None:
            sample["metadata"]["height"] = img.get("height")

        # Format annotations for Lightly
        anns = img.get("annotations", [])
        if anns:
            sample["annotations"] = []
            for a in anns:
                ann = {
                    "bbox": a.get("bbox"),
                    "category_id": a.get("label"),  # Lightly often uses category_id
                    "label": a.get("label"),  # Keep both for compatibility
                }
                if a.get("score") is not None:
                    ann["score"] = a.get("score")
                sample["annotations"].append(ann)

        samples.append(sample)

    return {
        "samples": samples,
        "version": manifest.get("version", "1.0"),
        "bbox_format": manifest.get("bbox_format", "pixels"),
    }


def view_with_spotlight(
    df: pd.DataFrame, *, format: str = "lightly", out_dir: str | None = None
) -> dict:
    """Convert a DataFrame to the requested format and optionally write to
    disk.

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

    # Validate DataFrame has required columns before processing
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
        # Allow df.spotlight(...) shorthand
        return self.view(*args, **kwargs)

    def view(self, *, format: str = "lightly", out_dir: str | None = None) -> dict:
        """Call the `view_with_spotlight` wrapper with this DataFrame.

        Returns the generated dict for the requested format and optionally
        writes `manifest.json` to `out_dir` when provided.
        """
        return view_with_spotlight(self._df, format=format, out_dir=out_dir)
