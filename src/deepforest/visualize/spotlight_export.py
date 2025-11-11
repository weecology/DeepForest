"""Export DeepForest galleries to Spotlight-compatible packages.

This module converts gallery directories (created by export_to_gallery)
into formats suitable for Renumics Spotlight visualization. Unlike the
spotlight_adapter module which converts DataFrames directly, this module
works with existing gallery thumbnails and metadata to create packaged
exports.

The module provides functions for creating Spotlight manifest files and
packaging gallery contents into structured directories suitable for
Spotlight ingestion.
"""

from __future__ import annotations

import json
import shutil

# tarfile import removed - archive functionality not needed
from pathlib import Path
from typing import Any


def _read_metadata(savedir: Path) -> list[dict[str, Any]]:
    meta_path = savedir / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"metadata.json not found in {savedir}")
    with meta_path.open("r", encoding="utf8") as fh:
        return json.load(fh)


def prepare_spotlight_package(
    savedir: str | Path,
    out_dir: str | Path | None = None,
    manifest_name: str = "spotlight_manifest.json",
) -> dict[str, Any]:
    """Prepare a Spotlight-compatible package from a DeepForest gallery.

    Takes a gallery directory (created by export_to_gallery) and packages it into
    a format suitable for Renumics Spotlight ingestion. The output follows the
    canonical Spotlight schema with proper structure and metadata.

    Args:
        savedir: Path to gallery directory containing thumbnails/ and metadata.json
        out_dir: Output directory for the Spotlight package (default: savedir/spotlight_package)
        manifest_name: Name of the manifest file (default: spotlight_manifest.json)

    Returns:
        Dict with package information: out_dir, manifest path, and item count

    Raises:
        ValueError: If thumbnails directory or metadata.json not found
    """
    savedir_p = Path(savedir)
    if out_dir is None:
        out_dir_p = savedir_p / "spotlight_package"
    else:
        out_dir_p = Path(out_dir)

    images_src = savedir_p / "thumbnails"
    if not images_src.exists():
        raise ValueError(f"Thumbnails directory not found in {savedir_p}")

    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_images = out_dir_p / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    metadata = _read_metadata(savedir_p)

    # Group annotations by source image to match Spotlight schema
    images_dict: dict[str, dict[str, Any]] = {}

    for rec in metadata:
        src_fname = rec.get("filename")
        if not src_fname:
            continue
        src_path = savedir_p / src_fname
        if not src_path.exists():
            continue

        # Copy thumbnail to images directory
        dst_fname = Path(src_fname).name
        dst_path = out_images / dst_fname
        shutil.copy2(src_path, dst_path)

        # Group by source image for proper Spotlight format
        source_img = rec.get("source_image", dst_fname)
        if source_img not in images_dict:
            images_dict[source_img] = {
                "file_name": str(Path("images") / dst_fname),
                "width": rec.get("width"),
                "height": rec.get("height"),
                "annotations": [],
            }

        # Add annotation
        annotation = {"bbox": rec.get("bbox", []), "label": rec.get("label", "Unknown")}
        if rec.get("score") is not None:
            annotation["score"] = rec.get("score")

        images_dict[source_img]["annotations"].append(annotation)

    # Create proper Spotlight manifest format
    manifest = {
        "version": "1.0",
        "bbox_format": "pixels",
        "images": list(images_dict.values()),
    }

    manifest_path = out_dir_p / manifest_name
    with manifest_path.open("w", encoding="utf8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    return {
        "out_dir": str(out_dir_p),
        "manifest": str(manifest_path),
        "count": len(manifest["images"]),
    }


__all__ = ["prepare_spotlight_package"]
