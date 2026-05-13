"""Prepare Zenodo marine mammal datasets for zero-shot DeepForest checks.

The script has two roles:
1. Write Zenodo download manifests without downloading image archives.
2. Convert extracted archives into DeepForest-style zero-shot CSVs when labels
   are available or when a qualitative whole-image proxy is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from PIL import Image

DATASETS = {
    "bigal_dolphins": {
        "record_id": 7013418,
        "title": "Drone images of dolphins",
        "citation": "Bigal et al. 2022",
    },
    "gray_cetaceans": {
        "record_id": 4933008,
        "title": "Drones and CNNs for cetacean species identification",
        "citation": "Gray et al. 2019",
    },
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def fetch_record(record_id: int) -> dict:
    """Fetch Zenodo record metadata from the public API."""
    url = f"https://zenodo.org/api/records/{record_id}"
    with urlopen(url) as response:
        return json.load(response)


def write_download_manifest(dataset: str, output_dir: Path) -> None:
    """Write file metadata and shell download commands for a Zenodo record."""
    dataset_config = DATASETS[dataset]
    record = fetch_record(dataset_config["record_id"])
    rows = []
    for file_info in record["files"]:
        rows.append(
            {
                "dataset": dataset,
                "record_id": record["id"],
                "doi": record["metadata"].get("doi"),
                "file_name": file_info["key"],
                "size": file_info["size"],
                "checksum": file_info.get("checksum"),
                "download_url": file_info["links"]["self"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / f"{dataset}_zenodo_files.csv", index=False)

    with (output_dir / f"{dataset}_download.sh").open("w") as download_script:
        download_script.write("#!/bin/bash\n")
        download_script.write("set -euo pipefail\n\n")
        download_script.write(f"mkdir -p {dataset}\n")
        for row in rows:
            download_script.write(
                f"curl -L '{row['download_url']}' -o '{dataset}/{row['file_name']}'\n"
            )


def prefixed_path(path: Path, root: Path, image_prefix: str | None) -> str:
    """Return the path to store in a DeepForest CSV."""
    rel_path = path.relative_to(root).as_posix()
    if image_prefix:
        return f"{image_prefix}/{rel_path}"
    return rel_path


def region_to_bbox(shape: dict) -> tuple[float, float, float, float] | None:
    """Convert VIA shape attributes to xmin/ymin/xmax/ymax."""
    name = shape.get("name")
    if name == "rect":
        x = float(shape["x"])
        y = float(shape["y"])
        width = float(shape["width"])
        height = float(shape["height"])
        return x, y, x + width, y + height

    if "all_points_x" in shape and "all_points_y" in shape:
        xs = [float(value) for value in shape["all_points_x"]]
        ys = [float(value) for value in shape["all_points_y"]]
        return min(xs), min(ys), max(xs), max(ys)

    return None


def iter_via_regions(via_path: Path) -> list[dict]:
    """Load image-level entries from common VIA JSON layouts."""
    with via_path.open() as src:
        data = json.load(src)
    if "_via_img_metadata" in data:
        return list(data["_via_img_metadata"].values())
    if isinstance(data, dict):
        return list(data.values())
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported VIA JSON structure: {via_path}")


def normalize_regions(regions) -> list[dict]:
    """Normalize VIA regions from dict/list storage to a list."""
    if isinstance(regions, dict):
        return list(regions.values())
    return list(regions or [])


def source_label(region: dict) -> str | None:
    """Return the first non-empty region attribute as source label metadata."""
    attrs = region.get("region_attributes") or {}
    for value in attrs.values():
        if value:
            return str(value)
    return None


def convert_gray_via(
    extract_root: Path,
    output_csv: Path,
    image_prefix: str | None,
) -> None:
    """Convert Gray et al. extracted VIA polygon labels to DeepForest boxes."""
    rows = []
    for via_path in sorted(extract_root.rglob("via_region_data.json")):
        image_dir = via_path.parent
        for entry in iter_via_regions(via_path):
            filename = entry.get("filename")
            if not filename:
                continue
            image_path = image_dir / filename
            if not image_path.exists():
                continue
            for region in normalize_regions(entry.get("regions")):
                bbox = region_to_bbox(region.get("shape_attributes", {}))
                if bbox is None:
                    continue
                rows.append(
                    {
                        "image_path": prefixed_path(
                            image_path,
                            root=extract_root,
                            image_prefix=image_prefix,
                        ),
                        "xmin": bbox[0],
                        "ymin": bbox[1],
                        "xmax": bbox[2],
                        "ymax": bbox[3],
                        "label": "Object",
                        "source_label": source_label(region),
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote {len(rows)} Gray et al. zero-shot annotations to {output_csv}")


def image_files(root: Path) -> list[Path]:
    """Find image files below root."""
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_bigal_source_label(path: Path, root: Path) -> str | None:
    """Infer species/condition metadata from folder names when available."""
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) <= 1:
        return None
    return "/".join(rel_parts[:-1])


def convert_bigal_whole_image(
    extract_root: Path,
    output_csv: Path,
    manifest_csv: Path,
    image_prefix: str | None,
) -> None:
    """Create qualitative whole-image proxy boxes for Bigal et al. images."""
    annotation_rows = []
    manifest_rows = []
    for path in image_files(extract_root):
        image_path = prefixed_path(path, root=extract_root, image_prefix=image_prefix)
        source = infer_bigal_source_label(path, root=extract_root)
        manifest_rows.append({"image_path": image_path, "source_label": source})
        with Image.open(path) as image:
            width, height = image.size
        annotation_rows.append(
            {
                "image_path": image_path,
                "xmin": 0,
                "ymin": 0,
                "xmax": width,
                "ymax": height,
                "label": "Object",
                "source_label": source,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annotation_rows).to_csv(output_csv, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(
        f"Wrote {len(annotation_rows)} Bigal et al. whole-image proxy rows to "
        f"{output_csv}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Zenodo marine mammal zero-shot data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest",
        help="Write Zenodo file manifests and download shell scripts.",
    )
    manifest.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
    )
    manifest.add_argument("--output-dir", type=Path, required=True)

    gray = subparsers.add_parser(
        "gray-cetaceans",
        help="Convert extracted Gray et al. VIA labels to a zero-shot CSV.",
    )
    gray.add_argument("--extract-root", type=Path, required=True)
    gray.add_argument("--output-csv", type=Path, required=True)
    gray.add_argument("--image-prefix", default="gray_cetaceans")

    bigal = subparsers.add_parser(
        "bigal-dolphins",
        help="Create qualitative whole-image zero-shot rows from extracted images.",
    )
    bigal.add_argument("--extract-root", type=Path, required=True)
    bigal.add_argument("--output-csv", type=Path, required=True)
    bigal.add_argument("--manifest-csv", type=Path, required=True)
    bigal.add_argument("--image-prefix", default="bigal_dolphins")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        datasets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
        for dataset in datasets:
            write_download_manifest(dataset=dataset, output_dir=args.output_dir)
        return

    if args.command == "gray-cetaceans":
        convert_gray_via(
            extract_root=args.extract_root,
            output_csv=args.output_csv,
            image_prefix=args.image_prefix,
        )
        return

    if args.command == "bigal-dolphins":
        convert_bigal_whole_image(
            extract_root=args.extract_root,
            output_csv=args.output_csv,
            manifest_csv=args.manifest_csv,
            image_prefix=args.image_prefix,
        )


if __name__ == "__main__":
    main()
