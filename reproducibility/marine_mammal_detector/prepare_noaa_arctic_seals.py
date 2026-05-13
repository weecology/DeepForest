"""Prepare NOAA Arctic Seals 2019 RGB annotations for DeepForest.

The script reads the public LILA metadata CSV and writes DeepForest-ready CSVs
plus image download manifests. It does not download images.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

import pandas as pd

METADATA_URL = (
    "https://storage.googleapis.com/public-datasets-lila/noaa-kotz/"
    "Detections/surv_test_kamera_detections_20210212_full_paths.csv"
)
GCP_PREFIX = "gs://public-datasets-lila/noaa-kotz"
AWS_PREFIX = "s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/noaa-kotz"
AZURE_PREFIX = "https://lilawildlife.blob.core.windows.net/lila-wildlife/noaa-kotz"
SEAL_LABELS = [
    "ringed_seal",
    "unknown_seal",
    "bearded_seal",
    "ringed_pup",
    "unknown_pup",
    "bearded_pup",
]


def deepforest_image_path(container_path: str, image_prefix: str | None) -> str:
    """Return the relative path expected under DeepForest root_dir."""
    if not image_prefix:
        return container_path
    return str(PurePosixPath(image_prefix) / container_path)


def sample_images(
    metadata: pd.DataFrame,
    max_images: int | None,
    seed: int,
) -> pd.DataFrame:
    """Sample complete images while keeping all annotations per selected image."""
    if max_images is None or max_images <= 0:
        return metadata.copy()

    images = pd.Series(sorted(metadata["rgb_image_path"].unique()))
    sampled = images.sample(n=min(max_images, len(images)), random_state=seed)
    return metadata[metadata["rgb_image_path"].isin(set(sampled))].copy()


def convert_to_deepforest(
    metadata: pd.DataFrame,
    image_prefix: str | None,
) -> pd.DataFrame:
    """Convert LILA RGB boxes into DeepForest Pascal VOC-style rows."""
    converted = pd.DataFrame(
        {
            "image_path": metadata["rgb_image_path"].map(
                lambda path: deepforest_image_path(path, image_prefix)
            ),
            "xmin": metadata[["rgb_left", "rgb_right"]].min(axis=1),
            "ymin": metadata[["rgb_top", "rgb_bottom"]].min(axis=1),
            "xmax": metadata[["rgb_left", "rgb_right"]].max(axis=1),
            "ymax": metadata[["rgb_top", "rgb_bottom"]].max(axis=1),
            "label": "Object",
        }
    )
    valid = (converted["xmax"] > converted["xmin"]) & (
        converted["ymax"] > converted["ymin"]
    )
    return converted.loc[valid].copy()


def split_by_image(
    annotations: pd.DataFrame,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split annotations by image so boxes from one image stay together."""
    images = pd.Series(sorted(annotations["image_path"].unique()))
    test_images = set(
        images.sample(
            n=max(1, round(len(images) * test_fraction)),
            random_state=seed,
        )
    )
    test = annotations[annotations["image_path"].isin(test_images)].copy()
    train = annotations[~annotations["image_path"].isin(test_images)].copy()
    return train, test


def build_manifest(metadata: pd.DataFrame, image_prefix: str | None) -> pd.DataFrame:
    """Create one row per selected RGB image with cloud download locations."""
    paths = pd.Series(sorted(metadata["rgb_image_path"].unique()), name="rgb_image_path")
    manifest = paths.to_frame()
    manifest["image_path"] = manifest["rgb_image_path"].map(
        lambda path: deepforest_image_path(path, image_prefix)
    )
    manifest["gcp_uri"] = manifest["rgb_image_path"].map(
        lambda path: f"{GCP_PREFIX}/{path}"
    )
    manifest["aws_uri"] = manifest["rgb_image_path"].map(
        lambda path: f"{AWS_PREFIX}/{path}"
    )
    manifest["azure_url"] = manifest["rgb_image_path"].map(
        lambda path: f"{AZURE_PREFIX}/{path}"
    )
    return manifest


def append_to_boem(
    boem_train_csv: Path,
    boem_test_csv: Path,
    noaa_train: pd.DataFrame,
    noaa_test: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Append NOAA rows to existing BOEM train/test CSVs."""
    boem_train = pd.read_csv(boem_train_csv, low_memory=False)
    boem_test = pd.read_csv(boem_test_csv, low_memory=False)
    pd.concat([boem_train, noaa_train], ignore_index=True).to_csv(
        output_dir / "train_with_noaa_arctic_seals.csv",
        index=False,
    )
    pd.concat([boem_test, noaa_test], ignore_index=True).to_csv(
        output_dir / "test_with_noaa_arctic_seals.csv",
        index=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NOAA Arctic Seals 2019 RGB annotations to DeepForest CSVs."
    )
    parser.add_argument("--metadata-csv", default=METADATA_URL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum unique annotated RGB images to keep. Use 0 for all images.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-prefix",
        default="noaa-kotz",
        help=(
            "Relative prefix prepended to image_path in output CSVs. Download RGB "
            "images under this folder inside the DeepForest root_dir."
        ),
    )
    parser.add_argument(
        "--seal-label",
        action="append",
        dest="seal_labels",
        default=None,
        help="NOAA detection_type to include. Repeat as needed. Defaults to all seals.",
    )
    parser.add_argument("--boem-train-csv", type=Path)
    parser.add_argument("--boem-test-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = args.seal_labels or SEAL_LABELS
    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    metadata = metadata[metadata["detection_type"].isin(labels)].copy()
    sampled = sample_images(metadata, max_images=args.max_images, seed=args.seed)

    annotations = convert_to_deepforest(sampled, image_prefix=args.image_prefix)
    train, test = split_by_image(
        annotations,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    manifest = build_manifest(sampled, image_prefix=args.image_prefix)

    annotations.to_csv(args.output_dir / "noaa_arctic_seals.csv", index=False)
    train.to_csv(args.output_dir / "noaa_arctic_seals_train.csv", index=False)
    test.to_csv(args.output_dir / "noaa_arctic_seals_test.csv", index=False)
    manifest.to_csv(args.output_dir / "noaa_arctic_seals_rgb_manifest.csv", index=False)
    manifest["gcp_uri"].to_csv(
        args.output_dir / "noaa_arctic_seals_gcp_uris.txt",
        index=False,
        header=False,
    )
    manifest["aws_uri"].to_csv(
        args.output_dir / "noaa_arctic_seals_aws_uris.txt",
        index=False,
        header=False,
    )

    if args.boem_train_csv or args.boem_test_csv:
        if not args.boem_train_csv or not args.boem_test_csv:
            raise ValueError("Provide both --boem-train-csv and --boem-test-csv.")
        append_to_boem(
            boem_train_csv=args.boem_train_csv,
            boem_test_csv=args.boem_test_csv,
            noaa_train=train,
            noaa_test=test,
            output_dir=args.output_dir,
        )

    print(
        f"Wrote {len(annotations)} annotations from "
        f"{manifest['rgb_image_path'].nunique()} RGB images to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
