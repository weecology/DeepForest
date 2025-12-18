"""Prepare bird detection training data from multiple sources.

This script collects annotations from multiple data sources, maps labels to "Bird",
creates symlinks to a single output directory, and generates train/test splits.
This is a documentation/example script - users should adapt paths to their own data.

Example paths are hardcoded below (actual data not publicly available).
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from deepforest.utilities import read_file


# Data source file paths (adapt these to your own data locations)
DATA_SOURCES = [
    "/orange/ewhite/b.weinstein/Drones_for_Ducks/uas-imagery-of-migratory-waterfowl/crowdsourced/20240220_dronesforducks_zooniverse_refined.json",
    "/orange/ewhite/b.weinstein/izembek-lagoon-waterfowl/izembek-lagoon-birds-metadata.json",
    "/orange/ewhite/b.weinstein/bird_detector/generalization/crops/training_annotations.csv",
    "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/train.csv",
]

# Nuisance labels to exclude (will be filtered out)
NUISANCE_LABELS = {"buoy", "buoys", "trash", "Trash"}

def load_coco_with_bboxes(json_file):
    """Load COCO format JSON file with bounding boxes (bbox) instead of segmentation.

    Args:
        json_file: Path to COCO JSON file with bbox annotations

    Returns:
        DataFrame with image_path, xmin, ymin, xmax, ymax, label columns
    """
    with open(json_file) as f:
        coco_data = json.load(f)

    # Create mapping from image_id to file_name
    image_ids = {image["id"]: image["file_name"] for image in coco_data["images"]}

    # Create mapping from category_id to category name (if available)
    category_ids = {}
    if "categories" in coco_data:
        category_ids = {cat["id"]: cat.get("name", f"category_{cat['id']}") for cat in coco_data["categories"]}

    annotations = []
    for annotation in coco_data["annotations"]:
        # Skip if image_id doesn't exist in images
        image_id = annotation["image_id"]
        if image_id not in image_ids:
            continue

        # COCO bbox format: [x, y, width, height] where (x, y) is top-left corner
        try:
            bbox = annotation["bbox"]
        except KeyError:
            continue

        x = bbox[0]
        y = bbox[1]
        width = bbox[2]
        height = bbox[3]

        # Convert to DeepForest format: xmin, ymin, xmax, ymax
        xmin = x
        ymin = y
        xmax = x + width
        ymax = y + height

        # Get category label
        category_id = annotation.get("category_id", 1)
        label = category_ids.get(category_id, "Bird")

        annotations.append({
            "image_path": image_ids[image_id],
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": label,
        })

    return pd.DataFrame(annotations)


def load_annotations_from_source(source_path):
    """Load annotations from a data source file.

    Args:
        source_path: Path to annotation file (CSV or JSON)

    Returns:
        DataFrame with annotations and root_dir attribute
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file does not exist: {source_path}")

    if source_path.endswith(".csv"):
        df = read_file(source_path)
    elif source_path.endswith(".json"):
        df = load_coco_with_bboxes(source_path)
    else:
        raise ValueError(f"Unsupported file type: {source_path}")

    # Add root_dir attribute (directory containing the annotation file)
    df.root_dir = os.path.dirname(source_path)
    
    return df


def map_labels_to_bird(df):
    """Map all labels to "Bird" except nuisance labels which are filtered out.

    Args:
        df: DataFrame with label column

    Returns:
        DataFrame with labels mapped to "Bird" and nuisance labels removed
    """
    # Filter out nuisance labels
    if "label" in df.columns:
        mask = ~df["label"].str.lower().isin([n.lower() for n in NUISANCE_LABELS])
        df = df[mask].copy()

        # Map all remaining labels to "Bird"
        df["label"] = "Bird"

    return df


def create_blank_images(output_dir, num_images=100, image_size=(400, 400)):
    """Create blank white images with empty annotations.

    Args:
        output_dir: Directory to save images and annotations
        num_images: Number of blank images to create
        image_size: Tuple of (width, height) for images

    Returns:
        DataFrame with empty annotations for blank images
    """
    blank_annotations = []

    for i in range(num_images):
        # Create blank white image
        blank_image = Image.new("RGB", image_size, color="white")
        image_filename = f"blank_image_{i:03d}.png"
        image_path = os.path.join(output_dir, image_filename)
        blank_image.save(image_path)

        # Create empty annotation (0,0,0,0 coordinates indicate empty frame)
        blank_annotations.append(
            {
                "image_path": image_filename,
                "xmin": 0,
                "ymin": 0,
                "xmax": 0,
                "ymax": 0,
                "label": "Bird",
            }
        )

    return pd.DataFrame(blank_annotations)


def create_symlink(source, target):
    """Create a symlink, handling existing files.

    Args:
        source: Source file path
        target: Target symlink path
    """
    # Remove target if it exists
    if os.path.exists(target) or os.path.islink(target):
        os.remove(target)

    # Create parent directory if needed
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # Create symlink
    os.symlink(source, target)


def main():
    """Main function to prepare bird detection training data."""
    parser = argparse.ArgumentParser(
        description="Prepare bird detection training data from multiple sources"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for prepared data (images and CSV files)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--num_blank_images",
        type=int,
        default=100,
        help="Number of blank white images to generate (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading annotations from multiple sources...")
    all_annotations = []
    image_files_map = {}  # Map from original path to symlink name

    # Load annotations from all sources
    for source_path in DATA_SOURCES:
        print(f"\nProcessing source: {source_path}")
        df = load_annotations_from_source(source_path)
        df = map_labels_to_bird(df)

        if df.empty:
            print(f"  No annotations after filtering for {source_path}")
            continue

        # Get root directory for images
        root_dir = df.root_dir if hasattr(df, 'root_dir') else os.path.dirname(source_path)
        
        # Special case: Drones for Ducks images are in /images subdirectory
        if "drones_for_ducks" in source_path.lower():
            root_dir = os.path.join(root_dir, "images")

        # Ensure required columns exist
        required_cols = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns {missing_cols}, skipping...")
            continue

        # Handle image paths - create symlinks
        unique_images = df["image_path"].unique()
        for img_path in unique_images:
            # Construct full source path
            if os.path.isabs(img_path):
                source_img_path = img_path
            else:
                source_img_path = os.path.join(root_dir, img_path)

            if not os.path.exists(source_img_path):
                # Try alternative locations
                alt_paths = [
                    os.path.join(root_dir, os.path.basename(img_path)),
                    os.path.join(os.path.dirname(source_path), img_path),
                ]
                found = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        source_img_path = alt_path
                        found = True
                        break
                if not found:
                    print(f"  Warning: Image not found: {source_img_path}")
                    continue

            # Create unique symlink name
            img_basename = os.path.basename(img_path)
            symlink_name = img_basename
            counter = 1
            while symlink_name in image_files_map.values():
                name, ext = os.path.splitext(img_basename)
                symlink_name = f"{name}_{counter}{ext}"
                counter += 1

            # Create symlink
            target_path = os.path.join(args.output_dir, symlink_name)
            try:
                create_symlink(source_img_path, target_path)
                image_files_map[img_path] = symlink_name
            except Exception as e:
                print(f"  Warning: Failed to create symlink for {img_path}: {e}")
                continue

            # Update image paths in dataframe to use symlink name
            df.loc[df["image_path"] == img_path, "image_path"] = symlink_name

        all_annotations.append(df)
        print(f"  Loaded {len(df)} annotations from {len(unique_images)} images")

    if not all_annotations:
        raise ValueError("No annotations were loaded from any source!")

    # Combine all annotations
    combined_df = pd.concat(all_annotations, ignore_index=True)

    # Add blank images
    print(f"\nGenerating {args.num_blank_images} blank white images...")
    blank_df = create_blank_images(args.output_dir, args.num_blank_images)
    combined_df = pd.concat([combined_df, blank_df], ignore_index=True)

    print(f"\nTotal annotations: {len(combined_df)}")
    print(f"Total unique images: {combined_df['image_path'].nunique()}")

    # Split into train/test by image_path (to avoid data leakage)
    print(f"\nSplitting into train/test ({1-args.test_size:.0%}/{args.test_size:.0%})...")
    unique_images = combined_df["image_path"].unique()
    train_images, test_images = train_test_split(
        unique_images, test_size=args.test_size, random_state=args.seed
    )

    train_df = combined_df[combined_df["image_path"].isin(train_images)].copy()
    test_df = combined_df[combined_df["image_path"].isin(test_images)].copy()

    # Save CSV files
    train_csv = os.path.join(args.output_dir, "train.csv")
    test_csv = os.path.join(args.output_dir, "test.csv")

    # Ensure required columns are present and in correct order
    required_cols = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    train_df = train_df[required_cols]
    test_df = test_df[required_cols]

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\nSaved training annotations: {train_csv} ({len(train_df)} annotations, {len(train_images)} images)")
    print(f"Saved test annotations: {test_csv} ({len(test_df)} annotations, {len(test_images)} images)")
    print(f"\nOutput directory: {args.output_dir}")
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()

