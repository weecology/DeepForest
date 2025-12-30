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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from deepforest.preprocess import split_raster
from deepforest.utilities import read_file


# Data source file paths (adapt these to your own data locations)
DATA_SOURCES = [
    "/orange/ewhite/b.weinstein/Drones_for_Ducks/uas-imagery-of-migratory-waterfowl/crowdsourced/20240220_dronesforducks_zooniverse_refined.json",
    "/orange/ewhite/b.weinstein/izembek-lagoon-waterfowl/izembek-lagoon-birds-metadata.json",
    "/orange/ewhite/b.weinstein/bird_detector/generalization/crops/training_annotations.csv",
    "/blue/ewhite/b.weinstein/BOEM/UBFAI Images with Detection Data/crops/train.csv",
]

# Nuisance labels to exclude (will be filtered out)
NUISANCE_LABELS = {"buoy", "buoys", "trash", "Trash",'boat','sargassum'}

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


def check_negative_coordinates(df):
    """Check for negative bounding box coordinates.

    Args:
        df: DataFrame with xmin, ymin, xmax, ymax columns

    Returns:
        DataFrame with rows that have negative coordinates
    """
    required_cols = ["xmin", "ymin", "xmax", "ymax"]
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    # Find rows with any negative coordinates
    negative_mask = (
        (df["xmin"] < 0) | (df["ymin"] < 0) | (df["xmax"] < 0) | (df["ymax"] < 0)
    )
    return df[negative_mask].copy()


def clip_boxes_to_image_bounds(df, image_dir):
    """Clip bounding box coordinates to image boundaries.

    Clips negative coordinates to 0 and coordinates beyond image dimensions
    to the image edges. Ensures boxes remain valid (xmax > xmin, ymax > ymin).

    Args:
        df: DataFrame with image_path, xmin, ymin, xmax, ymax columns
        image_dir: Directory containing the images

    Returns:
        DataFrame with clipped coordinates
    """
    df = df.copy()
    required_cols = ["image_path", "xmin", "ymin", "xmax", "ymax"]
    for col in required_cols:
        if col not in df.columns:
            return df

    # Track how many boxes were clipped
    clipped_count = 0
    invalid_count = 0

    # Process each unique image
    unique_images = df["image_path"].unique()
    for img_path in unique_images:
        # Get full image path
        full_img_path = os.path.join(image_dir, img_path)

        if not os.path.exists(full_img_path):
            continue

        try:
            # Load image to get dimensions
            img = Image.open(full_img_path)
            img_width, img_height = img.size

            # Get annotations for this image
            img_mask = df["image_path"] == img_path
            img_indices = df[img_mask].index

            for idx in img_indices:
                original_xmin = df.at[idx, "xmin"]
                original_ymin = df.at[idx, "ymin"]
                original_xmax = df.at[idx, "xmax"]
                original_ymax = df.at[idx, "ymax"]

                # Clip coordinates to image boundaries
                xmin = max(0, min(original_xmin, img_width - 1))
                ymin = max(0, min(original_ymin, img_height - 1))
                xmax = max(xmin + 1, min(original_xmax, img_width))
                ymax = max(ymin + 1, min(original_ymax, img_height))

                # Check if clipping occurred
                if (
                    xmin != original_xmin
                    or ymin != original_ymin
                    or xmax != original_xmax
                    or ymax != original_ymax
                ):
                    clipped_count += 1
                    df.at[idx, "xmin"] = xmin
                    df.at[idx, "ymin"] = ymin
                    df.at[idx, "xmax"] = xmax
                    df.at[idx, "ymax"] = ymax

                # Check if box is still valid
                if xmax <= xmin or ymax <= ymin:
                    invalid_count += 1

        except Exception as e:
            print(f"  Warning: Error processing image {img_path}: {e}")
            continue

    if clipped_count > 0:
        print(f"  Clipped {clipped_count} bounding boxes to image boundaries")
    if invalid_count > 0:
        print(f"  Warning: {invalid_count} boxes became invalid after clipping")

    return df


def process_izembek_with_splitting(df, root_dir, output_dir, image_files_map):
    """Process Izembek dataset by splitting images into 800-pixel crops.

    Args:
        df: DataFrame with annotations
        root_dir: Root directory for images
        output_dir: Output directory for crops
        image_files_map: Map from original path to symlink name

    Returns:
        DataFrame with crop annotations
    """
    import tempfile

    # Create temporary directory for crops
    crops_dir = os.path.join(output_dir, "izembek_crops")
    os.makedirs(crops_dir, exist_ok=True)

    crop_annotations_list = []
    unique_images = df["image_path"].unique()

    print(f"  Splitting {len(unique_images)} images into 2000-pixel crops...")

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

        # Get image basename for matching with annotations
        image_basename = os.path.basename(source_img_path)

        # Filter annotations for this image and update image_path to basename
        img_annotations = df[df["image_path"] == img_path].copy()
        if img_annotations.empty:
            continue

        # Update image_path to basename for split_raster matching
        img_annotations["image_path"] = image_basename

        # Save temporary annotations file for this image
        temp_annotations_file = os.path.join(crops_dir, f"temp_{image_basename}_annotations.csv")
        img_annotations.to_csv(temp_annotations_file, index=False)

        try:
            # Use split_raster to create crops
            crop_df = split_raster(
                annotations_file=temp_annotations_file,
                path_to_raster=source_img_path,
                root_dir=os.path.dirname(temp_annotations_file),
                patch_size=2000,
                patch_overlap=0,
                allow_empty=False,
                save_dir=crops_dir,
            )

            # Process each crop
            for crop_img_path in crop_df["image_path"].unique():
                crop_full_path = os.path.join(crops_dir, crop_img_path)

                if not os.path.exists(crop_full_path):
                    continue

                # Create unique symlink name
                crop_basename = crop_img_path
                symlink_name = crop_basename
                counter = 1
                while symlink_name in image_files_map.values():
                    name, ext = os.path.splitext(crop_basename)
                    symlink_name = f"{name}_{counter}{ext}"
                    counter += 1

                # Create symlink to crop
                target_path = os.path.join(output_dir, symlink_name)
                try:
                    create_symlink(crop_full_path, target_path)
                    image_files_map[crop_img_path] = symlink_name
                except Exception as e:
                    print(f"  Warning: Failed to create symlink for {crop_img_path}: {e}")
                    continue

                # Update image paths in crop dataframe to use symlink name
                crop_df.loc[crop_df["image_path"] == crop_img_path, "image_path"] = symlink_name

            crop_annotations_list.append(crop_df)

        except Exception as e:
            print(f"  Warning: Failed to split image {img_path}: {e}")
            continue
        finally:
            # Clean up temporary annotations file
            if os.path.exists(temp_annotations_file):
                os.remove(temp_annotations_file)

    if crop_annotations_list:
        return pd.concat(crop_annotations_list, ignore_index=True)
    else:
        return pd.DataFrame()


def filter_small_boxes(df, min_area=1, epsilon=1e-6):
    """Filter out bounding boxes with zero or single-pixel area.

    Args:
        df: DataFrame with xmin, ymin, xmax, ymax columns
        min_area: Minimum area (in pixels) for a box to be kept (default: 1)
        epsilon: Small value for floating point comparison (default: 1e-6)

    Returns:
        DataFrame with small boxes removed
    """
    df = df.copy()
    required_cols = ["xmin", "ymin", "xmax", "ymax"]
    for col in required_cols:
        if col not in df.columns:
            return df

    # Calculate width, height, and area
    width = df["xmax"] - df["xmin"]
    height = df["ymax"] - df["ymin"]
    area = width * height

    # Round area to handle floating point precision issues
    # Single-pixel boxes (width=1, height=1) should have area=1.0
    area_rounded = np.round(area, decimals=6)

    # Filter out boxes with invalid dimensions or area <= min_area
    # Filter if: width <= 0, height <= 0, or rounded area <= min_area
    # This catches single-pixel boxes (width=1, height=1, area=1)
    valid_mask = (width > epsilon) & (height > epsilon) & (area_rounded > min_area)

    removed_count = (~valid_mask).sum()
    if removed_count > 0:
        print(f"  Removed {removed_count} bounding boxes with area <= {min_area} pixel(s) or single-pixel dimensions")

    return df[valid_mask].copy()


def plot_negative_coordinate_examples(df, output_dir, root_dir=None, num_examples=3):
    """Plot examples of images with negative bounding box coordinates.

    Args:
        df: DataFrame with annotations that have negative coordinates
        output_dir: Directory to save plots
        root_dir: Root directory for images (optional)
        num_examples: Number of examples to plot (default: 3)
    """
    if df.empty:
        print("No examples with negative coordinates to plot.")
        return

    # Get unique images with negative coordinates
    unique_images = df["image_path"].unique()
    num_examples = min(num_examples, len(unique_images))

    print(f"\nPlotting {num_examples} examples with negative coordinates...")

    # Create plots directory
    plots_dir = os.path.join(output_dir, "negative_coordinate_examples")
    os.makedirs(plots_dir, exist_ok=True)

    for i, img_path in enumerate(unique_images[:num_examples]):
        # Get annotations for this image
        img_annotations = df[df["image_path"] == img_path].copy()

        # Try to load the image
        if root_dir:
            full_img_path = os.path.join(root_dir, img_path)
        else:
            full_img_path = img_path

        if not os.path.exists(full_img_path):
            print(f"  Warning: Image not found: {full_img_path}")
            continue

        try:
            img = Image.open(full_img_path)
            img_array = np.array(img)

            # Create plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img_array)
            ax.set_title(f"Image: {os.path.basename(img_path)}\nNegative coordinates detected", fontsize=12)

            # Draw bounding boxes
            for idx, row in img_annotations.iterrows():
                xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
                # Draw rectangle
                rect = plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                ax.add_patch(rect)
                # Add text with coordinates
                ax.text(
                    xmin,
                    ymin - 5,
                    f"xmin={xmin:.1f}, ymin={ymin:.1f}\nxmax={xmax:.1f}, ymax={ymax:.1f}",
                    color="red",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

            ax.axis("off")
            plt.tight_layout()

            # Save plot
            plot_filename = f"negative_coords_example_{i+1}_{os.path.basename(img_path)}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Saved: {plot_path}")

        except Exception as e:
            print(f"  Error plotting {img_path}: {e}")
            continue

    print(f"\nPlots saved to: {plots_dir}")


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

        # Special case: Izembek dataset - split into 2000-pixel crops
        if "izembek" in source_path.lower():
            print("  Using split_raster to create 2000-pixel crops with allow_empty=False")
            df = process_izembek_with_splitting(df, root_dir, args.output_dir, image_files_map)
            if df.empty:
                print(f"  No crop annotations generated for {source_path}")
                continue
            all_annotations.append(df)
            print(f"  Loaded {len(df)} crop annotations from {df['image_path'].nunique()} crop images")
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

    # Check for negative coordinates before clipping
    print("\nChecking for negative bounding box coordinates...")
    negative_coords_df = check_negative_coordinates(combined_df)
    if not negative_coords_df.empty:
        print(f"  Found {len(negative_coords_df)} annotations with negative coordinates")
        print(f"  Affected images: {negative_coords_df['image_path'].nunique()}")
        print("\n  Summary of negative coordinates:")
        print(f"    xmin < 0: {(combined_df['xmin'] < 0).sum()}")
        print(f"    ymin < 0: {(combined_df['ymin'] < 0).sum()}")
        print(f"    xmax < 0: {(combined_df['xmax'] < 0).sum()}")
        print(f"    ymax < 0: {(combined_df['ymax'] < 0).sum()}")

        # Plot examples before clipping
        plot_negative_coordinate_examples(negative_coords_df, args.output_dir, root_dir=args.output_dir)

        # Clip boxes to image boundaries
        print("\nClipping bounding boxes to image boundaries...")
        combined_df = clip_boxes_to_image_bounds(combined_df, args.output_dir)

        # Verify clipping worked
        negative_after = check_negative_coordinates(combined_df)
        if negative_after.empty:
            print("  All negative coordinates have been clipped.")
        else:
            print(f"  Warning: {len(negative_after)} annotations still have negative coordinates after clipping")
    else:
        print("  No negative coordinates found.")

    # Filter out boxes with zero or single-pixel area
    print("\nFiltering out boxes with zero or single-pixel area...")
    initial_count = len(combined_df)
    combined_df = filter_small_boxes(combined_df, min_area=1)
    removed_count = initial_count - len(combined_df)
    if removed_count > 0:
        print(f"  Removed {removed_count} boxes (kept {len(combined_df)} boxes)")

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

