"""Evaluate bird detection models on DeepWater Horizon imagery.

This script:
1. Loads shapefiles from the DeepWater Horizon monitoring program
2. Creates a test.csv file
3. Evaluates both the old and new bird detection models
4. Generates visualization comparisons
"""

import os
import glob

import pandas as pd
from deepforest import main as df_main
from deepforest.preprocess import split_raster
from deepforest.utilities import read_file
from deepforest.visualize import plot_results
import geopandas as gpd


def load_shapefiles_and_create_test_csv(data_dir, output_csv="test.csv", output_dir=None):
    """Load all shapefiles and create a test.csv file.

    Args:
        data_dir: Directory containing shapefiles and images
        output_csv: Name of output CSV file
        output_dir: Directory to write CSV file (default: tries data_dir, falls back to current directory)

    Returns:
        Path to the created CSV file
    """
    output_path = os.path.join(data_dir, output_csv)
    
    # Check if CSV already exists
    if os.path.exists(output_path):
        print(f"Test CSV already exists at {output_path}, skipping creation.")
        # Verify it's readable
        try:
            existing_df = pd.read_csv(output_path)
            print(f"Found existing {output_csv} with {len(existing_df)} annotations from {len(existing_df['image_path'].unique())} images")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}. Recreating...")
        else:
            return output_path
    
    # Find all shapefiles
    shapefiles = glob.glob(os.path.join(data_dir, "*_annotated.shp"))
    print(f"Found {len(shapefiles)} shapefiles")

    all_annotations = []

    for shp_path in shapefiles:
        # Extract base name to find corresponding image
        base_name = os.path.basename(shp_path).replace("_annotated.shp", "")
        
        # Find corresponding image file
        image_files = glob.glob(os.path.join(data_dir, f"{base_name}*.jpg"))
        if not image_files:
            print(f"Warning: No image found for {base_name}")
            continue
        
        image_path = image_files[0]
        image_filename = os.path.basename(image_path)
        
        print(f"Processing {base_name}: {image_filename}")
        
        # Read shapefile directly (coordinates are already in image space)
        gdf = gpd.read_file(shp_path)
        gdf.geometry = gdf.geometry.scale(xfact=1, yfact=-1, origin=(0, 0))
        
        # Set image_path
        gdf["image_path"] = image_filename
        gdf.crs = None
        gdf["label"] = "Bird"
        gdf = gdf[gdf.geometry.notna()]
        gdf = read_file(gdf, root_dir=data_dir)

        all_annotations.append(gdf)

    # Combine all annotations
    combined_df = pd.concat(all_annotations, ignore_index=True)
        
    combined_df.to_csv(output_path, index=False)
    print(f"\nCreated {output_csv} with {len(combined_df)} annotations from {len(combined_df['image_path'].unique())} images")
    print(f"Saved to: {output_path}")
    
    return output_path


def split_test_images_for_evaluation(test_csv, data_dir, patch_size=800, patch_overlap=0, output_dir=None, split_csv_name=None):
    """Split test images into smaller patches for evaluation using split_raster.

    Args:
        test_csv: Path to test CSV file with full image annotations
        data_dir: Directory containing test images
        patch_size: Size of patches for splitting (default: 800)
        patch_overlap: Overlap between patches (default: 0)
        output_dir: Directory to save split images (default: test_splits subdirectory of test_csv location)
        split_csv_name: Name of the output CSV file (default: test_split.csv)

    Returns:
        Tuple of (split_csv_path, split_dir) where split_csv_path is the path to
        the CSV file with split image annotations and split_dir is the directory
        containing the split images
    """
    # Create output directory for split images
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(test_csv), "test_splits")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default CSV name if not provided
    if split_csv_name is None:
        split_csv_name = "test_split.csv"

    # Read the test CSV
    test_df = read_file(test_csv)
    unique_images = test_df["image_path"].unique()

    print(f"\nSplitting {len(unique_images)} test images into {patch_size}-pixel patches...")
    print(f"Output directory: {output_dir}")

    all_split_annotations = []

    for image_name in unique_images:
        image_path = os.path.join(data_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        print(f"Processing {image_name}...")

        # Get annotations for this image
        image_annotations = test_df[test_df["image_path"] == image_name].copy()

        # Create temporary CSV file for this image's annotations
        temp_annotations_file = os.path.join(output_dir, f"temp_{image_name}_annotations.csv")
        image_annotations.to_csv(temp_annotations_file, index=False)

        # Use split_raster to create crops
        split_df = split_raster(
            annotations_file=temp_annotations_file,
            path_to_raster=image_path,
            root_dir=os.path.dirname(temp_annotations_file),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            allow_empty=False,
            save_dir=output_dir,
        )

        if not split_df.empty:
            all_split_annotations.append(split_df)
            print(f"  Created {len(split_df['image_path'].unique())} patches with {len(split_df)} annotations")
        else:
            print(f"  Warning: No patches created for {image_name}")

        # Clean up temporary annotations file
        if os.path.exists(temp_annotations_file):
            os.remove(temp_annotations_file)

    # Combine all split annotations
    if all_split_annotations:
        combined_split_df = pd.concat(all_split_annotations, ignore_index=True)
        
        # Save split CSV
        split_csv_path = os.path.join(output_dir, split_csv_name)
        combined_split_df.to_csv(split_csv_path, index=False)
        
        print(f"\nCreated split test CSV with {len(combined_split_df)} annotations from {len(combined_split_df['image_path'].unique())} patches")
        print(f"Saved to: {split_csv_path}")
        
        return split_csv_path, output_dir
    else:
        raise ValueError("No split annotations were created. Check that images exist and contain valid annotations.")


def evaluate_models(checkpoint_path, data_dir, test_csv, split_dir, iou_threshold=0.4):
    """Evaluate both old and new bird detection models.

    Args:
        checkpoint_path: Path to the new checkpoint model
        data_dir: Directory containing original test data (not used for evaluation)
        test_csv: Path to split test CSV file (for evaluation)
        split_dir: Directory containing split images (for evaluation)
        iou_threshold: IoU threshold for evaluation

    Returns:
        Dictionary with evaluation results for both models
    """
    results = {}
    
    # Evaluate new checkpoint model
    print("\n" + "=" * 80)
    print("Evaluating NEW checkpoint model...")
    print("=" * 80)
    checkpoint_model = df_main.deepforest.load_from_checkpoint(checkpoint_path)
    checkpoint_model.config.score_thresh = 0.25
    checkpoint_model.model.score_thresh = 0.25
    
    # Set up validation configuration using split CSV and split directory
    checkpoint_model.config.validation.csv_file = test_csv
    checkpoint_model.config.validation.root_dir = split_dir
    checkpoint_model.config.validation.iou_threshold = iou_threshold
    checkpoint_model.config.validation.val_accuracy_interval = 1
    checkpoint_model.config.workers = 0
    checkpoint_model.create_trainer()
    
    validation_results = checkpoint_model.trainer.validate(checkpoint_model)
    checkpoint_validate = validation_results[0] if validation_results else {}
    results["checkpoint"] = checkpoint_validate
    
    print(f"Box Precision: {checkpoint_validate.get('box_precision', 'N/A')}")
    print(f"Box Recall: {checkpoint_validate.get('box_recall', 'N/A')}")
    print(f"Empty Frame Accuracy: {checkpoint_validate.get('empty_frame_accuracy', 'N/A')}")
    
    # Evaluate old pretrained model
    print("\n" + "=" * 80)
    print("Evaluating OLD pretrained model (weecology/deepforest-bird)...")
    print("=" * 80)
    pretrained_model = df_main.deepforest()
    pretrained_model.load_model("weecology/deepforest-bird")
    pretrained_model.config.score_thresh = 0.25
    pretrained_model.model.score_thresh = 0.25
    
    # Set label dictionaries to match
    pretrained_model.label_dict = {"Bird": 0}
    pretrained_model.numeric_to_label_dict = {0: "Bird"}
    pretrained_model.config.label_dict = {"Bird": 0}
    pretrained_model.config.num_classes = 1
    
    # Set up validation configuration using split CSV and split directory
    pretrained_model.config.validation.csv_file = test_csv
    pretrained_model.config.validation.root_dir = split_dir
    pretrained_model.config.validation.iou_threshold = iou_threshold
    pretrained_model.config.validation.val_accuracy_interval = 1
    pretrained_model.config.workers = 0
    pretrained_model.create_trainer()
    
    validation_results = pretrained_model.trainer.validate(pretrained_model)
    pretrained_validate = validation_results[0] if validation_results else {}
    results["pretrained"] = pretrained_validate
    
    print(f"Box Precision: {pretrained_validate.get('box_precision', 'N/A')}")
    print(f"Box Recall: {pretrained_validate.get('box_recall', 'N/A')}")
    print(f"Empty Frame Accuracy: {pretrained_validate.get('empty_frame_accuracy', 'N/A')}")
    
    return results, checkpoint_model, pretrained_model


def generate_visualizations(
    checkpoint_model,
    pretrained_model,
    data_dir,
    test_csv,
    output_dir,
    num_images=2,
):
    """Generate side-by-side visualizations comparing old and new models.

    Args:
        checkpoint_model: New checkpoint model
        pretrained_model: Old pretrained model
        data_dir: Directory containing test data
        test_csv: Path to test CSV file
        output_dir: Directory to save visualizations
        num_images: Number of images to visualize
    """
    import matplotlib.pyplot as plt
    from deepforest.utilities import read_file
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read test CSV to get image list
    test_df = read_file(test_csv)
    unique_images = test_df["image_path"].unique()[:num_images]
    
    print(f"\nGenerating visualizations for {len(unique_images)} images...")
    
    for image_name in unique_images:
        image_path = os.path.join(data_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        print(f"Processing {image_name}...")
        
        # Get ground truth
        ground_truth = test_df[test_df["image_path"] == image_name].copy()
        
        # Predict with new model
        checkpoint_predictions = checkpoint_model.predict_tile(path=image_path, patch_size=800, patch_overlap=0)
        
        # Predict with old model
        pretrained_predictions = pretrained_model.predict_tile(path=image_path, patch_size=800, patch_overlap=0)

        # Create side-by-side comparison using savedir approach
        # Save individual plots first, then combine
        base_name = os.path.splitext(image_name)[0]
        plots_dir = "/blue/ewhite/b.weinstein/bird_detector_retrain/zero_shot/avian_images_annotated/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot new model
        if len(checkpoint_predictions) > 0:
            plot_results(
                checkpoint_predictions,
                ground_truth=ground_truth,
                image=image_path,
                savedir=plots_dir,
                basename=f"{base_name}_new",
                show=False,
            )
        else:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=16)
            ax.set_title("New Retrained Model - No Predictions", fontsize=14)
            plt.savefig(os.path.join(plots_dir, f"{base_name}_new.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        # Plot old model
        if len(pretrained_predictions) > 0:
            plot_results(
                pretrained_predictions,
                ground_truth=ground_truth,
                image=image_path,
                savedir=plots_dir,
                basename=f"{base_name}_old",
                show=False,
            )
        else:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=16)
            ax.set_title("Original Pretrained Model - No Predictions", fontsize=14)
            plt.savefig(os.path.join(plots_dir, f"{base_name}_old.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        # Combine the two images side by side
        from PIL import Image as PILImage
        img1 = PILImage.open(os.path.join(plots_dir, f"{base_name}_new.png"))
        img2 = PILImage.open(os.path.join(plots_dir, f"{base_name}_old.png"))
        
        # Resize to same height
        height = max(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * height / img1.height), height), PILImage.Resampling.LANCZOS)
        img2 = img2.resize((int(img2.width * height / img2.height), height), PILImage.Resampling.LANCZOS)
        
        # Combine
        combined = PILImage.new('RGB', (img1.width + img2.width, height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))
        
        # Save
        output_path = os.path.join(plots_dir, f"{base_name}_comparison.png")
        combined.save(output_path, dpi=(300, 300))
        
        print(f"Saved: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate bird detection models on DeepWater Horizon imagery"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/blue/ewhite/b.weinstein/bird_detector_retrain/zero_shot/avian_images_annotated",
        help="Directory containing shapefiles and images",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/blue/ewhite/b.weinstein/bird_detector_retrain/2022paper/checkpoints/f92a9384135f4481b7372b85d1da5b5f.ckpt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.4,
        help="IoU threshold for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: data_dir/visualizations)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=2,
        help="Number of images to visualize",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=800,
        help="Patch size for splitting images during evaluation (default: 800)",
    )
    parser.add_argument(
        "--patch_overlap",
        type=float,
        default=0.0,
        help="Patch overlap for splitting images during evaluation (default: 0.0)",
    )
    
    args = parser.parse_args()
    
    # Set default output directory (use current directory to avoid permission issues)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "visualizations")
    
    # Step 1: Load shapefiles and create test.csv
    print("=" * 80)
    print("Step 1: Loading shapefiles and creating test.csv")
    print("=" * 80)
    test_csv = load_shapefiles_and_create_test_csv(args.data_dir)
    
    # Step 2: Split test images for evaluation
    print("\n" + "=" * 80)
    print("Step 2: Splitting test images for evaluation")
    print("=" * 80)
    split_csv, split_dir = split_test_images_for_evaluation(
        test_csv=test_csv,
        data_dir=args.data_dir,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
    )
    
    # Step 3: Evaluate models using split images
    print("\n" + "=" * 80)
    print("Step 3: Evaluating models")
    print("=" * 80)
    results, checkpoint_model, pretrained_model = evaluate_models(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        test_csv=split_csv,
        split_dir=split_dir,
        iou_threshold=args.iou_threshold,
    )
    
    # Step 4: Generate visualizations using full images
    print("\n" + "=" * 80)
    print("Step 4: Generating visualizations")
    print("=" * 80)
    generate_visualizations(
        checkpoint_model=checkpoint_model,
        pretrained_model=pretrained_model,
        data_dir=args.data_dir,
        test_csv=test_csv,
        output_dir=args.output_dir,
        num_images=args.num_images,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print("\nNew Checkpoint Model:")
    print(f"  Box Precision: {results['checkpoint'].get('box_precision', 'N/A')}")
    print(f"  Box Recall: {results['checkpoint'].get('box_recall', 'N/A')}")
    print(f"  Empty Frame Accuracy: {results['checkpoint'].get('empty_frame_accuracy', 'N/A')}")
    
    print("\nOriginal Pretrained Model:")
    print(f"  Box Precision: {results['pretrained'].get('box_precision', 'N/A')}")
    print(f"  Box Recall: {results['pretrained'].get('box_recall', 'N/A')}")
    print(f"  Empty Frame Accuracy: {results['pretrained'].get('empty_frame_accuracy', 'N/A')}")
    
    print(f"\nVisualizations saved to: {args.output_dir}")
    print(f"Original test CSV saved to: {test_csv}")
    print(f"Split test CSV saved to: {split_csv}")
    print(f"Split images directory: {split_dir}")


if __name__ == "__main__":
    main()
