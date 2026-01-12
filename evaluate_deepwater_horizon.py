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
from deepforest.utilities import read_file, shapefile_to_annotations
from deepforest.visualize import plot_results
import geopandas as gpd


def load_shapefiles_and_create_test_csv(data_dir, output_csv="test.csv"):
    """Load all shapefiles and create a test.csv file.

    Args:
        data_dir: Directory containing shapefiles and images
        output_csv: Path to output CSV file

    Returns:
        Path to the created CSV file
    """
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
        
        # Read shapefile directly
        gdf = read_file(shp_path,image_path=image_path)
        
        # If CRS is EPSG:4326 but coordinates look like image coordinates, remove CRS
        if gdf.crs and gdf.crs.to_string() == "EPSG:4326":
            # Check if coordinates are in image space (not geographic)
            bounds = gdf.total_bounds
            if bounds[0] > -180 and bounds[2] < 180 and (bounds[0] > 100 or bounds[2] < -100):
                # These are likely image coordinates, not geographic
                gdf.crs = None
        
        # Set image_path
        gdf["image_path"] = image_filename
        
        # Set label if not present
        if "label" not in gdf.columns:
            gdf["label"] = "Bird"
        
        # Convert geometry to bounding boxes if needed
        if "geometry" in gdf.columns:
            # Get bounds for each geometry
            bounds = gdf.geometry.bounds
            # Handle non-finite values
            gdf["xmin"] = bounds["minx"].fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)
            gdf["ymin"] = bounds["miny"].fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)
            gdf["xmax"] = bounds["maxx"].fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)
            gdf["ymax"] = bounds["maxy"].fillna(0).replace([float('inf'), float('-inf')], 0).astype(int)
            
            # Filter out any rows with invalid coordinates
            # Note: ymin might be negative in image coordinates, so just check xmax > xmin and ymax > ymin
            initial_count = len(gdf)
            gdf = gdf[(gdf["xmax"] > gdf["xmin"]) & (gdf["ymax"] > gdf["ymin"])]
            
            # Skip if no valid rows
            if len(gdf) == 0:
                print(f"  Warning: No valid annotations after filtering (had {initial_count} before)")
                continue
            else:
                print(f"  Loaded {len(gdf)} annotations")
        
        # Ensure label column exists (should be "Bird" for bird annotations)
        if "label" not in gdf.columns:
            gdf["label"] = "Bird"
        else:
            # Standardize label to "Bird"
            gdf["label"] = "Bird"
        
        all_annotations.append(gdf)

    # Combine all annotations
    combined_df = pd.concat(all_annotations, ignore_index=True)
    
    # Ensure we have required columns for CSV format
    if "geometry" in combined_df.columns:
        # Convert geometry to xmin, ymin, xmax, ymax if needed
        if "xmin" not in combined_df.columns:
            bounds = combined_df.geometry.bounds
            combined_df["xmin"] = bounds["minx"].astype(int)
            combined_df["ymin"] = bounds["miny"].astype(int)
            combined_df["xmax"] = bounds["maxx"].astype(int)
            combined_df["ymax"] = bounds["maxy"].astype(int)
    
    # Select required columns for CSV
    csv_columns = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    csv_df = combined_df[csv_columns].copy()
    
    output_path = os.path.join(data_dir, output_csv)
    csv_df.to_csv(output_path, index=False)
    print(f"\nCreated {output_csv} with {len(csv_df)} annotations from {len(csv_df['image_path'].unique())} images")
    print(f"Saved to: {output_path}")
    
    return output_path


def evaluate_models(checkpoint_path, data_dir, test_csv, iou_threshold=0.4):
    """Evaluate both old and new bird detection models.

    Args:
        checkpoint_path: Path to the new checkpoint model
        data_dir: Directory containing test data
        test_csv: Path to test CSV file
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
    
    # Set up validation configuration
    checkpoint_model.config.validation.csv_file = test_csv
    checkpoint_model.config.validation.root_dir = data_dir
    checkpoint_model.config.validation.iou_threshold = iou_threshold
    checkpoint_model.config.validation.val_accuracy_interval = 1
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
    
    # Set up validation configuration
    pretrained_model.config.validation.csv_file = test_csv
    pretrained_model.config.validation.root_dir = data_dir
    pretrained_model.config.validation.iou_threshold = iou_threshold
    pretrained_model.config.validation.val_accuracy_interval = 1
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
    test_df = pd.read_csv(test_csv)
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
        checkpoint_predictions = checkpoint_model.predict_image(path=image_path)
        if checkpoint_predictions is not None and len(checkpoint_predictions) > 0:
            checkpoint_predictions = checkpoint_predictions[checkpoint_predictions.score > 0.25]
            # Ensure geometry column exists for visualization
            if "geometry" not in checkpoint_predictions.columns and all(col in checkpoint_predictions.columns for col in ["xmin", "ymin", "xmax", "ymax"]):
                import shapely.geometry
                checkpoint_predictions["geometry"] = checkpoint_predictions.apply(
                    lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
                )
        else:
            checkpoint_predictions = pd.DataFrame()
        
        # Predict with old model
        pretrained_predictions = pretrained_model.predict_image(path=image_path)
        if pretrained_predictions is not None and len(pretrained_predictions) > 0:
            pretrained_predictions = pretrained_predictions[pretrained_predictions.score > 0.25]
            # Ensure geometry column exists for visualization
            if "geometry" not in pretrained_predictions.columns and all(col in pretrained_predictions.columns for col in ["xmin", "ymin", "xmax", "ymax"]):
                import shapely.geometry
                pretrained_predictions["geometry"] = pretrained_predictions.apply(
                    lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
                )
        else:
            pretrained_predictions = pd.DataFrame()
        
        # Create side-by-side comparison using savedir approach
        # Save individual plots first, then combine
        base_name = os.path.splitext(image_name)[0]
        temp_dir = os.path.join(output_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Plot new model
        if len(checkpoint_predictions) > 0:
            plot_results(
                checkpoint_predictions,
                ground_truth=ground_truth,
                image=image_path,
                savedir=temp_dir,
                basename=f"{base_name}_new",
                show=False,
            )
        else:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=16)
            ax.set_title("New Retrained Model - No Predictions", fontsize=14)
            plt.savefig(os.path.join(temp_dir, f"{base_name}_new.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        # Plot old model
        if len(pretrained_predictions) > 0:
            plot_results(
                pretrained_predictions,
                ground_truth=ground_truth,
                image=image_path,
                savedir=temp_dir,
                basename=f"{base_name}_old",
                show=False,
            )
        else:
            # Create empty plot
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center", fontsize=16)
            ax.set_title("Original Pretrained Model - No Predictions", fontsize=14)
            plt.savefig(os.path.join(temp_dir, f"{base_name}_old.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        # Combine the two images side by side
        from PIL import Image as PILImage
        img1 = PILImage.open(os.path.join(temp_dir, f"{base_name}_new.png"))
        img2 = PILImage.open(os.path.join(temp_dir, f"{base_name}_old.png"))
        
        # Resize to same height
        height = max(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * height / img1.height), height), PILImage.Resampling.LANCZOS)
        img2 = img2.resize((int(img2.width * height / img2.height), height), PILImage.Resampling.LANCZOS)
        
        # Combine
        combined = PILImage.new('RGB', (img1.width + img2.width, height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))
        
        # Save
        output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        combined.save(output_path, dpi=(300, 300))
        
        # Clean up temp files
        os.remove(os.path.join(temp_dir, f"{base_name}_new.png"))
        os.remove(os.path.join(temp_dir, f"{base_name}_old.png"))
        
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
        default="/blue/ewhite/b.weinstein/bird_detector_retrain/data/checkpoints/6181df1ab7ac40f291b863a2a9b86024.ckpt",
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
    
    args = parser.parse_args()
    
    # Set default output directory (use current directory to avoid permission issues)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "visualizations")
    
    # Step 1: Load shapefiles and create test.csv
    print("=" * 80)
    print("Step 1: Loading shapefiles and creating test.csv")
    print("=" * 80)
    test_csv = load_shapefiles_and_create_test_csv(args.data_dir)
    
    # Step 2: Evaluate models
    print("\n" + "=" * 80)
    print("Step 2: Evaluating models")
    print("=" * 80)
    results, checkpoint_model, pretrained_model = evaluate_models(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        test_csv=test_csv,
        iou_threshold=args.iou_threshold,
    )
    
    # Step 3: Generate visualizations
    print("\n" + "=" * 80)
    print("Step 3: Generating visualizations")
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
    print(f"Test CSV saved to: {test_csv}")


if __name__ == "__main__":
    main()
