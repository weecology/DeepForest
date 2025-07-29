import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
import pycolmap
from pathlib import Path
from matplotlib import pyplot
import geopandas as gpd
from shapely.geometry import box

from deepforest import main
from hloc import extract_features, match_features, pairs_from_exhaustive
from hloc.utils.io import get_matches


def get_matching_points(h5_file, image1_name, image2_name, min_score=None):
    """Get matching points between two images from an h5 file."""
    matches, scores = get_matches(h5_file, image1_name, image2_name)
    if min_score is not None:
        matches = matches[scores > min_score]
    match_index = pd.DataFrame(matches, columns=["image1", "image2"])
    
    features_path = os.path.join(os.path.dirname(h5_file), "features.h5")
    with h5py.File(features_path, 'r') as features_h5_f:
        keypoints_image1 = pd.DataFrame(features_h5_f[image1_name]["keypoints"][:], columns=["x", "y"])
        keypoints_image2 = pd.DataFrame(features_h5_f[image2_name]["keypoints"][:], columns=["x", "y"])
        points1 = keypoints_image1.iloc[match_index["image1"].values].values
        points2 = keypoints_image2.iloc[match_index["image2"].values].values
    return points1, points2

def compute_homography_matrix(h5_file, image1_name, image2_name):
    """Compute the homography matrix between two images."""
    points1, points2 = get_matching_points(h5_file, image1_name, image2_name)
    if len(points1) < 4 or len(points2) < 4:
        raise ValueError(f"Not enough matching points (<4) found between images {image1_name} and {image2_name}")

    ransac_options = pycolmap.RANSACOptions(max_error=4.0)
    report = pycolmap.estimate_homography_matrix(points1, points2, ransac_options)

    if report is None:
        raise ValueError(f"Homography matrix estimation failed for images {image1_name} and {image2_name}")
    return report

def warp_box(xmin, ymin, xmax, ymax, homography):
    """Warp a bounding box using a homography matrix."""
    points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
    reshaped_points = points.reshape(-1, 1, 2)
    warped_points = cv2.perspectiveTransform(reshaped_points, homography).squeeze(1)
    
    warped_xmin, warped_ymin = warped_points.min(axis=0)
    warped_xmax, warped_ymax = warped_points.max(axis=0)
    return int(warped_xmin), int(warped_ymin), int(warped_xmax), int(warped_ymax)

def align_predictions(predictions, homography_matrix):
    """Aligns a DataFrame of predictions using a homography matrix."""
    transformed_predictions = predictions.copy()
    for index, row in transformed_predictions.iterrows():
        xmin, ymin, xmax, ymax = warp_box(row['xmin'], row['ymin'], row['xmax'], row['ymax'], homography_matrix)
        transformed_predictions.loc[index, ['xmin', 'ymin', 'xmax', 'ymax']] = xmin, ymin, xmax, ymax
    return transformed_predictions

def remove_predictions(src_predictions, dst_predictions, aligned_predictions, threshold, device, strategy='highest-score'):
    """Remove overlapping predictions using different strategies."""
    if strategy == "highest-score":
        dst_and_aligned_predictions = pd.concat([aligned_predictions, dst_predictions], ignore_index=True)
        boxes = torch.tensor(dst_and_aligned_predictions[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float).to(device)
        scores = torch.tensor(dst_and_aligned_predictions['score'].values, dtype=torch.float).to(device)
        
        keep_indices = torchvision.ops.nms(boxes, scores, threshold)
        indices_to_keep = dst_and_aligned_predictions.iloc[keep_indices.cpu()]
        
        src_filtered = src_predictions[src_predictions.box_id.isin(indices_to_keep.box_id)]
        dst_filtered = dst_predictions[dst_predictions.box_id.isin(indices_to_keep.box_id)]
    else:
        aligned_predictions["geometry"] = aligned_predictions.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
        dst_predictions["geometry"] = dst_predictions.apply(lambda row: box(row['xmin'], row['ymin'], row['xmax'], row['ymax']), axis=1)
        aligned_gdf = gpd.GeoDataFrame(aligned_predictions, geometry="geometry")
        dst_gdf = gpd.GeoDataFrame(dst_predictions, geometry='geometry')

        joined = gpd.sjoin(aligned_gdf, dst_gdf, how='inner', predicate='intersects')

        if strategy == "left-hand":
            src_indices_to_keep = src_predictions.box_id
            dst_indices_to_keep = dst_predictions[~dst_predictions.box_id.isin(joined.box_id_right)].box_id
        elif strategy == "right-hand":
            src_indices_to_keep = src_predictions[~src_predictions.box_id.isin(joined.box_id_left)].box_id
            dst_indices_to_keep = dst_predictions.box_id
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'highest-score', 'left-hand', 'right-hand'.")

        src_filtered = src_predictions[src_predictions.box_id.isin(src_indices_to_keep)]
        dst_filtered = dst_predictions[dst_predictions.box_id.isin(dst_indices_to_keep)]

    return src_filtered, dst_filtered

def align_and_delete(matching_h5_file, predictions, device, threshold=0.325, strategy='highest-score'):
    """Given predictions, align and delete overlapping boxes using a specified strategy."""
    image_names = sorted(predictions.image_path.unique())
    if len(image_names) < 2:
        return predictions

    predictions["box_id"] = range(len(predictions))
    filtered_predictions = {name: predictions[predictions.image_path == name] for name in image_names}
    
    num_pairs = len(image_names) * (len(image_names) - 1) // 2
    pair_count = 0

    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            src_image_name, dst_image_name = image_names[i], image_names[j]
            pair_count += 1
            print(f"Processing Pair {pair_count}/{num_pairs}: ({src_image_name}, {dst_image_name})")

            try:
                homography = compute_homography_matrix(h5_file=matching_h5_file, image1_name=src_image_name, image2_name=dst_image_name)
            except ValueError as e:
                print(f"Skipping pair, could not compute homography: {e}")
                continue

            src_preds, dst_preds = filtered_predictions[src_image_name], filtered_predictions[dst_image_name]
            
            if src_preds.empty or dst_preds.empty:
                continue

            aligned_src_preds = align_predictions(predictions=src_preds, homography_matrix=homography["H"])
            
            src_filtered, dst_filtered = remove_predictions(
                src_predictions=src_preds,
                dst_predictions=dst_preds,
                aligned_predictions=aligned_src_preds,
                threshold=threshold,
                device=device,
                strategy='left-hand'
            )
            
            filtered_predictions[src_image_name] = src_filtered
            filtered_predictions[dst_image_name] = dst_filtered
            
    return pd.concat(filtered_predictions.values()).drop_duplicates(subset="box_id")

def create_sfm_model(image_dir, output_path, references, overwrite=False):
    """Generate SfM feature files needed for matching."""
    feature_conf = extract_features.confs["disk"]
    matcher_conf = match_features.confs["disk+lightglue"]
    
    sfm_pairs, features, matches = output_path / 'pairs-sfm.txt', output_path / 'features.h5', output_path / 'matches.h5'
    
    extract_features.main(conf=feature_conf, image_dir=image_dir, image_list=references, feature_path=features, overwrite=overwrite)
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches, overwrite=overwrite)


# =============================MAIN FUNCTION====================================
def unique_predictions_images(image_dir, save_dir, strategy='highest-score', visualization=True):
    """
    High-level function to get unique predictions from a directory of overlapping images.

    Args:
        image_dir (str): Path to the directory containing input images.
        save_dir (str): Path to a directory for saving intermediate SfM files.
        strategy (str, optional): The strategy for deduplication. 
            Options: 'highest-score', 'left-hand', 'right-hand'. Defaults to 'highest-score'.
        visualization (bool, optional): If True, shows a plot comparing original and final predictions. Defaults to True.

    Returns:
        pandas.DataFrame: A DataFrame containing the final, deduplicated predictions.
    """
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = main.deepforest()
    model.use_release()
    model.to(device)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.tif', '.png', '.jpg'))])
    
    print(f"Found {len(image_files)} images to process: {image_files}")
    if not image_files:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")

    # --- 2. PRE-PROCESSING: CREATE SFM FEATURES ---
    print("\nStep 1: Creating SfM features...")
    create_sfm_model(
        image_dir=Path(image_dir),
        output_path=Path(save_dir),
        references=image_files,
        overwrite=True
    )
    print("SfM features created.")

    # 3. PREDICTION: GET INITIAL BOXES
    print("\nStep 2: Running prediction on all images...")
    all_predictions = []
    for image_file in image_files:
        print(f"Predicting on: {image_file}")
        image_path = os.path.join(image_dir, image_file)
        preds = model.predict_image(path=image_path, return_plot=False)
        if preds is not None and not preds.empty:
            preds["image_path"] = os.path.basename(image_file)
            all_predictions.append(preds)

    if not all_predictions:
        raise ValueError("No predictions were made on any images. Cannot proceed.")
        
    predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"Found {len(predictions)} total predictions before filtering.")

    # 4. DEDUPLICATION
    print("\nStep 3: Resolving overlaps using SfM...")
    matching_file = os.path.join(save_dir, "matches.h5")

    final_predictions = align_and_delete(
        predictions=predictions,
        matching_h5_file=matching_file,
        device=device,
        strategy=strategy
    )
    print(f"Overlap resolution complete. Final unique predictions: {len(final_predictions)}")

    # 5. VISUALIZATION
    if visualization and not final_predictions.empty:
        print("\nStep 4: Generating plots...")
        num_images = len(image_files)
        # Adjust subplot grid to fit all images
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        fig, axs = pyplot.subplots(rows, cols, figsize=(cols * 8, rows * 8))
        axs = axs.flatten()

        for i, image_path in enumerate(image_files):
            full_image_path = os.path.join(image_dir, image_path)
            image = cv2.imread(full_image_path)
            
            original_image_predictions = predictions[predictions["image_path"] == image_path]
            for _, row in original_image_predictions.iterrows():
                cv2.rectangle(image, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (255, 0, 0), 7) # Blue for original
            
            final_image_predictions_plot = final_predictions[final_predictions["image_path"] == image_path]
            for _, row in final_image_predictions_plot.iterrows():
                cv2.rectangle(image, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (182, 192, 255), 5) # Pink for final
                
            axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[i].set_title(f"Final predictions for {image_path}")
            axs[i].axis('off')
        
        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        pyplot.tight_layout()
        pyplot.show()

    return final_predictions


if __name__ == "__main__":
    
    base_path = r"C:\Users\Bhavya\GSoC\Predict_&_delete"
    image_directory = os.path.join(base_path, "Gregg1_2")
    save_directory = os.path.join(base_path, "Save_dir_new")

    final_results = unique_predictions_images(
        image_dir=image_directory,
        save_dir=save_directory,
        strategy='left-hand',
        visualization=True
    )

    print("\nFinal deduplicated predictions DataFrame:")
    print(final_results.head())