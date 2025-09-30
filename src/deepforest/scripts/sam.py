"""Convert DeepForest bounding box predictions to polygons using SAM2."""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm
from transformers import Sam2Model, Sam2Processor

from deepforest.utilities import mask_to_polygon
from deepforest.visualize import plot_results

logger = logging.getLogger(__name__)


def load_sam2_model(model_name: str, device: str):
    """Load SAM2 model and processor from HuggingFace.

    Args:
        model_name: Name of the SAM2 model on HuggingFace
        device: Device to load model on ('cuda', 'mps', or 'cpu')

    Returns:
        Tuple of (model, processor)
    """
    processor = Sam2Processor.from_pretrained(model_name)
    model = Sam2Model.from_pretrained(model_name)
    model = model.to(device)
    return model, processor


def process_image_group(
    image_path: str,
    detections: pd.DataFrame,
    model,
    processor,
    device: str,
    image_root: str = "",
    box_batch_size: int = 32,
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    viz_output_dir: str = None,
) -> list:
    """Process all detections for a single image.

    Args:
        image_path: Path to the image file
        detections: DataFrame of detections for this image
        model: SAM2 model
        processor: SAM2 processor
        device: Device to run inference on
        image_root: Root directory to prepend to image_path if needed
        box_batch_size: Maximum number of boxes to process per forward pass
        mask_threshold: Threshold for binarizing SAM2 mask outputs
        iou_threshold: Minimum IoU score to accept a polygon
        viz_output_dir: Directory to save visualizations (if not None)

    Returns:
        List of WKT polygon strings
    """
    full_path = os.path.join(image_root, image_path) if image_root else image_path
    image = Image.open(full_path).convert("RGB")

    boxes = detections[["xmin", "ymin", "xmax", "ymax"]].values.tolist()

    all_polygons = []
    for i in range(0, len(boxes), box_batch_size):
        box_chunk = boxes[i : i + box_batch_size]
        input_boxes = [box_chunk]

        inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            binarize=False,
            mask_interpolation_mode="nearest",
        )[0]

        iou_scores = outputs.iou_scores.cpu()

        for i, mask_set in enumerate(masks):
            best_idx = iou_scores[0, i].argmax().item()
            best_iou = iou_scores[0, i, best_idx].item()

            if best_iou < iou_threshold:
                all_polygons.append(Polygon().wkt)
                continue

            best_mask = mask_set[best_idx]
            mask_np = (best_mask.numpy() > mask_threshold).astype(np.uint8)
            polygon = mask_to_polygon(mask_np)
            all_polygons.append(polygon.wkt)

    if viz_output_dir is not None:
        viz_df = detections.copy()
        viz_df["polygon_geometry"] = all_polygons
        viz_df["geometry"] = viz_df["polygon_geometry"].apply(
            lambda x: wkt.loads(x) if pd.notna(x) else None
        )
        viz_df = viz_df[
            viz_df["geometry"].apply(lambda x: x is not None and not x.is_empty)
        ]

        if len(viz_df) > 0:
            if "label" not in viz_df.columns:
                viz_df["label"] = "Tree"
            if "score" not in viz_df.columns:
                viz_df["score"] = 1.0

            full_path = os.path.join(image_root, image_path) if image_root else image_path
            with Image.open(full_path) as img:
                width, height = img.size

            image_name = Path(image_path).stem
            viz_path = os.path.join(viz_output_dir, f"{image_name}_polygons.png")
            plot_results(
                results=viz_df,
                image=full_path,
                savedir=os.path.dirname(viz_path),
                basename=os.path.splitext(os.path.basename(viz_path))[0],
                height=height,
                width=width,
                show=False,
            )

    return all_polygons


def convert_boxes_to_polygons(
    input_csv: str,
    output_csv: str,
    model_name: str = "facebook/sam2.1-hiera-small",
    box_batch_size: int = 32,
    image_root: str = "",
    visualize: bool = False,
    viz_output_dir: str = ".",
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    device: str = None,
) -> None:
    """Convert DeepForest bounding boxes to polygons using SAM2.

    Args:
        input_csv: Path to input CSV with DeepForest predictions
        output_csv: Path to save output CSV with polygons
        model_name: HuggingFace model name for SAM2
        box_batch_size: Maximum number of boxes to process per forward pass
        image_root: Root directory to prepend to image paths in CSV
        visualize: Whether to create visualization images
        viz_output_dir: Directory to save visualization images
        mask_threshold: Threshold for binarizing SAM2 mask outputs
        iou_threshold: Minimum IoU score to accept a polygon
        device: Device to use ('cuda', 'mps', or 'cpu'). Auto-detects if None.
    """
    df = pd.read_csv(input_csv)

    required_cols = ["xmin", "ymin", "xmax", "ymax", "image_path"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    logger.info("Using device: %s", device)
    logger.info("Loading SAM2 model: %s", model_name)
    model, processor = load_sam2_model(model_name, device)

    grouped = df.groupby("image_path")
    total_images = len(grouped)

    all_polygons = []

    if visualize:
        os.makedirs(viz_output_dir, exist_ok=True)

    for image_path, group in tqdm(grouped, desc="Processing images", total=total_images):
        polygons = process_image_group(
            image_path,
            group,
            model,
            processor,
            device,
            image_root,
            box_batch_size,
            mask_threshold,
            iou_threshold,
            viz_output_dir=viz_output_dir if visualize else None,
        )
        all_polygons.extend(polygons)

    df["polygon_geometry"] = all_polygons

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved results to %s", output_csv)


def main():
    """CLI entrypoint for polygon conversion."""
    parser = argparse.ArgumentParser(
        description="Convert DeepForest bounding boxes to polygons using SAM2"
    )
    parser.add_argument("input", help="Path to input CSV with DeepForest predictions")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output CSV (default: input with '_polygons' suffix)",
    )
    parser.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-small",
        help="SAM2 model name from HuggingFace (default: facebook/sam2.1-hiera-small)",
    )
    parser.add_argument(
        "--box-batch",
        type=int,
        default=32,
        help="Maximum number of boxes to process per forward pass (default: 32)",
    )
    parser.add_argument(
        "--image-root",
        default="",
        help="Root directory to prepend to image paths in CSV",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization images with polygons overlaid",
    )
    parser.add_argument(
        "--viz-output-dir",
        default=".",
        help="Directory to save visualization images (default: current directory)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing SAM2 mask outputs (default: 0.5)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum IoU score to accept a polygon (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use for inference (default: auto-detect mps > cuda > cpu)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_polygons{input_path.suffix}"
        args.output = str(output_path)

    convert_boxes_to_polygons(
        args.input,
        args.output,
        model_name=args.model,
        box_batch_size=args.box_batch,
        image_root=args.image_root,
        visualize=args.visualize,
        viz_output_dir=args.viz_output_dir,
        mask_threshold=args.mask_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
