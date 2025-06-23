import logging
import os
import glob
import random
from typing import List, Optional, Tuple

import pandas as pd
import geopandas as gpd
from omegaconf import OmegaConf

from deepforest.label_studio import get_api_key
from label_studio_sdk import Client

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Detection:
    """
    Stub class for detection functionality. 
    Created only to satisfy syntax requirements. I will write this part later.
    """
    @staticmethod
    def load(model_name: str):
        """
        Load and return a detection model by name.
        """
        # TODO: implement actual model loading
        logger.info(f"Loading detection model '{model_name}'")
        return None

    @staticmethod
    def predict(
        m,
        model_path: Optional[str],
        image_paths: List[str],
        patch_size: int,
        patch_overlap: float,
        batch_size: int,
    ) -> List[pd.DataFrame]:
        """
        Run detection on the list of image paths and return a list of
        pandas DataFrames with prediction results. Each DataFrame should
        include at least columns ['image_path', 'geometry', 'score', ...].
        """
        # TODO: implement actual prediction logic
        logger.info(
            f"Running detection on {len(image_paths)} images with patch_size={patch_size}, "
            f"patch_overlap={patch_overlap}, batch_size={batch_size}"
        )
        return []

class Pipeline:
    """
    Minimal in-script pipeline to push selected images to a Label Studio project.
    """
    # TODO: Implement a pipeline that integrates with Label Studio

class ActiveLearning:
    """
    Active learning pipeline for DeepForest: predictions, human review,
    sampling and annotation.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_detection_score: float = 0.6,
        min_score: float = 0.1,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_detection_score = min_detection_score
        self.min_score = min_score
        logger.info(
            f"Initialized ActiveLearning with "
            f"confidence_threshold={confidence_threshold}, "
            f"min_detection_score={min_detection_score}, "
            f"min_score={min_score}"
        )

    def generate_predictions(
        self,
        images: List[str],
        patch_size: int = 512,
        patch_overlap: float = 0.1,
        batch_size: int = 16,
        pool_limit: int = 1000,
        model_path: Optional[str] = None,
    ) -> Optional[gpd.GeoDataFrame]:
        pool = images.copy()
        if len(pool) > pool_limit:
            pool = random.sample(pool, pool_limit)

        model = None
        if not model_path:
            logger.info("Loading default 'tree' model")
            model = Detection.load("tree")

        preds = Detection.predict(
            m=model,
            model_path=model_path,
            image_paths=pool,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
        )
        if not preds:
            return None

        df = pd.concat(preds, ignore_index=True)
        if df.empty:
            return None

        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        return gdf[gdf["score"] >= self.min_score]

    def human_review_split(
        self,
        predictions: gpd.GeoDataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if predictions.empty:
            logger.warning("Empty predictions for human review split")
            return pd.DataFrame(), pd.DataFrame()

        if "cropmodel_score" in predictions.columns:
            mask = (
                (predictions["score"] >= self.min_detection_score)
                & (predictions["cropmodel_score"] < self.confidence_threshold)
            )
            filtered = predictions[mask]
            uncertain = filtered[filtered["cropmodel_score"] < self.confidence_threshold]
            confident = filtered.drop(uncertain.index)
        else:
            confident = predictions[predictions["score"] >= self.confidence_threshold]
            uncertain = predictions.drop(confident.index)

        logger.info(
            f"Human review split -> confident: {len(confident)}, uncertain: {len(uncertain)}"
        )
        return confident, uncertain

    def select_samples(
        self,
        predictions: pd.DataFrame,
        strategy: str = "uncertainty",
        n_samples: int = 10,
        target_labels: Optional[List[str]] = None,
    ) -> Tuple[List[str], pd.DataFrame]:
        if predictions.empty:
            logger.warning("No predictions for sample selection")
            return [], pd.DataFrame()

        df = predictions.copy()

        if strategy == "random":
            candidates = df["image_path"].unique().tolist()
            chosen = random.sample(candidates, min(n_samples, len(candidates)))
        else:
            df = df[df["score"] >= self.min_score]
            if strategy == "most-detections":
                counts = df.groupby("image_path").size().nlargest(n_samples)
                chosen = counts.index.tolist()
            elif strategy == "uncertainty":
                df["uncertainty"] = 1 - df["score"]
                mean_unc = (
                    df.groupby("image_path")["uncertainty"]
                    .mean()
                    .nlargest(n_samples)
                )
                chosen = mean_unc.index.tolist()
            elif strategy == "rarest":
                if "cropmodel_label" not in df.columns:
                    raise ValueError("'rarest' requires 'cropmodel_label' column")
                counts = df["cropmodel_label"].value_counts()
                df["label_count"] = df["cropmodel_label"].map(counts)
                sorted_df = df.sort_values("label_count")
                chosen = (
                    sorted_df.drop_duplicates("image_path")
                    .head(n_samples)["image_path"]
                    .tolist()
                )
            elif strategy == "target-labels":
                if not target_labels:
                    raise ValueError("'target-labels' requires target_labels list")
                filtered = df[df["cropmodel_label"].isin(target_labels)]
                avg_score = (
                    filtered.groupby("image_path")["score"]
                    .mean()
                    .nlargest(n_samples)
                )
                chosen = avg_score.index.tolist()
            else:
                raise ValueError(f"Unknown strategy '{strategy}'")

        chosen_df = predictions[predictions["image_path"].isin(chosen)]
        logger.info(f"Selected {len(chosen)} images with strategy '{strategy}'")
        return chosen, chosen_df

    def run(
        self,
        image_folder: str,
        ls_project_id: str,
        annotations_csv: Optional[str] = None,
        strategy: str = "uncertainty",
        n: int = 10,
        **kwargs,
    ):
        if annotations_csv:
            logger.info(f"Loading preannotations from {annotations_csv}")
            predictions = pd.read_csv(annotations_csv)
        else:
            exts = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
            pool: List[str] = []
            for ext in exts:
                pool.extend(glob.glob(os.path.join(image_folder, ext)))
            if not pool:
                logger.error(f"No images found in {image_folder}")
                return
            logger.info(f"Found {len(pool)} images; generating predictions...")
            predictions = self.generate_predictions(pool, **kwargs)

        if predictions is None or predictions.empty:
            logger.info("No valid predictions; exiting.")
            return

        confident, uncertain = self.human_review_split(predictions)
        target_df = uncertain if not uncertain.empty else predictions
        chosen, chosen_df = self.select_samples(
            target_df, strategy=strategy, n_samples=n
        )

        out_images = kwargs.get("output_images", "selected_images.txt")
        out_csv = kwargs.get("output_csv", "selected_preannotations.csv")
        with open(out_images, "w") as f:
            for img in chosen:
                f.write(f"{img}\n")
        chosen_df.to_csv(out_csv, index=False)
        logger.info(
            f"Saved {len(chosen)} images to {out_images} and details to {out_csv}"
        )

        api_key = get_api_key()
        if api_key:
            os.environ["LABEL_STUDIO_API_KEY"] = api_key
            cfg = OmegaConf.create({
                "label_studio": {"project_id": ls_project_id},
                "pipeline": {"images_to_annotate": chosen, "gpus": 1},
            })
            Pipeline(cfg=cfg).run()
        else:
            logger.warning("No Label Studio API key; skipping push.")

def run_active_learning(
    image_folder: str,
    ls_project_id: str,
    annotations_csv: Optional[str] = None,
    strategy: str = "uncertainty",
    n: int = 10,
    confidence_threshold: float = 0.5,
    min_detection_score: float = 0.6,
    min_score: float = 0.1,
    **kwargs,
):
    al = ActiveLearning(
        confidence_threshold=confidence_threshold,
        min_detection_score=min_detection_score,
        min_score=min_score,
    )
    al.run(
        image_folder=image_folder,
        ls_project_id=ls_project_id,
        annotations_csv=annotations_csv,
        strategy=strategy,
        n=n,
        **kwargs,
    )
