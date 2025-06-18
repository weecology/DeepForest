"""
Active Learning module for DeepForest 

[WIP] This module provides simple active learning capabilities for selecting
the most informative samples for annotation.
"""

import random
import logging
from typing import List, Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearning:
    """
    Active learning class for DeepForest with multiple sampling strategies.
    
    Handles sample selection and human review workflows.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize Active Learning.
        
        Args:
            confidence_threshold: Threshold for confident vs uncertain predictions
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"ActiveLearning initialized with threshold: {confidence_threshold}")

    def _random_sampling(self, predictions: pd.DataFrame, n_samples: int) -> List[str]:
        """Randomly select image paths."""
        available_images = predictions["image_path"].unique().tolist()
        n_to_select = min(n_samples, len(available_images))
        return random.sample(available_images, n_to_select)

    def _uncertainty_sampling(self, predictions: pd.DataFrame,
                              n_samples: int) -> List[str]:
        """Select images with most uncertain predictions."""
        # Calculate uncertainty as (1 - confidence score)
        predictions = predictions.copy()
        predictions['uncertainty'] = 1 - predictions['score']

        # Group by image and get mean uncertainty per image
        image_uncertainty = (
            predictions.groupby('image_path')['uncertainty'].mean().sort_values(
                ascending=False))

        return image_uncertainty.head(n_samples).index.tolist()

    def _high_density_sampling(self, predictions: pd.DataFrame,
                               n_samples: int) -> List[str]:
        """Select images with highest number of detections."""
        # Count detections per image
        detection_counts = (predictions.groupby('image_path').size().sort_values(
            ascending=False))

        return detection_counts.head(n_samples).index.tolist()

    def human_review_split(
            self, predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split predictions into confident and uncertain for human review.
        
        Args:
            predictions: DataFrame with model predictions containing 'score' column
            
        Returns:
            Tuple of (confident_predictions, uncertain_predictions)
        """
        if predictions.empty:
            logger.warning("Empty predictions provided")
            return pd.DataFrame(), pd.DataFrame()

        # Split based on confidence threshold
        confident = predictions[predictions['score'] >= self.confidence_threshold]
        uncertain = predictions[predictions['score'] < self.confidence_threshold]

        logger.info(
            f"Split: {len(confident)} confident, {len(uncertain)} uncertain predictions")

        return confident, uncertain

    def select_samples(self,
                       predictions: pd.DataFrame,
                       strategy: str = 'uncertainty',
                       n_samples: int = 10) -> List[str]:
        """
        Select samples for annotation using specified strategy.
        
        Args:
            predictions: DataFrame with predictions
            strategy: Sampling strategy ('random', 'uncertainty', 'high_density')
            n_samples: Number of samples to select
            
        Returns:
            List of selected image paths
        """
        if predictions.empty:
            logger.warning("No predictions provided for sample selection")
            return []

        # Map strategy names to methods
        strategy_methods = {
            'random': self._random_sampling,
            'uncertainty': self._uncertainty_sampling,
            'high_density': self._high_density_sampling
        }

        if strategy not in strategy_methods:
            available = list(strategy_methods.keys())
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")

        selected = strategy_methods[strategy](predictions, n_samples)

        logger.info(f"Selected {len(selected)} samples using '{strategy}' strategy")

        return selected

    def run_iteration(self,
                      predictions: pd.DataFrame,
                      strategy: str = 'uncertainty',
                      n_samples: int = 10) -> dict:
        """
        Run a complete active learning iteration.
        
        Args:
            predictions: DataFrame with model predictions
            strategy: Sampling strategy to use
            n_samples: Number of samples to select
            
        Returns:
            Dictionary with iteration results
        """
        logger.info(
            f"Running active learning iteration with {len(predictions)} predictions")

        # Split predictions
        confident, uncertain = self.human_review_split(predictions)

        # Select samples from uncertain predictions (or all if no uncertain ones)
        target_predictions = uncertain if not uncertain.empty else predictions
        selected_images = self.select_samples(target_predictions, strategy, n_samples)

        results = {
            'total_predictions': len(predictions),
            'confident_predictions': len(confident),
            'uncertain_predictions': len(uncertain),
            'selected_images': selected_images,
            'strategy_used': strategy
        }

        logger.info(f"Iteration complete: selected {len(selected_images)} images")

        return results
