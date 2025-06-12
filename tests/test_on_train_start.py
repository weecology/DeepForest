"""
Tests for on_train_start hook that logs sample train/val images.
"""

import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pytorch_lightning.loggers import Logger

from deepforest import main, get_data


class MockLogger(Logger):
    """Mock logger to capture logged images."""
    
    def __init__(self):
        super().__init__()
        self.logged_images = []
        self.experiment = Mock()
        self.experiment.log_image = self._log_image
        
    def _log_image(self, path, metadata=None):
        """Capture logged images."""
        self.logged_images.append({
            'path': path,
            'metadata': metadata
        })
        
    @property
    def name(self):
        return "MockLogger"
    
    @property
    def version(self):
        return "0.0.0"
    
    def log_metrics(self, metrics, step):
        pass
    
    def log_hyperparams(self, params):
        pass


@pytest.fixture
def m_with_logger():
    """Create a model with mock logger."""
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.batch_size = 2
    m.config.workers = 0
    
    # Create trainer with mock logger
    logger = MockLogger()
    m.create_trainer(logger=logger, fast_dev_run=True)
    
    return m, logger


def test_on_train_start_logs_images(m_with_logger):
    """Test that on_train_start logs sample images from training dataset."""
    m, logger = m_with_logger
    
    # Fit the model to trigger on_train_start
    m.trainer.fit(m)
    
    # Check that images were logged
    assert len(logger.logged_images) > 0
    
    # Check that training images were logged
    train_images = [img for img in logger.logged_images 
                   if img['metadata'].get('context') == 'detection_train']
    assert len(train_images) > 0
    assert len(train_images) <= 5  # Should log at most 5 images
    
    # Check metadata
    for img in train_images:
        assert 'name' in img['metadata']
        assert 'context' in img['metadata']
        assert 'step' in img['metadata']
        assert img['metadata']['context'] == 'detection_train'


def test_on_train_start_logs_validation_images(m_with_logger):
    """Test that on_train_start logs sample images from validation dataset."""
    m, logger = m_with_logger
    
    # Fit the model to trigger on_train_start
    m.trainer.fit(m)
    
    # Check that validation images were logged
    val_images = [img for img in logger.logged_images 
                  if img['metadata'].get('context') == 'detection_val']
    assert len(val_images) > 0
    assert len(val_images) <= 5  # Should log at most 5 images
    
    # Check metadata
    for img in val_images:
        assert img['metadata']['context'] == 'detection_val'


def test_on_train_start_with_multiple_loggers():
    """Test that on_train_start works with multiple loggers."""
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.batch_size = 2
    m.config.workers = 0
    
    # Create multiple mock loggers
    logger1 = MockLogger()
    logger2 = MockLogger()
    
    m.create_trainer(logger=[logger1, logger2], fast_dev_run=True)
    m.trainer.fit(m)
    
    # Both loggers should have logged images
    assert len(logger1.logged_images) > 0
    assert len(logger2.logged_images) > 0
    
    # Same images should be logged to both
    assert len(logger1.logged_images) == len(logger2.logged_images)


def test_on_train_start_with_empty_annotations():
    """Test that on_train_start handles empty annotations gracefully."""
    m = main.deepforest()
    
    # Create empty CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_csv = pd.DataFrame({
            "image_path": [],
            "xmin": [],
            "xmax": [],
            "ymin": [],
            "ymax": [],
            "label": []
        })
        empty_csv_path = os.path.join(tmpdir, "empty.csv")
        empty_csv.to_csv(empty_csv_path, index=False)
        
        m.config.train.csv_file = empty_csv_path
        m.config.train.root_dir = tmpdir
        m.config.validation.csv_file = empty_csv_path
        m.config.validation.root_dir = tmpdir
        m.config.batch_size = 1
        m.config.workers = 0
        
        logger = MockLogger()
        m.create_trainer(logger=logger, fast_dev_run=True)
        
        # Should not crash with empty annotations
        m.trainer.fit(m)
        
        # No images should be logged
        assert len(logger.logged_images) == 0


def test_on_train_start_samples_correct_number():
    """Test that on_train_start samples the correct number of images."""
    m = main.deepforest()
    
    # Use a dataset with more than 5 images
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.batch_size = 2
    m.config.workers = 0
    
    logger = MockLogger()
    m.create_trainer(logger=logger, fast_dev_run=True)
    
    # Load the CSV to check how many unique images there are
    df = pd.read_csv(m.config.train.csv_file)
    n_unique_images = len(df.image_path.unique())
    
    m.trainer.fit(m)
    
    # Should log min(5, n_unique_images) for both train and val
    train_images = [img for img in logger.logged_images 
                   if img['metadata'].get('context') == 'detection_train']
    val_images = [img for img in logger.logged_images 
                  if img['metadata'].get('context') == 'detection_val']
    
    expected_count = min(5, n_unique_images)
    assert len(train_images) == expected_count
    assert len(val_images) == expected_count


def test_on_train_start_without_logger():
    """Test that on_train_start works without any loggers."""
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.validation.csv_file = get_data("example.csv")
    m.config.validation.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.batch_size = 2
    m.config.workers = 0
    
    # Create trainer without logger
    m.create_trainer(logger=False, fast_dev_run=True)
    
    # Should not crash
    m.trainer.fit(m)


@patch('deepforest.visualize.plot_annotations')
def test_on_train_start_calls_visualize(mock_plot_annotations, m_with_logger):
    """Test that on_train_start calls visualize.plot_annotations."""
    m, logger = m_with_logger
    
    # Configure mock to avoid actual plotting
    mock_plot_annotations.return_value = None
    
    m.trainer.fit(m)
    
    # Should have called plot_annotations
    assert mock_plot_annotations.called
    
    # Check that it was called with correct arguments
    calls = mock_plot_annotations.call_args_list
    assert len(calls) > 0
    
    for call in calls:
        args, kwargs = call
        # First argument should be a DataFrame with annotations
        assert isinstance(args[0], pd.DataFrame)
        # Should have savedir in kwargs
        assert 'savedir' in kwargs


def test_on_train_start_with_no_validation():
    """Test on_train_start when no validation dataset is provided."""
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.validation.csv_file = None
    m.config.validation.root_dir = None
    m.config.batch_size = 2
    m.config.workers = 0
    
    logger = MockLogger()
    m.create_trainer(logger=logger, fast_dev_run=True)
    m.trainer.fit(m)
    
    # Should only log training images
    train_images = [img for img in logger.logged_images 
                   if img['metadata'].get('context') == 'detection_train']
    val_images = [img for img in logger.logged_images 
                  if img['metadata'].get('context') == 'detection_val']
    
    assert len(train_images) > 0
    assert len(val_images) == 0


def test_on_train_start_preserves_parent_behavior():
    """Test that on_train_start still calls parent class method."""
    m = main.deepforest()
    m.config.train.csv_file = get_data("example.csv")
    m.config.train.root_dir = os.path.dirname(get_data("example.csv"))
    m.config.batch_size = 2
    m.config.workers = 0
    
    # Mock the parent on_train_start
    with patch.object(main.Model, 'on_train_start') as mock_parent:
        m.create_trainer(fast_dev_run=True)
        m.trainer.fit(m)
        
        # Parent on_train_start should have been called
        mock_parent.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
