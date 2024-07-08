#Test visualize
from deepforest import visualize
from deepforest import main
from deepforest import get_data
import os
import pytest
import numpy as np
import pandas as pd

def test_format_boxes(m):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        assert list(target_df.columns.values) == ["xmin","ymin","xmax","ymax","label"]
        assert not target_df.empty
        

#Test different color labels
@pytest.mark.parametrize("label",[0,1,20])
def test_plot_predictions(m, tmpdir,label):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        image = np.array(image)[:,:,::-1]
        image = np.rollaxis(image,0,3)
        target_df.label = label
        image = visualize.plot_predictions(image, target_df)

        assert image.dtype == "uint8"
        
def test_plot_prediction_dataframe(m, tmpdir):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        filenames = visualize.plot_prediction_dataframe(df=target_df,savedir=tmpdir, root_dir=m.config["validation"]["root_dir"])
        
    assert all([os.path.exists(x) for x in filenames])
        
def test_plot_predictions_and_targets(m, tmpdir):
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    m.model.eval()    
    predictions = m.model(images)    
    for path, image, target, prediction in zip(paths, images, targets, predictions):
        image = image.permute(1,2,0)
        save_figure_path = visualize.plot_prediction_and_targets(image, prediction, target, image_name=os.path.basename(path), savedir=tmpdir)
        assert os.path.exists(save_figure_path)

def test_convert_to_sv_format():
    # Create a mock DataFrame
    data = {
        'xmin': [0, 10],
        'ymin': [0, 20],
        'xmax': [5, 15],
        'ymax': [5, 25],
        'label': ['tree', 'tree'],
        'score': [0.9, 0.8],
        'image_path': ['image1.jpg', 'image1.jpg']
    }
    df = pd.DataFrame(data)
    
    # Call the function
    detections = visualize.convert_to_sv_format(df)
    
    # Expected values
    expected_boxes = np.array([[0, 0, 5, 5], [10, 20, 15, 25]], dtype=np.float32)
    expected_labels = np.array([0, 0])
    expected_scores = np.array([0.9, 0.8])

    
    # Assertions
    np.testing.assert_array_equal(detections.xyxy, expected_boxes)
    np.testing.assert_array_equal(detections.class_id, expected_labels)
    np.testing.assert_array_equal(detections.confidence, expected_scores)
    assert detections['class_name'] == ['tree', 'tree']
