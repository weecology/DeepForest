#Test evaluate
#Test IoU
from .conftest import download_release
from deepforest import evaluate
from deepforest import main
from deepforest import get_data
import os
import pytest
import pandas as pd
import numpy as np
import geopandas as gpd

def test_evaluate_image(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    ground_truth = pd.read_csv(csv_file)
    predictions.label = 0
    result = evaluate.evaluate_image(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file))     
        
    assert result.shape[0] == ground_truth.shape[0]
    assert sum(result.IoU) > 10 

def test_evaluate(m):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    predictions.label = "Tree"
    ground_truth = pd.read_csv(csv_file)
    predictions = predictions.loc[range(10)]
    results = evaluate.evaluate(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file))     
        
    assert results["results"].shape[0] == ground_truth.shape[0]
    assert results["box_recall"] > 0.1
    assert results["class_recall"].shape == (1,4)
    assert results["class_recall"].recall.values == 1
    assert results["class_recall"].precision.values == 1
    assert "score" in results["results"].columns
    assert results["results"].true_label.unique() == "Tree"

def test_evaluate_multi():
    csv_file = get_data("testfile_multi.csv")
    ground_truth = pd.read_csv(csv_file)
    ground_truth["label"] = ground_truth.label.astype("category").cat.codes
    
    #Manipulate the data to create some false positives
    predictions = ground_truth.copy()
    predictions["score"] = 1
    predictions.iloc[[36, 35, 34], predictions.columns.get_indexer(['label'])]
    results = evaluate.evaluate(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file))     
        
    assert results["results"].shape[0] == ground_truth.shape[0]
    assert results["class_recall"].shape == (2,4)
    
def test_evaluate_save_images(tmpdir):
    csv_file = get_data("testfile_multi.csv")
    ground_truth = pd.read_csv(csv_file)
    ground_truth["label"] = ground_truth.label.astype("category").cat.codes
    
    #Manipulate the data to create some false positives
    predictions = ground_truth.copy()
    predictions["score"] = 1
    predictions.iloc[[36, 35, 34], predictions.columns.get_indexer(['label'])]
    results = evaluate.evaluate(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file), savedir=tmpdir)     
    assert all([os.path.exists("{}/{}".format(tmpdir,x)) for x in ground_truth.image_path])

def test_evaluate_empty(m):
    m = main.deepforest()
    m.config["score_thresh"] = 0.8
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    results = m.evaluate(csv_file, root_dir, iou_threshold = 0.4)
    
    #Does this make reasonable predictions, we know the model works.
    assert np.isnan(results["box_precision"])
    assert results["box_recall"] == 0
    
    df = pd.read_csv(csv_file)
    assert results["results"].shape[0] == df.shape[0]

@pytest.fixture
def sample_results():
    # Create a sample DataFrame for testing
    data = {
        'true_label': [1, 1, 2],
        'predicted_label': [1, 2, 1]
    }
    return pd.DataFrame(data)

def test_compute_class_recall(sample_results):
    # Test case with sample data
    expected_recall = pd.DataFrame({
        'label': [1, 2],
        'recall': [0.5, 0],
        'precision': [0.5, 0],
        'size': [2, 1]
    }).reset_index(drop=True)

    assert evaluate.compute_class_recall(sample_results).equals(expected_recall)

@pytest.mark.parametrize("root_dir",[None,"tmpdir"])
def test_point_recall_image(root_dir, tmpdir):
    img_path = get_data("OSBS_029.png")
    if root_dir == "tmpdir":
        root_dir = os.path.dirname(img_path)
        savedir = tmpdir
    else:
        savedir = None

    # create sample dataframes
    predictions = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "xmin": [1, 150],
        "xmax": [100, 200],
        "ymin": [1, 75],
        "ymax": [50, 100],
        "label": ["A", "B"],
    })
    ground_df = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "x": [5, 20],
        "y": [30, 300],
        "label": ["A", "B"],
    })

    # run the function
    result = evaluate._point_recall_image_(predictions, ground_df, root_dir=root_dir, savedir=savedir)

    # check the output, 1 match of 2 ground truth
    assert all(result.predicted_label.isnull().values == [False,True])
    assert isinstance(result, gpd.GeoDataFrame)
    assert "predicted_label" in result.columns
    assert "true_label" in result.columns
    assert "geometry" in result.columns

def test_point_recall():
    # create sample dataframes
    predictions = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "xmin": [1, 150],
        "xmax": [100, 200],
        "ymin": [1, 75],
        "ymax": [50, 100],
        "label": ["A", "B"],
    })
    ground_df = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.png"],
        "x": [5, 20],
        "y": [30, 300],
        "label": ["A", "B"],
    })

    results = evaluate.point_recall(ground_df=ground_df, predictions=predictions)
    assert results["box_recall"] == 0.5
    assert results["class_recall"].recall[0] == 1