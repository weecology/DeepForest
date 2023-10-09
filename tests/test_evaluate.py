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