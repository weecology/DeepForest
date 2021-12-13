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

@pytest.fixture()
def m(download_release):
    m = main.deepforest()
    m.use_release(check_release=False)
    
    return m

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

def test_evaluate_multi(m):
    csv_file = get_data("testfile_multi.csv")
    m = main.deepforest(num_classes=2,label_dict={"Alive":0,"Dead":1})
    ground_truth = pd.read_csv(csv_file)
    ground_truth["label"] = ground_truth.label.astype("category").cat.codes
    
    #Manipulate the data to create some false positives
    predictions = ground_truth.copy()
    predictions["score"] = 1
    predictions.label.loc[[36,35,34]] = 0
    results = evaluate.evaluate(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file))     
        
    assert results["results"].shape[0] == ground_truth.shape[0]
    assert results["class_recall"].shape == (2,4)
    
def test_evaluate_save_images(m, tmpdir):
    csv_file = get_data("testfile_multi.csv")
    m = main.deepforest(num_classes=2,label_dict={"Alive":0,"Dead":1})
    ground_truth = pd.read_csv(csv_file)
    ground_truth["label"] = ground_truth.label.astype("category").cat.codes
    
    #Manipulate the data to create some false positives
    predictions = ground_truth.copy()
    predictions["score"] = 1
    predictions.label.loc[[36,35,34]] = 0
    results = evaluate.evaluate(predictions=predictions, ground_df=ground_truth, root_dir=os.path.dirname(csv_file), savedir=tmpdir)     
    assert all([os.path.exists("{}/{}".format(tmpdir,x)) for x in ground_truth.image_path])

def test_evaluate_empty():
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