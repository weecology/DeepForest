#test main
import os
import pytest
import pandas as pd
import numpy as np

from skimage import io

from deepforest import main
from deepforest import get_data

@pytest.fixture()
def m():
    csv_file = get_data("example.csv")
    m = main.deepforest()
    m.config["epochs"] = 1
    m.config["batch_size"] = 2
    m.create_model()
    m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), train=True)
    
    return m

@pytest.fixture()
def trained_model():
    csv_file = get_data("example.csv")
    m = main.deepforest()
    m.config["epochs"] = 1
    m.config["batch_size"] = 2
    m.create_model()
    m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), train=True)
    m.train(debug=True)        
    
    return m

def test_main():
    from deepforest import main

def test_train(m):
    m.train(debug=True)

def test_predict_image_empty(m):
    image = np.random.random((400,400,3))
    prediction = m.predict_image(image = image)
    assert prediction is None
    
def test_predict_image_fromfile(trained_model):
    path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    prediction = trained_model.predict_image(path = path)
    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","scores"}

def test_predict_image_fromarray(trained_model):
    image = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    image = io.imread(image)
    prediction = trained_model.predict_image(image = image)
    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","scores"}

def test_predict_file(trained_model):
    pass

def test_predict_tile():
    pass
