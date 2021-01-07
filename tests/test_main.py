#test main
import os
import pytest
import pandas as pd
import numpy as np

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

def test_main():
    from deepforest import main

def test_train(m):
    m.train(debug=True)

def test_predict_image(m):
    image = np.random.random((3,400,400))
    prediction = m.predict_image(image)
    assert isinstance(prediction, pd.DataFrame)
    
    prediction.columns == ["image_path","xmin","ymin","xmax","ymax","label"]
    
def test_predict_file():
    pass

def test_predict_tile():
    pass
