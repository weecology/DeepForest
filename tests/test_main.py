#test main
import os
import pytest

from deepforest import main
from deepforest import get_data

@pytest.fixture()
def model():
    csv_file = get_data("example.csv")
    m = main.deepforest()
    m.config["epochs"] = 1
    m.config["batch_size"] = 2
    m.create_model()
    m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), train=True)
    
    return m

def test_main():
    from deepforest import main

def test_train(model):
    model.train(debug=True)
