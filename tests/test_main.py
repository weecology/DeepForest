#test main
import os

from deepforest import main
from deepforest import get_data

def test_main():
    from deepforest import main

def test_train():
    csv_file = get_data("example.csv")
    m = main.deepforest()
    m.create_model()
    m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    m.train()
