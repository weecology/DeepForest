#test dataset model
from deepforest import get_data
from deepforest import dataset
from torch.utils.data import DataLoader
import os


def test_TreeDataset():
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=None)
    generator = DataLoader(ds)

    batch = next(iter(generator))
    image = batch["image"]
    boxes = batch["boxes"]

    #Between 0 and 1
    assert image.max() <= 1
    assert image.min() >= 0
    assert boxes.shape == (1, 79, 4)
