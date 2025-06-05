# Standard library imports
import os

# Third party imports
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from torch.utils.data import Dataset
from torchvision import transforms


def bounding_box_transform(augment=False):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(resnet_normalize)
    data_transforms.append(transforms.Resize([224, 224]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)


resnet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


class BoundingBoxDataset(Dataset):
    """An in memory dataset for bounding box predictions.

    Args:
        df: a pandas dataframe with image_path and xmin,xmax,ymin,ymax columns
        transform: a function to apply to the image
        root_dir: the directory where the image is stored

    Returns:
        rgb: a tensor of shape (3, height, width)
    """

    def __init__(self, df, root_dir, transform=None, augment=False):
        self.df = df

        if transform is None:
            self.transform = bounding_box_transform(augment=augment)
        else:
            self.transform = transform

        unique_image = self.df['image_path'].unique()
        assert len(unique_image
                  ) == 1, "There should be only one unique image for this class object"

        # Open the image using rasterio
        self.src = rio.open(os.path.join(root_dir, unique_image[0]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']

        # Read the RGB data
        box = self.src.read(window=Window(xmin, ymin, xmax - xmin, ymax - ymin))
        box = np.rollaxis(box, 0, 3)

        if self.transform:
            image = self.transform(box)
        else:
            image = box

        return image
