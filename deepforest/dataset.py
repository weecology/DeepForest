"""
Dataset model

https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection

During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

labels (Int64Tensor[N]): the class label for each ground-truth box

https://colab.research.google.com/github/benihime91/pytorch_retinanet/blob/master/demo.ipynb#scrollTo=0zNGhr6D7xGN

"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from deepforest.utilities import check_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image

def get_transform(augment):
    """Albumentations transformation of bounding boxs"""
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=["category_ids"]))
        
    else:
        transform = A.Compose([ToTensorV2()])
        
    return transform

class TreeDataset(Dataset):

    def __init__(self, csv_file, root_dir, transforms, label_dict = {"Tree": 0}):
        """
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = np.array(Image.open((img_name))).astype('float32')
        image = image / 255
        
        try:
            check_image(image)
        except Exception as e:
            raise Exception("dataloader failed with exception for image: {}",format(img_name))

        # select annotations
        image_annotations = self.annotations[self.annotations.image_path ==
                                             self.image_names[idx]]
        targets = {}
        targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                              "ymax"]].values.astype(float)
        
        # Labels need to be encoded
        targets["labels"] = image_annotations.label.apply(
            lambda x: self.label_dict[x]).values.astype(int)

        if self.transform:
            augmented = self.transform(image=image, bboxes=targets["boxes"], category_ids=targets["labels"])
            image = augmented["image"]
            
            boxes = np.array(augmented["bboxes"])
            boxes = torch.from_numpy(boxes)
            labels = np.array(augmented["category_ids"]) 
            labels = torch.from_numpy(labels)
            targets = {"boxes":boxes,"labels":labels}   
            
            #Check for blank tensors
            all_empty = all([len(x) == 0 for x in boxes])
            if all_empty:
                return None
            
        else:
            targets["boxes"] = torch.from_numpy(targets["boxes"])
            targets["labels"] = torch.from_numpy(targets["labels"])
            
        return self.image_names[idx], image, targets
