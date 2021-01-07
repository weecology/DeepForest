#Transforms from https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
import random
import torch
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image).float()
        target["boxes"] = torch.from_numpy(target["boxes"])
        target["labels"] = torch.from_numpy(target["labels"])

        return image, target
