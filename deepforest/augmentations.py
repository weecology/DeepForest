import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(augmentations=[], size=512):
    augmentation_list = [ToTensorV2()]
    for augmentation in augmentations:
        if augmentation.lower() == "horizontalflip":
            augmentation_list.append(A.HorizontalFlip(p=0.5))

        elif augmentation.lower() == "verticalflip":
            augmentation_list.append(A.VerticalFlip(p=0.5))

        elif augmentation.lower() == 'randomrotate90':
            augmentation_list.append(A.RandomRotate90())

        elif augmentation.lower() == 'randomcrop':
            augmentation_list.append(A.RandomCrop())

        elif augmentation.lower() == 'randombrightnesscontrast':
            augmentation_list.append(A.RandomBrightnessContrast())

        elif augmentation.lower() == 'normalize':
            augmentation_list.append(A.Normalize())

        else:
            raise NotImplementedError(f"Augmentation {augmentation} is not implemented.")

    transform = A.Compose(augmentation_list,
                          bbox_params=A.BboxParams(format='pascal_voc',
                                                   label_fields=["category_ids"]))

    return transform
