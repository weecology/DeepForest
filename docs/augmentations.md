# Augmentations 

## Overview

DeepForest allows users to customize data augmentations by defining them in a configuration file (`deepforest_config.yml`). This configuration is then passed to the model using the `get_augmentations` function. This guide will walk you through the steps to add custom augmentations and apply them in your DeepForest model.

## Configuration File (`deepforest_config.yml`)

The `deepforest_config.yml` file is where you define the data augmentations you want to apply during training or validation. The augmentations are specified under the `train` and `validation` sections.

### Example Configuration

Below is an example configuration file with custom augmentations:

```yaml
# Dataloaders
workers: 1
devices: auto
accelerator: auto
batch_size: 1

# Model Architecture
architecture: 'retinanet'
num_classes: 1
nms_thresh: 0.05

# Architecture specific params
retinanet:
    score_thresh: 0.1

train:
    csv_file: path/to/your/train/annotations.csv
    root_dir: path/to/your/train/images/
    augmentations:
        - type: RandomSizedBBoxSafeCrop
          params:
            height: 300
            width: 300
            erosion_rate: 0.0
            interpolation: 1
            always_apply: False
        - type: PadIfNeeded
          params:
            always_apply: False
            p: 1.0
            min_height: 1024
            min_width: 1024
            position: "center"
            border_mode: 0
            value: 0
        - type: ToTensorV2
        - type: BBox
          params:
            format: pascal_voc
            label_fields: [category_ids]

    default_augmentations:
        - type: HorizontalFlip
          params: 
            p: 0.5
        - type: VerticalFlip
          params:
            p: 0.5
        - type: RandomBrightnessContrast
          params:
            brightness_limit: 0.2
            contrast_limit: 0.2
            p: 0.5
        - type: RandomRotate90
          params:
            p: 0.5
        - type: Normalize
          params:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            max_pixel_value: 255.0
            p: 1.0
        - type: ToTensorV2
        - type: BBox
          params:
            format: pascal_voc
            label_fields: [category_ids]
    
validation:
    csv_file: path/to/your/val/annotations.csv
    root_dir: path/to/your/val/images/
    augmentations:
        - type: Normalize
          params:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            max_pixel_value: 255.0
            p: 1.0
        - type: ToTensorV2
        - type: BBox
          params:
            format: pascal_voc
            label_fields: [category_ids]
```
### Default Augmentations
If the `augmentations` key under `train` or `validation` is empty, DeepForest will automatically apply the `default_augmentations` specified in the configuration file. This ensures that some level of augmentation is always applied, even if no custom augmentations are defined.

Example with empty augmentations:
```yaml
train:
    csv_file: path/to/your/train/annotations.csv
    root_dir: path/to/your/train/images/
    augmentations:   # No custom augmentations
    default_augmentations:
        - type: HorizontalFlip
          params: 
            p: 0.5
        - type: VerticalFlip
          params:
            p: 0.5
        - type: RandomBrightnessContrast
          params:
            brightness_limit: 0.2
            contrast_limit: 0.2
            p: 0.5
        - type: RandomRotate90
          params:
            p: 0.5
        - type: Normalize
          params:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            max_pixel_value: 255.0
            p: 1.0
        - type: ToTensorV2

```

## Applying the Augmentations

After defining the augmentations in your configuration file, you can apply them to your DeepForest model by using the `get_augmentations` function.

### Example Usage

```python
from deepforest.augmentations import get_augmentations
from deepforest import main

# Initialize the model with custom augmentations
m = main.deepforest(transforms=get_augmentations)
```

### How to Add Custom Augmentations

If you want to add a new augmentation, follow these steps:

1. **Edit the `deepforest_config.yml` file**:
   - Add your desired augmentation under the `augmentations` section in either `train` or `validation`.
   - Specify the `type` of augmentation and the corresponding `params`.

2. **Pass the `get_augmentations` function to your model**:
   - As shown in the example above, initialize your model with the augmentations by passing the `get_augmentations` function.

### Supported Augmentations

Some of the supported augmentations in DeepForest include:

- `HorizontalFlip`: Flips the image horizontally with a given probability.
- `VerticalFlip`: Flips the image vertically with a given probability.
- `RandomRotate90`: Rotates the image by 90 degrees randomly.
- `RandomCrop`: Randomly crops a part of the image.
- `RandomSizedBBoxSafeCrop`: Crops a part of the image ensuring bounding boxes remain within the cropped area.
- `PadIfNeeded`: Pads the image to a minimum height and width.
- `RandomBrightnessContrast`: Randomly adjusts the brightness and contrast.
- `Normalize`: Normalizes the image with the given mean and standard deviation.
- `ToTensorV2`: Converts the image and bounding boxes to PyTorch tensors.

For a complete list of supported augmentations, refer to the [Albumentations documentation](https://albumentations.ai/docs/).

## Troubleshooting

- **Error: Missing key in the configuration file**: Ensure that all required keys and parameters are correctly defined in the `deepforest_config.yml` file.
- **Invalid Augmentation Type**: Double-check the spelling and case of the augmentation type in your configuration file.

## Conclusion

Customizing data augmentations in DeepForest is simple and flexible. By defining your desired augmentations in the `deepforest_config.yml` file and passing them to the model using `get_augmentations`, you can easily tailor the training and validation processes to suit your specific needs.
