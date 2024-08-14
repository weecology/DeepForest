import albumentations as A
from deepforest import utilities, get_data
import cv2
from albumentations.pytorch import ToTensorV2

def get_augmentations(augment=True):
    try:
        config_path = get_data("deepforest_config.yml")
    except Exception as e:
        raise ValueError("No config file provided and deepforest_config.yml "
                         "not found either in local directory or in installed "
                         "package location. {}".format(e))
    
    augmentations = []
    format_bbox='pascal_voc'
    label_fields=['category_ids']
    try:
        config = utilities.read_config(config_path)
        augment_config = config["train" if augment else "validation"]["augmentations"]
    except KeyError as e:
        raise ValueError("Missing key in the configuration file: {}".format(e))
    
    if not augment_config:
        default_augmentations=config["train" if augment else "validation"]["default_augmentations"]
        for augment in default_augmentations:
            augment_config_type=augment.get('type')
            params = augment.get('params', {})
            
            if augment_config_type.lower()=='horizontalflip':
                augmentations.append(A.HorizontalFlip(**params))
                
            elif augment_config_type.lower()=='verticalflip':
                augmentations.append(A.VerticalFlip(**params))
                
            elif augment_config_type.lower()=='randomrotate90':
                augmentations.append(A.RandomRotate90(**params))
                
            elif augment_config_type.lower()=='randomcrop':
                augmentations.append(A.RandomCrop(**params))
                
            elif augment_config_type.lower()=='randombrightnesscontrast':
                augmentations.append(A.RandomBrightnessContrast(**params))
                
            elif augment_config_type.lower()=='normalize':
                augmentations.append(A.Normalize(**params))
                
            elif augment_config_type.lower()=='totensorv2':
                augmentations.append(ToTensorV2())
                
            elif augment_config_type.lower()=='bbox':
                format_bbox=params['format']
                label_fields=params['label_fields']
                
        transform = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(format=format_bbox, label_fields=label_fields)
        )
                
        return transform
    
    for augment in augment_config:
        augment_config_type=augment.get('type')
        params = augment.get('params', {})
        if augment_config_type.lower() == "randomsizedbboxsafecrop":
            try:
                augmentations.append(A.RandomSizedBBoxSafeCrop(**params))
            except KeyError as e:
                raise ValueError("Missing parameter for RandomSizedBBoxSafeCrop: {}".format(e))
            
        elif augment_config_type.lower() == "padifneeded":
            min_height = params.get('min_height')
            pad_height_divisor = params.get('pad_height_divisor', 'None')
            
            try:
                min_width = params['min_width']
                position = params.get('position', 'top_left')
                border_mode = params.get('border_mode', cv2.BORDER_CONSTANT)
                value = params.get('value', 0)
                mask_value = params.get('mask_value', None)
                always_apply = params.get('always_apply', False)
                p = params.get('p', 1.0)
            except KeyError as e:
                raise ValueError("Missing parameter for PadIfNeeded: {}".format(e))
            
            if min_height == 'None' and pad_height_divisor != 'None':
                augmentations.append(A.PadIfNeeded(
                    min_height=None,
                    min_width=min_width,
                    pad_height_divisor=pad_height_divisor,
                    position=position,
                    border_mode=border_mode,
                    value=value,
                    mask_value=mask_value,
                    always_apply=always_apply,
                    p=p))
            elif min_height != "None" and pad_height_divisor == "None":
                augmentations.append(A.PadIfNeeded(
                    min_height=min_height,
                    min_width=min_width,
                    position=position,
                    border_mode=border_mode,
                    value=value,
                    mask_value=mask_value,
                    always_apply=always_apply,
                    p=p))
                
        elif augment_config_type.lower() == "totensorv2":
            augmentations.append(ToTensorV2())
            
        elif augment_config_type.lower()=='horizontalflip':
                augmentations.append(A.HorizontalFlip(**params))
                
        elif augment_config_type.lower()=='verticalflip':
            augmentations.append(A.VerticalFlip(**params))
            
        elif augment_config_type.lower()=='randomrotate90':
            augmentations.append(A.RandomRotate90(**params))
            
        elif augment_config_type.lower()=='randomcrop':
            augmentations.append(A.RandomCrop(**params))
            
        elif augment_config_type.lower()=='randombrightnesscontrast':
            augmentations.append(A.RandomBrightnessContrast(**params))
            
        elif augment_config_type.lower()=='normalize':
            augmentations.append(A.Normalize(**params))
            
        elif augment_config_type.lower()=='totensorv2':
            augmentations.append(ToTensorV2())
            
        elif augment_config_type.lower()=='bbox':
                format_bbox=params['format']
                label_fields=params['label_fields']
                
        transform = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(format=format_bbox, label_fields=label_fields)
        )
            
    return transform

if __name__ == "__main__":
    augmentations = get_augmentations()
    print("Augmentations created:", augmentations)

