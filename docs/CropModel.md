# The CropModel: Classifying objects after object detection

One of the most requested features since the early days of DeepForest was the ability to apply a follow-up model to predicted bounding boxes. For example, if we use the 'tree' or 'bird' backbone, but then we want to classify each of those detections with our model without retraining the upstream detector. Beginning in version 1.4.0, the CropModel class can be used in conjunction with predict_tile and predict_image methods. The general workflow is the object detection model is first applied, the prediction locations are extracted into images, optionally saved to disk, and a second model is applied on each crop image. A new column 'cropmodel_label' and 'cropmodel_score' will appear alongside the object detection model label and score. 

## Benefits

Why would you want to apply a model directly on each crop? Why not train a multi-class object detection model? This is certainly a reasonable approach, but there are a few benefits in particular common use-cases. 

* Object detection models require that all objects of a particular class are annotated within an image. This is often impossible for detailed category labels. For example, you might have bounding boxes for all 'trees' in an image, but only have species or health labels for a small portion of them based on ground surveys. 

* CropModels are simpler and more extendable. By decoupling the detection and classification workflows, you can seperately handle challenges like class imbalance and incomplete labels, without reducing the quality of the detections. We have found that training two stage object detection models to be finicky and involve reasonable knowledge on managing learning rates.

## Considerations

* Using a CropModel will be slower, since for each detection, the sensor data needs to be cropped and passed to the detector. This is definitely less efficient than using a combined classification/detection system like the multi-class detection models. With modern GPUs, this ofter matters less, but its something to be mindful of.

* The model knows only about the pixels that exist inside the crop, and cannot use features outside the bounding box. The lack of spatial awareness is a major limitation. It is possible, but untested that the multi-class detection model is better at this kind of task. Desgining a genuine box attention mechansim is probably better (https://arxiv.org/abs/2111.13087). 

## Use

To use the `CropModel` class, follow these steps:

1. Import the `CropModel` class into your code:

    ```python
    from crop_model import CropModel
    ```

2. Create an instance of the `CropModel` class:

    ```python
    model = CropModel()
    ```

3. Train the model using your training data:

    ```python
    model.train(X_train, y_train)
    ```

    Here, `X_train` represents the input features and `y_train` represents the corresponding crop yields.

4. Make predictions using the trained model:

    ```python
    predictions = model.predict(X_test)
    ```

    Here, `X_test` represents the input features for which you want to predict the crop yields.

## Training

To train the `CropModel`, you need a dataset containing 

## Customizing

The CropModel makes very few assumptions about the architecture and simply provides a container to make predictions at each detection. To specify a custom cropmodel, use the model argument.

```
from deepforest.model import CropModel
from torchvision.models import resnet101
backbone = resnet101(weights='DEFAULT')
crop_model = CropModel(num_classes=2, model=backbone)
```

One detail is that the preprocessing transform will differ for backbones, make sure to check the final lines

```
print(crop_model.get_transform(augment=True))
...
...
)>
    Resize(size=[224, 224], interpolation=bilinear, max_size=None, antialias=None)
    RandomHorizontalFlip(p=0.5)
)
```
To see the torchvision transform.Compose statement. You can overwrite this if needed for the torchvision.ImageFolder reader when reading existing images. 

```
def custom_transform(self, augment):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(self.normalize)
    <add transforms here>
    data_transforms.append(transforms.Resize([<new size>,<new size>]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)
crop_model.get_transform = custom_transform
```

Or if running from within memory crops during prediction, you can pass the transform and augment flag to predict methods

```
m.predict_tile(...,crop_transform=custom_transform, augment=False)
```

This allows full flexibility over the preprocessing steps. For further customization, you can subclass the CropModel object and change methods such as learning rate optimzation, evaluation steps and all other pytorch lightning hooks.

```
class CustomCropModel(CropModel):
    def training_step(self, batch, batch_idx):
        # Custom training step implementation
        # Add your code here
        return loss

# Create an instance of the custom CropModel
model = CustomCropModel()
```



