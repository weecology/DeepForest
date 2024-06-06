# The CropModel: Classifying objects after object detection

One of the most requested features since the early days of DeepForest was the ability to apply a follow-up model to predicted bounding boxes. For example, if we use the 'tree' or 'bird' backbone, but then we want to classify each of those detections with our model without retraining the upstream detector. Beginning in version 1.4.0, the CropModel class can be used in conjunction with predict_tile and predict_image methods. The general workflow is the object detection model is first applied, the prediction locations are extracted into images, optionally saved to disk, and a second model is applied on each crop image. A new column 'cropmodel_label' and 'cropmodel_score' will appear alongside the object detection model label and score. 

## Benefits

Why would you want to apply a model directly on each crop? Why not train a multi-class object detection model? This is certainly a reasonable approach, but there are a few benefits in particular common use-cases. 

* Object detection models require that all objects of a particular class are annotated within an image. This is often impossible for detailed category labels. For example, you might have bounding boxes for all 'trees' in an image, but only have species or health labels for a small portion of them based on ground surveys. 

* CropModels are simpler and more extendable. By decoupling the detection and classification workflows, you can seperately handle challenges like class imbalance and incomplete labels, without reducing the quality of the detections. We have found that training two stage object detection models to be finicky and involve reasonable knowledge on managing learning rates.

* New data and multi-sensor learning. For many applications the data needed for detection and classification may be different. The CropModel concept allows an extendable piece that can allow others to make more advanced pipelines.

## Considerations

* Using a CropModel will be slower, since for each detection, the sensor data needs to be cropped and passed to the detector. This is definitely less efficient than using a combined classification/detection system like the multi-class detection models. With modern GPUs, this ofter matters less, but its something to be mindful of.

* The model knows only about the pixels that exist inside the crop, and cannot use features outside the bounding box. The lack of spatial awareness is a major limitation. It is possible, but untested that the multi-class detection model is better at this kind of task. Desgining a genuine box attention mechansim is probably better (https://arxiv.org/abs/2111.13087). 

## Use

Consider a testfile with tree boxes and a 'Alive/Dead' label that comes with all DeepForest installations

```
df = pd.read_csv(get_data("testfile_multi.csv"))
crop_model = model.CropModel(num_classes=2)
```

This is a pytorch-lightning object and can be trained like any other DeepForest model. 

```
# Test forward pass
x = torch.rand(4, 3, 224, 224)
output = crop_model.forward(x)
assert output.shape == (4, 2)
```

The only difference is now we don't have boxes, we are classifier entire crops. We can do this within memory, or by writing a set of crops to disk. Let's start by writing to disk.

```
boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
image_path = os.path.join(os.path.dirname(get_data("SOAP_061.png")),df["image_path"].iloc[0])
crop_model.write_crops(boxes=boxes,labels=df.label.values,image_path=image_path, savedir=tmpdir)
```

This crops each box location and saves them in a folder with the label name. Now we have two folders in the savedir location, a 'Alive' and a 'Dead' folder.

## Training

We could train a new model from here in typical pytorch-lightning syntax. 

```
crop_model.create_trainer(fast_dev_run=True)
crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)
crop_model.trainer.fit(crop_model)
crop_model.trainer.validate(crop_model)
```

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



