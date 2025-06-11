# The CropModel

## Classifying Objects After Object Detection

One of the most requested features since the early days of DeepForest was the ability to apply a follow-up model to predicted bounding boxes. For example, if you use the 'tree' or 'bird' backbone, you might want to classify each detection with your own model without retraining the upstream detector.  

Beginning in version 1.4.0, the `CropModel` class can be used in conjunction with `predict_tile` and `predict_image` methods. The general workflow involves first applying the object detection model, extracting the prediction locations into images (which can optionally be saved to disk), and then applying a second model on each cropped image.  

New columns `cropmodel_label` and `cropmodel_score` will appear alongside the object detection model's label and score.  

## Benefits

Why would you want to apply a model directly to each crop? Why not train a multi-class object detection model?  

While that approach is certainly valid, there are a few key benefits to using CropModels, especially in common use cases:  

- **Flexible Labeling**: Object detection models require that all objects of a particular class be annotated within an image, which can be impossible for detailed category labels. For example, you might have bounding boxes for all 'trees' in an image, but only have species or health labels for a small portion of them based on ground surveys. Training a multi-class object detection model would mean training on only a portion of your available data. 
- **Simpler and Extendable**: CropModels decouple detection and classification workflows, allowing separate handling of challenges like class imbalance and incomplete labels, without reducing the quality of the detections. Two-stage object detection models can be finicky with similar classes and often require expertise in managing learning rates. 
- **New Data and Multi-sensor Learning**: In many applications, the data needed for detection and classification may differ. The CropModel concept provides an extendable piece that allows for advanced pipelines.

## Considerations

- **Efficiency**: Using a CropModel will be slower, as for each detection, the sensor data needs to be cropped and passed to the detector. This is less efficient than using a combined classification/detection system like multi-class detection models. While modern GPUs mitigate this to some extent, it is still something to be mindful of.
- **Lack of Spatial Awareness**: The model knows only about the pixels inside the crop and cannot use features outside the bounding box. This lack of spatial awareness can be a major limitation. It is possible, but untested, that multi-class detection models might perform better in such tasks. A box attention mechanism, like in [this paper](https://arxiv.org/abs/2111.13087), could be a better approach.

## Usage

### Single Crop Model

Consider a test file with tree boxes and an 'Alive/Dead' label that comes with all DeepForest installations:

```python

import pandas as pd
from deepforest import model
from deepforest import main as m
from deepforest.utilities import get_data

df = pd.read_csv(get_data("testfile_multi.csv"))
crop_model = model.CropModel(num_classes=2)
# Or set up the crop model or load weights model.CropModel.load_from_checkpoint(<path>)

m.create_trainer()
result = m.predict_tile(path=path, crop_model=crop_model)
```

```python
result.head()
# Output:
#    xmin   ymin   xmax  ...    image_path cropmodel_label  cropmodel_score
# 0  273.0  230.0  313.0  ...  SOAP_061.png               1         0.519510
# 1   47.0   82.0   81.0  ...  SOAP_061.png               1         0.506423
# 2    0.0   72.0   34.0  ...  SOAP_061.png               1         0.505258
# 3  341.0   40.0  374.0  ...  SOAP_061.png               1         0.517231
# 4    0.0  183.0   26.0  ...  SOAP_061.png               1         0.513122
```

### Multiple Crop Models

You can also pass multiple crop models to `predict_tile`. Each model's predictions and confidence scores will be stored in separate columns.

```python
crop_model1 = model.CropModel(num_classes=2)
crop_model2 = model.CropModel(num_classes=3)
result = m.predict_tile(path=path, crop_model=[crop_model1, crop_model2])
```

```python
result.head()
# Output:
#     xmin   ymin   xmax   ymax label     score    image_path  cropmodel_label_0  cropmodel_score_0  cropmodel_label_1  cropmodel_score_1  
# 0  273.0  230.0  313.0  275.0  Tree  0.882591  SOAP_061.png                   0           0.650223                  1           0.383726   
# 1   47.0   82.0   81.0  120.0  Tree  0.740889  SOAP_061.png                   0           0.621586                  1           0.376401   
# 2    0.0   72.0   34.0  116.0  Tree  0.735777  SOAP_061.png                   0           0.614928                  1           0.394649  
# 3  341.0   40.0  374.0   77.0  Tree  0.668367  SOAP_061.png                   0           0.598883                  1           0.386490  
# 4    0.0  183.0   26.0  235.0  Tree  0.664668  SOAP_061.png                   0           0.538162                  1           0.439823  
```

A `CropModel` is a PyTorch Lightning object and can also be used like any other model.

```python

import torch
from deepforest.model import CropModel

# Test forward pass
x = torch.rand(4, 3, 224, 224)
output = crop_model.forward(x)
assert output.shape == (4, 2)
```

## Writing Crops to Disk

We can either classify crops in memory or save them to disk.

```python

import os

boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
image_path = os.path.join(os.path.dirname(get_data("SOAP_061.png")), df["image_path"].iloc[0])
crop_model.write_crops(boxes=boxes, labels=df.label.values, image_path=image_path, savedir=tmpdir)
```

This saves each crop in labeled folders (`Alive/Dead`).

## Training

You can train a new model using PyTorch Lightning:

```python

from deepforest.model import CropModel

crop_model.create_trainer(fast_dev_run=True)
# Get the data stored from the write_crops step above.
crop_model.load_from_disk(train_dir=tmpdir, val_dir=tmpdir)
crop_model.trainer.fit(crop_model)
crop_model.trainer.validate(crop_model)
```

# Customizing

The `CropModel` makes very few assumptions about the architecture and simply provides a container to make predictions at each detection. To specify a custom CropModel, use the `model` argument.

```python
from deepforest.model import CropModel
from torchvision.models import resnet101
backbone = resnet101(weights='DEFAULT')
crop_model = CropModel(num_classes=2, model=backbone)
```

One detail to keep in mind is that the preprocessing transform will differ for backbones. Make sure to check the final lines:

```python
print(crop_model.get_transform(augment=True))

# Output:
# Resize(size=[224, 224], interpolation=bilinear, max_size=None, antialias=None)
# RandomHorizontalFlip(p=0.5)
```

To see the `torchvision` `transform.Compose` statement, you can overwrite this if needed for the `torchvision.ImageFolder` reader when reading existing images.

```python

from torchvision import transforms

def custom_transform(self, augment):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(self.normalize)
    # <add transforms here>
    data_transforms.append(transforms.Resize([<new size>, <new size>]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)
crop_model.get_transform = custom_transform
```

Or, when running from memory crops during prediction, you can pass the transform and augment flag to the `predict` methods.

```python
from deepforest import main as m

m.predict_tile(..., crop_transform=custom_transform, augment=False)
```

This allows full flexibility over the preprocessing steps. For further customization, you can subclass the `CropModel` object and change methods such as learning rate optimization, evaluation steps, and all other PyTorch Lightning hooks.

```python
class CustomCropModel(CropModel):
    def training_step(self, batch, batch_idx):
        # Custom training step implementation
        # Add your code here
        return loss

# Create an instance of the custom CropModel
model = CustomCropModel()
```

## Making Predictions Outside of predict_tile

While `predict_tile` provides a convenient way to run predictions on detected objects, you can also use the CropModel directly for classification tasks. This is useful when you have pre-cropped images or want to run classification independently.

### Loading a Trained Model

```python
from deepforest.model import CropModel
from pytorch_lightning import Trainer
from torchvision.datasets import ImageFolder
import numpy as np

# Load a trained model from checkpoint
cropmodel = CropModel.load_from_checkpoint("path/to/checkpoint.ckpt")

# The model will automatically load:
# - The model architecture and weights
# - The label dictionary mapping class names to indices
# - The number of classes
# - Any hyperparameters saved during training
```

### Making Predictions on a Dataset

```python
# Create dataset using the model's transform
val_ds = ImageFolder(
    root="path/to/images",
    transform=cropmodel.get_transform(augment=False)
)

# Create dataloader
crop_dataloader = cropmodel.predict_dataloader(val_ds)

# Run prediction
trainer = Trainer(
    gpus=1, 
    accelerator="gpu", 
    max_epochs=1, 
    logger=False, 
    enable_checkpointing=False
)
crop_results = trainer.predict(cropmodel, crop_dataloader)

# Process results using the built-in postprocessing method
label, score = cropmodel.postprocess_predictions(crop_results)

# Convert numeric labels to class names
label_names = [cropmodel.numeric_to_label_dict[x] for x in label]
```

### Making Predictions on Single Images

You can also make predictions on individual images or batches:

```python
import torch
from PIL import Image

# Load and preprocess a single image
image = Image.open("path/to/image.jpg")
transform = cropmodel.get_transform(augment=False)
tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = cropmodel(tensor)
    # Convert to numpy for postprocessing
    output = output.cpu().numpy()
    # Use the same postprocessing method
    label, score = cropmodel.postprocess_predictions([output])
    class_name = cropmodel.numeric_to_label_dict[label[0]]
    confidence = score[0]
```

## Model Architecture and Training

The CropModel uses a ResNet-50 backbone by default, but can be customized with any PyTorch model. The model includes:

- A classification head with the specified number of classes
- Standard image preprocessing (resize to 224x224, normalization)
- Data augmentation during training (random horizontal flips)
- Accuracy and precision metrics for evaluation

### Training Process

```python
# Initialize model
crop_model = CropModel(num_classes=2)

# Create trainer
crop_model.create_trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1
)

# Load data
crop_model.load_from_disk(
    train_dir="path/to/train",
    val_dir="path/to/val"
)

# Train
crop_model.trainer.fit(crop_model)

# Validate
crop_model.trainer.validate(crop_model)

# Save checkpoint
crop_model.trainer.save_checkpoint("model.ckpt")
```

### Evaluation

The model provides several evaluation metrics:

```python
# Get validation metrics
metrics = crop_model.trainer.validate(crop_model)

# Get confusion matrix
images, labels, predictions = crop_model.val_dataset_confusion(return_images=True)
```

### Confusion Matrix Visualization

You can visualize the confusion matrix in several ways:

```python
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns

# Method 1: Using torchmetrics
metric = MulticlassConfusionMatrix(num_classes=crop_model.num_classes)
metric.update(preds=predictions, target=labels)
fig, ax = metric.plot()
plt.title("Confusion Matrix")
plt.show()

# Method 2: Using seaborn with val_dataset_confusion
images, labels, predictions = crop_model.val_dataset_confusion(return_images=True)
confusion_matrix = np.zeros((crop_model.num_classes, crop_model.num_classes))
for true, pred in zip(labels, predictions):
    confusion_matrix[true][pred] += 1

# Plot with seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, 
            annot=True, 
            fmt='g',
            xticklabels=list(crop_model.label_dict.keys()),
            yticklabels=list(crop_model.label_dict.keys()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Get per-class metrics
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score

precision = MulticlassPrecision(num_classes=crop_model.num_classes)
recall = MulticlassRecall(num_classes=crop_model.num_classes)
f1 = MulticlassF1Score(num_classes=crop_model.num_classes)

precision_score = precision(torch.tensor(predictions), torch.tensor(labels))
recall_score = recall(torch.tensor(predictions), torch.tensor(labels))
f1_score = f1(torch.tensor(predictions), torch.tensor(labels))

print(f"Precision: {precision_score:.3f}")
print(f"Recall: {recall_score:.3f}")
print(f"F1 Score: {f1_score:.3f}")
```

This will give you a comprehensive view of your model's performance, including:
- A visual confusion matrix showing true vs predicted classes
- Per-class precision, recall, and F1 scores
- The ability to identify which classes are most commonly confused with each other

The confusion matrix is particularly useful for:
- Identifying class imbalance issues
- Finding classes that are frequently confused
- Understanding the model's strengths and weaknesses
- Guiding decisions about data collection and model improvement

## Advanced Usage

### Custom Model Architecture

You can use any PyTorch model as the backbone:

```python
from torchvision.models import resnet101

# Initialize with custom model
backbone = resnet101(weights='DEFAULT')
crop_model = CropModel(
    num_classes=2,
    model=backbone
)
```

### Custom Training Loop

You can subclass CropModel to customize the training process:

```python
class CustomCropModel(CropModel):
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs, y)
        
        # Add custom metrics
        self.log("custom_metric", value)
        
        return loss
```