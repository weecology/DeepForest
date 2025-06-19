# How do I load a pretrained model from Hugging Face?

This function loads pretrained DeepForest models from Hugging Face, with support for different model revisions. Additionally, you can save the model configuration and weights using `save_pretrained` and reload it later with `from_pretrained`.

## `load_model`

### Description

The `load_model` function loads a pretrained model from Hugging Face using the repository name (`model_name`) and the desired model version (`revision`). This is useful for tasks such as tree crown detection, but it can also load bird detection models with custom configurations.

### Arguments

- `model_name` (str): A repository ID for Hugging Face in the form `organization/repository`. Default is `"weecology/deepforest-tree"`.

  you can choose from:
    - weecology/deepforest-tree
    - weecology/deepforest-bird
    - weecology/deepforest-livestock
    - weecology/everglades-nest-detection
    - weecology/cropmodel-deadtrees
- `revision` (str): The model version (e.g., 'main', 'v1.0.0', etc.). Default is `'main'`.

### Returns

- `object`: A trained PyTorch model with its configuration and weights.

### Example Usage

#### Load a Model and Predict an Image

```python
from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_results
# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

sample_image_path = get_data("OSBS_029.png")
img = model.predict_image(path=sample_image_path)
plot_results(img)
```

1.5
# Load pretrained models – three common patterns

DeepForest ships with a small *model zoo* hosted on Hugging Face.  Below are the **three most frequent ways** to obtain a model object.

## 1. One-liner for the default tree model
```python
from deepforest import main
model = main.deepforest()
model.use_release()           # ≈200 MB download first time, then cached
```

## 2. Explicit repo / revision
```python
from deepforest import main
m = main.deepforest()
# Any public or private model on Hugging Face works
m.load_model("weecology/deepforest-tree", revision="v1.5.0")
```

## 3. Local checkpoint – resume training or inference
```python
from deepforest import main
ckpt_path = "checkpoints/epoch=4-step=500.ckpt"
model = main.deepforest()
model = model.load_from_checkpoint(ckpt_path)  # retains training config
```

Once loaded, the same prediction API is used:
```python
from deepforest import get_data
img_path = get_data("OSBS_029.png")
preds = model.predict_image(path=img_path)
print(preds.head())
```

---
