# DeepForest Model Loader

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

- `` (object): A trained PyTorch model with its configuration and weights.

### Example Usage

#### Load a Model

```python
from deepforest import main
from deepforest import get_data
import matplotlib.pyplot as plt

# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

sample_image_path = get_data("OSBS_029.png")
img = model.predict_image(path=sample_image_path, return_plot=True)

```
