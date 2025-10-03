# How do I use a pretrained model to predict an image?

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

```{image} ../../../www/getting_started1.png
:align: center
```

```{note}
**Please note that this video was made before the deepforest-pytorch -> deepforest name change.**
```

```{raw} html
<div style="position: relative; padding-bottom: 56.25%; height: 0;">
  <iframe src="https://www.loom.com/embed/f80ed6e3c7bd48d4a20ae32167af3d8c"
  frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen
  style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
  </iframe>
</div>
```

For single images, `predict_image` can read an image from memory or file and return predicted bounding boxes.
