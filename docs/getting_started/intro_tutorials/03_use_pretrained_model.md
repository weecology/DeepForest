# How do I use a pretrained model to predict an image?

1.5
# Predict a single image with a pretrained model

Below is the shortest path from a fresh install to plotted predictions on your screen.

```python
from deepforest import main, visualize
from importlib import resources

# 1️⃣  Load the pretrained tree-crown detector
model = main.deepforest()
model.use_release()  # one-time download and caching

# 2️⃣  Get the bundled demo image
img_path = resources.files("deepforest.data") / "OSBS_029.png"

# 3️⃣  Run the detector
crowns = model.predict_image(path=str(img_path))
print(crowns.head())  # pandas DataFrame (xmin, ymin, xmax, ymax, label, score)

# 4️⃣  Visualise results (opens a matplotlib window or inline in Jupyter)
visualize.plot_results(crowns)
```

Prediction uses CPU by default; add `model.to("cuda")` beforehand if you have a compatible GPU.

---

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