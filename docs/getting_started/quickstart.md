# Quickstart – Your First Prediction

Welcome to DeepForest!  This one–page guide shows how to install the package, load a ready-to-use model, and run your **first tree-crown prediction** in just a few lines of code.

## 1&nbsp;·&nbsp;Install

If you have conda available, the fastest way is:

```bash
conda install -c conda-forge deepforest
```

Prefer `pip`?  That works too:

```bash
pip install deepforest
```

## 2&nbsp;·&nbsp;Run a prediction

The snippet below will download a small sample image that ships with the library, run the pretrained *DeepForest-Tree* model, and pop up a figure with the predicted crowns.

```python
from deepforest import main, get_data, visualize

# 1.  Create a model object
model = main.deepforest()

# 2.  Download weights of the published tree-detection model
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# 3.  Grab a sample image (~1 MB) that is packaged with DeepForest
image_path = get_data("OSBS_029.png")

# 4.  Run the model
crowns = model.predict_image(path=image_path)

# 5.  Display the results
visualize.plot_results(crowns)
```

```{tip}
•  **Slow download the first time?**  The model weights (~95 MB) are cached, so future runs are instant.

•  Running on a machine **without a GPU** is perfectly fine for small images like this.  For large tiles see the *Predicting on large mosaics* tutorial.
```

## 3&nbsp;·&nbsp;Next steps

* **Predict on large mosaics** – read the [large-tiles tutorial](intro_tutorials/04_predict_large_tile.md).
* **Fine-tune on your data** – see the training walkthrough in the User Guide.
* **Need help?**  Ask on the [GitHub discussion board](https://github.com/weecology/DeepForest/discussions).