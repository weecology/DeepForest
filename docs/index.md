# DeepForest documentation

**Date**: |today| **Version**: |version|

<!-- **Download documentation**: [Zipped HTML](deepforest.zip) -->

**Previous versions**: Documentation of previous DeepForest versions is available at [weecology.org](https://deepforest.weecology.org/).

**Useful links**: [Binary Installers](https://pypi.org/project/deepforest) |  [Source Repository](https://github.com/weecology/DeepForest) |  [Issues & Ideas](https://github.com/weecology/DeepForest/issues) |  [Q&A Support](https://stackoverflow.com/questions/tagged/deepforest)

`DeepForest` is an open-source library providing tools for object detection and geospatial analysis in ecology, specifically for analyzing forest canopy data.

11.5
## Quick-start: detect tree crowns in seconds

Below is the shortest path from *zero* to predictions. Everything runs on a CPU-only laptop; a GPU just makes it faster.

```python
# Install once (uncomment if you have not installed DeepForest yet)
# !pip install deepforest

from deepforest import main, visualize
from importlib import resources
from pathlib import Path
import matplotlib.pyplot as plt

# 1) Load a pretrained tree-crown detector (≈200 MB download on first use)
model = main.deepforest()
model.use_release()  # or: model.load_model("weecology/deepforest-tree")

# 2) Grab a tiny demo image that ships with the package
sample_img = Path(resources.files("deepforest.data")) / "OSBS_029.png"

# 3) Predict crowns and visualise
crowns = model.predict_image(path=str(sample_img))
fig, ax = plt.subplots(figsize=(5, 5))
_ = visualize.plot_predictions(image_path=str(sample_img), pred_df=crowns, ax=ax)
plt.show()
```

👉 The code above downloads a single pretrained model the first time it is run and caches it locally for future sessions.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2
:margin: 0 0 0 0

:::{grid-item-card} Getting Started
:img-top: _static/index_getting_started.svg
:class-card: intro-card
:shadow: md

New to *DeepForest*? Check out the getting started guides. They contain an introduction to *DeepForest*'s main concepts and links to additional tutorials.

+++
```{button-ref} getting_started
:ref-type: ref
:color: secondary
:click-parent: true
:expand:
To the getting started guides
```
:::

:::{grid-item-card} User Guide
:img-top: _static/index_user_guide.svg
:class-card: intro-card
:shadow: md

The user guide provides in-depth information on the key concepts of DeepForest with useful background information and explanation.

+++
```{button-ref} user_guide
:ref-type: ref
:color: secondary
:class: sd-rounded-pill
:click-parent: true
:expand:
To the user guide
```
:::

:::{grid-item-card} Developer Guide
:img-top: _static/index_contribute.svg
:class-card: intro-card
:shadow: md

Want to improve DeepForest? The contributing guidelines will guide you through the process of improving and contributing to DeepForest.

+++
```{button-ref} development
:ref-type: ref
:color: secondary
:class: sd-rounded-pill
:click-parent: true
:expand:
To the development guide
```
:::

:::{grid-item-card} API Reference
:img-top: _static/index_api.svg
:class-card: intro-card
:shadow: md

Want to improve DeepForest? The contributing guidelines will guide you through the process of improving and contributing to DeepForest.

+++
```{button-ref} api
:ref-type: ref
:color: secondary
:class: sd-rounded-pill
:click-parent: true
:expand:
To the API reference
```
:::
::::


```{toctree}
:maxdepth: 3
:hidden:
:titlesonly:

getting_started/index
user_guide/index
source/modules
development/index
whatsnew/index
```
