# Using DeepForest from R

An R wrapper for DeepForest is available in the [deepforestr package](https://github.com/weecology/deepforestr).
Commands are very similar with some minor differences due to how the wrapping process
using [reticulate](https://rstudio.github.io/reticulate/) works.

## Installation

`deepforestr` is an R wrapper for the Python package, [DeepForest](https://deepforest.readthedocs.io/en/latest/).
This means that *Python* and the `DeepForest` Python package need to be installed first.

### Basic Installation

If you just want to use DeepForest from within R run the following commands in R.
This will create a local Python installation that will only be used by R and install the needed Python package for you.
If installing on Windows you need to [install RTools](https://cran.r-project.org/bin/windows/Rtools/) before installing the R package.

```R
install.packages('reticulate') # Install R package for interacting with Python
reticulate::install_miniconda() # Install Python
reticulate::py_install(c('gdal', 'rasterio', 'fiona')) # Install spatial dependencies via conda
reticulate::conda_remove('r-reticulate', packages = c('mkl')) # Remove package that causes conflicts on Windows (and maybe macOS)
reticulate::py_install('DeepForest', pip=TRUE) # Install the Python retriever package
devtools::install_github('weecology/deepforestr') # Install the R package for running the retriever
install.packages('raster') # For visualizing output for rasters
```

**After running these commands restart R.**

### Advanced Installation for Python Users

If you are using Python for other tasks you can use `deepforestr` with your existing Python installation
(though the [basic installation](#basic-installation) above will still work by creating a separate miniconda install and Python environment).

#### Install the `DeepForest` Python package

Install the `DeepForest` Python package into your prefered Python environment
using `pip`:

```bash
pip install DeepForest
```

#### Select the Python environment to use in R

`deepforestr` will try to find Python environments with `DeepForest`
(see the `reticulate` documentation on [order of discovery](https://rstudio.github.io/reticulate/articles/versions.html#order-of-discovery-1) for more details) installed.
Alternatively you can select a Python environment to use when working with `deepforestr` (and other packages using `reticulate`).

The most robust way to do this is to set the `RETICULATE_PYTHON` environment
variable to point to the preferred Python executable:

```R
Sys.setenv(RETICULATE_PYTHON = "/path/to/python")
```

This command can be run interactively or placed in `.Renviron` in your home directory.

Alternatively you can select the Python environment through the `reticulate` package for either `conda`:

```R
library(reticulate)
use_conda('name_of_conda_environment')
```

or `virtualenv`:

```R
library(reticulate)
use_virtualenv("path_to_virtualenv_environment")
```

You can check to see which Python environment is being used with:

```R
py_config()
```

#### Install the `deepforestr` R package

```R
remotes::install_github("weecology/deepforestr") # development version from GitHub
```

## Getting Started

### Load the current release model

```R
library(deepforestr)

model = df_model()
model$use_release()
```

### Predict a single image

#### Return the bounding boxes in a data frame

```R
image_path = get_data("OSBS_029.png") # Gets a path to an example image
bounding_boxes = model$predict_image(path=image_path, return_plot=FALSE)
head(bounding_boxes)
```

#### Return an image for visualization

```R
image_path = get_data("OSBS_029.png") # Gets a path to an example image
predicted_image = model$predict_image(path=image_path, return_plot=TRUE)
plot(raster::as.raster(predicted_image[,,3:1]/255))
```

### Predict a tile

#### Return the bounding boxes in a data frame 

```R
raster_path = get_data("OSBS_029.tif") # Gets a path to an example raster tile
bounding_boxes = model$predict_tile(raster_path, return_plot=FALSE)
head(bounding_boxes)
```

#### Return an image for visualization

```R
raster_path = get_data("OSBS_029.tif") # Gets a path to an example raster tile
predicted_raster = model$predict_tile(raster_path, return_plot=TRUE, patch_size=300L, patch_overlap=0.25)
plot(raster::as.raster(predicted_raster[,,3:1]/255))
```

Note at `patch_size` is an integer value in Python and therefore needs to have the `L` at the end of the number in R to make it an integer.

### Predict a set of annotations

```R
csv_file = get_data("testfile_deepforest.csv")
root_dir = get_data(".")
boxes = model$predict_file(csv_file=csv_file, root_dir = root_dir, savedir=".")
```

### Training

#### Set the training configuration

```R
annotations_file = get_data("testfile_deepforest.csv")

model$config$epochs = 1
model$config["save-snapshot"] = FALSE
model$config$train$csv_file = annotations_file
model$config$train$root_dir = get_data(".")
```

Optionally turn on `fast_dev_run` for testing and debugging:

```R
model$config$train$fast_dev_run = TRUE
```

#### Train the model

```R
model$create_trainer()
model$trainer$fit(model)
```

### Evaluation

```R
csv_file = get_data("OSBS_029.csv")
root_dir = get_data(".")
results = model$evaluate(csv_file, root_dir, iou_threshold = 0.4)
```

### Saving & Loading Models

#### Saving a model after training

```R
model$trainer$save_checkpoint("checkpoint.pl")
```

#### Loading a saved model

```R
new_model = df_model()
after = new_model$load_from_checkpoint("checkpoint.pl")
pred_after_reload = after$predict_image(path = img_path)
```

*Note that when reloading models, you should carefully inspect the model parameters, such as the score_thresh and nms_thresh.
These parameters are updated during model creation and the config file is not read when loading from checkpoint!
It is best to be direct to specify after loading checkpoint.*