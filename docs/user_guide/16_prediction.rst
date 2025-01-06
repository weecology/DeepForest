
There are atleast four ways to make predictions with DeepForest.

1. Predict an image using `model.predict_image <https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_image>`_
2. Predict a tile using `model.predict_tile <https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_tile>`_
3. Predict a directory of using a csv file using `model.predict_file <https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_file>`_
4. Predict a batch of images using `model.predict_batch <https://deepforest.readthedocs.io/en/latest/source/deepforest.html#deepforest.main.deepforest.predict_batch>`_

1. Predict an image using model.predict_image
====================================================

.. code-block:: python

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


2. Predict a tile using model.predict_tile
===========================================

Predict a tile
~~~~~~~~~~~~~~

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the ``predict_tile`` function, which splits the tile into overlapping windows, performs prediction on each of the windows, and then reassembles the resulting annotations.

Let's show an example with a small image. For larger images, patch_size should be increased.

.. code-block:: python

   from deepforest import main
   from deepforest import get_data
   import matplotlib.pyplot as plt

   # Initialize the model class
   model = main.deepforest()

   # Load a pretrained tree detection model from Hugging Face
   model.load_model(model_name="weecology/deepforest-tree", revision="main")
   
   # Predict on large geospatial tiles using overlapping windows
   raster_path = get_data("OSBS_029.tif")
   predicted_raster = model.predict_tile(raster_path, patch_size=300, patch_overlap=0.25)
   plot_results(results)

.. note::

   The *predict_tile* function is sensitive to *patch_size*, especially when using the prebuilt model on new data.
   We encourage users to experiment with various patch sizes. For 0.1m data, 400-800px per window is appropriate, but it will depend on the density of tree plots. For coarser resolution tiles, >800px patch sizes have been effective.

3. Predict a directory of using a csv file using model.predict_file
====================================================


4. Predict a batch of images using model.predict_batch
====================================================

