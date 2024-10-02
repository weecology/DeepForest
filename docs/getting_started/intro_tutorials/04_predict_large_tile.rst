How do I predict on large geospatial tiles?
===========================================

Predict a tile
~~~~~~~~~~~~~~

Large tiles covering wide geographic extents cannot fit into memory during prediction and would yield poor results due to the density of bounding boxes. Often provided as geospatial .tif files, remote sensing data is best suited for the ``predict_tile`` function, which splits the tile into overlapping windows, performs prediction on each of the windows, and then reassembles the resulting annotations.

Let’s show an example with a small image. For larger images, patch_size should be increased.

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

   # View boxes overlaid when return_plot=True; otherwise, boxes are returned
   plt.imshow(predicted_raster)
   plt.show()

.. note::

   The *predict_tile* function is sensitive to *patch_size*, especially when using the prebuilt model on new data.
   We encourage users to experiment with various patch sizes. For 0.1m data, 400-800px per window is appropriate, but it will depend on the density of tree plots. For coarser resolution tiles, >800px patch sizes have been effective.
