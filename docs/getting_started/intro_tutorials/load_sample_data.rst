How do I use the package sample data?
=====================================

Sample data
~~~~~~~~~~~

DeepForest comes with a small set of sample data that can be used to test out the provided examples. The data resides in the DeepForest data directory. Use the ``get_data`` helper function to locate the path to this directory, if needed.

.. code-block:: python

   from deepforest import get_data
   # Retrieve sample image path
   sample_image = get_data("OSBS_029.png")
   print(sample_image)
   # '[path]...../deepforest/data/OSBS_029.png'

To use images other than those in the sample data directory, provide the full path for the images.

.. code-block:: python

   from deepforest import main, get_data

   # Initialize the model and load the pre-trained release model
   model = main.deepforest()

   # Load a pretrained tree detection model from Hugging Face
   model.load_model(model_name="weecology/deepforest-tree", revision="main")

   # Use predict_image to get bounding boxes from a custom image path
   image_path = get_data("OSBS_029.png")
   boxes = model.predict_image(path=image_path, return_plot=False)

   # Output bounding boxes
   print(boxes)

::

   >>> boxes
        xmin   ymin   xmax   ymax label     score    image_path
   0   330.0  342.0  373.0  391.0  Tree  0.802979  OSBS_029.png
   1   216.0  206.0  248.0  242.0  Tree  0.778803  OSBS_029.png
   2   325.0   44.0  363.0   82.0  Tree  0.751573  OSBS_029.png
   3   261.0  238.0  296.0  276.0  Tree  0.748605  OSBS_029.png
   4   173.0    0.0  229.0   33.0  Tree  0.738210  OSBS_029.png
   5   258.0  198.0  291.0  230.0  Tree  0.716250  OSBS_029.png
   6    97.0  305.0  152.0  363.0  Tree  0.711664  OSBS_029.png
   7    52.0   72.0   85.0  108.0  Tree  0.698782  OSBS_029.png

