How do I make the predictions better?
-------------------------------------

Given the enormous array of background types and image acquisition
environments, it is unlikely that your image will be perfectly predicted
by the prebuilt model. Below are some tips and some general guidelines
to improve predictions.

Get suggestions on how to improve a model by using the `discussion
board <https://github.com/weecology/DeepForest/discussions>`__. Please
be aware that only feature requests or bug reports should be posted on
the `issues’ page <https://github.com/weecology/DeepForest/issues>`__.

Check patch size
----------------

The prebuilt model was trained on 10cm data at 400px crops. The model is
sensitive to predicting new image resolutions that differ. We have found
that increasing the patch size works better on higher quality data. For
example, here is a drone collected data at the standard 400px.

.. code:: python

   tile = model.predict_tile("/Users/ben/Desktop/test.jpg",return_plot=True,patch_overlap=0,iou_threshold=0.05,patch_size=400)

|image1|

Acceptable, but not ideal.

Here is 1000 px patches.

|image2|

improved.

For more on this theme see:
https://github.com/weecology/DeepForest_demos/blob/master/street_tree/StreetTrees.ipynb

The cross-resolution prediction remains an open area of debate and
should be explored carefully. For images similar to the 10cm data used
to train the release model, keeping a 40m focal view is likely a good
choice. For coarser resolutions, the answer will be best found through
trial and error, but it is almost certainly a wider field of view. This
is likely less of an issue if you retrain the model based on data at the
desired resolution rather than directly using the release model to
predict trees at novel resolutions.

IoU threshold
-------------

Object detection models have no inherent logic about overlapping
bounding box proposals. For tree crown detection, we expect trees in
dense forests to have some overlap but not be completely intertwined. We
therefore apply a postprocessing filter called ‘non-max-suppression’,
which is a common approach in the broader computer vision literature.
This routine starts with the boxes with the highest confidence scores
and removes any overlapping boxes greater than the
intersection-over-union threshold (IoU). Intersection-over-union is the
most common object detection metric, defined as the area of intersection
between two boxes divided by the area of union. If users find that there
is too much overlap among boxes, increasing the IoU will return fewer
boxes. To increase the number of overlapping boxes, reduce the IoU
threshold.

Annotate local training data
----------------------------

Ultimately, training a proper model with local data is the best chance
at getting good performance. See ``DeepForest.train()``

Here is a nice example of training a model on local data from a user in [cattle detection](https://edsbook.org/notebooks/gallery/95199651-9e81-4cae-a3a7-66398a9a5f62/notebook).

------

We welcome feedback on both the python package as well as the algorithm
performance. Please submit detailed issues to the Github repo `issues’
page <https://github.com/weecology/DeepForest/issues>`__.

.. |image1| image:: ../../www/example_patch400.png
.. |image2| image:: ../../www/example_patch1000.png