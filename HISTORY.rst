====================
DeepForest Changelog
====================

Version 1.3.5 (Date: Mar 12, 2024)
----------------------------------

- **New Feature:** Added support for multiple annotation types including point, box, and polygon.
- **New Utility:** Created ``utilities.download_ArcGIS_REST`` function to download tiles from the ArcGIS REST API (e.g., NAIP imagery).

Version 1.3.3 (Date: Mar 12, 2024)
----------------------------------

- **Enhancement:** ``split_raster`` now allows ``annotations_file`` to be ``None``, enabling flexibility during data preprocessing.

Version 1.3.0 (Date: Dec 3, 2023)
----------------------------------

- **Deprecation:** Removed ``IoU_Callback`` for better alignment with the PyTorch Lightning API (see `issue <https://github.com/Lightning-AI/pytorch-lightning/issues/19101>`_).
- **Refactor:** Evaluation code now leverages the PyTorch Lightning evaluation loop for result calculation during training.
- **Refactor:** Simplified ``image_callbacks`` by using predictions directly. No need to specify the root directory or CSV file, as the evaluation file is assumed.

Version 1.1.3 (Date: Nov 9, 2021)
----------------------------------

- **Enhancement:** Added box coordinates to the evaluation results frame for better result tracking.

Version 1.1.2 (Date: Sep 30, 2021)
----------------------------------

- **Bug Fix:** Fixed incorrect precision calculation in ``class_recall.csv`` when multiple classes were present.

Version 1.1.1 (Date: Sep 14, 2021)
----------------------------------

- **Update:** ``project_boxes`` now includes output options for both ``predict_tile`` and ``predict_image``.
- **New Feature:** Introduced ``annotations_to_shapefile``, which reverses ``shapefile_to_annotations`` functionality.
  Thanks to @sdtaylor for this contribution.

Version 1.1.0 (Date: Aug 5, 2021)
----------------------------------

1. **Enhancement:** Empty frames are now allowed by passing annotations with 0's for all coordinates. Example format:
   ::

     image_path, 0, 0, 0, 0, "Tree"

2. **New Feature:** Introduced ``check_release`` to reduce GitHub rate limit issues. When using ``use_release()``, the local model will be used if ``check_release = False``.

Version 1.0.9 (Date: Jul 14, 2021)
----------------------------------

- **Enhancement:** Improved default dtype for Windows users, thanks to @ElliotSalisbury for the contribution.

Version 1.0.0 (Date: Jun 7, 2021)
----------------------------------

- **Major Update:** Transitioned from TensorFlow to a PyTorch backend, enhancing performance and flexibility.

Version 0.1.34 (Date: )
-----------------------

- **Optimization:** Profiled dataset and evaluation code, significantly improving evaluation performance.

Version 0.1.30 (Date: )
-----------------------

- **Bug Fix:** Resolved issues to allow learning rate monitoring and decay based on ``val_classification_loss``.
