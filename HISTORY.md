# DeepForest Changelog

## Version x.x.x (Date: )

- **Deprecation:** `predict_tile` in `deepforest/main.py`. The `raster_path` argument is deprecated and will be removed in a future version.
- **Enhancement:** Updated the `predict_tile()` function to use the `path` argument (replacing `raster_path`) across all code, tests, and documentation for consistency.

## Version 1.5.2 (Date: Feb 7, 2025)

The major innovations are:

1. Improve Tests on edge cases

Additional features and enhancements include:

- **Documentation:** Improved documentation workflow and version available

## Version 1.5.0 (Date: Jan 15, 2024)

The major innovations are:

1. Restructured package layout by moving code into `src/` directory for better organization
2. Added batch prediction capabilities for improved processing of multiple images
3. Implemented support for image dataframes to allow more flexible input formats
4. Created `plot_annotations` method for better visualization of predictions
5. Added out-of-memory dataset sample for handling large datasets efficiently

Additional features and enhancements include:

- **Enhancement:** Reorganized package structure to follow modern Python packaging standards
- **Enhancement:** Enhanced test coverage for new features
- **Enhancement:** Improved code organization and maintainability
- **Documentation:** Added docformatter for consistent docstring formatting
- **Documentation:** Improved documentation clarity and organization
- **Documentation:** Added new contributor: Dingyi Fang

## Version 1.4.1 (Date: Oct 26, 2024)

- **Enhancement:** Use GitHub Actions to publish the package.

## Version 1.4.0 (Date: Oct 9, 2024)

The major innovations are:

1. New model loading framework using HuggingFace. DeepForest models are now available on [HuggingFace](https://huggingface.co/weecology). The models can be loaded using `load_model()` and used for inference.
2. An all-purpose `read_file` method is introduced to read annotations from various formats including CSV, JSON, and Pascal VOC.
3. The `CropModel` class is introduced to classify detected objects using a trained classification model. Use when a multi-class DeepForest model is not sufficiently flexible, such as when new data sources are used for fine-grained classification and class imbalance.
4. `deepforest.visualize.plot_results` is now the primary method for visualizing predictions. The function is more flexible and allows for customizing the plot using the supervision package.

Additional features and enhancements include:

- **New Feature:** A `crop_raster` function is introduced to crop a raster image using a bounding box.
- **New Feature:** Added beta support for multiple annotation types including point, box, and polygon.
- **New Feature:** Added support for learning rates scheduling using the `torch.optim.lr_scheduler` module. The learning rate scheduler can be specified in the configuration file.
- **New Utility:** Created `utilities.download_ArcGIS_REST` function to download tiles from the ArcGIS REST API (e.g., NAIP imagery).

- **Enhancement:** The training module better matches torchvision negative anchors format for empty frames.

- **Deprecation:** `shapefile_to_annotations` in `deepforest/utilities.py` is deprecated in favor of the more general `read_file` method.
- **Deprecation:** `predict` in `deepforest/main.py`. The `return_plot` argument is deprecated and will be removed in version 2.0. Use `visualize.plot_results` instead.
- **Deprecation:** `predict_tile` in `deepforest/main.py`. Deprecated arguments `return_plot`, `color`, and `thickness` will be removed in version 2.0.
- **Deprecation:** `crop_function` in `deepforest/preprocess.py`. The `base_dir` argument is deprecated and will be removed in version 2.0. Use `save_dir` instead.
- **Deprecation:** The `deepforest.visualize.IoU_Callback` for better alignment with the PyTorch Lightning API (see [issue](https://github.com/Lightning-AI/pytorch-lightning/issues/19101)).
- **Deprecation:** `deepforest.main.use_release` and `deepforest.main.use_bird_release` are deprecated in favor of the new model loading framework, for example using `deepforest.main.load_model("weecology/deepforest-bird")`.

## Version 1.3.3 (Date: Mar 12, 2024)

- **Enhancement:** `split_raster` now allows `annotations_file` to be `None`, enabling flexibility during data preprocessing.

## Version 1.3.0 (Date: Dec 3, 2023)

- **Deprecation:** Removed `IoU_Callback` for better alignment with the PyTorch Lightning API (see [issue](https://github.com/Lightning-AI/pytorch-lightning/issues/19101)).
- **Refactor:** Evaluation code now leverages the PyTorch Lightning evaluation loop for result calculation during training.
- **Refactor:** Simplified `image_callbacks` by using predictions directly. No need to specify the root directory or CSV file, as the evaluation file is assumed.

## Version 1.1.3 (Date: Nov 9, 2021)

- **Enhancement:** Added box coordinates to the evaluation results frame for better result tracking.

## Version 1.1.2 (Date: Sep 30, 2021)

- **Bug Fix:** Fixed incorrect precision calculation in `class_recall.csv` when multiple classes were present.

## Version 1.1.1 (Date: Sep 14, 2021)

- **Update:** `project_boxes` now includes output options for both `predict_tile` and `predict_image`.
- **New Feature:** Introduced `annotations_to_shapefile`, which reverses `shapefile_to_annotations` functionality.  
  Thanks to @sdtaylor for this contribution.

## Version 1.1.0 (Date: Aug 5, 2021)

1. **Enhancement:** Empty frames are now allowed by passing annotations with 0's for all coordinates. Example format:
   ```text
   image_path, 0, 0, 0, 0, "Tree"
   ```

2. **New Feature:** Introduced `check_release` to reduce GitHub rate limit issues. When using `use_release()`, the local model will be used if `check_release = False`.

## Version 1.0.9 (Date: Jul 14, 2021)

- **Enhancement:** Improved default dtype for Windows users, thanks to @ElliotSalisbury for the contribution.

## Version 1.0.0 (Date: Jun 7, 2021)

- **Major Update:** Transitioned from TensorFlow to a PyTorch backend, enhancing performance and flexibility.

## Version 0.1.34 (Date: )

- **Optimization:** Profiled dataset and evaluation code, significantly improving evaluation performance.

## Version 0.1.30 (Date: )

- **Bug Fix:** Resolved issues to allow learning rate monitoring and decay based on `val_classification_loss`.