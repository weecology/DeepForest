=============================
DeepForest Change Log
=============================

# 1.1.1

Update to project_boxes to include an output for predict_tile and predict_image, the function was split into two. annotations_to_shapefile reverses shapefile_to_annotations. Thanks to @sdtaylor for this contribution.

# 1.1.0

1.
Empty frames are now allowed by passing annotations with 0's for all coords. A single row for each blank image.

image_path, 0,0,0,0, "Tree"

2.
A check_release function was implemented to reduce github rate limit issues. on use_release(), the local model will be used if check_release = False

# 1.0.9

Minor update to improve windows users default dtype, thanks to @ElliotSalisbury

# 1.0.0

Major update to replace tensorflow backend with pytorch. 

#0.1.30
Bug fixes to allow learning rate monitoring and decay on val_classification_loss

#0.1.34
Profiled the dataset and evaluation code for performance. Evaluation should be much faster now.
