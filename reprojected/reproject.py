from osgeo import gdal
import os
import numpy as np
from deepforest import main
from deepforest import get_data

# I have used https://github.com/weecology/NeonTreeEvaluation/tree/master/evaluation/RGB for testing and also a raster file is attached which is currently used in code


def resample_image_in_place(image_path, new_res, resample_image):
    args = gdal.WarpOptions(
        xRes=new_res,
        yRes=new_res
    )
    gdal.Warp(resample_image, image_path, options=args)


model = main.deepforest()
model.use_release()

# Normal resolution prediction
image_path = 'ABBY_025_2019.tif'
boxes = model.predict_image(path=image_path, return_plot=False)
df = boxes.head()
# df['image_path'] = image_path
df.to_csv('file_normal.csv')

# Evaluation of normal
csv_file = 'file_normal.csv'
root_dir = os.path.dirname(csv_file)
results = model.evaluate(csv_file, root_dir, iou_threshold=0.4, savedir=None)
print(str(results["box_recall"]) + " is evaluation of normal data")


# 2px Sampling
# Resampling Image with 2px
resample_image_in_place(image_path, 2, 'temp_image2px.tif')


# 2px resoluton prediction
image_path2px = 'temp_image2px.tif'
boxes = model.predict_image(path=image_path2px, return_plot=False)
df = boxes.head()
df['image_path'] = image_path2px
df.to_csv('file_2px.csv')

# Evaluation of 2px
csv_file = 'file_2px.csv'
root_dir = os.path.dirname(csv_file)
results = model.evaluate(csv_file, root_dir, iou_threshold=0.4, savedir=None)
print(str(results["box_recall"]) + " is evaluation of projected data at 2px")


# 5px Sampling
# Resampling Image with 5px
resample_image_in_place(image_path, 5, 'temp_image5px.tif')


# 5px resoluton prediction
image_path5px = 'temp_image5px.tif'
boxes = model.predict_image(path=image_path2px, return_plot=False)
df = boxes.head()
df['image_path'] = image_path2px
df.to_csv('file_5px.csv')

# Evaluation of 5px
csv_file = 'file_5px.csv'
root_dir = os.path.dirname(csv_file)
results = model.evaluate(csv_file, root_dir, iou_threshold=0.4, savedir=None)
print(str(results["box_recall"]) + " is evaluation of projected data at 5px")

# I tried to resample the image to higher values too but results["box_recall"] value doesn't get changed and it gets stucked at 0.2 for the current raster file
