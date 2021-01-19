# FAQ

## DeepForest claims my image is not a 3 band raster!

Potential Solution: Check for an alpha channel when saving as a jpg.

```
OSError: cannot write mode RGBA as JPEG
```

or

```
IOError: DeepForest only accepts 3 band RGB rasters.
```

If you are manually cropping an image and saving a JPG, be careful not to save the alpha channel. For example, on OSX, the preview tool will save a 4 channel image (RGBA) instead of a three channel image (RGB) by default. When saving a crop, toggle alpha channel off.

If you need to select a set of rasters, the easiest thing to do is read in the data with a package like rasterio (not installed) and select the needed bands

```
src = rio.open("/orange/ewhite/everglades/Palmyra/example.tif")
numpy_image = src.read()
numpy_image = np.moveaxis(numpy_image,0,2)
numpy_image = numpy_image[:,:,:3]
```

Note that by default rasterio reads in the order, (channels, height, width), whereas DeepForest requires (height, width, channels). Using np.moveaxis to move the channels from the first to last positoon.

## The plotted image looks wrong, the trees are blue!
Solution: You are plotting the BlueGreenRed (BGR) image channel order. You want the RGB image channel ordering.

Unfortunately, python has two standard libraries for image processing and visualization, matplotlib and opencv, that do not have the same default channel order. Matplotlib reads images into RedGreenBlue order, whereas Opencv reads in BlueGreenRed order. The machine learning module and the default backbone weights assume the image is BGR. Therefore there is some uncomfortable moments of visualizing data and not anticipating the channel order.

![](../www/bgr_rgb.png)

The top can be converted into the bottom by reversing channel order.

```
bgr = rgb[...,::-1]
```

** Raw images that are fed into  should always be bgr order ** The model will perform slightly more poorly on rgb images, as shown above. If images are on path

```
deepforest.predict_image(image_path="path to image")
```

Deepforest will automatically read in the image as bgr, the user does not need to anything.

In general DeepForest has adopted the philosophy that functions which interact with the images should read in arrays in BGR or from file and return images in BGR foramt.

## How do I convert projected annotations to image coordinates?

I've made a small utility to convert projected data, such as utm coordinates, into the image coordinates.

[UTM_to_Image Gist](https://gist.github.com/bw4sz/e2fff9c9df0ae26bd2bfa8953ec4a24c)

DeepForest makes predictions in the image coordinate system with the top left of the image as 0,0. To convert these coordinates into the input prediction projection we need to know the bounds of the image and the resolution. Please note that this makes sense over small geographic areas in which we don't need to consider the curvature of the earth. I've written two utility functions that are useful. One for going from shapefiles to annotations. Another for going from predictions to projected boxes. Note that these require the geopandas library which is not installed in DeepForest.

## Linux Python 3.5 doesn't work.
Solution: Updated to Python >3.6

There is one known [bug](https://github.com/weecology/DeepForest/issues/64) in Ubuntu python 3.5 on use_release.
It is suggested to update to a more recent python version.

## DeepForest cannot read my image. I get "Cannot identify image".

```
>>> Image.open("/Users/ben/Downloads/NAIP/East_ben.tif")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/ben/Documents/DeepForest_French_Guiana/DeepForest/lib/python3.7/site-packages/PIL/Image.py", line 2818, in open
    raise IOError("cannot identify image file %r" % (filename if filename else fp))
OSError: cannot identify image file '/Users/ben/Downloads/NAIP/East_ben.tif'
```

The python image library requires three band unsigned 8 bit images. Sometimes geotiff files have no-data values greater than 255 or other file types.

Solution:

To ensure unsigned 8bit rasters, we can use any number of programs. For example in R using the raster package

```
library(raster)
r<-stack("/Users/ben/Downloads/NAIP/East_clip2.tif")
writeRaster(x=r,datatype="INT1U",filename="/Users/ben/Downloads/NAIP/East_ben.tiff")
```

## I'm seeing alot of deprecation warnings.

Occasionally users report that conda enforces incorrect versions on install from source.

```
 ERROR: keras-retinanet 0.5.1 has requirement keras-resnet==0.1.0, but you'll have keras-resnet 0.2.0 which is incompatible.
```
We have yet to find an example where this prevents DeepForest from operating successfully. From our perspective, this error can be ignored. If not, please open an [issue](https://github.com/weecology/DeepForest/issues) documenting your conda version and operating system.

### Tensorflow deprectation warnings

```
>>> from deepforest import deepforest
/anaconda3/envs/DeepForest/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
```

These warnings are upstream of DeepForest and can be ignored. They are there for developers and future users.
