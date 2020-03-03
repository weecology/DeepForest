"""
Prediction module. This module consists of predict utility function for the deepforest class
"""
import numpy as np
import copy
import glob
import keras
import cv2
import pandas as pd
import tensorflow as tf

#Retinanet-viz
from keras_retinanet.utils import image as keras_retinanet_image
from keras_retinanet.utils.visualization import draw_detections


def predict_image(model,
                  image_path=None,
                  raw_image=None,
                  score_threshold=0.05,
                  max_detections=200,
                  return_plot=True,
                  classes={"0": "Tree"},
                  color=None):
    """
    Predict invidiual tree crown bounding boxes for a single image
    
    Args:
        model (object): A keras-retinanet model to predict bounding boxes, either load a model from weights, use the latest release, or train a new model from scratch.  
        image_path (str): Path to image file on disk
        raw_image (str): Numpy image array in BGR channel order following openCV convention
        score_threshold (float): Minimum probability score to be included in final boxes, ranging from 0 to 1.
        max_detections (int): Maximum number of bounding box predictions per tile
        return_plot (bool):  If true, return a image object, else return bounding boxes as a numpy array
    
    Returns:
        raw_image (array): If return_plot is TRUE, the image with the overlaid boxes is returned
        image_detections: If return_plot is FALSE, a np.array of image_boxes, image_scores, image_labels
    """
    #Check for raw_image
    if raw_image is not None:
        numpy_image = raw_image.copy()
    else:
        #Read from path
        numpy_image = cv2.imread(image_path)

    #Make sure image exists
    try:
        numpy_image.shape
    except:
        raise IOError(
            "Image file {} cannot be read, check that it exists".format(image_path))

    #Check that its 3 band
    bands = numpy_image.shape[2]
    if not bands == 3:
        raise IOError(
            "Input file {} has {} bands. DeepForest only accepts 3 band RGB rasters. If the image was cropped and saved as a .jpg, please ensure that no alpha channel was used."
            .format(path_to_raster, bands))

    image = keras_retinanet_image.preprocess_image(numpy_image)
    image, scale = keras_retinanet_image.resize_image(image)

    if keras.backend.image_data_format() == 'channels_first':
        image = image.transpose((2, 0, 1))

    # run network
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

    # correct boxes for image scale
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes = boxes[0, indices[scores_sort], :]
    image_scores = scores[scores_sort]
    image_labels = labels[0, indices[scores_sort]]
    image_detections = np.concatenate([
        image_boxes,
        np.expand_dims(image_scores, axis=1),
        np.expand_dims(image_labels, axis=1)
    ],
                                      axis=1)

    df = pd.DataFrame(image_detections,
                      columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

    #Change numberic class into string label
    df.label = df.label.astype(int)
    df.label = df.label.apply(lambda x: classes[x])

    if return_plot:
        draw_detections(numpy_image,
                        image_boxes,
                        image_scores,
                        image_labels,
                        label_to_name=None,
                        score_threshold=score_threshold,
                        color=color)
        return numpy_image
    else:
        return df


def non_max_suppression(sess,
                        boxes,
                        scores,
                        labels,
                        max_output_size=200,
                        iou_threshold=0.15):
    '''
    Provide a tensorflow session and get non-maximum suppression
    Args:
        sess: a tensorfloe
    max_output_size, iou_threshold are passed to tf.image.non_max_suppression 
    '''
    non_max_idxs = tf.image.non_max_suppression(boxes,
                                                scores,
                                                max_output_size=max_output_size,
                                                iou_threshold=iou_threshold)
    new_boxes = tf.cast(tf.gather(boxes, non_max_idxs), tf.int32)
    new_scores = tf.gather(scores, non_max_idxs)
    new_labels = tf.gather(labels, non_max_idxs)
    return sess.run([new_boxes, new_scores, new_labels])
