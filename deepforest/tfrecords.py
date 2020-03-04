"""
Tfrecord module
Tfrecords creation and reader for improved performance across multi-gpu
There were a tradeoffs made in this repo. It would be natural to save the generated prepreprocessed image to tfrecord from the generator. This results in enormous (100x) files. 
The compromise was to read the original image from file using tensorflow's data pipeline. The opencv resize billinear method is marginally different then the tensorflow method, so we can't literally assert they are the same array. 
"""
import tensorflow as tf
import os
import csv
import numpy as np

from math import ceil
import keras
import pandas as pd

from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet import models
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet import losses

import matplotlib.pyplot as plt
import psutil
import gc


def create_tf_example(image, regression_target, class_target, fname, original_image):

    #Class labels can be stored as int list
    class_target = class_target.astype(int)

    #Save image information and metadata so that the tensors can be reshaped at runtime

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image/target_height':
                tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
            'image/target_width':
                tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
            'image/height':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[original_image.shape[0]])),
            'image/width':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[original_image.shape[1]])),
            'image/filename':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[fname.encode('utf-8')])),
            'image/object/regression_target':
                tf.train.Feature(float_list=tf.train.FloatList(
                    value=regression_target.flatten())),
            'image/object/class_target':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=class_target.flatten())),
            'image/object/n_anchors':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[regression_target.shape[0]]))
        }))

    # Serialize to string and write to file
    return example


def create_tfrecords(annotations_file,
                     class_file,
                     backbone_model="resnet50",
                     image_min_side=800,
                     size=1,
                     savedir="./"):
    """
    Args:
        annotations_file: path to 6 column data in form image_path, xmin, ymin, xmax, ymax, label
        backbone_model: A keras retinanet backbone
        image_min_side: resized image object minimum size
        size: Number of images per tfrecord
        savedir: dir path to save tfrecords files
    
    Returns:
        written_files: A list of path names of written tfrecords
    """
    memory_used = []

    #Image preprocess function
    backbone = models.backbone(backbone_model)

    #filebase name
    image_basename = os.path.splitext(os.path.basename(annotations_file))[0]

    ## Syntax checks
    ##Check annotations file only JPEG, PNG, GIF, or BMP are allowed.
    #df = pd.read_csv(annotations_file, names=["image_path","xmin","ymin","xmax","ymax","label"])
    #df['FileType'] = df.image_path.str.split('.').str[-1].str.lower()
    #bad_files = df[~df['FileType'].isin(["jpeg","jpg","png","gif","bmp"])]

    #if not bad_files.empty:
    #raise ValueError("Check annotations file, only JPEG, PNG, GIF, or BMP are allowed, {} incorrect files found /n {}: ".format(bad_files.shape[0],bad_files.head()))

    #Check dtypes, cannot use pandas, or will coerce in the presence of NAs
    with open(annotations_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        row = next(reader)
        if row[1].count(".") > 0:
            raise ValueError(
                "Annotation files should be headerless with integer box, {} is not a int".
                format(row[1]))

    #Create generator - because of how retinanet yields data, this should always be 1. Shape problems in the future?
    train_generator = CSVGenerator(annotations_file,
                                   class_file,
                                   batch_size=1,
                                   image_min_side=image_min_side,
                                   preprocess_image=backbone.preprocess_image)

    #chunk size
    indices = np.arange(train_generator.size())
    chunks = [
        indices[i * size:(i * size) + size] for i in range(ceil(len(indices) / size))
    ]

    written_files = []
    for chunk in chunks:
        #Create tfrecord dataset and save it for output
        fname = savedir + "{}_{}.tfrecord".format(image_basename, chunk[0])
        written_files.append(fname)
        writer = tf.io.TFRecordWriter(fname)
        images = []
        regression_targets = []
        class_targets = []
        filename = []
        original_image = []
        for i in chunk:

            #Original image
            original_image.append(train_generator.load_image(i))

            batch = train_generator.__getitem__(i),

            #split into images and tar  gets
            inputs, targets = batch[0]

            #grab image, asssume batch size of 1, squeeze
            images.append(inputs[0, ...])

            #Grab anchor targets
            regression_batch, labels_batch = targets

            #grab regression anchors
            #regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
            #where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
            #last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
            regression_anchors = regression_batch[0, ...]
            regression_targets.append(regression_anchors)

            #grab class labels - squeeze out batch size
            #From retinanet: labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
            #where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
            labels = labels_batch[0, ...]
            print("Label shape is: {}".format(labels.shape))
            class_targets.append(labels)

            #append filename by looking at group index
            current_index = train_generator.groups[i][0]

            #Grab filename and append to the full path
            fname = train_generator.image_names[current_index]
            fname = os.path.join(train_generator.base_dir, fname)

            filename.append(fname)

        for image, regression_target, class_target, fname, orig_image in zip(
                images, regression_targets, class_targets, filename, original_image):
            tf_example = create_tf_example(image, regression_target, class_target, fname,
                                           orig_image)
            writer.write(tf_example.SerializeToString())

        memory_used.append(psutil.virtual_memory().used / 2**30)

    plt.plot(memory_used)
    plt.title('Evolution of memory')
    plt.xlabel('iteration')
    plt.ylabel('memory used (GB)')
    plt.savefig(os.path.join(savedir, "memory.png"))

    return written_files


#Reading
def _parse_fn(example):

    #Define features
    features = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        "image/object/regression_target": tf.VarLenFeature(tf.float32),
        "image/object/class_target": tf.VarLenFeature(tf.int64),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/target_height": tf.FixedLenFeature([], tf.int64),
        "image/target_width": tf.FixedLenFeature([], tf.int64),
        "image/object/n_anchors": tf.FixedLenFeature([], tf.int64)
    }

    # Load one example and parse
    example = tf.io.parse_single_example(example, features)

    #Load image from file
    filename = tf.cast(example["image/filename"], tf.string)
    loaded_image = tf.read_file(filename)
    loaded_image = tf.image.decode_image(loaded_image, 3)

    #Reshape to known shape
    image_rows = tf.cast(example['image/height'], tf.int32)
    image_cols = tf.cast(example['image/width'], tf.int32)
    loaded_image = tf.reshape(loaded_image,
                              tf.stack([image_rows, image_cols, 3]),
                              name="cast_loaded_image")

    #needs to be float to subtract weights below
    loaded_image = tf.cast(loaded_image, tf.float32)

    #Turn loaded image from rgb into bgr and subtract imagenet means, see keras_retinanet.utils.image.preprocess_image
    red, green, blue = tf.unstack(loaded_image, axis=-1)

    #Subtract imagenet means
    blue = tf.subtract(blue, 103.939)
    green = tf.subtract(green, 116.779)
    red = tf.subtract(red, 123.68)

    #Recombine as BGR image
    loaded_image = tf.stack([blue, green, red], axis=-1)

    #Resize loaded image to desired target shape
    target_height = tf.cast(example['image/target_height'], tf.int32)
    target_width = tf.cast(example['image/target_width'], tf.int32)
    loaded_image = tf.image.resize(loaded_image, (target_height, target_width),
                                   align_corners=True)

    #Generated anchor data
    regression_target = tf.sparse_tensor_to_dense(
        example['image/object/regression_target'])
    class_target = tf.sparse_tensor_to_dense(example['image/object/class_target'])

    target_n_anchors = tf.cast(example['image/object/n_anchors'], tf.int32)
    regression_target = tf.cast(regression_target, tf.float32)
    class_target = tf.cast(class_target, tf.float32)

    regression_target = tf.reshape(regression_target, [target_n_anchors, 5],
                                   name="cast_regression")
    class_target = tf.reshape(class_target, [target_n_anchors, 2],
                              name="cast_class_label")

    return loaded_image, regression_target, class_target


def create_dataset(filepath, batch_size=1, shuffle=True):
    """
    Args:
        filepath: list of tfrecord files
        batch_size: number of images per batch
        
    Returns:
        dataset: a tensorflow dataset object for model training or prediction
    """

    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    ## Set the number of datapoints you want to load and shuffle
    if shuffle:
        dataset = dataset.shuffle(800)

    ## This dataset will go on forever
    dataset = dataset.repeat()

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ## Set the batchsize
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    #Collect a queue of data tensors
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ## Create an iterator
    iterator = dataset.make_one_shot_iterator()

    return iterator


def create_tensors(list_of_tfrecords, backbone_name="resnet50", shuffle=True):
    """
    Create a wired tensor target from a list of tfrecords
    
    Args:
        list_of_tfrecords: a list of tfrecord on disk to turn into a tfdataset
        backbone_name: keras retinanet backbone
        
    Returns:
        inputs: input tensors of images
        targets: target tensors of bounding boxes and classes
        """
    #Create tensorflow iterator
    iterator = create_dataset(list_of_tfrecords, shuffle=shuffle)
    next_element = iterator.get_next()

    #Split into inputs and targets
    inputs = next_element[0]
    targets = [next_element[1], next_element[2]]

    return inputs, targets
