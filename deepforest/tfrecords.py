from keras_retinanet import models
import tensorflow as tf
import numpy as np
from math import ceil
from keras_retinanet.preprocessing.csv_generator import CSVGenerator

def create_tf_example(image, regression_target, class_target, filename):
    #Image data 
    height = image.shape[0]
    width = image.shape[1]
    filename = filename
    image_format = b'jpg'
    classes_text = ['Tree']
    classes = [1]
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])), 
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
        'image/encoded':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
        'image/n_anchors': tf.train.Feature(int64_list=tf.train.Int64List(value=[regression_target.shape[0]])),         
        'image/object/regression_target': tf.train.Feature(float_list=tf.train.FloatList(value=regression_target.flatten())),
        'image/object/class_target': tf.train.Feature(float_list=tf.train.FloatList(value=class_target.flatten())),
    }))
    
    # Serialize to string and write to file
    return example

def create_tfrecords(backbone_model, image_min_side, size, savedir):
    """
    Args:
        backbone:
        image_min_side:
        size:
        savedir:
    Returns:
        NULL -> side effect
    """
    #Image preprocess function
    backbone = models.backbone(backbone_model)
    
    #Create generator - because of how retinanet yields data, this should always be 1. Shape problems in the future?
    train_generator = CSVGenerator(
        annotations_file,
        class_file,
         batch_size = 1,
         image_min_side = image_min_side,
         preprocess_image = backbone.preprocess_image
         )
    
    #chunk size 
    indices = np.arange(train_generator.size())
    chunks = [
        indices[i * size:(i * size) + size]
        for i in range(ceil(len(indices) / size))
    ]
    
    for chunk in chunks:
        #Create tfrecord dataset
        writer = tf.python_io.TFRecordWriter(savedir + "{}.tfrecords".format(chunk[0]))
        images = []
        regression_targets = []
        class_targets = []
        filenames = []
        
        for i in chunk:
            batch = train_generator.__getitem__(i),
            
            #split into images and targets
            inputs, targets =  batch[0]
           
            #grab image, asssume batch size of 1
            images.append(inputs)
            
            #Grab anchor targets
            regression_batch, labels_batch = targets
        
            #grab regression anchors
            #regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
            #where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
            #last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).       
            regression_anchors = regression_batch     
            regression_targets.append(regression_anchors)
            
            #grab class labels
            #From retinanet: labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
           #where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).        
            labels = labels_batch
            class_targets.append(labels)
            
            i +=1
                
        for image, regression_target, class_target, filename in zip(images,regression_targets, class_targets, filenames):
            tf_example = create_tf_example(image, regression_target, class_target, filename)
            writer.write(tf_example.SerializeToString())        
            

#Reading
def _parse_fn(example):
    
    #Define features
    features = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),       
        "image/n_anchors": tf.io.FixedLenFeature([], tf.int64),
        "image/object/regression_target": tf.FixedLenFeature([], tf.float32),
        "image/object/class_target": tf.FixedLenFeature([], tf.float32)
                        }
    
    # Load one example and parse
    example = tf.io.parse_single_example(example, features)
    image = tf.decode_raw(example['image/encoded'], tf.uint8)
    height = tf.cast(example['image/height'], tf.int32)
    width = tf.cast(example['image/width'], tf.int32)
    n_anchors = tf.cast(example['image/n_anchors'], tf.int32)
    regression_target = tf.cast(example['image/object/regression_target'], tf.float32)
    class_target = tf.cast(example['image/object/class_target'], tf.float32)
    
    return image, regression_target, class_target

def create_dataset(filepath):
    """
    filepath: list of tfrecord files
    """
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_fn)
    
    ## This dataset will go on forever
    dataset = dataset.repeat()
    
    ## Set the number of datapoints you want to load and shuffle 
    #dataset = dataset.shuffle(1000)
    
    ## Set the batchsize
    dataset = dataset.batch(1)
    
    ## Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    ## Create your tf representation of the iterator
    image, regression_target, class_target = iterator.get_next()
    
    #TODO allow this vary from config? Or read during sess?
    image = tf.reshape(image, [-1, 800, 800, 3])    
    
    regression_target = tf.reshape(regression_target, [-1, 120087, 5])
    class_target = tf.reshape(class_target, [-1, 120087, 2])
    
    #stack regression and class targets?    
    return image, [regression_target, class_target]

def train(path_to_tfrecord, steps_per_epoch=None):
    
    #Create tensorflow iterator
    inputs, targets = create_dataset(path_to_tfrecord)
    
    #Create model
    model = backbone.retinanet(num_classes=1)
    training_model = model
    
    # make prediction model
    prediction_model = retinanet_bbox(model=model)
        
    #Compile your model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        target_tensors = targets,
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    
    #Train model
    if steps_per_epoch is None:
        raise ValueError("Unknown steps for a tfrecord")
    training_model.fit(steps_per_epoch=steps_per_epoch)
    
    return training_model, prediction_model    