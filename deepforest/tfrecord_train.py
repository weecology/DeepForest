#Prepare tfrecord for retinanet reading
import tensorflow as tf

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

    #TODO allow this vary from config? Or sess?
    image = tf.reshape(image, [-1, 800, 800, 3])    
    
    return image, targets
    
def create_dataset(filepath, sess):
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
    
    #Reshape at run time
    regression_targets = tf.reshape(regression_target, [120087, 5])
    class_target = tf.reshape(regression_target, [120087, 2])
    
    #stack regression and class targets?    
    targets = tf.stack(regression_target, class_target)    

    return image, targets


if __name__ =="__main__":
    with tf.Session() as sess:
        tensors = create_dataset("/Users/ben/Documents/NeonTreeEvaluation_analysis/Weinstein_unpublished/pretraining/tfrecords/100.tfrecords", sess)

        