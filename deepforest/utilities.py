#utility functions for demo
import os
import yaml
import sys
import json
import pandas as pd
import numpy as np
import urllib
import xmltodict
import csv
import warnings
import tensorflow as tf

from tqdm import tqdm
from keras_retinanet import models
from keras.utils import multi_gpu_model
from deepforest import _ROOT


def label_to_name(class_dict, label):
    """ Map label to name.
        """
    name = class_dict[label]
    return name


def read_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(config_path, e))

    return config


def read_model(model_path, config):
    """
        Read keras retinanet model from keras.model.save()
        """
    #Suppress user warning, module does not need to be compiled for prediction
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = models.load_model(model_path, backbone_name='resnet50')

    return model


#Download progress bar
class DownloadProgressBar(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def use_release(save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="NEON"):
    '''Check the existance of, or download the latest model release from github
        
        Args:
                save_dir (str): Directory to save filepath, default to "data" in deepforest repo
                prebuilt_model: Currently only accepts "NEON", but could be expanded to include other prebuilt models. The local model will be called {prebuilt_model}.h5 on disk.
        
        Returns:
                release_tag, output_path (str): path to downloaded model 
        '''

    #Find latest github tag release from the DeepLidar repo
    _json = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                'https://api.github.com/repos/Weecology/DeepForest/releases/latest',
                headers={'Accept': 'application/vnd.github.v3+json'},
            )).read())
    asset = _json['assets'][0]
    url = asset['browser_download_url']

    #Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".h5")

    #Check the release tagged locally
    try:
        release_txt = pd.read_csv(save_dir + "current_release.csv")
    except:
        release_txt = pd.DataFrame({"current_release": [None]})

    #Download the current release it doesn't exist
    if not release_txt.current_release[0] == _json["html_url"]:

        print("Downloading model from DeepForest release {}, see {} for details".format(
            _json["tag_name"], _json["html_url"]))

        with DownloadProgressBar(unit='B',
                                 unit_scale=True,
                                 miniters=1,
                                 desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        print("Model was downloaded and saved to {}".format(output_path))

        #record the release tag locally
        release_txt = pd.DataFrame({"current_release": [_json["html_url"]]})
        release_txt.to_csv(save_dir + "current_release.csv")
    else:
        print(
            "Model from DeepForest release {} was already downloaded. Loading model from file."
            .format(_json["html_url"]))

    return _json["html_url"], output_path


def xml_to_annotations(xml_path):
    """Load annotations from xml format (e.g. RectLabel editor) and convert them into retinanet annotations format. 
        
        Args:
                xml_path (str): Path to the annotations xml, formatted by RectLabel
        
        Returns:
                Annotations (pandas dataframe): in the format -> path/to/image.png,x1,y1,x2,y2,class_name
        """
    #parse
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    #grab xml objects
    try:
        tile_xml = doc["annotation"]["object"]
    except Exception as e:
        raise Exception("error {} for path {} with doc annotation{}".format(
            e, xml_path, doc["annotation"]))

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    label = []

    if type(tile_xml) == list:
        treeID = np.arange(len(tile_xml))

        #Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['name'])
    else:
        #One tree
        treeID = 0

        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['name'])

    rgb_name = os.path.basename(doc["annotation"]["filename"])

    #set dtypes
    xmin = [int(x) for x in xmin]
    xmax = [int(x) for x in xmax]
    ymin = [int(x) for x in ymin]
    ymax = [int(x) for x in ymax]

    annotations = pd.DataFrame({
        "image_path": rgb_name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "label": label
    })
    return (annotations)


def create_classes(annotations_file):
    """Create a class list in the format accepted by keras retinanet
        
        Args:
                annotations_file: an annotation csv in the retinanet format path/to/image.png,x1,y1,x2,y2,class_name
        
        Returns:
                path to classes file
        """
    annotations = pd.read_csv(
        annotations_file, names=["image_path", "xmin", "ymin", "xmax", "ymax", "label"])

    #get dir to place along file annotations
    dirname = os.path.split(annotations_file)[0]
    classes_path = os.path.join(dirname, "classes.csv")

    #get unique labels
    labels = annotations.label.dropna().unique()
    n_classes = labels.shape[0]
    print("There are {} unique labels: {} ".format(n_classes, list(labels)))

    #write label
    with open(classes_path,'w', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for index, label in enumerate(labels):
            writer.writerow([label, index])

    return classes_path


def number_of_images(annotations_file):
    """How many images in the annotations file?
        
        Args:
                annotations_file (str):
        
        Returns:
                n (int): Number of images
        """

    df = pd.read_csv(annotations_file,
                     index_col=False,
                     names=["image_path", "xmin", "ymin", "xmax", "ymax"])
    n = len(df.image_path.unique())
    return n


def format_args(annotations_file, classes_file, config, images_per_epoch=None):
    """Format config file to match argparse list for retinainet
        
        Args:
                annotations_file: a path to a csv  dataframe of annotations to get number of images, no header
                config (dict): a dictionary object to convert into a list for argparse
                images_per_epoch (int): Override default steps per epoch (n images/batch size) by manually setting a number of images
        
        Returns:
                arg_list (list): a list structure that mimics argparse input arguments for retinanet
        """
    #Format args. Retinanet uses argparse, so they need to be passed as a list
    args = {}

    #remember that .yml reads None as a str
    if not config["weights"] == 'None':
        args["--weights"] = config["weights"]

    args["--backbone"] = config["backbone"]
    args["--image-min-side"] = config["image-min-side"]
    args["--multi-gpu"] = config["multi-gpu"]
    args["--epochs"] = config["epochs"]
    if images_per_epoch:
        args["--steps"] = round(images_per_epoch / int(config["batch_size"]))
    else:
        args["--steps"] = round(
            int(number_of_images(annotations_file)) / int(config["batch_size"]))

    args["--batch-size"] = config["batch_size"]
    args["--tensorboard-dir"] = None
    args["--workers"] = config["workers"]
    args["--max-queue-size"] = config["max_queue_size"]
    args["--freeze-layers"] = config["freeze_layers"]
    args["--score-threshold"] = config["score_threshold"]

    if config["save_path"]:
        args["--save-path"] = config["save_path"]

    if config["snapshot_path"]:
        args["--snapshot-path"] = config["snapshot_path"]

    arg_list = [[k, v] for k, v in args.items()]
    arg_list = [val for sublist in arg_list for val in sublist]

    #boolean arguments
    if config["save-snapshot"] is False:
        print("Disabling snapshot saving")
        arg_list = arg_list + ["--no-snapshots"]

    if config["freeze_resnet"] is True:
        arg_list = arg_list + ["--freeze-backbone"]

    if config["random_transform"] is True:
        print("Turning on random transform generator")
        arg_list = arg_list + ["--random-transform"]

    if config["multi-gpu"] > 1:
        arg_list = arg_list + ["--multi-gpu-force"]

    if config["multiprocessing"]:
        arg_list = arg_list + ["--multiprocessing"]

    #positional arguments first
    arg_list = arg_list + ["csv", annotations_file, classes_file]

    if not config["validation_annotations"] == "None":
        arg_list = arg_list + ["--val-annotations", config["validation_annotations"]]

    #All need to be str classes to mimic sys.arg
    arg_list = [str(x) for x in arg_list]

    return arg_list
