"""
Utilities model
"""
import json
import os
import pandas as pd
import urllib
from tqdm import tqdm
import yaml

from deepforest import _ROOT

def read_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

    return config


class DownloadProgressBar(tqdm):
    """Download progress bar class."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def use_release(save_dir=os.path.join(_ROOT, "data/"), prebuilt_model="NEON"):
    """Check the existence of, or download the latest model release from
    github.
    Args:
        save_dir: Directory to save filepath,
            default to "data" in deepforest repo
        prebuilt_model: Currently only accepts "NEON", but could be expanded to
            include other prebuilt models. The local model will be
            called {prebuilt_model}.h5 on disk.
    Returns:
        release_tag, output_path (str): path to downloaded model
    """

    # Find latest github tag release from the DeepLidar repo
    _json = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                'https://api.github.com/repos/Weecology/DeepForest/releases/latest',
                headers={'Accept': 'application/vnd.github.v3+json'},
            )).read())
    asset = _json['assets'][0]
    url = asset['browser_download_url']

    # Naming based on pre-built model
    output_path = os.path.join(save_dir, prebuilt_model + ".h5")

    # Check the release tagged locally
    try:
        release_txt = pd.read_csv(save_dir + "current_release.csv")
    except:
        release_txt = pd.DataFrame({"current_release": [None]})

    # Download the current release it doesn't exist
    if not release_txt.current_release[0] == _json["html_url"]:

        print("Downloading model from DeepForest release {}, see {} for details".format(
            _json["tag_name"], _json["html_url"]))

        with DownloadProgressBar(unit='B',
                                 unit_scale=True,
                                 miniters=1,
                                 desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

        print("Model was downloaded and saved to {}".format(output_path))

        # record the release tag locally
        release_txt = pd.DataFrame({"current_release": [_json["html_url"]]})
        release_txt.to_csv(save_dir + "current_release.csv")
    else:
        print("Model from DeepForest release {} was already downloaded. "
              "Loading model from file.".format(_json["html_url"]))

    return _json["html_url"], output_path

def xml_to_annotations(xml_path):
    """Load annotations from xml format (e.g. RectLabel editor) and convert
    them into retinanet annotations format.
    Args:
        xml_path (str): Path to the annotations xml, formatted by RectLabel
    Returns:
        Annotations (pandas dataframe): in the
            format -> path/to/image.png,x1,y1,x2,y2,class_name
    """
    # parse
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    # grab xml objects
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

        # Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['name'])
    else:
        # One tree
        treeID = 0

        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['name'])

    rgb_name = os.path.basename(doc["annotation"]["filename"])

    # set dtypes, check for floats and round
    xmin = [round_with_floats(x) for x in xmin]
    xmax = [round_with_floats(x) for x in xmax]
    ymin = [round_with_floats(x) for x in ymin]
    ymax = [round_with_floats(x) for x in ymax]

    annotations = pd.DataFrame({
        "image_path": rgb_name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "label": label
    })
    return (annotations)


def round_with_floats(x):
    """Check if string x is float or int, return int, rounded if needed."""

    try:
        result = int(x)
    except:
        warnings.warn(
            "Annotations file contained non-integer coordinates. "
            "These coordinates were rounded to nearest int. "
            "All coordinates must correspond to pixels in the image coordinate system. "
            "If you are attempting to use projected data, "
            "first convert it into image coordinates see FAQ for suggestions.")
        result = int(np.round(float(x)))

    return result

def load_saved_model():
    pass
