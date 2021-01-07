#entry point for deepforest model
import os
import torch

from deepforest import utilities
from deepforest import dataset
from deepforest import get_data
from deepforest import training
from deepforest import model
from deepforest import preprocess


class deepforest:
    """Class for training and predicting tree crowns in RGB images
    """
    def __init__(self, saved_model=None):
        # Read config file - if a config file exists in local dir use it,
        # if not use installed.
        if os.path.exists("deepforest_config.yml"):
            config_path = "deepforest_config.yml"
        else:
            try:
                config_path = get_data("deepforest_config.yml")
            except Exception as e:
                raise ValueError(
                    "No deepforest_config.yml found either in local "
                    "directory or in installed package location. {}".format(e))

        print("Reading config file: {}".format(config_path))
        self.config = utilities.read_config(config_path)

        # release version id to flag if release is being used
        self.__release_version__ = None

        if saved_model:
            utilities.load_saved_model(saved_model)

    def use_release(self):
        """Use the latest DeepForest model release from github and load model.
        Optionally download if release doesn't exist.
        Returns:
            model (object): A trained keras model
        """
        # Download latest model from github release
        release_tag, self.weights = utilities.use_release()

        # load saved model and tag release
        self.__release_version__ = release_tag
        print("Loading pre-built model: {}".format(release_tag))
    
    def create_model(self):
        """Define a deepforest retinanet architecture"""
        self.backbone = model.load_backbone()
        
    def predict_image(self, image):
        """Predict an image with a deepforest model"""
        self.backbone.eval()
        image = preprocess.preprocess_image(image)
        prediction = self.backbone(image)
        
    def predict_file(self, file):
        """Create a dataset and predict entire annotation file"""
        pass
    
    def predict_tile(self,
                     raster_path=None,
                     image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.
        Args:
            raster_path: Path to image on disk
            image (array): Numpy image array in BGR channel order
                following openCV convention
            patch_size: patch size default400,
            patch_overlap: patch overlap default 0.15,
            iou_threshold: Minimum iou overlap among predictions between
                windows to be suppressed. Defaults to 0.5.
                Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?
        Returns:
            boxes (array): if return_plot, an image.
                Otherwise a numpy array of predicted bounding boxes, scores and labels
        """
        
        pass
        

    def load_dataset(self, csv_file, root_dir=None, train=False):
        """Create a tree dataset for inference
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position. 
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line. 
        
        Args:
            csv_file: path to csv file 
            root_dir: directory of images. If none, uses "image_dir" in config
            train: Whether to create a training dataset, this deactivates data augmentations
        Returns:
            self.ds: a pytorch dataset
        """
        
        if root_dir is None:
            root_dir = self.config["image_dir"]
                
        self.ds = dataset.TreeDataset(csv_file=csv_file,
                              root_dir=root_dir,
                              transforms=dataset.get_transform(train=train))

    def train(self, debug=False):
        """Train on a loaded dataset"""
        #check is dataset has been created?
        if not self.ds:
            raise ValueError("Cannot train a model with first loading data, see deepforest.load_dataset(csv_file=<>)")
        #check is model has been created?
        if not self.backbone:
            raise ValueError("Cannot train a model with first creating a model instance, see deepforest.create_model().")
        
        self.backbone.train()
        training.run(train_ds=self.ds, model=self.backbone, config=self.config, debug=debug)
        
        
    def evaluate(csv_file):
        pass
