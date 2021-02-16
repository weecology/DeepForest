#entry point for deepforest model
import os
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from matplotlib import pyplot as plt
import torch
from torchvision.ops import nms 

from deepforest import utilities
from deepforest import dataset
from deepforest import get_data
from deepforest import training
from deepforest import model
from deepforest import preprocess
from deepforest import visualize

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
        
    def predict_image(self, image=None, path=None, return_plot=False,score_threshold=0.01):
        """Predict an image with a deepforest model
        Args:
            image: a numpy array of a RGB image ranged from 0-255
            path: optional path to read image from disk instead of passing image arg
            return_plot: Return image with plotted detections
            score_threshold: float [0,1] minimum probability score to return/plot.
        Returns:
            boxes: A pandas dataframe of predictions (Default)
            img: The input with predictions overlaid (Optional)
        """
        if isinstance(image,str):
            raise ValueError("Path provided instead of image. If you want to predict an image from disk, is path =")
  
        if path:
            if not isinstance(path,str):
                raise ValueError("Path expects a string path to image on disk")            
            image = io.imread(path)
        
        self.backbone.eval()        
        image = preprocess.preprocess_image(image)
        prediction = self.backbone(image)
        
        #return None for no predictions
        if len(prediction[0]["boxes"])==0:
            return None
        
        #This function on takes in a single image.
        df = visualize.format_predictions(prediction[0])
        df = df[df.scores > score_threshold]
        
        if return_plot:
            #Matplotlib likes no batch dim and channels first
            image = image.squeeze(0).permute(1,2,0)
            plot, ax = visualize.plot_predictions(image, df)
            return plot
        else:
            return df
                                                 
    def predict_file(self, csv_file, root_dir, save_dir=None):
        """Create a dataset and predict entire annotation file
        
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position. 
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line. 
        
        Args:
            csv_file: path to csv file 
            root_dir: directory of images. If none, uses "image_dir" in config
            savedir: Optional. Directory to save image plots.
        Returns:
            df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
        """
        
        self.backbone.eval()
        input_csv = pd.read_csv(csv_file)
        
        #Just predict each image once. 
        images = input_csv.image_path.unique() 
        
        if root_dir is None:
            root_dir = self.config["image_dir"]
        
        prediction_list = []
        for path in images:
            image = io.imread("{}/{}".format(root_dir,path))
            image = preprocess.preprocess_image(image)
            
            #Just predict the images, even though we have the annotations
            prediction = self.backbone(image)
            prediction = visualize.format_predictions(prediction[0])
            prediction["image_path"] = path
            prediction_list.append(prediction)
            
            if save_dir:
                image = image.squeeze(0).permute(1,2,0)                
                plot, ax = visualize.plot_predictions(image, prediction)
                annotations = input_csv[input_csv.image_path==path]
                plot = visualize.add_annotations(plot, ax, annotations)
                plot.savefig("{}/{}.png".format(save_dir,os.path.splitext(path)[0]))
            
        df = pd.concat(prediction_list,ignore_index=True)
        
        return df
    
    def predict_tile(self,
                     raster_path=None,
                     image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False,
                     soft_nms = False,
                     sigma = 0.5,
                     thresh = 0.001):
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
            soft_nms: whether to perform Gaussian Soft NMS or not, if false, default perform NMS. 
            sigma: variance of Gaussian function used in Gaussian Soft NMS
            thresh: the score thresh used to filter bboxes after soft-nms performed
        Returns:
            boxes (array): if return_plot, an image.
            Otherwise a numpy array of predicted bounding boxes, scores and labels
        """
        
        if image is not None:
            pass
        else:
            #load raster as image
            image = io.imread(raster_path)
        
        # Compute sliding window index
        windows = preprocess.compute_windows(image, patch_size, patch_overlap)
        # Save images to tempdir
        predicted_boxes = []
        
        for index, window in enumerate(tqdm(windows)):
            #crop window and predict
            crop = image[windows[index].indices()] 
            
            #crop is RGB channel order, change to BGR
            crop = crop[...,::-1]
            boxes = self.predict_image(image=crop,
                                    return_plot=False,
                                    score_threshold=self.config['score_threshold'])
            
            #transform the coordinates to original system
            xmin, ymin, xmax, ymax = windows[index].getRect()
            boxes.xmin = boxes.xmin + xmin
            boxes.xmax = boxes.xmax + xmin
            boxes.ymin = boxes.ymin + ymin
            boxes.ymax = boxes.ymax + ymin

            predicted_boxes.append(boxes)
        
        predicted_boxes = pd.concat(predicted_boxes)
        # Non-max supression for overlapping boxes among window
        if patch_overlap == 0:
            mosaic_df = predicted_boxes
        else:
            print(f"{predicted_boxes.shape[0]} predictions in overlapping windows, applying non-max supression")
            #move prediciton to tensor 
            boxes = torch.tensor(predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values, dtype = torch.float32)
            scores = torch.tensor(predicted_boxes.scores.values, dtype = torch.float32)
            labels = predicted_boxes.label.values
            
            if soft_nms == False:
                #Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).
                bbox_left_idx = nms(boxes = boxes, scores = scores, iou_threshold=iou_threshold)
            else:
                #Performs soft non-maximum suppression (soft-NMS) on the boxes.
                bbox_left_idx = utilities.soft_nms(boxes = boxes, scores = scores, sigma = sigma, thresh=thresh)
            
            bbox_left_idx = bbox_left_idx.numpy()
            new_boxes, new_labels, new_scores = boxes[bbox_left_idx].type(torch.int), labels[bbox_left_idx], scores[bbox_left_idx]
            
            #Recreate box dataframe
            image_detections = np.concatenate([
                    new_boxes,
                    np.expand_dims(new_labels, axis=1),
                    np.expand_dims(new_scores, axis=1)
                    ],axis=1)
            
            mosaic_df = pd.DataFrame(
                    image_detections,
                    columns=["xmin", "ymin", "xmax", "ymax", "label","score"])

            print(f"{mosaic_df.shape[0]} predictions kept after non-max suppression")
        
        if return_plot:
            # Draw predictions
            plot, _ = visualize.plot_predictions(image, mosaic_df)
            # Mantain consistancy with predict_image
            return plot
        else:
            return mosaic_df


    def load_dataset(self, csv_file, root_dir=None, augment=False):
        """Create a tree dataset for inference
        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position. 
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line. 
        
        Args:
            csv_file: path to csv file 
            root_dir: directory of images. If none, uses "image_dir" in config
            augment: Whether to create a training dataset, this deactivates data augmentations
        Returns:
            self.ds: a pytorch dataset
        """
        
        if root_dir is None:
            root_dir = self.config["image_dir"]
                
        self.ds = dataset.TreeDataset(csv_file=csv_file,
                              root_dir=root_dir,
                              transforms=dataset.get_transform(augment=augment))

    def train(self, callbacks=None, debug=False):
        """Train on a loaded dataset
        Args:
            debug: run a small training step for testing
            callbacks: a list of deepforest.callbacks that implements a training callback
        """
        #check is dataset has been created?
        if not self.ds:
            raise ValueError("Cannot train a model with first loading data, see deepforest.load_dataset(csv_file=<>)")
        #check is model has been created?
        if not self.backbone:
            raise ValueError("Cannot train a model with first creating a model instance, see deepforest.create_model().")
        
        self.backbone.train()
        self.model = training.run(train_ds=self.ds, model=self.backbone, config=self.config, debug=debug, callbacks=callbacks)
        
        
    def evaluate(self, csv_file, metrics, iou_threshold):
        pass

    def save(self):
        pass
    
    def load(self):
        pass
    