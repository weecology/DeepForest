"""
 A deepforest callback 
 
 Callbacks must have the following methods on_epoch_begin, on_epoch_end, on_fit_end, on_fit_begin methods and inject model and epoch kwargs.
"""

from deepforest import visualize
import pandas as pd
import numpy as np
import glob

from pytorch_lightning import Callback

class images_callback(Callback):
    """Run evaluation on a file of annotations during training
    Args:
        model: pytorch model
        csv_file: path to csv with columns, image_path, xmin, ymin, xmax, ymax, label
        epoch: integer. current epoch
        experiment: optional comet_ml experiment
        savedir: optional, directory to save predicted images
        project: whether to project image coordinates into geographic coordinations, see deepforest.evaluate
        root_dir: root directory of images to search for 'image path' values from the csv file
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate
        probability_threshold: minimum probablity for inclusion, see deepforest.evaluate
        n: run callback on every n epochs
    Returns:
        None: either prints validation scores or logs them to a comet experiment
        """
    
    def __init__(self, csv_file, root_dir, savedir, score_threshold=0, experiment=None, n=2):
        self.csv_file = csv_file
        self.experiment = experiment
        self.savedir = savedir
        self.root_dir = root_dir
        self.score_threshold = score_threshold
        self.n = n
        self.ground_truth = pd.read_csv(self.csv_file)
        
    def log_images(self, pl_module):
        ds = pl_module.load_dataset(self.csv_file, self.root_dir)
        
        #Make sure the n images is not larger than the dataset
        if self.n > len(ds):
            self.n = len(ds)
            
        for x in np.arange(self.n):
            batch = next(iter(ds))
            path, image, targets = batch
            pl_module.backbone.eval()
            predictions = pl_module.backbone(image)
            
            result = []
            for index, prediction in enumerate(predictions):
                formatted_prediction = visualize.format_predictions(prediction)
                formatted_prediction["image_path"] = path[index]
                result.append(formatted_prediction)
            df = pd.concat(result)
            
            df = df[df.scores > self.score_threshold]
            visualize.plot_prediction_dataframe(df, self.ground_truth, self.root_dir, self.savedir)
            
        if self.experiment:
            saved_plots = glob.glob("{}/*.png".format(self.savedir))
            for x in saved_plots:
                self.experiment.log_image(x)
        
    def on_epoch_end(self,trainer, pl_module):
        self.log_images(pl_module)

        
        
        
    