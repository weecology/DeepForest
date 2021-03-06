"""
 A deepforest callback 
 Callbacks must have the following methods on_epoch_begin, on_epoch_end, on_fit_end, on_fit_begin methods and inject model and epoch kwargs.
"""

from deepforest import visualize
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob

from pytorch_lightning import Callback
from deepforest import dataset
from deepforest import utilities

import torch
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
        n: number of images to upload
        every_n_epochs: run epoch interval
    Returns:
        None: either prints validation scores or logs them to a comet experiment
        """
    
    def __init__(self, csv_file, root_dir, savedir, n=2, every_n_epochs=5):
        self.csv_file = csv_file
        self.savedir = savedir
        self.root_dir = root_dir
        self.n = n
        self.ground_truth = pd.read_csv(self.csv_file)
        self.every_n_epochs = every_n_epochs
        
    def log_images(self, pl_module):
        
        ds = dataset.TreeDataset(csv_file=self.csv_file,
                              root_dir=self.root_dir, transforms=dataset.get_transform(augment=False))
        
        if self.n > len(ds):
            self.n = len(ds)
            
        ds = torch.utils.data.Subset(ds, np.arange(0,self.n,1))
        
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=utilities.collate_fn)
        
        pl_module.model.eval()

        for batch in data_loader:
            paths, images, targets = batch
            
            if not pl_module.device.type=="cpu":
                images = [x.to(pl_module.device) for x in images]
                
            predictions = pl_module.model(images)
            
            for path, image, prediction, target in zip(paths, images, predictions,targets):
                image = image.permute(1,2,0)
                image = image.cpu()
                visualize.plot_prediction_and_targets(
                    image=image,
                    predictions=prediction,
                    targets=target,
                    image_name=path,
                    savedir=self.savedir)
                plt.close()        
        try:
            saved_plots = glob.glob("{}/*.png".format(self.savedir))
            for x in saved_plots:
                pl_module.logger.experiment.log_image(x)
        except Exception as e:
            print("Could not find logger in ligthning module, skipping upload, images were saved to {}, error was rasied {}".format(self.savedir, e))
        
    def on_epoch_end(self,trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs  == 0:
            print("Running image callback")            
            self.log_images(pl_module)