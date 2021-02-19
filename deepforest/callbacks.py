"""
 A deepforest callback 
 
 Callbacks must have the following methods on_epoch_begin, on_epoch_end, on_fit_end, on_fit_begin methods and inject model and epoch kwargs.
"""

from deepforest import evaluate
from deepforest import predict
import pandas as pd        

from pytorch_lightning import Callback

class evaluate_callback(Callback):
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
    
    def __init__(self, csv_file, root_dir, iou_threshold=0.5, score_threshold=0, project=False, savedir=None, experiment=None, n=1):
        self.csv_file = csv_file
        self.experiment = experiment
        self.savedir = savedir
        self.project = project
        self.root_dir = root_dir
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.n = n
    
    def log_predictions(self, pl_module):
        pl_module.backbone.eval()
        predictions = predict.predict_file(pl_module.backbone, self.csv_file, self.root_dir, savedir=self.savedir)
        ground_df = pd.read_csv(self.csv_file)
        
        results = evaluate.evaluate(
            predictions=predictions,
            ground_df=ground_df,
            root_dir=self.root_dir,
            project=self.project,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            show_plot=False)
        
        if self.experiment:
            self.experiment.log_metric("Precision",results[0])
            self.experiment.log_metric("Recall",results[1])
        else:
            print("Validation precision at epoch {}: {}".format(pl_module.current_epoch, results[0]))
            print("Validation precision at epoch {}: {}".format(pl_module.current_epoch, results[1]))
     
    def on_init_end(self, trainer, pl_module):
        print('Running with comet validation callback')
        
    def on_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % self.n == 0:
            self.log_predictions(pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        self.log_predictions(pl_module)

        
        
        
    