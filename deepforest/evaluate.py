"""
Evaluation module
"""
import pandas as pd
import geopandas as gpd
from rasterio.plot import show 
import shapely
from matplotlib import pyplot
import numpy as np

from deepforest import IoU
from deepforest.utilities import check_file
from deepforest import visualize

def evaluate_image(predictions, ground_df, show_plot, root_dir, savedir):
    """
    df: a pandas dataframe with columns name, xmin, xmax, ymin, ymax, label. The 'name' column should be the path relative to the location of the file.
    show: Whether to show boxes as they are plotted
    summarize: Whether to group statistics by plot and overall score
    image_coordinates: Whether the current boxes are in coordinate system of the image, e.g. origin (0,0) upper left.
    root_dir: Where to search for image names in df
    """
        
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError("More than one plot passed to image crown: {}".format(plot_name))
    else:
        plot_name = plot_names[0]
    
    predictions['geometry'] = predictions.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    predictions = gpd.GeoDataFrame(predictions, geometry='geometry')
    
    ground_df['geometry'] = ground_df.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    ground_df = gpd.GeoDataFrame(ground_df, geometry='geometry')        
                
    if savedir:
        visualize.plot_prediction_dataframe(df=predictions, ground_truth=ground_df, root_dir=root_dir, savedir=savedir)        
    else:
        if show_plot:
            visualize.plot_prediction_dataframe(df=predictions, ground_truth=ground_df, root_dir=root_dir, savedir=savedir)        
        
    #match  
    result = IoU.compute_IoU(ground_df, predictions)
    
    return result    

def evaluate(predictions, ground_df, root_dir, show_plot=True, iou_threshold=0.4, savedir=None):
    """Image annotated crown evaluation routine
    submission can be submitted as a .shp, existing pandas dataframe or .csv path
    
    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        root_dir: location of files in the dataframe 'name' column.
        show_plot: Whether to show boxes as they are plotted
    Returns:
        results: a dataframe of match bounding boxes
        recall: proportion of true positives
        precision: proportion of predictions that are true positive
    """
    
    check_file(ground_df)
    check_file(predictions)
    
    #Run evaluation on all plots
    results = [ ]
    recalls = []
    precisions = []
    for image_path, group in predictions.groupby("image_path"):
        plot_ground_truth = ground_df[ground_df["image_path"] == image_path].reset_index()
        result = evaluate_image(predictions=group, ground_df=plot_ground_truth, show_plot=show_plot, root_dir=root_dir, savedir=savedir)
        result["image_path"] = image_path        
        result["match"] = result.IoU > iou_threshold
        true_positive = sum(result["match"] == True)
        recall = true_positive / result.shape[0]
        precision = true_positive / group.shape[0]
        
        recalls.append(recall)
        precisions.append(precision)
        results.append(result)
    
    if len(results)==0:
        print("No predictions made, setting precision and recall to 0")
        recall = 0
        precision = 0
    else:
        results = pd.concat(results)
        precision = np.mean(precisions)
        recall = np.mean(recalls)

    return {"results":results,"precision":precision, "recall":recall}