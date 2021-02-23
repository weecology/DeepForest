"""
Evaluation module
"""
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show 
import shapely
from matplotlib import pyplot

from deepforest import IoU
from deepforest.utilities import check_file
from deepforest.utilities import project_boxes

def evaluate_image(predictions, ground_df, project, score_threshold, show_plot, root_dir):
    """
    df: a pandas dataframe with columns name, xmin, xmax, ymin, ymax, label. The 'name' column should be the path relative to the location of the file.
    show: Whether to show boxes as they are plotted
    summarize: Whether to group statistics by plot and overall score
    score_threshold: minimum probability to be included in predictions
    image_coordinates: Whether the current boxes are in coordinate system of the image, e.g. origin (0,0) upper left.
    project: Logical. Whether to project predictions that are in image coordinates (0,0 origin) into the geographic coordinates of the ground truth image. The CRS is take from the image file using rasterio.crs
    root_dir: Where to search for image names in df
    """
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError("More than one plot passed to image crown: {}".format(plot_name))
    else:
        plot_name = plot_names[0]
    
    if project:
        predictions = project_boxes(predictions, root_dir, transform = project)
    else:
        predictions['geometry'] = predictions.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
        predictions = gpd.GeoDataFrame(predictions, geometry='geometry')
        
        ground_df['geometry'] = ground_df.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
        ground_df = gpd.GeoDataFrame(ground_df, geometry='geometry')        
                
    if show_plot:
        rgb_path = "{}/{}".format(root_dir,plot_name)
        rgb_src = rasterio.open(rgb_path)        
        fig, ax = pyplot.subplots(figsize=(6, 6))
        show(rgb_src, ax = ax)
        ground_df.geometry.boundary.plot(color="red", ax = ax)
        predictions.geometry.boundary.plot(ax=ax,color="blue")
        
    #match  
    result = IoU.compute_IoU(ground_df, predictions)
    
    return result    

def evaluate(predictions, ground_df, root_dir, project=False, show_plot=True, iou_threshold=0.4, score_threshold=0):
    """Image annotated crown evaluation routine
    submission can be submitted as a .shp, existing pandas dataframe or .csv path
    
    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        root_dir: location of files in the dataframe 'name' column.
        score_threshold: minimum probability to be included in predictions
        show_plot: Whether to show boxes as they are plotted
        project: Logical. Whether to project predictions that are in image coordinates (0,0 origin) into the geographic coordinates of the ground truth image. The CRS is take from the image file using rasterio.crs
    Returns:
        recall: proportion of true positives
        precision: proportion of predictions that are true positive
    """
    
    check_file(ground_df)
    check_file(predictions)
    
    #Run evaluation on all plots
    results = [ ]
    for image_path, group in predictions.groupby("image_path"):
        plot_ground_truth = ground_df[ground_df["image_path"] == image_path]
        result = evaluate_image(predictions=group, ground_df=plot_ground_truth, project=project, show_plot=show_plot, score_threshold=score_threshold, root_dir=root_dir)
        results.append(result)
    
    if len(results)==0:
        print("No predictions made, setting precision and recall to 0")
        recall = 0
        precision = 0
    else:
        results = pd.concat(results)
        
        results["match"] = results.score > iou_threshold
        true_positive = sum(results["match"] == True)
        recall = true_positive / results.shape[0]
        precision = true_positive / predictions.shape[0]
    
    return recall, precision