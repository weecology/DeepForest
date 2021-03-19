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
    Compute intersection-over-union matching among prediction and ground truth boxes for one image
    Args:
        df: a pandas dataframe with columns name, xmin, xmax, ymin, ymax, label. The 'name' column should be the path relative to the location of the file.
        show: Whether to show boxes as they are plotted
        summarize: Whether to group statistics by plot and overall score
        image_coordinates: Whether the current boxes are in coordinate system of the image, e.g. origin (0,0) upper left.
        root_dir: Where to search for image names in df
    Returns:
        result: pandas dataframe with crown ids of prediciton and ground truth and the IoU score.
    """

    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError("More than one plot passed to image crown: {}".format(plot_name))
    else:
        plot_name = plot_names[0]

    predictions['geometry'] = predictions.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    predictions = gpd.GeoDataFrame(predictions, geometry='geometry')

    ground_df['geometry'] = ground_df.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    ground_df = gpd.GeoDataFrame(ground_df, geometry='geometry')

    if savedir:
        visualize.plot_prediction_dataframe(df=predictions,
                                            ground_truth=ground_df,
                                            root_dir=root_dir,
                                            savedir=savedir)
    else:
        if show_plot:
            visualize.plot_prediction_dataframe(df=predictions,
                                                ground_truth=ground_df,
                                                root_dir=root_dir,
                                                savedir=savedir)

    # match
    result = IoU.compute_IoU(ground_df, predictions)
    
    #add the label classes
    result["predicted_label"] = result.prediction_id.apply(lambda x: predictions.label.loc[x])
    result["true_label"] = result.truth_id.apply(lambda x: ground_df.label.loc[x])

    return result


def evaluate(predictions,
             ground_df,
             root_dir,
             show_plot=True,
             iou_threshold=0.4,
             savedir=None):
    """Image annotated crown evaluation routine
    submission can be submitted as a .shp, existing pandas dataframe or .csv path

    Args:
        predictions: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        ground_df: a pandas dataframe, if supplied a root dir is needed to give the relative path of files in df.name
        root_dir: location of files in the dataframe 'name' column.
        show_plot: Whether to show boxes as they are plotted
    Returns:
        results: a dataframe of match bounding boxes
        box_recall: proportion of true positives of box position, regardless of class
        box_precision: proportion of predictions that are true positive, regardless of class
        class_recall: a pandas dataframe of class level recall and precision with class sizes
    """

    check_file(ground_df)
    check_file(predictions)

    # Run evaluation on all plots
    results = []
    box_recalls = []
    box_precisions = []
    for image_path, group in predictions.groupby("image_path"):
        plot_ground_truth = ground_df[ground_df["image_path"] == image_path].reset_index()
        result = evaluate_image(predictions=group,
                                ground_df=plot_ground_truth,
                                show_plot=show_plot,
                                root_dir=root_dir,
                                savedir=savedir)
        result["image_path"] = image_path
        result["match"] = result.IoU > iou_threshold
        true_positive = sum(result["match"])
        recall = true_positive / result.shape[0]
        precision = true_positive / group.shape[0]

        box_recalls.append(recall)
        box_precisions.append(precision)
        results.append(result)

    if len(results) == 0:
        print("No predictions made, setting precision and recall to 0")
        box_recall = 0
        box_precision = 0
    else:
        results = pd.concat(results)
        box_precision = np.mean(box_precisions)
        box_recall = np.mean(box_recalls)

    #Per class recall and precision
    class_recall_dict = {}
    class_precision_dict = {}
    class_size = {}
    
    for name, group in result.groupby("true_label"):
        class_recall_dict[name] = sum(group.true_label == group.predicted_label)/ground_df.shape[0]
        class_precision_dict[name] = sum(group.true_label == group.predicted_label)/predictions.shape[0]
        class_size[name] = group.size
    
    
    class_recall = pd.DataFrame({"label":class_recall_dict.keys(),"recall":pd.Series(class_recall_dict), "precision":pd.Series(class_precision_dict), "size":pd.Series(class_size)}).reset_index(drop=True)
            
    return {"results": results, "box_precision": box_precision, "box_recall": box_recall, "class_recall":class_recall}
