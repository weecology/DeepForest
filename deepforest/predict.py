#Prediction utilities
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import torch
from torchvision.ops import nms 

from deepforest import preprocess
from deepforest import visualize
from skimage import io

def predict_image(model, image, return_plot, device, iou_threshold=0.1):
    """Predict an image with a deepforest model
    
    Args:
        image: a numpy array of a RGB image ranged from 0-255
        path: optional path to read image from disk instead of passing image arg
        return_plot: Return image with plotted detections
        device: pytorch device of 'cuda' or 'cpu' for gpu prediction. Set internally.
    Returns:
        boxes: A pandas dataframe of predictions (Default)
        img: The input with predictions overlaid (Optional)
    """        
    image = preprocess.preprocess_image(image)
    image = image.to(device)
    prediction = model(image)
        
    if not device.type=="cpu":
        prediction = prediction[0].detach().cpu().numpy()
        
    #return None for no predictions
    if len(prediction[0]["boxes"])==0:
        return None
    
    #This function on takes in a single image.
    df = visualize.format_boxes(prediction[0])
    
    if return_plot:
        #Matplotlib likes no batch dim and channels first
        image = image.squeeze(0).permute(1,2,0)
        plot, ax = visualize.plot_predictions(image, df)
        return plot
    else:
        return df
    
def predict_file(model,csv_file,root_dir, savedir, device):
    """Create a dataset and predict entire annotation file
    
    Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position. 
    Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line. 
    
    Args:
        csv_file: path to csv file 
        root_dir: directory of images. If none, uses "image_dir" in config
        savedir: Optional. Directory to save image plots.
        device: pytorch device of 'cuda' or 'cpu' for gpu prediction. Set internally.
    Returns:
        df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
    """    
    
    input_csv = pd.read_csv(csv_file)
    
    #Just predict each image once. 
    images = input_csv["image_path"].unique()         
        
    prediction_list = []
    for path in images:
        image = io.imread("{}/{}".format(root_dir,path))
        
        image = preprocess.preprocess_image(image)
        
        #Just predict the images, even though we have the annotations
        if not device.type=="cpu":
            image = image.to(device)
        
        prediction = model(image)        
        
        prediction = visualize.format_boxes(prediction[0])
        prediction["image_path"] = path
        prediction_list.append(prediction)
        
        if savedir:
            image = image.squeeze(0).permute(1,2,0)                
            plot, ax = visualize.plot_predictions(image, prediction)
            annotations = input_csv[input_csv.image_path==path]
            plot = visualize.add_annotations(plot, ax, annotations)
            plot.savefig("{}/{}.png".format(savedir,os.path.splitext(path)[0]))
        
    df = pd.concat(prediction_list,ignore_index=True)
    
    return df

def predict_tile(model,
                 device,
                 raster_path=None,
                 image=None,
                 patch_size=400,
                 patch_overlap=0.05,
                 iou_threshold=0.15,
                 return_plot=False,
                 use_soft_nms = False,
                 sigma = 0.5,
                 thresh = 0.001):
    """For images too large to input into the model, predict_tile cuts the
    image into overlapping windows, predicts trees on each window and
    reassambles into a single array.
    
    Args:
        model: pytorch model
        raster_path: Path to image on disk
        image (array): Numpy image array in BGR channel order
            following openCV convention
        patch_size: patch size default400,
        patch_overlap: patch overlap default 0.15,
        iou_threshold: Minimum iou overlap among predictions between
            windows to be suppressed. Defaults to 0.5.
            Lower values suppress more boxes at edges.
        return_plot: Should the image be returned with the predictions drawn?
        use_soft_nms: whether to perform Gaussian Soft NMS or not, if false, default perform NMS. 
        sigma: variance of Gaussian function used in Gaussian Soft NMS
        thresh: the score thresh used to filter bboxes after soft-nms performed
        device: pytorch device of 'cuda' or 'cpu' for gpu prediction. Set internally.
    
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
        
        #crop is RGB channel order, change to BGR?
        boxes = predict_image(model=model,
                              image=crop,
                              return_plot=False,
                              device=device)
        if not boxes is None:
            #transform the coordinates to original system
            xmin, ymin, xmax, ymax = windows[index].getRect()
            boxes.xmin = boxes.xmin + xmin
            boxes.xmax = boxes.xmax + xmin
            boxes.ymin = boxes.ymin + ymin
            boxes.ymax = boxes.ymax + ymin
    
            predicted_boxes.append(boxes)
    
    if len(predicted_boxes) == 0:
        print("No predictions made, returning None")
        return None
    
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
        
        if use_soft_nms == False:
            #Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).
            bbox_left_idx = nms(boxes = boxes, scores = scores, iou_threshold=iou_threshold)
        else:
            #Performs soft non-maximum suppression (soft-NMS) on the boxes.
            bbox_left_idx = soft_nms(boxes = boxes, scores = scores, sigma = sigma, thresh=thresh)
        
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
    
def soft_nms(boxes,scores,sigma = 0.5, thresh = 0.001):
    '''
    Perform python soft_nms to reduce the confidances of the proposals proportional  to IoU value
    Paper: Improving Object Detection With One Line of Code
    Code : https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py
    Args:
        boxes: predicitons bounding boxes tensor format [x1,y1,x2,y2] 
        scores: the score corresponding to each box tensors
        sigma: variance of Gaussian function
        thresh: score thresh 
    Return:
        idxs_keep: the index list of the selected boxes
    
    '''
    # indexes concatenate boxes with the last column
    N = boxes.shape[0]
    indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)

    boxes = torch.cat((boxes, indexes), dim=1)


    # The order of boxes coordinate is [x1,y1,y2,x2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                boxes[i], boxes[maxpos.item() + i + 1] = boxes[maxpos.item() + i + 1].clone(), boxes[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        xx1 = np.maximum(boxes[i, 0].numpy(), boxes[pos:, 0].numpy())
        yy1 = np.maximum(boxes[i, 1].numpy(), boxes[pos:, 1].numpy())
        xx2 = np.minimum(boxes[i, 2].numpy(), boxes[pos:, 2].numpy())
        yy2 = np.minimum(boxes[i, 3].numpy(), boxes[pos:, 3].numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    idxs_keep = boxes[:, 4][scores > thresh].int()

    return idxs_keep