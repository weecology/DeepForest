#Visualize module for plotting and handling predictions
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def format_predictions(prediction):
    """Format a retinanet prediction into a pandas dataframe for a single image"""
    df = pd.DataFrame(prediction["boxes"].cpu().detach().numpy(),columns=["xmin","ymin","xmax","ymax"])
    df["label"] = prediction["labels"].cpu().detach().numpy()
    df["scores"] = prediction["scores"].cpu().detach().numpy()
    
    return df

def plot_predictions(image, df):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for index, row in df.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        rect = create_box(xmin=xmin,ymin=ymin, height=height, width=width)
        ax.add_patch(rect)
    #no axis show up
    plt.axis('off')

    return fig, ax
        
def create_box(xmin, ymin, height, width, color="cyan",linewidth=1):
    rect = patches.Rectangle((xmin,ymin),
                     height,
                     width,
                     linewidth=linewidth,
                     edgecolor=color,
                     fill = False)
    return rect

def add_annotations(plot, ax, annotations):
    """Add annotations to an already created visuale.plot_predictions
    Args:
        plot: matplotlib figure object
        ax: maplotlib axes object
        annotations: pandas dataframe of bounding box annotations
    Returns:
        plot: matplotlib figure object
    """
    for index, row in annotations.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        rect = create_box(xmin=xmin,ymin=ymin, height=height, width=width, color="orange")
        ax.add_patch(rect) 
    
    return plot