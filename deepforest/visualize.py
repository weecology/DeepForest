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
    plt.imshow(image)
    ax = plt.gca()
    for index, row in df.iterrows():
        xmin = row["xmin"]
        ymin = row["ymin"]
        width = row["xmax"] - xmin
        height = row["ymax"] - ymin
        rect = create_box(xmin=xmin,ymin=ymin, height=height, width=width)
    ax.add_patch(rect)
    
    return ax
        
def create_box(xmin, ymin, height, width, color="cyan",linewidth=2):
    rect = patches.Rectangle((xmin,ymin),
                     height,
                     width,
                     linewidth=2,
                     edgecolor=color,
                     fill = False)
    return rect