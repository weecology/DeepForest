from PIL import Image
import slidingwindow as sw
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches


def compute_windows(image,pixels,overlap):
    im = Image.open(image)
    data = np.array(im)    
    windows = sw.generate(image, sw.DimOrder.HeightWidthChannel, 300.05)
    return(windows)
    
    
def load(path):
    im = Image.open(path)
    data = np.array(im)
    windows = sw.generate(data, sw.DimOrder.HeightWidthChannel, 300, 0.05)
    
    # Do stuff with the generated windows
    fig,ax = pyplot.subplots(1)
    ax.imshow(data)
    
    for window in windows:
        
        x,y,w,h=window.getRect()
        print(x,y)
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')     
        ax.add_patch(rect)
    
    pyplot.show()
    for window in windows:        
        d=data[window.indices()]
        pyplot.imshow(d)
        pyplot.show()
   
if __name__=="__main__":   
    load(path="/Users/ben/Documents/DeepForest/data/OSBS_006_test.tif")