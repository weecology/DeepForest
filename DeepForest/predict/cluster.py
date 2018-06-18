'''
Generate clusters of trees using a watershed algorithm. The goal of the script is to take in a raw lidar cloud, find the treetops, 
perform watershed and return bounding box coordinates of clusters for tree crown prediction by neural network.
'''

import pyfor
import rasterio
from matplotlib.pyplot import imshow

def findTreeClusters(path):
    
    #Load 
    cloud=loadLidar(path)
    
    #Create CHM
    chm=calcCHM(cloud)

    #Watershed and return polygons
    boxes=watershed(chm,threshold=3,plot=False)
        
    return(boxes)

## Load lidar
def loadLidar(path):
    pc = pyfor.cloud.Cloud(path)
    pc.normalize(0.5)
    return(pc)

def calcCHM(cloud):
    chm = cloud.chm(0.5, interp_method = "nearest", pit_filter="median")
    return(chm)
    
def watershed(chm,threshold,plot=False):
    segmentation=chm.watershed_seg(threshold_abs=threshold, classify=False, plot=False)
    final_segments = segmentation[segmentation["raster_val"] > 0]
    return(final_segments)

def crop(chm,geometry):
    #crop and return image
    with MemoryFile(chm) as memfile:
        with memfile.open() as src:
            out_image, out_transform = mask(src, [geometry], crop=True)
    imshow(out_image)
    return(out_image)
        
if __name__=="__main__":
    segments=findTreeClusters("/Users/ben/Documents/TreeSegmentation/data/2017/Lidar/OSBS_003.laz")
    print(segments.head(n=10))
    