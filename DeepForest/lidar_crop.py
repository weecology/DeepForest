'''
Crop lidar canopy height model from NEON tile and sliding window coordinates
'''

import pandas as pd
import pyfor
from shapely import geometry
from matplotlib import pyplot
import re
import glob 
import numpy as np

def createPolygon(xmin,xmax,ymin,ymax):
    '''
    Convert a pandas row into a polygon bounding box
    ''' 
    p1 = geometry.Point(xmin,ymax)
    p2 = geometry.Point(xmax,ymax)
    p3 = geometry.Point(xmax,ymin)
    p4 = geometry.Point(xmin,ymin)
    
    pointList = [p1, p2, p3, p4, p1]
    
    poly = geometry.Polygon([[p.x, p.y] for p in pointList])
    
    return poly


def get_window_extent(annotations,row,windows,rgb_res):
    '''
    Get the geographic coordinates of the sliding window.
    Be careful that the ymin in the geographic data refers to the utm (bottom) and the ymin in the cartesian refers to origin (top). 
    '''
    #Select tile from annotations to get extent
    tile_annotations=annotations[annotations["rgb_path"]==row["image"]]
    
    #Set tile extent to convert to UTMs, flipped origin from R to Python
    tile_xmin=tile_annotations.tile_xmin.unique()[0]
    tile_ymax=tile_annotations.tile_ymax.unique()[0]
    tile_ymin=tile_annotations.tile_ymin.unique()[0]
    
    #Get window cartesian coordinates
    x,y,w,h= windows[row["windows"]].getRect()
    
    window_utm_xmin=x * rgb_res + tile_xmin
    window_utm_xmax=(x+w) * rgb_res + tile_xmin
    window_utm_ymin=tile_ymax - (y * rgb_res)
    window_utm_ymax= tile_ymax - ((y+h) * rgb_res)
    
    return(window_utm_xmin,window_utm_xmax,window_utm_ymin,window_utm_ymax)

def compute_chm(annotations,row,windows,rgb_res,lidar_path):
    """
    Computer a canopy height model based on the available laz file to align with the RGB data
    """
    #Find geographic coordinates of the rgb tile
    xmin,xmax,ymin,ymax=get_window_extent(annotations=annotations, row=row, windows=windows,rgb_res=rgb_res)

    #Add a small buffer based on rgb res
    xmin=xmin - rgb_res/2
    xmax=xmax + rgb_res/2
    
    ymin=ymin - rgb_res/2
    ymax=ymax + rgb_res/2
    
    laz_path=find_lidar_file(image_path=row["image"],lidar_path=lidar_path)
    
    pc=pyfor.cloud.Cloud(laz_path)
    pc.extension=".las"
    
    #Clip lidar to geographic extent
        
    #Create shapely polygon for clipping
    poly=createPolygon(xmin, xmax, ymin,ymax)
    
    clipped=pc.clip(poly)
    
    #TODO normalize properly, for now just subtract min value, see https://github.com/brycefrank/pyfor/issues/25
    clipped.data.points.z =  clipped.data.points.z  -  clipped.data.points.z.min()
    #clipped.normalize(1)
    
    #Quick filter for unreasonable points.
    clipped.filter(min = -5, max = 100, dim = "z")
    
    chm = clipped.chm(cell_size = rgb_res , interp_method = "nearest", pit_filter = "median", kernel_size = 11)
    
    return chm

def find_lidar_file(image_path,lidar_path):
    """
    Find the lidar file that matches RGB tile
    """
    #Look for lidar tile
    laz_files=glob.glob(lidar_path + "*.laz")
    
    #extract geoindex
    pattern=r"(\d+_\d+)_image"
    match=re.findall(pattern,image_path)[0]
    
    #Look for index in available laz
    laz_path=None
    for x in laz_files:
        if match in x:
            laz_path=x

    #Raise if no file found
    if not laz_path:
        print("No matching lidar file, check the lidar path: %s" %(lidar_path))
        FileNotFoundError
    
    return laz_path

def pad_array(image,chm):
    h,w=np.subtract(image.shape[0:2],chm.shape[0:2] )
    
    #distribute evenly to both sides as possible
    left = 0
    right = 0
    top=0
    bottom=0
    
    #allocate padding 'evenly'
    for x in np.arange(h):
        if x % 2 ==0:
            left +=1
        else:
            right +=1
    
    for x in np.arange(w):
        if x % 2 ==0:
            bottom +=1
        else:
            top +=1        
    #pad
    padded=np.pad(chm,((left,right),(top,bottom)),"constant")
    
    return padded    

if __name__=="__main__":
    lidar_path="/Users/ben/Documents/DeepForest/data/NEON_D03_OSBS_DP1_407000_3291000_classified_point_cloud.laz"