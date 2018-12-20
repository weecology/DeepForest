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
import os
from scipy.signal import medfilt2d

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
    tile_annotations=annotations[annotations["rgb_path"]==row["tile"]]
    
    #Set tile extent to convert to UTMs, flipped origin from R to Python
    tile_xmin=tile_annotations.tile_xmin.unique()[0]
    tile_ymax=tile_annotations.tile_ymax.unique()[0]
    tile_ymin=tile_annotations.tile_ymin.unique()[0]
    
    #Get window cartesian coordinates
    x,y,w,h= windows[row["window"]].getRect()
    
    window_utm_xmin=x * rgb_res + tile_xmin
    window_utm_xmax=(x+w) * rgb_res + tile_xmin
    window_utm_ymin=tile_ymax - (y * rgb_res)
    window_utm_ymax= tile_ymax - ((y+h) * rgb_res)
    
    return(window_utm_xmin,window_utm_xmax,window_utm_ymin,window_utm_ymax)

def fetch_lidar_filename(row,lidar_path,site):
    """
    Find lidar path in a directory.
    param: row a dictionary with tile "key" for filename to be searched for
    return: string location on disk
    """
    
    #first try identical name - this isn't great practice here, needs to be improved. How to direct the lidar path to the right directory?
    direct_filename=os.path.join("data" ,site,os.path.splitext(row["tile"])[0] + ".laz")

    if os.path.exists(direct_filename):
        laz_path=direct_filename
    else:
        print("Filename: %s does not exist, searching within %s" %(direct_filename,lidar_path))        
        laz_path=find_lidar_file(image_path=row["tile"],lidar_path=lidar_path)
        
    return laz_path

def load_lidar(laz_path):
    """
    Load lidar tile from file based on path name
    param: A string of the file path of the lidar tile
    return: A pyfor point cloud
    """
    
    try:
        pc=pyfor.cloud.Cloud(laz_path)
        pc.extension=".las"    
    except FileNotFoundError:
        print("Failed loading path: %s" %(laz_path))
        
    #normalize and filter
    zhang_filter=pyfor.ground_filter.Zhang2003(pc, cell_size=1)
    zhang_filter.normalize()    
    
    #Quick filter for unreasonable points.
    pc.filter(min = -5, max = pc.data.points.z.quantile(0.995), dim = "z")    
    
    #Check dim
    assert (not pc.data.points.shape[0] == 0), "Lidar tile is empty!"
    
    return pc

def compute_chm(lidar_tile,annotations,row,windows,rgb_res,kernel_size):
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
    
    #Create shapely polygon for clipping
    poly=createPolygon(xmin, xmax, ymin,ymax)
    
    #Clip lidar to geographic extent    
    clipped=lidar_tile.clip(poly)
        
    #Median filter
    chm =clipped.chm(cell_size = 0.1 , interp_method = "nearest")
    chm.array = medfilt2d(chm.array, kernel_size=kernel_size)
    
    #remove understory noise, anything under 2m.
    chm.array[chm.array < 2] = 0   
    
    return chm

def find_lidar_file(image_path,lidar_path):
    """
    Find the lidar file that matches RGB tile
    """
    #Look for lidar tile
    laz_files=glob.glob(lidar_path + "*.laz")
    
    #extract geoindex
    pattern=r"(\d+_\d+)_image"
    match=re.findall(pattern,image_path)
    
    if len(match)>0:
        match=match[0]
    else:
        return None
    
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
    """
    Enforce the same data structure between the rgb image and the canopy height model. 
    Add 0's around the edge to fix shape
    """
    
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

def bind_array(image,chm):
    '''
    Bind the rgb image and the lidar canopy height model
    Pad with zeros if needed
    '''
    
    #Check if arrays are same shape. If not, pad.
    if not chm.shape==image.shape:
        padded_chm=pad_array(image=image,chm=chm)
        
        #Append to bottom of image
        four_channel_image=np.dstack((image,padded_chm))
        
    else:
        four_channel_image=np.dstack((image,chm))    
    return four_channel_image