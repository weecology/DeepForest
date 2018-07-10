'''
Crop lidar canopy height model from NEON tile and sliding window coordinates
'''

import pandas as pd
import pyfor
from shapely import geometry
from matplotlib import pyplot

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
    
    #Set tile extent, remember to flip y axis from R -> Python
    tile_xmin=tile_annotations.tile_xmin.unique()[0]
    tile_xmax=tile_annotations.tile_xmax.unique()[0]
    tile_ymin=tile_annotations.tile_ymin.unique()[0]
    tile_ymax=tile_annotations.tile_ymax.unique()[0]
    
    #Get window cartesian coordinates
    x,y,w,h= windows[row["windows"]].getRect()
    
    window_xmin=x * rgb_res + tile_xmin
    window_xmax=(x+w) * rgb_res + tile_xmin
    window_ymin=(y+h) * rgb_res + tile_ymin
    window_ymax=y * rgb_res + tile_ymin
    
    return(window_xmin,window_xmax,window_ymin,window_ymax)

def compute_chm(annotations,row,windows,rgb_res,lidar_path):
    #Find geographic coordinates of the rgb tile
    xmin,xmax,ymin,ymax=get_window_extent(annotations=annotations, row=row, windows=windows,rgb_res=rgb_res)
    
    #Create shapely polygon
    poly=createPolygon(window_xmin, window_xmax, window_ymax,window_ymin)
    
    #Read lidar tile
    pc=pyfor.cloud.Cloud(lidar_path)
    
    #Clip to geographic extent
    clipped=pc.clip(poly)
    
    #Quick filter for unreasonable points.
    clipped.filter(min = -5, max = 100, dim = "z")
    
    clipped.normalize(1)
    chm = clipped.chm(cell_size = 0.1, interp_method = "nearest", pit_filter = "median", kernel_size = 11)
    
    return chm


if __name__=="__main__":
    lidar_path="/Users/ben/Documents/DeepForest/data/NEON_D03_OSBS_DP1_407000_3291000_classified_point_cloud.laz"