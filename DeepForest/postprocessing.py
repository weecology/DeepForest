''''
Bounding box post-processing
After neural network predicts a rectangular box, overlay it on the point cloud and assign points for instance based detections
Create a polygon from final tree to use for evaluation
'''

import os
import pyfor
import numpy as np
import geopandas as gp
from shapely import geometry

def drape_boxes(boxes, tilename, lidar_dir):
    '''
    boxes: predictions from retinanet
    cloud: pyfor cloud used to generate canopy height model
    '''
    
    #Find lidar path
    lidar_path = os.path.join(lidar_dir, tilename) + ".laz"
    
    #Load cloud
    pc = pyfor.cloud.Cloud(lidar_path)
    pc.normalize(1)
    
    #TODO order boxes by score. place best boxes on top.
    
    tree_counter = 0
    for box in boxes:

        #Find utm coordinates
        xmin, xmax, ymin, ymax = find_utm_coords(box = box, pc = pc)
        
        #Update points
        pc.data.points.loc[(pc.data.points.x > xmin) & (pc.data.points.x < xmax)  & (pc.data.points.y < ymin)   & (pc.data.points.y > ymax),"Tree"] = tree_counter
        
        #update counter
        tree_counter +=1 
        
    #remove ground points    
    pc.data.points.loc[pc.data.points.z < 2, "Tree"] = np.nan
    
    #TODO snap points to closest tree based on distance tolerance
            
    #View results        
    #pyfor.rasterizer.Grid(pc, cell_size=1).raster("max", "Tree").plot()

    return pc

    
def find_utm_coords(box,  pc, rgb_res = 0.1):
    
    """
    Turn cartesian coordinates back to projected utm
    """
    
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    
    tile_xmin = pc.data.points.x.min()
    tile_ymax = pc.data.points.y.max()
    
    window_utm_xmin = xmin * rgb_res + tile_xmin
    window_utm_xmax = xmax * rgb_res + tile_xmin
    window_utm_ymin = tile_ymax - (ymin * rgb_res)
    window_utm_ymax= tile_ymax - (ymax* rgb_res) 
    
    return(window_utm_xmin, window_utm_xmax, window_utm_ymin, window_utm_ymax)

def cloud_to_polygons(pc):
    ''''
    Turn a point cloud with a "Tree" attribute into 2d polygons for calculating IoU
    returns a geopandas frame of convex hulls
    '''
        
    hulls = [ ]
    
    tree_ids = pc.data.points.Tree.dropna().unique()
    
    for treeid in tree_ids:
        
        points = pc.data.points.loc[pc.data.points.Tree == 1,["x","y"]].values
        s = gp.GeoSeries(map(geometry.Point, zip(points[:,0], points[:,1])))
        point_collection = geometry.MultiPoint(list(s))
        convex_hull = point_collection.convex_hull
        hulls.append(convex_hull)
        
    hulldf = gp.GeoSeries(hulls)
    
    return hulldf
    