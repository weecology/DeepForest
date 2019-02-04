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
from DeepForest import Lidar 

def drape_boxes(boxes, tilename=None, lidar_dir=None, pc=None):
    '''
    boxes: predictions from retinanet
    cloud: pyfor cloud used to generate canopy height model
    tilename: name of the .laz file, without extension.
    lidar_dir: Where to look for lidar tile
    pc: Optional point cloud from memory, on the fly generation
    '''
    
    if not pc:
        #Find lidar path
        lidar_path = os.path.join(lidar_dir, tilename) + ".laz"
        
        #Load cloud
        pc = Lidar.load_lidar(lidar_path)
    
    density = Lidar.check_density(pc)
    
    print("Point density is {:.2f}".format(density))
    
    if density < 4:
        print("Point density of {:.2f} is too low, skipping image {}".format(density, tilename))
        return None
    
    #reset user_data column
    pc.data.points.user_data =  np.nan
        
    tree_counter = 1
    for box in boxes:

        #Find utm coordinates
        xmin, xmax, ymin, ymax = find_utm_coords(box = box, pc = pc)
        
        #Update points
        pc.data.points.loc[(pc.data.points.x > xmin) & (pc.data.points.x < xmax)  & (pc.data.points.y < ymin)   & (pc.data.points.y > ymax),"user_data"] = tree_counter
        
        #update counter
        tree_counter +=1 
        
    #remove ground points    
    pc.data.points.loc[pc.data.points.z < 2, "user_data"] = np.nan
    
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

def cloud_to_box(pc):
    ''''
    pc: a pyfor point cloud with labeled tree data in the 'user_data' column.
    Turn a point cloud with a "user_data" attribute into a numpy array of boxes
    '''
    tree_boxes = [ ]
    
    tree_ids = pc.data.points.user_data.dropna().unique()
    #Try to follow order of input boxes, start at Tree 1.
    tree_ids.sort()
    
    #For each tree, get the bounding box
    for tree_id in tree_ids:
        
        #Select points
        points = pc.data.points.loc[pc.data.points.user_data == tree_id,["x","y"]]
        
        #turn utm to cartesian, subtract min x and max y value, divide by cell size. Max y because numpy 0,0 origin is top left. utm N is top. 
        points.x = points.x - pc.data.points.x.min()
        points.y = pc.data.points.y.max() - points.y 
        points =  points.values/ 0.1
        
        s = gp.GeoSeries(map(geometry.Point, zip(points[:,0], points[:,1])))
        point_collection = geometry.MultiPoint(list(s))
        bounds = point_collection.bounds
        tree_boxes.append(bounds)
        
    #pass as numpy array of 3 dim
    tree_boxes =np.array(tree_boxes)
    
    return tree_boxes
    
def cloud_to_polygons(pc):
    ''''
    Turn a point cloud with a "Tree" attribute into 2d polygons for calculating IoU
    returns a geopandas frame of convex hulls
    '''
        
    hulls = [ ]
    
    tree_ids = pc.data.points.user_data.dropna().unique()
    
    for treeid in tree_ids:
        
        points = pc.data.points.loc[pc.data.points.user_data == treeid,["x","y"]].values
        s = gp.GeoSeries(map(geometry.Point, zip(points[:,0], points[:,1])))
        point_collection = geometry.MultiPoint(list(s))
        convex_hull = point_collection.convex_hull
        hulls.append(convex_hull)
        
    hulldf = gp.GeoSeries(hulls)
    
    return hulldf