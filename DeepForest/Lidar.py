'''
Sample point clouds for pointnet
'''

import h5py as h5
import numpy as np
import pyfor
import open3d
from scipy.stats import gaussian_kde
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
from shapely import geometry

def preprocess(lidar_path,bounding_box_path,num_points,view=False):
    '''
    Read in file, filter out ground, crop each tree, normalize to unit sphere and sample points
    path: Location of raw lidar tile
    num_points: Number of points to sample
    view: Visualize result in 3d plot
    '''
    
    # read bounding box
    bbox=pd.read_csv(bounding_box_path)
    
    # Read Lidar tile
    cloud=loadLidar(lidar_path)
        
    #filter out bare earth
    cloud.filter(min = 3, max = 200, dim = "z")
    
    #Crop out tree
    trees=[]
    for index,row in bbox.head().iterrows():
        
        print(index)
        #Create Polygon
        features=createPolygon(row)        
        
        #Crop
        clipped=cloud.clip(features)
        
        #Create numpy array
        clipped_numpy=clipped.las.points.iloc[:,0:3].values
        trees.append(clipped_numpy)
    
    
    #Subsampling on each tree
    sampled_trees=[]
    
    for tree in trees:
        sampled_trees.append(sample(tree,num_points,view))
    
    #get labels for each tree
    labels=bbox['label'].values
    
    return(sampled_trees,labels)
    
    # save as h5 file
    #save_h5(h5_filename, data, label)

## Load lidar
def loadLidar(path):
    pc = pyfor.cloud.Cloud(path)
    pc.normalize(0.5)
    return(pc)
    
def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

def min_distance(d):
    #compute pairwise distances and convert into square.
    distances=distance.pdist(d)
    distances=distance.squareform(distances)
    np.fill_diagonal(distances, np.inf)
    
    #min distance
    mind=distances.argmin(axis=0)
    return(mind)

def createPolygon(row):
    '''
    Convert a pandas row into a polygon bounding box
    ''' 
    
    p1 = geometry.Point(row.xmin,row.ymin)
    p2 = geometry.Point(row.xmin,row.ymax)
    p3 = geometry.Point(row.xmax,row.ymin)
    p4 = geometry.Point(row.xmax,row.ymax)
    
    pointList = [p1, p2, p3, p4, p1]
    
    poly = geometry.Polygon([[p.x, p.y] for p in pointList])
    
    return poly

def sample(tree,num_points,view):
    '''
    Grid Sample and weigthed sampling to create standard array
    tree: a 3d numpy array
    '''
    
    # normalize
    normal_xyz=normalize(tree)
    
    #convert to open3d format
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(normal_xyz)
    
    #subsample on fine grid
    downpcd = open3d.voxel_down_sample(pcd, voxel_size = 0.05)
    
    #Minimum distances among points
    dat=np.asarray(downpcd.points)
    d=min_distance(dat)
    
    print(d.shape[0])
    
    #Sample points weighted by min distance.
    #If num points is more than available, replace=T
    if(d.shape[0]>num_points):
        rows=np.random.choice(d.shape[0],size=num_points,replace=False)
    else:
        rows=np.random.choice(d.shape[0],size=num_points,replace=True)        
    
    sampled = open3d.PointCloud()
    sampled.points = open3d.Vector3dVector(dat[rows,])
    
    #View result if interested
    if view:
        pcd.paint_uniform_color([0.1,0.1,0.9])
        downpcd.paint_uniform_color([0.1,0.9,0.1])
        sampled.paint_uniform_color([0.9,0.1,0.1])
        open3d.draw_geometries([pcd+sampled+downpcd]) 
    
    #Return as numpy array
    return(np.asarray(sampled.points))
    
if __name__=="__main__":
    lidar_path="../data/OSBS_003.laz"    
    bounding_box_path="../data/bounding_boxes_OSBS_003.csv"
    trees=preprocess(lidar_path,bounding_box_path,num_points=100,view=True)    
