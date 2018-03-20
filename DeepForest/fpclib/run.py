# Author: Hamid Hamraz (hhamraz@cs.uky.edu)
# 2016, Department of Computer Science, University of Kentucky

import sys
import numpy
from operator import itemgetter

import fpclib.util as util
import fpclib.forest as forest
from fpclib.constants import *
import fpclib.crownSegmenter as crw_seg

# Loading the point cloud.  We assign 'True" to the skipHeader parameter, to skip the first line of the input.
cloud = util.loadData(sys.argv[1], skipHeader=True)

# Indexing the point cloud to a grid.  Normalized height of the LiDAR points are determined here.
grid = forest.indexPointCloud(cloud)

# Lower vegetation and ground points are removed since normalized heights were already calculated.
forest.excludeLowPoints(grid)

# If the highest point of the grid cell is not a first return, that grid cell won't be used.  
# This helps decrease non-necessary surface variability.
forest.excludeNonFirstReturnSurfaces(grid)

# Now, we segment the point cloud to individual trees.
# multilayer=0 means that the DSM-based segmentation is used. 
# multilayer=1 means multilayer segmentation with the regional layering method is used. 
# multilayer=2 means multilayer segmentation with the local layering method is used.  
# For the multilayer segmentations, if merging crowns across layers is desired, merge=True should be used.
# trees variable is a list of trees, where each tree is actually a list of LiDAR points.
# noise variable is a list of small noise pieces , where each piece is actually a list of LiDAR points.
trees, noise = crw_seg.segmentCanopy(grid, multilayer=0, merge=False)

print "Number of trees:", len(trees)

# 2D Visualizing the segmented crowns in distinct colors aerially.
# If the multilayer segmentation was performed, this visualization may look wierd because it paints layers on top of each other.
util.scatterPlot(trees, X, Y)

# If the path to a file is provided, the segmentation result is written to that file.
if len(sys.argv) > 2:
	util.outputPlotResult(trees, noise, sys.argv[2])
