from deepforest import main
from deepforest import utilities
import os
import glob

files = glob.glob("*.tif")
m = main.deepforest()
m.use_release()
for f in files:
    boxes = m.predict_tile(f, patch_size=2500)
    shp = utilities.boxes_to_shapefile(boxes,root_dir=os.getcwd())
    basename = os.path.splitext(os.path.basename(f))[0]
    shp.to_file("/Users/benweinstein/Downloads/{}.shp".format(basename))

m = main.deepforest()
m.use_release()
patch_sizes = [400, 800]
for patch_size in patch_sizes:
    f = "/Users/benweinstein/Downloads/5d618c861b68d80006a2ce64.tif"
    boxes = m.predict_tile(f, patch_size=patch_size)
    boxes = boxes[boxes.score>0.3]
    shp = utilities.boxes_to_shapefile(boxes,root_dir="/Users/benweinstein/Downloads/")
    basename = os.path.splitext(os.path.basename(f))[0]
    shp.to_file("/Users/benweinstein/Downloads/{}_{}.shp".format(basename,patch_size))