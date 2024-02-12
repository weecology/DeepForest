import glob
import pandas as pd
laz = glob.glob("/orange/ewhite/NeonData/*/DP1.30003.001/neon-aop-products/**/ClassifiedPointCloud/*.laz", recursive=True)
rgb = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/neon-aop-products/**/*.tif", recursive=True)
pd.DataFrame(laz).to_csv("lidar_tiles.csv")
pd.DataFrame(rgb).to_csv("rgb_tiles.csv")