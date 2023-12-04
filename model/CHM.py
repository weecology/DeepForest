#CHM height module. Given a x,y location and a pool of CHM images, find the matching location and extract the crown level CHM measurement
import glob
import numpy as np 
from src.utilities import find_sensor_path
import rasterstats
import geopandas as gpd
import pandas as pd
import traceback

def non_zero_99_quantile(x):
    """Get height quantile of all cells that are no zero"""
    if (x==0).all():
        return 0
    
    mdata = np.ma.masked_where(x < 0.5, x)
    mdata = np.ma.filled(mdata, np.nan)
    percentile = np.nanpercentile(mdata, 99)
    
    return (percentile)

def postprocess_CHM(df, lookup_pool):
    """Field measured height must be within min_diff meters of canopy model"""
    # Extract zonal stats, add a small offset, the min box can go to next tile.
    # Find the most recent year if labeled, contrib samples have no years 
    if "contrib" in df.plotID.iloc[0]:
        try:
            #Get all years of CHM data
            CHM_path = find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds)
        except ValueError as e:
            df["CHM_height"] = np.nan
            return df
    else:
        try:
            # Get all years of CHM data
            CHM_paths = find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds, all_years=True)
            survey_year = int(df.eventID.apply(lambda x: x.split("_")[-1]).max())
            
            # Check the difference in date
            CHM_years = [int(x.split("/")[-5].split("_")[0]) for x in CHM_paths]
            CHM_path = CHM_paths[np.argmin([abs(survey_year - x) for x in CHM_years])]     
        except ValueError as e:
            df["CHM_height"] = np.nan
            
            return df
 
    #buffer slightly, CHM model can be patchy
    geom = df.geometry.buffer(1)
    draped_boxes = rasterstats.zonal_stats(geom,
                                           CHM_path,
                                           add_stats={'q99': non_zero_99_quantile})
    df["CHM_height"] = [x["q99"] for x in draped_boxes]

    #if height is null, try to assign it
    df.height.fillna(df["CHM_height"], inplace=True)

    return df
        
def CHM_height(shp, CHM_pool):
        """For each plotID extract the heights from LiDAR derived CHM
        Args:
            shp: shapefile of data to filter
            config: DeepTreeAttention config file dict, parsed, see config.yml
        """    
        filtered_results = []
        for name, group in shp.groupby("plotID"):
            try:
                result = postprocess_CHM(group, lookup_pool=CHM_pool)
                filtered_results.append(result)
            except Exception as e:
                print("plotID {} raised: {}".format(name,e))                
                traceback.print_exc()
                
        filtered_shp = gpd.GeoDataFrame(pd.concat(filtered_results,ignore_index=True))
        
        return filtered_shp

def height_rules(df, min_CHM_height=1, max_CHM_diff=4, CHM_height_limit=8):
    """Which data points should be included based on a comparison of CHM and field heights
        This is asymmetric, field heights under CHM height are signs of subcanopy, whereas CHM under field height is mismeasurement and growth. 
        Do not filter NA heights
    Args:
        df: a pandas dataframe with CHM_height and height columns
        min_CHM_height: if CHM is avialble, remove saplings under X meters
        max_CHM_diff: max allowed difference between CHM and field height if CHM > field height
        CHM_height_limit: max allowed difference between CHM and field height if CHM < field height
    Returns:
       df: filtered dataframe
    """
    keep = []
    for index, row in df.iterrows():
        if np.isnan(row["CHM_height"]):
            keep.append(True)
        elif np.isnan(row["height"]):
            keep.append(True)
        elif row.CHM_height < min_CHM_height:
            keep.append(False)
        elif row.CHM_height > row.height:
            if (row.CHM_height - row.height) >= max_CHM_diff:
                keep.append(False)
            else:
                keep.append(True)
        elif row.CHM_height <= row.height:
            if (row.height - row.CHM_height) >= CHM_height_limit:
                keep.append(False)
            else:
                keep.append(True)
        else:
            print("No conditions applied to CHM_height {}, height {}".format(row.CHM_height,row.height))
            keep.append(True)
            
    df["keep"] = keep
    df = df[df.keep]
    
    return df

def filter_CHM(shp, CHM_pool, min_CHM_height=1, max_CHM_diff=4, CHM_height_limit=8):
    """Filter points by height rules"""
    if min_CHM_height is None:
        return shp
    
    #extract CHM height
    shp = CHM_height(shp, CHM_pool)
    shp = height_rules(
        df=shp,
        min_CHM_height=min_CHM_height,
        max_CHM_diff=max_CHM_diff,
        CHM_height_limit=CHM_height_limit)

    return shp