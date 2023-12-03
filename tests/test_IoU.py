#Test IoU
from .conftest import download_release
from deepforest import IoU
from deepforest import main
from deepforest import get_data
from deepforest import visualize

import os
import shapely
import geopandas as gpd
import pandas as pd

def test_compute_IoU(m, tmpdir):
    csv_file = get_data("OSBS_029.csv")
    predictions = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    ground_truth = pd.read_csv(csv_file)
    
    predictions['geometry'] = predictions.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    predictions = gpd.GeoDataFrame(predictions, geometry='geometry')
    
    ground_truth['geometry'] = ground_truth.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    ground_truth = gpd.GeoDataFrame(ground_truth, geometry='geometry')        
    
    ground_truth.label = 0
    predictions.label = 0
    visualize.plot_prediction_dataframe(
        df=predictions, 
        ground_truth=ground_truth, 
        root_dir=os.path.dirname(csv_file), 
        savedir=tmpdir)        
    
    result = IoU.compute_IoU(ground_truth, predictions)
    assert result.shape[0] == ground_truth.shape[0]
    assert sum(result.IoU) > 10 