"""Testing module for bounding box generation and merging. This is needed for deepforest.predict_tile and dealing with overlapping windows"""
from deepforest import boxes
import pytest
import pandas as pd
import shapely
import matplotlib.pyplot as plt
        
def plot_box(poly_list):
    #Plot a list of bounding boxes
    fig = plt.figure(1, figsize=(5,5), dpi=90)   
    ax = fig.add_subplot(111)    
    for poly in poly_list:
        x,y = poly.exterior.xy
        ax.plot(x, y, color='#6699cc', alpha=0.7,
            linewidth=3, solid_capstyle='round', zorder=2)
        ax.set_title('Polygon')
    plt.show()
    
@pytest.fixture()
def df():
    df = pd.DataFrame({"xmin":[1,2,9],"ymin":[0,1,9],"xmax":[4,5,10],"ymax":[2,5,10]})
    return df

def test_make_box(df):
    row = df.iloc[1]
    bbox = boxes.make_box(row.xmin,row.ymin,row.xmax,row.ymax)
    assert isinstance(bbox,shapely.geometry.Polygon)
    
def test_box_from_df(df):
    box_list = boxes.box_from_df(df)
    assert len(box_list) == 3
    
def test_merge_boxes(df):
    box_list = boxes.box_from_df(df)
    plot_box(box_list)
    
    #Create an example
    merge_boxes = [box_list[0]]
    tile_boxes = box_list[1:]
    
    merged_boxes = boxes.merge_boxes(tile_boxes,merge_boxes)
    plot_box(merged_boxes)
    
    assert len(merged_boxes) == 2
    