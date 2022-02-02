#Test visualize
from deepforest import visualize
from deepforest import main
from deepforest import get_data
from deepforest import utilities
import os
import pytest
import numpy as np

@pytest.fixture()
def visualize_config():
    visualize_config = utilities.read_config("deepforest_config.yml")
    visualize_config["train"]["csv_file"] = get_data("example.csv") 
    visualize_config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    visualize_config["train"]["fast_dev_run"] = True
    visualize_config["batch_size"] = 2
    visualize_config["workers"] = 0
    
    visualize_config["validation"]["csv_file"] = get_data("example.csv") 
    visualize_config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
        
    return visualize_config

def test_format_boxes(visualize_config):
    m = main.deepforest()
    m.use_release(check_release=False)     
    m.config = visualize_config
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        assert list(target_df.columns.values) == ["xmin","ymin","xmax","ymax","label"]
        assert not target_df.empty
        
#Test different color labels
@pytest.mark.parametrize("label",[0,1,20])
def test_plot_predictions(tmpdir,label, visualize_config):
    m = main.deepforest()
    m.config = visualize_config
    m.use_release(check_release=False) 
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        image = np.array(image)[:,:,::-1]
        image = np.rollaxis(image,0,3)
        target_df.label = label
        image = visualize.plot_predictions(image, target_df)

        assert image.dtype == "uint8"
        
def test_plot_prediction_dataframe(tmpdir, visualize_config):
    m = main.deepforest()
    m.config = visualize_config
    m.use_release(check_release=False) 
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    for path, image, target in zip(paths, images, targets):
        target_df = visualize.format_boxes(target, scores=False)
        target_df["image_path"] = path
        filenames = visualize.plot_prediction_dataframe(df=target_df,savedir=tmpdir, root_dir=m.config["validation"]["root_dir"])
        
    assert all([os.path.exists(x) for x in filenames])
        
def test_plot_predictions_and_targets(tmpdir, visualize_config):
    m = main.deepforest()
    m.config = visualize_config
    m.use_release(check_release=False)
    ds = m.val_dataloader()
    batch = next(iter(ds))
    paths, images, targets = batch
    m.model.eval()    
    predictions = m.model(images)    
    for path, image, target, prediction in zip(paths, images, targets, predictions):
        image = image.permute(1,2,0)
        save_figure_path = visualize.plot_prediction_and_targets(image, prediction, target, image_name=os.path.basename(path), savedir=tmpdir)
        assert os.path.exists(save_figure_path)
    