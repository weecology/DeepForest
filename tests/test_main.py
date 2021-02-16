#test main
import os
import glob
import pytest
import pandas as pd
import numpy as np

from skimage import io

from deepforest import main
from deepforest import get_data
        
@pytest.fixture()
def m():
    csv_file = get_data("example.csv")
    m = main.deepforest()
    m.config["epochs"] = 1
    m.config["batch_size"] = 2
    m.create_model()
    m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), augment=True)
    
    return m

@pytest.fixture()
def trained_model():
    csv_file = get_data("example.csv")
    m = main.deepforest()
    m.config["epochs"] = 1
    m.config["batch_size"] = 2
    m.create_model()
    m.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), augment=True)
    m.train(debug=True)        
    
    return m

def test_main():
    from deepforest import main

def test_train(m):
    m.train(debug=True)

def test_predict_image_empty(m):
    image = np.random.random((400,400,3))
    prediction = m.predict_image(image = image)
    assert prediction is None
    
def test_predict_image_fromfile(trained_model):
    path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    prediction = trained_model.predict_image(path = path)
    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","scores"}

def test_predict_image_fromarray(trained_model):
    image = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    image = io.imread(image)
    prediction = trained_model.predict_image(image = image)
    
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","scores"}

def test_predict_return_plot(trained_model):
    image = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    image = io.imread(image)
    plot = trained_model.predict_image(image = image, return_plot=True)
    assert not isinstance(plot, pd.DataFrame)
            
def test_predict_file(trained_model, tmpdir):
    csv_file = get_data("example.csv")
    df = trained_model.predict_file(csv_file, root_dir = os.path.dirname(csv_file), save_dir=tmpdir)
    assert set(df.columns) == {"xmin","ymin","xmax","ymax","label","scores","image_path"}
    
    printed_plots = glob.glob("{}/*.png".format(tmpdir))
    assert len(printed_plots) == 1

def test_predict_tile(trained_model):
    #test raster prediction 
    raster_path = get_data(path= 'OSBS_029.tif')
    prediction = trained_model.predict_tile(raster_path = raster_path,
                                            patch_size = 300,
                                            patch_overlap = 0.5,
                                            return_plot = False)
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin","ymin","xmax","ymax","label","score"}
    assert not prediction.empty

    #test soft-nms method
    soft_nms_pred = trained_model.predict_tile(raster_path = raster_path,
                                            patch_size = 300,
                                            patch_overlap = 0.5,
                                            return_plot = False,
                                            use_soft_nms =True)
    assert isinstance(soft_nms_pred, pd.DataFrame)
    assert set(soft_nms_pred.columns) == {"xmin","ymin","xmax","ymax","label","score"}
    assert not soft_nms_pred.empty

    #test predict numpy image
    image = io.imread(raster_path)
    prediction = trained_model.predict_tile(image = image,
                                patch_size = 300,
                                patch_overlap = 0.5,
                                return_plot = False)
    assert not prediction.empty

    # Test no non-max suppression
    prediction = trained_model.predict_tile(raster_path = raster_path,
                                       patch_size=300,
                                       patch_overlap=0,
                                       return_plot=False)
    assert not prediction.empty

def test_train_callbacks(m, capsys):
    
    class fake_callback():
        def on_epoch_end(self, epoch):
            print("Finished epoch {}")
        def on_fit_end(self):
            print("Done training")
    captured = capsys.readouterr()    
    m.train(debug=True, callbacks=[fake_callback()])
    assert captured.out == "Finished epoch 1\n Done Training\n"
    
def test_evaluate(m):
    csv_file = get_data("example.csv")
    root_dir = os.path.dirname(csv_file)
    precision, recall = m.evaluate(csv_file, root_dir, iou_threshold = 0.5)
    
    