# test main
import os
import glob
import pytest
import pandas as pd
import numpy as np
import cv2
import shutil
import torch
import tempfile
import copy
import importlib.util

import albumentations as A
from albumentations.pytorch import ToTensorV2

from deepforest import main, get_data, dataset, model
from deepforest.visualize import format_geometry
from deepforest.utilities import read_file

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import geopandas as gpd
from PIL import Image
import shapely

# Import release model from global script to avoid thrasing github during testing.
# Just download once.
from .conftest import download_release


@pytest.fixture()
def two_class_m():
    m = main.deepforest(config_args={"num_classes": 2},
                        label_dict={
                            "Alive": 0,
                            "Dead": 1
                        })
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv")
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2

    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv")
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["validation"]["val_accuracy_interval"] = 1

    m.create_trainer()

    return m


@pytest.fixture()
def m(download_release):
    m = main.deepforest()
    m.config["train"]["csv_file"] = get_data("example.csv")
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2

    m.config["validation"]["csv_file"] = get_data("example.csv")
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["workers"] = 0
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["train"]["epochs"] = 2

    m.create_trainer()
    m.load_model("weecology/deepforest-tree")
    return m


@pytest.fixture()
def m_without_release():
    m = main.deepforest()
    m.config["train"]["csv_file"] = get_data("example.csv")
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["train"]["fast_dev_run"] = True
    m.config["batch_size"] = 2

    m.config["validation"]["csv_file"] = get_data("example.csv")
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["workers"] = 0
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["train"]["epochs"] = 2

    m.create_trainer()
    return m


@pytest.fixture()
def path():
    return get_data(path='OSBS_029.tif')


def big_file():
    tmpdir = tempfile.gettempdir()
    csv_file = get_data("OSBS_029.csv")
    image_path = get_data("OSBS_029.png")
    df = pd.read_csv(csv_file)

    big_frame = []
    for x in range(3):
        img = Image.open("{}/{}".format(os.path.dirname(csv_file),
                                        df.image_path.unique()[0]))
        cv2.imwrite("{}/{}.png".format(tmpdir, x), np.array(img))
        new_df = df.copy()
        new_df.image_path = "{}.png".format(x)
        big_frame.append(new_df)

    big_frame = pd.concat(big_frame)
    big_frame.to_csv("{}/annotations.csv".format(tmpdir))

    return "{}/annotations.csv".format(tmpdir)


def test_tensorboard_logger(m, tmpdir):
    # Check if TensorBoard is installed
    if importlib.util.find_spec("tensorboard"):
        # Create model trainer and fit model
        annotations_file = get_data("testfile_deepforest.csv")
        logger = TensorBoardLogger(save_dir=tmpdir)
        m.config["train"]["csv_file"] = annotations_file
        m.config["train"]["root_dir"] = os.path.dirname(annotations_file)
        m.config["train"]["fast_dev_run"] = False
        m.config["validation"]["csv_file"] = annotations_file
        m.config["validation"]["root_dir"] = os.path.dirname(annotations_file)
        m.config["val_accuracy_interval"] = 1
        m.config["train"]["epochs"] = 2

        m.create_trainer(logger=logger, limit_train_batches=1, limit_val_batches=1)
        m.trainer.fit(m)

        assert m.trainer.logged_metrics["box_precision"]
        assert m.trainer.logged_metrics["box_recall"]
    else:
        print("TensorBoard is not installed. Skipping test_tensorboard_logger.")


def test_use_bird_release(m):
    imgpath = get_data("AWPE Pigeon Lake 2020 DJI_0005.JPG")
    m.load_model("Weecology/deepforest-bird")
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty

def test_load_model(m):
    imgpath = get_data("OSBS_029.png")
    m.load_model('ethanwhite/df-test')
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty


def test_train_empty(m, tmpdir):
    empty_csv = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.tif"],
        "xmin": [0, 10],
        "xmax": [0, 20],
        "ymin": [0, 20],
        "ymax": [0, 30],
        "label": ["Tree", "Tree"]
    })
    empty_csv.to_csv("{}/empty.csv".format(tmpdir))
    m.config["train"]["csv_file"] = "{}/empty.csv".format(tmpdir)
    m.config["batch_size"] = 2
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)

def test_train_with_empty_validation(m, tmpdir):
    empty_csv = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.tif"],
        "xmin": [0, 10],
        "xmax": [0, 20],
        "ymin": [0, 20],
        "ymax": [0, 30],
        "label": ["Tree", "Tree"]
    })
    empty_csv.to_csv("{}/empty.csv".format(tmpdir))
    m.config["train"]["csv_file"] = "{}/empty.csv".format(tmpdir)
    m.config["validation"]["csv_file"] = "{}/empty.csv".format(tmpdir)
    m.config["batch_size"] = 2
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)
    m.trainer.validate(m)


def test_validation_step(m):
    val_dataloader = m.val_dataloader()
    batch = next(iter(val_dataloader))
    m.predictions = []
    val_loss = m.validation_step(batch, 0)
    assert val_loss != 0

def test_validation_step_empty():
    """If the model returns an empty prediction, the metrics should not fail"""
    m = main.deepforest()
    m.config["validation"]["csv_file"] = get_data("example.csv")
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.create_trainer()

    val_dataloader = m.val_dataloader()
    batch = next(iter(val_dataloader))
    m.predictions = []
    val_loss = m.validation_step(batch, 0)
    assert len(m.predictions) == 1
    assert m.predictions[0].xmin.isna().all()
    assert m.iou_metric.compute()["iou"] == 0

def test_validate(m):
    m.trainer = None
    # Turn off trainer to test copying on some linux devices.
    before = copy.deepcopy(m)
    m.create_trainer()
    m.trainer.validate(m)
    # assert no weights have changed
    for p1, p2 in zip(before.named_parameters(), m.named_parameters()):
        assert p1[1].ne(p2[1]).sum() == 0


# Test train with each architecture
@pytest.mark.parametrize("architecture", ["retinanet", "FasterRCNN"])
def test_train_single(m_without_release, architecture):
    m_without_release.config["architecture"] = architecture
    m_without_release.create_model()
    m_without_release.config["train"]["fast_dev_run"] = False
    m_without_release.create_trainer(limit_train_batches=1)
    m_without_release.trainer.fit(m_without_release)


def test_train_preload_images(m):
    m.create_trainer(fast_dev_run=True)
    m.config["train"]["preload_images"] = True
    m.trainer.fit(m)


def test_train_geometry_column(m, tmpdir):
    """Test that training works with a geometry column from a shapefile"""

    # Get the source data
    df = read_file(get_data("OSBS_029.csv"))
    df["label"] = "Tree"

    # Create geodataframe with box geometries
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            shapely.geometry.box(xmin, ymin, xmax, ymax)
            for xmin, ymin, xmax, ymax in zip(df.xmin, df.ymin, df.xmax, df.ymax)
        ])

    # Add image path
    gdf["image_path"] = "OSBS_029.tif"

    gdf = gdf[["label", "geometry", "image_path"]]

    # Save to temporary shapefile with only a geometry and label column
    temp_shp = os.path.join(tmpdir, "temp_trees.shp")
    gdf.to_file(temp_shp)

    # Read shapefile using utilities
    df = read_file(temp_shp, root_dir=os.path.dirname(get_data("OSBS_029.tif")))
    df.to_csv(os.path.join(tmpdir, "OSBS_029.csv"), index=False)

    # Train model
    m.config["train"]["csv_file"] = os.path.join(tmpdir, "OSBS_029.csv")
    m.config["train"]["root_dir"] = os.path.dirname(get_data("OSBS_029.tif"))
    m.create_trainer(fast_dev_run=True)
    m.trainer.fit(m)

def test_train_multi(two_class_m):
    two_class_m.create_trainer(fast_dev_run=True)
    two_class_m.trainer.fit(two_class_m)

def test_train_no_validation(m):
    m.config["train"]["fast_dev_run"] = False
    m.config["validation"]["csv_file"] = None
    m.config["validation"]["root_dir"] = None
    m.create_trainer(limit_train_batches=1)
    m.trainer.fit(m)


def test_predict_image_empty(m):
    image = np.random.random((400, 400, 3)).astype("float32")
    prediction = m.predict_image(image=image)

    assert prediction is None


def test_predict_image_fromfile(m):
    path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    prediction = m.predict_image(path=path)

    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {
        "xmin", "ymin", "xmax", "ymax", "label", "score", "image_path", "geometry"
    }


def test_predict_image_fromarray(m):
    image_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")

    # assert error of dtype
    with pytest.raises(TypeError):
        image = Image.open(image_path)
        prediction = m.predict_image(image=image)

    image = np.array(Image.open(image_path).convert("RGB"))
    with pytest.warns(UserWarning, match="Image type is uint8, transforming to float32"):
        prediction = m.predict_image(image=image)

    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "geometry"}
    assert not hasattr(prediction, 'root_dir')

def test_predict_big_file(m, tmpdir):
    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()
    csv_file = big_file()
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file=csv_file,
                        root_dir=os.path.dirname(csv_file))
    assert set(df.columns) == {
        'label', 'score', 'image_path', 'geometry', "xmin", "ymin", "xmax", "ymax"
    }

def test_predict_small_file(m, tmpdir):
    csv_file = get_data("OSBS_029.csv")
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file, root_dir=os.path.dirname(csv_file))
    assert set(df.columns) == {
        'label', 'score', 'image_path', 'geometry', "xmin", "ymin", "xmax", "ymax"
    }

@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_dataloader(m, batch_size, path):
    m.config["batch_size"] = batch_size
    tile = np.array(Image.open(path))
    ds = dataset.TileDataset(tile=tile, patch_overlap=0.1, patch_size=100)
    dl = m.predict_dataloader(ds)
    batch = next(iter(dl))
    batch.shape[0] == batch_size


def test_predict_tile_empty(path):
    # Random weights
    m = main.deepforest()
    predictions = m.predict_tile(path=path, patch_size=300, patch_overlap=0)
    assert predictions is None

def test_predict_tile(m, path):
    m.create_model()
    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()

    prediction = m.predict_tile(path=path,
                               patch_size=300,
                               patch_overlap=0.1)

    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {
        "xmin", "ymin", "xmax", "ymax", "label", "score", "image_path", "geometry"
    }
    assert not prediction.empty

def test_predict_tile_from_array(m, path):
    # test predict numpy image
    image = np.array(Image.open(path))
    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()
    prediction = m.predict_tile(image=image,
                               patch_size=300)

    assert not prediction.empty

def test_predict_tile_with_crop_model(m, config):
    path = get_data("SOAP_061.png")
    patch_size = 400
    patch_overlap = 0.05
    # Set up the crop model
    crop_model = model.CropModel(num_classes=2)
    # Call the predict_tile method with the crop_model
    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                           patch_size=patch_size,
                           patch_overlap=patch_overlap,
                           crop_model=crop_model)

    # Assert the result
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "xmin", "ymin", "xmax", "ymax", "label", "score", "cropmodel_label", "geometry",
        "cropmodel_score", "image_path"
    }

def test_predict_tile_with_crop_model_empty():
    """If the model return is empty, the crop model should return an empty dataframe"""
    path = get_data("SOAP_061.png")
    m = main.deepforest()
    patch_size = 400
    patch_overlap = 0.05
    # Set up the crop model
    crop_model = model.CropModel(num_classes=2)
    # Call the predict_tile method with the crop_model
    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                           patch_size=patch_size,
                           patch_overlap=patch_overlap,
                           crop_model=crop_model)

    # Assert the result
    assert result is None or result.empty

def test_predict_tile_with_multiple_crop_models(m, config):
    path = get_data("SOAP_061.png")
    patch_size = 400
    patch_overlap = 0.05

    # Create multiple crop models
    crop_model = [model.CropModel(num_classes=2), model.CropModel(num_classes=3)]

    # Call predict_tile with multiple crop models
    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                           patch_size=patch_size,
                           patch_overlap=patch_overlap,
                           crop_model=crop_model)

    # Assert result type
    assert isinstance(result, pd.DataFrame)

    # Check column names dynamically for multiple crop models
    expected_cols = {"xmin", "ymin", "xmax", "ymax", "label", "score", "geometry", "image_path"}
    for i in range(len(crop_model)):
        expected_cols.add(f"cropmodel_label_{i}")
        expected_cols.add(f"cropmodel_score_{i}")

    assert set(result.columns) == expected_cols
    assert not result.empty

def test_predict_tile_with_multiple_crop_models_empty():
    """If no predictions are made, result should be empty"""
    path = get_data("SOAP_061.png")
    m = main.deepforest()
    patch_size = 400
    patch_overlap = 0.05

    # Create multiple crop models
    crop_model_1 = model.CropModel(num_classes=2)
    crop_model_2 = model.CropModel(num_classes=3)

    m.config["train"]["fast_dev_run"] = False
    m.create_trainer()
    result = m.predict_tile(path=path,
                           patch_size=patch_size,
                           patch_overlap=patch_overlap,
                           crop_model=[crop_model_1, crop_model_2])

    assert result is None or result.empty  # Ensure empty result is handled properly

def test_batch_prediction(m, path):
    # Prepare input data
    tile = np.array(Image.open(path))
    ds = dataset.TileDataset(tile=tile, patch_overlap=0.1, patch_size=300)
    dl = DataLoader(ds, batch_size=3)

    # Perform prediction
    predictions = []
    for batch in dl:
        prediction = m.predict_batch(batch)
        predictions.append(prediction)

    # Check results
    assert len(predictions) == len(dl)
    for batch_pred in predictions:
        for image_pred in batch_pred:
            assert isinstance(image_pred, pd.DataFrame)
            assert "label" in image_pred.columns
            assert "score" in image_pred.columns
            assert "geometry" in image_pred.columns

def test_batch_inference_consistency(m, path):
    tile = np.array(Image.open(path))
    ds = dataset.TileDataset(tile=tile, patch_overlap=0.1, patch_size=300)
    dl = DataLoader(ds, batch_size=4)

    batch_predictions = []
    for batch in dl:
        prediction = m.predict_batch(batch)
        batch_predictions.extend(prediction)

    single_predictions = []
    for image in ds:
        image = image.permute(1,2,0).numpy() * 255
        prediction = m.predict_image(image=image)
        single_predictions.append(prediction)

    batch_df = pd.concat(batch_predictions, ignore_index=True)
    single_df = pd.concat(single_predictions, ignore_index=True)

    # Make all xmin, ymin, xmax, ymax integers
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        batch_df[col] = batch_df[col].astype(int)
        single_df[col] = single_df[col].astype(int)
    pd.testing.assert_frame_equal(batch_df[["xmin", "ymin", "xmax", "ymax"]], single_df[["xmin", "ymin", "xmax", "ymax"]], check_dtype=False)


def test_epoch_evaluation_end(m):
    preds = [{
        'boxes': torch.tensor([
            [690.3572, 902.9113, 781.1031, 996.5151],
            [998.1990, 655.7919, 172.4619, 321.8518]
        ]),
        'scores': torch.tensor([
            0.6740, 0.6625
        ]),
        'labels': torch.tensor([
            0, 0
        ])
    }]
    targets = preds

    m.iou_metric.update(preds, targets)
    m.mAP_metric.update(preds, targets)

    boxes = format_geometry(preds[0])
    boxes["image_path"] = "test"
    m.predictions = [boxes]
    m.on_validation_epoch_end()

def test_epoch_evaluation_end_empty(m):
    """If the model returns an empty prediction, the metrics should not fail"""
    preds = [{
        'boxes': torch.zeros((1, 4)),
        'scores': torch.zeros(1),
        'labels': torch.zeros(1, dtype=torch.int64)
    }]
    targets = preds

    m.iou_metric.update(preds, targets)
    m.mAP_metric.update(preds, targets)

    boxes = format_geometry(preds[0])
    boxes["image_path"] = "test"
    m.predictions = [boxes]
    m.on_validation_epoch_end()
def test_empty_frame_accuracy_all_empty_with_predictions(m, tmpdir):
    """Test empty frame accuracy when all frames are empty but model predicts objects.
    The accuracy should be 0 since model incorrectly predicts objects in empty frames."""
    # Create ground truth with all empty frames
    ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    # Set all xmin, ymin, xmax, ymax to 0 to mark as empty
    ground_df.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = 0
    ground_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    m.config["validation"]["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    m.create_trainer()
    results = m.trainer.validate(m)
    assert results[0]["empty_frame_accuracy"] == 0

def test_empty_frame_accuracy_mixed_frames_with_predictions(m, tmpdir):
    """Test empty frame accuracy with a mix of empty and non-empty frames.
    Model predicts objects in all frames, so accuracy for empty frames should be 0."""
    # Create ground truth with mix of empty and non-empty frames
    tree_ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    empty_ground_df = pd.DataFrame({
        "image_path": ["AWPE Pigeon Lake 2020 DJI_0005.JPG"],
        "xmin": [0],
        "ymin": [0], 
        "xmax": [0],
        "ymax": [0],
        "label": ["Tree"]
    })

    ground_df = pd.concat([tree_ground_df, empty_ground_df])

    # Save the ground truth to a temporary file  
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    m.config["validation"]["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    m.create_trainer()
    results = m.trainer.validate(m)
    assert results[0]["empty_frame_accuracy"] == 0

def test_empty_frame_accuracy_without_predictions(tmpdir):
    """Create a ground truth with empty frames, the accuracy should be 1 with a random model"""
    m = main.deepforest()
    # Create ground truth with empty frames
    ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    # Set all xmin, ymin, xmax, ymax to 0
    ground_df.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = 0
    ground_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    m.config["validation"]["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    m.create_trainer()
    results = m.trainer.validate(m)
    assert results[0]["empty_frame_accuracy"] == 1

def test_multi_class_with_empty_frame_accuracy_without_predictions(two_class_m, tmpdir):
    """Create a ground truth with empty frames, the accuracy should be 1 with a random model"""
    # Create ground truth with empty frames
    ground_df = pd.read_csv(get_data("testfile_deepforest.csv"))
    # Set all xmin, ymin, xmax, ymax to 0
    ground_df.loc[:, ["xmin", "ymin", "xmax", "ymax"]] = 0
    ground_df.drop_duplicates(subset=["image_path"], keep="first", inplace=True)
    ground_df.loc[:, "label"] = "Alive"

    # Merge with a multi class ground truth
    multi_class_df = pd.read_csv(get_data("testfile_multi.csv"))
    ground_df = pd.concat([ground_df, multi_class_df])

    # Save the ground truth to a temporary file
    ground_df.to_csv(tmpdir.strpath + "/ground_truth.csv", index=False)
    two_class_m.config["validation"]["csv_file"] = tmpdir.strpath + "/ground_truth.csv"
    two_class_m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_deepforest.csv"))

    two_class_m.create_trainer()
    results = two_class_m.trainer.validate(two_class_m)
    assert results[0]["empty_frame_accuracy"] == 1

def test_evaluate_on_epoch_interval(m):
    m.config["validation"]["val_accuracy_interval"] = 1
    m.config["train"]["epochs"] = 1
    m.create_trainer()
    m.trainer.fit(m)
    assert m.trainer.logged_metrics["box_precision"]
    assert m.trainer.logged_metrics["box_recall"]

def test_set_labels_updates_mapping(m):
    new_mapping = {"Object": 0}
    m.set_labels(new_mapping)

    # Verify that the label dictionary has been updated.
    assert m.label_dict == new_mapping

    # Verify that the inverse mapping is correctly computed.
    expected_inverse = {0: "Object"}
    assert m.numeric_to_label_dict == expected_inverse

def test_set_labels_invalid_length(m): # Expect a ValueError when setting an invalid label mapping.
    # This mapping has two entries, which should be invalid since m.config["num_classes"] is 1.
    invalid_mapping = {"Object": 0, "Extra": 1}
    with pytest.raises(ValueError):
        m.set_labels(invalid_mapping)
