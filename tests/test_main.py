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
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
from .conftest import download_release

# Fixtures for setup
@pytest.fixture()
def common_config():
    return {
        "train_csv": get_data("example.csv"),
        "train_root": os.path.dirname(get_data("example.csv")),
        "val_csv": get_data("example.csv"),
        "val_root": os.path.dirname(get_data("example.csv")),
        "batch_size": 2,
        "epochs": 2,
        "fast_dev_run": True,
        "workers": 0,
        "val_accuracy_interval": 1
    }

@pytest.fixture()
def two_class_m(common_config):
    m = main.deepforest(config_args={"num_classes": 2}, label_dict={"Alive": 0, "Dead": 1})
    m.config["train"]["csv_file"] = get_data("testfile_multi.csv")
    m.config["train"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config["validation"]["csv_file"] = get_data("testfile_multi.csv")
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("testfile_multi.csv"))
    m.config.update(common_config)
    m.create_trainer()
    return m

@pytest.fixture()
def m(download_release, common_config):
    m = main.deepforest()
    m.config.update(common_config)
    m.create_trainer()
    m.use_release(check_release=False)
    return m

@pytest.fixture()
def m_without_release(common_config):
    m = main.deepforest()
    m.config.update(common_config)
    m.create_trainer()
    return m

@pytest.fixture()
def raster_path():
    return get_data(path='OSBS_029.tif')

# Helper function for creating big files
def big_file():
    tmpdir = tempfile.gettempdir()
    csv_file = get_data("OSBS_029.csv")
    image_path = get_data("OSBS_029.png")
    df = pd.read_csv(csv_file)
    big_frame = pd.concat([df.assign(image_path=f"{x}.png") for x in range(3)])
    for x in range(3):
        img = Image.open(os.path.join(os.path.dirname(csv_file), df.image_path.unique()[0]))
        cv2.imwrite(f"{tmpdir}/{x}.png", np.array(img))
    big_frame.to_csv(f"{tmpdir}/annotations.csv")
    return f"{tmpdir}/annotations.csv"

# Test cases
def test_tensorboard_logger(m, tmpdir):
    if importlib.util.find_spec("tensorboard"):
        annotations_file = get_data("testfile_deepforest.csv")
        logger = TensorBoardLogger(save_dir=tmpdir)
        m.config.update({
            "train_csv": annotations_file,
            "train_root": os.path.dirname(annotations_file),
            "fast_dev_run": False,
            "epochs": 2
        })
        m.create_trainer(logger=logger)
        m.trainer.fit(m)
        assert "box_precision" in m.trainer.logged_metrics
        assert "box_recall" in m.trainer.logged_metrics
    else:
        print("TensorBoard is not installed. Skipping test_tensorboard_logger.")

def test_use_bird_release(m):
    imgpath = get_data("AWPE Pigeon Lake 2020 DJI_0005.JPG")
    m.use_bird_release()
    boxes = m.predict_image(path=imgpath)
    assert not boxes.empty

def test_train_empty(m, tmpdir):
    empty_csv = pd.DataFrame({
        "image_path": ["OSBS_029.png", "OSBS_029.tif"],
        "xmin": [0, 10], "xmax": [0, 20],
        "ymin": [0, 20], "ymax": [0, 30],
        "label": ["Tree", "Tree"]
    })
    empty_csv.to_csv(f"{tmpdir}/empty.csv")
    m.config["train_csv"] = f"{tmpdir}/empty.csv"
    m.create_trainer()
    m.trainer.fit(m)

def test_validation_step(m):
    m.trainer = None
    before = copy.deepcopy(m)
    m.create_trainer()
    m.trainer.validate(m)
    for p1, p2 in zip(before.named_parameters(), m.named_parameters()):
        assert torch.equal(p1[1], p2[1])

@pytest.mark.parametrize("architecture", ["retinanet", "FasterRCNN"])
def test_train_single(m_without_release, architecture):
    m_without_release.config["architecture"] = architecture
    m_without_release.create_model()
    m_without_release.config["fast_dev_run"] = False
    m_without_release.create_trainer()
    m_without_release.trainer.fit(m_without_release)

def test_train_preload_images(m):
    m.create_trainer()
    m.config["train"]["preload_images"] = True
    m.trainer.fit(m)

def test_train_multi(two_class_m):
    two_class_m.trainer.fit(two_class_m)

def test_train_no_validation(m):
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
    assert set(prediction.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"}

def test_predict_image_fromarray(m):
    image_path = get_data(path="2019_YELL_2_528000_4978000_image_crop2.png")
    with pytest.raises(TypeError):
        image = Image.open(image_path)
        prediction = m.predict_image(image=image)
    image = np.array(Image.open(image_path).convert("RGB"))
    with pytest.warns(UserWarning, match="Image type is uint8, transforming to float32"):
        prediction = m.predict_image(image=image)
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score"}

def test_predict_return_plot(m):
    image = np.array(Image.open(get_data(path="2019_YELL_2_528000_4978000_image_crop2.png"))).astype('float32')
    plot = m.predict_image(image=image, return_plot=True)
    assert isinstance(plot, np.ndarray)

def test_predict_big_file(m, tmpdir):
    m.create_trainer()
    csv_file = big_file()
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file=csv_file, root_dir=os.path.dirname(csv_file), savedir=tmpdir)
    assert set(df.columns) == {'label', 'score', 'image_path', 'geometry', "xmin", "ymin", "xmax", "ymax"}
    printed_plots = glob.glob(f"{tmpdir}/*.png")
    assert len(printed_plots) == len(original_file.image_path.unique())

def test_predict_small_file(m, tmpdir):
    csv_file = get_data("OSBS_029.csv")
    original_file = pd.read_csv(csv_file)
    df = m.predict_file(csv_file, root_dir=os.path.dirname(csv_file), savedir=tmpdir)
    assert set(df.columns) == {'label', 'score', 'image_path', 'geometry', "xmin", "ymin", "xmax", "ymax"}
    printed_plots = glob.glob(f"{tmpdir}/*.png")
    assert len(printed_plots) == len(original_file.image_path.unique())

@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_dataloader(m, batch_size, raster_path):
    m.config["batch_size"] = batch_size
    tile = np.array(Image.open(raster_path))
    ds = dataset.TileDataset(tile=tile, patch_overlap=0.1, patch_size=100)
    dl = m.predict_dataloader(ds)
    batch = next(iter(dl))
    assert batch.shape[0] == batch_size

def test_predict_tile(m, raster_path):
    m.create_model()
    m.create_trainer()
    prediction = m.predict_tile(raster_path=raster_path, patch_size=300, patch_overlap=0.1, return_plot=False)
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"}
    assert not prediction.empty

@pytest.mark.parametrize("patch_overlap", [0.1, 0])
def test_predict_tile_from_array(m, patch_overlap, raster_path):
    image = np.array(Image.open(raster_path))
    m.create_trainer()
    prediction = m.predict_tile(image=image, patch_size=300, patch_overlap=patch_overlap, return_plot=False)
    assert isinstance(prediction, pd.DataFrame)
    assert set(prediction.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"}
    assert not prediction.empty

def test_predict_tile_return_plot(m, raster_path):
    m.create_trainer()
    plot = m.predict_tile(raster_path=raster_path, patch_size=300, patch_overlap=0.1, return_plot=True)
    assert isinstance(plot, np.ndarray)

def test_predict_generator(m):
    m.create_trainer()
    raster_path = get_data(path="OSBS_029.tif")
    generator = m.predict_tile_generator(raster_path=raster_path, patch_size=300, patch_overlap=0.1)
    for df in generator:
        assert set(df.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"}
        assert not df.empty

def test_predict_generator_from_array(m, raster_path):
    m.create_trainer()
    image = np.array(Image.open(raster_path))
    generator = m.predict_tile_generator(image=image, patch_size=300, patch_overlap=0.1)
    for df in generator:
        assert set(df.columns) == {"xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"}
        assert not df.empty

@pytest.mark.parametrize("augmentation", [None, A.Compose([A.HorizontalFlip(p=1), ToTensorV2()])])
def test_create_dataset(m, augmentation):
    df = pd.read_csv(get_data("example.csv"))
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(os.path.dirname(get_data("example.csv")), x))
    m.config["train"]["csv_file"] = get_data("example.csv")
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    ds = m.create_dataset(csv_file=m.config["train"]["csv_file"], root_dir=m.config["train"]["root_dir"], transforms=augmentation)
    sample = ds[0]
    assert isinstance(sample, dict)
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["bboxes"], list)
    assert isinstance(sample["labels"], list)
    if augmentation is not None:
        assert sample["image"].shape[0] == 3  # Check for ToTensorV2 augmentation

def test_train_recover(m, tmpdir):
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=1, monitor="val_box_mean_average_precision")
    m.config.update({
        "checkpoint_callback": checkpoint_callback,
        "train_csv": get_data("example.csv"),
        "train_root": os.path.dirname(get_data("example.csv")),
        "epochs": 2,
        "fast_dev_run": False
    })
    m.create_trainer()
    m.trainer.fit(m)
    # Simulate recovery by resetting trainer and reloading from checkpoint
    m.trainer = None
    m.create_trainer()
    ckpt_path = glob.glob(os.path.join(tmpdir, "*.ckpt"))[0]
    m.trainer.fit(m, ckpt_path=ckpt_path)

def test_train_evaluate(m):
    m.config.update({
        "train_csv": get_data("example.csv"),
        "train_root": os.path.dirname(get_data("example.csv")),
        "fast_dev_run": True
    })
    m.create_trainer()
    m.trainer.fit(m)
    val_results = m.trainer.validate(m)
    assert isinstance(val_results, list)
    assert len(val_results) > 0

def test_save_load_checkpoint(m, tmpdir):
    checkpoint_callback = ModelCheckpoint(dirpath=tmpdir, save_top_k=1, monitor="val_box_mean_average_precision")
    m.create_trainer(callbacks=[checkpoint_callback])
    m.trainer.fit(m)
    ckpt_path = checkpoint_callback.best_model_path
    m_loaded = model.load_from_checkpoint(ckpt_path)
    assert isinstance(m_loaded, model)

def test_load_model(m):
    m.model = None
    m.create_model()
    assert m.model is not None

@pytest.mark.parametrize("optimizer", ["Adam", "SGD"])
def test_optimizer_choice(m_without_release, optimizer):
    m_without_release.config["optimizer"] = optimizer
    m_without_release.create_trainer()
    assert m_without_release.trainer is not None

def test_custom_logger(m, tmpdir):
    logger = TensorBoardLogger(save_dir=tmpdir)
    m.create_trainer(logger=logger)
    m.trainer.fit(m)
    assert os.path.exists(logger.log_dir)

def test_large_data(m):
    m.config["batch_size"] = 16
    m.config["train"]["csv_file"] = big_file()
    m.create_trainer()
    m.trainer.fit(m)

def test_dataloader_augmentation(m):
    transforms = A.Compose([A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.5)])
    ds = m.create_dataset(transforms=transforms)
    dl = m.create_dataloader(dataset=ds, shuffle=True)
    batch = next(iter(dl))
    assert isinstance(batch, dict)
    assert "image" in batch
    assert "bboxes" in batch
    assert "labels" in batch

def test_evaluation_metrics(m):
    m.config.update({
        "train_csv": get_data("example.csv"),
        "train_root": os.path.dirname(get_data("example.csv")),
        "fast_dev_run": True
    })
    m.create_trainer()
    m.trainer.fit(m)
    metrics = m.trainer.logged_metrics
    assert "val_box_precision" in metrics
    assert "val_box_recall" in metrics
