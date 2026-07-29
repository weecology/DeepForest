import os

from deepforest import get_data, main


def test_default_batch_sizes():
    model = main.deepforest(
        config_args={
            "model": {"name": None},
            "num_classes": 1,
            "label_dict": {"Tree": 0},
        }
    )
    assert model.config.train_batch_size == 2
    assert model.config.predict_batch_size == 8


def test_custom_batch_sizes():
    custom_m = main.deepforest(
        config_args={
            "model": {"name": None},
            "num_classes": 1,
            "label_dict": {"Tree": 0},
            "train_batch_size": 4,
            "predict_batch_size": 16,
        }
    )
    assert custom_m.config.train_batch_size == 4
    assert custom_m.config.predict_batch_size == 16


def test_train_dataloader_batch_size():
    model = main.deepforest(
        config_args={
            "model": {"name": None},
            "num_classes": 1,
            "label_dict": {"Tree": 0},
        }
    )
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    model.config.train.csv_file = csv_file
    model.config.train.root_dir = root_dir
    loader = model.train_dataloader()
    assert loader.batch_size == model.config.train_batch_size


def test_val_dataloader_batch_size():
    model = main.deepforest(
        config_args={
            "model": {"name": None},
            "num_classes": 1,
            "label_dict": {"Tree": 0},
        }
    )
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    model.config.validation.csv_file = csv_file
    model.config.validation.root_dir = root_dir
    loader = model.val_dataloader()
    assert loader.batch_size == model.config.predict_batch_size


def test_predict_dataloader_batch_size():
    from deepforest.datasets import prediction

    model = main.deepforest(
        config_args={
            "model": {"name": None},
            "num_classes": 1,
            "label_dict": {"Tree": 0},
        }
    )
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)
    ds = prediction.FromCSVFile(csv_file=csv_file, root_dir=root_dir)
    loader = model.predict_dataloader(ds)
    assert loader.batch_size == model.config.predict_batch_size
