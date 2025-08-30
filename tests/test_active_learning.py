import math
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from deepforest import active_learning as al


@pytest.fixture
def tmp_paths(tmp_path):
    workdir = tmp_path / "work"
    images_dir = tmp_path / "images"
    workdir.mkdir()
    images_dir.mkdir()
    return workdir, images_dir


@pytest.fixture
def cfg(tmp_paths):
    workdir, images_dir = tmp_paths
    train_csv = workdir / "train.csv"
    val_csv = workdir / "val.csv"
    # Minimal DF-format CSVs
    df = pd.DataFrame([
        {"image_path": str(images_dir / "a.jpg"), "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "label": "tree"},
    ])
    df.to_csv(train_csv, index=False)
    df.to_csv(val_csv, index=False)
    return al.Config(
        workdir=str(workdir),
        images_dir=str(images_dir),
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        classes=["tree", "snag"],
        epochs_per_round=1,
        batch_size=1,
        precision=32,
        device="cpu",
        k_per_round=2,
        score_threshold_pred=0.2,
    )


class DummyModel:
    def __init__(self, batch_size=1, num_classes=2, predict_returns=None):
        self.config = {"train": {}, "val": {}, "batch_size": batch_size, "num_classes": num_classes}
        self._predict_returns = predict_returns or {}
        self.trainer = None

    def use_release(self):
        pass

    def create_trainer(self, trainer):
        self.trainer = trainer

    def eval(self):
        pass

    def predict_image(self, image_path, return_plot=False, score_threshold=0.2):
        # Return a small DF or None based on path
        ret = self._predict_returns.get(image_path)
        if ret is None:
            return None
        return pd.DataFrame(ret)

    def evaluate(self, csv_file, root_dir, iou_threshold, predictions=None):
        # Return a dict like DF evaluate usually does
        return {"val_map": 0.42, "iou_threshold": iou_threshold}


class DummyTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, model):
        # Simulate a training loop having run
        pass


class DummyCheckpoint:
    def __init__(self, dirpath, filename, monitor, mode, save_top_k, save_weights_only, auto_insert_metric_name):
        self.dirpath = dirpath
        self.best_model_path = str(Path(dirpath) / "best.ckpt")
        self.monitor = monitor
        self.mode = mode


class DummyEarlyStopping:
    def __init__(self, monitor, mode, patience):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience


@pytest.fixture(autouse=True)
def patch_lightning_and_df(monkeypatch):
    # Patch deepforest.main.deepforest factory to return DummyModel
    from deepforest import main as df_main

    def factory():
        return DummyModel()

    monkeypatch.setattr(df_main, "deepforest", factory, raising=True)

    # Patch PL Trainer and callbacks used by the module
    import pytorch_lightning as pl
    monkeypatch.setattr(pl, "Trainer", DummyTrainer, raising=True)

    from pytorch_lightning import callbacks as cb
    monkeypatch.setattr(cb, "ModelCheckpoint", DummyCheckpoint, raising=True)
    monkeypatch.setattr(cb, "EarlyStopping", DummyEarlyStopping, raising=True)

    # Also patch top-level imports used inside the module
    monkeypatch.setattr(al, "ModelCheckpoint", DummyCheckpoint, raising=True)
    monkeypatch.setattr(al, "EarlyStopping", DummyEarlyStopping, raising=True)


def test_config_fields_round_trip(cfg):
    assert cfg.classes == ["tree", "snag"]
    assert cfg.epochs_per_round == 1
    assert cfg.k_per_round == 2
    assert cfg.iou_eval == 0.5


def test_seed_function_does_not_error():
    # Should be silent even without CUDA
    al._seed_everything(123)


def test_resolve_device_cpu_when_no_cuda(monkeypatch):
    # Simulate no CUDA
    monkeypatch.setattr(al.torch, "cuda", type("cuda", (), {"is_available": staticmethod(lambda: False)}))
    accelerator, devices = al._resolve_device("auto")
    assert accelerator == "cpu"
    assert devices == 1


def test_read_paths_file_from_list(tmp_paths):
    _, images_dir = tmp_paths
    p1 = str(images_dir / "u1.jpg")
    p2 = str(images_dir / "u2.jpg")
    out = al._read_paths_file([p1, p2])
    assert out == [p1, p2]


def test_read_paths_file_from_text(tmp_path):
    f = tmp_path / "paths.txt"
    f.write_text("a.jpg\n\n# comment\nb.jpg\n", encoding="utf-8")
    # Current helper doesn't strip comments, but should ignore blanks
    out = al._read_paths_file(f)
    assert out == ["a.jpg", "# comment", "b.jpg"]


def test_entropy_empty_preds_returns_logC():
    entropy, n, mean = al._image_entropy_from_predictions(pd.DataFrame(), classes=["a", "b", "c"])
    assert pytest.approx(entropy, rel=1e-6) == math.log(3)
    assert n == 0
    assert mean == 0.0


def test_entropy_simple_distribution():
    df = pd.DataFrame(
        [
            {"label": "a", "score": 0.8},
            {"label": "a", "score": 0.2},
            {"label": "b", "score": 1.0},
        ]
    )
    entropy, n, mean = al._image_entropy_from_predictions(df, classes=["a", "b", "c"])
    # Mass: a=1.0, b=1.0, c=0.0 => probs=[0.5,0.5,0.0]
    assert pytest.approx(entropy, rel=1e-6) == -2 * (0.5 * math.log(0.5))
    assert n == 3
    assert pytest.approx(mean, rel=1e-6) == np.mean([0.8, 0.2, 1.0])


def test_active_learner_initializes_and_attaches_data(cfg):
    learner = al.ActiveLearner(cfg)
    # Data is attached into model.config
    t = learner.model.config["train"]
    v = learner.model.config["val"]
    assert Path(t["csv_file"]).name == Path(cfg.train_csv).name
    assert Path(v["csv_file"]).name == Path(cfg.val_csv).name
    assert Path(t["root_dir"]).name == Path(cfg.images_dir).name


def test_fit_one_round_returns_checkpoint_path(cfg):
    learner = al.ActiveLearner(cfg)
    ckpt = learner.fit_one_round()
    assert str(ckpt).endswith("best.ckpt")
    assert Path(ckpt).parent.name == "checkpoints"


def test_evaluate_returns_metrics(cfg):
    learner = al.ActiveLearner(cfg)
    metrics = learner.evaluate()
    assert "val_map" in metrics
    assert metrics["iou_threshold"] == cfg.iou_eval


def test_predict_images_handles_none_returns(cfg, tmp_paths, monkeypatch):
    workdir, images_dir = tmp_paths
    p1 = str(images_dir / "x.jpg")
    p2 = str(images_dir / "y.jpg")

    # Patch the DummyModel to return None for one path and a small DF for the other
    dm = DummyModel(predict_returns={
        p1: [{"label": "tree", "score": 0.7, "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}],
        # p2 -> None (implicit)
    })

    def factory():
        return dm

    from deepforest import main as df_main
    monkeypatch.setattr(df_main, "deepforest", factory, raising=True)

    learner = al.ActiveLearner(cfg)
    out = learner.predict_images([p1, p2])
    assert set(out.keys()) == {p1, p2}
    assert len(out[p1]) == 1
    # For None, code should create an empty DF with expected columns
    assert list(out[p2].columns) == ["xmin", "ymin", "xmax", "ymax", "label", "score", "image_path"]
    assert len(out[p2]) == 0


def test_select_for_labeling_writes_manifest_and_returns_topk(cfg, tmp_paths, monkeypatch):
    _, images_dir = tmp_paths
    u1 = str(images_dir / "u1.jpg")
    u2 = str(images_dir / "u2.jpg")
    u3 = str(images_dir / "u3.jpg")

    # Create a paths file
    paths_file = Path(cfg.workdir) / "unlabeled.txt"
    paths_file.write_text(f"{u1}\n{u2}\n{u3}\n", encoding="utf-8")

    # Build predictable predictions so entropy differs
    # u1: balanced mass across 2 classes -> higher entropy
    # u2: single-class mass -> lower entropy
    # u3: no preds -> max entropy (log C)
    dm = DummyModel(predict_returns={
        u1: [{"label": "tree", "score": 0.5}, {"label": "snag", "score": 0.5}],
        u2: [{"label": "tree", "score": 1.0}],
        # u3 -> None
    })

    def factory():
        return dm

    from deepforest import main as df_main
    monkeypatch.setattr(df_main, "deepforest", factory, raising=True)

    learner = al.ActiveLearner(cfg)
    topk = learner.select_for_labeling(paths_file, k=2)

    # Manifest file exists
    manifest_path = Path(cfg.workdir) / "acquisition" / "selection_round.csv"
    assert manifest_path.exists()

    # Top-2 should include u3 (empty -> max entropy) and u1 (balanced)
    selected = set(topk["image_path"].tolist())
    assert {u3, u1}.issubset(selected)
