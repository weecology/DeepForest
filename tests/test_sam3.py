import sys
import types

import numpy as np
import pytest
import shapely
import torch

from deepforest import get_data, utilities
from deepforest.main import deepforest
from deepforest.model import Sam3PolygonModel


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeSam3Processor:
    @classmethod
    def from_pretrained(cls, model_name, token=None):
        return cls()

    def __call__(
        self,
        images=None,
        text=None,
        input_boxes=None,
        input_boxes_labels=None,
        return_tensors="pt",
    ):
        _ = (text, input_boxes_labels, return_tensors)
        if input_boxes is None or len(input_boxes) == 0:
            raise ValueError("Fake processor expects input_boxes")
        return _FakeBatch(
            {
                "input_boxes": input_boxes,
                "original_sizes": torch.tensor([[images.height, images.width]]),
            }
        )

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=None,
    ):
        _ = (outputs, threshold, mask_threshold)
        height, width = target_sizes[0]
        boxes = self._latest_boxes
        masks = []
        scores = []
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 <= x1:
                x2 = min(width, x1 + 1)
            if y2 <= y1:
                y2 = min(height, y1 + 1)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            # Carve one corner so geometry is non-rectangular (polygon, not box)
            corner_h = max(1, (y2 - y1) // 3)
            corner_w = max(1, (x2 - x1) // 3)
            mask[y1 : y1 + corner_h, x1 : x1 + corner_w] = 0
            masks.append(mask)
            scores.append(0.9)
        return [{"masks": masks, "scores": scores}]

    @property
    def _latest_boxes(self):
        return self._cached_boxes

    @_latest_boxes.setter
    def _latest_boxes(self, boxes):
        self._cached_boxes = boxes


class _FakeSam3Model:
    @classmethod
    def from_pretrained(cls, model_name, token=None):
        _ = (model_name, token)
        return cls()

    def to(self, device):
        _ = device
        return self

    def __call__(self, **kwargs):
        return {"fake": True, "kwargs": kwargs}


@pytest.fixture()
def fake_sam3(monkeypatch):
    fake_module = types.SimpleNamespace(
        Sam3Model=_FakeSam3Model, Sam3Processor=_FakeSam3Processor
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_module)

    original_call = _FakeSam3Processor.__call__

    def _call_and_cache(self, *args, **kwargs):
        batch = original_call(self, *args, **kwargs)
        self._latest_boxes = batch["input_boxes"][0]
        return batch

    monkeypatch.setattr(_FakeSam3Processor, "__call__", _call_and_cache)


def test_load_sam3_model_with_mock(fake_sam3):
    sam = Sam3PolygonModel.load_model(model_name="facebook/sam3", hf_token="fake-token")
    assert sam.model is not None
    assert sam.processor is not None


def test_predict_polygons_from_predict_image(fake_sam3):
    model = deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    image_path = get_data("OSBS_029.png")
    results = model.predict_image(path=image_path)

    polygons = model.predict_polygons(
        results=results,
        path=image_path,
        model_name="facebook/sam3",
        hf_token="fake-token",
        prompt_mode="box",
    )

    assert polygons is not None
    assert len(polygons) > 0
    assert "geometry" in polygons.columns
    assert utilities.determine_geometry_type(polygons) == "polygon"


def test_predict_polygons_from_predict_tile(fake_sam3):
    model = deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    tile_path = get_data("OSBS_029.tif")
    results = model.predict_tile(path=tile_path, patch_size=300, patch_overlap=0.25)

    polygons = model.predict_polygons(
        results=results,
        path=tile_path,
        model_name="facebook/sam3",
        hf_token="fake-token",
        prompt_mode="box",
    )

    assert polygons is not None
    assert len(polygons) > 0
    assert "geometry" in polygons.columns
    assert utilities.determine_geometry_type(polygons) == "polygon"


def test_predict_polygons_point_workflow(fake_sam3):
    model = deepforest()
    model.load_model(model_name="weecology/deepforest-tree")
    image_path = get_data("OSBS_029.png")
    results = model.predict_image(path=image_path).copy()
    results["x"] = (results["xmin"] + results["xmax"]) / 2.0
    results["y"] = (results["ymin"] + results["ymax"]) / 2.0
    results["geometry"] = [
        shapely.geometry.Point(x, y)
        for x, y in zip(results["x"], results["y"], strict=False)
    ]
    results = results.drop(columns=["xmin", "ymin", "xmax", "ymax"])
    results = utilities.__pandas_to_geodataframe__(results)

    polygons = model.predict_polygons(
        results=results,
        path=image_path,
        model_name="facebook/sam3",
        hf_token="fake-token",
        prompt_mode="point",
        point_box_size=16.0,
    )

    assert polygons is not None
    assert len(polygons) > 0
    assert "geometry" in polygons.columns
    assert utilities.determine_geometry_type(polygons) == "polygon"
