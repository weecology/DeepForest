# Marine Biodiversity Inference Model

This page documents how to run inference with the marine biodiversity model.
It does **not** include training or data preparation workflows.

- Model hub: <https://huggingface.co/weecology/deepforest-marine-biodiversity>
- Project discussion and notes:
  <https://huggingface.co/weecology/deepforest-marine-biodiversity/discussions/1>

## Load the model

```python
from deepforest import main

model = main.deepforest()
model.load_model(
    model_name="weecology/deepforest-marine-biodiversity",
    revision="main",
)
```

If you are downloading from an environment with strict rate limits, authenticate
first with `huggingface-cli login` or set `HF_TOKEN`.

## Run single-image inference

```python
predictions = model.predict_image(path="/path/to/image.png")
print(predictions.head())
```

## Run tiled inference for large imagery

Use tiled prediction for large aerial or marine survey images.

```python
tile_predictions = model.predict_tile(
    path="/path/to/large_tile.tif",
    patch_size=600,
    patch_overlap=0.2,
)
print(tile_predictions.head())
```

## Notes

- Predictions are returned as a dataframe with box coordinates, scores, and labels.
- Tune `patch_size` and `patch_overlap` for your image resolution and object density.
- Training details are intentionally omitted here because the original training
  datasets are not broadly accessible.
