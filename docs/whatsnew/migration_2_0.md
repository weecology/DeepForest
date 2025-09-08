# Migrating to DeepForest 2.0

This guide summarizes breaking changes in 2.0 and how to update your code.

## Highlights
- Unified annotation geometry via `utilities.read_file()` with a `geometry` column.
- Visualization consolidated around `visualize.draw_*` and `visualize.plot_results`.
- Expanded model backbones including DETR and transformer model integrations
- Improved installation flexibility and uv support. 
- Model loading standardized via `deepforest.load_model()`.

## Deprecated functions and replacements

- `utilities.xml_to_annotations`
  - Use `utilities.read_pascal_voc(path)` or the general `utilities.read_file(path)`.

- `utilities.boxes_to_shapefile`
  - Use `utilities.image_to_geo_coordinates(df, root_dir=...)`.

- `utilities.annotations_to_shapefile` and `utilities.project_boxes`
  - Use `utilities.image_to_geo_coordinates(df, root_dir=...)`.

- `visualize.plot_points`
  - Use `visualize.plot_results` 

- `visualize.plot_predictions`
  - Use `visualize.plot_results(results_df, ground_truth_df=None, savedir=..., image=...)` for figures.

- `main.deepforest(..., augment=...)` and dataset `augment` flag
  - Configure augmentation via `config.train.augmentations` or pass `augmentations` to `load_dataset(...)`.

## Configuration example

```python
from deepforest.main import deepforest
m = deepforest(config_args={
    'num_classes': 1,
    'label_dict': {'Tree': 0},
    'train': {'augmentations': ['HorizontalFlip']}
})
```

## Visualization examples

- Draw predictions in-memory:
```python
from deepforest import visualize
# Results  should have a results_df.root_dir property from any deepforest predict function. If image is in memory, use 'image' argument.
visualize.plot_results(results=results_df)
```

## Geospatial conversion

- Convert prediction coordinates from image to geographic CRS:
```python
from deepforest.utilities import image_to_geo_coordinates
geo = image_to_geo_coordinates(results_df, root_dir='/path/to/images')
```

## Common migration steps

- Ensure result/annotation frames include a `geometry` column; use `utilities.read_file` to coerce.
- Replace `plot_predictions`/`plot_points` with `draw_predictions`/`draw_points` (or `plot_results`).
- Replace `boxes_to_shapefile`/`project_boxes` with `image_to_geo_coordinates`.
- Stop using `augment=True/False`; pass explicit `augmentations` instead and provide an empty list or None to disable augmentations."
