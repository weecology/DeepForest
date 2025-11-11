# Spotlight Integration

DeepForest integrates with [Renumics Spotlight](https://github.com/Renumics/spotlight) for interactive visualization of forest detection results. This integration allows you to explore predictions, analyze model performance, and assess data quality through Spotlight's web interface.

> **Note**: The Spotlight manifest format is experimental. For production use, consider the Hugging Face datasets export which offers broader tool compatibility.

## Quick Start

```python
from deepforest import get_data
from deepforest.utilities import read_file
from deepforest.visualize import view_with_spotlight


path = get_data("OSBS_029.csv")
df = read_file(path)

# Convert to Spotlight format
spotlight_data = df.spotlight()
# or
spotlight_data = view_with_spotlight(df)

# Generate new predictions and visualize
from deepforest.main import deepforest
model = deepforest()
model.load_model("Weecology/deepforest-tree")
image_path = get_data("OSBS_029.tif")
results = model.predict_image(path=image_path)

# Visualize with confidence scores preserved
spotlight_data = view_with_spotlight(results)

# Export to file for external tools
df.spotlight(format="lightly", out_dir="spotlight_export")
```

## API Reference

- `view_with_spotlight(df, format="lightly"|"objects", out_dir=...)` - Convert DeepForest DataFrame to Spotlight format
- `df.spotlight(...)` - DataFrame accessor method (calls `view_with_spotlight`)
- `prepare_spotlight_package(gallery_dir, out_dir)` - Package gallery thumbnails for Spotlight
- `export_to_spotlight_dataset(gallery_dir)` - Create Hugging Face Dataset from gallery

## Working with Predictions

When working with model predictions, the integration preserves confidence scores and detection metadata:

```python
from deepforest import get_data
from deepforest.main import deepforest
from deepforest.visualize import view_with_spotlight

# Generate predictions
model = deepforest()
model.load_model("Weecology/deepforest-tree")
image_path = get_data("OSBS_029.tif")
results = model.predict_image(path=image_path)

# Convert to Spotlight format
spotlight_data = results.spotlight()
# or
spotlight_data = view_with_spotlight(results)
```

The converted data includes:
- Bounding boxes (xmin, ymin, xmax, ymax)
- Class labels
- Confidence scores (0.0 to 1.0)
- Source image paths

This enables you to:
- Filter detections by confidence threshold
- Compare model performance across images
- Identify patterns in prediction quality
- Analyze spatial distribution of detections

## Example Output

The following example shows the Spotlight interface displaying DeepForest predictions:

```python
from deepforest import get_data
from deepforest.main import deepforest
from deepforest.visualize import view_with_spotlight

model = deepforest()
model.load_model("Weecology/deepforest-tree")
image_path = get_data("OSBS_029.tif")
results = model.predict_image(path=image_path)

# Launch Spotlight viewer
view_with_spotlight(results)
```

![Spotlight Interface](assets/spotlight_demo_screenshot.png)
*Main interface showing detection results in an interactive table*

![Detection Details](assets/spotlight_demo_screenshot1.png)
*Confidence scores and bounding box coordinates for each detection*

![Image Inspector](assets/spotlight_demo_screenshot2.png)
*Source imagery with detection metadata*

The interface displays:
- 55 tree detections from the sample image
- Confidence scores ranging from 0.303 to 0.799
- Bounding box coordinates for each detection
- Interactive sorting and filtering capabilities
- Source image visualization

## Demo Script

Test the integration with the included demo:

```bash
python demo_spotlight.py
```

The script will load a model, generate predictions, and launch the Spotlight viewer in your browser.

## Advanced Usage

For more complex workflows, you can combine Spotlight integration with gallery generation:

```python
from deepforest.visualize import (
    view_with_spotlight,
    export_to_gallery,
    write_gallery_html,
    export_to_spotlight_dataset
)

# Direct Spotlight integration
spotlight_data = view_with_spotlight(df, format="lightly")

# Create thumbnail gallery
metadata = export_to_gallery(df, "forest_gallery", max_crops=200)

# Generate HTML viewer
write_gallery_html("forest_gallery")

# Export as HuggingFace dataset
hf_dataset = export_to_spotlight_dataset("forest_gallery")
```

## Export Formats

| Format | Status | Files Created | Use Case |
|--------|--------|---------------|----------|
| Direct Spotlight | Stable | None | Interactive visualization |
| Gallery + HTML | Stable | Thumbnail images | Local browsing, sharing |
| HuggingFace Dataset | Stable | Uses existing images | Data science workflows |

## Command Line Interface

```bash
# Export predictions to gallery
python -m deepforest.scripts.cli gallery export predictions.csv --out forest_gallery

# Package for Spotlight
python -m deepforest.scripts.cli gallery spotlight --gallery forest_gallery --out spotlight_package

# Package for Spotlight
python -m deepforest.scripts.cli gallery spotlight --gallery forest_gallery --out spotlight_package
```

## Implementation Notes

The Spotlight export creates a minimal package containing:
- `manifest.json` - Image and annotation metadata
- `images/` folder - When generated from gallery export

See `src/deepforest/visualize/spotlight_export.py` for packaging utilities and `src/deepforest/visualize/spotlight_adapter.py` for the data format mapping.

## Alternative Export Options

For integration with other tools, you can export the gallery data in standard formats and use external tools for further processing.

## Testing

Run the test suite:

```bash
python -m pip install -U pytest pandas
python -m pytest -q tests/test_spotlight_adapter.py
```

Tests are located in `tests/test_spotlight_adapter.py` and `tests/test_spotlight_export.py`.
