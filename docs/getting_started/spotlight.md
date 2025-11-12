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
  - Supports flexible image reference columns: `image_path`, `file_name`, `source_image`, `image`
  - Handles NaN values in optional columns gracefully
  - Validates required bbox columns: `xmin`, `ymin`, `xmax`, `ymax`
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

```{image} ../assets/spotlight_demo_screenshot.png
:alt: Spotlight interface main view showing DeepForest predictions with data table and image viewer
:width: 600px
:align: center
```
*Main interface showing detection results in an interactive table*

```{image} ../assets/spotlight_demo_screenshot1.png
:alt: Spotlight interface showing detailed bounding box visualization on forest imagery
:width: 600px
:align: center
```
*Confidence scores and bounding box coordinates for each detection*

```{image} ../assets/spotlight_demo_screenshot2.png
:alt: Spotlight interface displaying confidence score distribution and filtering options
:width: 600px
:align: center
```
*Source imagery with detection metadata*

The screenshots show the Spotlight interface with:
- **Main view**: Data table displaying tree detections with confidence scores, bounding box coordinates, and interactive sorting capabilities
- **Image viewer**: Visual representation of detected trees with bounding boxes overlaid on the source forest imagery
- **Analytics panel**: Confidence score distribution charts and filtering options for analyzing model performance across detections

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

## Data Format Flexibility

The Spotlight integration supports flexible column naming for image references:

```python
# All of these column names work for image references:
df_path = pd.DataFrame({"image_path": ["img.jpg"], ...})
df_file = pd.DataFrame({"file_name": ["img.jpg"], ...})
df_source = pd.DataFrame({"source_image": ["img.jpg"], ...})
df_image = pd.DataFrame({"image": ["img.jpg"], ...})

# All will work with Spotlight
for df in [df_path, df_file, df_source, df_image]:
    spotlight_data = view_with_spotlight(df)
```

The integration also handles missing values gracefully:
- NaN values in optional columns (label, score) are excluded from output
- Empty DataFrames raise clear error messages
- Missing required columns (bbox coordinates) are validated

## Error Handling

The integration provides clear error messages for common issues:

```python
# Empty DataFrame
df_empty = pd.DataFrame()
# Raises: ValueError("DataFrame is empty")

# Missing image reference
df_no_image = pd.DataFrame({"xmin": [10], "ymin": [10], "xmax": [50], "ymax": [50]})
# Raises: ValueError("DataFrame must contain an image reference column")

# Missing bbox columns
df_no_bbox = pd.DataFrame({"image_path": ["test.jpg"], "label": ["Tree"]})
# Raises: ValueError("Missing required bbox column")

# Invalid format
df.spotlight(format="invalid")
# Raises: ValueError("Unsupported format: invalid")
```

## Testing

Run the comprehensive test suite:

```bash
python -m pip install -U pytest pandas
python -m pytest -q tests/test_spotlight.py
```

The test suite covers:
- Basic functionality and error handling
- Multiple image reference column formats
- NaN value handling
- Format consistency between "objects" and "lightly" formats
- Complete prediction workflows
- DataFrame accessor methods
- Export functionality

Tests are consolidated in `tests/test_spotlight.py`.
