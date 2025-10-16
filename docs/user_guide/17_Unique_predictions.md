# Unique Predictions: Double-Counting Removal

This guide covers DeepForest's advanced double-counting removal capabilities for overlapping imagery scenarios. The algorithm uses Structure-from-Motion (SfM) techniques to resolve duplicate detections across multiple images, ensuring accurate object counts in drone surveys.

## Overview

When analyzing overlapping aerial imagery (common in drone surveys), the same objects (e.g., tree crowns) often appear in multiple images, leading to inflated counts. Traditional approaches like simple IoU-based Non-Maximum Suppression fail because they don't account for the geometric relationship between images.

DeepForest's double-counting removal algorithm solves this by:

1. **Feature Extraction**: Creating SfM features for all images using DISK+LightGlue
2. **Geometric Alignment**: Computing homography matrices between image pairs
3. **Prediction Alignment**: Transforming predictions between coordinate systems
4. **Overlap Resolution**: Removing duplicates using configurable strategies
5. **Visualization**: Optional before/after comparison plots

## Use Cases

- **Drone Surveys**: Overlapping flight paths with redundant tree crown detections
- **Multi-Angle Detection**: Same objects viewed from different perspectives
- **Forest Inventory**: Accurate tree counts across overlapping imagery
- **Ecological Monitoring**: Precise wildlife or vegetation counts

## Quick Start

First, install DeepForest with the double-counting dependencies:

```bash
pip install deepforest[double_counting]
```

Then use the double-counting functionality:

```python
from deepforest import main

# Initialize model
model = main.deepforest()
model.use_release()

# Run double-counting removal
results = model.predict_unique(
    image_dir="/path/to/overlapping/images",
    save_dir="/path/to/sfm_output",
    strategy='highest-score',
    visualization=True
)

print(f"Found {len(results)} unique tree crowns")
```

## Algorithm Details

### Feature Extraction & Matching

The algorithm uses DISK+LightGlue for robust feature extraction and matching:

- **DISK descriptors**: Scale-invariant features effective for aerial imagery
- **LightGlue matcher**: High-quality correspondences with viewpoint robustness
- **Exhaustive pairing**: All possible image pairs are considered for matching

### Homography Estimation

For each image pair, a perspective transformation matrix is computed:

- **RANSAC algorithm**: Robust estimation in presence of outliers
- **Minimum 4 points**: Required for reliable homography computation
- **Error threshold**: 4.0 pixels (tunable for different image types)

### Prediction Alignment

Bounding box predictions are transformed between coordinate systems:

- **Corner transformation**: All four corners of each bounding box are transformed
- **Axis-aligned bounding**: Transformed quadrilaterals are converted to rectangles
- **Coordinate mapping**: Predictions aligned to common reference frame

### Overlap Resolution Strategies

Three strategies are available for removing overlapping detections:

#### 1. Highest-Score Strategy (`'highest-score'`)

**Recommended for most use cases**

- Uses Non-Maximum Suppression (NMS) based on confidence scores
- Keeps predictions with highest confidence, removes lower-scoring overlaps
- Most sophisticated approach considering both spatial overlap and confidence
- Requires GPU for optimal performance

```python
results = model.predict_unique(
    image_dir="images/",
    save_dir="output/",
    strategy='highest-score'  # Default
)
```

#### 2. Left-Hand Strategy (`'left-hand'`)

- Keeps all predictions from the first image
- Removes overlapping predictions from the second image
- Simple but may not be optimal for all scenarios
- Faster than highest-score strategy

```python
results = model.predict_unique(
    image_dir="images/",
    save_dir="output/",
    strategy='left-hand'
)
```

#### 3. Right-Hand Strategy (`'right-hand'`)

- Keeps all predictions from the second image
- Removes overlapping predictions from the first image
- Simple but may not be optimal for all scenarios
- Faster than highest-score strategy

```python
results = model.predict_unique(
    image_dir="images/",
    save_dir="output/",
    strategy='right-hand'
)
```

## Requirements

### Image Requirements

- **Overlap**: Images must have sufficient overlap (>30% recommended)
- **Viewpoints**: Similar viewpoints for reliable homography estimation
- **Quality**: Clear images with distinctive features for matching
- **Formats**: Supported formats include .tif, .png, .jpg

### System Requirements

- **Memory**: Scales with number of predictions per image
- **GPU**: Recommended for highest-score strategy (CUDA-compatible)
- **Storage**: Space for SfM feature files (typically 10-50MB per image)

### Dependencies

The double-counting functionality requires additional dependencies that are not included in the base DeepForest installation. Install them using:

```bash
pip install deepforest[double_counting]
```

This installs, among others:
- `opencv-python==4.11.0.86`: OpenCV (GUI wheel) used by the double-counting pipeline
- `pycolmap>=0.6.0`: For robust homography estimation
- `hloc>=1.4.0`: For feature extraction and matching
- `kornia>=0.7.0` and `kornia-feature>=0.7.0`: For feature ops/matching helpers

Important: Do not install `opencv-python` and `opencv-python-headless` in the same environment.

Alternatively, you can install the dependencies manually:

```bash
pip install opencv-python==4.11.0.86 pycolmap hloc kornia kornia-feature
```

## API Reference

### `predict_unique()`

High-level function for double-counting removal across multiple images.

```python
def predict_unique(image_dir, save_dir, strategy='highest-score', visualization=True):
    """
    Get unique predictions from overlapping images.
    
    Args:
        image_dir (str): Path to directory containing input images
        save_dir (str): Path for saving intermediate SfM files
        strategy (str): Deduplication strategy ('highest-score', 'left-hand', 'right-hand')
        visualization (bool): Show before/after comparison plots
        
    Returns:
        pandas.DataFrame: Deduplicated predictions with columns:
            ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label', 'image_path']
    """
```

### Core Functions

#### `create_sfm_model()`

Generate SfM feature files needed for geometric matching.

```python
from deepforest.evaluate import create_sfm_model

create_sfm_model(
    image_dir=Path("images/"),
    output_path=Path("sfm_output/"),
    references=["image1.tif", "image2.tif"],
    overwrite=True
)
```

#### `align_and_delete()`

Main function for removing double-counting across image pairs.

```python
from deepforest.evaluate import align_and_delete

final_predictions = align_and_delete(
    matching_h5_file="matches.h5",
    predictions=initial_predictions,
    device=device,
    threshold=0.325,
    strategy='highest-score'
)
```

## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(N²) where N is the number of images
- **Pair Processing**: N*(N-1)/2 image pairs must be processed
- **Feature Extraction**: Most time-consuming step (scales with image size)
- **GPU Acceleration**: Available for NMS operations

### Memory Usage

- **Scales with predictions**: More predictions per image = more memory
- **SfM Features**: Feature files require additional storage
- **Batch Processing**: Consider processing large datasets in batches

### Optimization Tips

1. **Use GPU**: Enable CUDA for highest-score strategy
2. **Batch Processing**: Process large datasets in smaller batches
3. **Image Resolution**: Consider downsampling very large images
4. **Strategy Selection**: Use simpler strategies for faster processing

## Troubleshooting

### Common Issues

#### Insufficient Matching Points

**Error**: `"Not enough matching points (<4) found between images"`

**Solutions**:
- Ensure images have sufficient overlap (>30%)
- Check image quality and clarity
- Verify images are from similar viewpoints
- Consider adjusting RANSAC parameters

#### Homography Estimation Failure

**Error**: `"Homography matrix estimation failed"`

**Solutions**:
- Increase image overlap
- Improve image quality
- Check for motion blur or distortion
- Verify feature extraction succeeded

#### Memory Issues

**Symptoms**: Out of memory errors during processing

**Solutions**:
- Reduce batch size
- Use CPU instead of GPU for NMS
- Process fewer images at once
- Consider image downsampling

### Performance Optimization

#### For Large Datasets

```python
# Process in batches
image_batches = [images[i:i+5] for i in range(0, len(images), 5)]

for batch in image_batches:
    results = model.predict_unique(
        image_dir=f"batch_{batch_idx}/",
        save_dir=f"output_{batch_idx}/",
        strategy='highest-score'
    )
```

#### For Speed-Critical Applications

```python
# Use faster strategy
results = model.predict_unique(
    image_dir="images/",
    save_dir="output/",
    strategy='left-hand',  # Faster than highest-score
    visualization=False    # Skip visualization
)
```

## Examples

### Basic Usage

```python
# First install: pip install deepforest[double_counting]
from deepforest import main

# Initialize model
model = main.deepforest()
model.use_release()

# Process overlapping images
results = model.predict_unique(
    image_dir="drone_survey/images/",
    save_dir="drone_survey/sfm_output/",
    strategy='highest-score',
    visualization=True
)

# Save results
results.to_csv("unique_predictions.csv", index=False)
print(f"Processed {len(results)} unique tree crowns")
```

### Advanced Configuration

```python
# First install: pip install deepforest[double_counting]
import torch
from deepforest import main
from deepforest.evaluate import create_sfm_model, align_and_delete

# Custom processing pipeline
model = main.deepforest()
model.use_release()

# Create SfM features with custom settings
create_sfm_model(
    image_dir=Path("images/"),
    output_path=Path("sfm_output/"),
    references=["img1.tif", "img2.tif", "img3.tif"],
    overwrite=True
)

# Run detection on individual images
all_predictions = []
for image_file in ["img1.tif", "img2.tif", "img3.tif"]:
    preds = model.predict_image(path=f"images/{image_file}")
    preds["image_path"] = image_file
    all_predictions.append(preds)

predictions = pd.concat(all_predictions)

# Custom double-counting removal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_results = align_and_delete(
    matching_h5_file="sfm_output/matches.h5",
    predictions=predictions,
    device=device,
    threshold=0.4,  # Custom IoU threshold
    strategy='highest-score'
)
```

### Batch Processing

```python
import os
from deepforest import main
import shutil
import pandas as pd

def process_image_batch(image_dir, batch_size=5):
    """Process images in batches to manage memory usage."""
    model = main.deepforest()
    model.use_release()
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.tif', '.png', '.jpg'))]
    
    # Process in batches
    batches = [image_files[i:i+batch_size] 
               for i in range(0, len(image_files), batch_size)]
    
    all_results = []
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        
        # Create temporary directory for batch
        batch_dir = f"batch_{i}"
        os.makedirs(batch_dir, exist_ok=True)
        
        # Copy batch images
        for img_file in batch:
            shutil.copy(f"{image_dir}/{img_file}", f"{batch_dir}/{img_file}")
        
        # Process batch
        results = model.predict_unique(
            image_dir=batch_dir,
            save_dir=f"sfm_batch_{i}",
            strategy='highest-score',
            visualization=False
        )
        
        all_results.append(results)
        
        # Cleanup
        shutil.rmtree(batch_dir)
    
    return pd.concat(all_results, ignore_index=True)

# Usage
results = process_image_batch("large_dataset/", batch_size=3)
```

## Best Practices

### Image Preparation

1. **Ensure Sufficient Overlap**: Aim for >30% overlap between adjacent images
2. **Maintain Consistent Viewpoints**: Similar camera angles improve matching
3. **Optimize Image Quality**: Clear, well-lit images with distinctive features
4. **Consider Resolution**: Balance between detail and processing speed

### Strategy Selection

- **Highest-Score**: Best accuracy, requires GPU, slower processing
- **Left/Right-Hand**: Faster processing, may be less accurate
- **Consider Use Case**: Choose based on accuracy vs. speed requirements

### Performance Optimization

1. **Use GPU**: Enable CUDA for NMS operations
2. **Batch Processing**: Process large datasets in manageable chunks
3. **Memory Management**: Monitor memory usage, adjust batch sizes
4. **Storage Planning**: Allocate sufficient space for SfM files

## Integration with Existing Workflows

### With DeepForest Training

```python
# Train model on non-overlapping data
model = main.deepforest()
model.train(
    csv_file="training_data.csv",
    root_dir="training_images/"
)

# Apply to overlapping survey data
survey_results = model.predict_unique(
    image_dir="survey_images/",
    save_dir="survey_sfm/",
    strategy='highest-score'
)
```

### With Evaluation Metrics

```python
# Evaluate unique predictions
evaluation_results = model.evaluate(
    csv_file="ground_truth.csv",
    root_dir="ground_truth_images/",
    predictions=survey_results
)

print(f"Precision: {evaluation_results['box_precision']:.3f}")
print(f"Recall: {evaluation_results['box_recall']:.3f}")
```

## Conclusion

DeepForest's double-counting removal algorithm provides a robust solution for accurate object counting in overlapping imagery scenarios. By leveraging SfM techniques and configurable strategies, users can achieve precise results while balancing accuracy and computational efficiency.

For additional support or questions about the double-counting removal functionality, please refer to the DeepForest documentation or community forums.

## Using only the double-counting tools (standalone)

If you only need the SfM-based de-duplication (without running DeepForest inference), you can use the building blocks in `deepforest.evaluate` directly: `create_sfm_model()` and `align_and_delete()`. In that case, you must supply your own predictions DataFrame with columns `['xmin','ymin','xmax','ymax','score','label','image_path']`.

### Why a separate environment is recommended

- Different OpenCV wheels conflict (GUI `opencv-python` vs `opencv-python-headless`). The repo uses the headless build; mixing both breaks imports.
- Torch/TorchVision/CUDA builds must be matched; installing other CV stacks may overwrite them.
- Geo stack (`geopandas`, `shapely`) often pulls native libs; isolating avoids version pin clashes.

Create a fresh environment to avoid these conflicts.

### Minimal standalone environment (no DeepForest inference)

Recommended with conda-forge (for clean Geo/CV binaries):

```bash
conda create -n df-doublecounting -c conda-forge python=3.11
conda activate df-doublecounting

# Core numeric/geo/cv
pip install opencv-python==4.11.0.86 numpy pandas shapely geopandas h5py matplotlib

# Torch + vision (pick the right CUDA/CPU build for your system if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# SfM and features
pip install pycolmap hloc kornia kornia-feature
```

Now you can run:

```python
from pathlib import Path
import pandas as pd
from deepforest.evaluate import create_sfm_model, align_and_delete
import torch

# 1) Build SfM artifacts once
create_sfm_model(
    image_dir=Path("/path/to/images"),
    output_path=Path("/path/to/sfm_output"),
    references=sorted(["img1.tif","img2.tif","img3.tif"]),
    overwrite=True,
)

# 2) Load your own predictions (must include image_path per row)
predictions = pd.read_csv("/path/to/predictions.csv")

# 3) De-duplicate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_predictions = align_and_delete(
    matching_h5_file=str(Path("/path/to/sfm_output")/"matches.h5"),
    predictions=predictions,
    device=device,
    threshold=0.325,
    strategy="highest-score",
)
```

### Using only the extras without DeepForest

If you still prefer pip extras management but won’t use DeepForest inference, you can install just the extra group and your own stack:

```bash
pip install deepforest[double_counting]
# Add your preferred torch build afterwards (to avoid pip picking an unintended one)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Notes:
- Ensure only one OpenCV wheel is present in the env. For double-counting, use `opencv-python==4.11.0.86` and avoid installing `opencv-python-headless` concurrently. If you hit conflicts, uninstall all OpenCV wheels and reinstall the one you need.
 