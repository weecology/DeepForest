"""
DeepForest Spotlight Integration Example

This example demonstrates how to use DeepForest predictions with Renumics Spotlight
for interactive data exploration and visualization.

Requirements:
    pip install renumics-spotlight

Usage:
    python demo_spotlight.py
"""

from deepforest import get_data, main
from deepforest.visualize import view_with_spotlight

# Load a DeepForest model
model = main.deepforest()
model.load_model("weecology/deepforest-tree")

# Make predictions on sample data
image_path = get_data("OSBS_029.tif")
results = model.predict_image(path=image_path)

print(f"Generated {len(results)} tree detections")
print(f"Score range: {results['score'].min():.3f} - {results['score'].max():.3f}")

# Method 1: Use DataFrame accessor
spotlight_data = results.spotlight()
print(f"Spotlight format created with {len(spotlight_data['samples'])} samples")

# Method 2: Use function directly
spotlight_data = view_with_spotlight(results)

# Optional: Save to file for later use
spotlight_data = view_with_spotlight(results, out_dir="spotlight_output")
print("Spotlight manifest saved to spotlight_output/manifest.json")

# Optional: Launch Spotlight viewer (requires renumics-spotlight)
try:
    import renumics.spotlight as spotlight
    import pandas as pd

    # Prepare data for Spotlight viewer
    spotlight_df = pd.DataFrame({
        'image': [image_path] * len(results),
        'label': results['label'],
        'confidence': results['score'],
        'xmin': results['xmin'],
        'ymin': results['ymin'],
        'xmax': results['xmax'],
        'ymax': results['ymax']
    })

    print("Opening Spotlight viewer in browser...")
    spotlight.show(spotlight_df, dtype={'image': spotlight.Image})

except ImportError:
    print("Install renumics-spotlight to launch the interactive viewer:")
    print("pip install renumics-spotlight")
