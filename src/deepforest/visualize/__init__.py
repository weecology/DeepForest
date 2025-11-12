"""Visualization module for DeepForest.

This module provides visualization functions for forest detection results,
including traditional plotting and interactive Spotlight integration.

Example usage:

    from deepforest.visualize import view_with_spotlight, plot_results

    # Traditional plotting
    plot_results(predictions_df)

    # Interactive Spotlight visualization
    spotlight_data = view_with_spotlight(predictions_df)

    # Or use DataFrame accessor
    spotlight_data = predictions_df.spotlight()
"""

# Import from the gallery and spotlight modules
# Import from the legacy visualize.py file to maintain backward compatibility
# We need to import from the parent module, not this package
import importlib.util
import sys
from pathlib import Path

from .gallery import export_to_gallery, write_gallery_html
from .spotlight_adapter import view_with_spotlight

# Get the path to the legacy visualize.py file
legacy_visualize_path = Path(__file__).parent.parent / "visualize.py"

# Import the legacy module
spec = importlib.util.spec_from_file_location("legacy_visualize", legacy_visualize_path)
legacy_visualize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_visualize)

# Import the functions we need from the legacy module
plot_results = legacy_visualize.plot_results
plot_annotations = legacy_visualize.plot_annotations
convert_to_sv_format = legacy_visualize.convert_to_sv_format
_load_image = legacy_visualize._load_image
label_to_color = legacy_visualize.label_to_color

__all__ = [
    "convert_to_sv_format",
    "export_to_gallery",
    "label_to_color",
    "plot_annotations",
    "plot_results",
    "view_with_spotlight",
    "write_gallery_html",
    "_load_image",
]
