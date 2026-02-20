"""Visualization module for DeepForest.

This module provides visualization functions for forest detection results,
including traditional plotting and interactive Spotlight integration.

Example usage::

    from deepforest.visualize import plot_results

    # Traditional plotting
    plot_results(df)

    # Interactive Spotlight visualization
    data = df.spotlight()

To view results interactively with Spotlight, use the spotlight visualization functions.
"""

# Import from the gallery and spotlight modules
# Import from the legacy visualize.py file to maintain backward compatibility
# Need to import from the parent module, not this package
import importlib.util
import sys
from pathlib import Path

# Gallery functionality removed - not needed for core Spotlight implementation
from .spotlight_adapter import SpotlightAccessor, view_with_spotlight

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
    "label_to_color",
    "plot_annotations",
    "plot_results",
    "SpotlightAccessor",
    "view_with_spotlight",
    "_load_image",
]
