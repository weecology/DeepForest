"""Spotlight integration for DeepForest detection results.

This module provides integration with Renumics Spotlight for interactive
visualization of forest detection results.

Example usage:

    from deepforest.visualize import view_with_spotlight

    # Convert predictions to Spotlight format
    spotlight_data = view_with_spotlight(predictions_df)

    # Or use DataFrame accessor
    spotlight_data = predictions_df.spotlight()
"""

# Import from the main visualize.py file
import sys
from pathlib import Path

from .gallery import export_to_gallery, write_gallery_html
from .spotlight_adapter import view_with_spotlight

# plot_results is provided by the (legacy) top-level module `visualize.py`.
# We import it lazily here to maintain backward compatibility while avoiding
# import-time dependency issues with the legacy module structure.
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from deepforest.visualize import plot_results
except ImportError:
    # Fallback if legacy visualize.py import fails
    def plot_results(*args, **kwargs):
        raise ImportError("plot_results function not available")


__all__ = [
    "export_to_gallery",
    "plot_results",
    "view_with_spotlight",
    "write_gallery_html",
]
