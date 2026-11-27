"""Visualization module for DeepForest.

This module provides visualization functions for forest detection results,
including traditional plotting and interactive Spotlight integration.

For backward compatibility, this module re-exports functionality from
the legacy ``visualize.py`` implementation, while Spotlight-specific
helpers are defined in ``spotlight_adapter.py``.

Example usage::

    from deepforest.visualize import plot_results

    # Traditional plotting
    plot_results(df)

    # Interactive Spotlight visualization
    data = df.spotlight()
"""

import importlib.util
from pathlib import Path

from .spotlight_adapter import (
    SpotlightAccessor,
    prepare_spotlight_package,
    view_with_spotlight,
)


def _load_legacy_visualize_module() -> object:
    """Load the legacy visualization module for backwards compatibility."""

    legacy_visualize_path = Path(__file__).parent.parent / "visualize.py"
    spec = importlib.util.spec_from_file_location(
        "legacy_visualize", legacy_visualize_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load legacy visualization module from {legacy_visualize_path}"
        )

    legacy_visualize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_visualize)
    return legacy_visualize


legacy_visualize = _load_legacy_visualize_module()

# Re-export legacy plotting helpers to preserve the public API.
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
    "prepare_spotlight_package",
    "_load_image",
]
