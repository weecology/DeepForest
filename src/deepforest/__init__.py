"""Top-level package for DeepForest."""

__author__ = """Ben Weinstein"""
__email__ = "ben.weinstein@weecology.org"

import os

from pytorch_lightning.utilities import disable_possible_user_warnings

from ._version import __version__

_ROOT = os.path.abspath(os.path.dirname(__file__))

# Disable PyTorch warnings that confuse users
disable_possible_user_warnings()


def get_data(path: str):
    """Get package sample data path.

    Intended for locating files packaged with DeepForest, like test data. DeepForest functions normally accept a relative or absolute path directly.

    Args:
        path: Data file name

    Returns:
        Full path to data file
    """
    if path == "config.yaml":
        rel_path = os.path.join(_ROOT, "conf", "config.yaml")
    else:
        rel_path = os.path.join(_ROOT, "data", path)

    if os.path.exists(rel_path):
        return rel_path
    else:
        raise FileNotFoundError(
            f"The file {rel_path} was not found relative to the DeepForest package. Note: this function should only be used for accessing test/demonstration data packaged with DeepForest, instead you can pass a path directly to most DeepForest functions."
        )


__all__ = ["__version__"]
