"""Top-level package for DeepForest."""

__author__ = """Ben Weinstein"""
__email__ = "ben.weinstein@weecology.org"

import os

from pytorch_lightning.utilities import disable_possible_user_warnings

from ._version import __version__

_ROOT = os.path.abspath(os.path.dirname(__file__))

# Disable PyTorch warnings that confuse users
disable_possible_user_warnings()


def get_data(path):
    """Get package sample data path.

    Args:
        path: Data file name

    Returns:
        Full path to data file
    """
    if path == "config.yaml":
        return os.path.join(_ROOT, "conf", "config.yaml")
    else:
        return os.path.join(_ROOT, "data", path)


__all__ = ["__version__"]
