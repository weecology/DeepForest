# -*- coding: utf-8 -*-
"""Top-level package for DeepForest."""
__author__ = """Ben Weinstein"""
__email__ = 'ben.weinstein@weecology.org'

import os
from ._version import __version__
from pytorch_lightning.utilities import disable_possible_user_warnings

_ROOT = os.path.abspath(os.path.dirname(__file__))

# Disable PyTorch warnings that confuse users
disable_possible_user_warnings()


def get_data(path):
    """Helper function to get package sample data."""
    if path == "config.yml":
        return os.path.join(_ROOT, "conf", "config.yml")
    else:
        return os.path.join(_ROOT, "data", path)
