# -*- coding: utf-8 -*-
"""Top-level package for DeepForest."""
__author__ = """Ben Weinstein"""
__email__ = 'ben.weinstein@weecology.org'
__version__ = '0.1.35'

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path):
    """helper function to get package sample data"""
    return os.path.join(_ROOT, 'data', path)
