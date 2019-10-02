#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""
import os
import sys
import pytest
import keras
from  deepforest import deepforest

def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

def test_download_release():
    test_model = deepforest.deepforest(weights=None)
    test_model.download_release()
    #Assert is model instance
    assert isinstance(test_model.model,keras.models.Model)
    
