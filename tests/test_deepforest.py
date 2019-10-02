#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""
import os
import sys
import pytest
import keras
from  deepforest import deepforest
from deepforest import utilities

@pytest.fixture()
def download_release(scope="session"):
    print("running fixtures")
    utilities.use_release()    
    
def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

def test_use_release(download_release):
    test_model = deepforest.deepforest() 
    test_model.use_release()
    #Assert is model instance    
    assert isinstance(test_model.model,keras.models.Model)

def test_predict(download_release):
    test_model = deepforest.deepforest(weights="data/universal_model_july30.h5")
    