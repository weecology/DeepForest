#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""
import os
import sys
import pytest
import keras
from  ..deepforest import deepforest

def test_deepforest():
    model = deepforest.deepforest(weights=None)
    assert model.weights is None

def test_download_release():
    test_model = deepforest.deepforest(weights=None)
    test_model.download_release()
    #Assert is model instance
    assert isinstance(test_model.model,keras.models.Model)
    
def test_deepforest_weights():
    model = deepforest.deepforest(weights="data/universal_model_july30.h5")
#@pytest.fixture
#def response():
    #"""Sample pytest fixture.

    #See more at: http://doc.pytest.org/en/latest/fixture.html
    #"""



#def test_content(response):
    #"""Sample pytest test function with the pytest fixture as an argument."""
    ## from bs4 import BeautifulSoup
    ## assert 'GitHub' in BeautifulSoup(response.content).title.string
