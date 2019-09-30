#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `deepforest` package."""

#Path hack
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_path)

import pytest
from  ..deepforest import deepforest

def test_deepforest():
    model = deepforest.deepforest()
    assert model.weights is None

#@pytest.fixture
#def response():
    #"""Sample pytest fixture.

    #See more at: http://doc.pytest.org/en/latest/fixture.html
    #"""



#def test_content(response):
    #"""Sample pytest test function with the pytest fixture as an argument."""
    ## from bs4 import BeautifulSoup
    ## assert 'GitHub' in BeautifulSoup(response.content).title.string
