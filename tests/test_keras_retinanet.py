# test loading of keras retinanet
import os
from deepforest.keras_retinanet.utils.anchors import compute_overlap
import numpy as np

def test_keras_retinanet():
    pass


def test_Cython_build():
    import deepforest.keras_retinanet.utils.compute_overlap
    assert os.path.exists(deepforest.keras_retinanet.utils.compute_overlap.__file__)

def test_iou():
    true_array = np.expand_dims(np.array([0.,0.,5.,5.]),axis=0)
    prediction_array = np.expand_dims(np.array([0.,0.,10.,10.]), axis=0)
    retinanet_iou = compute_overlap(prediction_array,true_array)
    assert retinanet_iou[0][0] == (5**2/10**2)