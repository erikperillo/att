"""
Module for feature calculations.
"""

import cv2
from cvx import inv
import numpy as np

#functions for getting components of Lab image. assumes image is already in Lab
LAB_ATTR_FUNCS = {
    "l1": lambda x: x[:, :, 0],
    "l0": lambda x: inv(LAB_ATTR_FUNCS["l1"](x)),
    "r": lambda x: x[:, :, 1],
    "g": lambda x: inv(LAB_ATTR_FUNCS["r"](x)),
    "y": lambda x: x[:, :, 2],
    "b": lambda x: inv(LAB_ATTR_FUNCS["y"](x))
}

def get_lab_attr(img, attr, cvt=True):
    """
    Gets Lab colorspace plane attr from image. Assumes img is in BGR.
    """
    #converting image to lab if required
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    #checking validity of argument
    if not attr in LAB_ATTR_FUNCS:
        raise ValueError("invalid lab attr '%s' (use: %s)" % \
            (attr, ",".join(LAB_ATTR_FUNCS)))

    return LAB_ATTR_FUNCS[attr](img)

def get_feature(img, feat):
    """
    Gets feature map from image. Assumes image comes in BGR colorspace.
    """
    #preprocessing input
    feat = feat.lower()

    if feat in LAB_ATTR_FUNCS:
        return get_lab_attr(img, attr=feat)

    raise ValueError("unknown feature '%s'" % feat)
