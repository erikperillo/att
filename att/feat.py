"""
Module for feature extractions.
"""

import cv2
import cvx
import numpy as np

#functions for getting components of Lab image. assumes image is already in Lab
LAB_ATTR_FUNCS = {
    #luminance on
    "l1": lambda x: x[:, :, 0],
    #luminance off
    "l0": lambda x: cvx.inv(LAB_ATTR_FUNCS["l1"](x)),
    #red
    "r": lambda x: x[:, :, 1],
    #green
    "g": lambda x: cvx.inv(LAB_ATTR_FUNCS["r"](x)),
    #yellow
    "y": lambda x: x[:, :, 2],
    #blue
    "b": lambda x: cvx.inv(LAB_ATTR_FUNCS["y"](x))
}

#default values for gabor kernel.
DEF_GABOR_K_PARAMS = {
    #kernel size
    "ksize": 2*(21,),
    #std of gaussian used in kernel
    "sigma": 5.0,
    #orientation in radians
    "theta": 0.0,
    #wavelenght of sinusoidal factor
    "lambd": 1.0,
    #spatial aspect ratio
    "gamma": 0.5,
    #phase offset
    "psi": 0.0,
    #kernel output type
    "ktype": cv2.CV_32F
}

#default orientations available
ORIENTATIONS = {
    #vertical
    "ver": 0.0,
    #horizontal
    "hor": np.pi/2,
    #left diagonal
    "l_diag": np.pi/4,
    #right diagonal
    "r_diag": -np.pi/4
}

def get_available_features():
    """
    Gets available features.
    """
    return list(LAB_ATTR_FUNCS.keys()) + list(ORIENTATIONS.keys())

def get_lab_attr(img, attr, cvt=True):
    """
    Gets Lab colorspace plane attr from image. 
    Assumes img is either in Lab or in BGR.
    """
    #converting image to lab if required
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    #checking validity of argument
    if not attr in LAB_ATTR_FUNCS:
        raise ValueError("invalid lab attr '%s' (use: %s)" % \
            (attr, ",".join(LAB_ATTR_FUNCS)))

    return LAB_ATTR_FUNCS[attr](img)

def _get_gabor_kernel(**custom_params):
    """
    Wrapper for OpenCV's getGaborKernel.
    """
    params = dict(DEF_GABOR_K_PARAMS) 
    params.update(custom_params)
    kernel = cv2.getGaborKernel(**params)

    return kernel

def gabor_filter(img, kernel_params={}, cvt=True):
    """
    Gets gabor kernel and applies to image.
    Assumes image comes in either grayscale or BGR.
    """
    #converting image to grayscale if required
    if cvt:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #getting kernel
    kernel = _get_gabor_kernel(**kernel_params) 
    #applying filter
    dtype = cv2.CV_32F if kernel.dtype == np.float32 else cv2.CV_64F
    img = cv2.filter2D(img, dtype, kernel)  

    return img

def get_orientation_map(img, orientation):
    """
    Gets orientation map in some of the directions.
    """
    #getting available directions
    rad_orientation = ORIENTATIONS[orientation]

    return gabor_filter(img, {"theta": rad_orientation})

def get_feature(img, feat):
    """
    Gets feature map from image. Assumes image comes in BGR colorspace.
    """
    #preprocessing input
    feat = feat.lower()

    if feat in LAB_ATTR_FUNCS:
        return get_lab_attr(img, attr=feat)
    elif feat in ORIENTATIONS:
        return get_orientation_map(img, orientation=feat)

    raise ValueError("unknown feature '%s'" % feat)
