"""!
@package feat
@brief Module for feature extractions.
"""

import cv2
import cvx
import numpy as np


##Functions for getting features from Lab image.
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

LAB_COLORS = [
    "r",
    "g",
    "b",
    "y"
]

LAB_CONTRASTS = [
    "l0",
    "l1"
]

##Default values for gabor kernel.
DEF_GABOR_K_PARAMS = {
    #kernel size
    "ksize": 2*(11,),
    #std of gaussian used in kernel
    "sigma": 2.0,
    #orientation in radians
    "theta": 0.0,
    #wavelenght of sinusoidal factor
    "lambd": 4.0,
    #spatial aspect ratio
    "gamma": 1.0,
    #phase offset
    "psi": 3.0,
    #kernel output type
    "ktype": cv2.CV_32F
}

##Default orientations available as features.
ORIENTATIONS = {
    #vertical
    "ver": 0.0,
    #horizontal
    "hor": np.pi/2,
    #left diagonal
    "r_diag": np.pi/4,
    #right diagonal
    "l_diag": -np.pi/4
}

def get_available_features():
    """!
    Gets available features.
    """
    return list(LAB_ATTR_FUNCS.keys()) + list(ORIENTATIONS.keys())

def get_lab_attr_map(img, attr, cvt=True):
    """!
    Gets feature calculable from Lab image.
    Assumes img is either in Lab or in BGR.

    @param img Input image.
    @param attr Feature to extract. See #LAB_ATTR_FUNCS.
    @param cvt If True, convert image to Lab colorspace.
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
    """!
    Gets gabor kernel for application on image.
    
    @param custom_params 
    Parameters for Gabor kernel to override #DEF_GABOR_K_PARAMS.
    """
    params = dict(DEF_GABOR_K_PARAMS) 
    params.update(custom_params)
    kernel = cv2.getGaborKernel(**params)

    return kernel

def gabor_filter(img, kernel_params={}, cvt=True, clip=True):
    """!
    Gets gabor kernel and applies to image.
    Assumes image comes in either grayscale or BGR.

    @param img Input image.
    @param kernel_params Parameters for Gabor kernel. 
    @param cvt If True, convert image to grayscale.
    @param clip If True, clip result for positive values only.
    """
    #converting image to grayscale if required
    if cvt and len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #getting kernel
    kernel = _get_gabor_kernel(**kernel_params) 
    #applying filter
    dtype = cv2.CV_32F if kernel.dtype == np.float32 else cv2.CV_64F
    img = cv2.filter2D(img, dtype, kernel)  

    return img.clip(min=0.0) if clip else img

def get_orientation_map(img, orientation="", gabor_kernel_params={}, 
    cvt=True, clip=True):
    """!
    Gets orientation map in some of the directions.
    Assumes image is in BGR.

    @param img Input image.
    @param orientation Orientation feature to compute.
    @param gabor_kernel_params Extra gabor kernel parameters.
    @param cvt Convert image from BGR to grayscale.
    @param clip It True, clip result for positive values only.
    """
    #getting available directions
    if orientation:
        rad_orientation = ORIENTATIONS[orientation]
        gabor_kernel_params.update({"theta": rad_orientation})

    return gabor_filter(img, gabor_kernel_params, cvt, clip)

def get_feature(img, feat, **kwargs):
    """!
    Gets feature map from image. 
    Assumes image comes in BGR colorspace.

    @param img Input image.
    @param feat Feature to extract.
    @param kwargs Extra arguments for feature extraction function.
    """
    #preprocessing input
    feat = feat.lower()

    if feat in LAB_ATTR_FUNCS:
        return get_lab_attr_map(img, attr=feat, **kwargs)
    elif feat in ORIENTATIONS:
        return get_orientation_map(img, orientation=feat, **kwargs)

    raise ValueError("unknown feature '%s'" % feat)
