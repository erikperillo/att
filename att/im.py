"""!
@package im
@brief Module for intensity map calculations.
"""

import cv2
import cvx
import numpy as np

def _is_even(num):
    """
    Self-explanatory.
    """
    return num % 2 == 0

def get_center_surround_kernel(size, dtype=np.float32):
    """!
    Gets kernel to be used in convolution to perform center-surround
    operation on image.

    @param size Size of kernel.
    @param dtype Type of returned kernel.
    """
    size = int(size)

    #checking validity of arguments
    if _is_even(size):
        raise InvalidDimensions("kernel size must be odd")
    if size < 2:
        raise InvalidDimensions("kernel size must be bigger than 1")
    if dtype != np.float32 and dtype != np.float64:
        raise InvalidDataType("kernel dtpe must be either float32 or float64")

    #creating kernel
    weight = -1.0/(size**2 - 1)
    kernel = weight*np.ones(dtype=dtype, shape=(size, size))
    kernel[size//2, size//2] = 1.0

    return kernel

def center_surround(img, kernel, clip=True):
    """!
    Performs center-surround operation on image with given kernel.
    Assumes image is one-dimensional.

    @param img Input image.
    @param kernel Center-surround kernel to apply to image.
    @param clip If True, clip result for positive values only.
    """
    #checking validity of image dimensions
    if len(img.shape) > 2:
        raise InvalidDimensions("image must have depth one")

    #applying operation through image
    ddepth = cv2.CV_32F if kernel.dtype == np.float32 else cv2.CV_64F
    cs_img = cv2.filter2D(img, ddepth, kernel)

    return cs_img.clip(min=0.0) if clip else cs_img

def im_pyr_lvl_one(lvl):
    """!
    Default weight function of pyramid level for intensity_map.

    @param lvl Pyramid level. 
    """
    return 1.0

def im_cs_ksize_one(size):
    """!
    Default weight function of center-surround kernel size for intensity_map.

    @param size Size of center-surround kernel.
    """
    return 1.0

def im_weight_one(pyr_lvl, cs_ksize):
    """!
    Default weight function for intensity map.

    @param lvl Pyramid level.
    @param cs_ksize Center-surround kernel size.
    """
    return im_pyr_lvl_one(pyr_lvl)*im_cs_ksize_one(cs_ksize)

class IMLCWeightFunc(object):
    """!
    Class for intensity map weight functions used for 
    linear combinations of maps.
    """
    def __init__(self, pyr_f, cs_ksize_f):
        """!
        Initializer for class.
        """
        ##Weight function for pyramid level.
        self.pyr_f = pyr_f
        ##Weight function for center-surround kernel size.
        self.cs_ksize_f = cs_ksize_f

    def __call__(self, pyr_lvl, cs_ksize):
        """!
        Computes total weight function as the product of the functions.
        
        @param pyr_lvl Pyramid level.
        @param cs_ksize Center-surround kernel size.
        """
        return self.pyr_f(pyr_lvl)*self.cs_ksize_f(cs_ksize)

def intensity_map(img, pyr_lvls=3, cs_ksizes=(3, 7), weight_f=im_weight_one,
    dtype=np.float32, debug=False):
    """!
    Gets intensity map by by summing up center_surround on
    multiple scales and kernel sizes.
    Assumes image is one-dimensional.
    The resultant intensity map is a linear combination of the intermediary
    maps.

    @param pyr_lvls Pyramids levels to calculate.
    @param cs_ksizes Center-surround kernel sizes.
    @param weight_f Function to compute weight of map from pyramid level and
        center-surround kernel size.
    @param dtype Type of output image.
    @param debug If True, returns intermediary intendity maps.
    """
    #getting all kernels
    cs_kernels = [get_center_surround_kernel(ks, dtype) for ks in cs_ksizes]
    #initial value for intensity map
    im_img = np.zeros(dtype=dtype, shape=img.shape)
    #debug image
    debug_img = None

    #iterating over pyramid levels (from 0 to pyr_lvls)
    for i in range(pyr_lvls+1):
        #getting downsampled image
        img = cv2.pyrDown(img) if i > 0 else img
        #partial debug image
        db_img = None

        for ki, k in enumerate(cs_kernels):
            #getting center-surround image 
            cs_img = center_surround(img, weight_f(i, ki)*k)
            #updating debug partial image
            if debug:
                db_img = cs_img if db_img is None else \
                    cvx.h_append(db_img, cs_img)
            #rescaling into original image dimensions
            cs_img = cvx.rep_pyrUp(cs_img, i)
            #updating debug partial image
            if debug and i > 0:
                db_img = cvx.h_append(db_img, cs_img)
            #summing contribution
            im_img += cs_img
    
        #updating debug image
        if debug:
            debug_img = db_img if debug_img is None else \
                cvx.v_append(debug_img, db_img) 

    return debug_img, im_img
