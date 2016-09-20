"""!
@package im
@brief Module for intensity map calculations.
"""

import cv2
import cvx
import numpy as np
from itertools import combinations

def _id(x):
    """!
    Identity function.
    """
    return x

def _is_even(num):
    """!
    Self-explanatory.
    """
    return num % 2 == 0

def _custom_dict(def_dict, custom_params):
    """!
    Returns a copy of def_dict with some of its values changed by custom_params.
    """
    new_dict = dict(def_dict)
    new_dict.update(custom_params)

    return new_dict

def _euclid_dist(vec_1, vec_2):
    """!
    Euclidean distance of two vectors.
    """
    return np.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(vec_1, vec_2)))

def _dists(points, dist_f=_euclid_dist):
    """!
    Computes the distances of a set of points.
    """
    return map(lambda vecs: dist_f(*vecs), combinations(points, 2))

def _cc(img, conn=8, dtype=cv2.CV_16U):
    """!
    Performs connected components on image. 
    Returns a set of stats for components (x, y, w, h) and centroids (x, y).
    """
    *__, stats, centroids = cv2.connectedComponentsWithStats(img, conn, dtype)

    return stats, centroids

def _frac_threshold(img, max_frac=0.75):
    """!
    Fractional Threshold. Only values above specified fraction will be white.
    """
    thresh = int(img.max()*max_frac)

    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

def _otsu_threshold(img):
    """!
    Adaptative Otshu threshold.
    """
    return cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

##Available threshold functions.
THRESH_FUNCS = {
    "otsu": _otsu_threshold,
    "frac": _frac_threshold
}

def _square(img):
    """!
    Self-explanatory.
    """
    return img**2

def _clahe(img, clip_limit=2.0, grid_size=8):
    """!
    Contrast Limited Adaptive Histogram Equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=2*(grid_size,))

    return clahe.apply(img)

##Available contrast operations.
CONTRAST_FUNCS = {
    #square of image
    "square": _square,
    "sq": _square,
    #adaptative contrast
    "clahe": _clahe,
    #nothing
    "none": _id
}

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

    @param pyr_lvl Pyramid level.
    @param cs_ksize Center-surround kernel size.
    """
    return im_pyr_lvl_one(pyr_lvl)*im_cs_ksize_one(cs_ksize)

def intensity_map(img, pyr_lvls=3, cs_ksizes=(3, 7), weight_f=im_weight_one,
    dtype=np.float32, debug=False):
    """!
    Gets intensity map by by summing up center_surround on
    multiple scales and kernel sizes.
    Assumes image is one-dimensional.
    The resultant intensity map is a linear combination of the intermediary
    maps.

    @param img Input image.
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

def _cc_cm_dists(cc_params):
    """!
    Computes distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    __, centroids = cc_params
    #calculating euclidean distance of centroids
    dists = _dists(centroids)

    return dists

def _cc_sum_cm_dists(cc_params):
    """!
    Sum of distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    return sum(_cc_cm_dists(cc_params))

def _cc_cm_dists_mean(cc_params):
    """!
    Mean distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    __, centroids = cc_params
    return _cc_sum_cm_dists(cc_params)/len(centroids)

def _cc_ssq_cm_dists(cc_params):
    """!
    Sum of squares of distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    return sum(x**2 for x in _cc_cm_dists(cc_params))

def _cc_rssq_cm_dists(cc_params):
    """!
    Root-sum-of-squares of distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    return np.sqrt(_cc_ssq_cm_dists(cc_params))

def _cc_sq_cm_dists_mean(cc_params):
    """!
    Mean of squares of distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    __, centroids = cc_params
    return _cc_ssq_cm_dists(cc_params)/len(centroids)

def _cc_sq_cm_dists_mean_sqr(cc_params):
    """!
    Root of mean of squares of distances of centroids of connected components.
    
    @param cc_params Connected components params in format (stats, centroids).
    """
    return np.sqrt(_cc_sq_cm_dists_mean(cc_params))

##Available weight functions for #cc_norm.
CC_NORM_SCORE_FUNCS = {
    #mean of center of mass distances
    "cmdm": _cc_cm_dists_mean,
    #root of sum of squares of center of mass distances
    "cmdrssq": _cc_rssq_cm_dists,
    #mean of squares of center of mass distances
    "cmdsqm": _cc_sq_cm_dists_mean,
    #root of mean of squares of center of mass distances
    "cmdrsqm": _cc_sq_cm_dists_mean_sqr
}

def cc_norm_(im, contrast_f, thresh_f, conn_comps_f, morph_op_f, score_f, 
    debug=False):
    """!
    Applies intensity map normalization via connected components method.

    @param im Input intensity map.
    @param contrast_f Contrast function to use.
    @param thresh_f Threshold function to use.
    @param conn_comps_f Connected components function to use.
    @param morph_op_f Morphological operation function to use.
    @param score_f Score (based on connected components) function to use.
    @param debug Returns intermediary images.
    """
    #debug values
    db_vals = None

    #applying contrast operation
    im = contrast_f(im)
    if debug:
        db_vals = [im]

    #converting to uint8 if required
    if im.dtype != np.uint8:
        _im = np.array(cvx.scale(im, 0, 255), dtype=np.uint8)
    else:
        _im = im

    #calculating threshold image
    thr_im = thresh_f(_im)
    if debug:
        db_vals.append(thr_im)

    #applying morphological operation
    thr_im = morph_op_f(thr_im)
    if debug:
        db_vals.append(thr_im)

    #computing connected components
    stats = conn_comps_f(thr_im)
    if debug:
        db_vals.append(stats)

    #getting 1/weight for image
    score = score_f(stats)
    if debug:
        db_vals.append(score)
    
    return db_vals, im/score

def cc_norm(im, 
    thr_type="otsu", thr_args={},
    contrast_type="sq", contrast_args={},
    morph_op_args={},
    cc_args={},
    score_type="cmdrssq", score_args={},
    debug=False):
    """!
    Builds functions and calls #cc_norm_.

    @param im Input intensity map.
    @param thr_type Threshold type to apply. See #THRESH_FUNCS.
    @param thr_args Arguments for threshold function.
    @param contrast_type Type of contrast to apply. See #CONTRAST_FUNCS.
    @param contrast_args Arguments for contrast function.
    @param morph_op_args Arguments for morphological operation.
    @param cc_args Arguments for connected components method.
    @param score_type Type of weight function. See #CC_NORM_SCORE_FUNCS.
    @param score_args Arguments for weight function.
    @param debug Returns intermediate images.
    """
    contrast_f = lambda x: CONTRAST_FUNCS[contrast_type](x, **contrast_args)
    thr_f = lambda x: THRESH_FUNCS[thr_type](x, **thr_args)
    cc_f = lambda x: _cc(x, **cc_args)
    morph_f = lambda x: cvx.morph_op(x, **morph_op_args)
    score_f = lambda x: CC_NORM_SCORE_FUNCS[score_type](x, **score_args)

    return cc_norm_(im, contrast_f, thr_f, cc_f, morph_f, score_f, debug=debug)

def frac_norm_(im, thr_f, debug=False):
    """!
    Applies intensity map normalization via fractional method.    
    
    @param im Input intensity map.
    @param thr_f Threshold function to apply.
    @param debug Returns intermediate images.
    """
    #converting to uint8 if necessary
    if im.dtype != np.uint8:
        _im = np.array(cvx.scale(im, 0, 255), dtype=np.uint8)
    else:
        _im = im

    #debug values
    db_vals = None

    #applying threshold
    _im = thr_f(img)
    if debug:
        db_vals = [_im]
    
    #getting number of pixels with high activations
    whites = cv2.countNonZero(_im)
    #total number of pixels
    size = im.shape[0]*im.shape[1]

    return db_vals, im*(1.0 - whites/size)**2

def frac_norm(im, thr_type="otsu", thr_args={}):
    """!
    Builds functions and calls #frac_norm_.

    @param im input intensity map.
    @param thr_type Threshold type to apply. See #THRESH_FUNCS.
    @param thr_args Arguments for threshold function.
    """
    thr_f = lambda x: THRESH_FUNCS[thr_type](x, **thr_args)

    return frac_norm_(im, thr_f)

##Intensity map normalization available methods.
IM_NORM_FUNCS = {
    #connected components normalization
    "cc": cc_norm,
    #fractional normalization
    "frac": frac_norm
}

def normalize(im, method="cc", **kwargs):
    """!
    Normalizes intensity-map.

    @param im Input intensity map.
    @param method Method to use. See #IM_NORM_FUNCS.
    @param kwargs Other arguments to be passed to chosen methods.
    """
    method = method.lower()

    return IM_NORM_FUNCS[method](im, **kwargs) 

##Available linear combination functions for intensity maps.
IM_COMBINE_FUNCS = {
    "sum": sum
}

def combine(ims, method="sum"):
    """!
    Returns linear combination of sum of intensity maps.

    @param ims Intensity maps list.
    @param method Method to use. See #IM_COMBINE_FUNCS.
    """
    method = method.lower()

    return IM_COMBINE_FUNCS[method](ims)
