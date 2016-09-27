"""!
@package im
@brief Module for intensity map calculations.
"""

import cv2
import cvx
import feat
import numpy as np
from itertools import combinations_with_replacement
from functools import reduce

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
    return map(lambda vecs: dist_f(*vecs), 
        combinations_with_replacement(points, 2))

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
    Adaptative Otsu threshold.
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

def _one(x):
    """!
    Default weight function.
    """
    return 1.0

def _inv(x):
    """!
    Inverse of x + 1.
    """
    return 1/(x + 1)

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
    #making functions to apply
    contrast_f = lambda x: CONTRAST_FUNCS[contrast_type](x, **contrast_args)
    thr_f = lambda x: THRESH_FUNCS[thr_type](x, **thr_args)
    cc_f = lambda x: _cc(x, **cc_args)
    morph_f = lambda x: cvx.morph_op(x, **morph_op_args)
    score_f = lambda x: CC_NORM_SCORE_FUNCS[score_type](x, **score_args)

    #calling higher-order function
    return cc_norm_(im, contrast_f, thr_f, cc_f, morph_f, score_f, debug=debug)

def frac_norm_(im, thr_f, debug=False):
    """!
    Applies intensity map normalization via fractional method.    
    
    @param im Input intensity map.
    @param thr_f Threshold function to apply.
    @param debug Returns intermediate images if True.
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

def _debug_unpack(img, db_imgs_and_imgs_iter, line=False):
    """!
    Gets an iterable with values in format (debug_img, img) and builds a 
    single debug image along with a single image.
    """
    #separating debug images and images
    db_imgs, imgs = zip(*db_imgs_and_imgs_iter)
    #appending all debug images together
    db_imgs = map(lambda xy: cvx.h_append(xy[0], xy[1]), zip(db_imgs, imgs))
    append = lambda x, y: cvx.v_append(x, y, put_line=line)
    db_imgs = reduce(append, db_imgs)

    return db_imgs, sum(imgs)

def _pyr_op(img, op_f, pyr_lvls, weight_f=_one, debug=False):
    """!
    Iterator that applies operation on different levels of pyramid of image.

    @param img Input image.
    @param op_f Operation to apply.
    @param pyr_lvls Pyramids levels to calculate. The original image is 0 level.
    @param weight_f Function to compute weight of map from pyramid level.
    @param debug If True, yields intermediary images.
    """
    #iterating over pyramid levels (from 0 to pyr_lvls)
    for i in range(pyr_lvls+1):
        #getting downsampled image
        img = cv2.pyrDown(img) if i > 0 else img
        #applying operation
        ds_op_img = weight_f(i)*op_f(img)
        #upsampling image again
        us_op_img = cvx.rep_pyrUp(ds_op_img, i)

        yield (ds_op_img if debug else None), us_op_img

def _pyr_op_sum(img, op_f, pyr_lvls=3, weight_f=_one, debug=False,
    prepare=True):
    """!
    Sums up results from operations on various levels of pyramid.
    See #_pyr_op.
    """
    #set suitable dimensions for pyramid if needed
    if prepare:
        img = cvx.pyr_prepare(img, pyr_lvls)

    #getting generator
    pyr_op_iter = _pyr_op(img, op_f=op_f, pyr_lvls=pyr_lvls, weight_f=weight_f,
        debug=debug)

    if debug:
        return _debug_unpack(img, pyr_op_iter)
    else:
        return None, sum(_img for __, _img in pyr_op_iter)

def _cs_op(img, ksizes, weight_f=_one, dtype=np.float32):
    """!
    Generator that appplies center surround with different kernels on image.

    @param img Input image.
    @param ksizes Kernel sizes.
    @param weight_f Function to compute weight of map from kernel size.
    @param dtype Output data type.
    """
    #iterating over kernel sizes
    for ks in ksizes:
        #getting kernel
        kernel = get_center_surround_kernel(ks, dtype)
        #applying operation
        cs_img = center_surround(img, weight_f(ks)*kernel)

        yield cs_img

def _cs_op_sum(img, ksizes=(3, 7), weight_f=_one, dtype=np.float32):
    """!
    Sums up results from #_cs_op.
    """
    return sum(_cs_op(img, ksizes, weight_f, dtype))

def _lab_attr_map(img, 
    pyr_lvls=3, cs_ksizes=(3, 7), pyr_w_f=_one, cs_w_f=_one,
    norm_method="cc", norm_params={},
    dtype=np.float32, debug=False):
    """!
    Gets intensity map of LAB colorspace feature.
    
    @param img Input image.
    @param pyr_lvls Pyramid levels.
    @param cs_ksizes Center-surround kernel sizes.
    @param pyr_w_f Weight function for pyramid level.
    @param cs_w_f Weight function for center-surround kernel size.
    @param dtype Data output type.
    @param debug If True, return intermediary images.
    """
    #center surround function to apply
    cs_f = lambda x: _cs_op_sum(x, cs_ksizes, cs_w_f, dtype=dtype)
    #applying center surround of all kernel sizes to all levels
    db_im, im = _pyr_op_sum(img, op_f=cs_f, pyr_lvls=pyr_lvls, 
        weight_f=pyr_w_f, debug=debug)

    #normalizing map
    db_norm_im, norm_im = normalize(im, method=norm_method, debug=debug, 
        **norm_params)

    #assembling debug image if required
    if debug:
        db_norm_im = [im] + db_norm_im[:-2]
        db_norm_im = reduce(cvx.h_append, 
            map(lambda x: db_im.max()*cvx.normalize(x), db_norm_im))
        db_im = cvx.v_append(db_im, db_norm_im)

    return db_im, norm_im

def _lab_map(img, pyr_lvls=3, cs_ksizes=(3, 7), pyr_w_f=_one, cs_w_f=_one, 
    feats=list(feat.LAB_ATTR_FUNCS.keys()), 
    norm_method="cc", norm_params={},
    dtype=np.float32, debug=False):
    """!
    Gets intensity map of a subset of LAB colorspace features.
    Assumes image comes in BGR.

    @param img Input image.
    @param pyr_lvls Pyramid levels.
    @param cs_ksizes Center-surround kernel sizes.
    @param pyr_w_f Weight function for pyramid level.
    @param cs_w_f Weight function for center-surround kernel size.
    @param feats LAB features to use. See #feat.LAB_ATTR_FUNCS.
    @param norm_method Method for maps normalization. See #IM_NORM_FUNCS.
    @param norm_params Additional normalization parameters.
    @param dtype Data output type.
    @param debug If True, return intermediary images.
    """
    #getting maps
    feat_iter = (feat.get_lab_attr_map(img, attr=ft, cvt=True) \
        for ft in feats)

    #getting center-surround on multiple kernel sizes and multiple pyramid
    #levels for each map
    pyr_w_f(1)
    map_iter = (_lab_attr_map(_img, pyr_lvls=pyr_lvls, cs_ksizes=cs_ksizes,
        norm_method=norm_method, norm_params=norm_params,
        pyr_w_f=pyr_w_f, cs_w_f=cs_w_f, dtype=dtype, debug=debug) \
        for _img in feat_iter)

    if debug:
        return _debug_unpack(img, map_iter, line=True)
    else:
        return None, sum(_img for __, _img in map_iter)

def color_map(img, pyr_lvls=3, cs_ksizes=(3, 7), pyr_w_f=_one, cs_w_f=_one, 
    norm_method="cc", norm_params={},
    colors=feat.LAB_COLORS, dtype=np.float32, debug=False):
    """!
    Computes color intensity map. See #_lab_map.
    Assumes image comes in BGR.
    
    @param colors Colors to use in map calculation.
    """
    return _lab_map(img, pyr_lvls=pyr_lvls, cs_ksizes=cs_ksizes, 
        pyr_w_f=pyr_w_f, cs_w_f=cs_w_f, 
        norm_method=norm_method, norm_params=norm_params,
        feats=colors, dtype=dtype, debug=debug)

def contrast_map(img, pyr_lvls=3, cs_ksizes=(3, 7), pyr_w_f=_one, cs_w_f=_one, 
    norm_method="cc", norm_params={},
    colors=feat.LAB_CONTRASTS, dtype=np.float32, debug=False):
    """!
    Computes contrast intensity map. See #_lab_map.
    Assumes image comes in BGR.
    
    @param colors Intensities to use in map calculation.
    """
    return _lab_map(img, pyr_lvls=pyr_lvls, cs_ksizes=cs_ksizes, 
        pyr_w_f=pyr_w_f, cs_w_f=cs_w_f, 
        norm_method=norm_method, norm_params=norm_params,
        feats=colors, dtype=dtype, debug=debug)

def _single_orientation_map(img, orientation, pyr_lvls=3, pyr_w_f=_one, 
    norm_method="cc", norm_params={},
    dtype=np.float32, debug=False, **kwargs):
    """!
    Gets intensity map of orientation feature.
    
    @param img Input image.
    @param orientation Orientation to compute intensity. 
    @param pyr_lvls Pyramid levels.
    @param pyr_w_f Pyramid weight function.
    @param norm_method Normalization method.
    @param norm_params Additional normalization parameters.
    @param dtype Data type. 
    @param debug If True, returns intermediary images.
    @param kwargs Additional parameters for feat.get_orientation_map.
    """
    #getting orientation map
    or_f = lambda _img: feat.get_orientation_map(_img, orientation, **kwargs)
    db_im, im = _pyr_op_sum(img, or_f, pyr_lvls, pyr_w_f, dtype, debug) 
    
    #normalizing
    db_norm_im, norm_im = normalize(im, method=norm_method, debug=debug,
        **norm_params)

    #assembling debug image if required
    if debug:
        db_norm_im = [im] + db_norm_im[:-2]
        db_norm_im = reduce(cvx.h_append, 
            map(lambda x: db_im.max()*cvx.normalize(x), db_norm_im))
        db_im = cvx.v_append(db_im, db_norm_im)

    return db_im, norm_im

def orientation_map(img, orientations=list(feat.ORIENTATIONS.keys()),
    norm_method="cc", norm_params={},
    pyr_lvls=3, pyr_w_f=_one, dtype=np.float32, debug=False, **kwargs):
    """!
    Gets intensity map of all orientation features.
    Assumes image comes in BGR.
    
    @param img Input image.
    @param orientations Orientations list.
    @param norm_method Normalization method.
    @param norm_params Additional normalization parameters.
    @param pyr_lvls Pyramid levels.
    @param dtype Data type.
    @param debug If True, returns intermediary images.
    @param kwargs Additional parameters for #_single_orientation_map.
    """ 
    #all maps iterator
    map_iter = (_single_orientation_map(img, ort, pyr_lvls=pyr_lvls, 
        norm_method=norm_method, norm_params=norm_params,
        pyr_w_f=pyr_w_f, dtype=dtype, debug=debug, **kwargs) \
        for ort in orientations)

    #combining result
    if debug:
        return _debug_unpack(img, map_iter, line=True)
    else:
        return sum(_img for __, _img in map_iter)

##Available saliency map functions.
MAP_FUNCTIONS = {
    "col": color_map,
    "cst": contrast_map,
    "ort": orientation_map,
}

def weighted_sum(maps):
    """!
    Performs a weighted sum of maps to produce final saliency map. 
    The weight of each map is 1/n, where n is the number of features in map.

    @param maps Intensity maps in format (color, contrast, orientation).
    """
    color_map, contrast_map, orientation_map = maps

    imap = np.zeros(shape=color_map.shape, dtype=color_map.dtype)

    if color_map is not None:
        imap += color_map/4
    if contrast_map is not None:
        imap += contrast_map/2
    if orientation_map is not None:
        imap += orientation_map/4

    return imap

##Available linear combination functions for intensity maps.
IM_COMBINE_FUNCS = {
    "sum": sum,
    "wsum": weighted_sum
}

def combine(maps, method="wsum"):
    """!
    Returns linear combination of sum of intensity maps.

    @param ims Intensity maps list.
    @param method Method to use. See #IM_COMBINE_FUNCS.
    """
    method = method.lower()

    return IM_COMBINE_FUNCS[method](maps)
