#!/usr/bin/env python3

"""
Test program for module.
"""

import im
import feat as ft
import numpy as np
import cvx
import cv2
import os
import sys
import oarg
from functools import reduce

#default parameters for gaussian blur
GAUSSIAN_BLUR_DEF_PARAMS = {
    "ksize": (5, 5),
    "sigmaX": 0.0,
    "sigmaY": 0.0,
    "borderType": cv2.BORDER_DEFAULT
}

#intensity map normalization method
IM_NORM_METHOD = "cc"
#intensity map normalization parameters
IM_NORM_PARAMS = {
    "thr_type": "otsu",
    "contrast_type": "sq",
    "score_type": "cmdrssq",
    "morph_op_args": {"erosion_ksize": 5, "dilation_ksize": 31, "opening": True}
}
#pyramid weight function
PYR_W_F = im._one
#center-surround kernel size weight function
CS_W_F = im._one

def error(msg, code=1):
    """
    Prints error message and exits with code.
    """
    print("error:", msg)
    exit(code)

def str_to_list(string, tp, delim=","):
    """
    Divides string separated by delimiter and converts to tp.
    """
    return list(map(tp, string.split(delim)))

def custom_dict(def_dict, custom_params):
    """
    Returns a copy of def_dict with some of its values changed by custom_params.
    """
    new_dict = dict(def_dict)
    new_dict.update(custom_params)

    return new_dict

def pre_proc(img, blur_params):
    """
    Performs pre-processing on image.
    """
    blur_dict = custom_dict(GAUSSIAN_BLUR_DEF_PARAMS, blur_params)

    if blur_dict["ksize"][0] > 1:
        return cv2.GaussianBlur(img, **blur_dict)

    return img

def mask(img, dilation_ksize=31, frac=0.8):
    """
    Marks whitest areas on image.
    """
    if img.dtype != np.uint8:
        img = np.array(cvx.scale(img, 0, 255), dtype=np.uint8)

    #getting most salient areas
    __, thr_img = cv2.threshold(img, int(frac*255), 255, cv2.THRESH_BINARY)
    thr_img = cvx.morph_op(thr_img, 
        erosion_ksize=None, dilation_ksize=dilation_ksize)

    return thr_img

def mark(img, dilation_ksize=31, frac=0.9, color=(0, 0, 255)):
    """
    Marks whitest areas on image.
    """
    thr_img = mask(img, dilation_ksize, frac)
    *__, stats, __ = cv2.connectedComponentsWithStats(thr_img)

    #drawing rectangles on salient areas
    if(len(img.shape) < 3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, __) in stats: 
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    return img

def gabor_kernel_view(kernel, size=100):
    """
    Returns a visualization of the gabor kernel.
    """
    return cv2.resize(kernel, (size, size))

def benchmark():
    bm_img_filepath = oarg.Oarg("-b --bm-img", "", "path to benchmark image", 0) 
    img_filepath = oarg.Oarg("-i --img", "", "path to image", 1) 
    display = oarg.Oarg("-D --display", True, "display images")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    oarg.parse(sys.argv)

    #help message
    if hlp.val:
        oarg.describeArgs("options:", def_val=True)
        exit()

    #checking validity of args
    if not img_filepath.found or not bm_img_filepath.found:
        error("image file not found (use -h for help)")

    bm_img = cv2.imread(bm_img_filepath.val, 0)
    img = cv2.imread(img_filepath.val, 0)

    #checking validity of image
    if img is None or bm_img is None:
        error("could not read image")

    img = cv2.resize(img, bm_img.shape[::-1])
    mask_img = img & bm_img

    #displaying images if required
    if display.val:
        cvx.display(bm_img, "bm_img")
        cvx.display(img, "img")
        cvx.display(mask_img, "mask_img")
        cv2.waitKey(0)

    print("Intersection:", mask_img.any())
    print("Intersection %:", mask_img.sum()/bm_img.sum())

def gabor_test():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0) 
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum hwight for image")
    blur_ksize = oarg.Oarg("-b --blur-ksize", 5, 
        "gaussian kernel size for blur")
    debug = oarg.Oarg("-d --debug", True, "debug mode")
    display = oarg.Oarg("-D --display", True, "display images")
    save_dir = oarg.Oarg("-S --save-dir", ".", "directory to save images")
    view_size = oarg.Oarg("-v --view-size", 100, "visualization size")
    clip = oarg.Oarg("-c --clip", True, "clip filter to positive values")
    #gabor kernel parameters
    ksize = oarg.Oarg("-k --ksize", ft.DEF_GABOR_K_PARAMS["ksize"][0], 
        "gabor kernel size")
    sigma = oarg.Oarg("-s --sigma", ft.DEF_GABOR_K_PARAMS["sigma"], 
        "gabor kernel sigma")
    theta = oarg.Oarg("-t --theta", ft.DEF_GABOR_K_PARAMS["theta"], 
        "gabor kernel theta")
    lambd = oarg.Oarg("-l --lambda", ft.DEF_GABOR_K_PARAMS["lambd"], 
        "gabor kernel lambda")
    gamma = oarg.Oarg("-g --gamma", ft.DEF_GABOR_K_PARAMS["gamma"], 
        "gabor kernel gamma")
    psi = oarg.Oarg("-p --psi", ft.DEF_GABOR_K_PARAMS["psi"], 
        "gabor kernel psi")
    ktype = oarg.Oarg("-T --ktype", ft.DEF_GABOR_K_PARAMS["ktype"], 
        "gabor kernel type")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    #parsing args
    oarg.parse(sys.argv, delim=":")

    #help message
    if hlp.val:
        oarg.describeArgs("options:", def_val=True)
        exit()

    #checking validity of args
    if not img_file.found:
        error("image file not found (use -h for help)")

    img = cv2.imread(img_file.val)

    #checking validity of image
    if img is None:
        error("could not read image")

    print("on file %s" % img_file.val)
    
    #converting to grayscale if needed
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resizing image
    img = cvx.resize(img, max_w.val, max_h.val)
    #pre-processing image
    img = pre_proc(img, {"ksize": 2*(blur_ksize.val,)})

    #building parameter dict
    kernel_params = {
        "ksize": 2*(ksize.val, ),
        "sigma": sigma.val,
        "theta": theta.val,
        "lambd": lambd.val,
        "gamma": gamma.val,
        "psi": psi.val,
        "ktype": ktype.val
    }

    if debug.val:
        print("gabor kernel params:", kernel_params)

    #getting gabor kernel
    kernel = ft._get_gabor_kernel(**kernel_params)
    kernel_view = gabor_kernel_view(kernel, view_size.val)

    #filtering image
    gabor_img = ft.gabor_filter(img, kernel_params, cvt=False, clip=clip.val)

    #displaying original image
    if display.val:
        cvx.display(img, "original image", False)
        cvx.display(kernel_view, "kernel view")
        cvx.display(gabor_img, "gabor image")

    if display.val:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def im_test():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0) 
    pyr_lvls = oarg.Oarg("-p --pyr-lvls", 3, "levels for pyramid") 
    #must be comma-separated
    str_cs_ksizes= oarg.Oarg("-c --cs-ks", "3,7", 
        "center-surround kernel sizes") 
    #must be comma-separated
    str_maps = oarg.Oarg("-m --maps", "col,cst,ort", "maps to use")
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum hwight for image")
    blur_ksize = oarg.Oarg("-b --blur-ksize", 5, 
        "gaussian kernel size for blur")
    debug = oarg.Oarg("-d --debug", True, "debug mode")
    display = oarg.Oarg("-D --display", True, "display images")
    list_fts = oarg.Oarg("-l --list-features", False, "list available features")
    save_dir = oarg.Oarg("-s --save-dir", ".", "directory to save images")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    #parsing args
    oarg.parse(sys.argv, delim=":")

    #help message
    if hlp.val:
        oarg.describeArgs("options:", def_val=True)
        exit()

    if list_fts.val:
        print("features:", ", ".join(ft.get_available_features()))
        exit()

    #checking validity of args
    if not img_file.found:
        error("image file not found (use -h for help)")

    img = cv2.imread(img_file.val)

    #checking validity of image
    if img is None:
        error("could not read image")
    if len(img.shape) < 3:
        error("image must be colored")

    print("on file %s" % img_file.val)

    #resizing image
    img = cvx.resize(img, max_w.val, max_h.val)
    #adapting image dimensions for proper pyramid up/downscaling
    img = cvx.pyr_prepare(img, pyr_lvls.val)
    #pre-processing image
    img = pre_proc(img, {"ksize": 2*(blur_ksize.val,)})

    #displaying original image
    if display.val:
        cvx.display(img, "original image", False)

    #getting center-surround kernel sizes
    cs_ksizes = str_to_list(str_cs_ksizes.val, int)
    maps = str_to_list(str_maps.val, str)

    #getting file name without extension
    f_name = ".".join(os.path.basename(img_file.val).split(".")[:-1])

    imaps = {"col": None, "cst": None, "ort": None}
    #color intensity map
    if "col" in maps:
        col_db, col_imap = im.color_map(img, 
            pyr_lvls=pyr_lvls.val, cs_ksizes=cs_ksizes, 
            pyr_w_f=PYR_W_F, cs_w_f=CS_W_F,
            norm_method=IM_NORM_METHOD, norm_params=IM_NORM_PARAMS,
            debug=debug.val)
        imaps["col"] = col_imap
        if debug.val:
            imaps["col_db"] = col_db
    #contrast intensity map
    if "cst" in maps:
        cst_db, cst_imap = im.contrast_map(img, 
            pyr_lvls=pyr_lvls.val, cs_ksizes=cs_ksizes, 
            pyr_w_f=PYR_W_F, cs_w_f=CS_W_F,
            norm_method=IM_NORM_METHOD, norm_params=IM_NORM_PARAMS,
            debug=debug.val)
        imaps["cst"] = cst_imap
        if debug.val:
            imaps["cst_db"] = cst_db
    #orientation intensity map
    if "ort" in maps:
        ort_db, ort_imap = im.orientation_map(img, 
            pyr_lvls=pyr_lvls.val, pyr_w_f=PYR_W_F,
            norm_method=IM_NORM_METHOD, norm_params=IM_NORM_PARAMS,
            debug=debug.val)
        imaps["ort"] = ort_imap
        if debug.val:
            imaps["ort_db"] = ort_db

    #getting final saliency map
    final_im = im.combine([imaps["col"], imaps["cst"], imaps["ort"]])
    #getting saliency mask
    final_im_mask = mask(final_im)

    #saving final result
    if save_dir.val:
        for name, imap in imaps.items():
            if imap is not None:
                cvx.save(imap, "%s/%s_%s_map.png" % \
                    (save_dir.val, f_name, name))
        cvx.save(final_im, "%s/%s_final_map.png" % (save_dir.val, f_name))
        cvx.save(final_im_mask, 
            "%s/%s_final_map_mask.png" % (save_dir.val, f_name))

    #displaying results if required
    if display.val:
        for name, imap in imaps.items():
            if imap is not None:
                cvx.display(imap, name)
        cvx.display(final_im, "final im")
        cvx.display(final_im_mask, "final im mask")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def feat_test():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0) 
    #must be comma-separated
    str_feats = oarg.Oarg("-f --features", 
        "l1,l0,r,g,b,y,hor,ver,l_diag,r_diag", "features to use")
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum hwight for image")
    blur_ksize = oarg.Oarg("-b --blur-ksize", 5, 
        "gaussian kernel size for blur")
    display = oarg.Oarg("-D --display", True, "display images")
    list_fts = oarg.Oarg("-l --list-features", False, "list available features")
    save_dir = oarg.Oarg("-s --save-dir", ".", "directory to save images")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    #parsing args
    oarg.parse(sys.argv, delim=":")

    #help message
    if hlp.val:
        oarg.describeArgs("options:", def_val=True)
        exit()

    if list_fts.val:
        print("features:", ", ".join(ft.get_available_features()))
        exit()

    #checking validity of args
    if not img_file.found:
        error("image file not found (use -h for help)")

    img = cv2.imread(img_file.val)

    #checking validity of image
    if img is None:
        error("could not read image")
    if len(img.shape) < 3:
        error("image must be colored")

    print("on file %s" % img_file.val)

    #resizing image
    img = cvx.resize(img, max_w.val, max_h.val)
    #pre-processing image
    img = pre_proc(img, {"ksize": 2*(blur_ksize.val,)})

    #getting center-surround kernel sizes
    feats = str_to_list(str_feats.val, str)

    #getting file name without extension
    f_name = ".".join(os.path.basename(img_file.val).split(".")[:-1])

    #displaying original image
    if display.val:
        cvx.display(img, "original image", False)
    #saving original image
    if save_dir.val:
        cvx.save(img, "%s/%s_original.png" % (save_dir.val, f_name))

    #computing features
    for feature in feats:
        print("\ton feature '%s' ..." % feature)
        feat_img = ft.get_feature(img, feature)
 
        #displaying images
        if display.val:
            cvx.display(feat_img, "feature %s" % feature)
        #saving images
        if save_dir.val:
            cvx.save(feat_img, "%s/%s_%s.png" % (save_dir.val, f_name, feature))

    if display.val:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("usage: test.py <test> [params]")
        exit()

    test = sys.argv[1]
    sys.argv = sys.argv[2:]

    #intensity maps test
    if test == "im":
        im_test()
    #gabor filter test
    elif test == "gabor":
        gabor_test()
    #features test
    elif test == "feat":
        feat_test()
    #comparison between model-made mask and benchmark mask
    elif test == "bm":
        benchmark()
    else:
        print("unknown test '%s' (use %s)" %\
             (test, " or ".join(("im", "gabor"))))

if __name__ == "__main__":
    main()
