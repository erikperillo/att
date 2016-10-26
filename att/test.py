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
import shelve
from functools import reduce

WEIGHT_FUNCS = {
    "one": lambda n: 1,
    "n": lambda n: n + 1,
    "one_over_n": lambda n: 1/(n + 1),
    "one_over_sqrt_n": lambda n: 1/np.sqrt(n + 1),
    "sqrt_n": lambda n: np.sqrt(n + 1)
}

#default parameters for gaussian blur
GAUSSIAN_BLUR_DEF_PARAMS = {
    "ksize": (5, 5),
    "sigmaX": 0.0,
    "sigmaY": 0.0,
    "borderType": cv2.BORDER_DEFAULT
}

def error(msg, code=1):
    """
    Prints error message and exits with code.
    """
    print("error:", msg)
    exit(code)

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

def gabor_kernel_view(kernel, size=100):
    """
    Returns a visualization of the gabor kernel.
    """
    return cv2.resize(kernel, (size, size))

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
    config_file = oarg.Oarg("-c --config-file", "config.db",
        "path to config file")
    max_w = oarg.Oarg("-W --maxw", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --maxh", 600, "maximum hwight for image")
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

    if not config_file.val:
        error("configuration file must be specified (use -h for help)")

    #configuring parameters
    with shelve.open(config_file.val) as conf:
        gaussian_blur_params = conf["gaussian_blur_params"]
        im_norm_method = conf["im_norm_method"]
        im_norm_params = conf["im_norm_params"]
        pyr_w_f = WEIGHT_FUNCS[conf["pyr_w_f"]]
        cs_w_f = WEIGHT_FUNCS[conf["cs_w_f"]]
        cs_ksizes = conf["cs_ksizes"]
        maps = conf["maps"]
        norm_score_f = conf["norm_score_f"]
        col_w = conf["col_w"]
        cst_w = conf["cst_w"]
        ort_w = conf["ort_w"]
        control = conf["control"]
        pyr_lvls = conf["pyr_lvls"]

    #print("on file %s" % img_file.val)

    #resizing image
    img = cvx.resize(img, max_w.val, max_h.val)
    #adapting image dimensions for proper pyramid up/downscaling
    img = cvx.pyr_prepare(img, pyr_lvls)
    #pre-processing image
    img = pre_proc(img, gaussian_blur_params)

    #displaying original image
    if display.val:
        cvx.display(img, "original image", False)

    #getting file name without extension
    f_name = ".".join(os.path.basename(img_file.val).split(".")[:-1])

    imaps = {"col": None, "cst": None, "ort": None}
    if control:
        final_im = np.random.rand(*img.shape[:2])
        final_im -= final_im.min()
    else:
        #color intensity map
        if "col" in maps:
            col_db, col_imap = im.color_map(img,
                pyr_lvls=pyr_lvls, cs_ksizes=cs_ksizes,
                pyr_w_f=pyr_w_f, cs_w_f=cs_w_f,
                norm_method=im_norm_method, norm_params=im_norm_params,
                debug=debug.val)
            imaps["col"] = col_imap
            if debug.val:
                imaps["col_db"] = col_db
        #contrast intensity map
        if "cst" in maps:
            cst_db, cst_imap = im.contrast_map(img,
                pyr_lvls=pyr_lvls, cs_ksizes=cs_ksizes,
                pyr_w_f=pyr_w_f, cs_w_f=cs_w_f,
                norm_method=im_norm_method, norm_params=im_norm_params,
                debug=debug.val)
            imaps["cst"] = cst_imap
            if debug.val:
                imaps["cst_db"] = cst_db
        #orientation intensity map
        if "ort" in maps:
            ort_db, ort_imap = im.orientation_map(img,
                pyr_lvls=pyr_lvls, pyr_w_f=pyr_w_f,
                norm_method=im_norm_method, norm_params=im_norm_params,
                debug=debug.val)
            imaps["ort"] = ort_imap
            if debug.val:
                imaps["ort_db"] = ort_db

        #getting final saliency map
        final_im = im.combine([
            col_w*imaps["col"] if imaps["col"] is not None else None,
            cst_w*imaps["cst"] if imaps["cst"] is not None else None,
            ort_w*imaps["ort"] if imaps["ort"] is not None else None])

    #saving final result
    if save_dir.val:
        for name, imap in imaps.items():
            if imap is not None:
                cvx.save(imap, "%s/%s_%s_map.png" % \
                    (save_dir.val, f_name, name))
        cvx.save(final_im, "%s/%s_final_map.png" % (save_dir.val, f_name))

    #displaying results if required
    if display.val:
        for name, imap in imaps.items():
            if imap is not None:
                cvx.display(imap, name)
        cvx.display(final_im, "final im")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def feat_test():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0)
    #must be comma-separated
    str_feats = oarg.Oarg("-f --features",
        "l1,l0,r,g,b,y,hor,ver,l_diag,r_diag", "features to use")
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum height for image")
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
    else:
        print("unknown test '%s' (use %s)" %\
             (test, " or ".join(("im", "gabor", "feat"))))

if __name__ == "__main__":
    main()
