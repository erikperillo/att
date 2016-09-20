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
import oarg

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

def mark(img, dilation_ksize=21, frac=0.8, color=(0, 0, 255)):
    """
    Marks whitest areas on image.
    """
    if img.dtype != np.uint8:
        img = np.array(cvx.scale(img, 0, 255), dtype=np.uint8)

    #getting most salient areas
    #__, thr_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    __, thr_img = cv2.threshold(img, int(frac*255), 255, cv2.THRESH_BINARY)
    thr_img = cvx.morph_op(thr_img, 
        erosion_ksize=None, dilation_ksize=dilation_ksize)
    *__, stats, __ = cv2.connectedComponentsWithStats(thr_img)

    #drawing rectangles on salient areas
    if(len(img.shape) < 3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h, __) in stats: 
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    return img

def main():
    #command-line arguments
    img_file = oarg.Oarg("-i --img", "", "path to image", 0) 
    pyr_lvl = oarg.Oarg("-p --pyr-lvl", 3, "levels for pyramid") 
    #must be comma-separated
    str_cs_ksizes= oarg.Oarg("-c --cs-ks", "3,7", 
        "center-surround kernel sizes") 
    #must be comma-separated
    str_feats = oarg.Oarg("-f --features", 
        "l1,l0,r,g,b,y,hor,ver,l_diag,r_diag", "features to use")
    max_w = oarg.Oarg("-W --max-w", 800, "maximum width for image")
    max_h = oarg.Oarg("-H --max-h", 600, "maximum hwight for image")
    blur_ksize = oarg.Oarg("-b --blur-ksize", 5, 
        "gaussian kernel size for blur")
    debug = oarg.Oarg("-d --debug", True, "debug mode")
    list_fts = oarg.Oarg("-l --list-features", False, "list available features")
    save_dir = oarg.Oarg("-s --save-dir", ".", "directory to save images")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    #parsing args
    oarg.parse(delim=":")

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
    img = cvx.pyr_prepare(img, pyr_lvl.val)
    #pre-processing image
    img = pre_proc(img, {"ksize": 2*(blur_ksize.val,)})

    #displaying original image
    cvx.display(img, "original image", False)

    #getting center-surround kernel sizes
    cs_ksizes = str_to_list(str_cs_ksizes.val, int)
    feats = str_to_list(str_feats.val, str)

    #getting file name without extension
    f_name = ".".join(os.path.basename(img_file.val).split(".")[:-1])

    ims = []
    norm_ims = []
    #computing intensity maps
    for feature in feats:
        print("\ton feature '%s' ..." % feature)
        input_img = ft.get_feature(img, feature)
        db_im, imap = im.intensity_map(input_img, pyr_lvl.val, cs_ksizes,    
            debug=debug.val)

        #normalizing map
        norm_db, norm_imap = im.normalize(imap, method="cc", 
            morph_op_args={"erosion_ksize": None, "dilation_ksize": 21,
                "opening": True},
            score_type="cmdrsqm",
            debug=True)
 
        #displaying images
        cvx.display(input_img, "input image %s" % feature)
        cvx.display(imap, "intensity map %s" % feature)
        if debug.val:
            cvx.display(db_im, "intermediary intensity_map %s" % feature)
            ctr_im, __, thr_im, __, score = norm_db
            print("\t\timap norm weight: 1/%f = %f" % (score, 1/score))
            cvx.display(ctr_im, "contrast for normed feature %s" % feature)
            cvx.display(thr_im, "conn comps in normed feature %s" % feature)

        #appending partial intensity map
        ims.append(imap)
        norm_ims.append(norm_imap)

        #saving images
        if save_dir.val:
            cvx.save(imap, "%s/%s_%s_map.png" % (save_dir.val, f_name, feature))
            if debug.val:
                cvx.save(db_im, "%s/%s_%s_db_map.png" % \
                    (save_dir.val, f_name, feature))

    #computing final intensity map
    final_im = sum(ims)
    final_im_norm = sum(norm_ims)

    #displaying final result
    cvx.display(final_im, "final intensity map")
    cvx.display(final_im_norm, "final intensity map normalized")
    cvx.display(mark(final_im_norm), "selected regions", False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #saving final result
    if save_dir.val:
        cvx.save(final_im, "%s/%s_final_map.png" % (save_dir.val, f_name))

if __name__ == "__main__":
    main()
