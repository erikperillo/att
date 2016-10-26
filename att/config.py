#!/usr/bin/env python3

#this utility is for generating configuration for test.py program.

import cv2
import shelve

OUT_FILENAME = "config.db"

PARAMS = dict(
    gaussian_blur_params = {
        "ksize": (5, 5),
        "sigmaX": 0.0,
        "sigmaY": 0.0,
        "borderType": cv2.BORDER_DEFAULT
    },

    #intensity map normalization method
    im_norm_method = "cc",

    #intensity map normalization parameters
    im_norm_params = {
        "thr_type": "otsu",
        "contrast_type": "none",
        "score_type": "num",
        "morph_op_args": {
            "erosion_ksize": 3,
            "dilation_ksize": 3,
            "opening": True
        }
    },

    #pyramid weight function
    pyr_w_f = "one",
    #center-surround kernel size weight function
    cs_w_f = "one",

    cs_ksizes = (3, 7),
    pyr_lvls = 3,

    maps = ("col", "cst", "ort"),

    norm_score_f = "num",

    col_w = 4,
    cst_w = 1,
    ort_w = 1/4,

    control = False
)

def gen_conf(filepath, params):
    with shelve.open(filepath) as conf:
        for k, v in params.items():
            conf[k] = v

def main():
    gen_conf(OUT_FILENAME, PARAMS)
    print("config written to '%s'" % OUT_FILENAME)

if __name__ == "__main__":
    main()
