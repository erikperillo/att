#!/usr/bin/env python3

#grid_search.py
#executes benchmark.sh script for various saliency map parameters

import subprocess as sp
import itertools
import datetime
import shutil
import shelve
import time
import cv2
import sys
import os

#base directory where each directory will be stored
base_dir = "/home/erik/gtest17"

#configuration file path
conf_filepath = "/home/erik/proj/att/att/grid_search_conf.db"

#benchmark script command
bm_cmd = "/home/erik/proj/att/test/benchmark/benchmark.sh"
bm_cmd_flags = ""

#resume whole grid search in csv files
resume = True

#metrics to use in resuming step
metrics = ["fp_auc_judd", "fp_nss", "cm_sim", "cm_cc"]

#resuming command
resume_cmd = "/home/erik/proj/att/test/benchmark/resume.sh"

#parameters for intensity map calculation
params = dict(
    gaussian_blur_params = [
        {
            "ksize": (5, 5),
            "sigmaX": 0.0,
            "sigmaY": 0.0,
            "borderType": cv2.BORDER_DEFAULT
        },
    ],

    im_norm_methods = [
        "cc",
    ],

    im_norm_params = [
        {
            "thr_type": "otsu",
            "contrast_type": "none",
            "score_type": "num",
            "morph_op_args": {
                "erosion_ksize": 3,
                "dilation_ksize": 3,
                "opening": True
            }
        },
    ],

    pyr_w_f = [
        "one",
        "one_over_sqrt_n",
        "sqrt_n"
    ],

    cs_w_f = [
        "one"
    ],

    cs_ksizes = [
        (3, 7),
    ],

    maps = [
        ("col", "cst", "ort"),
    ],

    norm_score_f = [
        "num",
    ],

    col_w = [
        4,
    ],

    cst_w = [
        1,
    ],

    ort_w = [
        1/4,
    ],

    control = [
        False,
    ]
)

def fmt_time(seconds):
    hours = int(seconds)//3600
    minutes = (int(seconds)%3600)//60
    seconds = seconds%60 

    return hours, minutes, seconds

def str_fmt_time(seconds):
    return "%.3dh%.2dm%.2ds" % fmt_time(seconds)

def mk_dir_name():
    return datetime.datetime.now().strftime("%b-%d-%Y_%Hh%Mm%Ss")

def ok_params(params_dict):
    colw = params_dict["col_w"]
    cstw = params_dict["cst_w"]
    ortw = params_dict["ort_w"]

    if colw == 0 and cstw == 0 and ortw == 0:
        return False
    if colw == cstw and cstw == ortw and colw != 1:
        return False
    if colw == 0 and cstw == ortw and cstw != 1:
        return False
    if cstw == 0 and colw == ortw and colw != 1:
        return False
    if ortw == 0 and colw == cstw and colw != 1:
        return False

    return True

def confs_gen(dict_of_lists):
    keys = dict_of_lists.keys()
    values_prod = itertools.product(*dict_of_lists.values())

    for vals in values_prod:
        params = {k: v for k, v in zip(keys, vals)}
        yield params

def mk_conf_file(filepath, conf_dict):
    with shelve.open(filepath) as conf:
        for k, v in conf_dict.items():
            conf[k] = v

def run_bm():
    #making run dir
    out_dir = mk_dir_name()
    out_dir = os.path.join(base_dir, out_dir)

    #making command
    flags = "%s" % bm_cmd_flags
    cmd = bm_cmd.split() + [out_dir] + [flags]

    #running
    ret = sp.run(cmd)

    return out_dir, ret

def grid_search():
    confs = (c for c in confs_gen(params))
    confs = filter(ok_params, confs)

    total_time = 0
    for conf in confs:
        print("generating config file '%s' ..." % conf_filepath)
        mk_conf_file(conf_filepath, conf)

        print("running bm ...")
        start_time = time.time()
        out_dir, __ = run_bm()
        partial_time = int(time.time() - start_time)

        print("[grid_search] run elapsed time: %s" % str_fmt_time(partial_time))
        total_time += partial_time
        print("[grid_search] total elapsed time: %s" % str_fmt_time(total_time))

        #writing configuration to dir
        with open(os.path.join(out_dir, "im_params.txt"), "w") as f:
            for k, v in conf.items():
                print("%s:" % k, v, file=f)

    if resume:
        for m in metrics:
            print("[grid_search] resuming for metric '%s'..." % m)
            ret = sp.run(resume_cmd.split() + [base_dir, m],
                    stdout=sp.PIPE, stderr=sp.PIPE)

            metric_filepath = os.path.join(base_dir, m + ".csv")
            with open(metric_filepath, "w") as f:
                f.write(ret.stdout.decode())

            print("done. saved in '%s'" % metric_filepath)

    print("copying generator script to file '%s'..." %\
            os.path.join(base_dir, "gen_grid_search.py"))
    shutil.copyfile(__file__, os.path.join(base_dir, "gen_grid_search.py"))
    print("done.")

def main():
    if os.path.exists(base_dir):
        print("error: directory '%s' already exists" % base_dir)
        exit()

    grid_search()

if __name__ == "__main__":
    main()
