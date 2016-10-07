#!/usr/bin/env python3

import scipy.io as sio
import numpy as np
import subprocess as sp
import os.path as op
import cv2

AUC_JUDD_SRC = "AUC_Judd.m"
AUC_SHUFFLED_SRC = "AUC_shuffled.m"
CC_SRC = "CC.m"
NSS_SRC = "NSS.m"
SIM_SRC = "similarity.m"
MATLAB_CMD = "octave --eval"

def _matlab_cmd(cmd_str):
    ret = sp.run(MATLAB_CMD.split() + [cmd_str], stdout=sp.PIPE, stderr=sp.PIPE)
    output = ret.stdout.decode()    
    #print(ret)
    return output

def matlab_cmd(cmd_str):
    output = _matlab_cmd(cmd_str)
    return output.split("=")[-1].strip()

def std_norm(arr):
    sigma = arr.std()
    u = arr.mean()
    return (arr - u)/(sigma if sigma else 1)

def load_and_fit_dims(img_filepath_1, img_filepath_2, 
    dtype=float, load_code=0):
    img_1 = np.array(cv2.imread(img_filepath_1, load_code), dtype=dtype)
    img_2 = np.array(cv2.imread(img_filepath_2, load_code), dtype=dtype)

    if img_1.shape != img_2.shape:
        img_1 = cv2.resize(img_1, img_2.shape[:2][::-1]) 

    return img_1, img_2 

def auc_judd(map_filepath, pts_filepath, jitter=1, to_plot=0):
    cmd = "; ".join([
        "addpath('{}')",
        "map = double(imread('{}'))",
        "pts = double(imread('{}'))",
        "[score, tp, fp, treshs] = AUC_Judd(map, pts>0, {}, {})",
        "score"
        ]).format(op.dirname(AUC_JUDD_SRC),
            map_filepath, 
            pts_filepath,
            jitter, to_plot)

    print("executing command '%s'" % cmd)
    score = float(matlab_cmd(cmd))

    return score

def auc_shuffled(map_filepath, pts_filepath, other_pts_filepath,
    n_splits=100, step_size=0.1, to_plot=0):
    cmd = "; ".join([
        "addpath('{}')",
        "map = double(imread('{}'))",
        "pts = double(imread('{}'))",
        "other_pts = double(imread('{}'))",
        "[score, tp, fp] = AUC_shuffled(map, pts>0, other_pts>0, {}, {}, {})",
        "score"
        ]).format(op.dirname(AUC_SHUFFLED_SRC),
            map_filepath, 
            pts_filepath,
            other_pts_filepath,
            n_splits, step_size, to_plot)

    print("executing command '%s'" % cmd)
    score = float(matlab_cmd(cmd))

    return score

def nss2(map_filepath, pts_filepath):
    cmd = "; ".join([
        "addpath('{}')",
        "pkg load image",
        "map = double(imread('{}'))",
        "pts = double(imread('{}'))",
        "score = NSS(map, pts>0)",
        "score"
        ]).format(op.dirname(AUC_JUDD_SRC),
            map_filepath, 
            pts_filepath)

    print("executing command '%s'" % cmd)
    score = float(matlab_cmd(cmd))

    return score

def _nss(sal_map, gt_pts_map):
    sal_map = std_norm(sal_map)
    gt_pts_map = gt_pts_map > 0
    
    return (sal_map*gt_pts_map).sum()/gt_pts_map.sum()

def nss(map_filepath, pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    score = _nss(map_img, pts_img)

    return score 
 
def cc2(map_filepath, gt_map_filepath):
    cmd = "; ".join([
        "addpath('{}')",
        "pkg load image",
        "map = double(imread('{}'))",
        "gt_map = double(imread('{}'))",
        "score = CC(map, gt_map)",
        "score"
        ]).format(op.dirname(CC_SRC),
            map_filepath, 
            gt_map_filepath)

    print("executing command '%s'" % cmd)
    score = float(matlab_cmd(cmd))

    return score

def _cc(sal_map, gt_sal_map):
    sal_map = std_norm(sal_map)
    gt_sal_map = std_norm(gt_sal_map)

    return (sal_map*gt_sal_map).mean()

def cc(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    score = _cc(map_img, gt_map_img)

    return score

def sim2(map_filepath, gt_map_filepath, to_plot=0):
    cmd = "; ".join([
        "addpath('{}')",
        "pkg load image",
        "map = double(imread('{}'))",
        "gt_map = double(imread('{}'))",
        "score = similarity(map, gt_map, {})",
        "score"
        ]).format(op.dirname(AUC_JUDD_SRC),
            map_filepath, 
            gt_map_filepath,
            to_plot)

    print("executing command '%s'" % cmd)
    score = float(matlab_cmd(cmd))

    return score

def _sim(sal_map, gt_sal_map):
    sal_map /= sal_map.sum()
    gt_sal_map /= gt_sal_map.sum()

    return np.minimum(sal_map, gt_sal_map).sum()

def sim(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    score = _sim(map_img, gt_map_img)

    return score

def _mae(sal_map, gt_sal_map):
    return abs(sal_map - gt_sal_map).mean()

def mae(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    map_img /= map_img.max()
    gt_map_img /= gt_map_img.max()
    score = _mae(map_img, gt_map_img)

    return score
 
def test():
    print("in auc_judd...")
    score = auc_judd("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

    print("in auc_shuffled...")
    score = auc_shuffled("map.jpg", "pts.jpg", "../code_forMetrics/other_pts.jpg")
    print("done. score =", score)
    print()

    print("in nss2...")
    score = nss2("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

    print("in nss...")
    score = nss("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

    print("in cc2...")
    score = cc2("map.jpg", "other_map.jpg")
    print("done. score =", score)
    print()

    print("in cc...")
    score = cc("map.jpg", "other_map.jpg")
    print("done. score =", score)
    print()

    print("in sim2...")
    score = sim2("map.jpg", "other_map.jpg")
    print("done. score =", score)
    print()

    print("in sim...")
    score = sim("map.jpg", "other_map.jpg")
    print("done. score =", score)
    print()

    print("in mae...")
    score = mae("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

def main():
    test()

if __name__ == "__main__":
    main()
