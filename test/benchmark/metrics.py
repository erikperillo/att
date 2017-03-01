#!/usr/bin/env python3

import scipy.io as sio
import numpy as np
import subprocess as sp
import os.path as op
import sys
import cv2

def std_norm(arr):
    sigma = arr.std()
    u = arr.mean()
    return (arr - u)/(sigma if sigma else 1)

def load_and_fit_dims(img_filepath_1, img_filepath_2,
    dtype=np.float32, load_code=0):
    img_1 = np.array(cv2.imread(img_filepath_1, load_code), dtype=dtype)
    img_2 = np.array(cv2.imread(img_filepath_2, load_code), dtype=dtype)

    if img_1.shape != img_2.shape:
        img_1 = cv2.resize(img_1, img_2.shape[:2][::-1])

    return img_1, img_2

def normalize(arr):
    return (arr - arr.min())/(arr.max() - arr.min())

def confusion_mtx(imap, bin_mask, divs=100, neg_bin_mask=None):
    imap = normalize(imap)

    positives = bin_mask.sum()
    if neg_bin_mask is None:
        neg_bin_mask = ~bin_mask
        negatives = bin_mask.shape[0]*bin_mask.shape[1] - positives
    else:
        neg_bin_mask &= ~bin_mask
        negatives = neg_bin_mask.sum()

    #print("pos={}, neg={}".format(positives, negatives))
    for thr in np.linspace(0.0, 1.0, divs):
        thr_imap = imap > thr
        tp = (thr_imap & bin_mask).sum()
        tn = (~thr_imap & neg_bin_mask).sum()
        fp = negatives - tn
        fn = positives - tp

        yield (tp, fp, tn, fn)
        #print("thr={}: tp={}, fp={}, tn={}, fn={}".format(thr, tp, fp, tn, fn))

def _auc_judd(imap, pts, divs=128):
    tpr = []
    fpr = []

    for tp, fp, tn, fn in confusion_mtx(imap, pts > 0, divs):
        tpr.append(tp/(tp + fn))
        fpr.append(fp/(fp + tn))

    auc = np.trapz(tpr[::-1], fpr[::-1]) + (1 - max(fpr))

    return auc

def auc_judd(map_filepath, pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    score = _auc_judd(map_img, pts_img)

    return score

def _auc_shuffled(imap, pts, other_pts, divs=128):
    tpr = []
    fpr = []

    for tp, fp, tn, fn in confusion_mtx(imap, pts > 0, divs, other_pts > 0):
        tpr.append(tp/(tp + fn))
        fpr.append(fp/(fp + tn))

    auc = np.trapz(tpr[::-1], fpr[::-1]) + (1 - max(fpr))

    return auc

def auc_shuffled(map_filepath, pts_filepath, other_pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    other_pts_img, __ = load_and_fit_dims(other_pts_filepath, pts_filepath)
    score = _auc_shuffled(map_img, pts_img, other_pts_img)
    return score

def _nss(sal_map, gt_pts_map):
    sal_map = std_norm(sal_map)
    gt_pts_map = gt_pts_map > 0
    return (sal_map*gt_pts_map).sum()/gt_pts_map.sum()

def nss(map_filepath, pts_filepath):
    map_img, pts_img = load_and_fit_dims(map_filepath, pts_filepath)
    score = _nss(map_img, pts_img)

    return score

def _cc(sal_map, gt_sal_map):
    sal_map = std_norm(sal_map)
    gt_sal_map = std_norm(gt_sal_map)

    return (sal_map*gt_sal_map).mean()

def cc(map_filepath, gt_map_filepath):
    map_img, gt_map_img = load_and_fit_dims(map_filepath, gt_map_filepath)
    score = _cc(map_img, gt_map_img)

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
    score = auc_shuffled("map.jpg", "pts.jpg", "other_pts.jpg")
    print("done. score =", score)
    print()
    exit()

    print("in nss...")
    score = nss("map.jpg", "pts.jpg")
    print("done. score =", score)
    print()

    print("in cc...")
    score = cc("map.jpg", "other_map.jpg")
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

#metrics to use
METRICS_FUNCS = {
    "auc_judd": auc_judd,
    "auc_shuffled": auc_shuffled,
    "nss": nss,
    "mae": mae,
    "cc": cc,
    "sim": sim
}

def compute():
    if len(sys.argv) < 2:
        print("usage: metrics.py <metric> <map> [other_args]")
        exit()

    #executing metric
    metric = sys.argv[1]
    try:
        score = METRICS_FUNCS[metric](*sys.argv[2:])
        print("%.6f" % score)
    except:
        print("_".join("(ERROR:[{}:{}])".format(sys.exc_info()[0].__name__,
            sys.exc_info()[1]).split()))

def main():
    #test()
    compute()

if __name__ == "__main__":
    main()
