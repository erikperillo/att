#!/usr/bin/env python3

import cv2
import os
import glob
import numpy as np
import pickle
import random
import gzip

INP_SHAPE = (58, 98)

STIMULI_DIR = "/home/erik/proj/ic/sal_benchmarks/bms/CAT2000/trainSet/Stimuli"
MAPS_DIR = "/home/erik/proj/ic/sal_benchmarks/bms/CAT2000/trainSet/FIXATIONMAPS"
OUT_DATA_FILEPATH = "./data/cat2000.gz"
#OUT_DATA_FILEPATH = "./data/test.pkl"

def normalize(data):
    return (data - data.mean())/data.std()

def bgr_to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

def swapax(img):
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

def augment(img):
    try:
        mirr = img[:, ::-1, :]
    except IndexError:
        mirr = img[:, ::-1]
    return (mirr,)

def files_to_mtx(dir_path, do_norm=True, do_augment=True, files_ext="*.jpg"):
    x = []
    y = []

    filepaths = glob.glob(os.path.join(dir_path, files_ext))
    for fp in filepaths:
        print("in", fp, "...")

        #reading image
        img = cv2.imread(fp)
        img = bgr_to_lab(img)

        #reading saliency map respective to image
        fn = os.path.basename(fp)
        sal_map_fp = os.path.join(MAPS_DIR, fn)
        sal_map = cv2.imread(sal_map_fp, 0)

        #resizing if necessary
        if img.shape != INP_SHAPE:
            old_shape = img.shape
            img = cv2.resize(img, INP_SHAPE[::-1])
            sal_map = cv2.resize(sal_map, INP_SHAPE[::-1])
            new_shape = img.shape
            print("\tWARNING: resized image/map from {} to {}".format(
                old_shape, new_shape))

        x.append(swapax(img).flatten())
        y.append(sal_map.flatten())
        if do_augment:
            for augm in augment(img):
                x.append(swapax(augm).flatten())
            for augm in augment(sal_map):
                y.append(augm.flatten())

    #creating matrices
    x_mtx = np.array(x, dtype=img.dtype)
    y_mtx = np.array(y, dtype=sal_map.dtype)
    if do_norm:
        x_mtx = normalize(x_mtx)

    #shuffling data...
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x_mtx = x_mtx[indexes]
    y_mtx = y_mtx[indexes]

    print("created x, y of shapes", x_mtx.shape, y_mtx.shape)

    return x_mtx, y_mtx

def main():
    print("reading files...")
    x, y = files_to_mtx(STIMULI_DIR)

    print("saving...")
    with gzip.open(OUT_DATA_FILEPATH, "wb") as f:
        pickle.dump((x, y), f)
    print("done.")

if __name__ == "__main__":
    main()
