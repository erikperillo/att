#!/usr/bin/env python3

import os
import glob
import numpy as np
import pickle
import random
from skimage import transform as tf, io, color, img_as_float
import gzip

#conversions from rgb to...
COL_CVT_FUNCS = {
    "lab": color.rgb2lab,
    "hsv": color.rgb2hsv,
    "luv": color.rgb2luv
}
COL_DCVT_FUNCS = {
    "lab": color.lab2rgb,
    "hsv": color.hsv2rgb,
    "luv": color.luv2rgb,
    "rgb": lambda x: x
}

#paths
DATASET_PATH = "/home/erik/proj/ic/saliency_datasets/judd"
OUT_DATA_FILEPATH = "./data/cat2000.gz"

#show images
SHOW_RAND_IMGS = 3

#image shape
SHAPE = (76, 100)

#float datatype
X_IMG_TO_FLOAT = True
Y_IMG_TO_FLOAT = True

#normalization
X_IMG_NORMALIZATION = None
Y_IMG_NORMALIZATION = None
X_SET_NORMALIZATION = "unit"
Y_SET_NORMALIZATION = "unit"

#input colorspace
X_IMG_COLSPACE = "lab"

#swap channel axis, eg. from shape (200, 200, 3) to (3, 200, 200)
SWAP_CHANNEL_AXIS = True

#augmentation techniques
AUGMENT = True
HOR_MIRROR = True
VER_MIRROR = False
AFFINE_TRANSFORMS = [
    #{
    #    "shear": 0.3,
    #}
]
TL_CORNER = 0.666
TR_CORNER = 0.666
BL_CORNER = 0.666
BR_CORNER = 0.666

def get_stimuli_paths(dataset_path, dataset_name=""):
    if not dataset_name:
        dataset_name = os.path.basename(dataset_path).lower()

    if dataset_name == "judd":
        filepaths = glob.glob(os.path.join(dataset_path, "stimuli", "*.jpeg"))
    elif dataset_name == "cat2000":
        filepaths = glob.glob(os.path.join(
            dataset_path, "trainSet", "Stimuli", "*.jpg"))
    elif dataset_name == "cssd":
        filepaths = glob.glob(os.path.join(dataset_path, "images", "*.jpg"))
    elif dataset_name == "ecssd":
        filepaths = glob.glob(os.path.join(dataset_path, "images", "*.jpg"))
    elif dataset_name == "mit_300":
        filepaths = glob.glob(os.path.join(
            dataset_path, "BenchmarkIMAGES", "BenchmarkIMAGES", "SM", "*.jpg"))
    else:
        raise ValueError("unknown dataset name '%s'" % dataset_name)

    return filepaths

def get_ground_truth_path(stimulus_path, dataset_path, dataset_name=""):
    if not dataset_name:
        dataset_name = os.path.basename(dataset_path).lower()

    stimulus_filename = os.path.basename(stimulus_path)
    stimulus_name = ".".join(stimulus_filename.split(".")[:-1])

    if dataset_name == "judd":
        map_filename = stimulus_name + "_fixMap.jpg"
        map_path = os.path.join(dataset_path, "maps", map_filename)
    elif dataset_name == "cat2000":
        map_filename = stimulus_filename
        map_path = os.path.join(
            dataset_path, "trainSet", "FIXATIONMAPS", map_filename)
    elif dataset_name == "cssd":
        print("WARNING: cssd has ground-truth masks and not maps.")
        map_filename = stimulus_name + ".png"
        map_path = os.path.join(dataset_path, "groud_truth_mask", map_filename)
    elif dataset_name == "ecssd":
        print("WARNING: cssd has ground-truth masks and not maps.")
        map_filename = stimulus_name + ".png"
        map_path = os.path.join(dataset_path, "groud_truth_mask", map_filename)
    elif dataset_name == "mit_300":
        raise ValueError("mit_300 has no saliency maps or ground truth masks")
    else:
        raise ValueError("unknown dataset name '%s'" % dataset_name)

    return map_path

def std_normalize(data):
    return (data - data.mean())/data.std()

def unit_normalize(data):
    return (data - data.min())/(data.max() - data.min())

def normalize(data, method=None):
    if method == "normal" or method == "std":
        return std_normalize(data)
    elif method == "unit":
        return unit_normalize(data)
    elif method == "none" or method is None:
        return data
    else:
        raise ValueError("unknown normalization method")

def swapax(img):
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

def mirror(img, hor=True):
    if hor:
        mirr = img[:, ::-1]
    else:
        mirr = img[::-1, :]
    return mirr

def affine_tf(img, **kwargs):
    at = tf.AffineTransform(**kwargs)
    return tf.warp(img, at)

def corner(img, frac, mode="tl"):
    h, w = img.shape[:2]
    x_cut = int(w*frac)
    y_cut = int(h*frac)

    if mode == "tl":
        ret = img[:y_cut, :x_cut]
    elif mode == "tr":
        ret = img[:y_cut, -x_cut:]
    elif mode == "bl":
        ret = img[-y_cut:, :x_cut]
    elif mode == "br":
        ret = img[-y_cut:, -x_cut:]
    else:
        raise ValueError("no known mode '%s'" % mode)

    return ret

def show_imgs(imgs, gray=False):
    import pylab

    if gray:
        pylab.gray()

    grid_w = int(np.ceil(np.sqrt(len(imgs))))
    grid_h = int(np.ceil(len(imgs)/grid_w))
    for i, img in enumerate(imgs):
        pylab.subplot(grid_h, grid_w, i+1)
        pylab.axis("off")
        pylab.imshow(img)
    pylab.show()

def augment(img,
    hor_mirr, ver_mirr,
    affine_tfs,
    tl_corner, tr_corner, bl_corner, br_corner):

    augmented = []

    if hor_mirr:
        mirr = mirror(img, hor=True)
        mirr_augm = augment(mirr, False, False,
            affine_tfs, tl_corner, tl_corner, bl_corner, bl_corner)
        augmented.append(mirr)
        augmented.extend(mirr_augm)
    if ver_mirr:
        mirr = mirror(img, hor=False)
        mirr_augm = augment(mirr, False, False,
            affine_tfs, tl_corner, tl_corner, bl_corner, bl_corner)
        augmented.append(mirr)
        augmented.extend(mirr_augm)

    for tf_args in affine_tfs:
        augmented.append(affine_tf(img, **tf_args))

    if tl_corner is not None:
        augmented.append(corner(img, tl_corner, "tl"))
    if tr_corner is not None:
        augmented.append(corner(img, tr_corner, "tr"))
    if bl_corner is not None:
        augmented.append(corner(img, bl_corner, "bl"))
    if br_corner is not None:
        augmented.append(corner(img, br_corner, "br"))

    return augmented

def files_to_mtx():
    x = []
    y = []

    filepaths = get_stimuli_paths(DATASET_PATH)
    for img_fp in filepaths[:2]:
        print("in", img_fp, "...")

        #reading image
        img = io.imread(img_fp, as_grey=False)
        if "rgb" != X_IMG_COLSPACE:
            img = COL_CVT_FUNCS[X_IMG_COLSPACE](img)
            print("\tconverted colspace from rgb to", X_IMG_COLSPACE)
        if img.dtype != np.float64 and X_IMG_TO_FLOAT:
            old_dtype = img.dtype
            img = img_as_float(img)
            print("\tconverted image dtype from {} to {}".format(
                old_dtype, img.dtype))
        x_imgs = [img]

        #reading ground truth
        gt_fp = get_ground_truth_path(img_fp, DATASET_PATH)
        gt = io.imread(gt_fp, as_grey=True)
        if gt.dtype != np.float64 and Y_IMG_TO_FLOAT:
            old_dtype = gt.dtype
            gt = img_as_float(gt)
            print("\tconverted ground truth dtype from {} to {}".format(
                old_dtype, gt.dtype))
        y_imgs = [gt]

        if AUGMENT:
            x_imgs += augment(img, HOR_MIRROR, VER_MIRROR, AFFINE_TRANSFORMS,
                TL_CORNER, TR_CORNER, BL_CORNER, BR_CORNER)
            y_imgs += augment(gt, HOR_MIRROR, VER_MIRROR, AFFINE_TRANSFORMS,
                TL_CORNER, TR_CORNER, BL_CORNER, BR_CORNER)
            print("\taugmented from 1 sample to %d" % len(x_imgs))

        #resizing if necessary
        if img.shape[:2] != SHAPE:
            old_shape = img.shape[:2]
            for i in range(len(x_imgs)):
                x_imgs[i] = tf.resize(x_imgs[i], SHAPE)
            print("\tresized stimulus from {} to {}".format(
                old_shape, x_imgs[-1].shape[:2]))
        if gt.shape[:2] != SHAPE:
            old_shape = gt.shape[:2]
            for i in range(len(x_imgs)):
                y_imgs[i] = tf.resize(y_imgs[i], SHAPE)
            print("\tresized ground truth from {} to {}".format(
                old_shape, y_imgs[-1].shape[:2]))

        if SHOW_IMGS:
            show_imgs([COL_DCVT_FUNCS[X_IMG_COLSPACE](x) for x in x_imgs] + \
                y_imgs)

        if SWAP_CHANNEL_AXIS:
            old_shape = x_imgs[-1].shape
            for i in range(len(x_imgs)):
                x_imgs[i] = swapax(x_imgs[i])
            print("\tswapped x images axis from {} to {}".format(
                old_shape, x_imgs[-1].shape))

        for xi, yi in zip(x_imgs, y_imgs):
            x.append(xi.flatten())
            y.append(yi.flatten())

    #creating matrices
    x_mtx = np.array(x)
    y_mtx = np.array(y)

    print("x_mtx shape: {}, dtype: {}".format(x_mtx.shape, x_mtx.dtype))
    print("y_mtx shape: {}, dtype: {}".format(y_mtx.shape, y_mtx.dtype))
    pixels = SHAPE[0]*SHAPE[1]
    a = x_mtx[0]
    import pylab
    pylab.gray()
    for i in range(3):
        _im = a[i*pixels:(i+1)*pixels]
        im = _im.reshape(SHAPE)
        print("_im shape: {}, im shape: {}".format(_im.shape, im.shape))
        pylab.subplot(1, 3, i+1)
        pylab.axis("off")
        pylab.imshow(im)
    pylab.show()
    exit()
    if do_norm:
        x_mtx = normalize(x_mtx)

    #shuffling data...
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x_mtx = x_mtx[indexes]
    y_mtx = y_mtx[indexes]

    print("x_mtx dtype, min, max, mean, std:",
        x_mtx.dtype, x_mtx.min(), x_mtx.max(), x_mtx.mean(), x_mtx.std())
    print("y_mtx dtype, min, max, mean, std:",
        y_mtx.dtype, y_mtx.min(), y_mtx.max(), y_mtx.mean(), y_mtx.std())
    print("created x, y of shapes", x_mtx.shape, y_mtx.shape)

    return x_mtx, y_mtx

def main():
    print("reading files...")
    x, y = files_to_mtx()

    print("saving...")
    with gzip.open(OUT_DATA_FILEPATH, "wb") as f:
        pickle.dump((x, y), f)
    print("done.")

if __name__ == "__main__":
    main()
