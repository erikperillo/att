#!/usr/bin/env python3

"""
This script takes images from a certain dataset and performs
data augmentation, preprocessing and formatting in order to make data
ready to be used as input by a certain estimator.

It produces (and pickles) a tuple (X, Y), where X and Y are 2D matrices.
"""

import os
import glob
import numpy as np
import random
from skimage import transform as tf, io, color, img_as_float
import gzip
import shutil
try:
    import pylab
    pylab_imported = True
except:
    print("WARNING: failed to import pylab, won't be able to show images")
    pylab_imported = False

import util
import config.datapreproc as cfg

#conversions from rgb to...
col_cvt_funcs = {
    "lab": color.rgb2lab,
    "hsv": color.rgb2hsv,
    "luv": color.rgb2luv
}
#converting back from ... to rgb
col_dcvt_funcs = {
    "lab": color.lab2rgb,
    "hsv": color.hsv2rgb,
    "luv": color.luv2rgb,
    "rgb": lambda x: x
}

def get_stimuli_paths(dataset_path, dataset_name="", shuffle=True):
    """
    Gets list of stimuli paths given a dataset path.
    Assumes a certain directory structure given the dataset.
    """
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
        filepaths = glob.glob(os.path.join(dataset_path, "stimuli", "*"))
        #raise ValueError("unknown dataset name '%s'" % dataset_name)

    if shuffle:
        random.shuffle(filepaths)
    return filepaths

def get_ground_truth_path(stimulus_path, dataset_path, dataset_name=""):
    """
    Gets ground truth (saliency mask/map) from the filepath of the respective
    stimulus.
    """
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
        match = os.path.join(dataset_path, "ground_truth", stimulus_name + "*")
        map_path = glob.glob(match)[0]
        #raise ValueError("unknown dataset name '%s'" % dataset_name)

    return map_path

def std_normalize(data):
    """
    Mean-std normalization.
    """
    return (data - data.mean())/data.std()

def unit_normalize(data):
    """
    Unit normalization.
    """
    return (data - data.min())/(data.max() - data.min())

def normalize(data, method=None):
    """
    Wrapper for normalization methods.
    """
    if method == "normal" or method == "std":
        return std_normalize(data)
    elif method == "unit":
        return unit_normalize(data)
    elif method == "none" or method is None:
        return data
    else:
        raise ValueError("unknown normalization method")

def swapax(img):
    """
    Makes image's shape from (rows, cols, depth) to (depth, rows, cols)
    """
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

def mirror(img, hor=True):
    """
    Reflects image around axis (hor=True <-> y).
    """
    if hor:
        mirr = img[:, ::-1]
    else:
        mirr = img[::-1, :]
    return mirr

def affine_tf(img, **kwargs):
    """
    Affine transformation on image.
    """
    at = tf.AffineTransform(**kwargs)
    return tf.warp(img, at)

def crop(img, x_frac, y_frac, mode="tl"):
    """
    Crops (x_frac, y_frac) of image's corner (selected by 'mode').
    """
    h, w = img.shape[:2]
    x_cut = int(w*x_frac)
    y_cut = int(h*y_frac)

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

def corner(img, frac, mode="tl"):
    """
    Wrapper for crop.
    """
    return crop(img, frac, frac, mode)

def show_imgs(imgs, gray=False, grid_w=None):
    """
    Displays list of images imgs.
    """
    if gray:
        pylab.gray()

    if grid_w is None:
        grid_w = int(np.ceil(np.sqrt(len(imgs))))
    grid_h = int(np.ceil(len(imgs)/grid_w))
    for i, img in enumerate(imgs):
        pylab.subplot(grid_h, grid_w, i+1)
        pylab.axis("off")
        pylab.imshow(img)
    pylab.show()

def augment(img, hor_mirr, ver_mirr, affine_tfs,
    tl_corner, tr_corner, bl_corner, br_corner):
    """
    Data augmentation for images.
    Applies a variety of techniques to make 1 image into N images.
    """

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
    """
    This routine assumes 3-channel stimuli loaded in RGB in shape (H, W, 3)
        and 1-channel maps loaded in grayscale in shape (H, W).
    """
    x = []
    y = []

    paths = get_stimuli_paths(cfg.dataset_path, cfg.dataset_name)
    for k, img_fp in enumerate(paths[slice(0, cfg.max_samples)]):
        print("in", img_fp, "...")

        #reading image
        img = io.imread(img_fp, as_grey=False)
        if len(img.shape) != 3:
            print("\tconverting grayscale image to rgb")
            img = color.gray2rgb(img)
        #converting colorspace if required
        if "rgb" != cfg.x_img_colspace:
            img = col_cvt_funcs[cfg.x_img_colspace](img)
            print("\tconverted colspace from rgb to", cfg.x_img_colspace)
        #converting datatype if required
        if img.dtype != np.float64 and cfg.x_img_to_float:
            old_dtype = img.dtype
            img = img_as_float(img)
            print("\tconverted image dtype from {} to {}".format(
                old_dtype, img.dtype))
        #list of all x images
        x_imgs = [img]

        #reading respective ground truth
        gt_fp = get_ground_truth_path(img_fp, cfg.dataset_path,
            cfg.dataset_name)
        print("\tground-truth filepath:", gt_fp)
        gt = io.imread(gt_fp, as_grey=True)
        #converting datatype if required
        if gt.dtype != np.float64 and cfg.y_img_to_float:
            old_dtype = gt.dtype
            gt = img_as_float(gt)
            print("\tconverted ground truth dtype from {} to {}".format(
                old_dtype, gt.dtype))
        #list of all ground truths
        y_imgs = [gt]

        #performing data augmentation
        if cfg.augment:
            x_imgs += augment(img, cfg.hor_mirror, cfg.ver_mirror,
                cfg.affine_transforms, cfg.tl_corner, cfg.tr_corner,
                cfg.bl_corner, cfg.br_corner)
            y_imgs += augment(gt, cfg.hor_mirror, cfg.ver_mirror,
                cfg.affine_transforms, cfg.tl_corner, cfg.tr_corner,
                cfg.bl_corner, cfg.br_corner)
            print("\taugmented from 1 sample to %d" % len(x_imgs))

        #resizing if necessary
        for imgs, shp, name in [[x_imgs, cfg.x_shape, "x"],
            [y_imgs, cfg.y_shape, "y"]]:

            if imgs[0].shape[:2] != shp:
                old_shape = imgs[0].shape[:2]
                if cfg.crop_on_resize:
                    crop_mode = "tl" if k%4 == 0 else ("tr" if k%4 == 1 else\
                        ("bl" if k%4 == 2 else "br"))
                    h1, w1 = imgs[0].shape[:2]
                    h2, w2 = shp
                    if w1/h1 > w2/h2:
                        x_frac, y_frac = (h1*w2)/(h2*w1), 1
                    elif w1/h1 < w2/h2:
                        x_frac, y_frac = 1, (h2*w1)/(h1*w2)
                    for i in range(len(imgs)):
                        imgs[i] = crop(imgs[i], x_frac, y_frac, crop_mode)
                    print(("\tcropped {}: x_frac: {:5f}, y_frac: {}, mode: {} "
                        "from {} to {}").format(name, x_frac, y_frac, crop_mode,
                            old_shape, imgs[-1].shape[:2]))
                old_shape = imgs[0].shape[:2]
                for i in range(len(imgs)):
                    imgs[i] = tf.resize(imgs[i], shp)
                print("\tresized {} from {} to {}".format(
                    name, old_shape, imgs[-1].shape[:2]))

        #displaying images and maps if required
        if pylab_imported and cfg.show_images:
            show_imgs([col_dcvt_funcs[cfg.x_img_colspace](x) for x in x_imgs] +\
                y_imgs)
        #displaying separate channels and maps if required
        if pylab_imported and cfg.show_channels:
            channels = [[x[:, :, i] for i in range(3)] + [y] for x, y in\
                zip(x_imgs, y_imgs)]
            channels = [item for sublist in channels for item in sublist]
            show_imgs(channels, gray=True, grid_w=4)

        #swapping channel axis
        if cfg.swap_channel_axis:
            old_shape = x_imgs[-1].shape
            for i in range(len(x_imgs)):
                x_imgs[i] = swapax(x_imgs[i])
            print("\tswapped x images axis from {} to {}".format(
                old_shape, x_imgs[-1].shape))

        #stacking stimuli/maps to matrices
        for xi, yi in zip(x_imgs, y_imgs):
            x.append(xi.flatten())
            y.append(yi.flatten())

    #creating numpy matrices
    x_mtx = np.array(x)
    y_mtx = np.array(y)

    print("x_mtx shape: {}, dtype: {}".format(x_mtx.shape, x_mtx.dtype))
    print("y_mtx shape: {}, dtype: {}".format(y_mtx.shape, y_mtx.dtype))

    #total number of pixels in each channel
    ch_len = cfg.x_shape[0]*cfg.x_shape[1]
    n_channels = len(x_mtx[0])//ch_len

    x_stats = []
    y_stats = []
    #x normalization
    if cfg.x_normalization is not None:
        if cfg.x_normalize_per_channel:
            print("normalizing x_mtx per channel")
            for i in range(n_channels):
                rng = slice(i*ch_len, (i+1)*ch_len)
                ch = x_mtx[:, rng]
                x_stats.append((ch.min(), ch.max(), ch.mean(), ch.std()))
                print("x_mtx channel", i, "min, max, mean, std:",
                    ch.min(), ch.max(), ch.mean(), ch.std(), ", normalizing...")
                x_mtx[:, rng] = normalize(x_mtx[:, rng], cfg.x_normalization)
        else:
            x_stats.append(
                (x_mtx.min(), x_mtx.max(), x_mtx.mean(), x_mtx.std()))
            print("x_mtx min, max, mean, std:", x_mtx.min(), x_mtx.max(),
                x_mtx.mean(), x_mtx.std(), ", normalizing...")
            x_mtx = normalize(x_mtx, cfg.x_normalization)
    #y normalization
    if cfg.y_normalization is not None:
        y_stats.append(
            (y_mtx.min(), y_mtx.max(), y_mtx.mean(), y_mtx.std()))
        print("y_mtx min, max, mean, std:", y_mtx.min(), y_mtx.max(),
            y_mtx.mean(), y_mtx.std(), ", normalizing...")
        y_mtx = normalize(y_mtx, cfg.y_normalization)

    #shuffling data
    print("shuffling data...")
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x_mtx = x_mtx[indexes]
    y_mtx = y_mtx[indexes]

    if x_mtx.dtype != cfg.x_dtype:
        print("setting x_mtx dtype from {} to {}".format(x_mtx.dtype,
            cfg.x_dtype))
        x_mtx = np.array(x_mtx, dtype=cfg.x_dtype)
    if y_mtx.dtype != cfg.y_dtype:
        print("setting y_mtx dtype from {} to {}".format(y_mtx.dtype,
            cfg.y_dtype))
        y_mtx = np.array(y_mtx, dtype=cfg.y_dtype)

    return x_mtx, y_mtx, x_stats, y_stats

def save_to_output_dir(x, y, x_stats, y_stats, base_dir=".", pattern="dataset"):
    """
    Saves matrices (and stats) obtained from files_to_mtx along with other info
    in a newly created dir created in base_dir.
    """
    #creating dir
    out_dir = util.uniq_filepath(base_dir, pattern)
    os.makedirs(out_dir)
    #saving data
    util.pkl((x, y), os.path.join(out_dir, "data.gz"))
    #saving data stats
    util.pkl((x_stats, y_stats), os.path.join(out_dir, "data_stats.gz"))
    #info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("dataset name:", cfg.dataset_name, file=f)
        print("x shape:", x.shape, file=f)
        print("y shape:", y.shape, file=f)
        print("git commit hash:", util.git_hash(), file=f)
    #copying configuration file
    shutil.copy(cfg.__file__, os.path.join(out_dir, "genconfig.py"))

    return out_dir

def main():
    random.seed(cfg.rand_seed)

    print("reading files...")
    x, y, x_stats, y_stats = files_to_mtx()

    if cfg.output_dir_basedir is not None:
        out_dir = save_to_output_dir(x, y, x_stats, y_stats,
            cfg.output_dir_basedir, cfg.dataset_name + "_dataset")
        print("saved dataset, info, stats to '%s'" % out_dir)
    else:
        print("saving data to '%s'..." % cfg.out_data_filepath)
        util.pkl((x, y), cfg.out_data_filepath)
        print("saving stats to '%s'..." % cfg.out_data_stats_filepath)
        util.pkl((x_stats, y_stats), cfg.out_data_stats_filepath)

    print("done.")

if __name__ == "__main__":
    main()
