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
import shutil
from math import ceil
from time import sleep
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

#counter for images processed
counter = 0
id_counter = 0

def dataset_name_from_dataset_path(dataset_path):
    return os.path.basename(dataset_path).lower()

def get_stimuli_paths(dataset_path, dataset_name="", shuffle=True):
    """
    Gets list of stimuli paths given a dataset path.
    Assumes a certain directory structure given the dataset.
    """
    if not dataset_name:
        dataset_name = dataset_name_from_dataset_path(dataset_path)

    if dataset_name == "judd":
        filepaths = glob.glob(os.path.join(dataset_path, "stimuli", "*.jpeg"))
    elif dataset_name == "cat2000":
        filepaths = glob.glob(os.path.join(
            dataset_path, "trainSet", "Stimuli", "*.jpg"))
    elif dataset_name == "salicon":
        fps = glob.glob(os.path.join(dataset_path, "images", "*train*.jpg"))
        fps.extend(glob.glob(os.path.join(dataset_path, "images", "*val*.jpg")))
        filepaths = fps
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
        dataset_name = dataset_name_from_dataset_path(dataset_path)

    stimulus_filename = os.path.basename(stimulus_path)
    stimulus_name = ".".join(stimulus_filename.split(".")[:-1])

    if dataset_name == "judd":
        map_filename = stimulus_name + "_fixMap.jpg"
        map_path = os.path.join(dataset_path, "maps", map_filename)
    elif dataset_name == "cat2000":
        map_filename = stimulus_filename
        map_path = os.path.join(
            dataset_path, "trainSet", "FIXATIONMAPS", map_filename)
    elif dataset_name == "salicon":
        map_filename = stimulus_filename
        map_path = os.path.join(dataset_path, "maps", map_filename)
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
        match = os.path.join(dataset_path, "fixmaps", stimulus_name + "*")
        map_path = glob.glob(match)[0]
        #raise ValueError("unknown dataset name '%s'" % dataset_name)

    return map_path

def std_normalize(data):
    """
    Mean-std normalization.
    """
    if data.std() == 0:
        raise Exception("ZERO STD")

    return (data - data.mean())/data.std()

def unit_normalize(data):
    """
    Unit normalization.
    """
    return (data - data.min())/(data.max() - data.min())

def max_normalize(data):
    """
    Max normalization.
    """
    return data/data.max()

def normalize(data, method=None):
    """
    Wrapper for normalization methods.
    """
    if method == "normal" or method == "std":
        return std_normalize(data)
    elif method == "unit":
        return unit_normalize(data)
    elif method == "max":
        return max_normalize(data)
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

def files_to_mtx(stimuli_paths):
    """
    This routine assumes 3-channel stimuli loaded in RGB in shape (H, W, 3)
        and 1-channel maps loaded in grayscale in shape (H, W).
    """
    x = []
    y = []
    fps = []
    global counter

    for k, img_fp in enumerate(stimuli_paths):
        print("[counter = {}] in {}...".format(counter, img_fp))
        counter += 1

        #reading image (x)
        img = io.imread(img_fp, as_grey=False)
        if len(img.shape) != 3:
            print("\tconverting grayscale image to rgb")
            img = color.gray2rgb(img)

        #converting x colorspace if required
        if "rgb" != cfg.x_img_colspace:
            img = col_cvt_funcs[cfg.x_img_colspace](img)
            print("\tconverted colspace from rgb to", cfg.x_img_colspace)

        #converting x datatype if required
        if img.dtype != np.float64 and cfg.x_img_to_float:
            old_dtype = img.dtype
            img = img_as_float(img)
            print("\tconverted image dtype from {} to {}".format(
                old_dtype, img.dtype))

        #list of all x images
        x_imgs = [img]

        #reading respective ground truth (y)
        gt_fp = get_ground_truth_path(img_fp, cfg.dataset_path,
            cfg.dataset_name)
        print("\tground-truth filepath:", gt_fp)
        gt = io.imread(gt_fp, as_grey=True)

        #converting y datatype if required
        if gt.dtype != np.float64 and cfg.y_img_to_float:
            old_dtype = gt.dtype
            gt = img_as_float(gt)
            print("\tconverted ground truth dtype from {} to {}".format(
                old_dtype, gt.dtype))

        #list of all ground truths
        y_imgs = [gt]

        #cropping if necessary
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
                    else:
                        x_frac, y_frac = 1, 1
                    for i in range(len(imgs)):
                        imgs[i] = crop(imgs[i], x_frac, y_frac, crop_mode)
                    print(("\tcropped {}: x_frac: {:5f}, y_frac: {}, mode: {} "
                        "from {} to {}").format(name, x_frac, y_frac, crop_mode,
                            old_shape, imgs[-1].shape[:2]))

        #performing data augmentation
        if cfg.augment:
            x_imgs += augment(x_imgs[0], cfg.hor_mirror, cfg.ver_mirror,
                cfg.affine_transforms, cfg.tl_corner, cfg.tr_corner,
                cfg.bl_corner, cfg.br_corner)
            y_imgs += augment(y_imgs[0], cfg.hor_mirror, cfg.ver_mirror,
                cfg.affine_transforms, cfg.tl_corner, cfg.tr_corner,
                cfg.bl_corner, cfg.br_corner)
            print("\taugmented from 1 sample to %d" % len(x_imgs))

        #resizing if necessary
        resized = {}
        for imgs, shp, name in [[x_imgs, cfg.x_shape, "x"],
            [y_imgs, cfg.y_shape, "y"]]:
            for i, __ in enumerate(imgs):
                if imgs[i].shape[:2] != shp:
                    old_shape = imgs[i].shape[:2]
                    imgs[i] = tf.resize(imgs[i], shp, mode="constant")
                    key = "{} -> {}".format(old_shape, imgs[i].shape[:2])
                    val = "{}_img[{}]".format(name, i)
                    if not key in resized:
                        resized[key] = []
                    resized[key].append(val)
        for k, v in resized.items():
            print("\t" + ", ".join(v), "resized:", k)

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
            x.append(xi.flatten().astype(cfg.x_dtype))
            y.append(yi.flatten().astype(cfg.x_dtype))
            #appending image/ground truth path to list
            fps.append((img_fp, gt_fp))

    #creating numpy matrices
    x_mtx = np.array(x, dtype=cfg.x_dtype)
    y_mtx = np.array(y, dtype=cfg.y_dtype)

    #shuffling data
    print("shuffling data...")
    indexes = np.arange(len(x))
    np.random.shuffle(indexes)
    x_mtx = x_mtx[indexes]
    y_mtx = y_mtx[indexes]
    fps = [fps[i] for i in indexes]

    print("x_mtx shape: {}, dtype: {}".format(x_mtx.shape, x_mtx.dtype))
    print("y_mtx shape: {}, dtype: {}".format(y_mtx.shape, y_mtx.dtype))

    #total number of pixels in each channel
    ch_len = cfg.x_shape[0]*cfg.x_shape[1]
    n_channels = len(x_mtx[0])//ch_len

    x_stats = []
    y_stats = []

    #x normalization
    if cfg.x_normalization is not None:
        for c in range(n_channels):
            print("\ton channel %d" % c)
            rng = slice(c*ch_len, (c+1)*ch_len)
            if cfg.x_normalize_per_image:
                print("\tnormalizing each x imagel...")
                for i in range(x_mtx.shape[0]):
                    print("\r\t on image %d    " % i, flush=True, end="")
                    x_mtx[i, rng] = normalize(x_mtx[i, rng],
                        cfg.x_normalization)
                print("\r                              ")
            else:
                print("\tnormalizing whole dataset")
                ch = x_mtx[:, rng]
                x_stats.append((ch.min(), ch.max(), ch.mean(), ch.std()))
                print("\tx_mtx channel", i, "min, max, mean, std:",
                    ch.min(), ch.max(), ch.mean(), ch.std())

    #y normalization
    if cfg.y_normalization is not None:
        if cfg.y_normalize_per_image:
            print("\tnormalizing y per image...")
            for i in range(y_mtx.shape[0]):
                print("\r\t on image %d    " % i, flush=True, end="")
                y_mtx[i, :] = normalize(y_mtx[i, :],
                    cfg.y_normalization)
            print("\r                                        ")
        else:
            y_stats.append(
                (y_mtx.min(), y_mtx.max(), y_mtx.mean(), y_mtx.std()))
            print("\ty_mtx min, max, mean, std:",
                y_mtx.min(), y_mtx.max(), y_mtx.mean(), y_mtx.std())

    return x_mtx, y_mtx, x_stats, y_stats, fps

def mk_output_dir(base_dir=".", pattern="dataset"):
    """
    Creates unique dir in base dir.
    """
    #creating dir
    out_dir = util.uniq_filepath(base_dir, pattern)
    os.makedirs(out_dir)

    #info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("dataset name:", cfg.dataset_name, file=f)
        print("git commit hash:", util.git_hash(), file=f)

    #copying configuration file
    shutil.copy(cfg.__file__, os.path.join(out_dir, "genconfig.py"))

    return out_dir

def save_to_output_dir(out_dir, x, y, x_stats, y_stats, fps, pattern="data"):
    """
    Saves matrices (and stats) obtained from files_to_mtx along with other info
    in out_dir.
    """
    global id_counter

    #saving data
    data_fp = os.path.join(out_dir, pattern + ".gz")
    util.pkl((x, y), data_fp)

    #saving data stats
    data_stats_fp = os.path.join(out_dir, pattern + "_stats.pkl")
    util.pkl((x_stats, y_stats), data_stats_fp)

    #saving filepaths
    filepaths_fp = os.path.join(out_dir, pattern + "_filepaths.csv")
    with open(filepaths_fp, "w") as f:
        print("id,stimulus,ground_truth", file=f)
        for xfp, yfp in fps:
            print("{},{},{}".format(id_counter, xfp, yfp), file=f)
            id_counter += 1

    return out_dir

def batch_stats_to_global_stats(x_stats, y_stats):
    """
    assumes x_stats in format:
    [(batch_1_weight, [(ch_1_min, ch_1_max, ch_1_mean, ch_1_std), ..., ]), ...]
    and y_stats in format:
    [(batch_1_weight, [(min, max, mean, std)]), ...]
    """
    #relative frequency of each batch
    weights = [st[0] for st in x_stats]
    #print("begin:", "X:", x_stats, "\nY:", y_stats)

    #getting y stats
    if cfg.y_normalization is not None and not cfg.y_normalize_per_image:
        _y_stats = [s[1][0] for s in y_stats]
        y_minn = min(s[0] for s in _y_stats)
        y_maxx = max(s[1] for s in _y_stats)
        y_mean = sum(w*s[2] for w, s in zip(weights, _y_stats))
        y_std = np.sqrt(sum(w*s[3]**2 for w, s in zip(weights, _y_stats)))
        y_stats = (y_minn, y_maxx, y_mean, y_std)

    #getting x stats channel-wise
    if cfg.x_normalization is not None and not cfg.x_normalize_per_image:
        n_x_channels = len(x_stats[0][1])
        x_ch_stats = []
        for i in range(n_x_channels):
            ch_stats = [s[1][i] for s in x_stats]
            minn = min(s[0] for s in ch_stats)
            maxx = max(s[1] for s in ch_stats)
            mean = sum(w*s[2] for w, s in zip(weights, ch_stats))
            std = np.sqrt(sum(w*s[3]**2 for w, s in zip(weights, ch_stats)))
            x_ch_stats.append((minn, maxx, mean, std))

    return x_ch_stats, y_stats

def _normalize_after_saving(x, y, x_stats, y_stats):
    #x_stats in format:
    #[(ch_1_min, ch_1_max, ch_1_mean, ch_1_std), ...]
    #y_stats in format:
    #(min, max, mean, std)
    if cfg.x_normalization is not None and not cfg.x_normalize_per_image:
        print("\tnormalizing x")
        ch_len = cfg.x_shape[0]*cfg.x_shape[1]
        for c in range(len(x_stats)):
            sl = slice(c*ch_len, (c+1)*ch_len)
            if cfg.x_normalization == "std":
                x[sl] = (x[sl]-x_stats[c][2])/x_stats[c][3]
            elif cfg.x_normalization == "unit":
                x[sl] = (x[sl]-x_stats[c][0])/(x_stats[c][1]-x_stats[c][0])
            elif cfg.x_normalization == "max":
                x[sl] = x[sl]/x_stats[c][1]
            else:
                raise ValueError("invalid x normalization method")

    if cfg.y_normalization is not None and not cfg.y_normalize_per_image:
        print("\tnormalizing y")
        if cfg.y_normalization == "std":
            y = (y - y_stats[2])/y_stats[3]
        elif cfg.y_normalization == "unit":
            y = (y - y_stats[0])/(y_stats[1] - y_stats[0])
        elif cfg.y_normalization == "max":
            y = y/y_stats[1]
        else:
            raise ValueError("invalid y normalization method")

    return x, y

def normalize_after_saving(out_dir, x_stats, y_stats, pattern="data_part_"):
    #making global stats and saving
    x_stats, y_stats = batch_stats_to_global_stats(x_stats, y_stats)
    util.pkl((x_stats, y_stats), os.path.join(out_dir, "data_stats.pkl"))

    for fn in glob.glob(os.path.join(out_dir, pattern + "*")):
        if "stats" in fn or "filepaths" in fn:
            continue
        print("normalizing data in '%s'..." % fn)
        x, y = util.unpkl(fn)
        x, y = _normalize_after_saving(x, y, x_stats, y_stats)
        util.pkl((x, y), fn)

def main():
    random.seed(cfg.rand_seed)

    #creating output dir
    out_dir = mk_output_dir(cfg.output_dir_basedir,
        cfg.dataset_name + "_dataset")
    print("created output dir in '%s'" % out_dir)

    #div is the factor by which each image will be augmented
    div = 1 if not cfg.augment else \
        (1 + cfg.hor_mirror + cfg.ver_mirror)*(1 +\
        (cfg.tl_corner is not None) + (cfg.tr_corner is not None) +\
        (cfg.bl_corner is not None) + (cfg.br_corner is not None) +\
        len(cfg.affine_transforms))
    print("will augment sources by a factor of", div)

    #paths of stimuli images
    stimuli_paths = get_stimuli_paths(cfg.dataset_path, cfg.dataset_name)
    if cfg.max_samples is not None:
        n_samples = cfg.max_samples
    else:
        n_samples = len(stimuli_paths)*div
    n_sources = n_samples//div
    print("n_samples: {}, n_sources: {}".format(n_samples, n_sources))
    stimuli_paths = stimuli_paths[:n_sources]

    #calculating size of sources batches
    if cfg.data_save_batch_size is not None:
        batch_size = cfg.data_save_batch_size//div
        n_batches = ceil(n_sources/batch_size)
    else:
        batch_size = n_samples
        n_batches = 1
    print("batch_size: {}, n_batches: {}".format(batch_size, n_batches))

    print("starting in ", end="", flush=True)
    for s in "...".join(map(str, range(5, 0, -1))):
        print(s, end="", flush=True)
        sleep(0.2)
    print()

    x_stats = []
    y_stats = []
    #main loop
    for i in range(n_batches):
        print("in batch %d" % (i+1))

        try:
            x, y, _x_stats, _y_stats, fps = files_to_mtx(
                stimuli_paths[:batch_size])
        except Exception as e:
            print("error in batch", i+1)
            raise e
        batch_w = len(stimuli_paths[:batch_size])/n_sources
        x_stats.append((batch_w, _x_stats))
        y_stats.append((batch_w, _y_stats))

        print("saving data part %d..." % (i+1))
        save_to_output_dir(out_dir, x, y, _x_stats, _y_stats, fps,
            "data_part_%d" % (i+1))

        stimuli_paths = stimuli_paths[batch_size:]

    #normalization after saving
    norm_x = cfg.x_normalization is not None and not cfg.x_normalize_per_image
    norm_y = cfg.y_normalization is not None and not cfg.y_normalize_per_image
    if norm_x or norm_y:
        print("normalizing after saving...")
        normalize_after_saving(out_dir, x_stats, y_stats)

    print("saved everything to '%s'" % out_dir)
    print("done.")

if __name__ == "__main__":
    main()
