#!/usr/bin/env python3

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from skimage import color, transform as tf
import lasagne
from deep import build_cnn
import gzip
import pickle

#MODEL_FILEPATH = "./model_best_so_far_trainloss07863valloss08151.npz"
MODEL_FILEPATH = "./model.npz"
#DATA_STATS_FILEPATH = "./stats.gz"
DATA_STATS_FILEPATH = "/home/erik/proj/att/att/deep/data/"\
    "judd_cat2000_dataset/data_stats.gz"
INPUT_SHAPE = (3, 80, 120)#(3, 76, 100)
OUTPUT_SHAPE = tuple(int(0.4*x) for x in INPUT_SHAPE[1:])#(38, 50)
CROP_ON_RESIZE = True

def swapax(img):
    """from shape (3, h, w) to (w, h, 3)"""
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

def _open(filepath, *args, **kwargs):
    ext = filepath.split(".")[-1]
    if ext == "gz":
        fn = gzip.open
    else:
        fn = open
    return fn(filepath, *args, **kwargs)

def crop(img, x_frac, y_frac, mode="tl"):
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

def crop_to_shape(img, tgt_shp, mode="tl"):
    h1, w1 = img.shape[:2]
    h2, w2 = tgt_shp
    if w1/h1 > w2/h2:
        x_frac, y_frac = (h1*w2)/(h2*w1), 1
    elif w1/h1 < w2/h2:
        x_frac, y_frac = 1, (h2*w1)/(h1*w2)
    return crop(img, x_frac, y_frac, mode)

def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    img = np.asarray(img)
    #if img.depth != 3:
    #    raise ValueError("Must pass a RGB image")

    img = color.rgb2lab(img)

    img_shape = img.shape[1:]
    if img_shape != INPUT_SHAPE[1:]:
        print("warning: resizing img from {} to {}".format(img_shape,
            INPUT_SHAPE))
        if CROP_ON_RESIZE:
            print("cropping before resizing")
            img = crop_to_shape(img, INPUT_SHAPE[1:])
        img = tf.resize(img, INPUT_SHAPE[1:])

    return img

def img_pre_proc(img):
    try:
        x_stats, __ = load_data_stats(DATA_STATS_FILEPATH)
    except:
        with gzip.open(DATA_STATS_FILEPATH, "rb") as f:
            x_stats = pickle.load(f)
    if len(x_stats) == 3:
        channels = []
        for i in range(len(x_stats)):
            minn, maxx, mean, std = x_stats[i]
            channels.append((img[:, :, i] - mean)/std)
        img = np.dstack(channels)

    img = swapax(img)

    return img

def load_data_stats(filepath):
    with _open(filepath, "rb") as f:
        x_stats, y_stats = pickle.load(f)
    return x_stats, y_stats

def load_model(network, filepath):
    with np.load(filepath) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network

def predict(img):
    inp = T.tensor4("inp")

    # Create neural network model
    _network = build_cnn((None,) + INPUT_SHAPE, inp)
    network = load_model(_network, MODEL_FILEPATH)

    pred = lasagne.layers.get_output(network, inputs=inp, deterministic=True)

    pred_f = theano.function([inp], pred)

    img = img.reshape((1,) + img.shape)

    start_time = time.time()
    sal_map = pred_f(img)
    pred_time = time.time() - start_time

    return sal_map, pred_time

def main():
    if len(sys.argv) < 2:
        print("usage: predict <img_filepath>")
        exit()

    print("loading image...")
    _img = load_img(sys.argv[1])
    img = img_pre_proc(_img)

    print("predicting...")
    pred, pred_time = predict(img)
    print("prediction took %f seconds" % pred_time)

    print("saving to 'pred.pkl'...")
    with open("pred.pkl", "wb") as f:
        pickle.dump(pred, f)

    pred = pred.reshape(OUTPUT_SHAPE)
    pred = (pred - pred.min())/(pred.max() - pred.min())
    print(pred.shape, pred.min(), pred.max(), pred.mean(), pred.std())
    pred = color.gray2rgb(pred)
    pred = tf.resize(pred, INPUT_SHAPE[1:])

    #_img = _img.copy()
    #_img.setflags(write=1)
    #_img[::5, :, :] = 0
    #_img[:, ::5, :] = 0
    try:
        import pylab
        print("displaying image...")
        pylab.gray()
        pylab.subplot(1, 2, 1)
        pylab.axis("off")
        pylab.imshow(color.lab2rgb(_img))
        pylab.subplot(1, 2, 2)
        pylab.axis("off")
        pylab.imshow(pred)
        pylab.show()
    except Exception:
        print("WARNING: could not display image")

if __name__ == '__main__':
    main()
