#!/usr/bin/env python3

import sys
import os
import time

import util

import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from skimage import color, transform as tf
import lasagne
import gzip
import pickle

MODEL_DIR_FILEPATH = "./data/trained_model_1"
MODEL_FILEPATH = os.path.join(MODEL_DIR_FILEPATH, "model.npz")
DATA_STATS_FILEPATH = "/home/erik/proj/att/att/deep/data/"\
    "judd_cat2000_dataset/data_stats.gz"
CROP_ON_RESIZE = True

if not MODEL_DIR_FILEPATH:
    import model
else:
    sys.path.append(MODEL_DIR_FILEPATH)
    import genmodel as model

def swapax(img):
    """from shape (3, h, w) to (w, h, 3)"""
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

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
    if img_shape != model.Model.INPUT_SHAPE[1:]:
        print("warning: resizing img from {} to {}".format(img_shape,
            model.Model.INPUT_SHAPE))
        if CROP_ON_RESIZE:
            print("cropping before resizing")
            img = crop_to_shape(img, model.Model.INPUT_SHAPE[1:])
        img = tf.resize(img, model.Model.INPUT_SHAPE[1:])

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
    return util.unpickle(filepath)

def predict(img):
    img = img.reshape((1,) + img.shape)

    inp = T.tensor4("inp")
    #neural network model
    net_model = model.Model(inp, load_net_from=MODEL_FILEPATH)
    #prediction function
    pred_f = theano.function([inp], net_model.test_pred)

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

    pred = pred.reshape(model.Model.OUTPUT_SHAPE[1:])
    pred = (pred - pred.min())/(pred.max() - pred.min())
    print(pred.shape, pred.min(), pred.max(), pred.mean(), pred.std())
    pred = color.gray2rgb(pred)
    print(pred.shape, model.Model.INPUT_SHAPE)
    pred = tf.resize(pred, model.Model.INPUT_SHAPE[1:])

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
