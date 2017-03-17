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
import pickle
import datapreproc
import config.model as model
import config.predict as cfg
try:
    import pylab
    pylab_imported = True
except:
    print("WARNING: failed to import pylab, won't be able to show images")
    pylab_imported = False

def crop_to_shape(img, tgt_shp, mode="tl"):
    """
    Crops img so as to have the same aspect ratio as tgt_shp.
    """
    h1, w1 = img.shape[:2]
    h2, w2 = tgt_shp
    if w1/h1 > w2/h2:
        x_frac, y_frac = (h1*w2)/(h2*w1), 1
    elif w1/h1 < w2/h2:
        x_frac, y_frac = 1, (h2*w1)/(h1*w2)
    return datapreproc.crop(img, x_frac, y_frac, mode)

def load_img(filepath):
    """
    Loads image in RGB format from filepath.
    """
    img = Image.open(filepath).convert("RGB")
    img = np.asarray(img)
    return img

def img_pre_proc(img):
    norm_f = lambda x: datapreproc.normalize(x, method=cfg.norm_method)

    #resizing if needed
    img_shape = img.shape[1:]
    if img_shape != model.Model.INPUT_SHAPE[1:]:
        print("warning: resizing img from {} to {}".format(img_shape,
            model.Model.INPUT_SHAPE))
        if cfg.crop_on_resize:
            print("cropping before resizing")
            img = crop_to_shape(img, model.Model.INPUT_SHAPE[1:])
        img = tf.resize(img, model.Model.INPUT_SHAPE[1:])

    #normalizing image
    x_stats, __ = util.unpkl(cfg.dataset_stats_filepath)
    if cfg.normalize_per_channel:
        channels = []
        for i in range(len(x_stats)):
            minn, maxx, mean, std = x_stats[i]
            channels.append(norm_f(img[:, :, i]))
        img = np.dstack(channels)
    else:
        minn, maxx, mean, std = x_stats[0]
        img = norm_f(img)

    img = datapreproc.swapax(img)

    return img

def load_data_stats(filepath):
    return util.unpickle(filepath)

def predict(img):
    img = img.reshape((1,) + img.shape)

    inp = T.tensor4("inp")
    #neural network model
    net_model = model.Model(inp, load_net_from=cfg.model_filepath)
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
    if pylab_imported:
        print("displaying image...")
        pylab.gray()
        pylab.subplot(1, 2, 1)
        pylab.axis("off")
        pylab.imshow(color.lab2rgb(_img))
        pylab.subplot(1, 2, 2)
        pylab.axis("off")
        pylab.imshow(pred)
        pylab.show()

if __name__ == '__main__':
    main()

