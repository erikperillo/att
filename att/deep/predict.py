#!/usr/bin/env python3

"""
This script takes as input an image (or directory with images) and
performs prediction using model and data configured in config.
"""

import sys
import os
import time
import numpy as np
import glob
import theano
import theano.tensor as T
from PIL import Image
from skimage import color, transform as tf
try:
    import pylab
    pylab_imported = True
except:
    print("WARNING: failed to import pylab, won't be able to show images")
    pylab_imported = False

import util
import datapreproc
import config.model as model
import config.predict as cfg

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

def _load_img(filepath):
    """
    Loads image in RGB format from filepath.
    """
    img = Image.open(filepath).convert("RGB")
    img = np.asarray(img)
    return img

def load_img(filepath):
    """
    Loads image in RGB from filepath and resizes if needed.
    """
    #getting image
    img = _load_img(filepath)

    #resizing if needed
    img_shape = img.shape[1:]
    if img_shape != model.Model.INPUT_SHAPE[1:]:
        print("\twarning: resizing img from {} to {}".format(img_shape,
            model.Model.INPUT_SHAPE))
        if cfg.crop_on_resize:
            print("\tcropping before resizing")
            img = crop_to_shape(img, model.Model.INPUT_SHAPE[1:])
        img = tf.resize(img, model.Model.INPUT_SHAPE[1:])

    return img

def predict(img, pred_f):
    """
    Takes pre-processed image as input and returns prediction.
    """
    img = img.reshape((1,) + img.shape)

    #getting prediction
    start_time = time.time()
    pred = pred_f(img)
    pred_time = time.time() - start_time

    #reshaping
    pred = pred.reshape(model.Model.OUTPUT_SHAPE[1:])
    #unit normalization
    pred = (pred - pred.min())/(pred.max() - pred.min())
    #resizing
    pred = tf.resize(pred, model.Model.INPUT_SHAPE[1:])

    return pred, pred_time

def img_pre_proc(img):
    """
    Takes an RGB image as input and pre-processes it so as to be a valid
    input to the model.
    """
    norm_f = lambda x: datapreproc.normalize(x, method=cfg.norm_method)

    #converting to proper colorspace
    img = datapreproc.col_cvt_funcs[cfg.img_colspace](img)

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

    if cfg.swap_img_channel_axis:
        img = datapreproc.swapax(img)

    return img

def img_filepath_to_fixmap_filepath(filepath):
    if "." in filepath:
        path = ".".join(filepath.split(".")[:-1])
    else:
        path = filepath
    path = os.path.basename(path + "_orig_and_fixmap.png")
    return os.path.join(os.getcwd(), path)

def main():
    if len(sys.argv) < 2:
        print("usage: predict <img_or_dir_path>")
        exit()

    if os.path.isdir(sys.argv[1]):
        filepaths = glob.glob(os.path.join(sys.argv[1], "*"))
    else:
        filepaths = [sys.argv[1]]

    #input
    inp = T.tensor4("inp")
    #neural network model
    net_model = model.Model(inp, load_net_from=cfg.model_filepath)
    #making prediction function
    #prediction function
    pred_f = theano.function([inp], net_model.test_pred)

    #iterating over images doing predictions
    for fp in filepaths:
        print("in image {}".format(fp))

        try:
            print("\tloading image...")
            img = load_img(fp)
            pre_proc_img = img_pre_proc(img)

            print("\tpredicting...", end=" ")
            pred, pred_time = predict(pre_proc_img, pred_f)
            print("done. took %f seconds" % pred_time)
        except:
            print("\tWARNING: could not load file")
            continue

        #displaying results if required
        if pylab_imported and cfg.show_images:
            print("\tdisplaying image...")
            pylab.subplot(1, 2, 1)
            pylab.axis("off")
            pylab.imshow(img)
            pylab.subplot(1, 2, 2)
            pylab.gray()
            pylab.axis("off")
            pylab.imshow(color.gray2rgb(pred))
            #saving if required
            if cfg.save_preds:
                fig = pylab.gcf()
                to_save = img_filepath_to_fixmap_filepath(fp)
                pylab.savefig(to_save)
                print("\tsaved to '%s'" % to_save)
            pylab.show()

if __name__ == '__main__':
    main()

