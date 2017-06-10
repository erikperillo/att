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

import scipy.ndimage as nd

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
    else:
        x_frac, y_frac = 1, 1
    return datapreproc.crop(img, x_frac, y_frac, mode)

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
    #pred = pred.reshape(model.Model.OUTPUT_SHAPE[1:])
    pred = pred.reshape(pred.shape[2:])
    pred = nd.zoom(pred, (4, 4), order=1)
    print("\tprediction pre-unit norm:",
        "min: {:5f}, max: {:5f}, mean: {:5f}, std: {:5f}".format(
        pred.min(), pred.max(), pred.mean(), pred.std()))
    #unit normalization
    pred = (pred - pred.min())/(pred.max() - pred.min())
    print("\tprediction pos-unit norm:",
        "min: {:5f}, max: {:5f}, mean: {:5f}, std: {:5f}".format(
        pred.min(), pred.max(), pred.mean(), pred.std()))
    #resizing
    if cfg.resize:
        print("resizing")
        pred = tf.resize(pred, model.Model.INPUT_SHAPE[1:], mode="constant")

    return pred, pred_time

def img_pre_proc(img):
    """
    Takes an RGB image as input and pre-processes it so as to be a valid
    input to the model.
    """
    #resizing if needed
    img_shape = img.shape[:-1] if cfg.swap_img_channel_axis else img.shape[1:]
    if img_shape != model.Model.INPUT_SHAPE[1:] and cfg.resize:
        print("\twarning: resizing img from {} to {}".format(img_shape,
            model.Model.INPUT_SHAPE[1:]), end="")
        if cfg.crop_on_resize:
            print(" (cropping before resizing)", end=" ")
            img = crop_to_shape(img, model.Model.INPUT_SHAPE[1:])
        print("resizing")
        img = tf.resize(img, model.Model.INPUT_SHAPE[1:], mode="constant")

    norm_f = lambda x: datapreproc.normalize(x, method=cfg.img_normalization)

    #converting to proper colorspace
    img = datapreproc.col_cvt_funcs[cfg.img_colspace](img)

    #normalizing image
    channels = []
    if not cfg.normalize_per_image:
        raise NotImplementedError("normalization per dataset not implemented")
    else:
        print("\tnormalizing input image per channel with method '{}'".format(
            cfg.img_normalization))
        for i in range(3):
            channels.append(norm_f(img[:, :, i]))
    img = np.dstack(channels)

    #swapping axis if necessary
    if cfg.swap_img_channel_axis:
        img = datapreproc.swapax(img)

    #converting to proper data type
    img = np.array(img, dtype=cfg.img_dtype)

    return img

def img_filepath_to_fixmap_filepath(filepath, ext="png"):
    if "." in filepath:
        path = ".".join(filepath.split(".")[:-1])
    else:
        path = filepath
    path = os.path.basename(path + "." + ext)
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
            img = util.load_image(fp)
            if cfg.max_img_shape is not None and\
                not (img.shape[:2] <= cfg.max_img_shape):
                print("warning: resizing img to {}".format(cfg.max_img_shape))
                h, w = cfg.max_img_shape
                scale = min(h/img.shape[0], w/img.shape[1])
                img = nd.zoom(img, (scale, scale, 1), order=1)
            pre_proc_img = img_pre_proc(img)
            print("\tpredicting...")
            pred, pred_time = predict(pre_proc_img, pred_f)
            print("\tdone predicting. took %f seconds" % pred_time)
        except FileNotFoundError:
            print("\tWARNING: could not load file")
            continue
        except:
            raise

        #displaying results if required
        if pylab_imported and cfg.show_images:
            pylab.subplot(1, 2, 1)
            pylab.axis("off")
            pylab.imshow(img)
            pylab.subplot(1, 2, 2)
            pylab.gray()
            pylab.axis("off")
            pylab.imshow(color.gray2rgb(pred))
            print("\tdisplaying image...")
            pylab.show()

        #saving if required
        if cfg.save_preds:
            #fig = pylab.gcf()
            to_save = img_filepath_to_fixmap_filepath(fp, ext="jpeg")
            print("\tsaving to '%s'" % to_save)
            #pylab.savefig(to_save)
            #print(pred.min(), pred.max(), pred.mean(), pred.std(), pred.shape)
            pred = nd.zoom(pred, (8, 8), order=1)
            util.save_image(255*pred, to_save)
            util.save_image(img, os.path.basename(fp).split(".")[0]+"_g.jpg")

if __name__ == '__main__':
    main()

