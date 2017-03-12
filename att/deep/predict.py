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
import gzip
import pickle

MODEL_FILEPATH = "./model.npz"
INPUT_SHAPE = (3, 76, 100)
#these values are for judd benchmark.
x_means_stds = [
    #channel 0
   (45.5519397, 26.7712818),
    #channel 1
   (1.6996536, 10.7172198),
    #channel 2
   (5.4866930, 16.5665818)
]
y_mean_std = (0.0513149, 0.1292755)

def swapax(img):
    """from shape (3, h, w) to (w, h, 3)"""
    return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)

def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    #if img.depth != 3:
    #    raise ValueError("Must pass a RGB image")

    img_shape = img.size[::-1]
    if img_shape != INPUT_SHAPE[1:]:
        print("warning: resizing img from {} to {}".format(img_shape,
            INPUT_SHAPE))
        img = img.resize(INPUT_SHAPE[1::][::-1], Image.ANTIALIAS)

    img = np.asarray(img)
    return img

def img_pre_proc(img):
    r = (img[:, :, 0] - x_means_stds[0][0])/x_means_stds[0][1]
    g = (img[:, :, 1] - x_means_stds[1][0])/x_means_stds[1][1]
    b = (img[:, :, 2] - x_means_stds[2][0])/x_means_stds[2][1]

    img = np.dstack((r, g, b))
    img = swapax(img)
    return img

def load_model(network, filepath):
    with np.load(filepath) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    return network

def build_cnn(input_shape, input_var=None):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)

    #input shape in form n_batches, depth, rows, cols
    output_shape = input_shape[-2]//2, input_shape[-1]//2

    #input
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    # Convolutional layer with 16 kernels of size 3x3. Strided and padded
    # convolutions are supported as well; see the docstring.
    #network = lasagne.layers.Conv2DLayer(
    #        network, num_filters=8, filter_size=(3, 3),
    #        nonlinearity=lasagne.nonlinearities.rectify,
    #        W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=48, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of x units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=int(1.5*output_shape[0]*output_shape[1]),
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=output_shape[0]*output_shape[1],
            nonlinearity=lasagne.nonlinearities.identity)

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

    pred = pred.reshape([x//2 for x in INPUT_SHAPE[1:]])
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
        pylab.imshow(_img)
        pylab.subplot(1, 2, 2)
        pylab.axis("off")
        pylab.imshow(pred)
        pylab.show()
    except Exception:
        print("WARNING: could not display image")

if __name__ == '__main__':
    main()
