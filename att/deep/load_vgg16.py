#!/usr/bin/env python3

import vgg16
import lasagne
from theano import tensor as T
import theano
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pickle

WEIGHTS_FILEPATH = "/home/erik/random/vgg16.pkl"

def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.
    Only really makes sense with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer.W.get_value()
    shape = W.shape
    print("shape=", shape)
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    for k in range(shape[0]):
        print("in k =", k)
        for feature_map in range(shape[1]):
            print("\tin i =", feature_map)
            figs, axes = plt.subplots(nrows, ncols, figsize=figsize,
                squeeze=False)

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')

            for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
                if i >= shape[0]:
                    break
                axes[r, c].imshow(W[i, feature_map], cmap='gray',
                                  interpolation='none')

            plt.show()

def main():
    print("building model...", end=" ", flush=True)
    net = vgg16.build_model()
    print("done.")

    print("loading weights...", end=" ", flush=True)
    with open(WEIGHTS_FILEPATH, "rb") as f:
        params = pickle.load(f, encoding="latin1")
    print("done.")

    print("setting layers...", end=" ", flush=True)
    out_layer = net["prob"]
    #net.initialize_layers()
    lasagne.layers.set_all_param_values(out_layer, params["param values"])
    print("done.")

    layers = lasagne.layers.get_all_layers(out_layer)
    for l in layers:
        if not "Conv2DLayer" in str(type(l)):
            continue
        print("in layer {}".format(type(l)))
        plot_conv_weights(l)

if __name__ == "__main__":
    main()
