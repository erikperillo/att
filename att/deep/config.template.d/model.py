import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer

def set_layer_as_immutable(layer):
    """Sets layer parameters so as not to be modified in training steps."""
    for k in layer.params.keys():
        layer.params[k] -= {"regularizable", "trainable"}

def cov(a, b):
    """Covariance."""
    return T.mean((a - T.mean(a))*(b - T.mean(b)))

def cc(pred, tgt):
    """Correlation Coefficient."""
    return cov(pred, tgt)/(T.std(pred)*T.std(tgt))

def mae(pred, tgt):
    """Mean-absolute-error."""
    return T.mean(abs(pred - tgt))

def mse(pred, tgt):
    """Mean-squared-error."""
    return lasagne.objectives.squared_error(pred, tgt).mean()

def build_inception_module(name, input_layer, filter_sizes):
    #filter_sizes: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}

    net['pool'] = PoolLayer(input_layer,
        pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(net['pool'],
        filter_sizes[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer,
        filter_sizes[1], 1, flip_filters=False)

    if filter_sizes[2] > 0:
        net['3x3_reduce'] = ConvLayer(input_layer,
            filter_sizes[2], 1, flip_filters=False)
        net['3x3'] = ConvLayer(net['3x3_reduce'],
            filter_sizes[3], 3, pad=1, flip_filters=False)
    else:
        net['3x3'] = ConvLayer(input_layer,
            filter_sizes[3], 3, pad=1, flip_filters=False)

    if filter_sizes[4] > 0:
        net['5x5_reduce'] = ConvLayer(input_layer,
            filter_sizes[4], 1, flip_filters=False)
        net['5x5'] = ConvLayer(net['5x5_reduce'],
            filter_sizes[5], 5, pad=2, flip_filters=False)
    else:
        net['5x5'] = ConvLayer(input_layer,
            filter_sizes[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}

class Model:
    #in format depth, rows, cols
    INPUT_SHAPE = (3, 480, 640)
    OUTPUT_SHAPE = (1, 60, 80)

    def __init__(self, input_var=None, target_var=None, load_net_from=None):
        self.input_var = T.tensor4('inps') if input_var is None else input_var
        self.target_var = T.matrix('tgts') if target_var is None else target_var

        #the network lasagne model
        self.net = self.get_net_model(input_var)
        if load_net_from is not None:
            self.load_net(load_net_from)

        #prediction train/test symbolic functions
        self.train_pred = lasagne.layers.get_output(self.net["output"],
            deterministic=False)
        self.test_pred = lasagne.layers.get_output(self.net["output"],
            deterministic=True)

        #loss train/test symb. functionS
        self.train_loss = -cc(self.train_pred, self.target_var)
        #optional regularization term
        #reg = lasagne.regularization.regularize_network_params(
        #    self.net["output"],
        #    lasagne.regularization.l2)
        #self.train_loss += reg*0.00001
        self.test_loss = -cc(self.test_pred, self.target_var)

        #updates symb. function for gradient descent
        self.params = lasagne.layers.get_all_params(self.net["output"],
            trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.train_loss, self.params, learning_rate=0.001, momentum=0.9)

        #train function
        self.train_fn = theano.function(
            inputs=[self.input_var, self.target_var],
            outputs={"cc": self.train_loss},
            updates=self.updates)
        #val function
        self.val_fn = theano.function(
            inputs=[self.input_var, self.target_var],
            outputs={
                "cc": self.test_loss,
                "mse": mse(self.test_pred, self.target_var)
            })

    def get_net_model(self, input_var=None, inp_shp=None):
        """
        Builds network.
        """
        if inp_shp is None:
            inp_shp = (None,) + Model.INPUT_SHAPE

        net = {}

        #input
        #net["input"] = lasagne.layers.InputLayer(shape=inp_shp,
        net["input"] = lasagne.layers.InputLayer(shape=(None, 3, None, None),
            input_var=input_var)

        #convpool layer
        net["conv1"] = lasagne.layers.Conv2DLayer(net["input"],
            num_filters=48//2, filter_size=(7, 7), stride=2, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv2"] = lasagne.layers.Conv2DLayer(net["conv1"],
            num_filters=48//2, filter_size=(5, 5), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv3"] = lasagne.layers.Conv2DLayer(net["conv2"],
            num_filters=48//2, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool1"] = lasagne.layers.MaxPool2DLayer(net["conv3"],
            pool_size=(2, 2))

        #inception layer
        net.update(build_inception_module("inception1",
            net["pool1"],
             #pool, 1x1, 3x3_red, 3x3, 5x5_red, 5x5
            [16, 16, 16, 64//2, 32//2, 48//2]))
        net.update(build_inception_module("inception2",
            net["inception1/output"],
            [32//2, 32//2, 32//2, 64//2, 32//2, 48//2]))
        net.update(build_inception_module("inception3",
            net["inception2/output"],
            [32//2, 32//2, 32//2, 64//2, 32//2, 48//2]))
        net["conv4"] = lasagne.layers.Conv2DLayer(net["inception3/output"],
            num_filters=96//2, filter_size=(1, 1), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool2"] = lasagne.layers.MaxPool2DLayer(net["conv4"],
            pool_size=(2, 2))

        #conv layer
        net["conv5"] = lasagne.layers.Conv2DLayer(net["pool2"],
            num_filters=96//2, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv6"] = lasagne.layers.Conv2DLayer(net["conv5"],
            num_filters=112//2, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net.update(build_inception_module("inception4",
            net["conv6"],
            [64//2, 64//2, 32//2, 128//2, 32//2, 96//2]))

        #output
        net["output"] = lasagne.layers.Conv2DLayer(net["inception4/output"],
            num_filters=1, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.identity)

        return net

    def save_net(self, filepath):
        """
        Saves net weights.
        """
        np.savez(filepath, *lasagne.layers.get_all_param_values(
            self.net["output"]))

    def load_net(self, filepath):
        """
        Loads net weights.
        """
        with np.load(filepath) as f:
            param_values = [f["arr_%d" % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.net["output"],
                param_values)
