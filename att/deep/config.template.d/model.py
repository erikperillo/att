import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

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

def pyr_down(input_var, shape=(None, None)):
    #g_kernel = (1/16.0)*np.array([
    #    [1.0, 2.0, 1.0],
    #    [2.0, 4.0, 2.0],
    #    [1.0, 2.0, 1.0]], dtype="float32")#.reshape((1, 1, 3, 3))
    #inp = lasagne.layers.InputLayer(shape=(None, 1) + shape,
    #    input_var=input_var)
    out = lasagne.layers.Conv2DLayer(input_var, num_filters=1,
        filter_size=(3, 3), stride=2, W=g_kernel, pad="same")
    return out
    #return lasagne.layers.get_output(out)

def pool_down(incoming):
    return lasagne.layers.MaxPool2DLayer(incoming, pool_size=(2, 2))

def build_inception_module(name, input_layer, filter_sizes):
    #filter_sizes: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}

    net['pool'] = lasagne.layers.MaxPool2DLayer(input_layer,
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
    INPUT_SHAPE = (3, 240, 320)
    OUTPUT_SHAPE = (1, 30, 40)

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
            #deterministic=True)
            deterministic=True)

        #loss train/test symb. functionS
        #self.train_loss = -cc(self.train_pred, self.target_var)
        self.train_loss = mse(self.train_pred, self.target_var)
        #optional regularization term
        #reg = lasagne.regularization.regularize_network_params(
        #    self.net["output"],
        #    lasagne.regularization.l2)
        #self.train_loss += reg*0.00001
        #self.test_loss = -cc(self.test_pred, self.target_var)
        self.test_loss = mse(self.test_pred, self.target_var)

        #updates symb. function for gradient descent
        self.params = lasagne.layers.get_all_params(self.net["output"],
            trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.train_loss, self.params, learning_rate=0.005, momentum=0.9)

        #train function
        self.train_fn = theano.function(
            inputs=[self.input_var, self.target_var],
            #outputs={"-cc": self.train_loss},
            outputs={"mse": self.train_loss},
            updates=self.updates)
        #val function
        self.val_fn = theano.function(
            inputs=[self.input_var, self.target_var],
            outputs={
                #"-cc": self.test_loss,
                "mse": self.test_loss,
                #"mse": mse(self.test_pred, self.target_var)
                "-cc": -cc(self.test_pred, self.target_var)
            })

    def get_net_model(self, input_var=None, inp_shp=None):
        """
        Builds network.
        """
        if inp_shp is None:
            inp_shp = (None,) + Model.INPUT_SHAPE

        net = {}

        #input
        net["input"] = lasagne.layers.InputLayer(shape=inp_shp,
            input_var=input_var)

        #layer 1: 240x320
        net["conv1.1"] = lasagne.layers.Conv2DLayer(net["input"],
            num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        #net["conv1.2"] = lasagne.layers.Conv2DLayer(net["conv1.1"],
        #    num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
        #    nonlinearity=lasagne.nonlinearities.rectify,
        #    pad="same")
        net["pool1"] = lasagne.layers.MaxPool2DLayer(net["conv1.1"],
            pool_size=(2, 2))

        #layer 2: 120x160
        net["conv2.1"] = lasagne.layers.Conv2DLayer(net["pool1"],
            num_filters=64, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv2.2"] = lasagne.layers.Conv2DLayer(net["conv2.1"],
            num_filters=96, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool2"] = lasagne.layers.MaxPool2DLayer(net["conv2.2"],
            pool_size=(2, 2))

        #layer 3: 60x80
        net["conv3.1"] = lasagne.layers.Conv2DLayer(net["pool2"],
            num_filters=128, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv3.2"] = lasagne.layers.Conv2DLayer(net["conv3.1"],
            num_filters=128, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv3.3"] = lasagne.layers.Conv2DLayer(net["conv3.2"],
            num_filters=144, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv3.4"] = lasagne.layers.Conv2DLayer(net["conv3.3"],
            num_filters=144, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool3"] = lasagne.layers.MaxPool2DLayer(net["conv3.4"],
            pool_size=(2, 2))

        #layer 4: 30x40
        net.update(build_inception_module("inception4.1",
            net["pool3"],
            [96, 128, 96, 192, 48, 96]))
        net.update(build_inception_module("inception4.2",
            net["inception4.1/output"],
            [64, 128, 80, 160, 24, 48]))
        net.update(build_inception_module("inception4.3",
            net["inception4.2/output"],
            [64, 128, 80, 160, 24, 48]))
        net.update(build_inception_module("inception4.4",
            net["inception4.3/output"],
            [64, 128, 96, 192, 28, 56]))
        net.update(build_inception_module("inception4.5",
            net["inception4.4/output"],
            [64, 128, 96, 192, 28, 56]))
        net.update(build_inception_module("inception4.6",
            net["inception4.5/output"],
            [64, 128, 112, 224, 32, 64]))
        net.update(build_inception_module("inception4.7",
            net["inception4.6/output"],
            [64, 128, 112, 224, 32, 64]))
        net.update(build_inception_module("inception4.8",
            net["inception4.7/output"],
            [112, 160, 128, 256, 40, 80]))

        #output
        net["output"] = lasagne.layers.Conv2DLayer(net["inception4.8/output"],
            num_filters=1, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.identity)
        print("# params:", lasagne.layers.count_params(net["output"]))

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
