import lasagne
import numpy as np
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode

def set_layer_as_immutable(layer):
    """Sets layer parameters so as not to be modified in training steps."""
    for k in layer.params.keys():
        layer.params[k] -= {"regularizable", "trainable"}

def std_norm(x):
    """Mean-std normalization."""
    return (x - T.mean(x))/T.std(x)

def cov(a, b):
    """Covariance."""
    return T.mean((a - T.mean(a))*(b - T.mean(b)))

def corr_coef(pred, tgt):
    """Correlation Coefficient."""
    return cov(pred, tgt)/(T.std(pred)*T.std(tgt))

def r_squared(pred, tgt):
    """R-squared."""
    return T.square(coef_corr(pred, tgt))

def sim(pred, tgt):
    """Similarity."""
    return T.sum(T.minimum(pred/pred.sum(), tgt/tgt.sum()))

def mse(pred, tgt):
    """Mean-squared-error."""
    return lasagne.objectives.squared_error(pred, tgt).mean()

def norm_mse(pred, tgt, alpha):
    """Normalized mean-squared-error."""
    return T.square((pred/pred.max() - tgt)/(alpha - tgt)).mean()

def mae(pred, tgt):
    """Mean-absolute-error."""
    return T.mean(abs(pred - tgt))

class Model:
    #in format depth, rows, cols
    INPUT_SHAPE = (3, 192, 256)
    OUTPUT_SHAPE = (1, 48, 64)

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
        self.train_loss = -corr_coef(self.train_pred, self.target_var)
        #self.train_loss = mse(self.train_pred, self.target_var)
        #optional regularization term
        #reg = lasagne.regularization.regularize_network_params(
        #    self.net["output"],
        #    lasagne.regularization.l2)
        #self.train_loss += reg*0.00001
        self.test_loss = -corr_coef(self.test_pred, self.target_var)
        #self.test_loss = mse(self.test_pred, self.target_var)

        #updates symb. function for gradient descent
        self.params = lasagne.layers.get_all_params(self.net["output"],
            trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.train_loss, self.params, learning_rate=0.001, momentum=0.9)

        #mean absolute error
        self.mae = mae(self.test_pred, self.target_var)

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

        #convpool layer
        net["conv1"] = lasagne.layers.Conv2DLayer(net["input"],
            num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv2"] = lasagne.layers.Conv2DLayer(net["conv1"],
            num_filters=40, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool1"] = lasagne.layers.MaxPool2DLayer(net["conv2"],
            pool_size=(2, 2))

        #convpool layer
        net["conv3"] = lasagne.layers.Conv2DLayer(net["pool1"],
            num_filters=48, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv4"] = lasagne.layers.Conv2DLayer(net["conv3"],
            num_filters=56, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool2"] = lasagne.layers.MaxPool2DLayer(net["conv4"],
            pool_size=(2, 2))

        #conv layer
        net["conv5"] = lasagne.layers.Conv2DLayer(net["pool2"],
            num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv6"] = lasagne.layers.Conv2DLayer(net["conv5"],
            num_filters=72, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv7"] = lasagne.layers.Conv2DLayer(net["conv6"],
            num_filters=80, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv8"] = lasagne.layers.Conv2DLayer(net["conv7"],
            num_filters=88, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv9"] = lasagne.layers.Conv2DLayer(net["conv8"],
            num_filters=96, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv10"] = lasagne.layers.Conv2DLayer(net["conv9"],
            num_filters=104, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")

        #output
        net["output"] = lasagne.layers.Conv2DLayer(net["conv10"],
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
