import lasagne
import numpy as np
from theano import tensor as T

class Model:
    #in format depth, rows, cols
    INPUT_SHAPE = (3, 192, 256)
    OUTPUT_SHAPE = (1, 12, 16)

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
        self.train_loss = lasagne.objectives.squared_error(self.train_pred,
            self.target_var).mean()
        #optional regularization term
        #reg = lasagne.regularization.regularize_network_params(
        #    self.net["output"],
        #    lasagne.regularization.l2)
        #self.train_loss += reg*0.00001
        self.test_loss = lasagne.objectives.squared_error(self.test_pred,
            self.target_var).mean()

        #updates symb. function for gradient descent
        self.params = lasagne.layers.get_all_params(self.net["output"],
            trainable=True)
        self.updates = lasagne.updates.nesterov_momentum(
            self.train_loss, self.params, learning_rate=0.01, momentum=0.9)

        #mean absolute error
        self.mae = T.mean(abs(self.test_pred - self.target_var))

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
        net["pool1"] = lasagne.layers.MaxPool2DLayer(net["conv1"],
            pool_size=(2, 2))

        #convpool layer
        net["conv2"] = lasagne.layers.Conv2DLayer(net["pool1"],
            num_filters=48, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool2"] = lasagne.layers.MaxPool2DLayer(net["conv2"],
            pool_size=(2, 2))

        #convpool layer
        net["conv3"] = lasagne.layers.Conv2DLayer(net["pool2"],
            num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool3"] = lasagne.layers.MaxPool2DLayer(net["conv3"],
            pool_size=(2, 2))

        #convpool layer
        net["conv4"] = lasagne.layers.Conv2DLayer(net["pool3"],
            num_filters=80, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool4"] = lasagne.layers.MaxPool2DLayer(net["conv4"],
            pool_size=(2, 2))

        #output
        net["output"] = lasagne.layers.Conv2DLayer(net["pool4"],
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
