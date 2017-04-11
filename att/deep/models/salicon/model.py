import lasagne
from lasagne.layers import InputLayer
from lasagne.layers.conv import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Upscale2DLayer as UpscaleLayer
from lasagne.layers import ConcatLayer

import numpy as np
from theano import tensor as T

def set_layer_as_rigid(layer):
    for k in layer.params.keys():
        layer.params[k] -= {"regularizable", "trainable"}

class Model:
    #in format depth, rows, cols
    INPUT_SHAPE = (3, 384, 512)
    OUTPUT_SHAPE = (3, 384//2, 512//2)

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

        #two inputs: the original image and the coarse image.
        net["input"] = InputLayer(shape=inp_shp,
            input_var=input_var)
        net["input_coarse"] = PoolLayer(net["input"],
            pool_size=(1, 1), stride=(2, 2), ignore_border=True)

        #vgg-16 layers on normal input
        net["conv1_1"] = ConvLayer(net["input"],
            64, 3, pad=1, flip_filters=False)
        net["conv1_2"] = ConvLayer(net["conv1_1"],
            64, 3, pad=1, flip_filters=False)
        net["pool1"] = PoolLayer(net["conv1_2"], 2)

        net["conv2_1"] = ConvLayer(net["pool1"],
            128, 3, pad=1, flip_filters=False)
        net["conv2_2"] = ConvLayer(net["conv2_1"],
            128, 3, pad=1, flip_filters=False)
        net["pool2"] = PoolLayer(net["conv2_2"], 2)

        net["conv3_1"] = ConvLayer(net["pool2"],
            256, 3, pad=1, flip_filters=False)
        net["conv3_2"] = ConvLayer(net["conv3_1"],
            256, 3, pad=1, flip_filters=False)
        net["conv3_3"] = ConvLayer(net["conv3_2"],
            256, 3, pad=1, flip_filters=False)
        net["pool3"] = PoolLayer(net["conv3_3"], 2)

        net["conv4_1"] = ConvLayer(net["pool3"],
            512, 3, pad=1, flip_filters=False)
        net["conv4_2"] = ConvLayer(net["conv4_1"],
            512, 3, pad=1, flip_filters=False)
        net["conv4_3"] = ConvLayer(net["conv4_2"],
            512, 3, pad=1, flip_filters=False)
        net["pool4"] = PoolLayer(net["conv4_3"], 2)

        net["conv5_1"] = ConvLayer(net["pool4"],
            512, 3, pad=1, flip_filters=False)
        net["conv5_2"] = ConvLayer(net["conv5_1"],
            512, 3, pad=1, flip_filters=False)
        net["conv5_3"] = ConvLayer(net["conv5_2"],
            512, 3, pad=1, flip_filters=False)
        net["partial_output"] = net["conv5_3"]

        #vgg-16 layers on coarse input
        net["conv1_1_coarse"] = ConvLayer(net["input_coarse"],
            64, 3, W=net["conv1_1"].W, pad=1, flip_filters=False)
        net["conv1_2_coarse"] = ConvLayer(net["conv1_1_coarse"],
            64, 3, W=net["conv1_2"].W, pad=1, flip_filters=False)
        net["pool1_coarse"] = PoolLayer(net["conv1_2_coarse"], 2)

        net["conv2_1_coarse"] = ConvLayer(net["pool1_coarse"],
            128, 3, W=net["conv2_1"].W, pad=1, flip_filters=False)
        net["conv2_2_coarse"] = ConvLayer(net["conv2_1_coarse"],
            128, 3, W=net["conv2_2"].W, pad=1, flip_filters=False)
        net["pool2_coarse"] = PoolLayer(net["conv2_2_coarse"], 2)

        net["conv3_1_coarse"] = ConvLayer(net["pool2_coarse"],
            256, 3, W=net["conv3_1"].W, pad=1, flip_filters=False)
        net["conv3_2_coarse"] = ConvLayer(net["conv3_1_coarse"],
            256, 3, W=net["conv3_2"].W, pad=1, flip_filters=False)
        net["conv3_3_coarse"] = ConvLayer(net["conv3_2_coarse"],
            256, 3, W=net["conv3_3"].W, pad=1, flip_filters=False)
        net["pool3_coarse"] = PoolLayer(net["conv3_3_coarse"], 2)

        net["conv4_1_coarse"] = ConvLayer(net["pool3_coarse"],
            512, 3, W=net["conv4_1"].W, pad=1, flip_filters=False)
        net["conv4_2_coarse"] = ConvLayer(net["conv4_1_coarse"],
            512, 3, W=net["conv4_2"].W, pad=1, flip_filters=False)
        net["conv4_3_coarse"] = ConvLayer(net["conv4_2_coarse"],
            512, 3, W=net["conv4_3"].W, pad=1, flip_filters=False)
        net["pool4_coarse"] = PoolLayer(net["conv4_3_coarse"], 2)

        net["conv5_1_coarse"] = ConvLayer(net["pool4_coarse"],
            512, 3, W=net["conv5_1"].W, pad=1, flip_filters=False)
        net["conv5_2_coarse"] = ConvLayer(net["conv5_1_coarse"],
            512, 3, W=net["conv5_2"].W, pad=1, flip_filters=False)
        net["conv5_3_coarse"] = ConvLayer(net["conv5_2_coarse"],
            512, 3, W=net["conv5_3"].W, pad=1, flip_filters=False)
        net["partial_output_coarse"] = UpscaleLayer(net["conv5_3_coarse"],
            scale_factor=2, mode="repeat")

        print("conv5_3:",
            lasagne.layers.get_output_shape(net["conv5_3"]))
        print("conv5_3_coarse:",
            lasagne.layers.get_output_shape(net["conv5_3_coarse"]))
        print("partial_output:",
            lasagne.layers.get_output_shape(net["partial_output"]))
        print("partial_output_coarse:",
            lasagne.layers.get_output_shape(net["partial_output_coarse"]))
        print("conv5_3_coarse:",
            lasagne.layers.get_output_shape(net["conv5_3_coarse"]))
        net["concat"] = ConcatLayer(
            [net["partial_output"], net["partial_output_coarse"]],
            axis=0)
        print("concat:",
            lasagne.layers.get_output_shape(net["concat"]))
        net["output"] = ConvLayer(net["concat"], 1, 1)
        print("output:",
            lasagne.layers.get_output_shape(net["output"]))

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
