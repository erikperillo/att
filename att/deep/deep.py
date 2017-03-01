#!/usr/bin/env python3

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle

DATASET_FILEPATH = "./data/cat2000.pkl"
INPUT_SHAPE = (3, 58, 98)

def load_dataset(filepath):
    with open(filepath, "rb") as f:
        X, y = pickle.load(f)
    return X, y

def tr_cv_te_split(X, y, cv_frac=0.2, te_frac=0.1):
    X, y = X[:100], y[:100]
    cv = int(cv_frac*len(y))
    te = int(te_frac*len(y))
    X_tr, y_tr = X[:-(cv+te)], y[:-(cv+te)]
    X_cv, y_cv = X[-(cv+te):-te], y[-(cv+te):-te]
    X_te, y_te = X[-te:], y[-te:]
    return X_tr, y_tr, X_cv, y_cv, X_te, y_te

def load_formatted_dataset(filepath, cv_frac=0.2, te_frac=0.1):
    print("Loading data...")
    X, y = load_dataset(filepath)
    print("X shape: {} | y shape: {}".format(X.shape, y.shape))

    print("Splitting...")
    X_tr, y_tr, X_cv, y_cv, X_te, y_te = tr_cv_te_split(X, y, cv_frac, te_frac)
    print("X_tr shape: {} | y_tr shape: {}".format(X_tr.shape, y_tr.shape))
    print("X_cv shape: {} | y_cv shape: {}".format(X_cv.shape, y_cv.shape))
    print("X_te shape: {} | y_te shape: {}".format(X_te.shape, y_te.shape))

    print("Reshaping...")
    X_tr = X_tr.reshape((-1,) + INPUT_SHAPE)
    X_cv = X_cv.reshape((-1,) + INPUT_SHAPE)
    X_te = X_te.reshape((-1,) + INPUT_SHAPE)
    print("X_tr shape: {} | y_tr shape: {}".format(X_tr.shape, y_tr.shape))
    print("X_cv shape: {} | y_cv shape: {}".format(X_cv.shape, y_cv.shape))
    print("X_te shape: {} | y_te shape: {}".format(X_te.shape, y_te.shape))

    return X_tr, y_tr, X_cv, y_cv, X_te, y_te

def build_cnn(input_shape, input_var=None):
    network = lasagne.layers.InputLayer(shape=input_shape,
                                        input_var=input_var)

    #input shape in form n_batches, depth, rows, cols
    output_shape = input_shape[-2], input_shape[-1]

    #input
    network = lasagne.layers.InputLayer(shape=input_shape, input_var=input_var)

    # Convolutional layer with 16 kernels of size 3x3. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of x units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=output_shape[0]*output_shape[1],
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=output_shape[0]*output_shape[1],
            nonlinearity=lasagne.nonlinearities.identity)

    return network

def main():
    X_tr, y_tr, X_cv, y_cv, X_te, y_te = load_formatted_dataset(
        DATASET_FILEPATH, cv_frac=0.2, te_frac=0.1)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    #target_var = T.ivector('targets')
    target_var = T.matrix('targets')

    # Create neural network model
    print("building network...", flush=True)
    network = build_cnn((None,) + INPUT_SHAPE, input_var)

    print("making symbolic functions...", flush=True)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = T.sqrt(loss.sum())
    # We could add some weight decay as well here, see lasagne.regularization.
    reg = lasagne.regularization.regularize_network_params(network,
        lasagne.regularization.l2)
    loss += reg*0.0001

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.95)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = T.sqrt(test_loss.sum())
    #test_loss = test_loss + reg*0.001
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(abs(test_prediction - target_var))

    print("compiling functions...", flush=True)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("calling loop")
    import trloop
    trloop.train_loop(
        X_tr, y_tr, train_fn,
        n_epochs=16, batch_size=20,
        X_val=X_cv, y_val=y_cv, val_f=val_fn,
        val_acc_tol=None,
        max_its=None,
        verbose=2)
    print("end.")
    err, acc = val_fn(X_test, y_test)
    print("test loss: %f | test acc: %f" % (err, acc))
    exit()
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        print("in epoch %d" % epoch)
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            err = train_fn(inputs, targets)
            train_err += err
            train_batches += 1
            print("\t\r in train batch %d | err: %.4g    " %\
                (train_batches, err), end="")

        print("")
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            print("\t\r in val batch %d | err: %.4g, acc: %f    " %\
                (val_batches, err, acc), end="")

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

if __name__ == '__main__':
    main()
