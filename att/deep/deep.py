#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T
import lasagne

#local modules
import model
import trloop
import util

DATASET_FILEPATH = "./data/judd.gz"

def load_dataset(filepath):
    return util.unpickle(filepath)

def tr_cv_te_split(X, y, cv_frac=0.2, te_frac=0.1):
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
    X_tr = X_tr.reshape((X_tr.shape[0],) + model.Model.INPUT_SHAPE)
    X_cv = X_cv.reshape((X_cv.shape[0],) + model.Model.INPUT_SHAPE)
    X_te = X_te.reshape((X_te.shape[0],) + model.Model.INPUT_SHAPE)
    print("X_tr shape: {} | y_tr shape: {}".format(X_tr.shape, y_tr.shape))
    print("X_cv shape: {} | y_cv shape: {}".format(X_cv.shape, y_cv.shape))
    print("X_te shape: {} | y_te shape: {}".format(X_te.shape, y_te.shape))

    return X_tr, y_tr, X_cv, y_cv, X_te, y_te

def main():
    X_tr, y_tr, X_cv, y_cv, X_te, y_te = load_formatted_dataset(
        DATASET_FILEPATH, cv_frac=0.1, te_frac=0.01)

    #theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    #neural network model
    print("building network...", flush=True)
    net_model = model.Model(input_var, target_var)

    print("compiling functions...", flush=True)
    #compiling function performing a training step on a mini-batch (by giving
    #the updates dictionary) and returning the corresponding training loss
    train_fn = theano.function([input_var, target_var],
        net_model.train_loss, updates=net_model.updates)
    #second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var],
        [net_model.test_loss, net_model.mae])

    print("calling loop")
    trloop.train_loop(
        X_tr, y_tr, train_fn,
        n_epochs=32, batch_size=20,
        X_val=X_cv, y_val=y_cv, val_f=val_fn,
        val_mae_tol=None,
        max_its=None,
        verbose=2)
    print("end.")

    err, mae = val_fn(X_te, y_te)
    print("test loss: %f | test mae: %f" % (err, mae))

    save_path = "model.npz"
    print("saving model to '%s'..." % save_path)
    net_model.save_net(save_path)

if __name__ == '__main__':
    main()
