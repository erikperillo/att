#!/usr/bin/env python3

import numpy as np
import theano
import theano.tensor as T
import lasagne
import random
import shutil
import sys
import os

#local modules
import model
import trloop
import util

DATASET_FILEPATH = "./data/judd_cat2000_dataset_1/data.gz"
OUTPUT_DIR_BASEDIR = "./data"
N_EPOCHS = 2
BATCH_SIZE = 1
MAX_ITS = None
VERBOSE = 2
CV_FRAC = 0.1
TE_FRAC = 0.1

def load_dataset(filepath):
    return util.unpkl(filepath)

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

def mk_output_dir(base_dir, pattern="train"):
    #creating dir
    out_dir = util.uniq_filepath(base_dir, pattern)
    os.makedirs(out_dir)
    return out_dir

def populate_output_dir(out_dir):
    #copying model generator file to dir
    shutil.copy(model.__file__, os.path.join(out_dir, "genmodel.py"))
    #copying this file to dir
    shutil.copy(__file__, os.path.join(out_dir, "genscript.py"))
    #info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("git commit hash:", util.git_hash(), file=f)

def main():
    X_tr, y_tr, X_cv, y_cv, X_te, y_te = load_formatted_dataset(
        DATASET_FILEPATH, cv_frac=CV_FRAC, te_frac=TE_FRAC)

    #theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    out_dir = mk_output_dir(OUTPUT_DIR_BASEDIR)
    print("created output dir '%s'..." % out_dir)
    populate_output_dir(out_dir)

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

    print("calling train loop")
    #creating logging object
    log = util.Tee([sys.stdout, open(os.path.join(out_dir, "train.log"), "w")])
    try:
        trloop.train_loop(
            X_tr, y_tr, train_fn,
            n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
            X_val=X_cv, y_val=y_cv, val_f=val_fn,
            val_mae_tol=None,
            max_its=MAX_ITS,
            verbose=VERBOSE,
            print_f=log.print)
    except KeyboardInterrupt:
        print("Keyboard Interrupt event.")

    print("end.")
    err, mae = val_fn(X_te, y_te)
    print("test loss: %f | test mae: %f" % (err, mae))

    model_path = os.path.join(out_dir, "model.npz")
    print("saving model to '{}'".format(model_path))
    net_model.save_net(model_path)

    print("\ndone.")

if __name__ == '__main__':
    main()
