"""
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import random
import numpy as np
import os
import glob

import util
import augment
import dproc

#data augmentation default operation sequence to be applied in about every op
_def_augm_ops = [
    ("blur", 0.15, {"rng": (0.5, 1.0)}),
    ("translation", 0.15, {"rng": (-0.1, 0.1)}),
    ("rotation", 0.15, {"rng": (-35, 35)}),
    ("shear", 0.15, {"rng": (-0.1, 0.1)}),
    ("add_noise", 0.15, {"rng": (-0.1, 0.1)}),
    ("mul_noise", 0.15, {"rng": (0.9, 1.1)}),
]

"""
Sequences of operations for data augmentation.
Each sequence spans a new image and applies operations randomly as defined by
    their probabilities.
Each sequence must contain operations in format (op_name, op_prob, op_kwargs).
"""
_augment_op_seqs = [
    [
        ("identity", 1.0, {}),
    ] + _def_augm_ops,

    [
        ("hmirr", 1.0, {}),
    ] + _def_augm_ops,
]

_model_data_dir = os.environ.get("MODEL_DATA_DIR_PATH",
    os.path.join(os.environ["HOME"], "data", "deeppeek", "self"))
_train_input_dir = os.environ.get("TRAIN_INPUT_DIR_PATH",
    os.path.join(os.environ["HOME"], "data", "sal"))
_train_output_dir = os.environ.get("TRAIN_OUTPUT_DIR_PATH",
    os.path.join(os.environ["HOME"], "data", "deeppeek", "train"))

#filepaths
#_fps = glob.glob(os.path.join(_train_input_dir, "salicon", "maps", "*.jpg"))
_fps = glob.glob(os.path.join(_train_input_dir, "judd", "stimuli", "*"))
random.seed(42)
random.shuffle(_fps)
_train_fps = _fps[:875]
_val_fps = _fps[875:]
#_test_fps = glob.glob(os.path.join(_train_input_dir, "judd", "stimuli", "*"))
_test_fps = glob.glob(os.path.join(_train_input_dir, "mit_300",
    "BenchmarkIMAGES", "BenchmarkIMAGES", "*.jpg"))

#augment function
_augment = lambda xy: augment.augment(xy, _augment_op_seqs, apply_on_y=True)

#arguments for train routine
train = {
    #base directory where new directory with train data will be created
    "out_dir_basedir": _train_output_dir,

    #use tensorboard summaries
    "use_tensorboard": True,
    "tensorboard_port": 6006,

    #path to directory containing data needed by tensorflow's SavedModel
    #can be None
    "pre_trained_model_path": \
        #None,
        os.path.join(_model_data_dir, "ckpts", "best"),

    #learning rate of the model
    "learning_rate": 1e-4,

    #list with filepaths of train files
    "train_set_fps": _train_fps,

    #list with filepaths of validation files
    "val_set_fps": _val_fps,

    #number of epochs for training loop. can be None
    "n_epochs": 64,

    #logs metrics every log_every_its, can be None
    "log_every_its": 30,

    #computes metrics on validation set every val_every_its. can be None
    "val_every_its": None,

    #number of times val set loss does not improve before early stopping.
    #can be None, in which case early stopping will never occur.
    "patience": 4,

    #save checkpoint with graph/weights every save_every_its besides epochs.
    #can be None
    "save_every_its": None,

    #verbosity
    "verbose": 2,

    #arguments to be provided by trloop.batch_gen function
    "batch_gen_kw": {
        #size of batch to be fed to model
        "batch_size": 12,

        #number of fetching threads for data loading/pre-processing/augmentation
        "n_threads": 10,

        #maximum number of samples to be loaded at a time.
        #the actual number may be slightly larger due to rounding.
        "max_n_samples": 8000,

        #the fetching threads loads a chunk of this size before augmentation
        #and pre-processing.
        #this spreads out the augmented versions of an image in the feeding line
        "fetch_thr_load_chunk_size": 8,

        #function to return tuple (x, y_true) given filepath
        "fetch_thr_load_fn": dproc.train_load,

        #function to return list of tuples [(_x, _y), ...] given (x, y) tuple
        "fetch_thr_augment_fn": _augment,

        #function to return batches of x (optionally (x, y))
        #given batches of x (optionally y as second argument)
        "fetch_thr_pre_proc_fn": dproc.train_pre_proc,

        #the maximum factor by which number of samples will be increased
        #due to data augmentation
        "max_augm_factor": len(_augment_op_seqs),
    },
}

#arguments for predict routine
infer = {
    #random seed to be used, can be None
    "rand_seed": 42,

    #list of input filepaths containing x values (optionally (x, y_true) tuples)
    "input_fps": _test_fps,

    #whether or not shuffle list of input filepaths
    "shuffle_input_fps": True,

    #path to directory containing meta-graph and weights for model
    "model_path": os.path.join(_model_data_dir, "ckpts", "best"),

    #base dir where new preds directory will be created
    "preds_save_dir_basedir": os.path.join(_train_output_dir, "inferences"),

    #if true, creates table.npz, containing x_fps, y_pred (and possibly y_true)
    "save_tables": not True,

    #if true, tries to load true values with load_fn
    "with_trues": not True,

    #maximum prediction data points to be stored, can be None
    "max_pred_points": 9999999,

    #maximum number of preds to save, can be None
    "max_n_preds_save": 999999,
}
