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
import json
from functools import partial

import util
import augment
import dproc
import model

def _read_stats_file(path):
    """
    Reads JSON statistics file from path, returning a dict.
    """
    with open(path, "r") as f:
        dct = json.load(f)
    dct = {int(k): v for k, v in dct.items()}
    return dct

def _last_touched(dir_path, pattern="*"):
    """
    Returns the path to file/dir in dir_path with most recent edit date.
    """
    paths = glob.glob(os.path.join(dir_path, pattern))
    if not paths:
        return ""
    path = max(paths, key=os.path.getmtime)
    return path

### PATHS CONFIGURATION
#directory to save sessions. a session dir contains a trained model and it's
#predictions/evaluations. assumes below dir is already created
_sessions_dir_path = "/home/erik/data/ml/sessions"
assert os.path.isdir(_sessions_dir_path)

#path for a to-be-created new session directory
_new_sess_dir_path = util.uniq_path(_sessions_dir_path, "sess")
#path to the most recently modified session dir
_last_touched_sess_dir_path = _last_touched(_sessions_dir_path)

#output to save all train-related output files for a given train run
_new_train_output_dir_path = os.path.join(_new_sess_dir_path, "trains")
_last_touched_train_output_dir_path = \
    os.path.join(_last_touched_sess_dir_path, "trains")
#output to save all infer-related output files
_infer_output_dir_path = os.path.join(_last_touched_sess_dir_path, "infers")
#output to save all assay-related output files
_assay_output_dir_path = os.path.join(_last_touched_sess_dir_path, "assays")

#path to base output dir of datagen pipeline
_datagen_output_dir_path = "/mnt/db/rand/croplandnet"
#path to dir that contains train/val/test set lists and train stats
dataset_dir_path = os.path.join(_datagen_output_dir_path, "refine_20m")
#path to train set statistics json file
#_train_set_stats_path = os.path.join(dataset_dir_path, "train_stats.json")

#directory containing this file
_config_file_dir = os.path.dirname(os.path.abspath(__file__))
#directory containing data
_data_dir = os.path.join(os.path.dirname(_config_file_dir), "data")

#path to dataset
dataset_path = "/home/erik/data/sal-dsets/salicon"

#path to statistics of train set
_stats_path = os.path.join(_data_dir, "salicon_train-set_lab-stats.json")

#path to list of train uids
_train_uids_list_path = os.path.join(_data_dir, "salicon_train-set.csv")
#path to list of val uids
_val_uids_list_path = os.path.join(_data_dir, "salicon_val-set.csv")
#path to list of test uids
_test_uids_list_path = os.path.join(_data_dir, "salicon_test-set.csv")

### DATA PROCESSING CONFIGURATION ###
#shape of input/output
_x_shape = model.X_SHAPE[-3:]
_y_shape = model.Y_SHAPE[-3:]

# these stats are fed to model; they are used for pre-processing before unet
_train_set_stats = _read_stats_file(_stats_path)

#data augmentation operations
_augment_ops = [
    ("hmirr", {}, 0.5),
    ("rotation", {"angle": (-40, 40)}, 0.15),
    ("add_noise", {"max_noise_amplitude": 0.25}, 0.15),
    ("mul_noise", {"noise": (0.7, 1.3)}, 0.15),
    ("blur", {"sigma": (1.0, 2.0)}, 0.15),
    ("translation", {"transl": (-0.3, 0.3)}, 0.15),
    ("shear", {"angle": (-0.3, 0.3)}, 0.15),
]


### TRAIN CONFIGURATION ###
train = {
    #base directory where new directory with train data will be created
    #NOTE that this argument can be overriden by the command line
    "output_dir_path": _new_train_output_dir_path,

    #path to directory containing data needed by tensorflow's SavedModel,
    #can be None
    #NOTE that this argument can be overriden by the command line
    "pre_trained_model_path": \
        None,

    #list (or path to list file) with paths of train set
    #NOTE that this argument can be overriden by the command line
    "train_set": _train_uids_list_path,

    #list (or path to list file) with paths of validation set
    #NOTE that this argument can be overriden by the command line
    "val_set": _val_uids_list_path,

    #use tensorboard summaries
    "use_tensorboard": True,
    #tensorboard server parameters
    "tensorboard_params": {
        "host": "0.0.0.0",
        "port": 6006,
    },

    #model construction args
    "meta_model_kwargs": {
        #train-set statistics
        # (in form of a dict {band_index: {'mean': mean, 'std': std}}
        #"stats": _train_set_stats,
    },

    #learning rate of the model
    "learning_rate": 1e-4,

    #number of epochs for training loop. can be None
    "n_epochs": 128,

    #logs metrics every log_every_its, can be None
    "log_every_its": 50,

    #computes metrics on validation set every val_every_its. can be None
    "val_every_its": None,

    #number of times val set loss does not improve before early stopping.
    #can be None, in which case early stopping will never occur.
    "patience": 2,

    #save checkpoint with graph/weights every save_every_its besides epochs.
    #can be None
    "save_every_its": None,

    #verbosity
    "verbose": 2,

    #arguments to be provided by trloop.batch_gen function
    "batch_gen_kw": {
        #size of batch to be fed to model
        "batch_size": 8,

        #number of fetching threads for data loading/pre-processing/augmentation
        "n_threads": 14,

        #maximum number of samples to be loaded at a time.
        #the actual number may be slightly larger due to rounding.
        "max_n_samples": 2048,

        #function to return tuple (x, y_true) given filepath
        "fetch_thr_load_fn": \
            partial(dproc.train_load, dset_path=dataset_path),
            #lambda uid: dproc.train_load(uid, dataset_path),

        #function to return (possibly) augmented image
        "fetch_thr_augment_fn": \
            partial(augment.augment, operations=_augment_ops),
            #lambda xy: augment.augment(xy, operations=_augment_ops),

        #function pre-process xy
        # NOTE that there are two pre-processing steps:
        # 1. the pre-proc that happens before feeding the input to the model;
        # 2. the pre-proc that happens before feeding the input to the unet.
        # fetch_thr_pre_proc_fn refers to the first step.
        "fetch_thr_pre_proc_fn": \
            partial(dproc.train_pre_proc,
                x_shape=_x_shape[-2:], y_shape=_y_shape[-2:]),
            #lambda xy: dproc.train_pre_proc(xy, _x_shape[-2:], _y_shape[-2:]),
    },

    "log_batch_gen_kw": {
        "n_threads": 2,
        "max_n_samples": 512,
    },

    #random seed for reproducibility
    "rand_seed": 135582,
}


### INFER CONFIGURATION ###
infer = {
    #input filepaths.
    #it's either a list or a path (str).
    #if it's a path to a .csv file, reads it's content line by line as a list.
    #if it's a path to a directory, gets the content of the directory.
    #otherwise consider it to be a path to a single file.
    #NOTE that this argument can be overriden by the command line
    "input_paths": \
        os.path.join(dataset_dir_path, "test_tiles.csv"),

    #base dir where new preds directory will be created
    #NOTE that this argument can be overriden by the command line
    "output_dir_path": _infer_output_dir_path,

    #path to directory containing meta-graph and weights for model
    #NOTE that this argument can be overriden by the command line
    "model_path": \
        os.path.join(_last_touched(_last_touched_train_output_dir_path),
            "self", "ckpts", "best"),

    #shape of cut for strided_predict method (can also be an int)
    "strided_predict_shape": _x_shape[-2:],

    #shape of stride for strided_predict method (can also be an int)
    "strided_predict_stride": 2*_x_shape[-1]//3,

    #rotations to use in averaged_predict method.
    #0 is the original image, 1 is image rotated 90 degrees (cc) 1 time etc
    "rot_averaged_predict_rotations": [0],

    #wether or not compute prediction for reflected image as well and
    #average the results
    "hmirr_averaged_predict": True,

    #maximum number of predictions. can be None
    "max_n_preds": None,

    #model construction args
    "meta_model_kwargs": {
    },

    #function to load input file
    "load_fn": dproc.infer_load_judd,

    #function to load input x (before going into model)
    "pre_proc_fn": partial(dproc.infer_pre_proc, x_shape=_x_shape[-2:]),

    #function to save prediction given path and y_pred
    "save_y_pred_fn": dproc.infer_save_y_pred,

    #random seed to be used, can be None
    "rand_seed": 88,
}

## ASSAY CONFIGURATION ##
assay = {
    #input filepaths.
    #it's either a list or a path (str).
    #if it's a path to a .csv file, reads it's content line by line as a list.
    #if it's a path to a directory, gets the content of the directory.
    #otherwise consider it to be a path to a single file.
    #NOTE that this argument can be overriden by the command line
    "input_paths": \
        _last_touched(_infer_output_dir_path),

    #directory to save evaluation output
    #NOTE that this argument can be overriden by the command line
    "output_dir_path": _assay_output_dir_path,

    #directory of true labels images (assumes all y_trues
    #are in one dir and they have the same filename as corresponding y_pred)
    #NOTE that this argument can be overriden by the command line
    "y_true_dir_path": \
        os.path.join(_datagen_output_dir_path, "tiles", "x_y_alpha"),

    #decision threshold to use in binary classification.
    #can be a float value in [0, 1] or 'auto'.
    #if 'auto' is used, gets best decision threshold via get_best_thr method
    #NOTE that this argument can be overriden by the command line
    "decision_thr": "auto",

    #maximum number of points to use for evaluation, will subsample
    #randomly from input paths. can be None (all points used)
    "max_n_points": 3*10**8,

    #threshold step to use in metrics computations (from 0 to 1)
    "thr_step": 0.01,

    #function to load prediction given path
    "load_y_pred_fn": dproc.assay_load_y_pred,

    #function to load true label given prediction path
    "load_y_true_fn": dproc.assay_load_y_true,

    #keyword arguments for plot_roc method
    "plot_roc_kwargs": {
        #display after generating
        "show": False,
    },

    #keyword arguments for plot_precision_recall_acc method
    "plot_precision_recall_acc_kwargs": {
        #plot f1-score
        "f1_score": True,
        #display after generating
        "show": False,
    },

    #keyword arguments for get_best_thr method
    "get_best_thr_kwargs": {
        #choose the threshold that maximizes this metric
        #"sort_by": "f1_score",
        "sort_by": "iou",
    },

    #keyword arguments for plot_conf_mtx method
    "plot_conf_mtx_kwargs": {
        #normalize numbers by total
        "normalize": True,
        #display after generating
        "show": False,
    },

    #random seed
    "rand_seed": 91,
}

#for dct in train, infer, assay:
#    for k, v in dct.items():
#        print("\t{}: {}".format(k, v))
#    print("\n-------------------\n")
#exit()
