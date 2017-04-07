import os
import glob

from . import datapreproc

#directory where dir with train info/model will be stored
output_dir_basedir = os.path.join(datapreproc.output_dir_basedir,
    datapreproc.dataset_name + "_dataset_2")

_dataset_filepaths = glob.glob(os.path.join(output_dir_basedir,"data_part*.gz"))
#filepaths of train batches
dataset_train_filepaths = _dataset_filepaths[0:2]
#filepaths of validation batches
dataset_val_filepaths = None#_dataset_filepaths[2:3]
#if not None, ignores dataset_{train,val}_filepaths and uses this as source
dataset_filepath = None#_dataset_filepaths[-1]
#validation fraction. ignored if dataset_filepath is not used
val_frac = 0.1

#number of epochs to use in train
n_epochs = 2
#batch size
batch_size = 10
#maximum number of iterations
max_its = None
#0 for nothing, 1 for only warnings, 2 for everything
verbose = 2
#validation function value tolerance
val_f_val_tol = None
