import os

from . import datapreproc

#filepath of dataset
dataset_filepath = os.path.join(datapreproc.output_dir_basedir,
    datapreproc.dataset_name + "_dataset",
    "data.gz")
#directory where dir with train info/model will be stored
output_dir_basedir = datapreproc.output_dir_basedir

#number of epochs to use in train
n_epochs = 2
#batch size
batch_size = 1
#maximum number of iterations
max_its = None
#0 for nothing, 1 for only warnings, 2 for everything
verbose = 2
#cross-validation fraction
cv_frac = 0.1
#train fraction
te_frac = 0.1
