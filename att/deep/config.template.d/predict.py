import os

from . import train
from . import datapreproc

dataset_dirpath = os.path.dirname(train.dataset_filepath)
dataset_stats_filepath = os.path.join(dataset_dirpath, "data_stats.gz")
normalize_per_channel = datapreproc.x_normalize_per_channel
norm_method = datapreproc.x_normalization
crop_on_resize = True
model_filepath = "/home/erik/proj/att/att/deep/data/train/model.npz"
