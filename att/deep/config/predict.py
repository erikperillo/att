"""
Configuration file for predict script.
"""

import os

from . import train
from . import datapreproc

#directory where dataset is stored. can be empty
dataset_dirpath = os.path.dirname(train.dataset_filepath)
#filepath of dataset statistics (mean/std etc)
dataset_stats_filepath = os.path.join(dataset_dirpath, "data_stats.gz")
#filepath of model
model_filepath = os.path.join(train.output_dir_basedir, "train", "model.npz")

#normalize images per channel
normalize_per_channel = datapreproc.x_normalize_per_channel
#normalization method (std, unit, nonde)
norm_method = datapreproc.x_normalization
#colorspace to transform image
img_colspace = datapreproc.x_img_colspace
#if true, crops image before resizing so as to get the same aspect ratio
#as the target shape
crop_on_resize = True
#if true, swaps axis
swap_img_channel_axis = datapreproc.swap_channel_axis

#if true, saves each prediction as <filepath>_fixmap.png
save_preds = False

#show images if true
show_images = True
