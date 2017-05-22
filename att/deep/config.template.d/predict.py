"""
Configuration file for predict script.
"""

import os
import glob

from . import train
from . import datapreproc

_data_dir_path = "/home/erik/proj/att/att/deep/data"
#filepath of dataset statistics (mean/std etc)
dataset_stats_filepath = None#os.path.join(_data_dir_path,
    #"judd_cat2000_dataset/data_stats.pkl")
#filepath of model
model_filepath = os.path.join(_data_dir_path, "train_3", "model.npz")

#colorspace to transform image
img_colspace = datapreproc.x_img_colspace
#normalization method (std, unit, nonde)
img_normalization = datapreproc.x_normalization
#normalize per image
normalize_per_image = datapreproc.x_normalize_per_image
#if true, crops image before resizing for the same aspect ratio as target shape
crop_on_resize = True
#if true, swaps axis
swap_img_channel_axis = datapreproc.swap_channel_axis

#data type
img_dtype = datapreproc.x_dtype

#if true, saves each prediction as <filepath>_fixmap.png
save_preds = True

#show images if true
show_images = False
