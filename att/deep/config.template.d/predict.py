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
model_filepath = os.path.join(_data_dir_path, "train_51", "model.npz")

#colorspace to transform image
img_colspace = "lab"
#normalization method (std, unit, none)
img_normalization = "std"
#normalize per image
normalize_per_image = True
#resize image to Model.INPUT_SHAPE if needed
resize = False
#if true, crops image before resizing for the same aspect ratio as target shape
crop_on_resize = True
#if true, swaps axis
swap_img_channel_axis = True

#data type
img_dtype = "float32"

#if true, saves each prediction as <filepath>_fixmap.png
save_preds = True

#show images if true
show_images = False

#maximum input shape, can be None
max_img_shape = (480, 640)
