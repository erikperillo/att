"""
Configuration file for eval script.
"""

import os
import glob

_data_dir_path = "/home/erik/docs/att_experiments/colorspace_rgb_vs_lab/"

#filepath of model
model_filepath = os.path.join(_data_dir_path, "train_salicon_lab", "model.npz")

filepaths = glob.glob(os.path.join("/home/erik/proj/att/att/deep/data",
    "salicon_dataset_lab", "data_part_4.gz"))

metrics = ["cc", "sim", "mse"]
