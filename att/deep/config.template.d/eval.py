"""
Configuration file for eval script.
"""

import os
import glob

_data_dir_path = "/home/erik/proj/att/att/deep/data"

#filepath of model
model_filepath = os.path.join(_data_dir_path, "train_8", "model.npz")

filepaths = [os.path.join(_data_dir_path, "salicon_dataset_lab",
    "data_part_4.gz")]

metrics = ["cc", "sim", "mse"]
