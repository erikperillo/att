"""
Configuration file for datapreproc script.
"""

import numpy as np
import os

#if dataset path isn't one in defaults (judd, cat2000, cssd, ecssd, mit_300),
#then it must contain a dir named stimuli (with only stimuli images)
#and a dir named ground_truth for ground-truths (with the same starting
#names as its respectives stimuli).
dataset_path = "/home/erik/data/saliency_datasets/salicon"
#name of dataset. it must be empty if not one in defaults.
dataset_name = os.path.basename(dataset_path).lower()
#save data, info, etc in dir created inside dir specified by this var
output_dir_basedir = "/home/erik/proj/att/att/deep/data"

#maximum number of samples to use, if None use all
max_samples = None
#save data in parts. if None, save everything in only one file.
data_save_batch_size = 500

#show images
show_images = False
show_channels = False

#downscale sampling factor of pyramid. if not None, ignores x_shape and y_shape.
x_pyramid = 0
y_pyramid = 3
#image shape
x_shape = (480, 640)
y_shape = (120, 160)

#crop image to have final dimension's proportions before resizing.
crop_on_resize = True

#end datatype
x_dtype = np.float32
y_dtype = np.float32

#float datatype
x_img_to_float = True
y_img_to_float = True

#x normalization (std/unit/max)
x_normalization = "std"
x_normalize_per_image = True
#y normalization
y_normalization = "std"
y_normalize_per_image = True

#input colorspace
x_img_colspace = "lab"

#swap channel axis, eg. from shape (200, 200, 3) to (3, 200, 200)
swap_channel_axis = True

#augmentation techniques
augment = True
#flip horizontally/vertically
hor_mirror = True
ver_mirror = False
#rotations, translations, etc
affine_transforms = [
    #{
    #    "shear": 0.3,
    #}
]
#gets a corner from image, eg. 0.6 tl_corner gets 60% of image from top left.
#top left
tl_corner = None
#top right
tr_corner = None#0.666
#bottom left
bl_corner = None#0.666
#bottom right
br_corner = None#0.8

#seed to be used by random module
rand_seed = 9
