import numpy as np
import os

#if dataset path isn't one in defaults (judd, cat2000, cssd, ecssd, mit_300),
#then it must contain a dir named stimuli (with only stimuli images)
#and a dir named ground_truth for ground-truths (with the same starting
#names as its respectives stimuli).
dataset_path = "/home/erik/proj/ic/saliency_datasets/judd_cat2000"
#name of dataset. it must be empty if not one in defaults.
dataset_name = os.path.basename(dataset_path).lower()
#output paths
out_data_filepath = ""
out_data_stats_filepath = ""
#if below is set, overrides out_data_{,stats_}filepath and puts data along
#with more info inside a directory created on dir specified by this variable.
output_dir_basedir = "/home/erik/proj/att/att/deep/data"

#maximum number of samples to use, if None use all
max_samples = 10

#show images
show_images = False
show_channels = False

#image shape
x_shape = (80, 120)
y_shape = (20, 30)

#crop image to have final dimension's proportions before resizing.
crop_on_resize = True

#end datatype
x_dtype = np.float64
y_dtype = np.float64

#float datatype
x_img_to_float = True
y_img_to_float = True

#normalization
x_normalization = "std"
x_normalize_per_channel = True
y_normalization = "std"

#input colorspace
x_img_colspace = "lab"

#swap channel axis, eg. from shape (200, 200, 3) to (3, 200, 200)
swap_channel_axis = True

#augmentation techniques
augment = False
#flip horizontally/vertically
hor_mirror = False
ver_mirror = False
#rotations, translations, etc
affine_transforms = [
    #{
    #    "shear": 0.3,
    #}
]
#gets a corner from image, eg. 0.6 tl_corner gets 60% of image from top left.
#top left
tl_corner = 0.666
#top right
tr_corner = None#0.666
#bottom left
bl_corner = None#0.666
#bottom right
br_corner = 0.666

#seed to be used by random module
rand_seed = 42
