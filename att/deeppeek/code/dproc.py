"""
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

"""
Module for data processing.
"""

from skimage import io
from skimage import transform as transf
from skimage import color
from skimage import img_as_float

import numpy as np

import glob
import os

import util

def _unit_norm(x, eps=1e-6):
    return (x - x.min())/max(x.max() - x.min(), eps)

def _std_norm(x, eps=1e-6):
    return (x - x.mean())/max(x.std(), eps)

def _hwc_to_chw(img):
    if img.ndim < 3:
        return img
    return img.swapaxes(2, 1).swapaxes(1, 0)

def _chw_to_hwc(img):
    if img.ndim < 3:
        return img
    return img.swapaxes(0, 1).swapaxes(1, 2)

def _gray_to_rgb(img):
    """assumes img in shape channels, height, width"""
    if img.shape[0] == 3:
        return img
    return np.concatenate(3*(img, ), axis=0)

def _load(path):
    return io.imread(path)

def _get_x_path(uid, dset_path):
    path = os.path.join(dset_path, "stimuli", "{}.jpg".format(uid))
    return path

def _get_y_path(uid, dset_path):
    path = os.path.join(dset_path, "maps", "{}.png".format(uid))
    return path

def _load_x(uid, dset_path):
    path = _get_x_path(uid, dset_path)
    x = _load(path)
    if x.ndim < 3:
        x = np.dstack([x, x, x])
    x = _hwc_to_chw(x)
    return x

def _load_y(uid, dset_path):
    path = _get_y_path(uid, dset_path)
    y = _load(path)
    return y

def _load_xy(uid, dset_path):
    x = _load_x(uid, dset_path)
    y = _load_y(uid, dset_path)
    return x, y

def train_load(uid, dset_path):
    return _load_xy(uid, dset_path)

def infer_load(path):
    return _load(path)

def assay_load_y_pred(path):
    pass

def assay_load_y_true(path):
    pass

def _pre_proc_x(x, shape=None):
    """
    assumes x in format height, width, channels
    """
    x = _chw_to_hwc(x)
    #converting to LAB colorspace
    x = color.rgb2lab(x, illuminant="D65", observer="2")
    #resizing input
    if shape is not None:
        x = transf.resize(
            x, shape, preserve_range=True, mode="constant", order=1)
    #normalizing each channel per mean and std
    for i in range(x.shape[0]):
        x[i] = _std_norm(x[i])
    x = _hwc_to_chw(x)
    x = x.astype("float32")
    return x

def _pre_proc_y(y, shape=None):
    """
    assumes y in format channels, height
    """
    if shape is not None:
        y = transf.resize(
            y, shape, preserve_range=True, mode="constant", order=1)
    y = y.reshape((1, ) + y.shape)
    y = _unit_norm(y)
    y = y.astype("float32")
    return y

def _pre_proc_xy(xy, x_shape=None, y_shape=None):
    x, y = xy
    x = _pre_proc_x(x, x_shape)
    y = _pre_proc_y(y, y_shape)
    return x, y

def train_pre_proc(xy, x_shape=None, y_shape=None):
    return _pre_proc_xy(xy, x_shape, y_shape)

def infer_pre_proc(x):
    return _pre_proc_x(x)

def infer_save_y_pred(path, y_pred):
    """
    assumes y_pred comes in shape (1, h, w), in [0, 1] float dtype
    """
    #converting to uint8 image
    y_pred = y_pred.reshape(y_pred.shape[-2:])
    y_pred = y_pred.clip(0, 1)
    y_pred = (255*y_pred).astype("uint8")
    io.imsave(path, y_pred)
