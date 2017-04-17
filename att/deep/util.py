"""
General-purpose utilities to be used in project.
"""

import os
import sys
import datetime
import subprocess as sp
import numpy as np
from PIL import Image
import pickle
import gzip
import bz2

def git_hash():
    """
    Gets git commit hash of project.
    """
    try:
        hsh = sp.getoutput("git rev-parse HEAD").strip("\n")
    except:
        hsh = ""

    return hsh

def time_str():
    """
    Returns string-formatted local time in format hours:minutes:seconds.
    """
    return "".join(str(datetime.datetime.now().time()).split(".")[0])

def date_str():
    """
    Returns string-formatted local date in format year-month-day.
    """
    return str(datetime.datetime.now().date())

def uniq_filename(dir_path, pattern, ext=""):
    """
    Returns an unique filename in directory path that starts with pattern.
    """
    dir_path = os.path.abspath(dir_path)

    files = [f for f in os.listdir(dir_path) if \
             os.path.exists(os.path.join(dir_path, f))]
    num = len([f for f in files if f.startswith(pattern)])

    filename = pattern + (("_" + str(num)) if num else "") + ext

    return filename

def uniq_filepath(dir_path, pattern, ext=""):
    """
    Returns an unique filepath in directory path that starts with pattern.
    """
    dir_path = os.path.abspath(dir_path)
    return os.path.join(dir_path, uniq_filename(dir_path, pattern, ext))

def get_ext(filepath, sep="."):
    """
    Gets extension of file given a filepath.
    """
    filename = os.path.basename(filepath.rstrip("/"))
    if not sep in filename:
        return ""
    return filename.split(sep)[-1]

def open_mp(filepath, *args, **kwargs):
    """
    Opens file with multiple protocols (gzip, bzip2...).
    """
    ext = get_ext(filepath)
    if ext == "gz":
        open_f = gzip.open
    elif ext == "bz2":
        open_f = bz2.open
    else:
        open_f = open
    return open_f(filepath, *args, **kwargs)

def pkl(obj, filepath, protocol=4):
    """
    Saves object using pickle.
    """
    with open_mp(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)

def unpkl(filepath):
    """
    Loads object using pickle.
    """
    with open_mp(filepath, "rb") as f:
        return pickle.load(f)

def load_image(filepath):
    """
    Loads image in RGB format from filepath.
    """
    img = Image.open(filepath).convert("RGB")
    img = np.asarray(img)
    return img

def txt_to_dict(info_file_path, sep=":"):
    """
    Gets a path to a text file with lines in format key: value and returns dict.
    """
    with open(info_file_path) as f:
        lines = [l.strip() for l in f]
    pairs = filter(lambda x: x, lines)
    pairs = (l.split(sep) for l in pairs)
    pairs = ((k.strip(), v.strip()) for k, v in pairs)
    return dict(pairs)

class Tee:
    """
    Broadcasts print message through list of open files.
    """
    def __init__(self, files):
        self.files = files

    def print(self, *args, **kwargs):
        for f in self.files:
            kwargs["file"] = f
            print(*args, **kwargs)

    def __del__(self):
        for f in self.files:
            if f != sys.stdout and f != sys.stderr:
                f.close()
