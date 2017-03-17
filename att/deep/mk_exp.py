#!/usr/bin/env python3

import util
import shutil
import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(SRC_DIR, "experiments")
DATA_PREPROC_FILE = os.path.join(SRC_DIR, "datapreproc.py")
TRAIN_FILE = os.path.join(SRC_DIR, "train.py")
MODEL_FILE = os.path.join(SRC_DIR, "model.py")
TRAIN_LOOP_FILE = os.path.join(SRC_DIR, "trloop.py")
UTIL_FILE = os.path.join(SRC_DIR, "util.py")

def main():
    dir_path = util.uniq_filepath(BASE_DIR, "exp")
    os.makedirs(dir_path)

    files_to_copy = [
        DATA_PREPROC_FILE,
        TRAIN_FILE,
        MODEL_FILE,
        TRAIN_LOOP_FILE,
        UTIL_FILE
    ]
    for fp in files_to_copy:
        filename = os.path.basename(fp)
        shutil.copy(fp, os.path.join(dir_path, filename))

    os.makedirs(os.path.join(dir_path, "data"))

    print("experiment env created in {}".format(dir_path))

if __name__ == "__main__":
    main()
