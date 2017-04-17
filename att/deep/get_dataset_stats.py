#!/usr/bin/env python3

import math
import glob
import random
import pickle
import numpy as np
import os
from skimage import io, color, img_as_float

DIR = "/home/erik/proj/ic/saliency_datasets/salicon/images/"

def main():
    stats = {
        "l": {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
        },
        "a": {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
        },
        "b": {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
        }
    }

    filepaths = glob.glob(os.path.join(DIR, "*"))
    random.shuffle(filepaths)
    counter = 0
    for fp in filepaths:
        print("[counter = {}] on image '{}'...".format(counter, fp))
        counter += 1

        img = io.imread(fp, as_grey=False)
        if len(img.shape) < 3:
            img = color.gray2rgb(img)
        img = color.rgb2lab(img)

        for i, c in zip(range(3), ("l", "a", "b")):
            channel = img[:, :, i]
            stats[c]["min"].append(channel.min())
            stats[c]["max"].append(channel.max())
            stats[c]["mean"].append(channel.mean())
            stats[c]["std"].append(channel.std())
            #for k, v in stats.items():
            #    print(k, "->", v)
            #print()

    for c in stats:
        n = len(stats[c]["min"])
        stats[c]["min"] = min(stats[c]["min"])
        stats[c]["max"] = max(stats[c]["max"])
        stats[c]["mean"] = sum(stats[c]["mean"])/n
        stats[c]["std"] = math.sqrt(sum(x**2 for x in stats[c]["std"])/n)

    for k, v in stats.items():
        print(k, "->", v)

    with open("lab_stats.csv", "w") as f:
        print("channel,min,max,mean,std", file=f)
        for k, v in stats.items():
            print("{},".format(k), end="", file=f)
            print(",".join(
                map(str, (v[s] for s in ("min", "max", "mean", "std")))),
                file=f)

    with open("lab_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    main()
