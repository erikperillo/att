#!/usr/bin/env python3

#show.py
#plots metrics values for different saliency map parameter configurations

import os.path
import pandas as pd
import matplotlib.pyplot as plt

data_filepaths = [
    "~/grid_search/auc.csv",
    "~/grid_search/nss.csv",
    "~/grid_search/cc.csv",
    "~/grid_search/sim.csv",
]

def label_fmt(label):
    return label

def path_to_title(filepath):
    return ".".join(os.path.basename(filepath).split(".")[:-1])

#assumes .csv file in format config,metric_mean,metric_ci_min,metric_ci_max
def draw(filepath, space=2):
    df = pd.read_csv(filepath, header=None)

    y = df[1].values
    x = [space*i for i in range(len(y))]
    y_lower_err = (df[1] - df[2]).values
    y_upper_err = (df[3] - df[1]).values
    labels = [label_fmt(l) for l in df[0].values]

    plt.figure()
    plt.errorbar(y=y, x=x, yerr=[y_lower_err, y_upper_err])
    plt.xticks(x, labels, rotation="vertical")
    plt.title(path_to_title(filepath))

def main():
    for fp in data_filepaths:
        draw(fp)

    plt.show()

if __name__ == "__main__":
    main()
