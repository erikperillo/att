#!/usr/bin/env python3

#show.py
#plots metrics values for different saliency map parameter configurations

import numpy as np
import os.path
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data_filepaths = [
    "auc_judd.csv",
    "nss.csv",
    "cc.csv",
    "sim.csv",
]

colors = list("bgmykwcr")

def label_fmt(label):
    label = os.path.basename(label)
    #label = [x for x in label.split("_") if "colw" in x][0]
    return label

def path_to_title(filepath):
    return ".".join(os.path.basename(filepath).split(".")[:-1])

#assumes .csv file in format config,metric_mean,metric_ci_min,metric_ci_max
def line_plot(filepath, space=2):
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

#assumes .csv file in format config,metric_mean,metric_ci_min,metric_ci_max
def _bar_plot(df, ax=None, indexes=None, width=0.35, col="r"):
    y = df[1].values
    y_lower_err = (df[1] - df[2]).values
    y_upper_err = (df[3] - df[1]).values

    if indexes is None:
        indexes = np.arange(len(y))
    if ax is None:
        fig, ax = plt.subplots()

    rects = ax.bar(indexes, y, width, color=col,
            yerr=[y_lower_err, y_upper_err],
            error_kw=dict(ecolor="gray", lw=2, capsize=3, capthick=2))

    return rects

def bar_plot(filepaths):
    fig, ax = plt.subplots()
    width = 0.80
    y_lim_ratio = 1.1
    min_y = 0.0
    max_y = 1.0
    labels = None
    indexes = None
    rects = []
    #number of data files
    n = len(data_filepaths)

    for i, fp in enumerate(filepaths):
        #reading dataframe from csv file
        df = pd.read_csv(fp, header=None)

        #updating y limits
        min_y = min(min_y, min(df[1]))
        max_y = max(max_y, max(df[1]))
        ax.set_ylim([y_lim_ratio*min_y, y_lim_ratio*max_y])

        if indexes is None:
            indexes = 1 + n*np.arange(len(df))
        if labels is None:
            labels = [label_fmt(l) for l in df[0].values]

        ret = _bar_plot(df, ax, indexes + i*width, width, colors[i%len(colors)])
        rects.append(ret)

    ax.set_xticks(1 +indexes + n*width/2)
    #ax.set_xticklabels(labels, rotation="-45", fontsize="large")
    ax.set_xticklabels(labels, rotation="horizontal", fontsize="large")
    ax.legend([r[0] for r in rects], [path_to_title(fp) for fp in filepaths])
    plt.show()

def main():
    bar_plot(data_filepaths)

if __name__ == "__main__":
    main()
