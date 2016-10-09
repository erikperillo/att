#!/usr/bin/env python3

import pandas as pd
import scipy.stats
import numpy as np
import sys

def ci(arr, conf=0.95):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    n = len(arr)
    mean = arr.mean()
    sem = scipy.stats.sem(arr)

    return scipy.stats.t.interval(conf, n-1, loc=mean, scale=sem)

def stats():
    if len(sys.argv) < 2:
        print("usage: stats <csv_file>")
        exit() 

    df = pd.read_csv(sys.argv[1])

    print("metric,mean,ci_min,ci_max,min,max")

    for col in df:
        try:
            data = df[col].values
            ci_min, ci_max = ci(data)
            print("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(
                col, data.mean(), ci_min, ci_max, data.min(), data.max()))
        except:
            print("could not get stats for col '%s'" % col, file=sys.stderr)
            continue

def main():
    stats()

if __name__ == "__main__":
    main()
