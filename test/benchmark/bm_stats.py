#!/usr/bin/env python3

import pandas as pd
import sys

def stats():
    if len(sys.argv) < 2:
        print("usage: stats <csv_file>")
        exit() 

    df = pd.read_csv(sys.argv[1])

    for col in df:
        try:
            data = df[col]
            print("{}: mean={:.6f}, std={:.6f}, max={:.6f}, min={:.6f}".\
                format(col, data.mean(), data.std(), data.max(), data.min()))
        except:
            print("could not get stats for col '{}'".format(col))
            continue

def main():
    stats()

if __name__ == "__main__":
    main()
