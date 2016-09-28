#!/usr/bin/env python3

import numpy as np

FILEPATH = "benchmark.txt"

STATS = [
    "Intersects with benchmark mask?",
    "Fraction in mask",
    "Fraction of model mask within benchmark mask",
    "Fraction of benchmark mask guessed by model"
]

def cvt(string):
    try:
        return float(string)
    except ValueError:
        return 1.0 if string == "True" else 0.

def get_stat(iterable, stat_str, delim=":"):
    pre_proc = lambda l: cvt(l.split(delim)[-1].strip())
    filter_f = lambda l: stat_str in l

    return np.array(list(map(pre_proc, filter(filter_f, iterable))), 
        dtype=float)

def main():
    with open(FILEPATH) as f:
        lines = [l for l in f]
        for stat in STATS:
            vals = get_stat(lines, stat)
            print("{}: mean = {:.3f}, std = {:.3f}"\
                .format(stat, vals.mean(), vals.std()))

if __name__ == "__main__":
    main()
