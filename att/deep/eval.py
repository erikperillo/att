#!/usr/bin/env python3

"""
Evaluation script.
"""

import os
import sys
import theano
import theano.tensor as T
import lasagne

import trloop
import objectives

import config.model as model
import config.eval as cfg

EVAL_FUNCS = {
    "cc": objectives.cc,
    "sim": objectives.sim,
    "mse": objectives.mse,
    "mae": objectives.mae,
}

def unit_norm(x):
    return (x - T.min(x))/(T.max(x) - T.min(x))

def main():
    #input
    inp = T.tensor4("inp")
    target = T.tensor4("target")

    #neural network model
    net_model = model.Model(inp, target, load_net_from=cfg.model_filepath)
    #making prediction function
    #prediction function
    x = unit_norm(net_model.test_pred)
    y = unit_norm(target)

    print(cfg.filepaths)

    eval_f = theano.function([inp, target],
        [EVAL_FUNCS[m](x, y) for m in cfg.metrics])

    values = {m: 0.0 for m in cfg.metrics}
    counter = 0

    print("evaluating...")
    for i, (x, y) in trloop.batches_gen_iter(cfg.filepaths, 1):
        vals = eval_f(x, y)
        for i, m in enumerate(cfg.metrics):
            values[m] += vals[i]
        counter += 1
        print("counter = %d      " % counter, end="\r", flush=True)

    for m in cfg.metrics:
        print("{}: {}".format(m, values[m]/max(counter, 1)))

if __name__ == "__main__":
    main()
