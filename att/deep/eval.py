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

import config.model as model
import config.eval as cfg

def std_norm(x):
    """Mean-std normalization."""
    return (x - T.mean(x))/T.std(x)

def unit_norm(x):
    """Unit normalization."""
    return (x - T.min(x))/(T.max(x) - T.min(x))

def cov(a, b):
    """Covariance."""
    return T.mean((a - T.mean(a))*(b - T.mean(b)))

def cc(pred, tgt):
    """Correlation Coefficient."""
    return cov(pred, tgt)/(T.std(pred)*T.std(tgt))

def r_squared(pred, tgt):
    """R-squared."""
    return T.square(coef_corr(pred, tgt))

def sim(pred, tgt):
    """Similarity."""
    return T.sum(T.minimum(pred/pred.sum(), tgt/tgt.sum()))

def mse(pred, tgt):
    """Mean-squared-error."""
    return lasagne.objectives.squared_error(pred, tgt).mean()

def norm_mse(pred, tgt, alpha):
    """Normalized mean-squared-error."""
    return T.square((pred/pred.max() - tgt)/(alpha - tgt)).mean()

def mae(pred, tgt):
    """Mean-absolute-error."""
    return T.mean(abs(pred - tgt))

EVAL_FUNCS = {
    "cc": cc,
    "sim": sim,
    "mse": mse,
    "mae": mae,
}

def main():
    #input
    inp = T.tensor4("inp")
    target = T.tensor4("target")

    #neural network model
    net_model = model.Model(inp, load_net_from=cfg.model_filepath)
    #making prediction function
    #prediction function
    x = unit_norm(net_model.test_pred)
    y = unit_norm(target)

    eval_f = theano.function([inp, target], 
        [EVAL_FUNCS[m](x, y) for m in cfg.metrics])

    values = {m: 0.0 for m in cfg.metrics}
    counter = 0

    print("evaluating...")
    for x, y in trloop.batches_gen_iter(cfg.filepaths, 1):
        vals = eval_f(x, y)
        for i, m in enumerate(cfg.metrics):
            values[m] += vals[i]
        counter += 1
        print("counter = %d      " % counter, end="\r", flush=True)

    for m in cfg.metrics:
        print("{}: {}".format(m, values[m]/max(counter, 1)))

if __name__ == "__main__":
    main()
