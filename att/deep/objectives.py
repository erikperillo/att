#!/usr/bin/env python3

"""
Objectives.
"""

import sys
import theano
import theano.tensor as T
import lasagne

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
