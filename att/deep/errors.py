import theano
import theano.tensor as T
import lasagne

def norm_max(pred, tgt, alpha=1.1):
    return lasagne.objectives.squared_error(
        (pred/pred.max() - tgt)/(alpha - tgt))
