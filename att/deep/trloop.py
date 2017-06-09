import numpy as np
import util
from config import model
import config.train as cfg
import time
import theano

from collections import defaultdict

import threading
import queue

def _str_fmt_time(seconds):
    int_seconds = int(seconds)
    hours = int_seconds//3600
    minutes = (int_seconds%3600)//60
    seconds = int_seconds%60 + (seconds - int_seconds)
    return "%.2dh:%.2dm:%.3fs" % (hours, minutes, seconds)

def _silence(*args, **kwargs):
    pass

def batches_gen(X, y, batch_size, shuffle=False):
    n_samples = len(y)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, max(n_samples-batch_size+1, 1), batch_size):
        excerpt = indices[start_idx:start_idx+batch_size]
        yield X[excerpt], y[excerpt]

def _inf_gen():
    n = 0
    while True:
        yield n
        n += 1

def batches_gen_iter(filepaths, batch_size, shuffle=False, print_f=_silence):
    if shuffle:
        np.random.shuffle(filepaths)

    for i, fp in enumerate(filepaths):
        msg = "    [loading file '{}'...]".format(fp)
        print_f(msg, end="\r", flush=True)
        X, y = util.unpkl(fp)
        print_f(len(msg)*" ", end="\r")

        X = X.astype(cfg.x_dtype, casting="same_kind")
        X = X.reshape((X.shape[0],) + model.Model.INPUT_SHAPE)
        y = y.astype(cfg.y_dtype, casting="same_kind")
        y = y.reshape((y.shape[0],) + model.Model.OUTPUT_SHAPE)

        for batch_X, batch_y in batches_gen(X, y, batch_size, shuffle):
            yield i, (batch_X, batch_y)

def load_data(filepaths, q, stop, print_f=print):
    i = 0
    while i < len(filepaths):
        if stop.is_set():
            break

        if q.empty():
            msg = "    [loading file '{}'...]".format(filepaths[i])
            print_f(msg, end="\r", flush=True)
            data = util.unpkl(filepaths[i])
            q.put(data)
            print_f(len(msg)*" ", end="\r")
            i += 1

        time.sleep(0.1)

    stop.set()

def batches_gen_async(filepaths, batch_size, shuffle=False, print_f=_silence):
    if shuffle:
        np.random.shuffle(filepaths)

    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    data_loader = threading.Thread(target=load_data, args=(filepaths, q, stop))
    data_loader.start()

    try:
        for i in _inf_gen():
            X, y = q.get()
            X = X.astype(cfg.x_dtype, casting="same_kind")
            X = X.reshape((X.shape[0],) + model.Model.INPUT_SHAPE)
            y = y.astype(cfg.y_dtype, casting="same_kind")
            y = y.reshape((y.shape[0],) + model.Model.OUTPUT_SHAPE)

            for batch_X, batch_y in batches_gen(X, y, batch_size, shuffle):
                yield i, (batch_X, batch_y)

            if q.empty() and stop.is_set():
                break
    except:
        raise
    finally:
        stop.set()
        data_loader.join()

def _str_fmt_dct(dct):
    """Assumes dict mapping str to float."""
    return " | ".join("%s: %.4g" % (k, v) for k, v in dct.items())

def run_epoch(
    data, func,
    batch_size=1,
    info=_silence, warn=_silence,
    shuf_data=True, async_data_load=True):

    vals_sum = defaultdict(float)
    n_its = 0
    batch_gen = batches_gen_async if async_data_load else batches_gen_iter

    for i, (bi, xy) in enumerate(batch_gen(data, batch_size, shuf_data, info)):
        vals = func(*xy)

        for k, v in vals.items():
            vals_sum[k] += v

        n_its += 1
        info("    [batch %d, data part %d/%d]" % (i, bi+1, len(data)),
            _str_fmt_dct(vals), 32*" ", end="\r")

    vals_mean = {k: v/n_its for k, v in vals_sum.items()}
    return vals_mean

def _str(obj):
    return str(obj) if obj is not None else "?"

def train_loop(
    tr_set, tr_f,
    n_epochs=10, batch_size=1,
    val_set=None, val_f=None, val_f_val_tol=None,
    async_data_load=True,
    verbose=2, print_f=print):
    """
    General Training loop.
    Parameters:
    tr_set: [str]
        list of str Training set containing X and y.
    tr_f : callable
        Training function giving a dict in format {"name": val, ...}
    n_epochs : int or None
        Number of epochs. If None, is infinite.
    batch_size : int
        Batch size.
    val_set: None or [str]
        Validation set containing X and y.
    val_f : callable or None
        Validation function giving a dict in format {"name": val, ...}
    verbose : int
        Prints nothing if 0, only warnings if 1 and everything if >= 2.
    """

    #info/warning functions
    info = print_f if verbose >= 2 else _silence
    warn = print_f if verbose >= 1 else _silence
    epoch_info = print if verbose >= 2 else _silence

    #checking if use validation
    validation = val_set is not None and val_f is not None

    if async_data_load:
        info("[info] using async loading of data")

    #initial values for some variables
    start_time = time.time()

    #main loop
    info("[info] starting training loop...")
    for epoch in _inf_gen():
        if n_epochs is not None and epoch >= n_epochs:
            warn("\nWARNING: maximum number of epochs reached")
            end_reason = "n_epochs"
            return end_reason

        info("epoch %d/%s:" % (epoch+1, _str(n_epochs)))

        tr_vals = run_epoch(tr_set, tr_f, batch_size,
            async_data_load=async_data_load, info=epoch_info, warn=warn)
        info("    train values:", _str_fmt_dct(tr_vals), 32*" ")

        if validation:
            val_vals = run_epoch(val_set, val_f, batch_size,
                async_data_load=async_data_load, info=epoch_info, warn=warn)
            info("    val values:", _str_fmt_dct(val_vals), 32*" ")

        info("    time so far:", _str_fmt_time(time.time() - start_time))
