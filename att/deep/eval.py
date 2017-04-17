#!/usr/bin/env python3

import oarg
import os
import sys
import numpy as np
import theano.tensor as T
from skimage import color, transform
import theano
import pylab

import config.model as model
import predict
import datapreproc
import util
sys.path.append(os.path.join("..", "..", "test", "benchmark"))
import metrics

def main():
    dataset_dir = oarg.Oarg("-s --dataset-dir", "", "path to dataset dir", 0)
    dataset_name = oarg.Oarg("-d --dataset-name", "", "dataset name")
    hlp = oarg.Oarg("-h --help", False, "this help message")

    oarg.parse()

    if hlp.val:
        oarg.describe_args("options:", True)
        exit()

    if not dataset_dir.val:
        print("error: must specify dataset dir (use -h for info)")
        eixt()

    if not os.path.isdir(dataset_dir.val):
        print("error: '{}' is not a dir".format(dataset_dir.val))
        exit()

    #input
    inp = T.tensor4("inp")
    #neural network model
    net_model = model.Model(inp, load_net_from=predict.cfg.model_filepath)
    #making prediction function
    #prediction function
    pred_f = theano.function([inp], net_model.test_pred)

    for fp in datapreproc.get_stimuli_paths(dataset_dir.val, dataset_name.val):
        print("in '{}'...".format(fp))

        print("\tloading stimulus...")
        stimulus = predict.load_img(fp)
        print("\tpreprocessing stimulus...")
        pre_proc_stimulus = predict.img_pre_proc(stimulus)
        print("\tpredicting...")
        pred, pred_time = predict.predict(pre_proc_stimulus, pred_f)
        print("\tdone predicting. took %f seconds" % pred_time)

        fixmap_fp = datapreproc.get_ground_truth_path(fp,
            dataset_dir.val, dataset_name.val)
        print("\tloading fixmap from {}...".format(fixmap_fp))
        fixmap = util.load_image(fixmap_fp)
        fixmap = transform.resize(fixmap, model.Model.INPUT_SHAPE[1:])
        fixmap = color.rgb2gray(fixmap)
        fixmap = datapreproc.unit_normalize(fixmap)

        #if pylab_imported and cfg.show_images:
        if True:
            print("\tdisplaying image...")
            pylab.subplot(1, 3, 1)
            pylab.axis("off")
            pylab.imshow(stimulus)
            pylab.subplot(1, 3, 2)
            pylab.gray()
            pylab.axis("off")
            pylab.imshow(color.gray2rgb(pred))
            pylab.subplot(1, 3, 3)
            pylab.gray()
            pylab.axis("off")
            pylab.imshow(color.gray2rgb(fixmap))
            pylab.show()

        print("cc:", metrics._cc(pred, fixmap))
        print("sim:", metrics._sim(pred, fixmap))
        print("mae:", metrics._mae(pred, fixmap))

if __name__ == "__main__":
    main()
