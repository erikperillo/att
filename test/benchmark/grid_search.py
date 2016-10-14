#!/usr/bin/env python3

#grid_search.py
#executes benchmark.sh script for various saliency map parameters

import subprocess as sp
import shutil
import itertools
import time
import sys
import os

#script variables
#pyramid levels
pyr_lvls = [3]
#normalization scores types
norm_scores = ["cmdrssq"]
#color map weights
col_ws = [16, 32, 64]
#contrast map weights
cst_ws = [1]
#orientation map weights
ort_ws = [0]
#base directory where each directory will be stored
base_dir = "/home/erik/grid_search/refined_weight_anal2"
#benchmark script command
bm_cmd = "/home/erik/proj/att/test/benchmark/benchmark.sh"
bm_cmd_flags = ""
#resume whole grid search in csv files
resume = True
#metrics to use in resuming step
metrics = ["fp_auc_judd", "fp_nss", "cm_sim", "cm_cc"]
#resuming command
resume_cmd = "/home/erik/proj/att/test/benchmark/resume.sh"

def fmt_time(seconds):
    hours = int(seconds)//3600
    minutes = (int(seconds)%3600)//60
    seconds = seconds%60 

    return hours, minutes, seconds

def str_fmt_time(seconds):
    return "%.3dh%.2dm%.2ds" % fmt_time(seconds)

def params_tuple_to_dict(run_params):
    pyrlvls, ns, colw, cstw, ortw = run_params
    
    params_dict = locals()
    del params_dict["run_params"]

    return params_dict

def mk_dir_name(params_dict, val_sep="-", param_sep="_"):
    return param_sep.join(k + val_sep + str(v) for k, v in params_dict.items())

def mk_flags(params_dict):
    return " ".join("--%s %s" % (k, str(v)) for k, v in params_dict.items())

def ok_params(params_dict):
    colw = params_dict["colw"]
    cstw = params_dict["cstw"]
    ortw = params_dict["ortw"]

    if colw == 0 and cstw == 0 and ortw == 0:
        return False
    if colw == cstw and cstw == ortw and colw != 1:
        return False
    if colw == 0 and cstw == ortw and cstw != 1:
        return False
    if cstw == 0 and colw == ortw and colw != 1:
        return False
    if ortw == 0 and colw == cstw and colw != 1:
        return False

    return True

def run_bm(params_dict):
    out_dir = mk_dir_name(params_dict)
    flags = "%s %s" % (bm_cmd_flags, mk_flags(params_dict))

    out_dir = os.path.join(base_dir, out_dir)
    cmd = bm_cmd.split() + [out_dir] + [flags]

    ret = sp.run(cmd)

    return ret

def grid_search():
    params = itertools.product(pyr_lvls, norm_scores, col_ws, cst_ws, ort_ws)
    params = map(params_tuple_to_dict, params)
    params = filter(ok_params, params)

    total_time = 0
    for p in params:
        start_time = time.time()
        run_bm(p)
        partial_time = int(time.time() - start_time)

        print("[grid_search] run elapsed time: %s" % str_fmt_time(partial_time))
        total_time += partial_time
        print("[grid_search] total elapsed time: %s" % str_fmt_time(total_time))

    if resume:
        for m in metrics:
            print("[grid_search] resuming for metric '%s'..." % m)
            ret = sp.run(resume_cmd.split() + [base_dir, m],
                    stdout=sp.PIPE, stderr=sp.PIPE)

            metric_filepath = os.path.join(base_dir, m + ".csv")
            with open(metric_filepath, "w") as f:
                f.write(ret.stdout.decode())

            print("done. saved in '%s'" % metric_filepath)

    print("copying generator script to file '%s'..." %\
            os.path.join(base_dir, "gen_grid_search.py"))
    shutil.copyfile(__file__, os.path.join(base_dir, "gen_grid_search.py"))
    print("done.")

def main():
    if os.path.exists(base_dir):
        print("error: directory '%s' already exists" % base_dir)
        exit()

    grid_search()

if __name__ == "__main__":
    main()
