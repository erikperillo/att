#!/bin/bash

cfg_dirs=""

mv config config.bak.d

for dir in $cfg_dirs; do
    mv $dir config
    ./train.py
    mv config $dir
done

mv config.bak.d config
