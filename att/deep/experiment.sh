#!/bin/bash

cfg_dirs="onemore twomore"

mv config config.bak.d

for dir in $cfg_dirs; do
    echo "in $dir"
    mv $dir config
    ./train.py
    mv config $dir
done

mv config.bak.d config
