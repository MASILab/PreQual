#!/bin/bash

proj_path=/CODE/dtiQA_v7

# matplotlib renders to files only; never let it pick an interactive backend (xvfb sets DISPLAY)
export MPLBACKEND=Agg

source $proj_path/venv/bin/activate
python $proj_path/run_dtiQA.py $@
deactivate
