#!/bin/bash

proj_path=/CODE/dtiQA_v7

source $proj_path/venv/bin/activate
python $proj_path/run_dtiQA.py $@
deactivate
