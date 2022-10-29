#!/bin/bash

L_file=$1
dwi_file=$2
org_bvec_file=$3
org_bval_file=$4
out_dir=$5
num_threads=$6

echo $L_file $dwi_file $org_bval_file $org_bvec_file $out_dir
# Set absolute path variable for our custom executables
#abs_path=/APPS/gradtensor # Change in container to /APPS/synb0

# Set path for executable
#export PATH=$PATH:$abs_path

# Set up venv
#source $abs_path/bin/activate
source /home/local/VANDERBILT/kanakap/py38-venv/bin/activate

# Prep inputs for gradtensor to b
echo Preparing the inputs for gradnonlinearity correction
python prep_inputs.py $org_bvec_file $org_bval_file $out_dir


# Run gradtensor to b #/usr/local/MATLAB/MATLAB_Runtime/v92 \
echo Computing bimages with gradnonlinearity tensor
/nfs/masi/kanakap/projects/LR/scripts/PreQual/src/APPS/gradtensor/run_apply_gradtensor_to_b.sh \
/home/local/VANDERBILT/kanakap/MATLAB_2017a_Runtime/v92 \
Limg_file $L_file \
refimg_file $dwi_file \
bval_file $out_dir/org.bval \
bvec_file $out_dir/org.bvec \
out_dir $out_dir || { echo 'apply_gradtensor_to_b failed' ; return 1; }

# Run bimages to corrected signal
echo Computing signal from bimages 
python bimages_to_sig.py $dwi_file $org_bvec_file $org_bval_file $out_dir $num_threads

# Done
echo gradtensor.sh complete!
