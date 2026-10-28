#!/bin/bash

L_file=$1
dwi_file=$2
org_bvec_file=$3
org_bval_file=$4
out_dir=$5
num_threads=$6

echo $L_file $dwi_file $org_bval_file $org_bvec_file $out_dir
# Set absolute path variable for our custom executables
abs_path=/APPS/gradtensor # Change in container to /APPS/gradtensor

# Set path for executable
export PATH=$PATH:$abs_path

# Set up venv
source $abs_path/gradvenv/bin/activate

# Compute the voxelwise corrected b-table from the gradient nonlinearity tensor
# (Python port of baxpr/gradtensor apply_gradtensor_to_b; no MATLAB runtime needed)
echo Computing bimages with gradnonlinearity tensor
python $abs_path/apply_gradtensor_to_b.py \
--Limg_file $L_file \
--refimg_file $dwi_file \
--bval_file $org_bval_file \
--bvec_file $org_bvec_file \
--out_dir $out_dir || { echo 'apply_gradtensor_to_b failed' ; exit 1; }

# Run bimages to corrected signal
echo Computing signal from bimages
python $abs_path/bimages_to_sig.py $dwi_file $org_bvec_file $org_bval_file $out_dir $num_threads || { echo 'bimages_to_sig failed' ; exit 1; }

# Done
echo gradtensor.sh complete!
