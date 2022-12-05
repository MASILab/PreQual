#!/bin/bash

L_file=/INPUTS/L.nii.gz
dwi_file=/INPUTS/dti.nii.gz
org_bvec_file=/INPUTS/bvec.txt
org_bval_file=/INPUTS/bval.txt
out_dir=/OUTPUTS
# num_threads=

echo $L_file $dwi_file $org_bval_file $org_bvec_file $out_dir
# Set absolute path variable for our custom executables
abs_path=/APPS/gradtensor # Change in container to /APPS/gradtensor

# Set path for executable
#export PATH=$PATH:$abs_path

# Set up venv
#source $abs_path/gradvenv/bin/activate

# # Prep inputs for gradtensor to b
# echo Preparing the inputs for gradnonlinearity correction
# python /nfs/masi/kanakap/projects/LR/scripts/PreQual/src/APPS/gradtensor/prep_inputs.py $org_bvec_file $org_bval_file $out_dir


# Run gradtensor to b #/usr/local/MATLAB/MATLAB_Runtime/v92 \
echo Computing bimages with gradnonlinearity tensor
$abs_path/run_apply_gradtensor_to_b.sh \
/usr/local/MATLAB/MATLAB_Runtime/v92 \
Limg_file $L_file \
refimg_file $dwi_file \
bval_file $org_bval_file \
bvec_file $org_bval_file \
out_dir $out_dir || { echo 'apply_gradtensor_to_b failed' ; return 1; }

# Run bimages to corrected signal
# echo Computing signal from bimages 
# python /nfs/masi/kanakap/projects/LR/scripts/PreQual/src/APPS/gradtensor/bimages_to_sig.py $dwi_file $org_bvec_file $org_bval_file $out_dir $num_threads

# Done
echo gradtensor.sh complete!
