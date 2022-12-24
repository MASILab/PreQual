#!/bin/bash

b0_d_file=$1
t1_file=$2
OUTPUTS=$3
mni_1mm_atlas_fname=$4 # Can either be with (default) or without skull, depending on the input T1

# Set absolute path variable for our custom executables
abs_path=/APPS/synb0 # Change in container to /APPS/synb0

# Set path for executable
export PATH=$PATH:$abs_path

## Set up freesurfer
#export FREESURFER_HOME=$abs_path/freesurfer # this line is executed in singularity environment set up
export FS_LICENSE=/APPS/freesurfer/license.txt
source $FREESURFER_HOME/SetUpFreeSurfer.sh # can't do this with singularity env set up because singularity uses sh and source and the setup file need bash, so need to do it before we need freesurfer, which for dtiQA is just here

# Set up pytorch
source $abs_path/pytorch/bin/activate

# Prepare input
prepare_input.sh $b0_d_file $t1_file $abs_path/atlases/$mni_1mm_atlas_fname $abs_path/atlases/mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz $OUTPUTS

# Run inference
NUM_FOLDS=5
for i in $(seq 1 $NUM_FOLDS);
  do echo Performing inference on FOLD: "$i"
  python3.6 $abs_path/inference.py $OUTPUTS/T1_norm_lin_atlas_2_5.nii.gz $OUTPUTS/b0_d_lin_atlas_2_5.nii.gz $OUTPUTS/b0_u_lin_atlas_2_5_FOLD_"$i".nii.gz $abs_path/dual_channel_unet/num_fold_"$i"_total_folds_"$NUM_FOLDS"_seed_1_num_epochs_100_lr_0.0001_betas_\(0.9\,\ 0.999\)_weight_decay_1e-05_num_epoch_*.pth
done

# Take mean
echo Taking ensemble average
fslmerge -t $OUTPUTS/b0_u_lin_atlas_2_5_merged.nii.gz $OUTPUTS/b0_u_lin_atlas_2_5_FOLD_*.nii.gz
fslmaths $OUTPUTS/b0_u_lin_atlas_2_5_merged.nii.gz -Tmean $OUTPUTS/b0_u_lin_atlas_2_5.nii.gz

# Apply inverse xform to undistorted b0
echo Applying inverse xform to undistorted b0
antsApplyTransforms -d 3 -i $OUTPUTS/b0_u_lin_atlas_2_5.nii.gz -r $b0_d_file -n BSpline -t [$OUTPUTS/epi_reg_d_ANTS.txt,1] -t [$OUTPUTS/ANTS0GenericAffine.mat,1] -o $OUTPUTS/b0_u.nii.gz

# Done
echo synb0.sh complete!

