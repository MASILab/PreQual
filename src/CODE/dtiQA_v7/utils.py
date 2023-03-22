# PreQual: Utilities
# Leon Cai, Qi Yang, and Praitayini Kanakaraj
# MASI Lab
# Vanderbilt University

# Set Up

import sys
import os
import subprocess

import numpy as np
import nibabel as nib
from skimage import measure
from scipy.optimize import fmin
import matplotlib.pyplot as plt

from vars import SHARED_VARS

# Class Definitions: dtiQA Error

class DTIQAError(Exception):
    pass

# Function Definitions: General File/NIFTI Management and Command Line Interface

def run_cmd(cmd):

    print('RUNNING: {}'.format(cmd))
    subprocess.check_call(cmd, shell=True)

def copy_file(in_file, out_file):

    cp_cmd = 'cp {} {}'.format(in_file, out_file)
    run_cmd(cp_cmd)

def move_file(in_file, out_file):

    mv_cmd = 'mv {} {}'.format(in_file, out_file)
    run_cmd(mv_cmd)

def rename_file(in_file, out_file):

    move_file(in_file, out_file)

    return out_file

def remove_dir(in_dir):

    rm_cmd = 'rm -r {}'.format(in_dir)
    run_cmd(rm_cmd)

def remove_file(in_file):

    rm_cmd = 'rm {}'.format(in_file)
    run_cmd(rm_cmd)

def make_dir(parent_dir, child_dir):

    new_dir = os.path.join(parent_dir, child_dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def get_prefix(file_path, file_ext='nii'):

    return os.path.split(file_path)[-1].split('.{}'.format(file_ext))[0]

def load_nii(nii_file, dtype='', ndim=-1):

    nii = nib.load(nii_file)
    img = nii.get_data()

    if not dtype == '':
        img = img.astype(dtype)
    if len(img.shape) < 3 or len(img.shape) > 4:
        raise DTIQAError('CANNOT LOAD NIFTI IMAGES THAT ARE NOT 3D OR 4D. REQUESTED IMAGE TO LOAD IS {}D.'.format(len(img.shape)))
    if ndim == 3 or ndim == 4:
        if ndim > len(img.shape): # ndim = 4, img = 3
            img = np.expand_dims(img, axis=3)
        elif ndim < len(img.shape): # ndim = 3, img = 4
            if img.shape[-1] == 1:
                img = img[..., 0]
            else:
                raise DTIQAError('CANNOT LOAD NIFTI IMAGE WITH FEWER DIMENSIONS THAT IT ALREADY HAS. REQUESTED {} DIMS, HAS {} DIMS'.format(ndim, len(img.shape)))

    img = np.array(img)
    aff = nii.affine
    hdr = nii.get_header()

    return img, aff, hdr

def save_nii(img, aff, nii_file, dtype='', ndim=-1):

    if not dtype == '':
        img = img.astype(dtype)
    if len(img.shape) < 3 or len(img.shape) > 4:
        raise DTIQAError('CANNOT SAVE NIFTI IMAGES THAT ARE NOT 3D OR 4D. REQUESTED IMAGE TO SAVE IS {}D.'.format(len(img.shape)))
    if ndim == 3 or ndim == 4:
        if ndim > len(img.shape): # ndim = 4, img = 3
            img = np.expand_dims(img, axis=3)
        elif ndim < len(img.shape): # ndim = 3, img = 4
            if img.shape[-1] == 1:
                img = img[..., 0]
            else:
                raise DTIQAError('CANNOT SAVE NIFTI IMAGE WITH FEWER DIMENSIONS THAT IT ALREADY HAS. REQUESTED {} DIMS, HAS {} DIMS'.format(ndim, len(img.shape)))

    nii = nib.Nifti1Image(img, aff)
    nib.save(nii, nii_file)

def load_txt(txt_file, txt_type=''): 

    txt_data = np.loadtxt(txt_file)
    if txt_type == 'bvals':
        if len(txt_data.shape) == 0:
            txt_data = np.array([txt_data])
    elif txt_type == 'bvecs':
        if len(txt_data.shape) == 1:
            txt_data = np.expand_dims(txt_data, axis=1)
    return txt_data

def save_txt(txt_data, txt_file):

    if len(txt_data.shape) > 2:
        raise DTIQAError('DATA MUST BE A NUMBER OR A 2D ARRAY TO BE SAVED. CURRENT DATA HAS DIMENSION {}.'.format(len(txt_data.shape)))
    if len(txt_data.shape) == 1: # data needs to be in a 2D array to be written (or a number) since bvals need to be written in rows for FSL
        txt_data = np.array([txt_data]) # 0D (i.e. number) or 2D data all fine, concern when data is 1D, it needs to be made 2D.
    np.savetxt(txt_file, txt_data, fmt='%1.7f', delimiter=' ', newline='\n')

def write_str(str_data, str_file):

    with open(str_file, 'w') as str_fobj:
        str_fobj.write(str_data)

def nii_sform(nii_file, sform_dir):

    nii_prefix = get_prefix(nii_file)

    nii = nib.load(nii_file)
    _, scode = nii.get_sform(coded=True)
    _, qcode = nii.get_qform(coded=True)

    if scode == 0 and qcode == 0:
        best_aff = 'FALL-BACK'
        txfm_warning_str = 'Both the sform and qform codes for {} were 0. The NIFTI file was resaved with the fall-back affine.'.format(nii_prefix)
    elif scode == 0 and qcode != 0:
        best_aff = 'Q-FORM'
        txfm_warning_str = 'The sform code for {} was 0. The NIFTI file was resaved with the qform affine.'.format(nii_prefix)
    elif scode != 0:
        best_aff = 'S-FORM'
        if qcode == 0:
            txfm_warning_str = ''
        elif qcode != 0:
            txfm_warning_str = 'Both the sform and qform codes for {} were non-zero. The NIFTI file was resaved with the sform affine overriding the qform.'.format(nii_prefix)
    
    print('PER NIBABEL STANDARDS, THE BEST AFFINE FOR {} IS THE {} AFFINE. SAVING THIS TRANSFORM INTO BOTH THE SFORM (CODE = 2) AND QFORM (CODE = 0) FIELDS.'.format(nii_prefix, best_aff))

    nii_sform_file = os.path.join(sform_dir, '{}_sform.nii.gz'.format(nii_prefix))
    nii_sform = nib.Nifti1Image(nii.get_data(), nii.affine)
    nib.save(nii_sform, nii_sform_file)

    return nii_sform_file, txfm_warning_str

# Function Definitions: Pipeline I/O

def load_config(in_dir):

    config_file = os.path.join(in_dir, 'dtiQA_config.csv')
    config_mat = np.genfromtxt(config_file, delimiter=',', dtype=np.str)
    if len(config_mat.shape) == 1:
        config_mat = np.expand_dims(config_mat, axis=0)
    dwi_prefixes = config_mat[:, 0]
    pe_dirs = list(config_mat[:, 1])
    readout_times = list(config_mat[:, 2].astype('float'))

    dwi_files = []
    bvals_files = []
    bvecs_files = []
    for dwi_prefix in dwi_prefixes:
        dwi_files.append(os.path.join(in_dir, '{}.nii.gz'.format(dwi_prefix)))
        bvals_files.append(os.path.join(in_dir, '{}.bval'.format(dwi_prefix)))
        bvecs_files.append(os.path.join(in_dir, '{}.bvec'.format(dwi_prefix)))

    return dwi_files, bvals_files, bvecs_files, pe_dirs, readout_times

# Function Definitions: DWI Manipulation

def bvals_threshold(bvals_file, threshold, threshold_dir):

    bvals_prefix = get_prefix(bvals_file, file_ext='bval')
    
    print('THRESHOLDING BVALS FOR {}...'.format(bvals_prefix))

    bvals = load_txt(bvals_file, txt_type='bvals')

    threshold = float(threshold)

    for i in range(0, len(bvals)):
        print('{}'.format(bvals[i]), end=' ')
        if bvals[i] < threshold:
            bvals[i] = 0
            print('--> {}'.format(bvals[i]), end=' ')
        print('\n', end='')

    bvals_thresholded_file = os.path.join(threshold_dir, '{}.bval'.format(bvals_prefix))
    save_txt(bvals, bvals_thresholded_file)

    print('FINISHED THRESHOLDING BVALS FOR {}'.format(bvals_prefix))

    return bvals_thresholded_file

def dwi_check(dwi_file, bvals_file, bvecs_file, check_dir):

    dwi_prefix = get_prefix(dwi_file)

    print('CHECKING BVAL/BVECS FOR {}...'.format(dwi_prefix))

    # Load data and calculate bvec magnitudes
    
    dwi_img, dwi_aff, _ = load_nii(dwi_file, ndim=4)
    bvals = load_txt(bvals_file, txt_type='bvals')
    bvecs = load_txt(bvecs_file, txt_type='bvecs')
    bvec_mags = np.sqrt(np.sum(np.square(bvecs), 0))
    
    # Check that the number of bvecs, bvals, and volumes are all the same

    if (not dwi_img.shape[3] == len(bvec_mags)) or (not dwi_img.shape[3] == len(bvals)) or (not len(bvals) == len(bvec_mags)):
        raise DTIQAError('INPUT {} HAS AN INCONSISTENT NUMBER OF BVALS, BVECS, AND/OR VOLUMES. PLEASE CHECK INPUTS.'.format(dwi_prefix))

    # Find unsupported bvec/bval combos

    keep_vols = []
    for i in range(len(bvec_mags)):
        keep_vol = True
        if bvec_mags[i] > 1.2 or (bvec_mags[i] < 0.8 and bvec_mags[i] > 0.01): # bvecs must be unit normalized or zero
            keep_vol = False
            print('DETECTED NON-UNIT NORMALIZED BVEC! THIS COULD INDICATE A TRACE VOLUME AND MAY CAUSE PIPELINE FAILURE. VOLUME FLAGGED FOR REMOVAL.')
        if bvals[i] > 0 and bvec_mags[i] <= 0.01: # If bvecs are zero, bval must be 0
            keep_vol = False
            print('DETECTED BVALUE > 0 WITH CORRESPONDING BVEC OF ZERO. THIS COULD INDICATE AN ADC VOLUME AND MAY CAUSE EDDY TO BE UNABLE TO CONVERGE AND PIPELINE FAILURE. VOLUME FLAGGED FOR REMOVAL.')
        keep_vols.append(keep_vol)
    keep_vols = np.array(keep_vols)

    # Remove those volumes and save
    
    dwi_checked_file = os.path.join(check_dir, '{}_checked.nii.gz'.format(dwi_prefix))
    if dwi_img.shape[3] == 1:
        if keep_vols:
            dwi_checked_img = dwi_img
        else:
            raise DTIQAError('INPUT {} IMAGE CONSISTS OF ONLY ONE VOLUME AND IT HAS BAD BVAL/BVEC COMBO! CANNOT REMOVE VOLUME WITHOUT REMOVING INPUT. PLEASE CHECK INPUTS.'.format(dwi_prefix))
    else:
        dwi_checked_img = dwi_img[:, :, :, keep_vols]
    save_nii(dwi_checked_img, dwi_aff, dwi_checked_file, ndim=4)

    bvals_checked_file = os.path.join(check_dir, '{}_checked.bval'.format(dwi_prefix))
    bvals_checked = bvals[keep_vols]
    save_txt(bvals_checked, bvals_checked_file)
    
    bvecs_checked_file = os.path.join(check_dir, '{}_checked.bvec'.format(dwi_prefix))
    bvecs_checked = bvecs[:, keep_vols]
    save_txt(bvecs_checked, bvecs_checked_file)

    # Document warnings

    num_bad_vols = np.sum(np.logical_not(keep_vols))
    if num_bad_vols > 0:
        dwi_check_warning_str = '{} problematic b-value / b-vector combinations detected in {}. The corresponding volumes were removed. Please verify all b-vectors are either unit normalized or zero and that they are only zero when the corresponding b value is also zero.'.format(num_bad_vols, dwi_prefix)
    else:
        print('NO BAD BVECS DETECTED!')
        dwi_check_warning_str = ''

    print('FINISHED CHECKING BVAL/BVECS FOR {}'.format(dwi_prefix))

    return dwi_checked_file, bvals_checked_file, bvecs_checked_file, dwi_check_warning_str

def dwi_denoise(dwi_file, denoised_dir):

    # Note: MRTrix3 build RC3.0 (newest at the time of writing this) prefs qform over sform across the board. Nibabel both prefs sform. I think FSL just saves both.
    # Be sure to put a file /etc/mrtrix.conf with the key: value pair "NIfTIUseSform: 1" to use the sform when both a qform and sform exist
    #
    # Edit: As of May 2022, we go through and only use the sform across the board. See nii_sform().

    dwi_prefix = get_prefix(dwi_file, file_ext='nii')

    print('DENOISING {}...'.format(dwi_prefix))

    dwi_denoised_file = os.path.join(denoised_dir, '{}_denoised.nii.gz'.format(dwi_prefix))
    dwi_noise_file = os.path.join(denoised_dir, '{}_noise.nii.gz'.format(dwi_prefix))

    dwi_img, _, _ = load_nii(dwi_file, ndim=4)
    if dwi_img.shape[3] > 1:
        denoise_cmd = 'dwidenoise {} {} -noise {} -force -nthreads {}'.format(dwi_file, dwi_denoised_file, dwi_noise_file, SHARED_VARS.NUM_THREADS-1)
        run_cmd(denoise_cmd)
        denoise_warning_str = ''
    else:
        print('3D VOLUME SUPPLIED, DENOISING REQUIRES 4D VOLUME. COPYING FILE INSTEAD AND LOGGING WARNING.')
        copy_file(dwi_file, dwi_denoised_file)
        denoise_warning_str = '{} was not a 4D image and thus was not denoised.'.format(dwi_prefix)

    print('FINISHED DENOISING {}'.format(dwi_prefix))

    return dwi_denoised_file, dwi_noise_file, denoise_warning_str

def dwi_rician(dwi_file, noise_file, rician_dir):

    temp_dir = make_dir(rician_dir, 'TEMP')

    dwi_prefix = get_prefix(dwi_file, file_ext='nii')

    print('REMOVING RICIAN BIAS FROM {}...'.format(dwi_prefix))

    if noise_file == '':
        print('NO NOISE VOLUME PROVIDED, RUNNING MP-PCA DENOISING TO CALCULATE, BUT MP-PCA DENOISING NOT APPLIED.')
        _, noise_file, _ = dwi_denoise(dwi_file, temp_dir)

    # Set NaNs and Infs in noise to zero so magnitude isn't affected

    noise_prefix = get_prefix(noise_file, file_ext='nii')
    clean_noise_file = os.path.join(temp_dir, '{}_clean.nii.gz'.format(noise_prefix))
    clean_cmd = 'mrcalc {} -finite {} 0 -if {} -force -nthreads {}'.format(noise_file, noise_file, clean_noise_file, SHARED_VARS.NUM_THREADS-1)
    run_cmd(clean_cmd)

    # Run Method of Moments: I' = sqrt(I^2 - n^2), I = image, n = noise

    dwi_raw_rician_file = os.path.join(temp_dir, '{}_raw_rician.nii.gz'.format(dwi_prefix))
    rician_cmd = 'mrcalc {} 2 -pow {} 2 -pow -sub -abs -sqrt {} -force -nthreads {}'.format(dwi_file, clean_noise_file, dwi_raw_rician_file, SHARED_VARS.NUM_THREADS-1)
    run_cmd(rician_cmd)

    # Set non-finite/imaginary/NaNs in corrected output to NaN
    
    dwi_rician_file = os.path.join(rician_dir, '{}_rician.nii.gz'.format(dwi_prefix))
    clean_cmd = 'mrcalc {} -finite {} NaN -if {} -force -nthreads {}'.format(dwi_raw_rician_file, dwi_raw_rician_file, dwi_rician_file, SHARED_VARS.NUM_THREADS-1)
    run_cmd(clean_cmd)

    # Finish Up

    rician_warning_str = ''

    remove_dir(temp_dir)

    print('FINISHED RICIAN CORRECTION FOR {}'.format(dwi_prefix))

    return dwi_rician_file, rician_warning_str

def dwi_degibbs(dwi_file, degibbs_dir):

    dwi_prefix = get_prefix(dwi_file, file_ext='nii')

    print('REMOVING GIBBS ARTIFACTS FROM {}...'.format(dwi_prefix))

    dwi_degibbs_file = os.path.join(degibbs_dir, '{}_degibbs.nii.gz'.format(dwi_prefix))
    degibbs_cmd = 'mrdegibbs {} {} -force -nthreads {}'.format(dwi_file, dwi_degibbs_file, SHARED_VARS.NUM_THREADS-1)
    run_cmd(degibbs_cmd)
    degibbs_warning_str = 'Gibbs de-ringing applied to {}. Because it can be unstable for partial Fourier acquisitions, we do NOT recommend this for most EPI images. It can also be very difficult to QA, so please carefully check the data in the DEGIBBS output folder in addition to the corresponding page in this PDF.'.format(dwi_prefix)

    print('FINISHED DEGIBBS {}'.format(dwi_prefix))

    return dwi_degibbs_file, degibbs_warning_str

def dwi_unbias(dwi_file, bvals_file, bvecs_file, unbias_dir):

    temp_dir = make_dir(unbias_dir, 'TEMP')

    dwi_prefix = get_prefix(dwi_file, file_ext='nii')

    print('CORRECTING {} BIAS FIELD...'.format(dwi_prefix))

    b0s_file, _, _ = dwi_extract(dwi_file, bvals_file, temp_dir, target_bval=0, first_only=False)
    b0s_avg_file = dwi_avg(b0s_file, temp_dir)
    mask_file = dwi_mask(b0s_avg_file, temp_dir)

    dwi_unbiased_file = os.path.join(unbias_dir, '{}_unbiased.nii.gz'.format(dwi_prefix))
    bias_field_file = os.path.join(unbias_dir, 'bias_field.nii.gz')

    unbias_cmd = 'dwibiascorrect ants {} {} -fslgrad {} {} -mask {} -bias {} -scratch {} -force -nthreads {}'.format(
        dwi_file, dwi_unbiased_file, bvecs_file, bvals_file, mask_file, bias_field_file, temp_dir, SHARED_VARS.NUM_THREADS-1)
    run_cmd(unbias_cmd)

    print('FINISHED UNBIASING {}'.format(dwi_prefix))

    remove_dir(temp_dir)

    return dwi_unbiased_file, bias_field_file

def dwi_norm(dwi_files, bvals_files, norm_dir):

    temp_dir = make_dir(norm_dir, 'TEMP')

    dwi_norm_files = []
    gains = []
    imgs = []
    imgs_normed = []

    # Calculate and apply gains to normalize dwi images

    for i in range(len(dwi_files)):

        dwi_prefix = get_prefix(dwi_files[i])

        print('NORMALIZING {}...'.format(dwi_prefix))

        b0s_file, _, _ = dwi_extract(dwi_files[i], bvals_files[i], temp_dir, target_bval=0, first_only=False)
        b0s_avg_file = dwi_avg(b0s_file, temp_dir)
        mask_file = dwi_mask(b0s_avg_file, temp_dir)

        b0s_avg_img, _, _ = load_nii(b0s_avg_file, ndim=3)
        mask_img, _, _ = load_nii(mask_file, dtype='bool', ndim=3)
        
        img = b0s_avg_img[mask_img]
        if i == 0:
            img_ref = img
            gain = 1
        else:
            img_in = img
            gain = _calc_gain(img_ref, img_in)

        print('GAIN: {}'.format(gain))

        dwi_img, dwi_aff, _ = load_nii(dwi_files[i])
        dwi_norm_img = dwi_img * gain
        dwi_norm_file = os.path.join(norm_dir, '{}_norm.nii.gz'.format(dwi_prefix))
        save_nii(dwi_norm_img, dwi_aff, dwi_norm_file)

        dwi_norm_files.append(dwi_norm_file)
        gains.append(gain)
        imgs.append(list(img))
        imgs_normed.append(list(img * gain))

    # Get average b0 histograms for visualization

    common_min_intensity = 0
    common_max_intensity = 0
    for img in imgs:
        img_max = np.nanmax(img)
        if img_max > common_max_intensity:
            common_max_intensity = img_max
    for img_normed in imgs_normed:
        img_normed_max = np.nanmax(img_normed)
        if img_normed_max > common_max_intensity:
            common_max_intensity = img_normed_max
    bins = np.linspace(common_min_intensity, common_max_intensity, 100)
    
    hists = []
    hists_normed = []
    for i in range(len(imgs)):
        hist, _ = np.histogram(imgs[i], bins=bins)
        hists.append(hist)
        hist_normed, _ = np.histogram(imgs_normed[i], bins=bins)
        hists_normed.append(hist_normed)

    remove_dir(temp_dir)

    return dwi_norm_files, gains, bins[:-1], hists, hists_normed

def dwi_merge(dwi_files, merged_prefix, merge_dir):

    merged_dwi_file = os.path.join(merge_dir, '{}.nii.gz'.format(merged_prefix))

    if len(dwi_files) == 1:

        print('ONLY ONE NII FILE PROVIDED FOR MERGING, COPYING AND RENAMING INPUT')
        copy_file(dwi_files[0], merged_dwi_file)

    else:

        print('MORE THAN ONE IMAGE PROVIDED FOR MERGING, PERFORMING MERGE')
        merge_cmd = 'fslmerge -t {} '.format(merged_dwi_file)
        for dwi_file in dwi_files:
            merge_cmd = '{}{} '.format(merge_cmd, dwi_file)
        run_cmd(merge_cmd)

    return merged_dwi_file

def bvals_merge(bvals_files, merged_prefix, merge_dir):

    merged_bvals_file = os.path.join(merge_dir, '{}.bval'.format(merged_prefix))

    if len(bvals_files) == 1:

        print('ONLY ONE BVAL FILE PROVIDED FOR MERGING, COPYING AND RENAMING INPUT')
        copy_file(bvals_files[0], merged_bvals_file)

    else:

        print('MORE THAN ONE BVALS FILE PROVIDED FOR MERGING, PERFORMING MERGE')
        merged_bvals = np.array([])
        for bvals_file in bvals_files:
            merged_bvals = np.hstack((merged_bvals, load_txt(bvals_file, txt_type='bvals')))
        save_txt(merged_bvals, merged_bvals_file)

    return merged_bvals_file

def bvecs_merge(bvecs_files, merged_prefix, merge_dir):

    merged_bvecs_file = os.path.join(merge_dir, '{}.bvec'.format(merged_prefix))

    if len(bvecs_files) == 1:

        print('ONLY ONE BVEC FILE PROVIDED FOR MERGING, COPYING AND RENAMING INPUT')
        copy_file(bvecs_files[0], merged_bvecs_file)

    else:

        print('MORE THAN ONE BVECS FILE PROVIDED FOR MERGING, PERFORMING MERGE')
        merged_bvecs = np.array([[], [], []])
        for bvecs_file in bvecs_files:
            merged_bvecs = np.hstack((merged_bvecs, load_txt(bvecs_file, txt_type='bvecs')))
        save_txt(merged_bvecs, merged_bvecs_file)

    return merged_bvecs_file

def dwi_volume_prefixes(dwi_files): 

    # File assignments by volume, using file prefix for splitting

    volume_prefixes = []
    for dwi_file in dwi_files:
        dwi_prefix = get_prefix(dwi_file)
        dwi_img, _, _ = load_nii(dwi_file, ndim=4)
        num_vols = dwi_img.shape[3]
        for i in range(num_vols):
            volume_prefixes.append(dwi_prefix)

    return volume_prefixes

def dwi_split(dwi_merged_file, volume_prefixes, split_dir):

    dwi_merged_img, dwi_merged_aff, _ = load_nii(dwi_merged_file, ndim=4)

    dwi_split_files = []

    volume_prefixes = np.array(volume_prefixes)
    unique_volume_prefixes = _unique_prefixes(volume_prefixes)
    for volume_prefix in unique_volume_prefixes:
        dwi_split_img = dwi_merged_img[:, :, :, volume_prefixes == volume_prefix]
        dwi_split_file = os.path.join(split_dir, '{}.nii.gz'.format(volume_prefix))
        save_nii(dwi_split_img, dwi_merged_aff, dwi_split_file)
        
        dwi_split_files.append(dwi_split_file)

    return dwi_split_files

def bvals_split(bvals_merged_file, volume_prefixes, split_dir):

    bvals_merged = load_txt(bvals_merged_file, txt_type='bvals')

    bvals_split_files = []

    volume_prefixes = np.array(volume_prefixes)
    unique_volume_prefixes = _unique_prefixes(volume_prefixes)
    for volume_prefix in unique_volume_prefixes:

        bvals_split = bvals_merged[volume_prefixes == volume_prefix]
        bvals_split_file = os.path.join(split_dir, '{}.bval'.format(volume_prefix))
        save_txt(bvals_split, bvals_split_file)
        
        bvals_split_files.append(bvals_split_file)

    return bvals_split_files

def bvecs_split(bvecs_merged_file, volume_prefixes, split_dir):

    bvecs_merged = load_txt(bvecs_merged_file, txt_type='bvecs')

    bvecs_split_files = []

    volume_prefixes = np.array(volume_prefixes)
    unique_volume_prefixes = _unique_prefixes(volume_prefixes)
    for volume_prefix in unique_volume_prefixes:

        bvecs_split = bvecs_merged[:, volume_prefixes == volume_prefix]
        bvecs_split_file = os.path.join(split_dir, '{}.bvec'.format(volume_prefix))
        save_txt(bvecs_split, bvecs_split_file)
        
        bvecs_split_files.append(bvecs_split_file)

    return bvecs_split_files

def dwi_extract(dwi_file, bvals_file, extract_dir, target_bval=0, first_only=False):

    dwi_prefix = get_prefix(dwi_file)

    print('EXTRACTING {} {} VOLUME(S) FROM {}'.format('FIRST' if first_only else 'ALL', 'B = {}'.format(target_bval), dwi_prefix))

    dwi_img, dwi_aff, _ = load_nii(dwi_file, ndim=4)
    bvals = load_txt(bvals_file, txt_type='bvals')

    num_total_vols = dwi_img.shape[3]
    index = np.array(range(0, num_total_vols))
    index = index[bvals == target_bval]

    if first_only:

        print('EXTRACTING FIRST VOLUME ONLY => 3D OUTPUT')
        dwi_extracted_img = dwi_img[:, :, :, index[0]]
        num_extracted_vols = 1

    else:

        print('EXTRACTING ALL VALID VOLUMES => 4D OUTPUT')
        dwi_extracted_img = dwi_img[:, :, :, index]
        num_extracted_vols = len(index)

    print('EXTRACTED IMAGE HAS SHAPE {}'.format(dwi_extracted_img.shape))

    dwi_extracted_file = os.path.join(extract_dir, '{}_b{}_{}.nii.gz'.format(dwi_prefix, target_bval, 'first' if first_only else 'all'))
    save_nii(dwi_extracted_img, dwi_aff, dwi_extracted_file, ndim=4)

    return dwi_extracted_file, num_extracted_vols, num_total_vols

def dwi_avg(dwi_file, avg_dir):

    dwi_prefix = get_prefix(dwi_file)

    print('AVERAGING {}'.format(dwi_prefix))

    dwi_img, dwi_aff, _ = load_nii(dwi_file, ndim=4)
    dwi_avg_img = np.nanmean(dwi_img, axis=3)
    dwi_avg_file = os.path.join(avg_dir, '{}_avg.nii.gz'.format(dwi_prefix))
    save_nii(dwi_avg_img, dwi_aff, dwi_avg_file, ndim=3)

    return dwi_avg_file

def dwi_mask(dwi_file, mask_dir):

    temp_dir = make_dir(mask_dir, 'TEMP')
    
    # Compute bet mask on 3D DWI image

    bet_file = os.path.join(temp_dir, 'bet.nii.gz')
    bet_mask_file = os.path.join(temp_dir, 'bet_mask.nii.gz')
    bet_cmd = 'bet {} {} -f 0.25 -m -n -R'.format(dwi_file, bet_file)
    run_cmd(bet_cmd)

    # Move binary mask out of temp directory

    dwi_prefix = get_prefix(dwi_file)
    mask_file = os.path.join(mask_dir, '{}_mask.nii.gz'.format(dwi_prefix))
    move_file(bet_mask_file, mask_file)

    # Clean up

    remove_dir(temp_dir)

    return mask_file

def dwi_improbable_mask(mask_file, dwi_file, bvals_file, mask_dir):

    mask_prefix = get_prefix(mask_file)

    print('IDENTIFYING VOXELS FOR IMPROBABLE MASK, BUILDING ON EXISTING MASK {}'.format(mask_prefix))

    # Load mask, DWI, and b-values 

    mask_img, mask_aff, _ = load_nii(mask_file, dtype='bool', ndim=3)
    dwi_img, _, _ = load_nii(dwi_file, ndim=4)
    bvals = load_txt(bvals_file, txt_type='bvals')

    # Keep voxels where the minimum value across b0s is greater than the minimum value across dwis
    # and its in the original mask

    b0_min_img = np.amin(dwi_img[:, :, :, bvals == 0], axis=3)
    dwi_min_img = np.amin(dwi_img[:, :, :, bvals != 0], axis=3)
    improbable_mask_img = np.logical_and(b0_min_img > dwi_min_img, mask_img)

    # Compute Percent of intra-mask voxels that are improbable

    percent_improbable = 100 * (1 - np.sum(improbable_mask_img)/np.sum(mask_img))
    print('WITHIN MASK {}, {:.2f}% OF VOXELS WERE IMPROBABLE'.format(mask_prefix, percent_improbable))

    # Save improbable mask

    improbable_mask_file = os.path.join(mask_dir, '{}_improbable.nii.gz'.format(mask_prefix))
    save_nii(improbable_mask_img.astype(int), mask_aff, improbable_mask_file, ndim=3)

    return improbable_mask_file, percent_improbable

def dwi_smooth(dwi_file, smooth_dir):

    dwi_prefix = get_prefix(dwi_file)

    print('SMOOTHING {}'.format(dwi_prefix))

    # Smooth diffusion image (primarily for use with synb0 pipeline)

    dwi_smooth_file = os.path.join(smooth_dir, '{}_smooth.nii.gz'.format(dwi_prefix))
    smooth_cmd = 'fslmaths {} -s 1.15 {}'.format(dwi_file, dwi_smooth_file)
    run_cmd(smooth_cmd)

    return dwi_smooth_file

def bvals_scale(bvals_file, scale_factor, scaled_dir):

    bvals_prefix = get_prefix(bvals_file, file_ext='bval')

    print('SCALING {} BVALS BY {}'.format(bvals_prefix, scale_factor))

    bvals = load_txt(bvals_file, txt_type='bvals')
    bvals_scaled = scale_factor * bvals
    bvals_scaled_file = os.path.join(scaled_dir, '{}_x{}.bval'.format(bvals_prefix, scale_factor))
    save_txt(bvals_scaled, bvals_scaled_file)

    return bvals_scaled_file

# Function Definitions: Phase Encoding Scheme Manipulation

def pescheme2params(pe_axis, pe_dir, readout_time):

    pe_scheme = '{}{}'.format(pe_axis, pe_dir)

    params_dict = {
        'i+': [1, 0, 0],
        'i-': [-1, 0, 0],
        'j+': [0, 1, 0],
        'j-': [0, -1, 0],
    }
    params = params_dict.get(pe_scheme, 'INVALID')
    if params == 'INVALID':
        raise DTIQAError('INVALID PHASE ENCODING SCHEME SPECIFIED!')
    params.append(readout_time)
    
    params = [str(param) for param in params]
    params_line = ' '.join(params)

    return params_line

def pescheme2axis(pe_axis, pe_dir, aff):

    if pe_axis == 'i':
        axis_idx = 0
    elif pe_axis == 'j':
        axis_idx = 1
    else:
        raise DTIQAError('INVALID PHASE ENCODING AXIS SPECIFIED!')

    axis_codes = nib.orientations.aff2axcodes(aff)
    axis_name = axis_codes[axis_idx]

    if pe_dir == '-':
        dir_name = 'From'
    elif pe_dir == '+':
        dir_name = 'To'
    else:
        raise DTIQAError('INVALID PHASE ENCODING DIRECTION SPECIFIED!')

    axis_str = '{} {}'.format(dir_name, axis_name)

    return axis_str

# Function Definitions: Visualization

def slice_nii(nii_file, offsets=[0], custom_aff=[], min_percentile=0, max_percentile=100, min_intensity=np.nan, max_intensity=np.nan, det=False):

    if det:
        img, aff, hdr = load_nii(nii_file, ndim=4)
        img = det_matrix(img)
    else:
        img, aff, hdr = load_nii(nii_file, ndim=3)
    # Extract voxel dimensions and reorient image in radiological view

    vox_dim = hdr.get_zooms()
    if len(custom_aff) > 0:
        aff = custom_aff
    img, vox_dim = _radiological_view(img, aff, vox_dim)

    # Extract min and max of entire volume so slices can be plotted with homogenous scaling

    if np.isnan(min_intensity):
        img_min = np.nanpercentile(img, min_percentile)
    else:
        img_min = min_intensity
        
    if np.isnan(max_intensity):
        img_max = np.nanpercentile(img, max_percentile)
    else:
        img_max = max_intensity

    # Extract center triplanar slices with offsets.

    i0 = int(round(img.shape[0] / 2, 1))
    i1 = int(round(img.shape[1] / 2, 1))
    i2 = int(round(img.shape[2] / 2, 1))

    i0s = []
    i1s = []
    i2s = []
    for offset in offsets:
        i0s.append(i0 + offset)
        i1s.append(i1 + offset)
        i2s.append(i2 + offset)

    s0s = img[i0s, :, :]
    s1s = img[:, i1s, :]
    s2s = img[:, :, i2s]

    slices = (s0s, s1s, s2s)

    # Output Descriptions:

    # slices: a list of sagittal, coronal, and axial volumes.
    # - slices[1] gives the coronal volume
    # - slices[1][:, 0, :] gives the first coronal slices offset offsets[0] from the center coronal slice
    # - np.rot90(np.squeeze(slices[1][:, 0, :])) prepares it for plotting
    # vox_dim: a list of 3 values corresponding to the real-life sizes of each voxel, needed for proper axis scaling when plotting slices
    # img_min and img_max: the min and max values of the img volume, needed for proper homogenous intensity scaling when plotting different slices

    return slices, vox_dim, img_min, img_max

def plot_slice(slices, img_dim, offset_index, vox_dim, img_min, img_max, alpha=1, cmap='gray'):

    s = slices[img_dim]
    if img_dim == 0:
        s = s[offset_index, :, :]
        vox_ratio = vox_dim[2]/vox_dim[1]
    elif img_dim == 1:
        s = s[:, offset_index, :]
        vox_ratio = vox_dim[2]/vox_dim[0]
    elif img_dim == 2:
        s = s[:, :, offset_index]
        vox_ratio = vox_dim[1]/vox_dim[0]
    s = np.rot90(np.squeeze(s))
    im = plt.imshow(s, cmap=cmap, vmin=img_min, vmax=img_max, aspect=vox_ratio, alpha=alpha)
    plt.xticks([], [])
    plt.yticks([], [])
    return im

def plot_slice_contour(slices, img_dim, offset_index, color):

    s = slices[img_dim]
    if img_dim == 0:
        s = s[offset_index, :, :]
    elif img_dim == 1:
        s = s[:, offset_index, :]
    elif img_dim == 2:
        s = s[:, :, offset_index]
    s = np.rot90(np.squeeze(s))
    
    slice_contours = measure.find_contours(s, 0.9)
    for slice_contour in enumerate(slice_contours):
        plt.plot(slice_contour[1][:,1], slice_contour[1][:,0], linewidth=1, color=color)

def merge_pdfs(pdf_files, merged_prefix, pdf_dir):

    print('MERGING PDFS')

    pdf_files_str = ' '.join(pdf_files)
    merged_pdf_file = os.path.join(pdf_dir, '{}.pdf'.format(merged_prefix))
    gs_cmd = 'gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE={} -dBATCH {}'.format(merged_pdf_file, pdf_files_str)
    run_cmd(gs_cmd)

    print('CLEANING UP COMPONENT PDFS')
    remove_file(pdf_files_str)

    return merged_pdf_file

# Function Definitions: Math Helper Functions

def nearest(value, array):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def round(num, base):

    d = num / base
    if d % 1 >= 0.5:
        return base*np.ceil(d)
    else:
        return base*np.floor(d)

# Private Helper Functions

def _calc_gain(img_ref, img_in):

    gain_inits = np.linspace(0.5, 1.5, 10)
    gains = np.zeros(len(gain_inits))
    errors = np.zeros(len(gain_inits))
    for i in range(len(gain_inits)):
        gain, error, _, _, _ = fmin(_err, gain_inits[i], args=(img_ref, img_in), full_output=True)
        gains[i] = gain[0]
        errors[i] = error
    return gains[np.argmin(errors)]

def _err(gain, img_ref, img_in):

    img_in_gained = img_in * gain
    common_min_intensity = 0
    common_max_intensity = np.amax((np.amax(img_ref), np.amax(img_in_gained)))
    bins = np.linspace(common_min_intensity, common_max_intensity, 1000)
    hist_ref, _ = np.histogram(img_ref, bins=bins)
    hist_in_gained, _ = np.histogram(img_in_gained, bins=bins)
    return _hi_loss(hist_ref, hist_in_gained)

def _mse_loss(hist_ref, hist_in):

    return np.mean(np.square(hist_ref - hist_in))

def _hi_loss(hist_ref, hist_in):

    return -np.sum(np.minimum(hist_ref, hist_in))

def _radiological_view(img, aff, vox_dim=(1, 1, 1)):

    # RAS defined by nibabel as L->R, P->A, I->S. Orientation functions from nibabel assume this.
    # "Radiological view" is LAS. Want to view in radiological view for doctors.
    # NIFTIs are required to have world coordinates in RAS.

    # Some helpful links:
    # https://users.fmrib.ox.ac.uk/~paulmc/fsleyes/userdoc/latest/display_space.html#radiological-vs-neurological
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained

    orientations = nib.orientations.io_orientation(aff) # Get orientations relative to RAS in nibabel

    old_axis_order = np.array(orientations[:, 0]) # Permute to get RL, AP, IS axes in right order
    new_axis_order = np.array([0, 1, 2])
    permute_axis_order = list(np.array([new_axis_order[old_axis_order == 0],
                                        new_axis_order[old_axis_order == 1],
                                        new_axis_order[old_axis_order == 2]]).flatten())
    img = np.transpose(img, axes=permute_axis_order)
    orientations_permute = orientations[permute_axis_order]

    vox_dim = np.array(vox_dim)[permute_axis_order] # Do the same to reorder pixel dimensions

    for orientation in orientations_permute: # Flip axes as needed to get R/A/S as positive end of axis (into radiological view)
        if (orientation[1] == 1 and orientation[0] == 0) or (orientation[1] == -1 and orientation[0] > 0):
            img = nib.orientations.flip_axis(img, axis=orientation[0].astype('int'))
    
    return img, vox_dim

def _unique_prefixes(volume_prefixes):

    unique_volume_prefix_indices = np.unique(volume_prefixes, return_index=True)[1]
    unique_volume_prefix_indices.sort()
    unique_volume_prefixes = volume_prefixes[unique_volume_prefix_indices]
    return unique_volume_prefixes

def compute_FA(resmaple_gradtensor_file, gradtensor_fa_file):
    
    img = nib.load(resmaple_gradtensor_file)
    affine = img.affine
    LR_mat = img.get_fdata()
    eig_vol = np.zeros([LR_mat.shape[0],LR_mat.shape[1],LR_mat.shape[2],3])
    for i in range(LR_mat.shape[0]):
        for j in range(LR_mat.shape[1]):
            for k in range(LR_mat.shape[2]):
                r = np.reshape(LR_mat[i,j,k,:],[3,3])
                w, _ = np.linalg.eig(r)
                eig_vol[i,j,k,:] = w

    ev1 = eig_vol[:,:,:,0]
    ev2 = eig_vol[:,:,:,1]
    ev3 = eig_vol[:,:,:,2]
    FA = np.sqrt(0.5) * ( np.sqrt ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2) / (np.sqrt (ev1 **2) + (ev2 **2) + (ev3 **2)))
    fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
    nib.save(fa_img, gradtensor_fa_file)

def det_matrix(l):
        x_dim = l.shape[0]
        y_dim = l.shape[1]
        z_dim = l.shape[2]
        vL = np.zeros((3,3,x_dim,y_dim,z_dim))
        vL[0,0,:,:,:] = l[:,:,:,0]
        vL[0,1,:,:,:] = l[:,:,:,1]
        vL[0,2,:,:,:] = l[:,:,:,2]
        vL[1,0,:,:,:] = l[:,:,:,3]
        vL[1,1,:,:,:] = l[:,:,:,4]
        vL[1,2,:,:,:] = l[:,:,:,5]
        vL[2,0,:,:,:] = l[:,:,:,6]
        vL[2,1,:,:,:] = l[:,:,:,7]
        vL[2,2,:,:,:] = l[:,:,:,8]

        L_det = np.zeros((x_dim,y_dim,z_dim))

        # Along all axis obtain the determinant of LR matrix
        for x in range(x_dim):
                for y in range(y_dim):
                        for z in range(z_dim):
                                L_mat = np.squeeze(vL[:,:,x,y,z])
                                L_det[x,y,z] = np.linalg.det(L_mat[:,:])
        return L_det
