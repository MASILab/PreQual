# PreQual
# Leon Cai and Qi Yang
# MASI Lab
# Vanderbilt University

# Set Up

import sys
import os
import argparse as ap
import time

import numpy as np

import preproc
import stats
import vis
import utils
from vars import SHARED_VARS

# Run Pipeline

def main():

    # DEFINE ARGUMENTS

    parser = ap.ArgumentParser(description='PreQual (dtiQA v7 Multi): An automated pipeline for integrated preprocessing and quality assurance of diffusion weighted MRI images')
    parser.add_argument('in_dir', help='A path to the INPUTS directory that must contain dtiQA_config.csv')
    parser.add_argument('out_dir', help='A path to the OUTPUTS directory')
    parser.add_argument('pe_axis', help='Phase encoding axis (direction agnostic) (i.e. i or j)')
    parser.add_argument('--bval_threshold', metavar='N', default='50', help='Non-negative integer threshold under which to consider a b-value to be 0 (default = 50)')
    parser.add_argument('--nonzero_shells', metavar='s1,s2,...,sn/auto', default='auto', help='Comma separated list of positive integers indicating nonzero shells for SNR/CNR analysis when there are more unique b-values than shells determined by eddy or automatically determine shells by rounding to nearest 100 (default = auto)')
    parser.add_argument('--denoise', metavar='on/off', default='on', help='Denoise images prior to preprocessing (default = on)')
    parser.add_argument('--prenormalize', metavar='on/off', default='on', help='Normalize intensity distributions before preprocessing (default = on)')
    parser.add_argument('--synb0', metavar='on/off', default='on', help='Run topup with a synthetic b0 generated with Synb0-DisCo if no reverse phase encoded images are supplied and a T1 is supplied (default = on)')
    parser.add_argument('--extra_topup_args', metavar='string', default='', help='Extra arguments to pass to topup')
    parser.add_argument('--eddy_cuda', metavar='8.0/9.1/off', default='off', help='Run eddy with CUDA 8.0 or 9.1 or without GPU acceleration and with OPENMP only (default = off)')
    parser.add_argument('--eddy_mask', metavar='on/off', default='on', help='Use a brain mask for eddy (default = on)')
    parser.add_argument('--eddy_bval_scale', metavar='N/off', default='off', help='Positive number with which to scale b-values for eddy only in order to perform distortion correction on super low shells (default = off)')
    parser.add_argument('--extra_eddy_args', metavar='string', default='', help='Extra arguments to pass to eddy')
    parser.add_argument('--postnormalize', metavar='on/off', default='off', help='Normalize intensity distributions after preprocessing (default = off)')
    parser.add_argument('--correct_bias', metavar='on/off', default='off', help='Perform N4 bias field correction as implemented in ANTS (default = off)')
    parser.add_argument('--split_outputs', action='store_true', help='Split preprocessed output to match structure of input files (default = do NOT split)')
    parser.add_argument('--keep_intermediates', action='store_true', help='Keep intermediate copies of data (default = do NOT keep)')
    parser.add_argument('--num_threads', metavar='N', default=1, help='Non-negative integer indicating number of threads to use when running multi-threaded steps of this pipeline (default = 1)')
    parser.add_argument('--project', metavar='string', default='proj', help='Project ID (default = proj)')
    parser.add_argument('--subject', metavar='string', default='subj', help='Subject ID (default = subj)')
    parser.add_argument('--session', metavar='string', default='sess', help='Session ID (default = sess)')
    args = parser.parse_args()

    # START PIPELINE

    print('***********************************')
    print('*** DTIQA V7: PIPELINE STARTING ***')
    print('***********************************\n')

    Ti = time.time()

    # PREPARE LIST TO SAVE ALL WARNINGS

    warning_strs = []

    # INPUTS AND OUTPUTS

    print('*****************************')
    print('*** DTIQA V7: PARSING I/O ***')
    print('*****************************')

    ti = time.time()

    # PARSE INPUTS

    in_dir = args.in_dir
    out_dir = args.out_dir
    pe_axis = args.pe_axis

    # PARSE PARAMETERS INTO params DICTIONARY

    params = {}

    bval_threshold = float(args.bval_threshold)
    if int(bval_threshold) == bval_threshold and int(bval_threshold) > 0:
        params['bval_threshold'] = int(bval_threshold)
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --bval_threshold PARAMETER. EXITING.')

    if args.nonzero_shells == 'auto':
        params['shells'] = []
    else:
        try:
            shells = [int(s) for s in args.nonzero_shells.split(',')]
            if np.amin(shells) < 0:
                raise utils.DTIQAError('INVALID INPUT FOR --nonzero_shells PARAMETER. EXITING.')
            shells.append(0)
            params['shells'] = np.sort(np.unique(shells))
        except:
            raise utils.DTIQAError('INVALID INPUT FOR --nonzero_shells PARAMETER. EXITING.')

    if args.denoise == 'on':
        params['use_denoise'] = True
    elif args.denoise == 'off':
        params['use_denoise'] = False
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --denoise PARAMETER. EXITING.')

    if args.prenormalize == 'on':
        params['use_prenormalize'] = True
    elif args.prenormalize == 'off':
        params['use_prenormalize'] = False
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --prenormalize PARAMETER. EXITING.')

    if args.synb0 == 'on':
        params['use_synb0_user'] = True
    elif args.synb0 == 'off':
        params['use_synb0_user'] = False
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --synb0 PARAMETER. EXITING.')

    params['extra_topup_args'] = args.extra_topup_args
    if len(params['extra_topup_args']) > 0:
        warning_strs.append('Additional inputs given to topup. These are untested and may produce pipeline failure or inaccurate results.')

    if args.eddy_cuda == '8.0':
        params['eddy_cuda_version'] = 8.0
    elif args.eddy_cuda == '9.1':
        params['eddy_cuda_version'] = 9.1
    elif args.eddy_cuda == 'off':
        params['eddy_cuda_version'] = 0
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --eddy_cuda PARAMETER. EXITING.')

    if args.eddy_mask == 'on':
        params['eddy_mask_type'] = 'brain'
    elif args.eddy_mask == 'off':
        params['eddy_mask_type'] = 'volume'
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --eddy_mask PARAMETER. EXITING.')

    eddy_bval_scale = args.eddy_bval_scale
    if eddy_bval_scale == 'off':
        eddy_bval_scale = 1
    eddy_bval_scale = float(eddy_bval_scale)
    if eddy_bval_scale > 0:
        params['eddy_bval_scale'] = eddy_bval_scale
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --eddy_bval_scale PARAMETER. EXITING.')

    params['extra_eddy_args'] = args.extra_eddy_args
    if len(params['extra_eddy_args']) > 0:
        warning_strs.append('Additional inputs given to eddy. These are untested and may produce pipeline failure or inaccurate results.')

    if args.postnormalize == 'on':
        params['use_postnormalize'] = True
    elif args.postnormalize == 'off':
        params['use_postnormalize'] = False
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --postnormalize PARAMETER. EXITING.')

    if args.correct_bias == 'on':
        params['use_unbias'] = True
    elif args.correct_bias == 'off':
        params['use_unbias'] = False
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --correct_bias PARAMETER. EXITING.')

    params['split_outputs'] = args.split_outputs
    params['keep_intermediates'] = args.keep_intermediates

    num_threads = float(args.num_threads)
    if int(num_threads) == num_threads and int(num_threads) > 0:
        SHARED_VARS.NUM_THREADS = int(num_threads)
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --num_threads PARAMETER. EXITING.')

    params['project'] = args.project
    params['subject'] = args.subject
    params['session'] = args.session

    # GATHER FILES AND CHECK CONFIGURATION

    dwi_files, bvals_files, bvecs_files, pe_dirs, readout_times = utils.load_config(in_dir)

    num_pe_dirs = len(np.unique(pe_dirs))

    if not (len(dwi_files) > 0 and len(dwi_files) == len(bvals_files) and len(dwi_files) == len(bvecs_files)):
        raise utils.DTIQAError('NUMBER OF INPUT IMAGES AND BVAL/BVEC FILES MISMATCH. EXITING.')
    if num_pe_dirs > 2 or num_pe_dirs < 1:
        raise utils.DTIQAError('NUMBER OF PHASE ENCODING DIRECTIONS INVALID. EXITING.')

    # DETERMINE IF TOPUP CAN RUN: RPE OR T1 MUST BE PROVIDED

    t1_file = os.path.join(in_dir, 't1.nii.gz')
    if num_pe_dirs == 2:
        use_synb0 = False
        use_topup = True
    if num_pe_dirs == 1:
        use_synb0 = os.path.exists(t1_file) and params['use_synb0_user'] # if not RPE provided, only run synb0 if T1 provided AND user desires it
        use_topup = use_synb0

    # PRINT I/O

    print('PROJECT DETAILS:')
    print('- PROJECT: {}'.format(params['project']))
    print('- SUBJECT: {}'.format(params['subject']))
    print('- SESSION: {}'.format(params['session']))
    print('INPUTS:')
    for i in range(len(dwi_files)):
        print('- {} ({}{}, {})\n- {}\n- {}'.format(dwi_files[i], pe_axis, pe_dirs[i], readout_times[i], bvals_files[i], bvecs_files[i]))
    if os.path.exists(t1_file):
        print('- {}'.format(t1_file))
    print('PREPROCESSING:')
    print('- {}'.format('Eddy {}'.format('Only' if not use_topup else '+ Topup ({})'.format('Synb0' if use_synb0 else 'RPE'))))
    print('PARAMETERS:')
    print('- BValue Threshold: {}'.format(params['bval_threshold']))
    print('- Shells: {}'.format(params['shells']))
    print('- Denoise: {}'.format(params['use_denoise']))
    print('- Prenormalize: {}'.format(params['use_prenormalize']))
    print('- Extra Topup Args: {}'.format(params['extra_topup_args']))
    print('- Eddy CUDA: {}'.format(params['eddy_cuda_version']))
    print('- Eddy Mask Type: {}'.format(params['eddy_mask_type']))
    print('- Eddy BValue Scale: {}'.format(params['eddy_bval_scale']))
    print('- Extra Eddy Args: {}'.format(params['extra_eddy_args']))
    print('- Postnormalize: {}'.format(params['use_postnormalize']))
    print('- N4 Bias Correction: {}'.format(params['use_unbias']))
    print('- Split Outputs: {}'.format(params['split_outputs']))
    print('- Keep Intermediates: {}'.format(params['keep_intermediates']))
    print('- Number of Threads: {}'.format(SHARED_VARS.NUM_THREADS))
    print('OUTPUT DIRECTORY: {}'.format(out_dir))

    tf = time.time()
    dt = round(tf - ti)

    print('*************************************')
    print('*** DTIQA V7: I/O PARSED ({:05d}s) ***'.format(dt))
    print('*************************************\n')

    # THRESHOLD B = 0

    print('********************************************')
    print('*** DTIQA V7: SETTING B = 0 IF B < {:05d} ***'.format(params['bval_threshold']))
    print('********************************************')

    ti = time.time()

    threshold_dir = utils.make_dir(out_dir, 'THRESHOLDED_BVALS')

    bvals_thresholded_files = []
    for bvals_file in bvals_files:
        bvals_thresholded_file = utils.bvals_threshold(bvals_file, params['bval_threshold'], threshold_dir)
        bvals_thresholded_files.append(bvals_thresholded_file)

    tf = time.time()
    dt = round(tf - ti)

    print('********************************************')
    print('*** DTIQA V7: BVALS THRESHOLDED ({:05d}s) ***'.format(dt))
    print('********************************************\n')

    # CHECK BVECS FOR UNIT NORMALIZATION

    print('*******************************************')
    print('*** DTIQA V7: CHECKING BVAL/BVEC COMBOS ***')
    print('*******************************************')

    ti = time.time()

    check_dir = utils.make_dir(out_dir, 'CHECKED')

    dwi_checked_files = []
    bvals_checked_files = []
    bvecs_checked_files = []
    for i in range(len(dwi_files)):
        dwi_checked_file, bvals_checked_file, bvecs_checked_file, dwi_check_warning_str = utils.dwi_check(dwi_files[i], bvals_thresholded_files[i], bvecs_files[i], check_dir)
        dwi_checked_files.append(dwi_checked_file)
        bvals_checked_files.append(bvals_checked_file)
        bvecs_checked_files.append(bvecs_checked_file)
        if not dwi_check_warning_str == '':
            warning_strs.append(dwi_check_warning_str)

    tf = time.time()
    dt = round(tf - ti)

    print('***************************************************')
    print('*** DTIQA V7: BVAL/BVEC COMBOS CHECKED ({:05d}s) ***'.format(dt))
    print('***************************************************\n')

    # DENOISE

    print('************************************')
    print('*** DTIQA V7: STARTING DENOISING ***')
    print('************************************')

    ti = time.time()

    denoised_dir = utils.make_dir(out_dir, 'DENOISED')

    if params['use_denoise']:

        dwi_denoised_files = []
        for dwi_checked_file in dwi_checked_files:
            dwi_denoised_file, denoise_warning_str = utils.dwi_denoise(dwi_checked_file, denoised_dir)
            dwi_denoised_files.append(dwi_denoised_file)
            if not denoise_warning_str == '':
                warning_strs.append(denoise_warning_str)

    else:

        print('SKIPPING DENOISING')
        dwi_denoised_files = dwi_checked_files

    tf = time.time()
    dt = round(tf - ti)

    print('*********************************************')
    print('*** DTIQA V7: FINISHED DENOISING ({:05d}s) ***'.format(dt))
    print('*********************************************\n')

    # PRENORMALIZE OR GAIN CALCULATIONS

    print('*******************************************')
    print('*** DTIQA V7: STARTING {} ***'.format('PRENORMALIZATION' if params['use_prenormalize'] else 'GAIN ESTIMATIONS'))
    print('*******************************************')

    ti = time.time()

    if params['use_prenormalize']:
        prenorm_dir = utils.make_dir(out_dir, 'PRENORMALIZED')
    else:
        print('RUNNING PRENORMALIZATION ALGORITHMS FOR GAIN CALCULATIONS ONLY. PRENORMALIZATION WILL NOT BE APPLIED TO IMAGES.')
        prenorm_dir = utils.make_dir(out_dir, 'GAIN_CHECK')

    dwi_prenorm_files, dwi_prenorm_gains, dwi_prenorm_bins, dwi_denoised_hists, dwi_prenormed_hists = \
        utils.dwi_norm(dwi_denoised_files, bvals_checked_files, prenorm_dir)

    if not params['use_prenormalize']:
        dwi_prenorm_files = dwi_denoised_files
        for dwi_prenorm_gain in dwi_prenorm_gains:
            if dwi_prenorm_gain > 1.05 or dwi_prenorm_gain < 0.95:
                print('PRENORMALIZE IS OFF AND GAINS > 5% WERE DETECTED!')
                warning_strs.append('Images with gain differences greater than 5% were detected. Please see the Gain QA page of this PDF for more information.')
                break
    
    tf = time.time()
    dt = round(tf - ti)

    print('****************************************************')
    print('*** DTIQA V7: FINISHED {} ({:05d}s) ***'.format('PRENORMALIZATION' if params['use_prenormalize'] else 'GAIN ESTIMATIONS', dt))
    print('****************************************************\n')

    # PREPROCESSING PREPARATION

    print('*********************************************')
    print('*** DTIQA V7: PREPARING FOR PREPROCESSING ***')
    print('*********************************************')

    ti = time.time()

    topup_dir = utils.make_dir(out_dir, 'TOPUP')
    eddy_dir = utils.make_dir(out_dir, 'EDDY')
    
    topup_input_b0s_file, topup_acqparams_file, b0_d_file, b0_syn_file, eddy_input_dwi_file, eddy_input_bvals_file, eddy_input_bvecs_file, eddy_acqparams_file, eddy_index_file = \
        preproc.prep(dwi_prenorm_files, bvals_checked_files, bvecs_checked_files, pe_axis, pe_dirs, readout_times, use_topup, use_synb0, t1_file, topup_dir, eddy_dir, params['eddy_bval_scale'])

    tf = time.time()
    dt = round(tf - ti)

    print('*************************************************************')
    print('*** DTIQA V7: FINISHED PREPROCESSING PREPARATION ({:05d}s) ***'.format(dt))
    print('*************************************************************\n')
    
    # TOPUP

    print('*******************************')
    print('*** DTIQA V7: RUNNING TOPUP ***')
    print('*******************************')

    ti = time.time()

    if use_topup:

        topup_results_prefix, topup_output_b0s_file = preproc.topup(topup_input_b0s_file, topup_acqparams_file, params['extra_topup_args'], topup_dir)

    else:

        print('SKIPPING TOPUP')

        topup_results_prefix = ''
        topup_output_b0s_file = ''
        warning_strs.append('No RPE or T1 image provided, skipping topup.')

    tf = time.time()
    dt = round(tf - ti)

    print('*****************************************')
    print('*** DTIQA V7: TOPUP FINISHED ({:05d}s) ***'.format(dt))
    print('*****************************************\n')

    # EDDY

    print('******************************')
    print('*** DTIQA V7: RUNNING EDDY ***')
    print('******************************')

    ti = time.time()

    eddy_output_dwi_file, eddy_output_bvals_file, eddy_output_bvecs_file, eddy_mask_file, eddy_warning_str = \
        preproc.eddy(eddy_input_dwi_file, eddy_input_bvals_file, eddy_input_bvecs_file, eddy_acqparams_file, eddy_index_file, params['eddy_mask_type'], params['eddy_cuda_version'], params['eddy_bval_scale'], topup_results_prefix, topup_output_b0s_file, params['extra_eddy_args'], eddy_dir)
    if not eddy_warning_str == '':
        warning_strs.append(eddy_warning_str)

    tf = time.time()
    dt = round(tf - ti)

    print('****************************************')
    print('*** DTIQA V7: EDDY FINISHED ({:05d}s) ***'.format(dt))
    print('****************************************\n')

    # POST-NORMALIZE

    print('********************************************')
    print('*** DTIQA V7: STARTING POSTNORMALIZATION ***')
    print('********************************************')

    ti = time.time()

    norm_dir = utils.make_dir(out_dir, 'POSTNORMALIZED')

    if params['use_postnormalize']:

        # Split preprocessed data for normalization

        eddy_volume_prefixes = utils.dwi_volume_prefixes(dwi_prenorm_files)
        for i in range(len(eddy_volume_prefixes)):
            eddy_volume_prefixes[i] = '{}_{}'.format(eddy_volume_prefixes[i], 'topup_eddy' if use_topup else 'eddy')

        dwi_eddy_files = utils.dwi_split(eddy_output_dwi_file, eddy_volume_prefixes, norm_dir)
        bvals_eddy_files = utils.bvals_split(eddy_output_bvals_file, eddy_volume_prefixes, norm_dir)
        bvecs_eddy_files = utils.bvecs_split(eddy_output_bvecs_file, eddy_volume_prefixes, norm_dir)

        # Normalize

        dwi_norm_files, dwi_norm_gains, dwi_norm_bins, dwi_eddy_hists, dwi_normed_hists = \
            utils.dwi_norm(dwi_eddy_files, bvals_eddy_files, norm_dir)

        # Remerge 

        norm_prefix = 'normed'

        dwi_norm_file = utils.dwi_merge(dwi_norm_files, norm_prefix, norm_dir)
        bvals_norm_file = utils.bvals_merge(bvals_eddy_files, norm_prefix, norm_dir)
        bvecs_norm_file = utils.bvecs_merge(bvecs_eddy_files, norm_prefix, norm_dir)

    else:

        print('SKIPPING POSTNORMALIZATION')

        dwi_norm_file = eddy_output_dwi_file
        bvals_norm_file = eddy_output_bvals_file
        bvecs_norm_file = eddy_output_bvecs_file

    tf = time.time()
    dt = round(tf - ti)

    print('*****************************************************')
    print('*** DTIQA V7: FINISHED POSTNORMALIZATION ({:05d}s) ***'.format(dt))
    print('*****************************************************\n')

    # PERFORM N4 BIAS FIELD CORRECTION

    print('********************************************')
    print('*** DTIQA V7: CORRECTING BIAS FIELD (N4) ***')
    print('********************************************')

    ti = time.time()

    unbias_dir = utils.make_dir(out_dir, 'UNBIASED')

    if params['use_unbias']:

        dwi_unbiased_file, bias_field_file = utils.dwi_unbias(dwi_norm_file, bvals_norm_file, bvecs_norm_file, unbias_dir)

    else:

        print('SKIPPING BIAS FIELD CORRECTION')

        dwi_unbiased_file = dwi_norm_file
        bias_field_file = ''

    bvals_unbiased_file = bvals_norm_file
    bvecs_unbiased_file = bvecs_norm_file

    tf = time.time()
    dt = round(tf - ti)

    print('************************************************')
    print('*** DTIQA V7: BIAS FIELD CORRECTED ({:05d}s) ***'.format(dt))
    print('************************************************\n')

    # FORMALIZE OUTPUTS IN PREPROCESSED FOLDER

    print('*************************************')
    print('*** DTIQA V7: FORMALIZING OUTPUTS ***')
    print('*************************************')

    ti = time.time()

    preproc_dir = utils.make_dir(out_dir, 'PREPROCESSED')

    preproc_prefix = 'dwmri'
    dwi_preproc_file = os.path.join(preproc_dir, '{}.nii.gz'.format(preproc_prefix))
    bvals_preproc_file = os.path.join(preproc_dir, '{}.bval'.format(preproc_prefix))
    bvecs_preproc_file = os.path.join(preproc_dir, '{}.bvec'.format(preproc_prefix))

    utils.copy_file(dwi_unbiased_file, dwi_preproc_file)
    utils.copy_file(bvals_unbiased_file, bvals_preproc_file)
    utils.copy_file(bvecs_unbiased_file, bvecs_preproc_file)

    tf = time.time()
    dt = round(tf - ti)

    print('*********************************************')
    print('*** DTIQA V7: OUTPUTS FORMALIZED ({:05d}s) ***'.format(dt))
    print('*********************************************\n')

    # FINAL MASK

    print('********************************************')
    print('*** DTIQA V7: CREATING PREPROCESSED MASK ***')
    print('********************************************')

    ti = time.time()

    mask_file = preproc.mask(dwi_preproc_file, bvals_preproc_file, 'brain', preproc_dir)

    mask_file = utils.rename_file(mask_file, os.path.join(preproc_dir, 'mask.nii.gz')) # Name it mask.nii.gz in the PREPROCESSED directory

    tf = time.time()
    dt = round(tf - ti)

    print('***************************************************')
    print('*** DTIQA V7: PREPROCESSED MASK SAVED ({:05d}s) ***'.format(dt))
    print('***************************************************\n')

    # GENERATE TENSOR

    print('**************************************************************')
    print('*** DTIQA V7: GENERATING TENSORS AND RECONSTRUCTING SIGNAL ***')
    print('**************************************************************')

    ti = time.time()

    tensor_dir = utils.make_dir(out_dir, 'TENSOR')

    tensor_file, dwi_recon_file = preproc.tensor(dwi_preproc_file, bvals_preproc_file, bvecs_preproc_file, mask_file, tensor_dir)

    tf = time.time()
    dt = round(tf - ti)

    print('*********************************************************************')
    print('*** DTIQA V7: TENSORS GENERATED AND SIGNAL RECONSTRUCTED ({:05d}s) ***'.format(dt))
    print('*********************************************************************\n')

    # GENERATE SCALAR MAPS

    print('************************************')
    print('*** DTIQA V7: GENERATING SCALARS ***')
    print('************************************')

    ti = time.time()

    scalars_dir = utils.make_dir(out_dir, 'SCALARS')

    fa_file, md_file = preproc.scalars(tensor_file, mask_file, scalars_dir)

    tf = time.time()
    dt = round(tf - ti)

    print('********************************************')
    print('*** DTIQA V7: SCALARS GENERATED ({:05d}s) ***'.format(dt))
    print('********************************************\n')

    # CALCULATE STATS

    print('*************************************************')
    print('*** DTIQA V7: PERFORMING STATISTICAL ANALYSES ***')
    print('*************************************************')

    ti = time.time()

    stats_dir = utils.make_dir(out_dir, 'STATS')

    chisq_mask_file = stats.chisq_mask(dwi_preproc_file, bvals_preproc_file, mask_file, stats_dir)
    chisq_matrix_file = stats.chisq(dwi_preproc_file, dwi_recon_file, chisq_mask_file, stats_dir)

    motion_dict, motion_stats_out_list = stats.motion(eddy_dir, stats_dir)
    cnr_dict, bvals_preproc_shelled, cnr_stats_out_list, cnr_warning_str = stats.cnr(dwi_preproc_file, bvals_preproc_file, mask_file, eddy_dir, stats_dir, shells=params['shells'])
    if not cnr_warning_str == '':
        warning_strs.append(cnr_warning_str)
    
    roi_names, roi_avg_fa, atlas_ants_fa_file, cc_center_voxel, fa_stats_out_list = stats.fa_info(fa_file, stats_dir)

    stats.stats_out(motion_stats_out_list, cnr_stats_out_list, fa_stats_out_list, stats_dir)

    opt_dir = utils.make_dir(out_dir, 'OPTIMIZED_BVECS')

    bvals_corrected_file, bvecs_corrected_file = stats.gradcheck(dwi_preproc_file, bvals_preproc_file, bvecs_preproc_file, mask_file, opt_dir)

    tf = time.time()
    dt = round(tf - ti)

    print('********************************************************')
    print('*** DTIQA V7: FINISHED STATISTICAL ANALYSES ({:05d}s) ***'.format(dt))
    print('********************************************************\n')

    # GENERATE PDF

    print('******************************')
    print('*** DTIQA V7: CREATING PDF ***')
    print('******************************')

    ti = time.time()

    vis_dir = utils.make_dir(out_dir, 'PDF')

    # Generate component PDFs

    title_vis_file = vis.vis_title(dwi_files, t1_file, pe_axis, pe_dirs, readout_times, use_topup, use_synb0, params, warning_strs, vis_dir)
    pedir_vis_file = vis.vis_pedir(dwi_checked_files, bvals_checked_files, pe_axis, pe_dirs, vis_dir)
    if use_synb0:
        synb0_vis_file = vis.vis_synb0(b0_d_file, t1_file, b0_syn_file, vis_dir)
    if params['use_prenormalize']:
        prenorm_label = 'Prenormalization'
    else:
        prenorm_label = 'Gain QA of {}Inputs'.format('Denoised ' if params['use_denoise'] else '')
    prenorm_vis_file = vis.vis_norm(dwi_denoised_files, dwi_prenorm_files, dwi_prenorm_gains, dwi_prenorm_bins, dwi_denoised_hists, dwi_prenormed_hists, prenorm_label, vis_dir)
    prenorm_vis_file = utils.rename_file(prenorm_vis_file, os.path.join(vis_dir, 'prenorm.pdf'))
    if params['use_postnormalize']:
        norm_vis_file = vis.vis_norm(dwi_eddy_files, dwi_norm_files, dwi_norm_gains, dwi_norm_bins, dwi_eddy_hists, dwi_normed_hists, 'Postnormalization', vis_dir)
    preproc_vis_file = vis.vis_preproc(dwi_checked_files, bvals_checked_files, dwi_preproc_file, bvals_preproc_file, eddy_mask_file, mask_file, chisq_mask_file, vis_dir)
    stats_vis_file = vis.vis_stats(dwi_preproc_file, bvals_preproc_file, mask_file, chisq_matrix_file, motion_dict, eddy_dir, vis_dir)
    gradcheck_vis_file = vis.vis_gradcheck(bvals_checked_files, bvecs_checked_files, bvals_preproc_file, bvecs_preproc_file, bvals_corrected_file, bvecs_corrected_file, vis_dir)
    if params['use_unbias']:
        bias_vis_file = vis.vis_bias(dwi_norm_file, bvals_norm_file, dwi_unbiased_file, bias_field_file, vis_dir)
    dwi_vis_files = vis.vis_dwi(dwi_preproc_file, bvals_preproc_shelled, bvecs_preproc_file, cnr_dict, vis_dir)
    tensor_vis_file = vis.vis_tensor(tensor_file, fa_file, cc_center_voxel, vis_dir)
    fa_vis_file = vis.vis_scalar(fa_file, vis_dir, name='FA')
    fa_stats_vis_file = vis.vis_fa_stats(roi_names, roi_avg_fa, fa_file, atlas_ants_fa_file, vis_dir)
    md_vis_file = vis.vis_scalar(md_file, vis_dir, name='MD')

    # Combine component PDFs

    vis_files = []
    vis_files.append(title_vis_file)
    vis_files.append(pedir_vis_file)
    vis_files.append(prenorm_vis_file) # prenorm_vis_file = prenorm or gain check histograms
    if params['use_postnormalize']:
        vis_files.append(norm_vis_file)
    if use_synb0:
        vis_files.append(synb0_vis_file)
    vis_files.append(preproc_vis_file)
    vis_files.append(stats_vis_file)
    if params['use_unbias']:
        vis_files.append(bias_vis_file)
    vis_files.append(gradcheck_vis_file)
    for dwi_vis_file in dwi_vis_files:
        vis_files.append(dwi_vis_file)
    vis_files.append(tensor_vis_file)
    vis_files.append(fa_vis_file)
    vis_files.append(fa_stats_vis_file)
    vis_files.append(md_vis_file)

    pdf_file = utils.merge_pdfs(vis_files, 'dtiQA', vis_dir)

    tf = time.time()
    dt = round(tf - ti)

    print('************************************')
    print('*** DTIQA V7: PDF SAVED ({:05d}s) ***'.format(dt))
    print('************************************\n')

    # SPLIT OUTPUTS IF DESIRED

    print('***********************************')
    print('*** DTIQA V7: SPLITTING OUTPUTS ***')
    print('***********************************')

    ti = time.time()

    if params['split_outputs']:

        print('SPLITTING OUTPUTS')

        preproc_volume_prefixes = utils.dwi_volume_prefixes(dwi_checked_files)
        for i in range(len(preproc_volume_prefixes)):
            preproc_volume_prefixes[i] = preproc_volume_prefixes[i].replace('_checked', '_preproc')

        dwi_preproc_files = utils.dwi_split(dwi_preproc_file, preproc_volume_prefixes, preproc_dir)
        bvals_preproc_files = utils.bvals_split(bvals_preproc_file, preproc_volume_prefixes, preproc_dir)
        bvecs_preproc_files = utils.bvecs_split(bvecs_preproc_file, preproc_volume_prefixes, preproc_dir)

    else:

        print('NOT SPLITTING OUTPUTS')

    tf = time.time()
    dt = round(tf - ti)

    print('*****************************************************')
    print('*** DTIQA V7: FINISHED SPLITTING OUTPUTS ({:05d}s) ***'.format(dt))
    print('*****************************************************\n')

    # REMOVE UNNECESSARY OUTPUTS

    print('********************************')
    print('*** DTIQA V7: FINAL CLEAN UP ***')
    print('********************************')

    ti = time.time()

    # Always clear these intermediates:

    if params['eddy_mask_type'] == 'volume':
        print('CLEARING EDDY PSEUDO-MASK')
        utils.remove_file(eddy_mask_file)

    # Clear these intermediates if user desires:

    if not params['keep_intermediates']:

        print('CLEARING CHECKED DATA')
        utils.remove_dir(check_dir)

        print('CLEARING DENOISED DATA')
        utils.remove_dir(denoised_dir)

        print('CLEARING {} DATA'.format('PRENORMALIZED' if params['use_prenormalize'] else 'GAIN CHECK'))
        utils.remove_dir(prenorm_dir)

        if use_synb0:
            print('CLEARING SYNB0 INPUTS')
            utils.remove_file(b0_d_file)

        if use_topup:
            print('CLEARING TOPUP INPUTS')
            utils.remove_file(topup_input_b0s_file)
            print('CLEARING TOPUP OUTPUTS')
            utils.remove_file(topup_output_b0s_file)

        print('CLEARING EDDY INPUTS')
        utils.remove_file(eddy_input_dwi_file)
        utils.remove_file(eddy_input_bvals_file)
        utils.remove_file(eddy_input_bvecs_file)

        print('CLEARING EDDY OUTPUTS')
        utils.remove_file(eddy_output_dwi_file)
        utils.remove_file(eddy_output_bvals_file)
        utils.remove_file(eddy_output_bvecs_file)

        print('CLEARING POSTNORMALIZED DATA')
        utils.remove_dir(norm_dir)

        print('CLEARING UNBIASED DATA')
        if params['use_unbias']:
            utils.remove_file(dwi_unbiased_file)
        else:
            utils.remove_dir(unbias_dir)

        print('CLEARING TENSOR-RECONSTRUCTED SIGNAL')
        utils.remove_file(dwi_recon_file)

        if params['split_outputs']:
            print('OUTPUTS HAVE BEEN SPLIT, CLEARING MERGED OUTPUT')
            utils.remove_file(dwi_preproc_file)
            utils.remove_file(bvals_preproc_file)
            utils.remove_file(bvecs_preproc_file)

    tf = time.time()
    dt = round(tf - ti)

    print('*************************************')
    print('*** DTIQA V7: ALL CLEAN! ({:05d}s) ***'.format(dt))
    print('*************************************\n')

    # FINISH UP

    Tf = time.time()
    dT = round(Tf - Ti)

    print('*********************************************')
    print('*** DTIQA V7: PIPELINE FINISHED ({:06d}s) ***'.format(dT))
    print('*********************************************\n')

    return pdf_file

if __name__ == '__main__':
    main()