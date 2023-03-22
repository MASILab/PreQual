# PreQual: Preprocessing
# Leon Cai, Qi Yang, and Praitayini Kanakaraj
# MASI Lab
# Vanderbilt University

# Set Up

import sys
import os

import nibabel as nib
import numpy as np

import utils
from vars import SHARED_VARS

# Define Preprocessing Functions

def prep(dwi_files, bvals_files, bvecs_files, pe_axis, pe_dirs, readout_times, use_topup, use_synb0, t1_file, t1_stripped, topup_dir, eddy_dir, eddy_bval_scale, topup_first_b0s_only):

    print('PREPARING FOR PREPROCESSING...')

    # Prepare eddy inputs: merge into one file
    
    eddy_input_prefix = 'preproc_input'
    eddy_input_dwi_file = utils.dwi_merge(dwi_files, eddy_input_prefix, eddy_dir)
    eddy_input_bvals_file = utils.bvals_merge(bvals_files, eddy_input_prefix, eddy_dir)
    if not eddy_bval_scale == 1: # If we need to scale b-values for eddy, do so here
        eddy_input_bvals_file = utils.bvals_scale(eddy_input_bvals_file, float(eddy_bval_scale), eddy_dir)
    eddy_input_bvecs_file = utils.bvecs_merge(bvecs_files, eddy_input_prefix, eddy_dir)

    # Set up data for acqparams and eddy_index files

    num_b0s = []
    num_vols = []
    for i in range(len(bvals_files)):
        bvals = utils.load_txt(bvals_files[i], txt_type='bvals')
        if topup_first_b0s_only: # if not using all b0s, use first b0 from each dwi file
            num_b0s.append(1)
        else: # if using all b0s, get an accurate count for each input dwi file 
            num_b0s.append(np.sum(bvals == 0))
        num_vols.append(len(bvals))

    # Prepare for topup if applicable

    topup_input_b0s_file = ''
    b0_d_file = ''
    b0_syn_file = ''

    if use_topup:

        # Extract b0s for topup

        if topup_first_b0s_only:

            print('EXTRACTING FIRST B0 FROM EACH DWI INPUT FOR TOPUP')
            temp_dir = utils.make_dir(topup_dir, 'TEMP')
            topup_input_b0s_files = []
            for i in range(len(dwi_files)):
                b0_file, _, _ = utils.dwi_extract(dwi_files[i], bvals_files[i], temp_dir, target_bval=0, first_only=True)
                topup_input_b0s_files.append(b0_file)
            topup_input_b0s_file = utils.dwi_merge(topup_input_b0s_files, '{}_first_b0s_only'.format(eddy_input_prefix), topup_dir)
            utils.remove_dir(temp_dir)

        else:

            print('EXTRACTING ALL B0S FROM DWI INPUT FOR TOPUP')
            topup_input_b0s_file, _, _ = utils.dwi_extract(eddy_input_dwi_file, eddy_input_bvals_file, topup_dir, target_bval=0, first_only=False)

        # Run synb0 if needed

        if use_synb0:

            # Run synb0 on first b0 of first DWI 4D sequence

            b0_d_file, _, _ = utils.dwi_extract(eddy_input_dwi_file, eddy_input_bvals_file, topup_dir, target_bval=0, first_only=True)
            b0_syn_file = synb0(b0_d_file, t1_file, topup_dir, stripped=t1_stripped)

            # Smooth input b0s file as per Justin's optimization of synb0

            topup_input_b0s_smooth_file = utils.dwi_smooth(topup_input_b0s_file, topup_dir)
            utils.remove_file(topup_input_b0s_file)
            topup_input_b0s_file = topup_input_b0s_smooth_file
            
            # Merge all raw b0s with b0_syn (no normalization needed, because synb0 takes intensity into account) and overwrite all b0s file

            topup_input_b0s_prefix = utils.get_prefix(topup_input_b0s_file)
            topup_input_b0s_with_syn_file = utils.dwi_merge([topup_input_b0s_file, b0_syn_file], '{}_with_b0_syn'.format(topup_input_b0s_prefix), topup_dir)
            utils.remove_file(topup_input_b0s_file)
            topup_input_b0s_file = topup_input_b0s_with_syn_file

            # Performed synb0 on first b0 of first dwi 4D volume in sequence: update pe_dir and readout time accordingly

            pe_dirs.append(pe_dirs[0])
            readout_times.append(0)

            num_b0s.append(1) # will be incorporated into acqparams
            num_vols.append(0) # will not be incorporated into eddy index
    
    # Write acquisition parameters file to topup (if applicable) and eddy directory: define a phase-encoding scheme for each b0

    print('GENERATING ACQPARAMS FILES')

    acqparams_list = []
    for i in range(len(pe_dirs)):
        for j in range(num_b0s[i]):
            acqparams_list.append(utils.pescheme2params(pe_axis, pe_dirs[i], readout_times[i]))
    acqparams_str = '\n'.join(acqparams_list)
    
    eddy_acqparams_file = os.path.join(eddy_dir, 'acqparams.txt')
    utils.write_str(acqparams_str, eddy_acqparams_file)

    if use_topup:

        topup_acqparams_file = os.path.join(topup_dir, 'acqparams.txt')
        utils.write_str(acqparams_str, topup_acqparams_file)

    else:

        topup_acqparams_file = ''
    
    # Write eddy index file: need to associate each volume in an image with the line in the acqparams file corresponding to the first b0 of that image

    print('GENERATING EDDY INDEX FILES')

    eddy_index_list = []

    total_num_b0s = 1
    for i in range(len(num_vols)):
        num_b0 = num_b0s[i]
        num_vol = num_vols[i]
        for i in range(num_vol):
            eddy_index_list.append(str(total_num_b0s))
        total_num_b0s += num_b0
    eddy_index_str = ' '.join(eddy_index_list)

    eddy_index_file = os.path.join(eddy_dir, 'index.txt')
    utils.write_str(eddy_index_str, eddy_index_file)

    print('FINISHED PREPROCESSING PREPARATIONS')

    return topup_input_b0s_file, topup_acqparams_file, b0_d_file, b0_syn_file, eddy_input_dwi_file, eddy_input_bvals_file, eddy_input_bvecs_file, eddy_acqparams_file, eddy_index_file

def synb0(b0_d_file, t1_file, synb0_dir, stripped=False):

    print('RUNNING SYNB0 ON A 3D DISTORTED B0 AND A {}T1...'.format('SKULL STRIPPED ' if stripped else ''))

    temp_dir = utils.make_dir(synb0_dir, 'TEMP')

    mni_fname = 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii.gz' if stripped else 'mni_icbm152_t1_tal_nlin_asym_09c.nii.gz'

    synb0_cmd = 'export OMP_NUM_THREADS={} ; bash {} {} {} {} {}'.format(SHARED_VARS.NUM_THREADS, SHARED_VARS.SYNB0_EXEC_FILE, b0_d_file, t1_file, temp_dir, mni_fname)
    utils.run_cmd(synb0_cmd)

    print('MOVE OUTPUT TO B0_SYN.NII.GZ')

    b0_u_file = os.path.join(temp_dir, 'b0_u.nii.gz') # original output
    b0_syn_file = os.path.join(synb0_dir, 'b0_syn.nii.gz') # renamed output
    utils.move_file(b0_u_file, b0_syn_file)

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    print('SYNB0 FINISHED')

    return b0_syn_file

def gradtensor(gradtensor_file, dwi_preproc_file, bvecs_preproc_file, bvals_preproc_file, grad_nonlinear_dir,num_cores):

    print('RUNNING GRADNONLIEARITY CORRECTION WITH {} ...'.format(gradtensor_file))

    gradtensor_cmd = 'bash {} {} {} {} {} {} {}'.format(SHARED_VARS.GRADTENSOR_EXEC_FILE, gradtensor_file, dwi_preproc_file, bvecs_preproc_file, bvals_preproc_file, grad_nonlinear_dir, num_cores)
    utils.run_cmd(gradtensor_cmd)

    dwi_grad_corrected_file = os.path.join(grad_nonlinear_dir, 'Lcorrected_sig.nii.gz') # original output

    return dwi_grad_corrected_file

def topup(topup_input_b0s_file, topup_acqparams_file, extra_topup_args, topup_dir):

    print('TOPPING UP...')

    # Prepare outputs

    topup_output_b0s_file = os.path.join(topup_dir, '{}_topped_up.nii.gz'.format(utils.get_prefix(topup_input_b0s_file)))
    topup_results_prefix = os.path.join(topup_dir, 'topup_results')
    topup_output_field_file = os.path.join(topup_dir, 'topup_field.nii.gz')

    # Run topup on all b0s

    topup_cmd = 'topup --verbose --imain={} --datain={} --config={} --iout={} --out={} --fout={} {}'.format(
        topup_input_b0s_file, topup_acqparams_file, SHARED_VARS.TOPUP_CNF_FILE, topup_output_b0s_file, topup_results_prefix, topup_output_field_file, extra_topup_args)
    utils.run_cmd(topup_cmd)

    print('FINISHED TOPPING UP')

    return topup_results_prefix, topup_output_b0s_file

def eddy(eddy_input_dwi_file, eddy_input_bvals_file, eddy_input_bvecs_file, eddy_acqparams_file, eddy_index_file, eddy_mask_type, eddy_cuda_version, eddy_bval_scale, topup_results_prefix, topup_output_b0s_file, extra_eddy_args, eddy_dir):

    print('RUNNING EDDY...')

    print('CREATING A {} MASK FOR EDDY'.format(eddy_mask_type)) # name it eddy_mask.nii.gz in the EDDY directory

    temp_dir = utils.make_dir(eddy_dir, 'TEMP')

    if not topup_output_b0s_file == '': # ran topup
        topup_output_b0s_avg_file = utils.dwi_avg(topup_output_b0s_file, temp_dir)
        eddy_mask_file = utils.dwi_mask(topup_output_b0s_avg_file, temp_dir)
    else: # no topup
        eddy_mask_file = mask(eddy_input_dwi_file, eddy_input_bvals_file, eddy_mask_type, temp_dir)

    eddy_mask_renamed_file = os.path.join(eddy_dir, 'eddy_mask.nii.gz')
    utils.move_file(eddy_mask_file, eddy_mask_renamed_file)
    eddy_mask_file = eddy_mask_renamed_file

    utils.remove_dir(temp_dir)

    # Determine version of eddy to run

    if eddy_cuda_version == 8.0:
        eddy_exec = 'eddy_cuda8.0'
    elif eddy_cuda_version == 9.1:
        eddy_exec = 'eddy_cuda9.1'
    else:
        eddy_exec = 'eddy_openmp'

    # Set outputs

    eddy_results_prefix = os.path.join(eddy_dir, 'eddy_results')

    topup_options = ''
    if topup_results_prefix != '':
        topup_options = '--topup={}'.format(topup_results_prefix)

    # Run eddy and rerun if data not shelled

    eddy_cmd = 'export OMP_NUM_THREADS={} ; {} --verbose --imain={} --bvals={} --bvecs={} --mask={} --acqp={} --index={} --repol --cnr_maps --out={} {} {}'.format(
        SHARED_VARS.NUM_THREADS, eddy_exec, 
        eddy_input_dwi_file, eddy_input_bvals_file, eddy_input_bvecs_file, eddy_mask_file, 
        eddy_acqparams_file, eddy_index_file, 
        eddy_results_prefix, 
        topup_options, 
        extra_eddy_args
    )

    try:
        dsi_found = False
        utils.run_cmd(eddy_cmd)
    except:
        print('EDDY ENCOUNTERED AN ERROR, POSSIBLY BECAUSE EDDY DETECTED NON-SHELLED DATA. NOW ATTEMPTING TO FORCE EDDY TO RUN ON NON-SHELLED DATA.')
        dsi_found = True
        eddy_force_cmd = '{} --data_is_shelled'.format(eddy_cmd)
        utils.run_cmd(eddy_force_cmd)

    # Rename outputs

    eddy_input_prefix = utils.get_prefix(eddy_input_dwi_file)
    eddy_output_dwi_file = os.path.join(eddy_dir, '{}_eddyed.nii.gz'.format(eddy_input_prefix))
    eddy_output_bvals_file = os.path.join(eddy_dir, '{}_eddyed.bval'.format(eddy_input_prefix))
    eddy_output_bvecs_file = os.path.join(eddy_dir, '{}_eddyed.bvec'.format(eddy_input_prefix))

    utils.move_file('{}.nii.gz'.format(eddy_results_prefix), eddy_output_dwi_file)
    utils.copy_file(eddy_input_bvals_file, eddy_output_bvals_file)
    if not eddy_bval_scale == 1: # If we need to undo scaling b-values for eddy, do so here
        eddy_output_bvals_file = utils.bvals_scale(eddy_output_bvals_file, float(1)/float(eddy_bval_scale), eddy_dir)
    utils.copy_file('{}.eddy_rotated_bvecs'.format(eddy_results_prefix), eddy_output_bvecs_file)

    # Document warnings

    eddy_warning_str = ''
    if dsi_found:
        eddy_warning_str = 'Eddy encountered an error while running and successfully reran when forced to run on non-shelled data, suggesting the input data was a non-shelled (i.e. DSI) image. Please note that eddy does not currently support DSI data and may produce spurious results. Please verify your data is shelled or that this behavior is expected.'

    print('FINISHED EDDY')

    return eddy_output_dwi_file, eddy_output_bvals_file, eddy_output_bvecs_file, eddy_mask_file, eddy_warning_str

def mask(dwi_file, bvals_file, mask_type, mask_dir):

    dwi_prefix = utils.get_prefix(dwi_file, file_ext='nii')

    print('GENERATING MASK FOR {}...'.format(dwi_prefix))

    mask_file = os.path.join(mask_dir, '{}_mask.nii.gz'.format(dwi_prefix))

    if mask_type == 'brain':

        print('CALCULATING BRAIN MASK ON AVERAGED B0S')

        temp_dir = utils.make_dir(mask_dir, 'TEMP')

        b0s_file, _, _ = utils.dwi_extract(dwi_file, bvals_file, temp_dir, target_bval=0, first_only=False)
        b0s_avg_file = utils.dwi_avg(b0s_file, temp_dir)
        b0s_mask_file = utils.dwi_mask(b0s_avg_file, temp_dir)

        utils.move_file(b0s_mask_file, mask_file)

        print('CLEANING UP')

        utils.remove_dir(temp_dir)

    elif mask_type == 'volume':

        print('RETURNING MASK OF FULL VOLUME')

        dwi_img, dwi_aff, _ = utils.load_nii(dwi_file, ndim=4)
        mask_img = np.ones(dwi_img.shape[:-1]).astype('int')
        utils.save_nii(mask_img, dwi_aff, mask_file)

    else:

        raise utils.DTIQAError('INVALID MASK TYPE \"{}\" SPECIFIED!'.format(mask_type))

    print('FINISHED GENERATING MASK FOR {}'.format(dwi_prefix))

    return mask_file

def tensor(dwi_file, bvals_file, bvecs_file, mask_file, tensor_dir):

    dwi_prefix = utils.get_prefix(dwi_file)

    print('CONVERTING {} TO TENSOR WITH RECONSTRUCTED SIGNAL...'.format(dwi_prefix))

    tensor_file = os.path.join(tensor_dir, '{}_tensor.nii.gz'.format(dwi_prefix))  # make tensor dwi2tensor then use that for fa tensor2metric, volumes 0-5: D11, D22, D33, D12, D13, D23
    dwi_recon_file = os.path.join(tensor_dir, '{}_recon.nii.gz'.format(dwi_prefix))
    tensor_cmd = 'dwi2tensor {} {} -fslgrad {} {} -mask {} -predicted_signal {} -force -nthreads {}'.format(dwi_file, tensor_file, bvecs_file, bvals_file, mask_file, dwi_recon_file, SHARED_VARS.NUM_THREADS-1)
    utils.run_cmd(tensor_cmd)

    print('FINISHED CONVERTING {} TO TENSOR WITH RECONSTRUCTED SIGNAL'.format(dwi_prefix))

    return tensor_file, dwi_recon_file

def scalars(tensor_file, mask_file, scalars_dir):

    tensor_prefix = utils.get_prefix(tensor_file)

    print('CONVERTING {} TENSOR TO SCALARS...'.format(tensor_prefix))

    print('CONVERTING TENSOR TO FA...')

    fa_file = os.path.join(scalars_dir, '{}_fa.nii.gz'.format(tensor_prefix))
    fa_cmd = 'tensor2metric {} -fa {} -mask {} -force -nthreads {}'.format(tensor_file, fa_file, mask_file, SHARED_VARS.NUM_THREADS-1)
    utils.run_cmd(fa_cmd)

    print('CONVERTING TENSOR TO MD...')

    md_file = os.path.join(scalars_dir, '{}_md.nii.gz'.format(tensor_prefix))
    md_cmd = 'tensor2metric {} -adc {} -mask {} -force -nthreads {}'.format(tensor_file, md_file, mask_file, SHARED_VARS.NUM_THREADS-1)
    utils.run_cmd(md_cmd)

    print('CONVERTING TENSOR TO AD...')

    ad_file = os.path.join(scalars_dir, '{}_ad.nii.gz'.format(tensor_prefix))
    ad_cmd = 'tensor2metric {} -ad {} -mask {} -force -nthreads {}'.format(tensor_file, ad_file, mask_file, SHARED_VARS.NUM_THREADS - 1)
    utils.run_cmd(ad_cmd)

    print('CONVERTING TENSOR TO RD...')

    rd_file = os.path.join(scalars_dir, '{}_rd.nii.gz'.format(tensor_prefix))
    rd_cmd = 'tensor2metric {} -rd {} -mask {} -force -nthreads {}'.format(tensor_file, rd_file, mask_file, SHARED_VARS.NUM_THREADS - 1)
    utils.run_cmd(rd_cmd)

    print('CONVERTING TENSOR TO V1...')

    v1_file = os.path.join(scalars_dir, '{}_v1.nii.gz'.format(tensor_prefix))
    v1_cmd = 'tensor2metric {} -vector {} -num 1 -modulate none -mask {} -force -nthreads {}'.format(tensor_file, v1_file, mask_file, SHARED_VARS.NUM_THREADS - 1)
    utils.run_cmd(v1_cmd)

    print('ALL SCALARS GENERATED')

    return fa_file, md_file, ad_file, rd_file, v1_file
