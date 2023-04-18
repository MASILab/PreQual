# PreQual: Visualization
# Leon Cai, Qi Yang, and Praitayini Kanakaraj
# MASI Lab
# Vanderbilt University

# Set Up

import os
from datetime import datetime
from io import StringIO

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
#from mpl_toolkits.mplot3d import Axes3D

import utils
from vars import SHARED_VARS

# Define Visualization Functions

def vis_title(dwi_files, t1_file, pe_axis, pe_dirs, readout_times, use_topup, use_synb0, params, warning_strs, vis_dir):

    print('RENDERING TITLE PAGE')

    title_str = \
    str('PreQual v{} (dtiQA v7 Multi)\n'
        'Creation Date: {}\n\n'

        'Authors:\n'
        '- Leon Cai (leon.y.cai@vanderbilt.edu)\n'
        '- Qi Yang (qi.yang@vanderbilt.edu)\n'
        '- Praitayini Kanakaraj (praitayini.kanakaraj@vanderbilt.edu)\n\n'

        'Run Date: {}\n\n'

        'Project: {}\n'
        'Subject: {}\n'
        'Session: {}\n'
        .format(SHARED_VARS.VERSION, SHARED_VARS.CREATION_DATE, datetime.now(), params['project'], params['subject'], params['session']))

    params_str = \
    str('Parameters:\n'
        '- B-Value Threshold: {}\n'
        '- Shells: {}\n'
        '- Run Denoise: {}\n'
        '- Run Degibbs: {}\n'
        '- Run Rician: {}\n'
        '- Run Prenormalize: {}\n'
        '- Topup B0s: {}\n'
        '- Try Synb0-DisCo: {}\n'
        '- Extra Topup Args: {}\n'
        '- Eddy Mask: {}\n'
        '- Eddy B-Value Scale: {}\n'
        '- Extra Eddy Args: {}\n'
        '- Run Postnormalize: {}\n'
        '- Run N4 Bias Field Correction: {}\n'
        '- Run Gradient Nonlinearity Correction: {}\n'
        '- Mask Improbable Voxels: {}\n'
        '- Glyph Visualization Type: {}\n'
        '- Atlas Registration Type: {}\n'
        '- Split Outputs: {}\n'
        '- Keep Intermediates: {}\n'
        .format(params['bval_threshold'],
                params['shells'] if len(params['shells']) > 0 else 'Auto',
                params['use_denoise'],
                params['use_degibbs'],
                params['use_rician'],
                params['use_prenormalize'],
                'First' if params['topup_first_b0s_only'] else 'All',
                params['use_synb0_user'],
                params['extra_topup_args'],
                True if params['eddy_mask_type'] == 'brain' else False,
                params['eddy_bval_scale'],
                params['extra_eddy_args'],
                params['use_postnormalize'],
                params['use_unbias'],
                params['use_grad'],
                params['improbable_mask'],
                params['glyph_type'],
                params['atlas_reg_type'],
                params['split_outputs'],
                params['keep_intermediates']))

    inputs_str = 'Preprocessing: {}\nInputs (w/ PE direction and readout time):\n'.format('Eddy only' if not use_topup else 'Topup (RPE) + Eddy' if not use_synb0 else 'Topup (Synb0) + Eddy')
    for i in range(len(dwi_files)):
        inputs_str = '{}- {} ({}{}, {})\n'.format(inputs_str, utils.get_prefix(dwi_files[i]), pe_axis, pe_dirs[i], readout_times[i])
    if os.path.exists(t1_file):
        inputs_str = '{}- {} ({})'.format(inputs_str, utils.get_prefix(t1_file), 'unused' if not use_synb0 else 'skull-stripped' if params['t1_stripped'] else 'raw')

    method_strs, ref_strs = _methods_strs(use_topup, use_synb0, params)

    methods_str = \
    str('Warnings (See \"PreQual User Guide\" at github.com/MASILab/PreQual for more information):\n\n'
        
        '- All input volumes must have the same phase encoding axis, as input into this pipeline and as reflected above. Please see the PE Direction page of this PDF for more information.\n'
        '{}'

        'Methods Summary:\n\n'

        '{}\n\n'

        'References:\n\n'

        '{}'
        
        .format('- {}\n\n'.format('\n- '.join(warning_strs)) if len(warning_strs) > 0 else '\n',
                ' '.join(method_strs),
                '\n'.join(ref_strs)
               )
       )

    title_vis_file = os.path.join(vis_dir, 'title.pdf')

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)
    plt.axis([0, 1, 0, 1])
    plt.tight_layout()
    plt.text(-0.025, 1, title_str, ha='left', va='top', wrap=True, fontsize=6)
    plt.text(0.25, 1, params_str, ha='left', va='top', wrap=True, fontsize=6)
    plt.text(0.625, 1, inputs_str, ha='left', va='top', wrap=True, fontsize=6)
    plt.text(-0.025, 0, methods_str, ha='left', va='bottom', wrap=True, fontsize=6)
    plt.axis('off')
    plt.savefig(title_vis_file)
    plt.close()

    return title_vis_file

def vis_pedir(dwi_files, bvals_files, pe_axis, pe_dirs, vis_dir):

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)
    num_dwi = len(dwi_files)
    dwi_prefixes = []
    dwi_pe_strs = []
    for i in range(num_dwi):

        dwi_file = dwi_files[i]
        bvals_file = bvals_files[i]

        b0_file, _, _ = utils.dwi_extract(dwi_file, bvals_file, temp_dir, target_bval=0, first_only=True)
        _, b0_aff, _ = utils.load_nii(b0_file)
        b0_slices, b0_vox_dim, b0_min, b0_max = utils.slice_nii(b0_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)
        
        dwi_prefixes.append(utils.get_prefix(dwi_file))
        dwi_pe_strs.append(utils.pescheme2axis(pe_axis, pe_dirs[i], b0_aff))

        plt.subplot(4, num_dwi, i+1)
        utils.plot_slice(slices=b0_slices, img_dim=0, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
        if i == 0:
            plt.ylabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.title('{}'.format(i+1), fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.xlabel('P | A', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)

        plt.subplot(4, num_dwi, i+1+num_dwi)
        utils.plot_slice(slices=b0_slices, img_dim=1, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
        if i == 0:
            plt.ylabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.xlabel('R | L', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)

        plt.subplot(4, num_dwi, i+1+num_dwi*2)
        utils.plot_slice(slices=b0_slices, img_dim=2, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
        if i == 0:
            plt.ylabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.xlabel('R | L', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)

    plt.tight_layout()

    plt.subplots_adjust(top=0.925)
    plt.suptitle('PE Direction', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    lbl_strs = []
    for i in range(num_dwi):
        lbl_strs.append('{}) {} ({})'.format(i+1, dwi_prefixes[i], dwi_pe_strs[i]))
    lbl_str = '\n'.join(lbl_strs)
    lbl_ax = plt.subplot(4, 2, 7)
    plt.text(0, 0.5, lbl_str, ha='left', va='center', wrap=True, fontsize=SHARED_VARS.LABEL_FONTSIZE)
    lbl_ax.axis('off')

    pe_dir_info_str = 'The supplied phase encoding direction for the input images was \"{}\". Thus, the best interpreted anatomical axes for these images based on their affines are described to the left. Please ensure that these axes (direction agnostic) are interpreted to be the same and are visually distorted above for all images. It is an underlying assumption of this pipeline that all images be phase encoded on the same axis with varying direction as appropriate.'.format(pe_axis)
    txt_ax = plt.subplot(4, 2, 8)
    plt.text(0, 0.5, pe_dir_info_str, ha='left', va='center', wrap=True, fontsize=SHARED_VARS.LABEL_FONTSIZE)
    txt_ax.axis('off')

    pedir_vis_file = os.path.join(vis_dir, 'pedir.pdf')
    plt.savefig(pedir_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    return pedir_vis_file

def vis_synb0(b0_d_file, t1_file, b0_syn_file, vis_dir):

    print('VISUALIZING SYNB0-DISCO')

    b0_d_slices, b0_d_vox_dim, b0_d_min, b0_d_max = utils.slice_nii(b0_d_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)
    t1_slices, t1_vox_dim, t1_min, t1_max = utils.slice_nii(t1_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)
    b0_syn_slices, b0_syn_vox_dim, b0_syn_min, b0_syn_max = utils.slice_nii(b0_syn_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    plt.subplot(3, 3, 1)
    utils.plot_slice(slices=b0_d_slices, img_dim=0, offset_index=0, vox_dim=b0_d_vox_dim, img_min=b0_d_min, img_max=b0_d_max)
    plt.title('Distorted b0', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.ylabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 4)
    utils.plot_slice(slices=b0_d_slices, img_dim=1, offset_index=0, vox_dim=b0_d_vox_dim, img_min=b0_d_min, img_max=b0_d_max)
    plt.ylabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 7)
    utils.plot_slice(slices=b0_d_slices, img_dim=2, offset_index=0, vox_dim=b0_d_vox_dim, img_min=b0_d_min, img_max=b0_d_max)
    plt.ylabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 2)
    utils.plot_slice(slices=t1_slices, img_dim=0, offset_index=0, vox_dim=t1_vox_dim, img_min=t1_min, img_max=t1_max)
    plt.title('T1', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 5)
    utils.plot_slice(slices=t1_slices, img_dim=1, offset_index=0, vox_dim=t1_vox_dim, img_min=t1_min, img_max=t1_max)

    plt.subplot(3, 3, 8)
    utils.plot_slice(slices=t1_slices, img_dim=2, offset_index=0, vox_dim=t1_vox_dim, img_min=t1_min, img_max=t1_max)

    plt.subplot(3, 3, 3)
    utils.plot_slice(slices=b0_syn_slices, img_dim=0, offset_index=0, vox_dim=b0_syn_vox_dim, img_min=b0_syn_min, img_max=b0_syn_max)
    plt.title('Synthetic b0', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 6)
    utils.plot_slice(slices=b0_syn_slices, img_dim=1, offset_index=0, vox_dim=b0_syn_vox_dim, img_min=b0_syn_min, img_max=b0_syn_max)

    plt.subplot(3, 3, 9)
    utils.plot_slice(slices=b0_syn_slices, img_dim=2, offset_index=0, vox_dim=b0_syn_vox_dim, img_min=b0_syn_min, img_max=b0_syn_max)

    plt.tight_layout()

    plt.subplots_adjust(top=0.925)
    plt.suptitle('Synb0-DisCo', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    synb0_vis_file = os.path.join(vis_dir, 'synb0.pdf')
    plt.savefig(synb0_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    return synb0_vis_file

def vis_preproc(dwi_files, bvals_files, dwi_preproc_file, bvals_preproc_file, eddy_mask_file, mask_file, percent_improbable, stats_mask_file, vis_dir):

    print('VISUALIZING RAW AND PREPROCESSED DATA + MASKS')

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    eddy_mask_slices, _, _, _ = utils.slice_nii(eddy_mask_file)

    num_dwi = len(dwi_files)
    dwi_prefixes = []
    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    for i in range(num_dwi):

        dwi_file = dwi_files[i]
        bvals_file = bvals_files[i]

        dwi_prefixes.append(utils.get_prefix(dwi_file))

        b0_file, _, _ = utils.dwi_extract(dwi_file, bvals_file, temp_dir, target_bval=0, first_only=True)
        b0_slices, b0_vox_dim, b0_min, b0_max = utils.slice_nii(b0_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)

        plt.subplot(5, num_dwi+1, i+1)
        utils.plot_slice(slices=b0_slices, img_dim=0, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
        utils.plot_slice_contour(slices=eddy_mask_slices, img_dim=0, offset_index=0, color='r')
        if i == 0:
            plt.ylabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.title('{}'.format(i+1), fontsize=SHARED_VARS.LABEL_FONTSIZE)

        plt.subplot(5, num_dwi+1, i+1+num_dwi+1)
        utils.plot_slice(slices=b0_slices, img_dim=1, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
        utils.plot_slice_contour(slices=eddy_mask_slices, img_dim=1, offset_index=0, color='r')
        if i == 0:
            plt.ylabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

        plt.subplot(5, num_dwi+1, i+1+(num_dwi+1)*2)
        utils.plot_slice(slices=b0_slices, img_dim=2, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
        utils.plot_slice_contour(slices=eddy_mask_slices, img_dim=2, offset_index=0, color='r')
        if i == 0:
            plt.ylabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    b0_preproc_file, _, _ = utils.dwi_extract(dwi_preproc_file, bvals_preproc_file, temp_dir, target_bval=0, first_only=True)
    b0_preproc_slices, b0_preproc_vox_dim, b0_preproc_min, b0_preproc_max = utils.slice_nii(b0_preproc_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)

    mask_slices, _, _, _ = utils.slice_nii(mask_file)
    stats_mask_slices, _, _, _ = utils.slice_nii(stats_mask_file)

    plt.subplot(5, num_dwi+1, num_dwi+1)
    utils.plot_slice(slices=b0_preproc_slices, img_dim=0, offset_index=0, vox_dim=b0_preproc_vox_dim, img_min=b0_preproc_min, img_max=b0_preproc_max)
    utils.plot_slice_contour(slices=mask_slices, img_dim=0, offset_index=0, color='c')
    utils.plot_slice_contour(slices=stats_mask_slices, img_dim=0, offset_index=0, color='m')
    plt.title('Preprocessed', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(5, num_dwi+1, 2*(num_dwi+1))
    utils.plot_slice(slices=b0_preproc_slices, img_dim=1, offset_index=0, vox_dim=b0_preproc_vox_dim, img_min=b0_preproc_min, img_max=b0_preproc_max)
    utils.plot_slice_contour(slices=mask_slices, img_dim=1, offset_index=0, color='c')
    utils.plot_slice_contour(slices=stats_mask_slices, img_dim=1, offset_index=0, color='m')

    plt.subplot(5, num_dwi+1, 3*(num_dwi+1))
    utils.plot_slice(slices=b0_preproc_slices, img_dim=2, offset_index=0, vox_dim=b0_preproc_vox_dim, img_min=b0_preproc_min, img_max=b0_preproc_max)
    utils.plot_slice_contour(slices=mask_slices, img_dim=2, offset_index=0, color='c')
    utils.plot_slice_contour(slices=stats_mask_slices, img_dim=2, offset_index=0, color='m')

    plt.tight_layout()

    plt.subplots_adjust(top=0.925)
    plt.suptitle('Preprocessing and Masks', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    lbl_ax = plt.subplot(5, 2, 7)
    lbl_strs = []
    for i in range(num_dwi):
        lbl_strs.append('{}) {}'.format(i+1, dwi_prefixes[i]))
    lbl_str = '\n'.join(lbl_strs)
    plt.text(0, 0.5, lbl_str, ha='left', va='center', wrap=True, fontsize=SHARED_VARS.LABEL_FONTSIZE)
    lbl_ax.axis('off')

    legend_ax = plt.subplot(5, 2, 8)
    plt.plot([], linewidth=2, color='r', label='Eddy Mask')
    plt.plot([], linewidth=2, color='c', label='Preprocessed Mask ({:.2f}% Voxels Improbable)'.format(percent_improbable))
    plt.plot([], linewidth=2, color='m', label=r'$\chi^2$' ' Mask')
    plt.legend(bbox_to_anchor=(0.5, 0.5), loc='center', ncol=1)
    legend_ax.axis('off')

    txt_ax = plt.subplot(5, 1, 5)
    mask_info_str = '- Eddy Mask: If --eddy_mask is \"on\" (default), this is calculated on the averaged raw b0s if topup is not run and on the averaged topped-up b0s if it is. If --eddy_mask is \"off\", this mask is not applied and eddy is performed on the entire volume. This mask need only be approximate and is used for motion and eddy-current correction.\n- Preprocessed Mask: This is calculated on the preprocessed averaged b0s. It is used for tensor visualization and other analyses (i.e. SNR and CNR calculations, gradient table checks, etc.) and is the basis for the chi-squared mask.\n- Chi-Squared Mask: This is calculated from the preprocessed mask by subtracting the CSF and eroding the result. It is used to determine the voxels in which to perform the chi-squared analysis.'
    plt.text(0, 0.5, mask_info_str, ha='left', va='center', wrap=True, fontsize=SHARED_VARS.LABEL_FONTSIZE)
    txt_ax.axis('off')

    preproc_vis_file = os.path.join(vis_dir, 'preproc_masks.pdf')
    plt.savefig(preproc_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    return preproc_vis_file

def vis_norm(dwi_files, dwi_norm_files, gains, bins, hists, hists_normed, title, vis_dir):

    print('VISUALIZING NORMALIZATION')

    norm_vis_file = os.path.join(vis_dir, 'norm.pdf')

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    plt.subplot(2, 1, 1)
    for i in range(len(dwi_files)):
        dwi_prefix = utils.get_prefix(dwi_files[i])
        plt.plot(bins, hists[i], label=dwi_prefix)
    plt.legend()
    plt.title('Before', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.ylabel('Frequency', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.grid()

    plt.subplot(2, 1, 2)
    for i in range(len(dwi_norm_files)):
        dwi_norm_prefix = utils.get_prefix(dwi_norm_files[i])
        plt.plot(bins, hists_normed[i], label='{} (Gain Correction: {:.3f})'.format(dwi_norm_prefix, gains[i]))
    plt.legend()
    plt.title('After', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.xlabel('Intensity', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.ylabel('Frequency', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.grid()

    plt.suptitle('{}:\nAverage b0 Intensity Distributions By Scan Within Approximate Masks'.format(title), fontsize=SHARED_VARS.TITLE_FONTSIZE)

    plt.savefig(norm_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    return norm_vis_file

def vis_bias(dwi_file, bvals_file, dwi_unbiased_file, bias_field_file, vis_dir):

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    print('VISUALIZING N4 BIAS FIELD CORRECTION')

    b0_file, _, _ = utils.dwi_extract(dwi_file, bvals_file, temp_dir, target_bval=0, first_only=True)
    b0_unbiased_file, _, _ = utils.dwi_extract(dwi_unbiased_file, bvals_file, temp_dir, target_bval=0, first_only=True)

    b0_slices, b0_vox_dim, b0_min, b0_max = utils.slice_nii(b0_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)
    bias_field_slices, bias_field_vox_dim, bias_field_min, bias_field_max = utils.slice_nii(bias_field_file)
    b0_unbiased_slices, b0_unbiased_vox_dim, b0_unbiased_min, b0_unbiased_max = utils.slice_nii(b0_unbiased_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    plt.subplot(3, 3, 1)
    utils.plot_slice(slices=b0_slices, img_dim=0, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
    plt.colorbar()
    plt.title('Biased', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.ylabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 4)
    utils.plot_slice(slices=b0_slices, img_dim=1, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
    plt.colorbar()
    plt.ylabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 7)
    utils.plot_slice(slices=b0_slices, img_dim=2, offset_index=0, vox_dim=b0_vox_dim, img_min=b0_min, img_max=b0_max)
    plt.colorbar()
    plt.ylabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 2)
    utils.plot_slice(slices=bias_field_slices, img_dim=0, offset_index=0, vox_dim=bias_field_vox_dim, img_min=bias_field_min, img_max=bias_field_max)
    plt.colorbar()
    plt.title('Bias Field', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 5)
    utils.plot_slice(slices=bias_field_slices, img_dim=1, offset_index=0, vox_dim=bias_field_vox_dim, img_min=bias_field_min, img_max=bias_field_max)
    plt.colorbar()

    plt.subplot(3, 3, 8)
    utils.plot_slice(slices=bias_field_slices, img_dim=2, offset_index=0, vox_dim=bias_field_vox_dim, img_min=bias_field_min, img_max=bias_field_max)
    plt.colorbar()

    plt.subplot(3, 3, 3)
    utils.plot_slice(slices=b0_unbiased_slices, img_dim=0, offset_index=0, vox_dim=b0_unbiased_vox_dim, img_min=b0_unbiased_min, img_max=b0_unbiased_max)
    plt.colorbar()
    plt.title('Unbiased', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 6)
    utils.plot_slice(slices=b0_unbiased_slices, img_dim=1, offset_index=0, vox_dim=b0_unbiased_vox_dim, img_min=b0_unbiased_min, img_max=b0_unbiased_max)
    plt.colorbar()

    plt.subplot(3, 3, 9)
    utils.plot_slice(slices=b0_unbiased_slices, img_dim=2, offset_index=0, vox_dim=b0_unbiased_vox_dim, img_min=b0_unbiased_min, img_max=b0_unbiased_max)
    plt.colorbar()

    plt.tight_layout()

    plt.subplots_adjust(top=0.925)
    plt.suptitle('N4 Bias Field Correction', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    bias_vis_file = os.path.join(vis_dir, 'bias.pdf')
    plt.savefig(bias_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    utils.remove_dir(temp_dir)

    return bias_vis_file


def vis_grad(bvals_file, shells, dwi_corr_file, grad_field_file, fa_grad_field_file, mask_file, vis_dir):

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    print('VISUALIZING GRADIENT NONLINEAR FIELD CORRECTION')
    shells = np.unique(shells)
    print(shells)
    bval = shells[shells != 0][0]
    print(bval)
    b0_corr_file, _, _ = utils.dwi_extract(dwi_corr_file, bvals_file, temp_dir, target_bval=bval, first_only=True)
    b0_corr_slices, b0_corr_vox_dim, b0_corr_min, b0_corr_max = utils.slice_nii(b0_corr_file, min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)

    mask_slices, _, _, _ = utils.slice_nii(mask_file)
    grad_field_slices, grad_field_vox_dim, grad_field_min, grad_field_max = utils.slice_nii(grad_field_file, det=True)
    fa_grad_field_slices, fa_grad_field_vox_dim, fa_grad_field_min, fa_grad_field_max = utils.slice_nii(fa_grad_field_file)

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    plt.subplot(3, 3, 1)
    utils.plot_slice_contour(mask_slices, img_dim=0, offset_index=0, color='b')
    utils.plot_slice(slices=b0_corr_slices, img_dim=0, offset_index=0, vox_dim=b0_corr_vox_dim, img_min=b0_corr_min, img_max=b0_corr_max)
    plt.colorbar()
    plt.title('Corrected b = ' + str(bval), fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.ylabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 4)
    utils.plot_slice_contour(mask_slices, img_dim=1, offset_index=0, color='b')
    utils.plot_slice(slices=b0_corr_slices, img_dim=1, offset_index=0, vox_dim=b0_corr_vox_dim, img_min=b0_corr_min, img_max=b0_corr_max)
    plt.colorbar()
    plt.ylabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 7)
    utils.plot_slice_contour(mask_slices, img_dim=2, offset_index=0, color='b')
    utils.plot_slice(slices=b0_corr_slices, img_dim=2, offset_index=0, vox_dim=b0_corr_vox_dim, img_min=b0_corr_min, img_max=b0_corr_max)
    plt.colorbar()
    plt.ylabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 2)
    utils.plot_slice_contour(mask_slices, img_dim=0, offset_index=0, color='b')
    utils.plot_slice(slices=grad_field_slices, img_dim=0, offset_index=0, vox_dim=grad_field_vox_dim, img_min=grad_field_min, img_max=grad_field_max)
    plt.colorbar()
    plt.title('Det. of \n Gradient Nonlinear Field', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 5)
    utils.plot_slice_contour(mask_slices, img_dim=1, offset_index=0, color='b')
    utils.plot_slice(slices=grad_field_slices, img_dim=1, offset_index=0, vox_dim=grad_field_vox_dim, img_min=grad_field_min, img_max=grad_field_max)
    plt.colorbar()

    plt.subplot(3, 3, 8)
    utils.plot_slice_contour(mask_slices, img_dim=2, offset_index=0, color='b')
    utils.plot_slice(slices=grad_field_slices, img_dim=2, offset_index=0, vox_dim=grad_field_vox_dim, img_min=grad_field_min, img_max=grad_field_max)
    plt.colorbar()

    plt.subplot(3, 3, 3)
    utils.plot_slice(slices=fa_grad_field_slices, img_dim=0, offset_index=0, vox_dim=fa_grad_field_vox_dim, img_min=fa_grad_field_min, img_max=fa_grad_field_max)
    plt.colorbar()
    plt.title('FA of \n Gradient Nonlinear Field', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 3, 6)
    utils.plot_slice(slices=fa_grad_field_slices, img_dim=1, offset_index=0, vox_dim=fa_grad_field_vox_dim, img_min=fa_grad_field_min, img_max=fa_grad_field_max)
    plt.colorbar()

    plt.subplot(3, 3, 9)
    utils.plot_slice(slices=fa_grad_field_slices, img_dim=2, offset_index=0, vox_dim=fa_grad_field_vox_dim, img_min=fa_grad_field_min, img_max=fa_grad_field_max)
    plt.colorbar()

    plt.tight_layout()

    plt.subplots_adjust(top=0.925)
    plt.suptitle('Nonlinear Gradient Field Correction', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    grad_vis_file = os.path.join(vis_dir, 'grad.pdf')
    plt.savefig(grad_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    utils.remove_dir(temp_dir)

    return grad_vis_file


def vis_degibbs(dwi_files, bvals_files, dwi_degibbs_files, gains, vis_dir):

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    print('VISUALIZING DEGIBBS')

    # Scale all inputs by prenormalization gains

    dwi_scaled_files = []
    dwi_degibbs_scaled_files = []
    for i in range(len(dwi_files)):
        # Pregibbs
        dwi_prefix = utils.get_prefix(dwi_files[i], file_ext='nii')
        dwi_img, dwi_aff, _ = utils.load_nii(dwi_files[i])
        dwi_scaled_img = dwi_img * gains[i]
        dwi_scaled_file = os.path.join(temp_dir, '{}_scaled.nii.gz'.format(dwi_prefix))
        utils.save_nii(dwi_scaled_img, dwi_aff, dwi_scaled_file)
        dwi_scaled_files.append(dwi_scaled_file)
        # Postgibbs
        dwi_degibbs_prefix = utils.get_prefix(dwi_degibbs_files[i], file_ext='nii')
        dwi_degibbs_img, dwi_degibbs_aff, _ = utils.load_nii(dwi_degibbs_files[i])
        dwi_degibbs_scaled_img = dwi_degibbs_img * gains[i]
        dwi_degibbs_scaled_file = os.path.join(temp_dir, '{}_scaled.nii.gz'.format(dwi_degibbs_prefix))
        utils.save_nii(dwi_degibbs_scaled_img, dwi_degibbs_aff, dwi_degibbs_scaled_file)
        dwi_degibbs_scaled_files.append(dwi_degibbs_scaled_file)

    # Load common bvals

    gibbs_bval_file = utils.bvals_merge(bvals_files, 'gibbs', temp_dir)

    # Load pregibbs b0s, scaled by prenorm gains

    pregibbs_prefix = 'pregibbs_scaled'
    pregibbs_dwi_file = utils.dwi_merge(dwi_scaled_files, pregibbs_prefix, temp_dir)
    pregibbs_b0s_file, _, _ = utils.dwi_extract(pregibbs_dwi_file, gibbs_bval_file, temp_dir, target_bval=0, first_only=False)
    pregibbs_b0s_img, gibbs_b0s_aff, _ = utils.load_nii(pregibbs_b0s_file, ndim=4)

    # Load postgibbs b0s, scaled by prenorm gains

    postgibbs_prefix = 'postgibbs_scaled'
    postgibbs_dwi_file = utils.dwi_merge(dwi_degibbs_scaled_files, postgibbs_prefix, temp_dir)
    postgibbs_b0s_file, _, _ = utils.dwi_extract(postgibbs_dwi_file, gibbs_bval_file, temp_dir, target_bval=0, first_only=False)
    postgibbs_b0s_img, _, _ = utils.load_nii(postgibbs_b0s_file, ndim=4)

    # Calculate average absolute residuals

    res_img = np.nanmean(np.abs(postgibbs_b0s_img - pregibbs_b0s_img), axis=3)
    res_aff = gibbs_b0s_aff
    res_file = os.path.join(temp_dir, 'gibbs_residuals.nii.gz')
    utils.save_nii(res_img, res_aff, res_file, ndim=3)

    # Plot 5 central triplanar views

    res_slices, res_vox_dim, res_min, res_max = utils.slice_nii(res_file, offsets=[-10, -5, 0, 5, 10], min_intensity=0, max_percentile=99)
    temp_vis_file = _vis_vol(res_slices, res_vox_dim, res_min, res_max, temp_dir, name='Gibbs_Deringing,_Averaged_Residuals_of_b_=_0_Volumes', comment='Residuals should be larger at high-contrast interfaces', colorbar=False)
    degibbs_vis_file = utils.rename_file(temp_vis_file, os.path.join(vis_dir, 'degibbs.pdf'))

    # Finish Up

    utils.remove_dir(temp_dir)

    return degibbs_vis_file

def vis_rician(dwi_files, bvals_files, dwi_rician_files, gains, shells, vis_dir):

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    print('VISUALIZING RICIAN CORRECTION')

    # For each input...

    dwi_scaled_files = []
    dwi_rician_scaled_files = []
    bvals_shelled_files = []
    shells = np.unique(shells)
    for i in range(len(dwi_files)):
        # Prerician
        dwi_prefix = utils.get_prefix(dwi_files[i], file_ext='nii')
        dwi_img, dwi_aff, _ = utils.load_nii(dwi_files[i])
        dwi_scaled_img = dwi_img * gains[i] # ...scale by prenormalization gains...
        dwi_scaled_file = os.path.join(temp_dir, '{}_scaled.nii.gz'.format(dwi_prefix))
        utils.save_nii(dwi_scaled_img, dwi_aff, dwi_scaled_file)
        dwi_scaled_files.append(dwi_scaled_file)
        # Postrician
        dwi_rician_prefix = utils.get_prefix(dwi_rician_files[i], file_ext='nii')
        dwi_rician_img, dwi_rician_aff, _ = utils.load_nii(dwi_rician_files[i])
        dwi_rician_scaled_img = dwi_rician_img * gains[i] # ...scale by prenormalization gains...
        dwi_rician_scaled_file = os.path.join(temp_dir, '{}_scaled.nii.gz'.format(dwi_rician_prefix))
        utils.save_nii(dwi_rician_scaled_img, dwi_rician_aff, dwi_rician_scaled_file)
        dwi_rician_scaled_files.append(dwi_rician_scaled_file)
        # Bvals
        bvals_prefix = utils.get_prefix(bvals_files[i], file_ext='bval')
        bvals = utils.load_txt(bvals_files[i], txt_type='bvals')
        for j in range(len(bvals)):
            bvals[j] = utils.nearest(bvals[j], shells) # ...and round all bvals to the nearest shell.
        bvals_shelled_file = os.path.join(temp_dir, '{}_shelled.bval'.format(bvals_prefix))
        utils.save_txt(bvals, bvals_shelled_file)
        bvals_shelled_files.append(bvals_shelled_file)

    # For each image, calculate a distribution of intramask (csf or brain) intensities for each shell

    rician_labels = [] # labels to track the images/shells from which histograms come from
    rician_bins = [] # bins to track the x-axis of the histograms
    prerician_hists = [] # hists to track the histograms
    postrician_hists = []
    
    for i in range(len(dwi_scaled_files)): # iterate through images

        rician_labels.append([]) # each image needs a list of shell labels
        rician_bins.append([]) # each image needs a list of histogram x-axes
        prerician_hists.append([]) # each image needs a list of histograms (y-axes)
        postrician_hists.append([])

        bvals = utils.load_txt(bvals_shelled_files[i]) # iterate through non-zero shells
        bvals_unique = np.sort(np.unique(bvals[bvals!=0]))
        for bval in bvals_unique:

            # record shell being investigated
            rician_labels[i].append(bval) 

            # extract volumes from that shell
            dwi_shell_file, _, _ = utils.dwi_extract(dwi_scaled_files[i], bvals_shelled_files[i], temp_dir, target_bval=bval, first_only=False) 
            dwi_rician_shell_file, _, _ = utils.dwi_extract(dwi_rician_scaled_files[i], bvals_shelled_files[i], temp_dir, target_bval=bval, first_only=False)
            
            # generate bg mask
            dwi_b0s_file, _, _ = utils.dwi_extract(dwi_scaled_files[i], bvals_shelled_files[i], temp_dir, target_bval=0, first_only=False) 
            dwi_b0s_avg_file = utils.dwi_avg(dwi_b0s_file, temp_dir)
            dwi_mask_file = utils.dwi_mask(dwi_b0s_avg_file, temp_dir)
            dwi_mask_img = utils.load_nii(dwi_mask_file, dtype='bool', ndim=3)[0]

            # # CSF Mask
            #
            # # calculate a csf mask after masking out background: mask bg of avg shell image
            # dwi_shell_avg_file = utils.dwi_avg(dwi_shell_file, temp_dir) 
            # dwi_shell_avg_img, dwi_shell_avg_aff, _ = utils.load_nii(dwi_shell_avg_file)
            # dwi_shell_avg_img[np.logical_not(dwi_mask_img)] = 0
            # dwi_shell_avg_masked_file = os.path.join(temp_dir, '{}_masked.nii.gz'.format(utils.get_prefix(dwi_shell_avg_file, file_ext='nii')))
            # utils.save_nii(dwi_shell_avg_img, dwi_shell_avg_aff, dwi_shell_avg_masked_file)
            #
            # # calculate a csf mask after masking out background: get csf mask from bg-masked avg shell image
            # fast_prefix = '{}_fast'.format(utils.get_prefix(dwi_shell_avg_masked_file, file_ext='nii'))
            # fast_cmd = 'fast -o {} -v {}'.format(os.path.join(temp_dir, fast_prefix), dwi_shell_avg_masked_file)
            # utils.run_cmd(fast_cmd)
            # csf_file = os.path.join(temp_dir, '{}_pve_0.nii.gz'.format(fast_prefix))
            # csf_img = utils.load_nii(csf_file, ndim=4)[0]
            #
            # # Apply csf masks
            # dwi_shell_img, _, _ = utils.load_nii(dwi_shell_file, ndim=4) # load in volumes and prep mask
            # dwi_rician_shell_img, _, _ = utils.load_nii(dwi_rician_shell_file, ndim=4)
            # mask_img = np.tile(csf_img, [1, 1, 1, dwi_shell_img.shape[3]]) > 0.5

            # Brain Mask

            dwi_shell_img, _, _ = utils.load_nii(dwi_shell_file, ndim=4)
            dwi_rician_shell_img, _, _ = utils.load_nii(dwi_rician_shell_file, ndim=4)
            mask_img = np.tile(np.expand_dims(dwi_mask_img, axis=3), [1, 1, 1, dwi_shell_img.shape[3]])

            # Calculate histograms across all volumes of that shell with mask
            dwi_shell_intensities = dwi_shell_img[mask_img]
            dwi_rician_shell_intensities = dwi_rician_shell_img[mask_img]
            common_min_intensity = 0
            common_max_intensity = np.amax((np.amax(np.nanpercentile(dwi_shell_intensities, 99)), np.amax(np.nanpercentile(dwi_rician_shell_intensities, 99))))
            bins = np.linspace(common_min_intensity, common_max_intensity, 100)
            prerician_hist, _ = np.histogram(dwi_shell_intensities, bins=bins)
            postrician_hist, _ = np.histogram(dwi_rician_shell_intensities, bins=bins)
            prerician_hists[i].append(prerician_hist) # record histograms
            postrician_hists[i].append(postrician_hist)
            rician_bins[i].append(bins[:-1]) # record bins

    # Plot histograms

    fig = plt.figure(0, figsize=SHARED_VARS.PAGESIZE)
    num_subplots = len(rician_labels)

    for i in range(num_subplots):
        plt.subplot(num_subplots, 1, i+1)
        for j in range(len(rician_labels[i])):
            plt.plot(rician_bins[i][j], prerician_hists[i][j], label='b = {} (Without)'.format(rician_labels[i][j]))
            plt.plot(rician_bins[i][j], postrician_hists[i][j], label='b = {} (With)'.format(rician_labels[i][j]))
        plt.legend()
        plt.title(utils.get_prefix(dwi_files[i], file_ext='nii'), fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.ylabel('Frequency', fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.xlabel('Intensity', fontsize=SHARED_VARS.LABEL_FONTSIZE)
        plt.grid()

    plt.tight_layout()

    plt.subplots_adjust(top=0.85)
    plt.suptitle('Intramask Intensity Distributions With and Without Rician Correction\n(Intensities should decrease slightly with correction)', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    rician_vis_file = os.path.join(vis_dir, 'rician.pdf')
    plt.savefig(rician_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    # Finish Up

    utils.remove_dir(temp_dir)
    
    return rician_vis_file

def vis_stats(dwi_file, bvals_file, mask_file, chisq_matrix_file, motion_dict, eddy_dir, vis_dir):

    print('VISUALIZING MOTION AND CHI SQUARED STATISTICS')

    # Load data

    eddy_outlier_map_file = os.path.join(eddy_dir, 'eddy_results.eddy_outlier_map')
    bvals = utils.load_txt(bvals_file, txt_type='bvals')

    # Configure figure

    fig = plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    # Visualize rotations

    rotations = motion_dict['rotations']

    ax = plt.subplot(5, 3, 1)
    plt.plot(range(0, rotations.shape[0]), rotations[:, 0], color='b', label='x ({:.3f})'.format(motion_dict['eddy_avg_rotations'][0]))
    plt.plot(range(0, rotations.shape[0]), rotations[:, 1], color='c', label='y ({:.3f})'.format(motion_dict['eddy_avg_rotations'][1]))
    plt.plot(range(0, rotations.shape[0]), rotations[:, 2], color='m', label='z ({:.3f})'.format(motion_dict['eddy_avg_rotations'][2]))
    plt.xlim((-1, rotations.shape[0]+1))
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    plt.ylabel('Rotation' '\n' r'($^\circ$)', fontsize=3*SHARED_VARS.LABEL_FONTSIZE/4)
    plt.grid()
    plt.legend(fontsize=SHARED_VARS.LABEL_FONTSIZE/2, loc='upper left')
    plt.title('Motion and Intensity', fontsize=SHARED_VARS.TITLE_FONTSIZE)

    # Visualize translations

    translations = motion_dict['translations']

    ax = plt.subplot(5, 3, 4)
    plt.plot(range(0, translations.shape[0]), translations[:, 0], color='b', label='x ({:.3f})'.format(motion_dict['eddy_avg_translations'][0]))
    plt.plot(range(0, translations.shape[0]), translations[:, 1], color='c', label='y ({:.3f})'.format(motion_dict['eddy_avg_translations'][1]))
    plt.plot(range(0, translations.shape[0]), translations[:, 2], color='m', label='z ({:.3f})'.format(motion_dict['eddy_avg_translations'][2]))
    plt.xlim((-1, translations.shape[0]+1))
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    plt.ylabel('Translation\n(mm)', fontsize=3*SHARED_VARS.LABEL_FONTSIZE/4)
    plt.grid()
    plt.legend(fontsize=SHARED_VARS.LABEL_FONTSIZE/2, loc='upper left')

    # Visualize RMS Displacement

    abs_displacement = motion_dict['abs_displacement']
    rel_displacement = motion_dict['rel_displacement']

    ax = plt.subplot(5, 3, 7)
    plt.plot(range(0, abs_displacement.shape[0]), abs_displacement, color='b', label='Abs. ({:.3f})'.format(motion_dict['eddy_avg_abs_displacement'][0]))
    plt.plot(range(0, rel_displacement.shape[0]), rel_displacement, color='m', label='Rel. ({:.3f})'.format(motion_dict['eddy_avg_rel_displacement'][0]))
    plt.xlim((-1, abs_displacement.shape[0]+1))
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    plt.ylabel('Displacement\n(mm)', fontsize=3*SHARED_VARS.LABEL_FONTSIZE/4)
    plt.grid()
    plt.legend(fontsize=SHARED_VARS.LABEL_FONTSIZE/2, loc='upper left')

    # Visualize median intensity

    dwi_img, _, _ = utils.load_nii(dwi_file, ndim=4)
    mask_img, _, _ = utils.load_nii(mask_file, dtype='bool', ndim=3)
    median_intensities = np.zeros(dwi_img.shape[3])
    for i in range(dwi_img.shape[3]):
        dwi_vol = dwi_img[:, :, :, i]
        median_intensities[i] = np.nanmedian(dwi_vol[mask_img])

    ax = plt.subplot(5, 3, 10)
    ax.plot(range(0, dwi_img.shape[3]), median_intensities, color='c')
    ax.plot(range(0, dwi_img.shape[3]), median_intensities, '.', color='b')
    plt.xlim((-1, dwi_img.shape[3]+1))
    ax.set_ylabel('Median\nIntensity', fontsize=3*SHARED_VARS.LABEL_FONTSIZE/4)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(labelbottom=False)
    ax.grid()

    # Visualize B values

    ax = plt.subplot(5, 3, 13)
    ax.plot(range(0, len(bvals)), bvals, color='c', label='B Value')
    ax.plot(range(0, len(bvals)), bvals, '.', color='b', label='B Value')
    ax.set_ylabel('B Value', fontsize=3*SHARED_VARS.LABEL_FONTSIZE/4)
    plt.xlim((-1, len(bvals)+1))
    ax.grid()

    # Finish up motion

    ax.set_xlabel('Diffusion Volume', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    # Visualize outlier map

    outlier_map = _get_outlier_map(eddy_outlier_map_file)
    outlier_map = np.transpose(outlier_map)

    ax = plt.subplot(4, 2, 2)
    ax.matshow(outlier_map, aspect='auto', origin='lower')
    ax.set_title('Eddy Outlier Slices', fontsize=SHARED_VARS.TITLE_FONTSIZE)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Diffusion Volume', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    ax.set_ylabel('Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.xlim(0, outlier_map.shape[1])
    plt.ylim(0, outlier_map.shape[0])
    ax.grid()

    # Visualize Chi Squared

    chisq_matrix = utils.load_txt(chisq_matrix_file)
    chisq_slices = np.nanmedian(chisq_matrix, axis=1)
    chisq_vols = np.nanmedian(chisq_matrix, axis=0)

    ax = plt.subplot(12, 2, 12)
    ax.plot(list(range(0, len(chisq_vols))), chisq_vols)
    plt.xlim(0, len(chisq_vols))
    ax.grid()
    ax.set_title(r'$\chi^2$' ' Analysis After Eddy Outlier Correction', fontsize=SHARED_VARS.TITLE_FONTSIZE)
    ax.spines['bottom'].set_visible(False)

    ax = plt.subplot(2, 16, 24)
    ax.plot(chisq_slices, list(range(0, len(chisq_slices))))
    ax.invert_xaxis()
    plt.xticks(rotation=60)
    plt.ylim(0, len(chisq_slices))
    ax.grid()
    ax.spines['right'].set_visible(False)

    ax = plt.subplot(2, 2, 4)
    im = ax.imshow(chisq_matrix, aspect='auto', origin='lower', interpolation='nearest', vmin=0, vmax=0.2)
    ax.set_xlabel('Diffusion Volume', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Slice', rotation=-90, fontsize=SHARED_VARS.LABEL_FONTSIZE)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position('right')
    plt.xlim(0, chisq_matrix.shape[1])
    plt.ylim(0, chisq_matrix.shape[0])
    ax.grid()

    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.4])
    plt.colorbar(im, cax=cbar_ax)

    # Finish Up Figure

    stats_vis_file = os.path.join(vis_dir, 'stats.pdf')
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(stats_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    return stats_vis_file

def vis_gradcheck(bvals_files, bvecs_files, bvals_preproc_file, bvecs_preproc_file, bvals_corrected_file, bvecs_corrected_file, vis_dir):

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    print('VISUALIZING GRADIENT CHECK')

    # Load and prepare gradients

    bvals = utils.load_txt(utils.bvals_merge(bvals_files, 'raw_merged', temp_dir), txt_type='bvals')
    bvecs = utils.load_txt(utils.bvecs_merge(bvecs_files, 'raw_merged', temp_dir), txt_type='bvecs')

    bvals_preproc = utils.load_txt(bvals_preproc_file, txt_type='bvals')
    bvals_corrected = utils.load_txt(bvals_corrected_file, txt_type='bvals')
    bvecs_preproc = utils.load_txt(bvecs_preproc_file, txt_type='bvecs')
    bvecs_corrected = utils.load_txt(bvecs_corrected_file, txt_type='bvecs')

    scaled_bvecs = np.array([np.multiply(bvals, bvecs[0, :]), np.multiply(bvals, bvecs[1, :]), np.multiply(bvals, bvecs[2, :])])
    scaled_bvecs_preproc = np.array([np.multiply(bvals_preproc, bvecs_preproc[0, :]), np.multiply(bvals_preproc, bvecs_preproc[1, :]), np.multiply(bvals_preproc, bvecs_preproc[2, :])])
    scaled_bvecs_corrected = np.array([np.multiply(bvals_corrected, bvecs_corrected[0, :]), np.multiply(bvals_corrected, bvecs_corrected[1, :]), np.multiply(bvals_corrected, bvecs_corrected[2, :])])

    # Visualize

    gradcheck_vis_file = os.path.join(vis_dir, 'gradcheck.pdf')

    fig = plt.figure(0, figsize=SHARED_VARS.PAGESIZE)
    
    ax = plt.axes(projection='3d')
    ax.scatter(scaled_bvecs[0, :], scaled_bvecs[1, :], scaled_bvecs[2, :], c='r', marker='o', label='Original')
    ax.scatter(scaled_bvecs_preproc[0, :], scaled_bvecs_preproc[1, :], scaled_bvecs_preproc[2, :], c='b', marker='x', label='Preprocessed')
    ax.scatter(scaled_bvecs_corrected[0, :], scaled_bvecs_corrected[1, :], scaled_bvecs_corrected[2, :], c='g', marker='+', label='Preprocessed + Optimized')
    plt.title('Gradient Check', fontsize=SHARED_VARS.TITLE_FONTSIZE)
    plt.legend(fontsize=SHARED_VARS.LABEL_FONTSIZE)
    ax.set_box_aspect([1,1,1]) # Make sure axes have equal aspect ratios
    ax_radius = 1.1*np.amax(bvals)
    ax.set_xlim3d((-ax_radius, ax_radius))
    ax.set_ylim3d((-ax_radius, ax_radius))
    ax.set_zlim3d((-ax_radius, ax_radius))

    plt.tight_layout()

    plt.subplots_adjust(bottom=0.2)
    fig.add_axes([0.025, 0.05, 0.95, 0.15])
    info_str = '- Original: Raw gradients (b-vectors scaled by b-value) given as input.\n- Preprocessed: Gradients output and rotated by eddy. Often slightly different than the original gradients.\n- Preprocessed + Optimized: Preprocessed gradients that have been sign and order permuted to produce the optimal tract length as determined by dwigradcheck in MRTrix3. Ideally identical to the preprocessed gradients. If not, this suggests an incorrect sign or axis permutation in the b-vectors. Tensor or vector glyph visualization in this PDF can help support this.'
    plt.text(0, 0, info_str, ha='left', va='bottom', wrap=True, fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.axis('off')

    plt.savefig(gradcheck_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    return gradcheck_vis_file

def vis_dwi(dwi_file, bvals_shelled, bvecs_file, cnr_dict, vis_dir):

    print('VISUALIZING DWI')

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    dwi_vis_files = []
    
    bvals = np.sort(np.unique(bvals_shelled))

    for i in range(len(bvals)):

        bvals_shelled_file = StringIO(' '.join([str(bval) for bval in bvals_shelled]))

        bX = bvals[i]
        bXs_file, num_bXs, _ = utils.dwi_extract(dwi_file, bvals_shelled_file, temp_dir, target_bval=bX, first_only=False)
        bXs_avg_file = utils.dwi_avg(bXs_file, temp_dir)
        bXs_avg_slices, bXs_avg_vox_dim, bXs_avg_min, bXs_avg_max = utils.slice_nii(bXs_avg_file, offsets=[-10, -5, 0, 5, 10], min_intensity=0, max_percentile=SHARED_VARS.VIS_PERCENTILE_MAX)
        
        cnr = '{:.3f}'.format(cnr_dict[bX])
        cnr_label = 'SNR' if bX == 0 else 'CNR'

        bXs_vis_file = _vis_vol(bXs_avg_slices, bXs_avg_vox_dim, bXs_avg_min, bXs_avg_max, vis_dir, name='Preprocessed_b_=_{},_{}_scan_average,_{}_=_{}'.format(bX, num_bXs, cnr_label, cnr), colorbar=False)
        dwi_vis_files.append(bXs_vis_file)

        bvals_shelled_file.close()

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    return dwi_vis_files

def vis_scalar(scalar_file, vis_dir, name='?', comment=''):

    print('VISUALIZING {} SCALAR MAP'.format(name))
    
    scalar_min = 0
    if name == 'FA':
        scalar_max = 1
    elif name == 'MD':
        scalar_max = SHARED_VARS.ADC_WATER
    
    scalar_slices, scalar_vox_dim, _, _ = utils.slice_nii(scalar_file, offsets=[-10, -5, 0, 5, 10])
    scalar_vis_file = _vis_vol(scalar_slices, scalar_vox_dim, scalar_min, scalar_max, vis_dir, name=name, comment=comment, colorbar=True)
    
    return scalar_vis_file

def vis_fa_stats(roi_names, roi_med_fa, fa_file, atlas_ants_fa_file, vis_dir):

    print('VISUALIZING FA STATISTICS')

    fa_slices, fa_vox_dim, _, _ = utils.slice_nii(fa_file)
    _, fa_aff, _ = utils.load_nii(fa_file)

    atlas_slices, atlas_vox_dim, _, _ = utils.slice_nii(atlas_ants_fa_file, custom_aff=fa_aff) # Using fa affine because atlas has been registered to fa and should have voxel correspondence with same affine, but ants seems to have some rounding error. Only a problem on edge cases (i.e. rotated 45° exactly)

    overlay_colormap = mcm.jet
    overlay_colormap.set_under('k', alpha=0)

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    plt.subplot(1, 2, 1)
    plt.barh(roi_names, roi_med_fa, height=0.5)
    plt.xticks(fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.title('Median FA per ROI', fontsize=SHARED_VARS.TITLE_FONTSIZE)
    plt.grid()

    plt.subplot(3, 2, 2)
    utils.plot_slice(slices=fa_slices, img_dim=0, offset_index=0, vox_dim=fa_vox_dim, img_min=0, img_max=1)
    utils.plot_slice(slices=atlas_slices, img_dim=0, offset_index=0, vox_dim=atlas_vox_dim, img_min=1, img_max=len(roi_names), alpha=0.5, cmap=overlay_colormap)
    plt.title('FA ROI Alignment', fontsize=SHARED_VARS.TITLE_FONTSIZE)
    plt.ylabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 2, 4)
    utils.plot_slice(slices=fa_slices, img_dim=1, offset_index=0, vox_dim=fa_vox_dim, img_min=0, img_max=1)
    utils.plot_slice(slices=atlas_slices, img_dim=1, offset_index=0, vox_dim=atlas_vox_dim, img_min=1, img_max=len(roi_names), alpha=0.5, cmap=overlay_colormap)
    plt.ylabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(3, 2, 6)
    utils.plot_slice(slices=fa_slices, img_dim=2, offset_index=0, vox_dim=fa_vox_dim, img_min=0, img_max=1)
    utils.plot_slice(slices=atlas_slices, img_dim=2, offset_index=0, vox_dim=atlas_vox_dim, img_min=1, img_max=len(roi_names), alpha=0.5, cmap=overlay_colormap)
    plt.ylabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.tight_layout()

    fa_stats_vis_file = os.path.join(vis_dir, 'fa_stats.pdf')
    plt.savefig(fa_stats_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    return fa_stats_vis_file

def vis_glyphs(tensor_file, v1_file, fa_file, cc_center_voxel, vis_dir, glyph_type='tensor'):

    print('VISUALIZING TENSORS')

    temp_dir = utils.make_dir(vis_dir, 'TEMP')

    # Prepare mrview command: Location to visualize

    cc_center_voxel_str = ','.join([str(np.round(loc)) for loc in cc_center_voxel])

    # Prepare mrview command: Planes to visualize + correspondence with file names

    planes = {
        0: 'sagittal',
        1: 'coronal',
        2: 'axial'
    }

    # Prepare mrview command: Glyphs, either tensor or v1

    bad_adc_mask_nii = _bad_adc_nii_mask(tensor_file, temp_dir)
    if glyph_type == 'tensor':
        tensor_filtered_file = _filter_bad_adc(tensor_file, bad_adc_mask_nii, temp_dir)
        glyph_file = tensor_filtered_file
        glyph_load_str = '-odf.load_tensor'
        glyph_title_str = 'Tensors'
    elif glyph_type == 'vector':
        v1_filtered_file = _filter_bad_adc(v1_file, bad_adc_mask_nii, temp_dir)
        glyph_file = v1_filtered_file
        glyph_load_str = '-fixel.load'
        glyph_title_str = 'Principal Eigenvectors'

    # Generate mrview commands and plot glyphs

    for i in planes:
        vis_cmd = 'mrview -load {} {} {} -mode 1 -plane {} -fov 160 -voxel {} -focus 0 -size 1200,1200 -capture.folder {} -capture.prefix {} -capture.grab -noannotations -exit -nthreads {}'.format(
            fa_file, glyph_load_str, glyph_file, i, cc_center_voxel_str, temp_dir, planes[i], SHARED_VARS.NUM_THREADS-1
        )
        utils.run_cmd(vis_cmd) # will save as '<planes[i]>0000.png'
        vis_zoom_cmd = 'mrview -load {} {} {} -mode 1 -plane {} -fov 80 -voxel {} -focus 0 -size 1200,1200 -capture.folder {} -capture.prefix {} -capture.grab -noannotations -exit -nthreads {}'.format(
            fa_file, glyph_load_str, glyph_file, i, cc_center_voxel_str, temp_dir, '{}_zoom'.format(planes[i]), SHARED_VARS.NUM_THREADS-1 # will save as 'planes[i]_zoom0000.png'
        )
        utils.run_cmd(vis_zoom_cmd)

    # Put PDF page together

    glyph_vis_file = os.path.join(vis_dir, 'glyphs.pdf')

    plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    plt.subplot(2, 3, 1)
    plt.imshow(plt.imread(os.path.join(temp_dir, 'sagittal0000.png')))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel('160 mm FOV', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(2, 3, 4)
    plt.imshow(plt.imread(os.path.join(temp_dir, 'sagittal_zoom0000.png')))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.ylabel('80 mm FOV', fontsize=SHARED_VARS.LABEL_FONTSIZE)
    plt.xlabel('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(2, 3, 2)
    plt.imshow(plt.imread(os.path.join(temp_dir, 'coronal0000.png')))
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(2, 3, 5)
    plt.imshow(plt.imread(os.path.join(temp_dir, 'coronal_zoom0000.png')))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.subplot(2, 3, 3)
    plt.imshow(plt.imread(os.path.join(temp_dir, 'axial0000.png')))
    plt.xticks([], [])
    plt.yticks([], [])

    plt.subplot(2, 3, 6)
    plt.imshow(plt.imread(os.path.join(temp_dir, 'axial_zoom0000.png')))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)

    plt.tight_layout()

    plt.suptitle('{} (Non-physiologic Eigenvalues Omitted)'.format(glyph_title_str), fontsize=SHARED_VARS.TITLE_FONTSIZE)
    plt.savefig(glyph_vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    return glyph_vis_file

# Helper Functions

def _methods_strs(use_topup, use_synb0, params):

    c = 1
    s = []
    r = []
    
    s.append('The diffusion data were preprocessed and quality-checked with the following pipeline built around the MRTrix3 [{}], FSL [{}], and ANTs [{}] software packages.'.format(c, c+1, c+2))
    r.append('[{}] Tournier, J. D. et al. (2019). MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. NeuroImage, 116137.'.format(c))
    r.append('[{}] Jenkinson, M. et al. (2012). Fsl. Neuroimage, 62(2), 782-790.'.format(c+1))
    r.append('[{}] Tustison, N. J. et al. (2014). Large-scale evaluation of ANTs and FreeSurfer cortical thickness measurements. Neuroimage, 99, 166-179.'.format(c+2))
    c += 3

    s.append('First, any volumes with a corresponding b value less than {} were treated as b0 volumes for the remainder of the pipeline.'.format(params['bval_threshold']))

    if params['use_denoise']:
        s.append('The diffusion data were denoised with the provided dwidenoise (MP-PCA) function included with MRTrix3 [{}][{}][{}].'.format(c, c+1, c+2))
        r.append('[{}] Veraart, J. et al. (2016). Denoising of diffusion MRI using random matrix theory. Neuroimage, 142, 394-406.'.format(c))
        r.append('[{}] Veraart, J. et al. (2016). Diffusion MRI noise mapping using random matrix theory. Magnetic resonance in medicine, 76(5), 1582-1593.'.format(c+1))
        r.append('[{}] Cordero-Grande, L. et al. (2019). Complex diffusion-weighted image estimation via matrix recovery under general noise models. NeuroImage, 200, 391-404.'.format(c+2))             
        c += 3

    if params['use_degibbs']:
        s.append('Gibbs de-ringing followed with the local subvoxel-shifts method [{}].'.format(c))
        r.append('[{}] Kellner, E. et al. (2016). Gibbs‐ringing artifact removal based on local subvoxel‐shifts. Magnetic resonance in medicine, 76(5), 1574-1581.'.format(c))
        c += 1

    if params['use_rician']:
        s.append('Rician correction was performed with the method of moments [{}].'.format(c))
        r.append('[{}] Koay, C. G. et al. (2006). Analytically exact correction scheme for signal extraction from noisy magnitude MR signals. Journal of magnetic resonance, 179(2), 317-322.'.format(c))
        c += 1

    s.append('The images were then {}concatenated for further processing.'.format('intensity-normalized to the first image and ' if params['use_prenormalize'] else ''))
    
    if use_topup:
        if use_synb0:
            pre_str = 'No reverse phase encoded images were acquired, but corresponding T1 images of the subjects were available. Thus, a T1 image was used to generate a synthetic susceptibility-corrected b0 volume using SYNB0-DISCO, a deep learning framework by Schilling et al. [{}]. This synthetic b0 image was used in conjunction with FSL\'s topup to correct for susceptibility-induced artifacts in the diffusion data. FSL\'s eddy algorithm was then used to correct for'.format(c)
            r.append('[{}] Schilling, K. G. et al. (2019). Synthesized b0 for diffusion distortion correction (Synb0-DisCo). Magnetic resonance imaging, 64, 62-70.'.format(c))
            c += 1
        else:
            pre_str = 'FSL\'s topup and eddy algorithms were used to correct for susceptibilty-induced and'
    else:
            pre_str = 'FSL\'s eddy algorithm was then used to correct for'
    s.append('{} motion artifacts and eddy currents and to remove outlier slices [{}][{}][{}][{}].'.format(pre_str, c, c+1, c+2, c+3))
    r.append('[{}] Andersson, J. L. et al. (2003). How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. Neuroimage, 20(2), 870-888.'.format(c))
    r.append('[{}] Smith, S. M. et al. (2004). Advances in functional and structural MR image analysis and implementation as FSL. Neuroimage, 23, S208-S219.'.format(c+1))
    r.append('[{}] Andersson, J. L. et al. (2016). An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. Neuroimage, 125, 1063-1078.'.format(c+2))
    r.append('[{}] Andersson, J. L. et al. (2016). Incorporating outlier detection and replacement into a non-parametric framework for movement and distortion correction of diffusion MR images. NeuroImage, 141, 556-572.'.format(c+3))
    c += 4

    if params['use_postnormalize']:
        s.append('The images were then intensity-normalized to the first image{}.'.format(' a second time' if params['use_prenormalize'] else ''))
    
    if params['use_unbias']:
        s.append('N4 bias field correction was then performed [{}].'.format(c))
        r.append('[{}] Tustison, N. J. et al. (2010). N4ITK: improved N3 bias correction. IEEE transactions on medical imaging, 29(6), 1310-1320.'.format(c))
        c += 1
    
    s.append('Lastly, the preprocessed data were fitted with a tensor model using the dwi2tensor function included with MRTrix3 using an iterative reweighted least squares estimator [{}].'.format(c))
    r.append('[{}] Veraart, J. et al. (2013). Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. Neuroimage, 81, 335-346.'.format(c))
    c += 1

    s.append('The quality of this preprocessing pipeline was then assessed qualitatively for gross errors and quantitatively analyzed using a three-step approach.')

    s.append('In the first step, the preprocessed data were analyzed in accordance with the method outlined by Lauzon et al. [{}].'.format(c))
    r.append('[{}] Lauzon, C. B. et al. (2013). Simultaneous analysis and quality assurance for diffusion tensor imaging. PloS one, 8(4).'.format(c))
    c += 1

    s.append('The brain parenchyma without CSF were masked in a restrictive manner by using an eroded brain mask generated on the average b0 image using the bet2 function included with FSL [{}].'.format(c))
    r.append('[{}] Smith, S. M. (2002). Fast robust automated brain extraction. Human brain mapping, 17(3), 143-155.'.format(c))
    c += 1

    s.append('Then, the tensor fits of the masked data were backpropagated through the diffusion model to reconstruct the original diffusion signal.')
    s.append('The goodness-of-fit for the tensor model was then assessed using a modified pixel chi-squared value per slice per volume.')
    
    s.append('In the second step, the tensor fit was converted to a fractional anisotropy (FA) image [{}][{}].'.format(c, c+1))
    r.append('[{}] Basser, P. J. et al. (1994). MR diffusion tensor spectroscopy and imaging. Biophysical journal, 66(1), 259-267.'.format(c))
    r.append('[{}] Westin, C. F. (1997). Geometrical diffusion measures for MRI from tensor basis analysis. Proc. ISMRM\'97.'.format(c+1))
    c += 2
    
    s.append('The ICBM FA MNI atlas with 48 white matter tract labels provided with FSL were then non-rigidly registered to the subject\'s FA image with the ANTs software package [{}][{}][{}][{}].'.format(c, c+1, c+2, c+3))
    s.append('The average FA for each tract was then quantified and assessed for physiologic congruence.')
    r.append('[{}] Mori, S. et al. (2005). MRI atlas of human white matter. Elsevier.'.format(c))
    r.append('[{}] Wakana, S. et al. (2007). Reproducibility of quantitative tractography methods applied to cerebral white matter. Neuroimage, 36(3), 630-644.'.format(c+1))
    r.append('[{}] Hua, K. et al. (2008). Tract probability maps in stereotaxic spaces: analyses of white matter anatomy and tract-specific quantification. Neuroimage, 39(1), 336-347.'.format(c+2))
    r.append('[{}] Avants, B. B. et al. (2008). Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical image analysis, 12(1), 26-41.'.format(c+3))
    c += 4

    s.append('Lastly, the gradient orientations were visualized and checked using the dwigradcheck script included with MRTrix [{}].'.format(c))
    r.append('[{}] Jeurissen, B. et al. (2014). Automated correction of improperly rotated diffusion gradient orientations in diffusion weighted MRI. Medical image analysis, 18(7), 953-962.'.format(c))
    c += 1   

    return s, r

def _vis_vol(slices, vox_dim, min, max, vis_dir, name='?', comment='', colorbar=False):

    title = name.replace('_', ' ')
    if not comment == '':
        title = '{}\n({})'.format(title, comment)

    print('VISUALIZING 3D VOLUME: {}'.format(name))

    fig = plt.figure(0, figsize=SHARED_VARS.PAGESIZE)

    for i in range(0, 5):

        plt.subplot(3, 5, i+1)
        utils.plot_slice(slices=slices, img_dim=0, offset_index=i, vox_dim=vox_dim, img_min=min, img_max=max)
        if i == 0:
            plt.xlabel('Right-most Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)
        if i == 2:
            plt.title('Sagittal', fontsize=SHARED_VARS.LABEL_FONTSIZE)
            plt.xlabel('P | A', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)
        if i == 4:
            plt.xlabel('Left-most Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)

        plt.subplot(3, 5, i+1 + 5)
        utils.plot_slice(slices=slices, img_dim=1, offset_index=i, vox_dim=vox_dim, img_min=min, img_max=max)
        if i == 0:
            plt.xlabel('Posterior-most Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)
        if i == 2:
            plt.title('Coronal', fontsize=SHARED_VARS.LABEL_FONTSIZE)
            plt.xlabel('R | L', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)
        if i == 4:
            plt.xlabel('Anterior-most Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)

        plt.subplot(3, 5, i+1 + 10)
        im = utils.plot_slice(slices=slices, img_dim=2, offset_index=i, vox_dim=vox_dim, img_min=min, img_max=max)
        if i == 0:
            plt.xlabel('Inferior-most Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)
        if i == 2:
            plt.title('Axial', fontsize=SHARED_VARS.LABEL_FONTSIZE)
            plt.xlabel('R | L', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)
        if i == 4:
            plt.xlabel('Superior-most Slice', fontsize=SHARED_VARS.LABEL_FONTSIZE / 2)

    plt.tight_layout()

    plt.subplots_adjust(top=0.925)
    plt.suptitle('{}'.format(title), fontsize=SHARED_VARS.TITLE_FONTSIZE)

    if colorbar:
        plt.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.875, 0.1, 0.025, 0.8])
        plt.colorbar(im, cax=cbar_ax)

    vis_file = os.path.join(vis_dir, '{}.pdf'.format(name))
    plt.savefig(vis_file, dpi=SHARED_VARS.PDF_DPI)
    plt.close()

    return vis_file

def _get_outlier_map(path):

    rows = []
    with open(path,'r') as f:
        txt = f.readlines()
        for i in np.arange(1,len(txt)):
            rows.append(_str2list(txt[i].strip('\n')))
    outlier_array = np.array(rows)
    return outlier_array

def _str2list(string):

    row = []
    for i in range(len(string)):
        if (not np.mod(i,2)):
            row.append(int(string[i]))
    return row

def _bad_adc_nii_mask(tensor_file, filter_dir):

    print('LOCATING IMPROBABLE TENSOR VALUES (DIAGONAL ELEMENTS < 0 OR > 3xADC_WATER)')

    tensor_img, tensor_aff, _ = utils.load_nii(tensor_file, ndim=4)

    tensor_diag_img = tensor_img[:, :, :, 0:3]
    bad_adcdiag_img = np.logical_or(tensor_diag_img < 0, tensor_diag_img > 3*SHARED_VARS.ADC_WATER)
    bad_adc3d_img = bad_adcdiag_img[:, :, :, 0]
    for i in range(1, bad_adcdiag_img.shape[3]):
        bad_adc3d_img = np.logical_or(bad_adc3d_img, bad_adcdiag_img[:, :, :, i])
    bad_adc3d_nii = nib.Nifti1Image(bad_adc3d_img.astype('int'), tensor_aff)

    return bad_adc3d_nii

def _filter_bad_adc(glyph_file, bad_adc3d_nii, filter_dir):

    glyph_prefix = utils.get_prefix(glyph_file)

    print('REMOVING IMPROBABLE EIGENVALUE GLYPHS FROM {}'.format(glyph_prefix))

    glyph_img, glyph_aff, _ = utils.load_nii(glyph_file, ndim=4)

    bad_adc3d_nii_list = []
    for i in range(0, glyph_img.shape[3]):
        bad_adc3d_nii_list.append(bad_adc3d_nii)
    bad_adc4d_nii = nib.concat_images(bad_adc3d_nii_list, axis=None)
    bad_adc4d_img = bad_adc4d_nii.get_data().astype('bool')

    glyph_filtered_img = glyph_img
    glyph_filtered_img[bad_adc4d_img] = np.nan
    glyph_filtered_file = os.path.join(filter_dir, '{}_filtered.nii.gz'.format(glyph_prefix))
    utils.save_nii(glyph_filtered_img, glyph_aff, glyph_filtered_file)

    return glyph_filtered_file
