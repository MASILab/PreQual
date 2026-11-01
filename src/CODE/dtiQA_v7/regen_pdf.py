# PreQual: regenerate the brain-figure pages of the QA PDF from a finished output directory
# MASI Lab
# Vanderbilt University
#
# A development aid for checking the visualization code (in particular the TARDIS composites) without
# rerunning the pipeline. It reads what a normal run leaves behind (PREPROCESSED, TENSOR, SCALARS, STATS,
# EDDY and TARDIS) and writes <out_dir>/PDF_CHECK/dtiQA_check.pdf with the pages whose content depends on
# the image data: preprocessing and masks, the TARDIS pages, the per-shell averages, tensor glyphs, FA,
# FA ROI alignment and MD. The title, phase encoding, normalization, motion and gradient check pages are
# not rebuilt. The existing PDF/dtiQA.pdf is left alone.
#
# If TARDIS ran (TARDIS/PREPROCESSED, TENSOR, SCALARS and LOGS present) but its COMPOSITE folder is missing,
# the composites are built first with 006_composite_outputs.py, exactly as tardis.sh does. The TARDIS QA PNGs
# in TARDIS/LOGS/QA are rewritten by the TARDIS pages. Outputs of PreQual versions with the older per-shell
# TARDIS layout (fod_b<label>/, native/) are not supported; rerun TARDIS for those.
#
# Run inside the container with the dtiQA venv and under xvfb (mrview renders the glyph page), e.g.
#
#   singularity exec -B /path/to/PreQual/src:/mnt/src -B $OUT:$OUT prequal.sif bash -c \
#     'mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"; cd /tmp; \
#      xvfb-run -a --server-args="-screen 0 1600x1280x24 -ac" \
#      /CODE/dtiQA_v7/venv/bin/python /mnt/src/CODE/dtiQA_v7/regen_pdf.py $OUT --num_threads 4'
#
# Binding the repository's src folder makes the script import the vis.py and utils.py next to it and use
# the 006_composite_outputs.py of the same checkout, so the current code is tested against the container's
# tools and Python environments.

import os
import glob
import argparse as ap

import numpy as np

import vis
import utils
from vars import SHARED_VARS

def main():

    parser = ap.ArgumentParser(description='Regenerate the brain-figure pages of the PreQual QA PDF from a finished output directory (for checking the visualization code). Writes <out_dir>/PDF_CHECK/dtiQA_check.pdf.')
    parser.add_argument('out_dir', help='A finished PreQual OUTPUTS directory (with PREPROCESSED, TENSOR, SCALARS, STATS and EDDY folders)')
    parser.add_argument('--in_dir', metavar='path', default='', help='The INPUTS directory of that run (dtiQA_config.csv and the raw scans). Used for the raw columns of the preprocessing page; without it the CHECKED folder is used if it was kept, otherwise only the preprocessed column is drawn')
    parser.add_argument('--glyph_type', metavar='tensor/vector', default='tensor', help='Glyph type for the tensor page (default = tensor)')
    parser.add_argument('--num_threads', metavar='N', default=1, help='Number of threads for mrview and tensor2metric (default = 1)')
    parser.add_argument('--tardis_py', metavar='path', default='', help='Python interpreter with nibabel for building missing TARDIS composites (default = $TARDIS_HOME/venv/bin/python)')
    parser.add_argument('--composite_script', metavar='path', default='', help='006_composite_outputs.py to use for missing composites (default = the copy in this checkout, else $TARDIS_HOME)')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    if not os.path.isdir(out_dir):
        raise utils.DTIQAError('OUTPUT DIRECTORY {} NOT FOUND. EXITING.'.format(out_dir))

    num_threads = float(args.num_threads)
    if int(num_threads) == num_threads and int(num_threads) > 0:
        SHARED_VARS.NUM_THREADS = int(num_threads)
    else:
        raise utils.DTIQAError('INVALID INPUT FOR --num_threads PARAMETER. EXITING.')

    if args.glyph_type not in ('tensor', 'vector'):
        raise utils.DTIQAError('INVALID INPUT FOR --glyph_type PARAMETER. EXITING.')

    tardis_home = os.environ.get('TARDIS_HOME', '/APPS/tardis')

    # LOCATE THE OUTPUTS OF THE FINISHED RUN

    def need(path):
        if not os.path.exists(path):
            raise utils.DTIQAError('{} NOT FOUND. THIS SCRIPT NEEDS A FINISHED, UNSPLIT PREQUAL OUTPUT DIRECTORY. EXITING.'.format(path))
        return path

    preproc_dir = os.path.join(out_dir, 'PREPROCESSED')
    dwi_preproc_file = need(os.path.join(preproc_dir, 'dwmri.nii.gz'))
    bvals_preproc_file = need(os.path.join(preproc_dir, 'dwmri.bval'))
    bvecs_preproc_file = need(os.path.join(preproc_dir, 'dwmri.bvec'))
    mask_file = need(os.path.join(preproc_dir, 'mask.nii.gz'))

    tensor_file = need(os.path.join(out_dir, 'TENSOR', 'dwmri_tensor.nii.gz'))
    fa_file = need(os.path.join(out_dir, 'SCALARS', 'dwmri_tensor_fa.nii.gz'))
    md_file = need(os.path.join(out_dir, 'SCALARS', 'dwmri_tensor_md.nii.gz'))
    v1_file = need(os.path.join(out_dir, 'SCALARS', 'dwmri_tensor_v1.nii.gz'))

    stats_dir = os.path.join(out_dir, 'STATS')
    chisq_mask_file = need(os.path.join(stats_dir, 'chisq_mask.nii.gz'))
    cnr_file = need(os.path.join(stats_dir, 'eddy_median_cnr.txt'))
    roi_med_fa_file = need(os.path.join(stats_dir, 'roi_med_fa.txt'))
    atlas2subj_file = need(os.path.join(stats_dir, 'atlas2subj.nii.gz'))

    eddy_mask_file = os.path.join(out_dir, 'EDDY', 'eddy_mask.nii.gz')
    if not os.path.exists(eddy_mask_file):
        print('NOTE: {} NOT FOUND (--eddy_mask off?), DRAWING THE PREPROCESSED MASK IN ITS PLACE'.format(eddy_mask_file))
        eddy_mask_file = mask_file

    tardis_dir = os.path.join(out_dir, 'TARDIS')
    tardis_files = [os.path.join(tardis_dir, 'LOGS', 'fov.json'), os.path.join(tardis_dir, 'LOGS', 'fov_native.json'),
                    os.path.join(tardis_dir, 'PREPROCESSED', 'dwmri_extended.nii.gz'), os.path.join(tardis_dir, 'TENSOR', 'dwmri_extended_tensor.nii.gz'),
                    os.path.join(tardis_dir, 'SCALARS', 'dwmri_extended_tensor_fa.nii.gz'), os.path.join(tardis_dir, 'MODEL_SPACE', 'fa.nii.gz')]
    use_tardis = all([os.path.exists(f) for f in tardis_files])
    if os.path.isdir(tardis_dir) and not use_tardis:
        print('NOTE: TARDIS FOLDER FOUND BUT INCOMPLETE OR IN THE OLD PER-SHELL LAYOUT (MISSING {}), PAGES WILL USE THE NATIVE DATA'.format(', '.join([f for f in tardis_files if not os.path.exists(f)])))

    vis_dir = utils.make_dir(out_dir, 'PDF_CHECK')
    temp_dir = utils.make_dir(vis_dir, 'TEMP_REGEN') # vis functions use their own TEMP folder inside vis_dir

    print('REGENERATING PDF PAGES FROM {}'.format(out_dir))
    print('- TARDIS: {}'.format('found' if use_tardis else 'not found, native data only'))

    # RAW SCANS FOR THE PREPROCESSING PAGE: INPUTS DIRECTORY, ELSE THE CHECKED FOLDER IF IT WAS KEPT

    dwi_raw_files = []
    bvals_raw_files = []
    if args.in_dir != '':
        dwi_raw_files, bvals_in_files, _, _, _ = utils.load_config(os.path.abspath(args.in_dir))
        threshold_dir = os.path.join(out_dir, 'THRESHOLDED_BVALS')
        for bvals_in_file in bvals_in_files: # the thresholded b-values are what the pipeline treated as b0
            bvals_thresholded_file = os.path.join(threshold_dir, os.path.split(bvals_in_file)[-1])
            bvals_raw_files.append(bvals_thresholded_file if os.path.exists(bvals_thresholded_file) else bvals_in_file)
    else:
        for dwi_checked_file in sorted(glob.glob(os.path.join(out_dir, 'CHECKED', '*_checked.nii.gz'))):
            bvals_checked_file = dwi_checked_file.replace('.nii.gz', '.bval')
            if os.path.exists(bvals_checked_file):
                dwi_raw_files.append(dwi_checked_file)
                bvals_raw_files.append(bvals_checked_file)
    if len(dwi_raw_files) == 0:
        print('NOTE: NO RAW SCANS FOUND (GIVE --in_dir), THE PREPROCESSING PAGE WILL ONLY SHOW THE PREPROCESSED COLUMN')

    # SNR/CNR PER SHELL AND THE SHELLED B-VALUES, AS SAVED BY stats.cnr

    cnrs = np.loadtxt(cnr_file)
    if len(cnrs.shape) == 1:
        cnrs = np.expand_dims(cnrs, axis=0)
    shell_labels = cnrs[:, 0]
    cnr_dict = {}
    for i in range(cnrs.shape[0]):
        cnr_dict[float(shell_labels[i])] = float(cnrs[i, 1])
    bvals_preproc = utils.load_txt(bvals_preproc_file, txt_type='bvals')
    bvals_preproc_shelled = [float(utils.nearest(bval, shell_labels)) for bval in bvals_preproc]

    # PERCENT OF IMPROBABLE VOXELS (RECOMPUTED, NOT SAVED BY THE PIPELINE)

    _, percent_improbable = utils.dwi_improbable_mask(mask_file, dwi_preproc_file, bvals_preproc_file, temp_dir)

    # MEDIAN FA PER ROI AND THE CORPUS CALLOSUM CENTER, AS IN stats.scalar_info

    roi_names = []
    roi_med_fa = []
    with open(roi_med_fa_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line == '':
                continue
            roi_name, roi_val = line.rsplit(' ', 1)
            roi_names.append(roi_name)
            roi_med_fa.append(float(roi_val))

    atlas2subj_img, _, _ = utils.load_nii(atlas2subj_file, ndim=3)
    cc_genu_val = 3 # taken from label.txt
    cc_splenium_val = 5
    cc_index = np.logical_or(atlas2subj_img == cc_genu_val, atlas2subj_img == cc_splenium_val)
    cc_locs = np.column_stack(np.where(cc_index))
    cc_center_voxel = (np.nanmean(cc_locs[:, 0]), np.nanmean(cc_locs[:, 1]), np.nanmean(cc_locs[:, 2]))

    # TARDIS COMPOSITES: BUILD THEM IF THIS RUN PREDATES THEM, THEN LOAD THEM

    fov = None
    if use_tardis:
        composite_dir = os.path.join(tardis_dir, 'COMPOSITE')
        if not os.path.exists(os.path.join(composite_dir, 'composite.json')):
            print('TARDIS COMPOSITE VOLUMES NOT FOUND, BUILDING THEM')
            composite_script = args.composite_script
            if composite_script == '':
                checkout_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'APPS', 'tardis', '006_composite_outputs.py')
                composite_script = os.path.abspath(checkout_script) if os.path.exists(checkout_script) else os.path.join(tardis_home, '006_composite_outputs.py')
            tardis_py = args.tardis_py if args.tardis_py != '' else os.path.join(tardis_home, 'venv', 'bin', 'python')
            need(composite_script)
            need(tardis_py)
            composite_cmd = '{} {} --mask {} --tensor {} --fa {} --md {} --tardis-dir {}'.format(
                tardis_py, composite_script, mask_file, tensor_file, fa_file, md_file, tardis_dir)
            utils.run_cmd(composite_cmd)
            v1_cmd = 'tensor2metric {} -vector {} -num 1 -modulate none -mask {} -nthreads {} -force -quiet'.format(
                os.path.join(composite_dir, 'tensor.nii.gz'), os.path.join(composite_dir, 'v1.nii.gz'), os.path.join(composite_dir, 'mask.nii.gz'), SHARED_VARS.NUM_THREADS - 1)
            utils.run_cmd(v1_cmd)
        fov = utils.load_tardis_composites(tardis_dir)
        if fov is None:
            print('WARNING: TARDIS COMPOSITES COULD NOT BE LOADED, PAGES WILL SHOW THE NATIVE (UNEXTENDED) DATA')

    # GENERATE THE PAGES, IN THE ORDER OF dtiQA.pdf

    vis_files = []
    vis_files.append(vis.vis_preproc(dwi_raw_files, bvals_raw_files, dwi_preproc_file, bvals_preproc_file, eddy_mask_file, mask_file, percent_improbable, chisq_mask_file, vis_dir, fov=fov))
    if use_tardis:
        vis_files.extend(vis.vis_tardis(tardis_dir, vis_dir))
    vis_files.extend(vis.vis_dwi(dwi_preproc_file, bvals_preproc_shelled, bvecs_preproc_file, cnr_dict, vis_dir, mask_file=mask_file))
    vis_files.append(vis.vis_glyphs(tensor_file, v1_file, fa_file, cc_center_voxel, vis_dir, glyph_type=args.glyph_type, fov=fov))
    vis_files.append(vis.vis_scalar(fa_file, vis_dir, name='FA', comment='White matter should be bright', fov=fov))
    vis_files.append(vis.vis_fa_stats(roi_names, roi_med_fa, fa_file, atlas2subj_file, vis_dir, fov=fov))
    vis_files.append(vis.vis_scalar(md_file, vis_dir, name='MD', comment='CSF should be bright', fov=fov))

    pdf_file = utils.merge_pdfs(vis_files, 'dtiQA_check', vis_dir)

    print('CLEANING UP TEMP DIRECTORY')
    utils.remove_dir(temp_dir)

    print('PDF SAVED: {}'.format(pdf_file))

    return pdf_file

if __name__ == '__main__':
    main()
