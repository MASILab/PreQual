# PreQual Outputs

## If you just want the preprocessed diffusion MRI

Grab these three things from the output directory and you are done:

| Path | What it is |
|---|---|
| `PREPROCESSED/dwmri.nii.gz`, `dwmri.bval`, `dwmri.bvec` | The preprocessed diffusion data, all input scans concatenated, with eddy-rotated b-vectors in FSL format. This is the input to tractography, model fitting, or any further analysis. |
| `PREPROCESSED/mask.nii.gz` | The brain mask used by the pipeline, on the same grid. |
| `PDF/dtiQA.pdf` | The quality report. Read it before trusting the data. See [How to QA](QA_GUIDE.md). |

If you are interested in the individual outputs of the various PreQual options, read on. The rest of this page is the exhaustive list: every folder and every file the pipeline can write, in the order it writes them.

## Other files you may want

Depending on what you are doing:

| Path | When |
|---|---|
| `SCALARS/dwmri_tensor_fa.nii.gz`, `_md`, `_ad`, `_rd`, `_v1` | You want DTI maps without fitting a tensor yourself. |
| `STATS/stats.csv` | You want per-scan motion, CNR and regional FA numbers in a table, for example to screen a cohort. |
| `EDDY/eddy_results.eddy_outlier_map` and the other `eddy_results.*` text files | You want eddy's own motion and outlier bookkeeping. |
| `TARDIS/PREPROCESSED/dwmri_extended.nii.gz`, `.bval`, `.bvec` | A T1 was supplied and you want the field-of-view-extended diffusion data: a synthetic DWI on the native grid padded along the slice axis, laid out like PREPROCESSED. |
| `TARDIS/SCALARS/dwmri_extended_tensor_fa.nii.gz`, `_md`, `_ad`, `_rd`, `_v1` | DTI maps of the extended data, fit like the PreQual ones. |
| `TARDIS/COMPOSITE/fa.nii.gz`, `md`, `tensor`, `v1`, `mask` | The PreQual maps where the data were acquired and the TARDIS maps where slices were regenerated, in one volume. This is what the PDF shows. |

Everything else is intermediate. Most of it is only present when PreQual is run with `--keep_intermediates`.

## Folder by folder

Folders appear in the order the pipeline creates them. "Kept" means the folder survives a normal run; "intermediate" means it is removed unless `--keep_intermediates` is given. `<image>` stands for the prefix of an input scan as listed in `dtiQA_config.csv`; `<image_%>` is that prefix with suffixes appended by each step that ran (for example `dwi1_denoised_norm`).

### THRESHOLDED_BVALS (kept)

One `.bval` file per input scan with every value below `--bval_threshold` (default 50) set to 0. These are the b-values the rest of the pipeline treats as b0.

### CHECKED (intermediate)

`<image>_checked.nii.gz / .bval / .bvec` per scan. Volumes whose b-value and b-vector passed the consistency checks (a zero b-vector must have a zero b-value and vice versa). Volumes that fail are dropped and reported on the title page of the PDF.

### DENOISED (intermediate, only if `--denoise on`)

`<image_%>_denoised.nii.gz` per scan after MP-PCA denoising, plus `<image_%>_noise.nii.gz`, the estimated noise map that Rician correction uses.

### DEGIBBS (intermediate, only if `--degibbs on`)

`<image_%>_degibbs.nii.gz` per scan after Gibbs ringing removal.

### RICIAN (intermediate, only if `--rician on`)

`<image_%>_rician.nii.gz` per scan after Rician bias correction by the method of moments.

### PRENORMALIZED or GAIN_CHECK (intermediate)

`<image_%>_norm.nii.gz` per scan. With `--prenormalize on` (default) each scan's intensity is scaled to match the first scan's b0 histogram, and the folder is called PRENORMALIZED. With it off, the same scaling is computed but only reported, and the folder is called GAIN_CHECK. Gain differences above 5% are flagged in the PDF.

### TOPUP (kept, only if topup ran)

Topup runs when a reverse phase-encoded scan was supplied, or when a T1 was supplied and Synb0-DisCo generated a synthetic undistorted b0.

| File | What it is |
|---|---|
| `acqparams.txt` | Phase encoding direction and readout time for every b0 topup used, one line per b0. With Synb0, the last line is the synthetic b0 with readout time 0. |
| `b0_syn.nii.gz` | The Synb0-DisCo synthetic b0 (Synb0 runs only). |
| `preproc_input_b0_*.nii.gz` | The b0 volumes given to topup (`_all` or `_first_b0s_only` depending on `--topup_first_b0s_only`), with `_smooth_with_b0_syn` when the synthetic b0 was appended. |
| `*_topped_up.nii.gz` | The same b0s after topup correction. |
| `topup_field.nii.gz` | The estimated susceptibility field in Hz. |
| `topup_results_fieldcoef.nii.gz`, `topup_results_movpar.txt` | Topup's spline field coefficients and movement parameters, consumed by eddy. |
| `*.topup_log` | Topup's log. |

### EDDY (kept)

| File | What it is |
|---|---|
| `acqparams.txt` | Phase encoding and readout time per acquired b0. Unlike the topup file, this one never contains the synthetic Synb0 line, because current FSL eddy rejects readout times outside 0.01 to 0.2 s. |
| `index.txt` | For each volume of the merged input, the line of `acqparams.txt` that applies to it. |
| `preproc_input.nii.gz / .bval / .bvec` (intermediate) | All scans merged into the single 4D input eddy received. |
| `preproc_input_eddyed.nii.gz / .bval / .bvec` (intermediate) | Eddy's corrected output, renamed. The `.bvec` file is eddy's rotated b-vectors. |
| `eddy_mask.nii.gz` | The brain mask eddy was given (`--eddy_mask on`, the default). |
| `eddy_results.eddy_command_txt` | The exact eddy command, useful for checking which executable and options ran. |
| `eddy_results.eddy_parameters` | Per volume: 3 translations, 3 rotations, then the eddy current terms. |
| `eddy_results.eddy_movement_rms` | Per volume: RMS displacement relative to the first volume, and relative to the previous volume. |
| `eddy_results.eddy_restricted_movement_rms` | Same, excluding translation along the phase encoding axis, which is confounded by the susceptibility field. |
| `eddy_results.eddy_outlier_map` | A 0/1 matrix, one row per volume and one column per slice. 1 marks a slice eddy judged a signal-dropout outlier and replaced. |
| `eddy_results.eddy_outlier_n_stdev_map`, `_n_sqr_stdev_map` | How far each slice's mean signal fell below the prediction, in standard deviations. |
| `eddy_results.eddy_outlier_report` | Human-readable list of outlier slices. |
| `eddy_results.eddy_outlier_free_data.nii.gz` | The input data with outlier slices replaced but before motion correction. |
| `eddy_results.eddy_cnr_maps.nii.gz` | SNR of the b0s (first volume) and CNR of each shell (following volumes). |
| `eddy_results.eddy_rotated_bvecs` | The b-vectors after eddy's rotation correction. This is what becomes `PREPROCESSED/dwmri.bvec`. |
| `eddy_results.eddy_post_eddy_shell_alignment_parameters`, `eddy_results.eddy_post_eddy_shell_PE_translation_parameters` | Residual alignment between shells after correction, overall and along the phase encoding axis. |
| `eddy_results.eddy_values_of_all_input_parameters` | Every eddy parameter, including defaults. |

### POSTNORMALIZED (intermediate, only if `--postnormalize on`)

`<image_%>_topup_eddy_norm.nii.gz` per scan. A deprecated second intensity normalization after eddy; off by default.

### UNBIASED (only if `--correct_bias on`)

`*_unbiased.nii.gz`, the data after N4 bias field correction, and `bias_field.nii.gz`, the estimated multiplicative field.

### GRADNONLINEAR_CORRECTED (only if `--correct_grad on`)

Requires `gradtensor.nii` in the input directory.

| File | What it is |
|---|---|
| `Lcorrected_sig.nii.gz` | The diffusion signal corrected for gradient nonlinearity. This is what feeds PREPROCESSED. |
| `L_resamp.nii.gz` | The gradient tensor resampled onto the diffusion grid, 9 volumes (Lxx, Lxy, ..., Lzz). NaN where the diffusion grid lies outside the tensor's field of view. |
| `gradtensor_fa.nii.gz` | FA of the resampled tensor, for the PDF. |
| `bvec_XXXX.nii.gz`, `bval_XXXX.nii.gz` | Voxelwise corrected b-vector (3 volumes) and b-value images, one pair per diffusion volume. |
| `org.bval`, `org.bvec` | Copy of the b-table that was corrected. |

### PREPROCESSED (kept)

The final output.

| File | What it is |
|---|---|
| `dwmri.nii.gz`, `dwmri.bval`, `dwmri.bvec` | All input scans, concatenated in `dtiQA_config.csv` order, after every preprocessing step that ran. The b-vectors are eddy's rotated vectors in FSL convention. |
| `<image>_preproc.nii.gz / .bval / .bvec` | Only with `--split_outputs`: the same data split back into one file per input scan. The merged `dwmri.*` files are then removed. |
| `mask.nii.gz` | Brain mask from `bet` on the average b0, `-f 0.25 -R`. Used for the tensor fit and all statistics. |
| `improbable_mask.nii.gz` | Only with `--improbable_mask on`: the brain mask with voxels removed where the minimum b0 signal is lower than the minimum diffusion-weighted signal. |

### TENSOR (kept)

| File | What it is |
|---|---|
| `dwmri_tensor.nii.gz` | The MRtrix3 tensor fit, 6 volumes in MRtrix order: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz. Fit inside `mask.nii.gz` with iteratively reweighted least squares. |
| `dwmri_recon.nii.gz` (intermediate) | The signal predicted by that tensor, used for the goodness-of-fit statistics. |

### SCALARS (kept)

`dwmri_tensor_fa.nii.gz`, `_md`, `_ad`, `_rd` and `_v1.nii.gz`: fractional anisotropy, mean, axial and radial diffusivity (mm²/s), and the principal eigenvector (3 volumes, unit length). All computed from `TENSOR/dwmri_tensor.nii.gz` inside the mask.

### STATS (kept)

| File | What it is |
|---|---|
| `stats.csv` | One `name,value` row per statistic: average rotations and translations per axis, average absolute and relative displacement (all from eddy, in degrees and mm), median SNR of the b0s and median CNR per shell inside the mask, and median FA in each of the 48 JHU white matter tract ROIs. The quickest file to screen a cohort with. |
| `eddy_avg_rotations.txt`, `eddy_avg_translations.txt`, `eddy_avg_abs_displacement.txt`, `eddy_avg_rel_displacement.txt` | The motion numbers above, one file each. |
| `eddy_median_cnr.txt` | Median SNR/CNR per shell. |
| `roi_avg_fa.txt` | Median FA per JHU ROI. |
| `chisq_mask.nii.gz` | The mask used for goodness of fit: the brain mask eroded by one voxel with CSF removed (FSL FAST on the per-shell mean images, CSF probability above 15%). |
| `chisq_matrix.txt` | Modified chi-squared of the tensor fit, one row per slice and one column per volume. High values mark slices and volumes the tensor explains poorly. |
| `atlas2subj.nii.gz` | The JHU ICBM white matter atlas labels warped into subject space. |
| `fa2template_*` or `b02template_*` | The ANTs transforms from subject to the JHU template (affine, warp, inverse warp), named after `--atlas_reg_type`. |

### OPTIMIZED_BVECS (kept)

`dwmri.bval` and `dwmri.bvec` as permuted and sign-flipped by MRtrix3's `dwigradcheck`. These are for QA only: if they differ from `PREPROCESSED/dwmri.bvec`, the gradient table may be misoriented. The PDF's gradient check page shows the comparison. Do not use these files for analysis.

### TARDIS (kept, only if a T1 was supplied and `--tardis on`, the default)

The field-of-view extension. Everything here is additional to PREPROCESSED, which is never modified. The folder mirrors the PreQual output: PREPROCESSED, TENSOR and SCALARS hold the extended data and its tensor fit on the native grid padded along the slice axis (same voxel size and orientation as PREPROCESSED, more slices), COMPOSITE merges the PreQual and TARDIS maps, MODEL_SPACE holds the model-grid versions and LOGS everything else.

| Path | What it is |
|---|---|
| `PREPROCESSED/dwmri_extended.nii.gz`, `.bval`, `.bvec`, `dwmri_extended_grad.txt` | One synthetic DWI for all TARDIS-processed shells: a constant b0 (intensity `S0`, 1000 by default) followed by every processed shell on the 30 MRtrix directions the FOD model works with (tensor model: the shell's own directions). Shells TARDIS did not process (fewer than 28 directions under the FOD model, or all but the chosen shell under the tensor model) are not included, so the b-values are a subset of `PREPROCESSED/dwmri.bval`. `S = S0 exp(-b ADC)` from the model's ADC; see the note below. The FSL bvec is for this file's orientation, the grad.txt is the same table in MRtrix world convention. |
| `PREPROCESSED/mask_extended.nii.gz` | The T1 brain mask on the padded grid, used for the tensor fit below. |
| `PREPROCESSED/regenerated_slices.nii.gz` | 1 where the model produced the data, 0 where it is the (band-limited) acquired signal. Use it to tell synthetic from measured voxels. |
| `TENSOR/dwmri_extended_tensor.nii.gz` | `dwi2tensor` on `dwmri_extended` inside `mask_extended`, exactly as PreQual fits `PREPROCESSED/dwmri`. MRtrix order Dxx, Dyy, Dzz, Dxy, Dxz, Dyz. |
| `SCALARS/dwmri_extended_tensor_fa.nii.gz`, `_md`, `_ad`, `_rd`, `_v1` | FA, mean, axial and radial diffusivity (mm²/s) and the principal eigenvector of that tensor, same commands as PreQual's SCALARS. |
| `COMPOSITE/fa.nii.gz`, `md`, `tensor`, `v1`, `mask` | Ready-to-use extended maps on the padded native grid: the PreQual result where the data were acquired, the TARDIS result where slices were regenerated. `mask` is the PreQual mask with the TARDIS brain mask in the regenerated slices. These are what the brain figures in the PDF show (preprocessed masks, tensor glyphs, FA, MD, FA atlas page). |
| `COMPOSITE/fov_boundary.nii.gz`, `regenerated_slab.nii.gz`, `acquired_slab.nii.gz`, `composite.json` | Marker volumes (1 on the first regenerated slice at each extended end, on every regenerated slice, on every acquired slice) and the description of the composites: slice axis, padding, acquired slice range, regenerated and boundary slice indices. The blue FOV line in the PDF is drawn from `fov_boundary`. |
| `MODEL_SPACE/` | On the TARDIS grid (2x2x1 mm, canonical RAS): `T1w`, `brain_mask`, `dwi_coverage` (where acquired signal exists), `b0_mean`, `regenerated_slices`, the same synthetic `dwmri_extended.*` before resampling, `fa.nii.gz` (its FA, shown beside the T1 on the TARDIS pages), and what the model produced per shell: `adc_b<label>.nii.gz` (30-direction ADC, FOD model) or `adc6_b<label>.nii.gz` and `tensor_b<label>.nii.gz` (tensor model). |
| `LOGS/summary.txt` | One-screen summary: model, shells, how many slices were regenerated at each end, output folders. Read this first. |
| `LOGS/plan.json` | Shell clustering and which model was chosen (FOD model for shells with 28 or more directions, tensor model otherwise). |
| `LOGS/fov.json`, `fov_slices.csv` | Where the T1 brain starts and ends, where the acquired DWI starts and ends, which slices were regenerated, and the per-slice fraction of brain voxels with valid signal (overall and b0 only). Slices inside the acquired footprint that were regenerated because they were only partly valid are listed under `partial_slices`. |
| `LOGS/fov_native.json` | The padding of the native grid (slice axis, slices added before and after) and which padded-native slices are regenerated. |
| `LOGS/inference.json`, `meta.json` | Device, ODE steps, seed, timing; model grid and shell metadata. |
| `LOGS/t1_preproc/`, `T1w_mask.nii.gz`, `b0_mean.nii.gz` | The T1 after Rician denoising and N4 bias correction (plus the bias field), its SynthStrip brain mask, and the mean b0 the T1 was registered to. |
| `LOGS/reg/` | `epi_reg` outputs: the b0-to-T1 affine and its inverse, the world-space version, and the T1 and mask resampled onto the padded b0 grid. |
| `LOGS/QA/` | PNG copies of the TARDIS pages of the PDF, plus one overview figure per processed shell. |

Two things to keep in mind when using `dwmri_extended`. Its b0 is a constant with no T2 contrast, because the model works on ADC; use single-shell or per-shell methods on it, not multi-tissue CSD. And inside the acquired field of view it is the measured signal band-limited to lmax 6 and re-expressed on the model directions, not the raw measurements; for those slices use PREPROCESSED, or use the COMPOSITE maps, which keep the PreQual values there.

### PDF (kept)

`dtiQA.pdf`, the quality report. The per-page PDFs it was assembled from (`title.pdf`, `pedir.pdf`, `preproc.pdf`, `stats.pdf`, `gradcheck.pdf`, `fa.pdf`, `md.pdf`, `tardis_*.pdf` and so on) sit in the same folder. The pages and what to look for in them are described in [How to QA](QA_GUIDE.md).
