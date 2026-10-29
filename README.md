# PreQual

Integrated preprocessing and quality assurance for diffusion MRI, in one container.

PreQual takes raw diffusion weighted images (NIfTI, bval, bvec), runs denoising, susceptibility and eddy current distortion correction, motion correction and optional bias field, gradient nonlinearity and field-of-view corrections, and produces a preprocessed dataset together with a PDF quality report.

![Pipeline Overview](https://github.com/MASILab/PreQual/blob/master/overview.png?raw=true)

This page is the quick start. Every option, processing step and output file is documented in the **[Advanced User Guide](docs/USER_GUIDE.md)**.

## Quick start

### 1. Get the container

Build from this repository (Apptainer, root permissions, see [Containerization](docs/USER_GUIDE.md#containerization-of-source-code) for details on the base image and the TARDIS checkpoints):

    git clone https://github.com/MASILab/PreQual.git
    cd PreQual
    cp /path/to/neuro_base_with_ants.tar .                 # base image (Docker archive)
    cp /path/to/flow_difov_2d_*_best.pt models/tardis/      # TARDIS checkpoints
    sudo apptainer build prequal.sif Singularity

### 2. Prepare an input directory

    inputs/
      dwi1.nii.gz  dwi1.bval  dwi1.bvec       one or more diffusion scans
      rpe.nii.gz   rpe.bval   rpe.bvec        optional reverse phase-encoded scan
      t1.nii.gz                               optional T1, enables Synb0-DisCo and TARDIS
      dtiQA_config.csv                        one line per scan: <prefix>,<pe_dir>,<readout_time>

Example `dtiQA_config.csv` for two forward scans and one reverse phase-encoded scan, phase encoded along `j` with a 0.05 s readout:

    dwi1,+,0.05
    dwi2,+,0.05
    rpe,-,0.05

`pe_dir` is `+` or `-` along the phase encoding axis; `readout_time` is the total readout time in seconds, between 0.01 and 0.2 s as FSL's eddy requires. See [Configuration File](docs/USER_GUIDE.md#configuration-file) and [Examples](docs/USER_GUIDE.md#examples).

### 3. Run

    apptainer run -e --contain --home /path/to/inputs \
      -B /path/to/inputs:/INPUTS \
      -B /path/to/outputs:/OUTPUTS \
      -B /tmp:/tmp \
      -B /path/to/freesurfer/license.txt:/APPS/freesurfer/license.txt \
      --nv \
      prequal.sif j [options]

* The positional argument is the phase encoding axis of all inputs, `i` or `j`.
* The FreeSurfer license bind is only needed when a T1 is supplied (Synb0-DisCo and TARDIS use FreeSurfer tools). Get one at <https://surfer.nmr.mgh.harvard.edu/fswiki/License>.
* `--nv` is optional and enables the GPU for `--eddy_cuda on` and for TARDIS. Everything runs on the CPU without it.

### 4. Results

    outputs/
      PREPROCESSED/dwmri.nii.gz  dwmri.bval  dwmri.bvec    final preprocessed data
      PDF/dtiQA.pdf                                        quality assurance report
      TARDIS/                                              field-of-view extension (if a T1 was supplied)

All other folders are described under [Outputs](docs/USER_GUIDE.md#outputs).

## Common options

Full list with defaults and explanations: [Options](docs/USER_GUIDE.md#options).

| Option | What it does | Default |
|---|---|---|
| `--synb0 raw/stripped/off` | Use the T1 to synthesize a distortion-free b0 (Synb0-DisCo) when no reverse phase-encoded scan exists | `raw` |
| `--tardis on/off` | Regenerate superior/inferior brain slices cut off by the acquisition with TARDIS (needs a T1) | `on` |
| `--eddy_cuda on/off` | Run eddy on the GPU (requires `--nv`) | `off` |
| `--denoise on/off`, `--degibbs on/off`, `--rician on/off` | MP-PCA denoising, Gibbs de-ringing, Rician correction | `on`, `off`, `off` |
| `--correct_bias on/off` | N4 bias field correction | `off` |
| `--correct_grad on/off` | Gradient nonlinearity correction (needs `gradtensor.nii` in the inputs) | `off` |
| `--num_threads N` | Threads for multi-threaded steps | `1` |
| `--extra_eddy_args`, `--extra_topup_args` | Pass extra arguments to eddy or topup, separated by `+` | none |
| `--keep_intermediates` | Keep intermediate files from every stage | off |
| `--project`, `--subject`, `--session` | Labels printed on the PDF | `proj`, `subj`, `sess` |

Run with `--help` for the complete list.

## What the pipeline does

1. Checks and thresholds b-values, verifies the gradient table.
2. MP-PCA denoising, optional Gibbs de-ringing and Rician correction, inter-scan intensity normalization.
3. Susceptibility distortion correction with topup, using a reverse phase-encoded scan or a Synb0-DisCo synthetic b0 from the T1.
4. Eddy current and motion correction with FSL eddy, including slice-wise outlier replacement.
5. Optional N4 bias field correction and gradient nonlinearity correction.
6. Brain masking, tensor fitting, FA/MD/AD/RD/V1 maps, goodness-of-fit, SNR/CNR and motion statistics, atlas-based FA checks.
7. Optional TARDIS field-of-view extension of the preprocessed data, with its own outputs and QA pages.
8. A PDF report covering each step.

Details: [Processing Steps](docs/USER_GUIDE.md#pipeline-processing-steps) and [Quality Assurance Steps](docs/USER_GUIDE.md#pipeline-quality-assurance-steps). Input requirements and caveats: [Pipeline Assumptions](docs/USER_GUIDE.md#pipeline-assumptions). Running BIDS-organized data without copying: [Running BIDS Data](docs/USER_GUIDE.md#running-bids-data).

## Versions

v2.0.0 rebuilds the container on Ubuntu 22.04 with current FSL, FreeSurfer 8, MRtrix3 and ANTs, replaces the MATLAB gradient nonlinearity binary with Python, adds TARDIS, and uses SynthStrip for T1 skull stripping. Outputs are therefore not numerically identical to v1.1.x and earlier. See the change list in the [Advanced User Guide](docs/USER_GUIDE.md#containerization-of-source-code) and the [note for VUIIS XNAT users](docs/USER_GUIDE.md#note-on-versioning-for-vuiis-xnat-users).

## Citation

If you use PreQual, please cite:

Leon Y. Cai, Qi Yang, Colin B. Hansen, Vishwesh Nath, Karthik Ramadass, Graham W. Johnson, Benjamin N. Conrad, Brian D. Boyd, John P. Begnoche, Lori L. Beason-Held, Andrea T. Shafer, Susan M. Resnick, Warren D. Taylor, Gavin R. Price, Victoria L. Morgan, Baxter P. Rogers, Kurt G. Schilling, Bennett A. Landman. *PreQual: An automated pipeline for integrated preprocessing and quality assurance of diffusion weighted MRI images*. [Magnetic Resonance in Medicine](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28678), 2021.

If Synb0-DisCo was used (`--synb0`, a T1 and no reverse phase-encoded scan), please also cite:

Kurt G. Schilling, Justin Blaber, Yuankai Huo, Allen Newton, Colin Hansen, Vishwesh Nath, Andrea T. Shafer, Owen Williams, Susan M. Resnick, Baxter Rogers, Adam W. Anderson, Bennett A. Landman. *Synthesized b0 for diffusion distortion correction (Synb0-DisCo)*. [Magnetic Resonance Imaging](https://doi.org/10.1016/j.mri.2019.05.008), 64, 62-70, 2019.

Kurt G. Schilling, Justin Blaber, Colin Hansen, Leon Cai, Baxter Rogers, Adam W. Anderson, Seth Smith, Praitayini Kanakaraj, Tonia Rex, Susan M. Resnick, Lori L. Beason-Held, Bennett A. Landman. *Distortion correction of diffusion weighted MRI without reverse phase-encoding scans or field-maps*. [PLoS ONE](https://doi.org/10.1371/journal.pone.0236418), 15(7): e0236418, 2020.

If the TARDIS field-of-view extension was used (`--tardis`), please also cite:

Gaurav Rudravaram, Lianrui Zuo, Ema Topolnjak, Michael E. Kim, Adam M. Saunders, Elyssa M. McMaster, Paula Trujillo, Ely E. Wesley, Colin D. McKnight, Pratik P. Pandharipande, Mayur B. Patel, Cayce Strength, Brenda T. Pund, Amelia W. Maiga, Robert S. Dittus, David A. Bennett, Julie A. Schneider, Konstantinos Arfanakis, The HABS-HD Study Team, Derek B. Archer, Timothy J. Hohman, Yelena Bodien, Daniel C. Moyer, Kurt G. Schilling, Bennett A. Landman. *Generalizable field of view extension for diffusion MRI via conditional flow matching*. [arXiv:2610.17401](https://arxiv.org/abs/2610.17401), 2026.

[Medical-image Analysis and Statistical Interpretation (MASI) Lab](https://my.vanderbilt.edu/masi), Vanderbilt University, Nashville, TN, USA. Licensed under the terms in [LICENSE.md](LICENSE.md).
