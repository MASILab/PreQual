Bootstrap: docker-archive
From: /nfs2/harmonization/singularities/neuro_base_with_ants.tar

# ============================================================================
#  PreQual (dtiQA v7 Multi) v2.0.0
#
#  Base image: neuro_base_with_ants (Containers_Gaurav/brainstemseg/Dockerfile,
#  saved with `docker save`): Ubuntu 22.04 + MRtrix3 (/opt/mrtrix3) + FSL
#  (fslconda, /opt/fsl/fsl_install_dir) + FreeSurfer 8 (/opt/freesurfer) + ANTs
#  (/opt/ants) + uv (/opt/uv).
#
#  This recipe adds what PreQual needs on top of that:
#    - Convert3D (c3d_affine_tool, used by Synb0-DisCo)
#    - TARDIS FOV extension (scripts from Containers_Gaurav/TARDIS, checkpoints from models/tardis/)
#    - four uv-managed Python venvs (dtiQA, Synb0-DisCo, gradtensor, TARDIS)
#    - xvfb + ghostscript for PDF generation
#    - the PreQual source from this repository (build context = repo root)
#
#  BUILD (from the repository root, with the base tar next to this file or
#  pointed at by an absolute path in the From: line, and the two TARDIS
#  checkpoints in models/tardis/, see models/tardis/README.md):
#      sudo apptainer build prequal.sif Singularity
#  For a CPU-only torch in the TARDIS venv:
#      sudo APPTAINERENV_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu apptainer build ...
#
#  RUN: see README.md. Bind the FreeSurfer license to
#      /APPS/freesurfer/license.txt and pass --nv for --eddy_cuda on.
# ============================================================================

%files
    src/APPS/synb0                     /APPS/synb0
    src/APPS/gradtensor                /APPS/gradtensor
    src/APPS/tardis                    /APPS/tardis
    models/tardis/flow_difov_2d_basegd_multiB_best.pt   /APPS/tardis/checkpoints/flow_difov_2d_basegd_multiB_best.pt
    models/tardis/flow_difov_2d_multicond_t1_best.pt    /APPS/tardis/checkpoints/flow_difov_2d_multicond_t1_best.pt
    src/CODE/dtiQA_v7                  /CODE/dtiQA_v7
    src/CODE/run_dtiQA.sh              /CODE/run_dtiQA.sh
    src/SUPPLEMENTAL                   /SUPPLEMENTAL
    venv/pip_install_dtiQA.txt         /INSTALLERS/pip_install_dtiQA.txt
    venv/pip_install_synb0.txt         /INSTALLERS/pip_install_synb0.txt
    venv/pip_install_gradtensor.txt    /INSTALLERS/pip_install_gradtensor.txt
    venv/pip_install_tardis.txt        /INSTALLERS/pip_install_tardis.txt

%environment
    export DEBIAN_FRONTEND=noninteractive

    # MRtrix3 (base build)
    export MRTRIX3_HOME=/opt/mrtrix3
    export MRTRIX_TMPFILE_DIR=/tmp

    # FSL (fslconda)
    export FSLDIR=/opt/fsl/fsl_install_dir
    export FSLOUTPUTTYPE=NIFTI_GZ
    if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then . "${FSLDIR}/etc/fslconf/fsl.sh"; fi

    # FreeSurfer (SetUpFreeSurfer.sh is sourced in synb0.sh, the only step that needs it)
    export FREESURFER_HOME=/opt/freesurfer
    export FS_LICENSE=/APPS/freesurfer/license.txt

    # ANTs
    export ANTSPATH=/opt/ants/bin
    export LD_LIBRARY_PATH="/opt/ants/lib:${LD_LIBRARY_PATH:-}"

    # TARDIS FOV extension
    export TARDIS_HOME=/APPS/tardis

    # uv-managed Python interpreters
    export XDG_DATA_HOME=/opt/uv-data

    # matplotlib (all venvs) renders to files only; xvfb sets DISPLAY, which would otherwise select Tk
    export MPLBACKEND=Agg

    # Keep every per-user cache inside the container's /tmp so nothing is written to (or needs) a home
    # directory: matplotlib config/font cache, fontconfig, Qt (mrview under xvfb), uv. This lets the
    # image run with --no-home.
    export MPLCONFIGDIR=/tmp/prequal_mpl
    export XDG_CACHE_HOME=/tmp/prequal_cache
    export XDG_RUNTIME_DIR=/tmp/prequal_runtime
    export UV_CACHE_DIR=/tmp/prequal_uv

    export TMPDIR=/tmp
    export PATH="/opt/ants/bin:/opt/mrtrix3/bin:${FSLDIR}/bin:${FREESURFER_HOME}/bin:/APPS/c3d-1.0.0-Linux-x86_64/bin:/opt/uv:${PATH}"

%post
    set -eu
    export DEBIAN_FRONTEND=noninteractive
    export PATH="/opt/uv:${PATH}"
    export XDG_DATA_HOME=/opt/uv-data
    umask 022
    NPROC=$(nproc)

    mkdir -p /APPS /INSTALLERS /INPUTS /OUTPUTS /CODE /SUPPLEMENTAL /opt/uv-data

    # ---- Sanity-check the base image -------------------------------------------------------
    for t in /opt/mrtrix3/bin/dwidenoise /opt/mrtrix3/bin/mrdegibbs /opt/mrtrix3/bin/dwigradcheck \
             /opt/mrtrix3/bin/mrgrid /opt/mrtrix3/bin/transformconvert /opt/mrtrix3/bin/mrtransform \
             /opt/mrtrix3/bin/dwi2response /opt/mrtrix3/bin/dwi2fod /opt/mrtrix3/bin/sh2peaks /opt/mrtrix3/bin/tensor2metric \
             /opt/fsl/fsl_install_dir/bin/topup /opt/fsl/fsl_install_dir/bin/eddy_cpu /opt/fsl/fsl_install_dir/bin/epi_reg \
             /opt/fsl/fsl_install_dir/bin/convert_xfm /opt/fsl/fsl_install_dir/bin/flirt \
             /opt/freesurfer/bin/mri_convert /opt/freesurfer/bin/mri_synthstrip \
             /opt/ants/bin/antsApplyTransforms /opt/ants/bin/antsRegistrationSyNQuick.sh \
             /opt/ants/bin/DenoiseImage /opt/ants/bin/N4BiasFieldCorrection /opt/uv/uv; do
        [ -x "$t" ] || { echo "missing $t in base image" >&2; exit 1; }
    done
    if [ ! -x /opt/fsl/fsl_install_dir/bin/eddy_cuda ]; then
        echo "WARNING: eddy_cuda not found in base FSL; --eddy_cuda on will not work" >&2
    fi

    # ---- System packages ---------------------------------------------------------------------
    apt-get -y update
    apt-get -y install --no-install-recommends \
        xvfb xauth ghostscript \
        wget curl ca-certificates tar bzip2 git

    # ---- Convert3D 1.0.0 (c3d_affine_tool for Synb0-DisCo) -----------------------------------
    wget -q -O /INSTALLERS/c3d.tar.gz "https://downloads.sourceforge.net/project/c3d/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz"
    tar -xf /INSTALLERS/c3d.tar.gz -C /APPS/
    test -x /APPS/c3d-1.0.0-Linux-x86_64/bin/c3d_affine_tool

    # ---- FreeSurfer license location ---------------------------------------------------------
    # Users bind their license to /APPS/freesurfer/license.txt (same path as previous PreQual
    # versions). $FREESURFER_HOME/license.txt points at it so FreeSurfer tools find it too.
    mkdir -p /APPS/freesurfer
    echo "This is a dummy license file. Please bind your freesurfer license file to this file." > /APPS/freesurfer/license.txt
    ln -sfn /APPS/freesurfer/license.txt /opt/freesurfer/license.txt

    # ---- Python environments (uv) ------------------------------------------------------------
    # dtiQA: Python 3.8 with the legacy pinned scientific stack
    uv python install 3.8 3.12
    cd /CODE/dtiQA_v7
    rm -rf venv
    uv venv --python 3.8 venv
    uv pip install --python venv/bin/python -r /INSTALLERS/pip_install_dtiQA.txt
    venv/bin/python -c "import numpy, scipy, nibabel, matplotlib, skimage, sklearn, h5py; print('dtiQA venv ok')"

    # Synb0-DisCo: Python 3.12, CPU-only torch (inference runs on the CPU)
    cd /APPS/synb0
    rm -rf pytorch
    uv venv --python 3.12 pytorch
    uv pip install --python pytorch/bin/python --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python pytorch/bin/python -r /INSTALLERS/pip_install_synb0.txt
    pytorch/bin/python -c "import torch, nibabel, numpy; print('synb0 venv ok, torch', torch.__version__)"
    chmod +x /APPS/synb0/*.sh

    # gradtensor: Python 3.12. apply_gradtensor_to_b.py is a Python port of the former
    # MATLAB binary, so neither the MATLAB runtime nor scilpy is needed any more.
    cd /APPS/gradtensor
    rm -rf gradvenv
    uv venv --python 3.12 gradvenv
    uv pip install --python gradvenv/bin/python -r /INSTALLERS/pip_install_gradtensor.txt
    gradvenv/bin/python -c "import dipy, joblib, nibabel, scipy; print('gradtensor venv ok, dipy', dipy.__version__)"
    gradvenv/bin/python /APPS/gradtensor/apply_gradtensor_to_b.py --help > /dev/null
    chmod +x /APPS/gradtensor/*.sh /APPS/gradtensor/*.py

    # TARDIS: Python 3.12 with torch. The CUDA build from PyPI is installed by default and runs on the GPU
    # when the container is started with --nv (falls back to the CPU otherwise). Set
    # TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu (APPTAINERENV_TORCH_INDEX_URL) for a CPU-only image.
    cd /APPS/tardis
    rm -rf venv
    uv venv --python 3.12 venv
    if [ -n "${TORCH_INDEX_URL:-}" ]; then
        uv pip install --python venv/bin/python torch --index-url "${TORCH_INDEX_URL}"
    else
        uv pip install --python venv/bin/python torch
    fi
    uv pip install --python venv/bin/python -r /INSTALLERS/pip_install_tardis.txt
    venv/bin/python -c "import torch, dipy, nibabel, scipy, matplotlib, tqdm, joblib; print('tardis venv ok, torch', torch.__version__, 'cuda build:', torch.version.cuda)"
    for f in checkpoints/flow_difov_2d_basegd_multiB_best.pt checkpoints/flow_difov_2d_multicond_t1_best.pt; do
        [ -s "$f" ] || { echo "missing TARDIS checkpoint /APPS/tardis/$f" >&2; exit 1; }
    done
    for s in 001_plan_shells.py 002_build_volumes.py 003_extend_fov.py 004_write_outputs.py 005_native_outputs.py; do
        venv/bin/python "/APPS/tardis/$s" --help > /dev/null
    done
    chmod +x /APPS/tardis/*.sh /APPS/tardis/*.py

    # ---- Permissions -------------------------------------------------------------------------
    chmod +x /CODE/run_dtiQA.sh
    chmod 755 /INPUTS /SUPPLEMENTAL /APPS /CODE
    chmod 775 /OUTPUTS
    chmod -R a+rX /APPS /CODE /SUPPLEMENTAL /opt/uv-data

    # ---- Clean up ----------------------------------------------------------------------------
    cd /
    rm -rf /INSTALLERS
    apt-get -y clean
    rm -rf /var/lib/apt/lists/*

%runscript

    mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$XDG_RUNTIME_DIR" "$UV_CACHE_DIR" 2>/dev/null
    chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null
    cd /tmp # never run from the (possibly bound) home directory: MRtrix scripts default their scratch to the cwd
    xvfb-run -a --server-args="-screen 0 1600x1280x24 -ac" bash /CODE/run_dtiQA.sh /INPUTS /OUTPUTS "$@"

%test
    set -eu
    for t in dwidenoise mrdegibbs dwigradcheck dwi2tensor tensor2metric topup eddy_cpu bet epi_reg mri_convert mri_synthstrip antsApplyTransforms antsRegistrationSyNQuick.sh c3d_affine_tool xvfb-run gs; do
        command -v "$t" > /dev/null || { echo "missing $t" >&2; exit 1; }
    done
    /CODE/dtiQA_v7/venv/bin/python -c "import numpy, nibabel, matplotlib, skimage"
    /APPS/synb0/pytorch/bin/python -c "import torch, nibabel"
    /APPS/gradtensor/gradvenv/bin/python -c "import dipy, nibabel, scipy, joblib"
    /APPS/gradtensor/gradvenv/bin/python /APPS/gradtensor/apply_gradtensor_to_b.py --help > /dev/null
    /APPS/tardis/venv/bin/python -c "import torch, dipy, nibabel"
    test -s /APPS/tardis/checkpoints/flow_difov_2d_basegd_multiB_best.pt
    test -s /APPS/tardis/checkpoints/flow_difov_2d_multicond_t1_best.pt
    for t in mrgrid transformconvert mrtransform dwi2response dwi2fod sh2peaks convert_xfm flirt; do command -v "$t" > /dev/null; done
    echo "PreQual container tests passed"

%help
    PreQual (dtiQA v7 Multi): integrated preprocessing and QA for diffusion MRI.

    apptainer run -e --contain --home /path/to/inputs \
        -B /path/to/inputs:/INPUTS -B /path/to/outputs:/OUTPUTS -B /tmp:/tmp \
        -B /path/to/freesurfer/license.txt:/APPS/freesurfer/license.txt \
        [--nv] prequal.sif pe_axis [options]

    See README.md (github.com/MASILab/PreQual) for inputs, the configuration file and all options.
