#!/bin/bash
# TARDIS FOV extension inside PreQual.
#
# Usage: tardis.sh <dwi.nii.gz> <dwi.bval> <dwi.bvec> <t1.nii.gz> <out_dir> <num_threads>
#                  [<mask.nii.gz> <tensor.nii.gz> <fa.nii.gz> <md.nii.gz>]
#
# With the four optional native PreQual files, composite volumes (native metric where acquired, TARDIS
# where regenerated, on the padded native grid) are written to <out_dir>/native for the PDF and for users.
#
# Regenerates the superior/inferior brain slices cut off by the diffusion acquisition with the TARDIS
# rectified-flow models (FOD model for shells with >= 28 directions, tensor model otherwise), writes
# the extended data on the TARDIS model grid (2x2x1 mm, RAS) and resampled onto the native DWI grid
# padded along the slice axis, and fits CSD / tensor metrics (FA, MD) on the extended data.
#
# Exit codes: 0 ok | 3 no shell with enough directions for either model | other = failure
#
# Adapted from Containers_Gaurav/TARDIS/pipeline.sh. Runs on the GPU when the container is started
# with --nv and a CUDA device is visible, otherwise on the CPU.
set -euo pipefail

DWI=$1
BVAL=$2
BVEC=$3
T1=$4
OUT=$5
THREADS=${6:-1}
NATIVE_MASK=${7:-}
NATIVE_TENSOR=${8:-}
NATIVE_FA=${9:-}
NATIVE_MD=${10:-}

TARDIS_HOME=${TARDIS_HOME:-/APPS/tardis}
PY=${TARDIS_PY:-$TARDIS_HOME/venv/bin/python}
FOD_CKPT=${TARDIS_FOD_CKPT:-$TARDIS_HOME/checkpoints/flow_difov_2d_basegd_multiB_best.pt}
TENSOR_CKPT=${TARDIS_TENSOR_CKPT:-$TARDIS_HOME/checkpoints/flow_difov_2d_multicond_t1_best.pt}
DIRS=$TARDIS_HOME/30_dir_mrtrix_dirgen.txt

# TARDIS parameters (defaults from the standalone container)
ODE_STEPS=${TARDIS_ODE_STEPS:-}          # empty = 20 (FOD) / 15 (tensor)
MIN_DIRS_FOD=${TARDIS_MIN_DIRS_FOD:-28}
MIN_DIRS_TENSOR=${TARDIS_MIN_DIRS_TENSOR:-6}
TENSOR_BMAX=${TARDIS_TENSOR_BMAX:-1500}
COVERAGE_FRAC=${TARDIS_COVERAGE_FRAC:-0.9}
MIN_BRAIN_VOXELS=${TARDIS_MIN_BRAIN_VOXELS:-1}
ERASE_EDGE=${TARDIS_ERASE_EDGE:-0}
PAD_MM=${TARDIS_PAD_MM:-80}
SEED=${TARDIS_SEED:-0}
S0=${TARDIS_S0:-1000}
DEVICE=${TARDIS_DEVICE:-auto}

export TMPDIR=${TMPDIR:-/tmp}
export MRTRIX_TMPFILE_DIR=${MRTRIX_TMPFILE_DIR:-$TMPDIR}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$TMPDIR/tardis_mpl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$TMPDIR/tardis_cache}
export FSLOUTPUTTYPE=NIFTI_GZ
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$MRTRIX_TMPFILE_DIR" 2>/dev/null || true

log() { echo "[TARDIS] $*"; }
die() { echo "[TARDIS][ERROR] $*" >&2; exit "${2:-1}"; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1" 127; }

for f in "$DWI" "$BVAL" "$BVEC" "$T1"; do [[ -f "$f" ]] || die "missing input: $f" 2; done
[[ -x "$PY" ]] || die "python venv not found at $PY"
for c in flirt epi_reg convert_xfm fslmaths mrgrid mrinfo transformconvert mrtransform dwi2tensor tensor2metric dwi2response dwi2fod sh2peaks mrcalc mrcat DenoiseImage N4BiasFieldCorrection; do need "$c"; done

WORK=$OUT/work; REG=$OUT/reg; MS=$OUT/model_space
mkdir -p "$WORK" "$REG" "$MS" "$OUT/QA"
GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && GPU=1
log "dwi $DWI | t1 $T1 | out $OUT | threads $THREADS | gpu visible: $GPU"

# ── 0. T1 preprocessing: Rician denoising + N4 bias field correction (ANTs) ─────
# Everything downstream (SynthStrip, epi_reg, the T1 conditioning channel of the model) uses this T1.
log "=== 0/8 T1 denoising + N4 bias field correction ==="
T1P=$OUT/t1_preproc
mkdir -p "$T1P"
T1_DENOISED=$T1P/T1w_denoised.nii.gz
T1_PREPROC=$T1P/T1w_denoised_n4.nii.gz
DenoiseImage -d 3 -i "$T1" -o "$T1_DENOISED" -n Rician -s 1 -p 1 -r 2 -v 0
N4BiasFieldCorrection -d 3 -i "$T1_DENOISED" -o "[$T1_PREPROC,$T1P/T1w_bias_field.nii.gz]" \
    -s 4 -b [200] -c [50x50x50x50,0.0000001] -v 0
[[ -s "$T1_PREPROC" ]] || die "T1 preprocessing produced no output"
T1=$T1_PREPROC

# ── 1. shells, model choice, mean b0 ─────────────────────────────────────────
log "=== 1/8 gradient table: shells + model choice ==="
set +e
"$PY" "$TARDIS_HOME/001_plan_shells.py" --dwi "$DWI" --bval "$BVAL" --bvec "$BVEC" --out-dir "$OUT" \
    --min-dirs-fod "$MIN_DIRS_FOD" --min-dirs-tensor "$MIN_DIRS_TENSOR" --tensor-bmax "$TENSOR_BMAX"
rc=$?
set -e
if [[ $rc -eq 3 ]]; then
    die "no shell with >= $MIN_DIRS_TENSOR directions; TARDIS cannot run on this acquisition" 3
elif [[ $rc -ne 0 ]]; then
    die "001_plan_shells.py failed (exit $rc)" "$rc"
fi
MODEL=$(cat "$WORK/model.txt")
B0=$OUT/b0_mean.nii.gz
if [[ "$MODEL" == "fod" ]]; then CKPT=$FOD_CKPT; else CKPT=$TENSOR_CKPT; fi
[[ -s "$CKPT" ]] || die "checkpoint for the $MODEL model not found: $CKPT"

# ── 2. T1 brain mask (SynthStrip, fallback bet) ──────────────────────────────
log "=== 2/8 T1 brain mask ==="
MASK=$OUT/T1w_mask.nii.gz
ss_ok=0
if command -v mri_synthstrip >/dev/null 2>&1; then
    if [[ $GPU -eq 1 ]]; then mri_synthstrip -i "$T1" -m "$MASK" -g && ss_ok=1 || log "synthstrip on GPU failed, retrying on CPU"; fi
    [[ $ss_ok -eq 1 ]] || { mri_synthstrip -i "$T1" -m "$MASK" && ss_ok=1 || true; }
fi
if [[ $ss_ok -ne 1 || ! -s "$MASK" ]]; then
    log "WARNING: mri_synthstrip unavailable or failed; falling back to bet -R"
    bet "$T1" "$WORK/T1w_bet" -m -R -f 0.5 && mv "$WORK/T1w_bet_mask.nii.gz" "$MASK"
fi
[[ -s "$MASK" ]] || die "no T1 brain mask produced"

# ── 3. T1 -> b0 registration (epi_reg), resampled onto a padded b0 grid ──────
log "=== 3/8 T1 -> b0 registration ==="
T1_DWI=$REG/T1w_in_dwi.nii.gz; MASK_DWI=$REG/T1w_mask_in_dwi.nii.gz; XFM=$REG/t1_to_dwi.mat
fslmaths "$T1" -mas "$MASK" "$REG/T1w_brain.nii.gz"
epi_reg --epi="$B0" --t1="$T1" --t1brain="$REG/T1w_brain.nii.gz" --out="$REG/epi2struct" -v
convert_xfm -omat "$XFM" -inverse "$REG/epi2struct.mat"
# pad every axis of the b0 grid by PAD_MM so the head outside the DWI FOV survives the resampling
read -r -a SP < <(mrinfo -spacing "$B0")
pad_args=()
for ax in 0 1 2; do
    n=$(awk -v mm="$PAD_MM" -v s="${SP[$ax]}" 'BEGIN{printf "%d", (mm/s)+0.999}')
    pad_args+=(-axis "$ax" "$n,$n")
done
mrgrid "$B0" pad "${pad_args[@]}" "$REG/b0_padded.nii.gz" -force -quiet
# flirt matrices live in each image's own voxel frame, so they cannot be applied with a padded reference;
# convert to a world-space matrix and regrid with MRtrix.
transformconvert "$XFM" "$T1" "$B0" flirt_import "$REG/t1_to_dwi_world.txt" -force -quiet
mrtransform "$T1" -linear "$REG/t1_to_dwi_world.txt" -template "$REG/b0_padded.nii.gz" -interp linear "$T1_DWI" -force -quiet
mrtransform "$MASK" -linear "$REG/t1_to_dwi_world.txt" -template "$REG/b0_padded.nii.gz" -interp nearest "$MASK_DWI" -force -quiet

# ── 4. model grid + model input ──────────────────────────────────────────────
log "=== 4/8 model-space volumes ($MODEL model) ==="
"$PY" "$TARDIS_HOME/002_build_volumes.py" --plan "$OUT/plan.json" --dwi "$DWI" --bval "$BVAL" --bvec "$BVEC" \
    --t1 "$T1_DWI" --mask "$MASK_DWI" --dirs "$DIRS" --out-dir "$OUT" \
    --coverage-frac "$COVERAGE_FRAC" --min-brain-voxels "$MIN_BRAIN_VOXELS" --erase-edge "$ERASE_EDGE" --threads "$THREADS"

# ── 5. inference ─────────────────────────────────────────────────────────────
log "=== 5/8 FOV extension ==="
ext_args=(--out-dir "$OUT" --checkpoint "$CKPT" --device "$DEVICE" --seed "$SEED" --threads "$THREADS")
[[ -n "$ODE_STEPS" ]] && ext_args+=(--ode-steps "$ODE_STEPS")
"$PY" "$TARDIS_HOME/003_extend_fov.py" "${ext_args[@]}"

# ── 6. NIfTI outputs on the model grid ───────────────────────────────────────
log "=== 6/8 writing model-grid outputs ==="
"$PY" "$TARDIS_HOME/004_write_outputs.py" --out-dir "$OUT" --s0 "$S0"
cp -f "$WORK/inference.json" "$WORK/meta.json" "$OUT/"

# ── 7. fits: CSD (FOD model) or tensor metrics (tensor model), plus FA/MD for every output ─
log "=== 7/8 fits ==="
BMASK=$MS/brain_mask.nii.gz
while read -r kind label dir; do
    [[ -n "$kind" ]] || continue
    if [[ "$kind" == "fod" ]]; then
        log "b=$label: dwi2response tournier -> dwi2fod csd (lmax 6) -> sh2peaks"
        dwi2response tournier "$dir/dwi_extended.nii.gz" "$dir/response.txt" -grad "$dir/grad.txt" -mask "$BMASK" \
            -lmax 6 -scratch "$TMPDIR" -nthreads "$THREADS" -force -quiet
        dwi2fod csd "$dir/dwi_extended.nii.gz" "$dir/response.txt" "$dir/fod.nii.gz" -grad "$dir/grad.txt" -mask "$BMASK" \
            -lmax 6 -nthreads "$THREADS" -force -quiet
        sh2peaks "$dir/fod.nii.gz" "$dir/peaks.nii.gz" -num 3 -mask "$BMASK" -nthreads "$THREADS" -force -quiet
        log "b=$label: dwi2tensor -> FA / MD of the extended synthetic DWI"
        dwi2tensor "$dir/dwi_extended.nii.gz" "$dir/tensor_extended.nii.gz" -grad "$dir/grad.txt" -mask "$BMASK" -nthreads "$THREADS" -force -quiet
        tensor2metric "$dir/tensor_extended.nii.gz" -fa "$dir/fa.nii.gz" -adc "$dir/md.nii.gz" -mask "$BMASK" -nthreads "$THREADS" -force -quiet
    else
        log "b=$label: tensor2metric (FA, MD, V1) + DTI peaks [V1*FA,0,0]"
        tensor2metric "$dir/tensor_extended.nii.gz" -fa "$dir/fa.nii.gz" -adc "$dir/md.nii.gz" -vector "$dir/v1.nii.gz" \
            -mask "$BMASK" -nthreads "$THREADS" -force -quiet
        tensor2metric "$dir/tensor_extended.nii.gz" -vector "$WORK/v1fa.nii.gz" -modulate FA -mask "$BMASK" -nthreads "$THREADS" -force -quiet
        mrcalc "$WORK/v1fa.nii.gz" 0 -mult "$WORK/zeros3.nii.gz" -force -quiet
        mrcat "$WORK/v1fa.nii.gz" "$WORK/zeros3.nii.gz" "$WORK/zeros3.nii.gz" "$dir/peaks_dti.nii.gz" -axis 3 -force -quiet
    fi
done < "$WORK/outputs.txt"

# Primary output (used for the QA pages and the native-grid outputs): the selected shell closest to b=1000
PRIMARY=$(awk 'BEGIN{best=""; bd=1e9} {d=$2-1000; if (d<0) d=-d; if (d<bd) {bd=d; best=$0}} END{print best}' "$WORK/outputs.txt")
echo "$PRIMARY" > "$OUT/primary_output.txt"
PRIMARY_DIR=$(echo "$PRIMARY" | awk '{print $3}')
log "primary output: $PRIMARY"

# ── 8. native-grid outputs ───────────────────────────────────────────────────
log "=== 8/8 resampling the extended data onto the native DWI grid ==="
"$PY" "$TARDIS_HOME/005_native_outputs.py" --dwi "$DWI" --tardis-dir "$OUT" --primary-dir "$PRIMARY_DIR" --out-dir "$OUT/native"

# ── 9. composites: PreQual metric where acquired, TARDIS where regenerated (padded native grid) ──
if [[ -n "$NATIVE_MASK" && -f "$NATIVE_MASK" && -f "$NATIVE_TENSOR" && -f "$NATIVE_FA" && -f "$NATIVE_MD" ]]; then
    log "=== 9/9 composite volumes on the padded native grid ==="
    "$PY" "$TARDIS_HOME/006_composite_outputs.py" --dwi "$DWI" --bval "$BVAL" --mask "$NATIVE_MASK" --tensor "$NATIVE_TENSOR" \
        --fa "$NATIVE_FA" --md "$NATIVE_MD" --tardis-dir "$OUT" --native-dir "$OUT/native"
    tensor2metric "$OUT/native/tensor_composite.nii.gz" -vector "$OUT/native/v1_composite.nii.gz" -num 1 -modulate none \
        -mask "$OUT/native/mask_composite.nii.gz" -nthreads "$THREADS" -force -quiet
else
    log "=== 9/9 composites skipped (native mask/tensor/FA/MD not given) ==="
fi

rm -rf "$WORK"
log "done. see $OUT/summary.txt"
cat "$OUT/summary.txt"
