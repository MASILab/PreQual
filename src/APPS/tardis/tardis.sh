#!/bin/bash
# TARDIS FOV extension inside PreQual.
#
# Usage: tardis.sh <dwi.nii.gz> <dwi.bval> <dwi.bvec> <t1.nii.gz> <out_dir> <num_threads>
#                  [<mask.nii.gz> <tensor.nii.gz> <fa.nii.gz> <md.nii.gz>]
#
# Regenerates the superior/inferior brain slices cut off by the diffusion acquisition with the TARDIS
# rectified-flow models (FOD model for shells with >= 28 directions, tensor model otherwise) and writes
# the result in the same shape as PreQual's own output:
#
#   <out_dir>/PREPROCESSED/  dwmri_extended.nii.gz/.bval/.bvec  one synthetic DWI (constant b0 + every processed
#                            shell) on the native grid padded along the slice axis; mask_extended, regenerated_slices
#   <out_dir>/TENSOR/        dwmri_extended_tensor.nii.gz       fit with dwi2tensor like PreQual
#   <out_dir>/SCALARS/       dwmri_extended_tensor_{fa,md,ad,rd,v1}.nii.gz
#   <out_dir>/COMPOSITE/     PreQual maps where acquired, TARDIS maps where regenerated (what the PDF shows);
#                            only written when the four optional native PreQual files are given
#   <out_dir>/MODEL_SPACE/   the same on the 2x2x1 mm RAS model grid, plus the per-shell ADC the model produced
#   <out_dir>/LOGS/          plan, fov, meta, inference, summary, registration, T1 preprocessing, QA figures
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
for c in flirt epi_reg convert_xfm fslmaths mrgrid mrinfo transformconvert mrtransform dwi2tensor tensor2metric DenoiseImage N4BiasFieldCorrection; do need "$c"; done

# MRtrix: -nthreads 0 disables multithreading, so the thread count given to it is THREADS-1 (as in PreQual)
MR_THREADS=$(( THREADS > 1 ? THREADS - 1 : 0 ))

WORK=$OUT/work; LOGS=$OUT/LOGS; REG=$LOGS/reg; MS=$OUT/MODEL_SPACE
PRE=$OUT/PREPROCESSED; TEN=$OUT/TENSOR; SCA=$OUT/SCALARS; COMP=$OUT/COMPOSITE
mkdir -p "$WORK" "$LOGS" "$REG" "$MS" "$PRE" "$TEN" "$SCA" "$LOGS/QA"
GPU=0; command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && GPU=1
log "dwi $DWI | t1 $T1 | out $OUT | threads $THREADS | gpu visible: $GPU"

# ── 0. T1 preprocessing: Rician denoising + N4 bias field correction (ANTs) ─────
# Everything downstream (SynthStrip, epi_reg, the T1 conditioning channel of the model) uses this T1.
log "=== 0/10 T1 denoising + N4 bias field correction ==="
T1P=$LOGS/t1_preproc
mkdir -p "$T1P"
T1_DENOISED=$T1P/T1w_denoised.nii.gz
T1_PREPROC=$T1P/T1w_denoised_n4.nii.gz
DenoiseImage -d 3 -i "$T1" -o "$T1_DENOISED" -n Rician -s 1 -p 1 -r 2 -v 0
N4BiasFieldCorrection -d 3 -i "$T1_DENOISED" -o "[$T1_PREPROC,$T1P/T1w_bias_field.nii.gz]" \
    -s 4 -b [200] -c [50x50x50x50,0.0000001] -v 0
[[ -s "$T1_PREPROC" ]] || die "T1 preprocessing produced no output"
T1=$T1_PREPROC

# ── 1. shells, model choice, mean b0 ─────────────────────────────────────────
log "=== 1/10 gradient table: shells + model choice ==="
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
B0=$LOGS/b0_mean.nii.gz
if [[ "$MODEL" == "fod" ]]; then CKPT=$FOD_CKPT; else CKPT=$TENSOR_CKPT; fi
[[ -s "$CKPT" ]] || die "checkpoint for the $MODEL model not found: $CKPT"

# ── 2. T1 brain mask (SynthStrip, fallback bet) ──────────────────────────────
log "=== 2/10 T1 brain mask ==="
MASK=$LOGS/T1w_mask.nii.gz
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
log "=== 3/10 T1 -> b0 registration ==="
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
log "=== 4/10 model-space volumes ($MODEL model) ==="
"$PY" "$TARDIS_HOME/002_build_volumes.py" --plan "$LOGS/plan.json" --dwi "$DWI" --bval "$BVAL" --bvec "$BVEC" \
    --t1 "$T1_DWI" --mask "$MASK_DWI" --dirs "$DIRS" --out-dir "$OUT" \
    --coverage-frac "$COVERAGE_FRAC" --min-brain-voxels "$MIN_BRAIN_VOXELS" --erase-edge "$ERASE_EDGE" --threads "$THREADS"

# ── 5. inference ─────────────────────────────────────────────────────────────
log "=== 5/10 FOV extension ==="
ext_args=(--out-dir "$OUT" --checkpoint "$CKPT" --device "$DEVICE" --seed "$SEED" --threads "$THREADS")
[[ -n "$ODE_STEPS" ]] && ext_args+=(--ode-steps "$ODE_STEPS")
"$PY" "$TARDIS_HOME/003_extend_fov.py" "${ext_args[@]}"

# ── 6. one synthetic DWI (all processed shells) + ADC per shell on the model grid ─
log "=== 6/10 writing model-grid outputs ==="
"$PY" "$TARDIS_HOME/004_write_outputs.py" --out-dir "$OUT" --s0 "$S0"
cp -f "$WORK/inference.json" "$WORK/meta.json" "$LOGS/"

# ── 7. FA on the model grid (for the TARDIS pages of the PDF, which show it beside the registered T1) ─
log "=== 7/10 tensor fit on the model grid ==="
BMASK=$MS/brain_mask.nii.gz
dwi2tensor "$MS/dwmri_extended.nii.gz" "$WORK/tensor_model_grid.nii.gz" -grad "$MS/dwmri_extended_grad.txt" -mask "$BMASK" \
    -nthreads "$MR_THREADS" -force -quiet
tensor2metric "$WORK/tensor_model_grid.nii.gz" -fa "$MS/fa.nii.gz" -mask "$BMASK" -nthreads "$MR_THREADS" -force -quiet

# ── 8. native-grid outputs: PREPROCESSED/dwmri_extended.* on the padded native grid ─
log "=== 8/10 resampling the extended data onto the native DWI grid ==="
"$PY" "$TARDIS_HOME/005_native_outputs.py" --dwi "$DWI" --tardis-dir "$OUT"

# ── 9. tensor + scalars on the native grid, the way PreQual fits PREPROCESSED/dwmri ─
log "=== 9/10 tensor fit and scalars on the native grid ==="
EXT=$PRE/dwmri_extended; EXT_MASK=$PRE/mask_extended.nii.gz; EXT_TENSOR=$TEN/dwmri_extended_tensor.nii.gz
dwi2tensor "$EXT.nii.gz" "$EXT_TENSOR" -fslgrad "$EXT.bvec" "$EXT.bval" -mask "$EXT_MASK" -nthreads "$MR_THREADS" -force -quiet
tensor2metric "$EXT_TENSOR" -fa "$SCA/dwmri_extended_tensor_fa.nii.gz" -mask "$EXT_MASK" -nthreads "$MR_THREADS" -force -quiet
tensor2metric "$EXT_TENSOR" -adc "$SCA/dwmri_extended_tensor_md.nii.gz" -mask "$EXT_MASK" -nthreads "$MR_THREADS" -force -quiet
tensor2metric "$EXT_TENSOR" -ad "$SCA/dwmri_extended_tensor_ad.nii.gz" -mask "$EXT_MASK" -nthreads "$MR_THREADS" -force -quiet
tensor2metric "$EXT_TENSOR" -rd "$SCA/dwmri_extended_tensor_rd.nii.gz" -mask "$EXT_MASK" -nthreads "$MR_THREADS" -force -quiet
tensor2metric "$EXT_TENSOR" -vector "$SCA/dwmri_extended_tensor_v1.nii.gz" -num 1 -modulate none -mask "$EXT_MASK" -nthreads "$MR_THREADS" -force -quiet

# ── 10. composites: PreQual metric where acquired, TARDIS where regenerated (padded native grid) ──
if [[ -n "$NATIVE_MASK" && -f "$NATIVE_MASK" && -f "$NATIVE_TENSOR" && -f "$NATIVE_FA" && -f "$NATIVE_MD" ]]; then
    log "=== 10/10 composite volumes on the padded native grid ==="
    "$PY" "$TARDIS_HOME/006_composite_outputs.py" --mask "$NATIVE_MASK" --tensor "$NATIVE_TENSOR" \
        --fa "$NATIVE_FA" --md "$NATIVE_MD" --tardis-dir "$OUT"
    tensor2metric "$COMP/tensor.nii.gz" -vector "$COMP/v1.nii.gz" -num 1 -modulate none -mask "$COMP/mask.nii.gz" \
        -nthreads "$MR_THREADS" -force -quiet
else
    log "=== 10/10 composites skipped (native mask/tensor/FA/MD not given) ==="
fi

rm -rf "$WORK"
log "done. see $LOGS/summary.txt"
cat "$LOGS/summary.txt"
