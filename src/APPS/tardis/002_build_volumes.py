#!/usr/bin/env python
"""
002: resample the DWI, the registered T1 and its brain mask onto the model grid (037's grid: canonical RAS,
2x2x1 mm, 128x128 in-plane, z extended so the whole T1 brain fits), build the model input (FOD model:
tournier07 lmax-6 SH fit of each selected shell projected onto the 30 mrtrix directions -> ADC, x1000, clip 3.5,
[-1,1]; tensor model: dipy WLS tensor on the selected shell -> ADC along 6 icosahedral directions, x1000) and
locate the acquired DWI footprint inside the T1 brain (the slices to regenerate).
Inputs : --plan plan.json --dwi --bval --bvec --t1 reg/T1w_in_dwi.nii.gz --mask reg/T1w_mask_in_dwi.nii.gz --dirs 30_dir_mrtrix_dirgen.txt
Writes : OUT/work/model_input.npz, OUT/work/meta.json, OUT/fov.json, OUT/fov_slices.csv,
         OUT/model_space/{T1w,brain_mask,dwi_coverage,b0_mean}.nii.gz
Invoke : python 002_build_volumes.py --plan P --dwi D --bval B --bvec V --t1 T --mask M --dirs F --out-dir OUT
         [--coverage-frac 0.9] [--min-brain-voxels 1] [--erase-edge 0] [--z-margin 2] [--threads N]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from joblib import Parallel, delayed
from nibabel.affines import apply_affine
from tqdm import tqdm

VOX = (2.0, 2.0, 1.0)          # 037 grid (x, y, z) mm, canonical RAS
CROP = 128                     # 037 in-plane crop/pad
B0_THRESHOLD = 50.0
LMAX, SH_BASIS, SMOOTH = 6, "tournier07", 0.006      # 037 SH fit
ADC_CLIP, SIG_FLOOR = 4e-3, 1e-6                      # 037 ADC clip (mm^2/s), signal floor
CLIP_MAX, ADC_SCALE = 3.5, 1000.0                     # preprocess.py: x1000, clip 3.5 -> [-1,1]
DIRECTIONS6 = np.array([        # util_scripts/006: 6 icosahedral directions of the tensor model
    (0.0000, 0.5257, 0.8507), (0.0000, 0.5257, -0.8507), (0.5257, 0.8507, 0.0000),
    (0.5257, -0.8507, 0.0000), (0.8507, 0.0000, 0.5257), (-0.8507, 0.0000, 0.5257)])


def load_gradients(bval_path, bvec_path, n_vols):
    bvals = np.loadtxt(bval_path).ravel().astype(float)
    bvecs = np.loadtxt(bvec_path).astype(float)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3 or bvecs.shape == (3, 3):
        bvecs = bvecs.T
    assert len(bvals) == n_vols and bvecs.shape == (n_vols, 3)
    return bvals, bvecs


def world_dirs(bvecs, affine):
    """FSL bvecs (voxel axes, x negated when det > 0) -> unit vectors in world (scanner RAS) coordinates."""
    R = affine[:3, :3]
    Rn = R / np.linalg.norm(R, axis=0, keepdims=True)
    d = bvecs.copy()
    if np.linalg.det(R) > 0:
        d[:, 0] *= -1
    w = d @ Rn.T
    n = np.linalg.norm(w, axis=1, keepdims=True)
    return np.where(n > 0, w / np.maximum(n, 1e-12), 0.0)


def model_grid(dwi_img, mask_img, z_margin):
    """037's resample_to_output grid (ref) and the model grid: ref cropped/padded to 128x128 in-plane (037's
    fit_in_plane) and extended in z to cover the registered T1 mask. -> (ref_img, shape, affine, info).
    DWI volumes are resampled onto ref exactly as in training and embedded by integer offsets (embed())."""
    ref = nibproc.resample_to_output(nib.Nifti1Image(np.zeros(dwi_img.shape[:3], np.float32), dwi_img.affine),
                                     voxel_sizes=VOX, order=1)
    X, Y, Z = ref.shape
    A = ref.affine
    ox = (X - CROP) // 2 if X > CROP else -((CROP - X) // 2)
    oy = (Y - CROP) // 2 if Y > CROP else -((CROP - Y) // 2)
    ijk = np.argwhere(np.asarray(mask_img.dataobj) > 0.5)
    if ijk.size == 0:
        sys.exit("registered T1 mask is empty")
    vz = apply_affine(np.linalg.inv(A), apply_affine(mask_img.affine, ijk))[:, 2]
    z0 = min(0, int(np.floor(vz.min())) - z_margin)
    z1 = max(Z - 1, int(np.ceil(vz.max())) + z_margin)
    aff = A.copy()
    aff[:3, 3] = apply_affine(A, np.array([ox, oy, z0], float))
    shape = (CROP, CROP, z1 - z0 + 1)
    info = dict(ref_shape=[int(X), int(Y), int(Z)], inplane_offset=[int(ox), int(oy)], z_offset=int(z0),
                dwi_z_range_in_grid=[int(-z0), int(-z0 + Z - 1)])
    return ref, shape, aff, info


def fit_in_plane(vol, size=CROP):
    """037: center-crop / zero-pad axes 0,1 of an (X, Y, ...) array to size x size."""
    X, Y = vol.shape[0], vol.shape[1]
    if X > size:
        x0 = (X - size) // 2
        vol = vol[x0:x0 + size]
    if Y > size:
        y0 = (Y - size) // 2
        vol = vol[:, y0:y0 + size]
    X2, Y2 = vol.shape[0], vol.shape[1]
    px, py = max(0, size - X2), max(0, size - Y2)
    if px or py:
        pad = [(px // 2, px - px // 2), (py // 2, py - py // 2)] + [(0, 0)] * (vol.ndim - 2)
        vol = np.pad(vol, pad, mode="constant", constant_values=0)
    return vol


def embed(vol_ref, shape, info):
    """(Xref, Yref, Zref[, ...]) on 037's grid -> (128, 128, Zext[, ...]) on the model grid, zeros elsewhere."""
    v = fit_in_plane(vol_ref)
    out = np.zeros(shape + v.shape[3:], v.dtype)
    z0 = info["dwi_z_range_in_grid"][0]
    out[:, :, z0:z0 + v.shape[2]] = v
    return out


def resample(data, affine, grid, order):
    img = nib.Nifti1Image(np.ascontiguousarray(data, dtype=np.float32), affine)
    return np.asarray(nibproc.resample_from_to(img, grid, order=order).dataobj, dtype=np.float32)


def fov_analysis(brain, covered, min_vox, cov_frac, erase):
    """Per-slice brain / coverage -> known FOV and brain extent along z (model grid). -> (fov dict, rows)."""
    nb = brain.sum(axis=(0, 1))
    nc = (brain & covered).sum(axis=(0, 1))
    frac = np.where(nb > 0, nc / np.maximum(nb, 1), 0.0)
    is_brain = nb >= min_vox
    bz = np.where(is_brain)[0]
    if bz.size == 0:
        sys.exit(f"no slice with >= {min_vox} brain voxels")
    kz = np.where(is_brain & (frac >= cov_frac))[0]
    if kz.size == 0:
        sys.exit(f"no brain slice with >= {cov_frac:.0%} DWI coverage: check the registration")
    z_lo_b, z_hi_b = int(bz[0]), int(bz[-1])
    z_lo_k, z_hi_k = int(kz[0]) + erase, int(kz[-1]) - erase
    if z_hi_k - z_lo_k + 1 < 2:
        sys.exit(f"known FOV of {z_hi_k - z_lo_k + 1} slices after --erase-edge {erase} is too thin")
    holes = [int(z) for z in range(z_lo_k, z_hi_k + 1) if is_brain[z] and frac[z] < cov_frac]
    sup = list(range(z_hi_k + 1, z_hi_b + 1))
    inf = list(range(z_lo_k - 1, z_lo_b - 1, -1))
    fov = dict(z_lo_brain=z_lo_b, z_hi_brain=z_hi_b, z_lo_known=z_lo_k, z_hi_known=z_hi_k,
               n_known=z_hi_k - z_lo_k + 1, extend_superior=sup, extend_inferior=inf,
               n_extend_superior=len(sup), n_extend_inferior=len(inf),
               mm_extend_superior=len(sup) * VOX[2], mm_extend_inferior=len(inf) * VOX[2],
               coverage_frac=cov_frac, min_brain_voxels=min_vox, erase_edge=erase,
               interior_slices_below_coverage=holes)
    rows = [(z, int(nb[z]), int(nc[z]), round(float(frac[z]), 4), int(z_lo_k <= z <= z_hi_k),
             int(z in sup or z in inf)) for z in range(len(nb))]
    return fov, rows


def shell_adc30(data, idx, dirs, bvals, S0, brain, mrtrix30):
    """037's shell ADC on the brain voxels: SH fit -> 30 mrtrix directions -> ADC (mm^2/s), clipped."""
    b = float(bvals[idx].mean())
    sig = data[..., idx][brain]                                        # (n_vox, n_dirs)
    sh = sf_to_sh(sig, Sphere(xyz=dirs[idx]), sh_order_max=LMAX, basis_type=SH_BASIS, smooth=SMOOTH)
    s30 = np.clip(sh_to_sf(sh, Sphere(xyz=mrtrix30), sh_order_max=LMAX, basis_type=SH_BASIS), SIG_FLOOR, None)
    adc = -1.0 / b * np.log(s30 / S0[brain][:, None])
    adc = np.clip(np.nan_to_num(adc, nan=0.0), 0.0, ADC_CLIP).astype(np.float32)
    vol = np.zeros(brain.shape + (mrtrix30.shape[0],), np.float32)
    vol[brain] = adc
    return vol, b


def normalise_adc(vol):
    """mm^2/s (0 outside brain) -> [-1,1] float32, z-major (Z, X, Y, nd); background -> -1."""
    v = np.clip(vol * ADC_SCALE, 0.0, CLIP_MAX) / CLIP_MAX * 2.0 - 1.0
    return np.ascontiguousarray(np.moveaxis(v, 2, 0)).astype(np.float32)


def tensor_adc6(data, idx, b0_idx, dirs, bvals, fit_mask):
    """007's tensor -> ADC6: dipy WLS on b0s + the shell, quadratic form -> ADC along the 6 directions, x1000."""
    sel = np.array(list(b0_idx) + list(idx))
    gtab = gradient_table(bvals[sel], bvecs=dirs[sel], b0_threshold=B0_THRESHOLD)
    fit = TensorModel(gtab, fit_method="WLS").fit(np.nan_to_num(data[..., sel], nan=0.0), mask=fit_mask)
    q = np.nan_to_num(fit.quadratic_form, nan=0.0)
    u = DIRECTIONS6
    adc6 = np.einsum("...ij,di,dj->...d", q, u, u) * ADC_SCALE
    return np.nan_to_num(adc6, nan=0.0).astype(np.float32) * fit_mask[..., None], float(bvals[idx].mean())


def main(args):
    out = Path(args.out_dir)
    work, ms = out / "work", out / "model_space"
    work.mkdir(parents=True, exist_ok=True)
    ms.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan).read_text())
    model = plan["model"]
    shells = [s for s in plan["shells"] if s["label"] in plan["selected"]]
    b0_idx = plan["b0_indices"]

    dwi_img, t1_img, mask_img = nib.load(args.dwi), nib.load(args.t1), nib.load(args.mask)
    bvals, bvecs = load_gradients(args.bval, args.bvec, dwi_img.shape[3])
    wdirs = world_dirs(bvecs, dwi_img.affine)
    ref, shape, aff, info = model_grid(dwi_img, mask_img, args.z_margin)
    grid = (shape, aff)
    print(f"model grid {shape} @ {VOX} mm; DWI occupies z {info['dwi_z_range_in_grid']} of {shape[2]}", flush=True)

    needed = sorted(set(b0_idx) | {i for s in shells for i in s["indices"]})
    dwi = np.asarray(dwi_img.dataobj, dtype=np.float32)
    vols = Parallel(n_jobs=args.threads, prefer="threads")(
        delayed(resample)(dwi[..., t], dwi_img.affine, ref, 1)
        for t in tqdm(needed, desc="resample DWI", unit="vol"))
    data = np.zeros(shape + (dwi.shape[3],), np.float32)
    for t, v in zip(needed, vols):
        data[..., t] = embed(v, shape, info)
    del dwi, vols
    footprint = embed(resample(np.ones(dwi_img.shape[:3], np.float32), dwi_img.affine, ref, 0), shape, info) > 0.5
    S0 = np.clip(data[..., b0_idx].mean(axis=-1), SIG_FLOOR, None)
    b0m = data[..., b0_idx].mean(axis=-1)
    covered = footprint & (b0m > 0)

    brain = resample(np.asarray(mask_img.dataobj, dtype=np.float32), mask_img.affine, grid, 0) > 0.5
    t1 = np.nan_to_num(resample(np.asarray(t1_img.dataobj, dtype=np.float32), t1_img.affine, grid, 1), nan=0.0)
    t1[~brain] = 0.0
    fov, rows = fov_analysis(brain, covered, args.min_brain_voxels, args.coverage_frac, args.erase_edge)
    p99 = float(np.percentile(t1[:, :, fov["z_lo_brain"]:fov["z_hi_brain"] + 1], 99))
    p99 = p99 if np.isfinite(p99) and p99 > 0 else 1.0
    t1n = np.ascontiguousarray(np.moveaxis(np.clip(t1 / p99, 0.0, 1.0) * 2.0 - 1.0, 2, 0)).astype(np.float32)

    arrays, meta_shells = {}, []
    fit_mask = brain & covered
    mrtrix30 = np.loadtxt(args.dirs, comments="#")
    mrtrix30 /= np.linalg.norm(mrtrix30, axis=1, keepdims=True)
    for s in tqdm(shells, desc=f"{model} input", unit="shell"):
        idx = np.array(s["indices"])
        if model == "fod":
            vol, b = shell_adc30(data, idx, wdirs, bvals, S0, fit_mask, mrtrix30)
            arrays[f"adc_{s['label']}"] = normalise_adc(vol)
        else:
            adc6, b = tensor_adc6(data, idx, b0_idx, wdirs, bvals, fit_mask)
            arrays[f"adc_{s['label']}"] = np.ascontiguousarray(np.moveaxis(adc6, 2, 0)).astype(np.float32)
        meta_shells.append(dict(label=s["label"], b_mean=b, n_dirs=int(len(idx)), indices=s["indices"],
                                bvals=[float(v) for v in bvals[idx]], dirs_world=wdirs[idx].tolist()))
    meta = dict(model=model, shape=list(shape), affine=aff.tolist(), vox=list(VOX), grid=info, t1_p99=p99,
                clip_max=CLIP_MAX, adc_scale=ADC_SCALE, shells=meta_shells,
                dirs30=mrtrix30.tolist(), dirs6=DIRECTIONS6.tolist())
    np.savez(work / "model_input.npz", t1=t1n, brain=np.moveaxis(brain, 2, 0), covered=np.moveaxis(covered, 2, 0),
             **arrays)
    (work / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "fov.json").write_text(json.dumps(fov, indent=2))
    with (out / "fov_slices.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "brain_voxels", "covered_voxels", "covered_frac", "known_fov", "regenerated"])
        w.writerows(rows)
    for name, arr, dt in (("T1w", t1, np.float32), ("brain_mask", brain, np.uint8),
                          ("dwi_coverage", covered, np.uint8), ("b0_mean", b0m * brain, np.float32)):
        nib.save(nib.Nifti1Image(arr.astype(dt), aff), ms / f"{name}.nii.gz")
    print(f"brain z {fov['z_lo_brain']}..{fov['z_hi_brain']} | acquired z {fov['z_lo_known']}..{fov['z_hi_known']} | "
          f"regenerate {fov['n_extend_inferior']} inferior + {fov['n_extend_superior']} superior slices "
          f"({fov['mm_extend_inferior']:.0f} + {fov['mm_extend_superior']:.0f} mm)", flush=True)
    if fov["interior_slices_below_coverage"]:
        print(f"WARNING: slices {fov['interior_slices_below_coverage']} inside the acquired FOV are below "
              f"{args.coverage_frac:.0%} coverage", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True)
    p.add_argument("--dwi", required=True)
    p.add_argument("--bval", required=True)
    p.add_argument("--bvec", required=True)
    p.add_argument("--t1", required=True, help="T1 registered onto the (padded) b0 grid")
    p.add_argument("--mask", required=True, help="T1 brain mask on the same grid")
    p.add_argument("--dirs", required=True, help="30_dir_mrtrix_dirgen.txt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--coverage-frac", type=float, default=0.9)
    p.add_argument("--min-brain-voxels", type=int, default=1)
    p.add_argument("--erase-edge", type=int, default=0)
    p.add_argument("--z-margin", type=int, default=2, help="extra slices beyond the T1 brain in the model grid")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
