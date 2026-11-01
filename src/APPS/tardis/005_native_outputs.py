#!/usr/bin/env python
"""
005: resample the extended data from the model grid onto the native DWI grid, padded along its
superior-inferior axis so that the regenerated slices fit. Voxel size and orientation match the native
DWI; only the slice count differs. This gives TARDIS the same shape of output as PreQual itself.
Inputs : --dwi PREPROCESSED/dwmri.nii.gz --tardis-dir OUT (reads OUT/MODEL_SPACE/{dwmri_extended.*,brain_mask,
         regenerated_slices,T1w}.nii.gz and OUT/LOGS/fov.json)
Writes : OUT/PREPROCESSED/dwmri_extended.{nii.gz,bval,bvec}, dwmri_extended_grad.txt, mask_extended.nii.gz,
         regenerated_slices.nii.gz, OUT/LOGS/fov_native.json
Invoke : python 005_native_outputs.py --dwi D --tardis-dir OUT
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
from nibabel.affines import apply_affine
from tqdm import tqdm


def fsl_bvecs(dirs_world, affine):
    """World unit vectors -> FSL bvec convention for an image with this affine (voxel axes, x negated if det > 0)."""
    R = affine[:3, :3]
    Rn = R / np.linalg.norm(R, axis=0, keepdims=True)
    v = dirs_world @ np.linalg.inv(Rn).T
    if np.linalg.det(R) > 0:
        v[:, 0] *= -1
    return v


def padded_native_grid(native_img, model_img, fov):
    """Native grid extended along its superior-inferior axis to cover the model-grid brain extent."""
    A = native_img.affine.copy()
    shape = list(native_img.shape[:3])
    # native axis most aligned with world z
    axis = int(np.argmax(np.abs(A[:3, :3][2, :]) / np.linalg.norm(A[:3, :3], axis=0)))
    # model-grid corners of the brain-extent slab -> native voxel coordinates
    X, Y = model_img.shape[0], model_img.shape[1]
    z0, z1 = fov["z_lo_brain"], fov["z_hi_brain"]
    corners = np.array([[x, y, z] for x in (0, X - 1) for y in (0, Y - 1) for z in (z0, z1)], float)
    vox = apply_affine(np.linalg.inv(A), apply_affine(model_img.affine, corners))[:, axis]
    pad_before = int(max(0, np.ceil(-vox.min())))
    pad_after = int(max(0, np.ceil(vox.max() - (shape[axis] - 1))))
    A[:3, 3] = A[:3, 3] - pad_before * A[:3, axis]
    shape[axis] += pad_before + pad_after
    return tuple(shape), A, dict(slice_axis=axis, pad_before=pad_before, pad_after=pad_after,
                                 native_shape=list(native_img.shape[:3]), padded_shape=shape)


def resample(img, grid, order):
    return np.asarray(nibproc.resample_from_to(img, grid, order=order, mode="constant", cval=0.0).dataobj, dtype=np.float32)


def main(args):
    tardis = Path(args.tardis_dir)
    ms, logs, pre = tardis / "MODEL_SPACE", tardis / "LOGS", tardis / "PREPROCESSED"
    pre.mkdir(parents=True, exist_ok=True)
    fov = json.loads((logs / "fov.json").read_text())

    native = nib.load(args.dwi)
    model_t1 = nib.load(ms / "T1w.nii.gz")
    shape, aff, info = padded_native_grid(native, model_t1, fov)
    grid = (shape, aff)
    print(f"native grid {native.shape[:3]} -> padded {shape} along axis {info['slice_axis']} "
          f"(+{info['pad_before']} before, +{info['pad_after']} after)", flush=True)

    # masks (nearest neighbour)
    for name, src in (("mask_extended", ms / "brain_mask.nii.gz"), ("regenerated_slices", ms / "regenerated_slices.nii.gz")):
        arr = resample(nib.load(src), grid, 0)
        nib.save(nib.Nifti1Image((arr > 0.5).astype(np.uint8), aff), pre / f"{name}.nii.gz")

    # synthetic DWI (linear), volume by volume, with its gradient table expressed for the native axes
    dwi_img = nib.load(ms / "dwmri_extended.nii.gz")
    vols = [resample(nib.Nifti1Image(np.asarray(dwi_img.dataobj[..., t], dtype=np.float32), dwi_img.affine), grid, 1)
            for t in tqdm(range(dwi_img.shape[3]), desc="resample DWI", unit="vol")]
    nib.save(nib.Nifti1Image(np.stack(vols, axis=-1), aff), pre / "dwmri_extended.nii.gz")
    grad = np.loadtxt(ms / "dwmri_extended_grad.txt", comments="#")
    dirs_world, bvals = grad[:, :3], grad[:, 3]
    np.savetxt(pre / "dwmri_extended.bval", bvals[None], fmt="%.1f")
    np.savetxt(pre / "dwmri_extended.bvec", fsl_bvecs(dirs_world, aff).T, fmt="%.8f")
    np.savetxt(pre / "dwmri_extended_grad.txt", grad, fmt="%.8f", header="MRtrix gradient table: world (scanner RAS) x y z b")

    # which padded-native slices are regenerated
    regen = np.asarray(nib.load(pre / "regenerated_slices.nii.gz").dataobj) > 0
    other = tuple(i for i in range(3) if i != info["slice_axis"])
    info["regenerated_slice_indices"] = [int(i) for i in np.where(regen.sum(axis=other) > 0)[0]]
    info["n_volumes"] = int(dwi_img.shape[3])
    info["source_fov"] = fov
    (logs / "fov_native.json").write_text(json.dumps(info, indent=2))
    print(f"native outputs -> {pre}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dwi", required=True, help="native preprocessed DWI (defines the grid)")
    p.add_argument("--tardis-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
