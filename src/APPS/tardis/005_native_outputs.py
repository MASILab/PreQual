#!/usr/bin/env python
"""
005: resample the TARDIS outputs of the primary shell from the model grid (2x2x1 mm RAS) back onto the
native preprocessed DWI grid, padded along the native slice (superior-inferior) axis so that the regenerated
slices are kept. Voxel size and orientation of the native grid are preserved; only the slice axis grows.
Inputs : --dwi PREPROCESSED/dwmri.nii.gz --tardis-dir OUT --primary-dir OUT/<kind>_b<label>
Writes : OUT/native/{fa,md,adc_mean,brain_mask,regenerated_slices,dwi}_extended.nii.gz (+ .bval/.bvec/grad.txt
         for the DWI), OUT/native/fov_native.json
Invoke : python 005_native_outputs.py --dwi D --tardis-dir OUT --primary-dir P --out-dir OUT/native
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np
from nibabel.affines import apply_affine


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
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tardis = Path(args.tardis_dir)
    primary = Path(args.primary_dir)
    fov = json.loads((tardis / "fov.json").read_text())

    native = nib.load(args.dwi)
    model_t1 = nib.load(tardis / "model_space" / "T1w.nii.gz")
    shape, aff, info = padded_native_grid(native, model_t1, fov)
    grid = (shape, aff)
    print(f"native grid {native.shape[:3]} -> padded {shape} along axis {info['slice_axis']} "
          f"(+{info['pad_before']} before, +{info['pad_after']} after)", flush=True)

    # scalar maps (linear) and masks (nearest)
    for name, src, order, dt in (("fa", primary / "fa.nii.gz", 1, np.float32),
                                 ("md", primary / "md.nii.gz", 1, np.float32),
                                 ("brain_mask", tardis / "model_space" / "brain_mask.nii.gz", 0, np.uint8),
                                 ("regenerated_slices", tardis / "model_space" / "regenerated_slices.nii.gz", 0, np.uint8)):
        arr = resample(nib.load(src), grid, order)
        nib.save(nib.Nifti1Image(arr.astype(dt), aff), out / f"{name}_extended.nii.gz")

    # mean ADC over the model directions
    adc_file = primary / ("adc30_extended.nii.gz" if primary.name.startswith("fod") else "adc6_extended.nii.gz")
    adc_img = nib.load(adc_file)
    adc_mean = nib.Nifti1Image(np.asarray(adc_img.dataobj, dtype=np.float32).mean(axis=-1), adc_img.affine)
    nib.save(nib.Nifti1Image(resample(adc_mean, grid, 1), aff), out / "adc_mean_extended.nii.gz")

    # extended synthetic DWI with its gradient table expressed for the native axes
    dwi_img = nib.load(primary / "dwi_extended.nii.gz")
    vols = [resample(nib.Nifti1Image(np.asarray(dwi_img.dataobj[..., t], dtype=np.float32), dwi_img.affine), grid, 1)
            for t in range(dwi_img.shape[3])]
    nib.save(nib.Nifti1Image(np.stack(vols, axis=-1), aff), out / "dwi_extended.nii.gz")
    grad = np.loadtxt(primary / "grad.txt", comments="#")
    dirs_world, bvals = grad[:, :3], grad[:, 3]
    np.savetxt(out / "dwi_extended.bval", bvals[None], fmt="%.1f")
    np.savetxt(out / "dwi_extended.bvec", fsl_bvecs(dirs_world, aff).T, fmt="%.8f")
    np.savetxt(out / "grad.txt", grad, fmt="%.8f", header="MRtrix gradient table: world (scanner RAS) x y z b")

    # which padded-native slices are regenerated
    regen = np.asarray(nib.load(out / "regenerated_slices_extended.nii.gz").dataobj) > 0
    other = tuple(i for i in range(3) if i != info["slice_axis"])
    info["regenerated_slice_indices"] = [int(i) for i in np.where(regen.sum(axis=other) > 0)[0]]
    info["primary_output"] = primary.name
    info["source_fov"] = fov
    (out / "fov_native.json").write_text(json.dumps(info, indent=2))
    print(f"native outputs -> {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dwi", required=True, help="native preprocessed DWI defining the grid to pad")
    p.add_argument("--tardis-dir", required=True)
    p.add_argument("--primary-dir", required=True, help="the <kind>_b<label> output folder to resample")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
