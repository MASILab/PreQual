#!/usr/bin/env python
"""
006: composite volumes on the padded native grid for the PDF and for users: the PreQual metric where the
data were acquired, the TARDIS result where slices were regenerated.

    fa_composite, md_composite, tensor_composite (MRtrix order xx yy zz xy xz yz), mask_composite,
    shell_b<label>_mean_composite (per processed shell: mean DW image, synthetic part scaled to the native
    intensity inside the acquired FOV), plus three marker volumes used to draw the extension on every brain
    figure: fov_boundary (1 on the first regenerated slice at each extended end), regenerated_slab (1 on every
    regenerated slice) and acquired_slab (1 on every acquired slice). composite.json describes them.

Inputs : --dwi PREPROCESSED/dwmri.nii.gz --bval --mask PREPROCESSED/mask.nii.gz --tensor TENSOR/dwmri_tensor.nii.gz
         --fa SCALARS/..._fa.nii.gz --md SCALARS/..._md.nii.gz --tardis-dir OUT --native-dir OUT/native
Writes : OUT/native/*_composite.nii.gz, OUT/native/{fov_boundary,regenerated_slab,acquired_slab}.nii.gz,
         OUT/native/composite.json
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np


def resample(img, grid, order):
    return np.asarray(nibproc.resample_from_to(img, grid, order=order, mode="constant", cval=0.0).dataobj, dtype=np.float32)


def main(args):
    native_dir = Path(args.native_dir)
    tardis = Path(args.tardis_dir)
    info = json.loads((native_dir / "fov_native.json").read_text())
    plan = json.loads((tardis / "plan.json").read_text())
    primary = tardis / info["primary_output"]
    axis = int(info["slice_axis"])
    pad_before = int(info["pad_before"])

    ref = nib.load(native_dir / "fa_extended.nii.gz")
    aff = ref.affine
    shape = tuple(int(s) for s in ref.shape[:3])
    grid = (shape, aff)
    n_native = int(info["native_shape"][axis])

    def pad(arr):
        """Embed a native-grid array (3D or 4D) into the padded grid along the slice axis."""
        out = np.zeros(shape + tuple(arr.shape[3:]), np.float32)
        sl = [slice(None)] * 3
        sl[axis] = slice(pad_before, pad_before + n_native)
        out[tuple(sl)] = arr
        return out

    def slab(indices):
        """Full-plane marker: 1 on the given slice indices along the slice axis."""
        out = np.zeros(shape, np.uint8)
        for k in indices:
            sl = [slice(None)] * 3
            sl[axis] = int(k)
            out[tuple(sl)] = 1
        return out

    other_axes = tuple(i for i in range(3) if i != axis)
    regen = np.asarray(nib.load(native_dir / "regenerated_slices_extended.nii.gz").dataobj) > 0
    brain_ext = np.asarray(nib.load(native_dir / "brain_mask_extended.nii.gz").dataobj) > 0
    regen_slices = sorted(int(k) for k in np.where(regen.sum(axis=other_axes) > 0)[0])
    regen_slab = slab(regen_slices) > 0

    # acquired slices: native extent minus the regenerated ones (partly valid edge slices are regenerated)
    native_range = set(range(pad_before, pad_before + n_native))
    acquired = sorted(native_range - set(regen_slices))
    if not acquired:
        raise SystemExit("no acquired slice left after removing regenerated ones")
    acq_lo, acq_hi = acquired[0], acquired[-1]
    boundary = []
    if (acq_lo - 1) in regen_slices:
        boundary.append(acq_lo - 1)
    if (acq_hi + 1) in regen_slices:
        boundary.append(acq_hi + 1)
    print(f"slice axis {axis}: acquired {acq_lo}..{acq_hi}, regenerated {len(regen_slices)} slices, boundary at {boundary}", flush=True)

    nib.save(nib.Nifti1Image(slab(boundary), aff), native_dir / "fov_boundary.nii.gz")
    nib.save(nib.Nifti1Image(slab(regen_slices), aff), native_dir / "regenerated_slab.nii.gz")
    nib.save(nib.Nifti1Image(slab(acquired), aff), native_dir / "acquired_slab.nii.gz")

    # masks and scalar maps
    mask_native = pad(np.asarray(nib.load(args.mask).dataobj, dtype=np.float32)) > 0.5
    mask_comp = np.where(regen_slab, brain_ext & regen, mask_native)
    nib.save(nib.Nifti1Image(mask_comp.astype(np.uint8), aff), native_dir / "mask_composite.nii.gz")

    for name, native_file in (("fa", args.fa), ("md", args.md)):
        nat = pad(np.nan_to_num(np.asarray(nib.load(native_file).dataobj, dtype=np.float32)))
        ext = np.nan_to_num(np.asarray(nib.load(native_dir / f"{name}_extended.nii.gz").dataobj, dtype=np.float32))
        comp = np.where(regen, ext, np.where(regen_slab, 0.0, nat)).astype(np.float32)
        nib.save(nib.Nifti1Image(comp, aff), native_dir / f"{name}_composite.nii.gz")

    # tensor: TARDIS tensor (model grid, MRtrix component order, scanner frame) resampled component-wise
    t_img = nib.load(primary / "tensor_extended.nii.gz")
    t_ext = np.stack([resample(nib.Nifti1Image(np.asarray(t_img.dataobj[..., c], dtype=np.float32), t_img.affine), grid, 1)
                      for c in range(t_img.shape[3])], axis=-1)
    t_nat = pad(np.nan_to_num(np.asarray(nib.load(args.tensor).dataobj, dtype=np.float32)))
    t_comp = np.where(regen[..., None], t_ext, np.where(regen_slab[..., None], 0.0, t_nat)).astype(np.float32)
    nib.save(nib.Nifti1Image(t_comp, aff), native_dir / "tensor_composite.nii.gz")

    # per processed shell: mean DW image, synthetic part scaled to the native intensity inside the acquired FOV
    dwi_img = nib.load(args.dwi)
    acq_brain = mask_native & ~regen_slab
    shells_out = []
    for s in plan["shells"]:
        if s["label"] not in plan["selected"]:
            continue
        d = tardis / f"{plan['model']}_b{s['label']}"
        if not (d / "dwi_extended.nii.gz").exists():
            continue
        idx = s["indices"]
        nat_mean = np.zeros(dwi_img.shape[:3], np.float64)
        for t in idx:
            nat_mean += np.nan_to_num(np.asarray(dwi_img.dataobj[..., t], dtype=np.float32))
        nat_mean = pad((nat_mean / len(idx)).astype(np.float32))
        syn_img = nib.load(d / "dwi_extended.nii.gz")
        syn_mean = np.asarray(syn_img.dataobj[..., 1:], dtype=np.float32).mean(axis=-1)  # volume 0 is the constant b0
        syn_res = resample(nib.Nifti1Image(syn_mean, syn_img.affine), grid, 1)
        sel = acq_brain & (syn_res > 0) & (nat_mean > 0)
        scale = float(np.median(nat_mean[sel]) / np.median(syn_res[sel])) if sel.sum() > 100 else 1.0
        comp = np.where(regen & brain_ext, syn_res * scale, np.where(regen_slab, 0.0, nat_mean)).astype(np.float32)
        out_file = native_dir / f"shell_b{s['label']}_mean_composite.nii.gz"
        nib.save(nib.Nifti1Image(comp, aff), out_file)
        shells_out.append(dict(label=int(s["label"]), n_dirs=int(s["n_dirs"]), b_mean=float(s["b_mean"]),
                               native_volumes=[int(t) for t in idx], scale=scale, file=out_file.name))
        print(f"shell b={s['label']}: synthetic scaled by {scale:.3f} to match the acquired FOV", flush=True)

    meta = dict(slice_axis=axis, pad_before=pad_before, pad_after=int(info["pad_after"]), native_shape=info["native_shape"],
                padded_shape=list(shape), affine=aff.tolist(), acquired_range=[acq_lo, acq_hi],
                regenerated_slices=regen_slices, boundary_slices=boundary, primary_output=info["primary_output"],
                files=dict(fa="fa_composite.nii.gz", md="md_composite.nii.gz", tensor="tensor_composite.nii.gz",
                           v1="v1_composite.nii.gz", mask="mask_composite.nii.gz", boundary="fov_boundary.nii.gz",
                           regenerated="regenerated_slab.nii.gz", acquired="acquired_slab.nii.gz"),
                shells=shells_out)
    (native_dir / "composite.json").write_text(json.dumps(meta, indent=2))
    print(f"composites -> {native_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dwi", required=True)
    p.add_argument("--bval", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--fa", required=True)
    p.add_argument("--md", required=True)
    p.add_argument("--tardis-dir", required=True)
    p.add_argument("--native-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
