#!/usr/bin/env python
"""
006: composite volumes on the padded native grid for the PDF and for users: the PreQual metric where the
data were acquired, the TARDIS result where slices were regenerated.

    COMPOSITE/fa, md, tensor (MRtrix order xx yy zz xy xz yz), mask, plus three marker volumes used to draw the
    extension on every brain figure: fov_boundary (1 on the first regenerated slice at each extended end),
    regenerated_slab (1 on every regenerated slice) and acquired_slab (1 on every acquired slice).
    composite.json describes them. (v1 is added by tardis.sh with tensor2metric.)

Inputs : --mask PREPROCESSED/mask.nii.gz --tensor TENSOR/dwmri_tensor.nii.gz --fa SCALARS/..._fa.nii.gz
         --md SCALARS/..._md.nii.gz (the native PreQual results) --tardis-dir OUT
         (reads OUT/PREPROCESSED/{dwmri_extended.nii.gz,mask_extended,regenerated_slices}, OUT/TENSOR/dwmri_extended_tensor,
         OUT/SCALARS/dwmri_extended_tensor_{fa,md}, OUT/LOGS/fov_native.json)
Writes : OUT/COMPOSITE/*.nii.gz, OUT/COMPOSITE/composite.json
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main(args):
    tardis = Path(args.tardis_dir)
    pre, ten, sca, logs, comp = (tardis / d for d in ("PREPROCESSED", "TENSOR", "SCALARS", "LOGS", "COMPOSITE"))
    comp.mkdir(parents=True, exist_ok=True)
    info = json.loads((logs / "fov_native.json").read_text())
    axis = int(info["slice_axis"])
    pad_before = int(info["pad_before"])

    ref = nib.load(pre / "dwmri_extended.nii.gz")
    aff = ref.affine
    shape = tuple(int(s) for s in ref.shape[:3])
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
    regen = np.asarray(nib.load(pre / "regenerated_slices.nii.gz").dataobj) > 0
    brain_ext = np.asarray(nib.load(pre / "mask_extended.nii.gz").dataobj) > 0
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

    nib.save(nib.Nifti1Image(slab(boundary), aff), comp / "fov_boundary.nii.gz")
    nib.save(nib.Nifti1Image(slab(regen_slices), aff), comp / "regenerated_slab.nii.gz")
    nib.save(nib.Nifti1Image(slab(acquired), aff), comp / "acquired_slab.nii.gz")

    # masks and scalar maps: native inside the acquired slices, TARDIS (fit on dwmri_extended) in the regenerated ones
    mask_native = pad(np.asarray(nib.load(args.mask).dataobj, dtype=np.float32)) > 0.5
    mask_comp = np.where(regen_slab, brain_ext & regen, mask_native)
    nib.save(nib.Nifti1Image(mask_comp.astype(np.uint8), aff), comp / "mask.nii.gz")

    for name, native_file in (("fa", args.fa), ("md", args.md)):
        nat = pad(np.nan_to_num(np.asarray(nib.load(native_file).dataobj, dtype=np.float32)))
        ext = np.nan_to_num(np.asarray(nib.load(sca / f"dwmri_extended_tensor_{name}.nii.gz").dataobj, dtype=np.float32))
        comp_arr = np.where(regen, ext, np.where(regen_slab, 0.0, nat)).astype(np.float32)
        nib.save(nib.Nifti1Image(comp_arr, aff), comp / f"{name}.nii.gz")

    t_ext = np.nan_to_num(np.asarray(nib.load(ten / "dwmri_extended_tensor.nii.gz").dataobj, dtype=np.float32))
    t_nat = pad(np.nan_to_num(np.asarray(nib.load(args.tensor).dataobj, dtype=np.float32)))
    t_comp = np.where(regen[..., None], t_ext, np.where(regen_slab[..., None], 0.0, t_nat)).astype(np.float32)
    nib.save(nib.Nifti1Image(t_comp, aff), comp / "tensor.nii.gz")

    meta = dict(slice_axis=axis, pad_before=pad_before, pad_after=int(info["pad_after"]), native_shape=info["native_shape"],
                padded_shape=list(shape), affine=aff.tolist(), acquired_range=[acq_lo, acq_hi],
                regenerated_slices=regen_slices, boundary_slices=boundary,
                files=dict(fa="fa.nii.gz", md="md.nii.gz", tensor="tensor.nii.gz", v1="v1.nii.gz", mask="mask.nii.gz",
                           boundary="fov_boundary.nii.gz", regenerated="regenerated_slab.nii.gz", acquired="acquired_slab.nii.gz"))
    (comp / "composite.json").write_text(json.dumps(meta, indent=2))
    print(f"composites -> {comp}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True)
    p.add_argument("--tensor", required=True)
    p.add_argument("--fa", required=True)
    p.add_argument("--md", required=True)
    p.add_argument("--tardis-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
