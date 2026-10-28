#!/usr/bin/env python
"""
apply_gradtensor_to_b.py: voxelwise b-table correction for gradient coil nonlinearity.

Python port of apply_gradtensor_to_b.m / compute_b_images.m from
https://github.com/baxpr/gradtensor (Rogers et al., Proc SPIE 2018), which previously ran
in PreQual as a MATLAB-compiled binary requiring the MATLAB Compiler Runtime R2017a.

Given a gradient coil tensor image L (x, y, z, 9 with volumes Lxx, Lxy, Lxz, Lyx, Lyy, Lyz,
Lzx, Lzy, Lzz), a reference diffusion image defining the output grid, and an FSL-style
b-table, this script

  1. resamples L onto the reference grid with trilinear interpolation (SPM reslice,
     interp=1, mask=true: voxels outside the L field of view are set to NaN),
  2. for every voxel v and every gradient direction g computes

         ab      = L_v @ bvec_g                 (in RAS, so the FSL bvec x is flipped first)
         bval'   = bval_g * |ab|^2              (b scales with G^2)
         bvec'   = ab / |ab|                    (flipped back to the FSL convention)

  3. writes, exactly like the MATLAB version,
         <out_dir>/L_resamp.nii.gz        resampled tensor (x, y, z, 9), float32
         <out_dir>/bvec_%04d.nii.gz       corrected b-vector image per volume (x, y, z, 3)
         <out_dir>/bval_%04d.nii.gz       corrected b-value image per volume (x, y, z)
     plus copies of the input b-table (org.bval / org.bvec) for provenance.

Usage:
    apply_gradtensor_to_b.py --Limg_file L.nii.gz --refimg_file dwi.nii.gz \
        --bval_file dwi.bval --bvec_file dwi.bvec --out_dir OUT
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from dipy.io import read_bvals_bvecs

# SPM reslice tolerance (in voxels) for deciding that a sample falls outside the source image
SPM_MASK_TOLERANCE = 5e-2


def resample_to_reference(L_img, ref_img):
    """Trilinearly resample the 4D tensor image L onto the 3D grid of ref_img.

    Returns an array of shape ref_shape + (9,) with NaN wherever the reference voxel maps
    outside the field of view of L (SPM reslice with mask=true on a float image).
    """

    ref_shape = tuple(ref_img.shape[:3])
    L = np.asanyarray(L_img.dataobj).astype(np.float64)
    if L.ndim == 5 and L.shape[3] == 1:
        L = L[:, :, :, 0, :]
    if L.ndim != 4 or L.shape[3] != 9:
        raise ValueError('Gradient tensor image must be 4D with 9 volumes (Lxx, Lxy, ..., Lzz), got shape {}'.format(L.shape))

    # Reference voxel centers -> world (RAS) -> L voxel coordinates
    ii, jj, kk = np.meshgrid(np.arange(ref_shape[0]), np.arange(ref_shape[1]), np.arange(ref_shape[2]), indexing='ij')
    vox_ref = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
    ref_to_L = np.linalg.inv(L_img.affine) @ ref_img.affine
    vox_L = (ref_to_L @ vox_ref)[:3]

    # SPM masks any voxel sampled outside the source image (with a small tolerance)
    dims = np.array(L.shape[:3], dtype=np.float64)[:, None]
    outside = np.any((vox_L < -SPM_MASK_TOLERANCE) | (vox_L > dims - 1 + SPM_MASK_TOLERANCE), axis=0)

    out = np.empty((vox_L.shape[1], 9), dtype=np.float64)
    for e in range(9):
        out[:, e] = map_coordinates(L[..., e], vox_L, order=1, mode='nearest')
    out[outside, :] = np.nan

    return out.reshape(ref_shape + (9,))


def adjust_btable(L_resamp, bvals, bvecs):
    """Apply the voxelwise tensor to an FSL b-table.

    L_resamp: (x, y, z, 9) row-major tensor components
    bvals:    (N,)
    bvecs:    (N, 3) FSL convention (unit vectors, x flipped relative to RAS world coordinates)

    Returns adjbvec (x, y, z, 3, N) in the FSL convention and adjbval (x, y, z, N).
    """

    spatial_shape = L_resamp.shape[:3]
    Lv = L_resamp.reshape(-1, 3, 3)  # row-major: volume e -> L[e // 3, e % 3], as in the MATLAB code
    nv = Lv.shape[0]
    nb = bvals.shape[0]

    # FSL bvecs are LAS while NIfTI world coordinates are RAS: flip x before applying L
    #   https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/FAQ#What_conventions_do_the_bvecs_use.3F
    #   http://www.mrtrix.org/2016/04/15/bug-in-fsl-bvecs-handling/
    bvec_ras = bvecs.T.astype(np.float64).copy()  # (3, N)
    bvec_ras[0, :] *= -1

    adjbvec = np.empty((nv, 3, nb), dtype=np.float64)
    adjbval = np.empty((nv, nb), dtype=np.float64)

    for b in range(nb):
        ab = Lv @ bvec_ras[:, b]  # (nv, 3): L_v * bvec for every voxel
        len2 = np.sum(ab ** 2, axis=1)
        adjbval[:, b] = bvals[b] * len2  # b has a G^2 term, so scale by the squared length change
        length = np.sqrt(len2)
        keep = length != 0  # skip renormalization of zero vectors (b = 0); NaN voxels stay NaN
        ab[keep, :] /= length[keep, None]
        adjbvec[:, :, b] = ab

    # Flip x back to the FSL convention
    adjbvec[:, 0, :] *= -1

    return adjbvec.reshape(spatial_shape + (3, nb)), adjbval.reshape(spatial_shape + (nb,))


def main(argv=None):

    parser = argparse.ArgumentParser(description='Compute voxelwise gradient-nonlinearity-corrected b-values and b-vectors (Python port of baxpr/gradtensor apply_gradtensor_to_b).')
    parser.add_argument('--Limg_file', required=True, help='Gradient coil tensor image (x, y, z, 9)')
    parser.add_argument('--refimg_file', required=True, help='Diffusion image defining the output grid')
    parser.add_argument('--bval_file', required=True, help='FSL b-values file')
    parser.add_argument('--bvec_file', required=True, help='FSL b-vectors file')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading gradient tensor {}'.format(args.Limg_file))
    L_img = nib.load(args.Limg_file)
    print('Loading reference image {}'.format(args.refimg_file))
    ref_img = nib.load(args.refimg_file)

    bvals, bvecs = read_bvals_bvecs(args.bval_file, args.bvec_file)
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    nb = bvals.shape[0]
    print('Loaded b-table with {} volumes'.format(nb))

    # Keep the b-table that was applied next to the outputs (same format the MATLAB version consumed)
    np.savetxt(os.path.join(args.out_dir, 'org.bval'), bvals[None, :])
    np.savetxt(os.path.join(args.out_dir, 'org.bvec'), bvecs.T)

    print('Resampling gradient tensor to the reference grid')
    L_resamp = resample_to_reference(L_img, ref_img)
    n_outside = int(np.sum(np.isnan(L_resamp[..., 0])))
    if n_outside > 0:
        print('WARNING: {} of {} reference voxels fall outside the gradient tensor field of view and are set to NaN (no correction applied there).'.format(n_outside, L_resamp[..., 0].size))
    L_resamp_file = os.path.join(args.out_dir, 'L_resamp.nii.gz')
    nib.save(nib.Nifti1Image(L_resamp.astype(np.float32), ref_img.affine), L_resamp_file)

    print('Flipping bvec X (LAS to RAS) and adjusting b-table voxelwise')
    adjbvec, adjbval = adjust_btable(L_resamp, bvals, bvecs)

    print('Writing per-volume bvec/bval images to {}'.format(args.out_dir))
    for b in range(nb):
        bvec_img = nib.Nifti1Image(adjbvec[..., b].astype(np.float32), ref_img.affine)
        bvec_img.header['descrip'] = b'Adjusted bvec'
        nib.save(bvec_img, os.path.join(args.out_dir, 'bvec_{:04d}.nii.gz'.format(b + 1)))

        bval_img = nib.Nifti1Image(adjbval[..., b].astype(np.float32), ref_img.affine)
        bval_img.header['descrip'] = b'Adjusted bval'
        nib.save(bval_img, os.path.join(args.out_dir, 'bval_{:04d}.nii.gz'.format(b + 1)))

    print('apply_gradtensor_to_b complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
