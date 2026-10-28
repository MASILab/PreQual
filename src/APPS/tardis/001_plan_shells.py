#!/usr/bin/env python
"""
001: read the gradient table, cluster the shells, pick the model and write the mean b0.
FOD model if any shell has >= --min-dirs-fod directions (every such shell is processed); else the
tensor model on the shell with the most directions among those with b <= --tensor-bmax, else on any
shell with >= --min-dirs-tensor directions (warning); else exit 3.
Inputs : --dwi dwmri.nii.gz --bval --bvec
Writes : OUT/plan.json, OUT/b0_mean.nii.gz, OUT/work/model.txt
Invoke : python 001_plan_shells.py --dwi D --bval B --bvec V --out-dir OUT [--min-dirs-fod 28]
         [--min-dirs-tensor 6] [--tensor-bmax 1500] [--force-model fod|tensor]
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

B0_THRESHOLD = 50.0     # 037: b < 50 is a b0
CLUSTER_TOL = 10.0      # 034: consecutive b-values <= 2*tol apart share a shell
LABEL_ROUND = 5         # 034: shell label = mean b rounded to 5


def load_gradients(bval_path, bvec_path, n_vols):
    bvals = np.loadtxt(bval_path).ravel().astype(float)
    bvecs = np.loadtxt(bvec_path).astype(float)
    if bvecs.ndim != 2:
        sys.exit(f"bvec must be a 2-D table, got shape {bvecs.shape}")
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3 or bvecs.shape == (3, 3):
        bvecs = bvecs.T
    if len(bvals) != n_vols or bvecs.shape != (n_vols, 3):
        sys.exit(f"gradient table mismatch: {n_vols} volumes, {len(bvals)} bvals, bvecs {bvecs.shape}")
    return bvals, bvecs


def shell_clusters(bvals):
    """034/051 clustering of the non-b0 volumes -> [(label, b_mean, indices)] sorted by b."""
    idx = sorted(np.where(bvals >= B0_THRESHOLD)[0], key=lambda i: bvals[i])
    clusters = [[idx[0]]] if idx else []
    for i in tqdm(idx[1:], desc="cluster shells", unit="vol", leave=False):
        if bvals[i] - bvals[clusters[-1][-1]] <= 2 * CLUSTER_TOL:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    out = []
    for c in clusters:
        b_mean = float(bvals[c].mean())
        out.append(dict(label=int(round(b_mean / LABEL_ROUND) * LABEL_ROUND), b_mean=b_mean,
                        n_dirs=len(c), indices=[int(i) for i in c]))
    return out


def choose(shells, args):
    warnings = []
    fod = [s for s in shells if s["n_dirs"] >= args.min_dirs_fod]
    tensor_ok = [s for s in shells if s["n_dirs"] >= args.min_dirs_tensor]
    model = args.force_model or ("fod" if fod else "tensor")
    if model == "fod":
        if not fod:
            sys.exit(f"--force-model fod but no shell has >= {args.min_dirs_fod} directions")
        return model, fod, warnings
    if not tensor_ok:
        return None, [], [f"no shell with >= {args.min_dirs_tensor} directions; nothing to fit"]
    low = [s for s in tensor_ok if s["label"] <= args.tensor_bmax]
    pool = low or tensor_ok
    if not low:
        warnings.append(f"no shell with b <= {args.tensor_bmax:.0f} and >= {args.min_dirs_tensor} dirs; "
                        f"fitting the tensor on b={pool[0]['label']} shells is outside the DTI regime")
    pick = sorted(pool, key=lambda s: (-s["n_dirs"], abs(s["label"] - 1000)))[0]
    if args.force_model != "tensor" and not fod:
        warnings.append(f"no shell with >= {args.min_dirs_fod} directions: tensor model on b={pick['label']} "
                        f"({pick['n_dirs']} dirs)")
    return model, [pick], warnings


def main(args):
    out = Path(args.out_dir)
    (out / "work").mkdir(parents=True, exist_ok=True)
    img = nib.load(args.dwi)
    if img.ndim != 4:
        sys.exit(f"{args.dwi}: expected a 4-D DWI, got shape {img.shape}")
    bvals, bvecs = load_gradients(args.bval, args.bvec, img.shape[3])
    b0_idx = [int(i) for i in np.where(bvals < B0_THRESHOLD)[0]]
    if not b0_idx:
        sys.exit(f"no b0 volume (b < {B0_THRESHOLD:.0f})")
    shells = shell_clusters(bvals)
    print(f"{img.shape[3]} volumes, {len(b0_idx)} b0, shells: " +
          ", ".join(f"b{s['label']} x{s['n_dirs']}" for s in shells), flush=True)
    model, selected, warnings = choose(shells, args)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)
    plan = dict(model=model, n_volumes=int(img.shape[3]), b0_threshold=B0_THRESHOLD, b0_indices=b0_idx,
                shells=shells, selected=[s["label"] for s in selected], warnings=warnings,
                min_dirs_fod=args.min_dirs_fod, min_dirs_tensor=args.min_dirs_tensor, tensor_bmax=args.tensor_bmax)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    if model is None:
        sys.exit(3)
    (out / "work" / "model.txt").write_text(model)
    print(f"model: {model} on shell(s) {plan['selected']}", flush=True)

    b0 = np.zeros(img.shape[:3], np.float64)
    for i in tqdm(b0_idx, desc="mean b0", unit="vol", leave=False):
        b0 += np.nan_to_num(np.asarray(img.dataobj[..., i], dtype=np.float32), nan=0.0)
    b0 = (b0 / len(b0_idx)).astype(np.float32)
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(b0, img.affine, hdr), out / "b0_mean.nii.gz")
    print(f"mean b0 -> {out / 'b0_mean.nii.gz'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dwi", required=True)
    p.add_argument("--bval", required=True)
    p.add_argument("--bvec", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-dirs-fod", type=int, default=28)
    p.add_argument("--min-dirs-tensor", type=int, default=6)
    p.add_argument("--tensor-bmax", type=float, default=1500.0)
    p.add_argument("--force-model", choices=("fod", "tensor"), default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
