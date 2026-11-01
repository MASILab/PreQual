#!/usr/bin/env python
"""
004: write the extended volumes as NIfTI on the model grid, as one multi-shell synthetic DWI.
FOD model: per processed shell the 30-direction ADC (mm^2/s) is saved, and S = S0 exp(-b ADC) on the 30
mrtrix directions goes into the DWI. Tensor model: the tensor (MRtrix order xx yy zz xy xz yz, mm^2/s) and
the 6-direction ADC are saved, and the DWI holds the shell's own directions. The DWI is one constant b0
followed by every processed shell, with FSL bval/bvec and an MRtrix gradient table. Shells TARDIS did not
process are not in it. Plus the regenerated-slice mask, a QA figure per shell and LOGS/summary.txt.
Inputs : OUT/work/{extended.npz,meta.json,inference.json}, OUT/LOGS/{fov,plan}.json, OUT/MODEL_SPACE/T1w.nii.gz
Writes : OUT/MODEL_SPACE/dwmri_extended.{nii.gz,bval,bvec}, dwmri_extended_grad.txt, regenerated_slices.nii.gz,
         adc_b<label>.nii.gz (FOD model) or adc6_b<label>.nii.gz + tensor_b<label>.nii.gz (tensor model),
         OUT/LOGS/QA/adc_b<label>.png, OUT/LOGS/summary.txt
Invoke : python 004_write_outputs.py --out-dir OUT [--s0 1000]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from tqdm import tqdm


def fsl_bvecs(dirs_world, affine):
    """World unit vectors -> FSL bvec convention for an image with this affine (voxel axes, x negated if det > 0)."""
    R = affine[:3, :3]
    Rn = R / np.linalg.norm(R, axis=0, keepdims=True)
    v = dirs_world @ np.linalg.inv(Rn).T
    if np.linalg.det(R) > 0:
        v[:, 0] *= -1
    return v


def write_gradients(prefix, dirs_world, bvals, affine):
    """<prefix>.bval, <prefix>.bvec (FSL) and <prefix>_grad.txt (MRtrix, world frame)."""
    dirs_world = np.asarray(dirs_world, float)
    bvals = np.asarray(bvals, float)
    np.savetxt(f"{prefix}.bval", bvals[None], fmt="%.1f")
    np.savetxt(f"{prefix}.bvec", fsl_bvecs(dirs_world, affine).T, fmt="%.8f")
    np.savetxt(f"{prefix}_grad.txt", np.column_stack([dirs_world, bvals]), fmt="%.8f",
               header="MRtrix gradient table: world (scanner RAS) x y z b")


def save(path, arr, aff, dtype=np.float32):
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr).astype(dtype), aff), path)


def tensor_from_adc6(adc6, dirs6, adc_scale):
    """(..., 6 dirs) x adc_scale -> (..., 6 comps xx yy zz xy xz yz) mm^2/s (exact inverse of 002's projection)."""
    u = np.asarray(dirs6)
    B = np.column_stack([u[:, 0] ** 2, u[:, 1] ** 2, u[:, 2] ** 2,
                         2 * u[:, 0] * u[:, 1], 2 * u[:, 0] * u[:, 2], 2 * u[:, 1] * u[:, 2]])
    return np.einsum("...d,cd->...c", adc6 / adc_scale, np.linalg.inv(B)).astype(np.float32)


def qa_figure(t1, scalar, brain, fov, title, out_png, vox):
    """Sagittal + coronal mid-brain views of the T1 and of the extended scalar map, FOV edges marked."""
    ijk = np.argwhere(brain)
    cx, cy = (int(round(ijk[:, 0].mean())), int(round(ijk[:, 1].mean()))) if ijk.size else (t1.shape[0] // 2, t1.shape[1] // 2)
    views = [("sagittal", lambda a: a[cx], vox[2] / vox[1]), ("coronal", lambda a: a[:, cy], vox[2] / vox[0])]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5), facecolor="black", gridspec_kw=dict(hspace=0.12, wspace=0.05))
    vmax = float(np.percentile(scalar[brain], 99)) if brain.any() else 1.0
    for col, (name, cut, aspect) in enumerate(views):
        for row, (img, cmap, vm, lab) in enumerate(((t1, "gray", None, "T1 (registered)"),
                                                   (scalar, "gray", vmax, "mean ADC, extended"))):
            ax = axes[row, col]
            ax.imshow(cut(img).T, origin="lower", cmap=cmap, vmin=0, vmax=vm, aspect=aspect, interpolation="nearest")
            for z in (fov["z_lo_known"] - 0.5, fov["z_hi_known"] + 0.5):
                ax.axhline(z, color="cyan", lw=0.8, ls="--")
            for z in (fov["z_lo_brain"] - 0.5, fov["z_hi_brain"] + 0.5):
                ax.axhline(z, color="orange", lw=0.8, ls=":")
            ax.set_title(f"{lab} - {name}", color="white", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{title}\ncyan: acquired DWI FOV | orange: T1 brain extent | regenerated "
                 f"{fov['n_extend_inferior']} inferior + {fov['n_extend_superior']} superior slices", color="white", fontsize=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main(args):
    out = Path(args.out_dir)
    work, ms, logs = out / "work", out / "MODEL_SPACE", out / "LOGS"
    qa = logs / "QA"
    qa.mkdir(parents=True, exist_ok=True)
    meta = json.loads((work / "meta.json").read_text())
    fov = json.loads((logs / "fov.json").read_text())
    plan = json.loads((logs / "plan.json").read_text())
    inf = json.loads((work / "inference.json").read_text())
    aff = np.array(meta["affine"])
    vox = meta["vox"]
    kind, clip_max, adc_scale = meta["model"], meta["clip_max"], meta["adc_scale"]
    ext = np.load(work / "extended.npz")
    brain = np.moveaxis(np.load(work / "model_input.npz")["brain"], 0, 2)        # (X, Y, Z)
    t1 = np.asarray(nib.load(ms / "T1w.nii.gz").dataobj, dtype=np.float32)
    Z = brain.shape[2]
    regen = np.zeros(Z, bool)
    regen[fov["extend_superior"] + fov["extend_inferior"]] = True
    save(ms / "regenerated_slices.nii.gz", brain & regen[None, None, :], aff, np.uint8)
    out_mask = brain.copy()
    out_mask[:, :, :fov["z_lo_brain"]] = False
    out_mask[:, :, fov["z_hi_brain"] + 1:] = False

    # one synthetic DWI for all processed shells: a constant b0, then each shell's directions
    vols = [(args.s0 * out_mask).astype(np.float32)[..., None]]
    dirs_all = [[0.0, 0.0, 0.0]]
    bvals_all = [0.0]
    lines = []
    for s in tqdm(meta["shells"], desc="write", unit="shell"):
        label = s["label"]
        arr = np.moveaxis(ext[f"adc_{label}"], 0, 2)                              # (X, Y, Z, C)
        if kind == "fod":
            adc = (arr + 1.0) / 2.0 * clip_max / adc_scale * out_mask[..., None]    # mm^2/s
            save(ms / f"adc_b{label}.nii.gz", adc, aff)
            dirs = np.asarray(meta["dirs30"])
            b = s["b_mean"]
            vols.append(args.s0 * np.exp(-b * adc) * out_mask[..., None])
            dirs_all.extend(dirs.tolist())
            bvals_all.extend([b] * len(dirs))
            scalar = adc.mean(axis=-1)
            desc = f"b={label} ({s['n_dirs']} acquired directions, mean b {b:.1f}): 30-direction ADC, {len(dirs)} synthetic volumes"
        else:
            adc6 = arr * out_mask[..., None]                                       # x adc_scale
            tensor = tensor_from_adc6(adc6, meta["dirs6"], adc_scale) * out_mask[..., None]
            save(ms / f"tensor_b{label}.nii.gz", tensor, aff)
            save(ms / f"adc6_b{label}.nii.gz", adc6 / adc_scale, aff)
            g = np.asarray(s["dirs_world"])
            bv = np.asarray(s["bvals"])
            u = np.column_stack([g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2,
                                 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]])
            adc_g = np.einsum("...c,dc->...d", tensor, u)
            vols.append(args.s0 * np.exp(-bv * adc_g) * out_mask[..., None])
            dirs_all.extend(g.tolist())
            bvals_all.extend(bv.tolist())
            scalar = adc6.mean(axis=-1) / adc_scale
            desc = f"b={label} ({s['n_dirs']} acquired directions, mean b {s['b_mean']:.1f}): tensor, {len(bv)} synthetic volumes on the acquired directions"
        qa_figure(t1, scalar, out_mask, fov, f"{kind} model, {desc}", qa / f"adc_b{label}.png", vox)
        lines.append(f"  {desc}")

    dwi = np.concatenate(vols, axis=-1).astype(np.float32)
    save(ms / "dwmri_extended.nii.gz", dwi, aff)
    write_gradients(str(ms / "dwmri_extended"), dirs_all, bvals_all, aff)
    print(f"MODEL_SPACE/dwmri_extended.nii.gz: {dwi.shape[3]} volumes (1 b0 + {dwi.shape[3] - 1} synthetic), S0 = {args.s0:g}", flush=True)

    summary = [
        f"TARDIS FOV extension - {kind} model ({inf['ode_steps']} ODE steps, {inf['device']}, seed {inf['seed']})",
        f"shells: " + ", ".join(f"b{s['label']} x{s['n_dirs']}" for s in plan["shells"]) + f" | processed {plan['selected']}",
        f"model grid {meta['shape']} @ {vox} mm (RAS); T1 brain z {fov['z_lo_brain']}..{fov['z_hi_brain']}, "
        f"acquired DWI z {fov['z_lo_known']}..{fov['z_hi_known']}",
        f"regenerated: {fov['n_extend_inferior']} inferior slices ({fov['mm_extend_inferior']:.0f} mm), "
        f"{fov['n_extend_superior']} superior slices ({fov['mm_extend_superior']:.0f} mm)",
        "processed shells:"] + lines + [
        "outputs:",
        f"  PREPROCESSED/    dwmri_extended.nii.gz/.bval/.bvec: synthetic DWI on the native grid padded along the slice axis "
        f"(1 constant b0 of S0={args.s0:g} + the processed shells, {'30 mrtrix directions per shell' if kind == 'fod' else 'the acquired directions'}), "
        "mask_extended, regenerated_slices",
        "  TENSOR/, SCALARS/  tensor and FA/MD/AD/RD/V1 fit on dwmri_extended the way PreQual fits PREPROCESSED/dwmri",
        "  COMPOSITE/       PreQual maps where acquired, TARDIS maps where regenerated (what the PDF shows)",
        "  MODEL_SPACE/     the same synthetic DWI, ADC per shell, T1w, brain_mask, dwi_coverage, b0_mean, regenerated_slices on the 2x2x1 mm model grid",
        "  LOGS/            plan, fov, fov_slices.csv, fov_native, meta, inference, this summary, registration (reg/), T1 preprocessing, QA/*.png"]
    summary += [f"WARNING: {w}" for w in plan.get("warnings", [])]
    (logs / "summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--s0", type=float, default=1000.0, help="constant b0 intensity inside the brain of the written DWI")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
