import logging
import warnings
import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from scilpy.reconst.raw_signal import compute_dwi_attenuation
from scilpy.utils.bvec_bval_tools import (check_b0_threshold, is_normalized_bvecs, normalize_bvecs)

def reconstruct_signal_at_voxel(i,j,k,n,og_bvec,og_bval,bvec_stack,bval_stack,dwi_hat,sh_order):
        warnings.filterwarnings("ignore")
        dwi = n[i][j][k]
        og_gradient_table = gradient_table_from_bvals_bvecs(og_bval, og_bvec)
        vec = bvec_stack[i,j,k,:,:]
        val = bval_stack[i,j,k,:]
        gradient_table = gradient_table_from_bvals_bvecs(val, vec)
        sh_order=sh_order
        basis_type='tournier07'
        smooth=0.00
        use_attenuation=True
        force_b0_threshold=False
        mask=None
        sphere=None

        # Extracting infos
        b0_mask = gradient_table.b0s_mask
        bvecs = gradient_table.bvecs
        bvals = gradient_table.bvals

        dwi = np.reshape(dwi,[1,1,1,bvals.shape[0]])

        if not is_normalized_bvecs(bvecs):
                logging.warning("Your b-vectors do not seem normalized...")
                bvecs = normalize_bvecs(bvecs)

        b0_threshold = check_b0_threshold(force_b0_threshold, bvals.min())

        # Keeping b0-based infos
        bvecs = bvecs[np.logical_not(b0_mask)]
        weights = dwi[..., np.logical_not(b0_mask)]

        # scale singal with bval correction
        b0 = dwi[..., b0_mask].mean(axis=3)
        norm_gg = np.divide(bvals[np.logical_not(b0_mask)] , og_bval[np.logical_not(b0_mask)])
        weights_scaled = b0 * np.exp (np.divide( (np.log (np.divide(weights,b0)) ) , norm_gg))

        # Compute attenuation using the b0.
        if use_attenuation:
                weights_scaled = compute_dwi_attenuation(weights_scaled, b0)

        # Get cartesian coords from bvecs
        sphere = Sphere(xyz=bvecs)

        # Fit SH (SF TO SH)
        sh = sf_to_sh(weights_scaled, sphere, sh_order, basis_type, smooth=smooth)

        # Apply mask
        if mask is not None:
                sh *= mask[..., None]

        # Reconstructing DWI (SH to SF)
        og_bvecs = og_gradient_table.bvecs

        if not is_normalized_bvecs(og_bvecs):
                logging.warning("Your b-vectors do not seem normalized...")
                og_bvecs = normalize_bvecs(og_bvecs)

        og_bvecs = og_bvecs[np.logical_not(b0_mask)]

        og_sphere = Sphere(xyz=og_bvecs)

        sf = sh_to_sf(sh, og_sphere, sh_order=sh_order, basis_type=basis_type)

        # SF TO DWI (inverse of compute_dwi_attenuation) here weights_hat is DWI with bvec corrected
        b0 = b0[..., None]
        weights_hat = sf * b0
        dwi_hat[i,j,k,:] = weights_hat

def get_sh_order(ndirection):

    # Given the number of directions returns the sh order
    order_dir_table = {0:1,2:6,4:15,6:28,8:45,10:66,12:91}
    min_sh_order = [(key, value) for key, value in order_dir_table.items() if value < ndirection]
    sh_order = min_sh_order[-1][0]
    return sh_order
