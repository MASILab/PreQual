import logging
import warnings
import numpy as np
from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.core.gradients import gradient_table_from_bvals_bvecs

# ---------------------------------------------------------------------------------------------
# Helpers vendored from scilpy 1.4.0 (MIT license, github.com/scilus/scilpy) so that this venv
# no longer depends on scilpy and can run on a current Python / dipy.
# ---------------------------------------------------------------------------------------------

DEFAULT_B0_THRESHOLD = 20


def is_normalized_bvecs(bvecs):
    """Check if b-vectors (N, 3) are normalized (zero vectors are allowed)."""

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3, bvecs_norm == 0))


def normalize_bvecs(bvecs):
    """Normalize non-zero b-vectors (N, 3) to unit length."""

    bvecs = np.array(bvecs, dtype=np.float64, copy=True)
    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    idx = bvecs_norm != 0
    bvecs[idx] /= bvecs_norm[idx, None]
    return bvecs


def check_b0_threshold(force_b0_threshold, bvals_min, b0_thr=DEFAULT_B0_THRESHOLD):
    """Warn (or raise) if the minimal b-value is negative or above the b0 threshold."""

    if bvals_min < 0:
        logging.warning('Warning: Your dataset contains negative b-values (minimal bvalue of {}). This is suspicious. We recommend you check your data.'.format(bvals_min))

    if bvals_min > b0_thr:
        if force_b0_threshold:
            logging.warning('Warning: Your minimal bvalue is {}, but the threshold is set to {}. Proceeding with a threshold of {}.'.format(bvals_min, b0_thr, bvals_min))
            return bvals_min
        raise ValueError('The minimal bvalue ({}) is greater than the threshold ({}). No b0 volumes can be found.'.format(bvals_min, b0_thr))

    return b0_thr


def compute_dwi_attenuation(dwi_weights, b0):
    """Signal attenuation: DWI divided by the b0, clipped to <= 1 and with non-finite values set to 0."""

    b0 = b0[..., None]
    dwi_weights = np.minimum(dwi_weights, b0)
    dwi_attenuation = dwi_weights / b0
    dwi_attenuation[np.logical_not(np.isfinite(dwi_attenuation))] = 0.
    return dwi_attenuation


# ---------------------------------------------------------------------------------------------
# dipy compatibility: sh_order was renamed sh_order_max in dipy 1.9 and removed in 2.0
# ---------------------------------------------------------------------------------------------

def _sf_to_sh(sf, sphere, sh_order, basis_type, smooth):
    try:
        return sf_to_sh(sf, sphere, sh_order_max=sh_order, basis_type=basis_type, smooth=smooth)
    except TypeError:
        return sf_to_sh(sf, sphere, sh_order, basis_type, smooth=smooth)


def _sh_to_sf(sh, sphere, sh_order, basis_type):
    try:
        return sh_to_sf(sh, sphere, sh_order_max=sh_order, basis_type=basis_type)
    except TypeError:
        return sh_to_sf(sh, sphere, sh_order=sh_order, basis_type=basis_type)


# ---------------------------------------------------------------------------------------------
# Signal reconstruction
# ---------------------------------------------------------------------------------------------

def reconstruct_signal_at_voxel(i,j,k,n,og_bvec,og_bval,bvec_stack,bval_stack,dwi_hat,sh_order):
        warnings.filterwarnings("ignore")
        dwi = n[i][j][k]
        og_gradient_table = gradient_table_from_bvals_bvecs(og_bval, og_bvec)
        og_b0_mask = og_gradient_table.b0s_mask
        vec = bvec_stack[i,j,k,:,:]
        val = bval_stack[i,j,k,:]

        # Voxels outside the gradient tensor field of view have no corrected b-table (NaN):
        # leave the original signal untouched there.
        if not (np.all(np.isfinite(vec)) and np.all(np.isfinite(val))):
                dwi_hat[i,j,k,:] = dwi[np.logical_not(og_b0_mask)]
                return

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
        sh = _sf_to_sh(weights_scaled, sphere, sh_order, basis_type, smooth)

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

        sf = _sh_to_sf(sh, og_sphere, sh_order, basis_type)

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
