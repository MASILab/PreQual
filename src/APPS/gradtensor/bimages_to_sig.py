import os
import sys
import shutil
import tempfile
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from dipy.io import read_bvals_bvecs
from utils import reconstruct_signal_at_voxel, get_sh_order

# Assign inputs
n = nib.load(sys.argv[1]).get_fdata()
vol = nib.load(sys.argv[1])
og_file = sys.argv[2]
ob_file = sys.argv[3]
out_dir = sys.argv[4]
num_cores = sys.argv[5]

# Load the voxelwise bvec and bvals
bvec_vols = []
for i in sorted(os.listdir(out_dir)):
    if i.endswith('.nii.gz') and i.startswith('bvec_'):
        bvec_vol = nib.load(out_dir + '/' + i).get_fdata()
        bvec_vol = np.expand_dims(bvec_vol,4)
        bvec_vol = np.transpose(bvec_vol,(0,1,2,4,3))
        bvec_vols.append(bvec_vol)
bvec_stack = np.stack(bvec_vols,3)
bvec_stack = bvec_stack.squeeze()

bval_vols = []
for i in sorted(os.listdir(out_dir)):
    if i.endswith('.nii.gz') and i.startswith('bval_'):
        bval_vol = nib.load(out_dir + '/' + i).get_fdata()
        bval_vols.append(bval_vol)
bval_stack = np.stack(bval_vols,3)

# Get the number of shells - to run each separate shell 
og_bval, og_bvec = read_bvals_bvecs(ob_file,og_file)
bval = np.unique(og_bval)
bval = bval.astype(int)
bval = bval[bval!=0]
bval.sort()

# Run bimage to signal for each separate shell
dwmri_corrected = np.zeros((n.shape[0],n.shape[1],n.shape[2],n.shape[3]))
for i in bval:
    # Get the index for bval
    ind_b = np.where(og_bval == i)
    ind_b0 = np.nonzero(og_bval == 0)
    ind_b0 = np.squeeze(ind_b0)
    ind_0_b = np.where((og_bval == 0) | (og_bval == i))

    # Set up memory for corrected signal and data for bval
    path = os.path.join(out_dir,'TMP')
    os.mkdir(path)
    xaxis = range(n.shape[0])
    yaxis = range(n.shape[1])
    zaxis = range(n.shape[2]) 
    len1 = ind_0_b[0]
    dwi_hat_path = os.path.join(path,'dwi_hat_'+ str(i) +'.mmap')
    dwi_hat = np.memmap(dwi_hat_path, dtype=float, shape=(n.shape[0],n.shape[1],n.shape[2],ind_b[0].shape[0]), mode='w+')
    data = n[:,:,:,len1]
    org_bvec = og_bvec[len1,:]
    org_bval = og_bval[len1]
    corr_bvec = bvec_stack[:,:,:,len1,:]
    corr_bval = bval_stack[:,:,:,len1]
    sh_order = get_sh_order(ind_b[0].shape[0])

    # Call the main reconstruct_signal_at_voxel script
    print('b-value:',str(i),', No. of directions:',str(ind_b[0].shape[0]) ,', SH order',sh_order)
    results = Parallel(n_jobs=int(num_cores))(delayed(reconstruct_signal_at_voxel)(i,j,k,data,org_bvec,org_bval,corr_bvec,corr_bval,dwi_hat,sh_order) for k in zaxis for j in yaxis for i in xaxis)
    dwmri_corrected[:,:,:,ind_b0] = n[:,:,:,ind_b0] 
    dwmri_corrected[:,:,:,ind_b[0]] = dwi_hat

# Save the corrected signal
dwmri_corrected = np.nan_to_num(dwmri_corrected)
out_name = 'Lcorrected_sig.nii.gz'
output_file = os.path.join(out_dir, out_name )
nib.save(nib.Nifti1Image(dwmri_corrected.astype(np.float32),vol.affine),output_file )


# Remove tmp file
try:
    shutil.rmtree(path)
except:
    print("Couldn't delete folder")


