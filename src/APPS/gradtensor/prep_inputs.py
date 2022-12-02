import sys
import numpy as np
from dipy.io import read_bvals_bvecs


print(sys.argv)
og_file = sys.argv[1]
ob_file = sys.argv[2]
out_dir = sys.argv[3]

# Load the bvec and bval
org_bval, org_bvec = read_bvals_bvecs(ob_file,og_file)
org_bval = np.reshape(org_bval,[org_bval.shape[0],1])

# Set output filenames
out_bval_file = out_dir + '/org.bval'
out_bvec_file = out_dir + '/org.bvec'


# Write the bval bvec with the correct dims for apply_gradten
np.savetxt(out_bval_file,np.transpose(org_bval))
np.savetxt(out_bvec_file,np.transpose(org_bvec))


