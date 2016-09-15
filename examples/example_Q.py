# -*- coding: utf-8 -*-
import skimage.data
from skimage.measure import compare_nrmse
import bart_cy as bart
import numpy as np
from pyir.operators_private import MRI_Operator
from pyir.operators import DiagonalOperator
from pyvolplot import volshow
from pyir.utils import fftn, ifftn, embed
from pyir.nufft.nufft import compute_Q
from matplotlib import pyplot as plt
nread = 256
traj_rad = bart.traj(X=2*nread, Y=64, radial=True).real
traj_rad = traj_rad.reshape((3, -1), order='F')
kspace = traj_rad.transpose((1, 0))[:, :2] * 0.5
mask = np.ones((nread, nread), dtype=np.bool)
nufft_kwargs = dict(mode='table0',  # 'table1',
                    use_CUDA=False,
                    kernel='kb:beatty',
                    phasing='complex')
Nd = np.asarray((nread, nread))

osfvals = [1.25, 1.5, 1.75, 2]
all_nrmse = np.zeros(len(osfvals))
for n, osf in enumerate(osfvals):
    G = MRI_Operator(Nd=Nd,
                     Kd=(osf*Nd).astype(np.intp),
                     Jd=(4, 4),
                     fov=(1, 1),
                     kspace=kspace,
                     mask=np.ones(Nd, dtype=np.bool),
                     # weights=np.sqrt(np.linalg.norm(kspace, axis=1)),
                     **nufft_kwargs)
    
    weights = np.linalg.norm(kspace, axis=1)
    weights = DiagonalOperator(weights, order='F')
    x = skimage.data.camera()[::2, ::2].astype(np.complex64)
    tmp0 = embed(G.Gnufft.H * (G.Gnufft * x), mask)
    tmp = embed(G.Gnufft.H * weights * (G.Gnufft * x), mask)
    
    volshow(tmp)
    
    # psf = nufft_adj_psf(G.Gnufft, np.ones(kspace.shape[0]))
    # volshow(np.abs(psf)**0.25)
    # PSF = fftn(psf)
    
    Q = compute_Q(G)
    # tmp0_approx = ifftn(Q * fftn(x, G.Gnufft.Kd))[:nread, :nread]
    
    tmp0_approx = ifftn(Q * fftn(x, Q.shape))[:nread, :nread]
    
    nrmse = compare_nrmse(np.abs(tmp0), np.abs(tmp0_approx))
    errscale = 100
    volshow([tmp0, tmp0_approx, errscale*(tmp0-tmp0_approx)], vmax=np.abs(tmp0).max())
    all_nrmse[n] = nrmse

plt.figure()
plt.plot(osfvals, all_nrmse)

