# -*- coding: utf-8 -*-
import skimage.data
import bart_cy as bart
import numpy as np
from pyir.operators_private import MRI_Operator
from pyir.operators import DiagonalOperator
from pyvolplot import volshow
from pyir.utils import fftn, ifftn, embed
nread = 256
traj_rad = bart.traj(X=2*nread, Y=512, radial=True).real
traj_rad = traj_rad.reshape((3, -1), order='F')
kspace = traj_rad.transpose((1, 0))[:, :2] * 0.5
mask = np.ones((nread, nread), dtype=np.bool)
nufft_kwargs = dict(mode='table1',
                    use_CUDA=False,
                    kernel_type='kb:beatty',
                    phasing='real',
                    Ld=151)
Nd = np.asarray((nread, nread))
J = 4
osf = 1.05
G = MRI_Operator(Nd=Nd,
                 Kd=(4*((osf*Nd)//4)).astype(np.intp),
                 Jd=(J, J),
                 fov=(1, 1),
                 kspace=kspace,
                 mask=np.ones(Nd, dtype=np.bool),
                 # weights=np.sqrt(np.linalg.norm(kspace, axis=1)),
                 **nufft_kwargs)

G_ref = MRI_Operator(Nd=Nd,
                     Kd=(2*Nd).astype(np.intp),
                     Jd=(6, 6),
                     fov=(1, 1),
                     kspace=kspace,
                     mask=np.ones(Nd, dtype=np.bool),
                     # weights=np.sqrt(np.linalg.norm(kspace, axis=1)),
                     **nufft_kwargs)

if False:
    from pyir.nufft import NufftBase
    from pyir.nufft.nufft_utils import _nufft_coef
    K = G.Gnufft.Kd[0]
    L = G.Gnufft.Ld[0]
    N = G.Gnufft.Nd[0]
    J = G.Gnufft.Jd[0]
    t1 = J / 2. - 1 + np.arange(L) / L  # [L]
    om1 = t1 * 2 * pi / K       # * gam
    s1 = NufftBase(om=om1, Nd=N, Kd=K, **nufft_args)
    h = np.asarray(
        s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(order='F')
    h = np.concatenate((h, np.asarray([h[0], ])), axis=0)  # [J*L+1,]
    [c, arg] = _nufft_coef(om1, J, K, G.Gnufft.kernel.kernel[0])
nufft_kwargs['kernel_type'] = 'mols'
G_MOLS = MRI_Operator(Nd=Nd,
                      Kd=(4*((osf*Nd)//4)).astype(np.intp),
                      Jd=(J, J),
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
tmp_ref = embed(G_ref.Gnufft.H * weights * (G_ref.Gnufft * x), mask)
volshow(tmp)
from matplotlib import pyplot as plt

plt.figure(); plt.plot(G.Gnufft.h[0].real); plt.plot(G.Gnufft.h[0].imag)
plt.plot(G_MOLS.Gnufft.h[0].real/G_MOLS.Gnufft.h[0].real.max()); plt.plot(G_MOLS.Gnufft.h[0].imag)

tmp_MOLS = embed(G_MOLS.Gnufft.H * weights * (G_MOLS.Gnufft * x), mask)
volshow([tmp, tmp_MOLS])

#tmp_MOLS = np.abs(tmp_MOLS)/np.linalg.norm(tmp_MOLS)*np.linalg.norm(x)
#tmp = np.abs(tmp)/np.linalg.norm(tmp)*np.linalg.norm(x)
from skimage.measure import compare_nrmse
print("NRMSE (kb:beatty): = {}".format(compare_nrmse(np.abs(tmp_ref), np.abs(tmp))))
print("NRMSE (MOLS-U): = {}".format(compare_nrmse(np.abs(tmp_ref), np.abs(tmp_MOLS))))
volshow([np.abs(tmp_ref-tmp), np.abs(tmp_ref-tmp_MOLS)], vmax=np.max(np.abs(tmp_ref-tmp)))

tmp_hybrid = embed(G_MOLS.Gnufft.H * weights * (G.Gnufft * x), mask)
volshow(tmp_hybrid)

