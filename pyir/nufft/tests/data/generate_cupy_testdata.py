from __future__ import division, print_function, absolute_import

import os
import shutil
import numpy as np
from pyir.nufft.nufft import NufftBase

from pyir.nufft.tests.test_dtft import _uniform_freqs

os.chdir('/media/lee8rx/3TB_Data11/cupy_testdata')
# have to also set hardcoded save_testdata = True in _nufft_table_adj and
#  _nufft_table_forward


def _perturbed_gridpoints(Nd, rel_std=0.5, seed=1234, xp=np):
    """Generate a uniform cartesian frequency grid of shape Nd and then
    perturb each point's position by an amount between [0, rel_std) of the
    grid spacing along each axis.

    (for testing NUFFT routines vs. the DTFT)
    """
    rstate = np.random.RandomState(seed)
    if np.isscalar(Nd):
        Nd = (Nd, )
    Nd = np.asarray(Nd)
    if np.any(Nd < 2):
        raise ValueError("must be at least size 2 on all dimensions")
    omega = _uniform_freqs(Nd)  # [npoints, ndim]
    df = 2*np.pi/Nd
    npoints = omega.shape[0]
    for d in range(len(Nd)):
        # don't randomize edge points so values will still fall within
        # a 2*pi range
        omega[:, d] += df[d] * rel_std * rstate.rand(npoints)
        # rescale to keep values within a 2*pi range
        omega[:, d] += np.min(omega[:, d])
        omega[:, d] *= (2*np.pi)/np.max(omega[:, d])

    return xp.asarray(omega)


kernel_type = 'kb:beatty'
for mode in ['table']:
    for precision in ['single', 'double']:
        for phasing in ['real', 'complex']:
            for ndim in [1, 2, 3]:
                rstate = np.random.RandomState(1234)
                if ndim == 1:
                    Nd = 64
                    Kd = 128
                    Jd = 6
                    Ld = 512
                    n_shift = Nd // 2
                    om = _perturbed_gridpoints(Nd)
                elif ndim == 2:
                    Nd = [16, ] * ndim
                    Kd = [32, ] * ndim
                    Jd = [6, ] * ndim
                    Ld = 512
                    n_shift = np.asarray(Nd) / 2
                    om = _perturbed_gridpoints(Nd)
                elif ndim == 3:
                    Nd = [8, ] * ndim
                    Kd = [16, ] * ndim
                    Jd = [5, ] * ndim  # use odd kernel for variety (even in 1D, 2D tests)
                    Ld = 512
                    n_shift = np.asarray(Nd) / 2
                    om = _perturbed_gridpoints(Nd)
                A = NufftBase(om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift,
                              mode=mode, Ld=Ld,
                              kernel_type=kernel_type,
                              precision=precision,
                              phasing=phasing)  # TODO:  'table' case

                if ndim == 1:
                    x = rstate.randn(Nd) + 1j * rstate.randn(Nd)
                else:
                    x = rstate.randn(*tuple(Nd)) + 1j * rstate.randn(*tuple(Nd))
                y = A._nufft_forward(x)

                shutil.move('table{}d_forward_data.npz'.format(ndim),
                            'table{}d_forward_order{}_{}_{}.npz'.format(ndim, mode[-1], precision, phasing))

                x_adj = A._nufft_adj(y)
                shutil.move('table{}d_adj_data.npz'.format(ndim),
                            'table{}d_adj_order{}_{}_{}.npz'.format(ndim, mode[-1], precision, phasing))


if False:
    from __future__ import division, print_function, absolute_import

    import os
    import shutil
    import numpy as np
    from pyir.nufft.nufft import NufftBase

    from pyir.nufft.tests.test_dtft import _uniform_freqs

    from pyir.cuda.autoinit_cupy import _get_gridding_funcs
    from pyir.cuda.CUDA_MRI_cupy import default_device, default_context, get_1D_block_table_gridding
    import cupy

    os.chdir('/media/lee8rx/3TB_Data11/cupy_testdata')

    mode = 'table'
    precision = 'double'
    phasing = 'complex'
    ndim = 1
    for ndim in [1, 2, 3]:
        for mode in ['table']:
            for precision in ['single', 'double']:
                for phasing in ['real', 'complex']:
                    print(ndim, mode, precision, phasing)
                    if ndim == 1:
                        Nd = [64, ]
                        Kd = [128, ]
                        Jd = 6
                        Ld = 512
                        #om = _perturbed_gridpoints(Nd)
                    elif ndim == 2:
                        Nd = [16, ] * ndim
                        Kd = [32, ] * ndim
                        Jd = 6
                        Ld = 512
                        #om = _perturbed_gridpoints(Nd)
                    elif ndim == 3:
                        Nd = [8, ] * ndim
                        Kd = [16, ] * ndim
                        Jd = 5  # use odd kernel for variety (even in 1D, 2D tests)
                        Ld = 512
                        #om = _perturbed_gridpoints(Nd)

                    forward_data = np.load('table{}d_forward_order{}_{}_{}.npz'.format(ndim, mode[-1], precision, phasing))
                    print(list(forward_data.keys()))
                    adj_data = np.load('table{}d_adj_order{}_{}_{}.npz'.format(ndim, mode[-1], precision, phasing))

                    M = forward_data['tm'].shape[0]
                    kern_forward, kern_adj = _get_gridding_funcs(Kd=Kd,
                                                                 M=M,
                                                                 J=Jd,
                                                                 L=Ld,
                                                                 is_complex_kernel=(phasing=='complex'),
                                                                 precision=precision)

                    block, grid = get_1D_block_table_gridding(M, default_device)
                    if precision == 'single':
                        real_dtype = np.float32
                        cplx_dtype = np.complex64
                        rtol = atol = 1e-3
                    else:
                        real_dtype = np.float64
                        cplx_dtype = np.complex128
                        rtol = atol = 1e-13

                    print("forward")
                    ck = cupy.asarray(forward_data['Xk'].squeeze(), dtype=cplx_dtype)
                    fm = cupy.empty(forward_data['X'].squeeze().shape)
                    fm.fill(1e-16)  # .fill(0) causing error in memset_async
                    default_context.synchronize()
                    fm = fm.astype(cplx_dtype)
                    if phasing == 'real':
                        h1 = cupy.asarray(forward_data['h0'].squeeze(), dtype=real_dtype)
                    else:
                        h1 = cupy.asarray(forward_data['h0'].squeeze(), dtype=cplx_dtype)
                    if ndim > 1:
                        if phasing == 'real':
                            h2 = cupy.asarray(forward_data['h1'].squeeze(), dtype=real_dtype)
                        else:
                            h2 = cupy.asarray(forward_data['h1'].squeeze(), dtype=cplx_dtype)
                    if ndim > 2:
                        if phasing == 'real':
                            h3 = cupy.asarray(forward_data['h2'].squeeze(), dtype=real_dtype)
                        else:
                            h3 = cupy.asarray(forward_data['h2'].squeeze(), dtype=cplx_dtype)

                    tm = cupy.asarray(forward_data['tm'].squeeze(), dtype=real_dtype)

                    if ndim == 1:
                        kern_forward(block, grid, (ck, h1, tm, fm))
                    elif ndim == 2:
                        kern_forward(block, grid, (ck, h1, h2, tm, fm))
                    elif ndim == 3:
                        kern_forward(block, grid, (ck, h1, h2, h3, tm, fm))
                    fm_expected = np.asarray(forward_data['X'].squeeze(), dtype=cplx_dtype)
                    cupy.testing.assert_allclose(fm_expected, fm, rtol=rtol, atol=atol)
                    # print(np.max(np.abs(fm_expected - fm.get())))

                    print("adjoint")
                    ck = cupy.empty(adj_data['Xk'].squeeze().shape, dtype=real_dtype)
                    ck.fill(1e-16)  # .fill(0) causing error in memset_async
                    default_context.synchronize()
                    ck = ck.astype(cplx_dtype)
                    fm = cupy.asarray(adj_data['X'].squeeze().astype(cplx_dtype))

                    if phasing == 'real':
                        h1 = cupy.asarray(adj_data['h0'].squeeze(), dtype=real_dtype)
                    else:
                        h1 = cupy.asarray(adj_data['h0'].squeeze(), dtype=cplx_dtype)
                    if ndim > 1:
                        if phasing == 'real':
                            h2 = cupy.asarray(adj_data['h1'].squeeze(), dtype=real_dtype)
                        else:
                            h2 = cupy.asarray(adj_data['h1'].squeeze(), dtype=cplx_dtype)
                    if ndim > 2:
                        if phasing == 'real':
                            h3 = cupy.asarray(adj_data['h2'].squeeze(), dtype=real_dtype)
                        else:
                            h3 = cupy.asarray(adj_data['h2'].squeeze(), dtype=cplx_dtype)

                    tm = cupy.asarray(adj_data['tm'].squeeze(), dtype=real_dtype)

                    if ndim == 1:
                        kern_adj(block, grid, (ck, h1, tm, fm))
                    elif ndim == 2:
                        kern_adj(block, grid, (ck, h1, h2, tm, fm))
                    elif ndim == 3:
                        kern_adj(block, grid, (ck, h1, h2, h3, tm, fm))
                    ck_expected = np.asarray(adj_data['Xk'].squeeze(), dtype=cplx_dtype)
                    cupy.testing.assert_allclose(ck_expected, ck, rtol=rtol, atol=atol)
                    # print(np.max(np.abs(ck_expected - ck.get())))
