# -*- coding: utf-8 -*-
import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import (run_module_suite,
                           assert_almost_equal,
                           assert_allclose,
                           assert_equal,
                           assert_,
                           dec)

import pyir.nufft
from pyir.nufft import dtft, dtft_adj
from pyir.nufft.nufft import NufftBase, nufft_forward, nufft_adj
from pyir.nufft.tests.test_dtft import _uniform_freqs
from pyir.utils import max_percent_diff
pkg_dir = os.path.dirname(pyir.nufft.__file__)
data_dir = pjoin(pkg_dir, 'data', 'mat_files')

__all__ = ['test_nufft_init',
           'test_nufft',
           'test_nufft_adj']


def _perturbed_gridpoints(Nd, rel_std=0.5, seed=1234):
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

    return omega


def _randomized_gridpoints(Nd, rel_std=0.5, seed=1234):
    """Generate a uniform cartesian frequency grid of shape Nd and then
    perturb each point's position by an amount between [0, rel_std) of the
    grid spacing along each axis.

    (for testing NUFFT routines vs. the DTFT)
    """
    rstate = np.random.RandomState(seed)
    ndim = len(Nd)
    omegas = rstate.uniform(size=(tuple(Nd) + (ndim, )), low=0, high=2*np.pi)
    return omegas.reshape((-1, ndim), order='F')


def test_nufft_init(verbose=False):
    Nd = np.array([20, 10])
    st = NufftBase(om='epi', Nd=Nd, Jd=[5, 5], Kd=2 * Nd)
    om = st.om
    if verbose:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(om[:, 0], om[:, 1], 'o-')
        plt.axis([-np.pi, np.pi, -np.pi, np.pi])
        print(st)


def test_nufft_adj():
    """ test nufft_adj() """
    N1 = 4
    N2 = 8
    n_shift = [2.7, 3.1]    # random shifts to stress it
    o1 = 2 * np.pi * np.array([0.0, 0.1, 0.3, 0.4, 0.7, 0.9])
    o2 = np.flipud(o1)
    om = np.vstack((o1.ravel(), o2.ravel())).T
    for mode in ['sparse', 'table1', 'table0']:
        for phasing in ['real', 'complex']:
            st = NufftBase(om=om, Nd=np.array([N1, N2]), Jd=[4, 6],
                           Kd=2 * np.array([N1, N2]), n_shift=n_shift,
                           kernel_type='minmax:tuned',
                           # kernel_type='kb:beatty',
                           phasing=phasing,
                           mode=mode)

            X = np.arange(1, o1.size + 1).ravel() ** 2    # test spectrum
            xd = dtft_adj(X, om, Nd=[N1, N2], n_shift=n_shift)  # TODO...
            XXX = np.vstack((X, X, X)).T
            xn = nufft_adj(st, XXX)
            # print('nufft vs dtft max%%diff = %g' %
            # max_percent_diff(xd,xn[:,:,-1]))  #TODO
            try:
                assert_almost_equal(
                    np.squeeze(xd), np.squeeze(xn[:, :, -1]), decimal=1)
                print("Success for mode: {}, {}".format(mode, phasing))
            except:
                print("Failed for mode: {}, {}".format(mode, phasing))
    return


@dec.skipif(True)  # test data not available
def test_nufft(verbose=False):
    for test3d in [False, True]:
        for initialize_from_Matlab in [True, ]:  # False,]:
            if verbose:
                make_fig = True
                print("\n\nRunning _nufft_test with " +
                      "test3d={}, Matlab_init={}: ".format(
                          test3d,  initialize_from_Matlab))
            else:
                make_fig = False
            _nufft_test(
                test3d=test3d,
                initialize_from_Matlab=initialize_from_Matlab,
                make_fig=make_fig)


def test_nufft_1d():
    Nd = 64
    Kd = 128
    Jd = 6
    Ld_table1 = 512
    n_shift = Nd // 2
    om = _perturbed_gridpoints(Nd)
    rstate = np.random.RandomState(1234)

    rtol = 1e-3
    atol = 1e-5
    maxdiff_forward = {}
    maxdiff_adjoint = {}
    for kernel_type in ['kb:beatty', ]:
        for mode in ['table1', 'table0', 'sparse']:
            for precision in ['single', 'double']:
                for phasing in ['real', 'complex']:
                    if mode == 'table0':
                        Ld = Ld_table1 * 100
                    else:
                        Ld = Ld_table1
                    A = NufftBase(om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift,
                                  mode=mode, Ld=Ld,
                                  kernel_type=kernel_type,
                                  precision=precision,
                                  phasing=phasing)  # TODO:  'table' case

                    x = rstate.randn(Nd) + 1j * rstate.randn(Nd)
                    y = A._nufft_forward(x)

                    y2 = dtft(x, omega=om, Nd=Nd, n_shift=n_shift)
                    assert_allclose(y, y2, rtol=rtol, atol=atol)

                    x_adj = A._nufft_adj(y)
                    x_adj2 = dtft_adj(y, omega=om, Nd=Nd, n_shift=n_shift)
                    assert_allclose(x_adj, x_adj2, rtol=rtol, atol=atol)

                    maxdiff_forward[
                        (kernel_type, mode, precision, phasing)] = \
                        max_percent_diff(y, y2)

                    maxdiff_adjoint[
                        (kernel_type, mode, precision, phasing)] = \
                        max_percent_diff(x_adj, x_adj2)


def test_nufft_2d():
    ndim = 2
    Nd = [16, ] * ndim
    Kd = [32, ] * ndim
    Jd = [6, ] * ndim
    Ld_table1 = 512
    n_shift = np.asarray(Nd) / 2
    om = _perturbed_gridpoints(Nd)
    rstate = np.random.RandomState(1234)

    rtol = 1e-3
    atol = 1e-5
    maxdiff_forward = {}
    maxdiff_adjoint = {}
    for kernel_type in ['kb:beatty', ]:
        for mode in ['table1', 'table0', 'sparse']:
            for precision in ['single', 'double']:
                for phasing in ['real', 'complex']:  # 'complex']:
                    if mode == 'table0':
                        Ld = Ld_table1 * 100
                    else:
                        Ld = Ld_table1
                    A = NufftBase(om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift,
                                  mode=mode, Ld=Ld,
                                  kernel_type=kernel_type,
                                  precision=precision,
                                  phasing=phasing)  # TODO:  'table' case
                    x = rstate.standard_normal(Nd)
                    x = x + 1j * rstate.standard_normal(Nd)
                    y = A._nufft_forward(x)
                    y2 = dtft(x, omega=om, Nd=Nd, n_shift=n_shift)

                    # max_percent_diff(y, y2[:, 0])
                    assert_allclose(y, y2, rtol=rtol, atol=atol)

                    x_adj = A._nufft_adj(y)
                    x_adj2 = dtft_adj(y, omega=om, Nd=Nd, n_shift=n_shift)
                    assert_allclose(x_adj, x_adj2, rtol=rtol, atol=atol)

                    maxdiff_forward[
                        (kernel_type, mode, precision, phasing)] = \
                        max_percent_diff(y, y2)

                    maxdiff_adjoint[
                        (kernel_type, mode, precision, phasing)] = \
                        max_percent_diff(x_adj, x_adj2)


def test_nufft_3d():
    ndim = 3
    Nd = [8, ] * ndim
    Kd = [16, ] * ndim
    Jd = [5, ] * ndim  # use odd kernel for variety (even in 1D, 2D tests)
    Ld_table1 = 512
    n_shift = np.asarray(Nd) / 2
    om = _perturbed_gridpoints(Nd)

    maxdiff_forward = {}
    maxdiff_adjoint = {}
    for kernel_type in ['kb:beatty', ]:
        for mode in ['table1', 'table0', 'sparse']:
            for precision in ['single', 'double']:
                for phasing in ['real', 'complex']:  # 'complex']:
                    rstate = np.random.RandomState(1234)
                    if mode == 'table0':
                        Ld = Ld_table1 * 100
                    else:
                        Ld = Ld_table1
                    A = NufftBase(om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift,
                                  mode=mode, Ld=Ld,
                                  kernel_type=kernel_type,
                                  precision=precision,
                                  phasing=phasing)  # TODO:  'table' case
                    x = rstate.standard_normal(Nd)
                    x = x + 1j * rstate.standard_normal(Nd)
                    y = A._nufft_forward(x)
                    y2 = dtft(x, omega=om, Nd=Nd, n_shift=n_shift)
                    assert_(max_percent_diff(y, y2) < 0.02)

                    x_adj = A._nufft_adj(y)
                    x_adj2 = dtft_adj(y, omega=om, Nd=Nd, n_shift=n_shift)
                    assert_(max_percent_diff(x_adj, x_adj2) < 0.02)

                    maxdiff_forward[
                        (kernel_type, mode, precision, phasing)] = \
                        max_percent_diff(y, y2)

                    maxdiff_adjoint[
                        (kernel_type, mode, precision, phasing)] = \
                        max_percent_diff(x_adj, x_adj2)

# TODO: test other nshift, odd shape, odd Kd, etc
# TODO: test order='F'/'C'


def test_nufft_dtypes():
    Nd = 64
    Kd = 128
    Jd = 6
    Ld = 4096
    n_shift = Nd // 2
    om = _perturbed_gridpoints(Nd)

    kernel_type = 'kb:beatty'
    mode = 'table1'
    for precision in ['single', 'double']:
        A = NufftBase(om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift,
                      mode=mode, Ld=Ld,
                      kernel_type=kernel_type,
                      precision=precision)  # TODO:  'table' case

        if precision == 'single':
            assert_equal(A._cplx_dtype, np.complex64)
            assert_equal(A._real_dtype, np.float32)
        else:
            assert_equal(A._cplx_dtype, np.complex128)
            assert_equal(A._real_dtype, np.float64)

        # set based on precision of om rather than the precision argument
        A2 = NufftBase(om=om.astype(A._real_dtype),
                       Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift,
                       mode=mode, Ld=Ld,
                       precision=None,
                       kernel_type=kernel_type)  # TODO:  'table' case

        if precision == 'single':
            assert_equal(A2._cplx_dtype, np.complex64)
            assert_equal(A2._real_dtype, np.float32)
        else:
            assert_equal(A2._cplx_dtype, np.complex128)
            assert_equal(A2._real_dtype, np.float64)

    x = np.random.randn(Nd) + 1j * np.random.randn(Nd)

    # output matches operator dtype regardless of input dtype
    y = A._nufft_forward(x.astype(np.complex64))
    assert_equal(y.dtype, A._cplx_dtype)
    y = A._nufft_forward(x.astype(np.complex128))
    assert_equal(y.dtype, A._cplx_dtype)

    # real input also gives complex output
    y = A._nufft_forward(x.real.astype(np.float32))
    assert_equal(y.dtype, A._cplx_dtype)
    y = A._nufft_forward(x.real.astype(np.float64))
    assert_equal(y.dtype, A._cplx_dtype)


def test_nufft_table_make1():
    from pyir.nufft.nufft import _nufft_table_make1
    from numpy.testing import assert_almost_equal
    decimal = 6
    for N in [64, ]:   # [33, 64]:
        for phasing in ['real', 'complex']:
            h0, t0 = _nufft_table_make1(how='slow', N=N, J=6, K=2 * N, L=2048,
                                        phasing=phasing,
                                        kernel_type='kb:beatty',
                                        kernel_kwargs={})
            h1, t1 = _nufft_table_make1(how='fast', N=N, J=6, K=2 * N, L=2048,
                                        phasing=phasing,
                                        kernel_type='kb:beatty',
                                        kernel_kwargs={})
            h2, t2 = _nufft_table_make1(how='ratio', N=N, J=6, K=2 * N, L=2048,
                                        phasing=phasing,
                                        kernel_type='kb:beatty',
                                        kernel_kwargs={})
            assert_almost_equal(h0, h1, decimal=decimal)
            assert_almost_equal(h0, h2, decimal=decimal)
            assert_almost_equal(h1, h2, decimal=decimal)
            assert_almost_equal(t0, t1, decimal=decimal)
            assert_almost_equal(t0, t2, decimal=decimal)


if __name__ == "__main__":
    run_module_suite()
