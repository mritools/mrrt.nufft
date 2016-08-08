# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
from pyir.nufft import dtft, dtft_adj
from pyir.utils import max_percent_diff

from numpy.testing import assert_allclose, run_module_suite


__all__ = ['test_dtft_3d',
           'test_dtft_2d',
           'test_dtft_1d',
           'test_dtft_adj_3d',
           'test_dtft_adj_2d',
           'test_dtft_adj_1d']


def _uniform_freqs(Nd):
    """Generate a uniform cartesian frequency grid of shape Nd.

    (for testing DTFT routines vs. a standard Cartesian FFT)
    """
    # TODO: simplify via np.indices?
    if np.isscalar(Nd):
        Nd = (Nd, )
    ndim = len(Nd)
    fs = [2 * np.pi * np.arange(Nd[d])/Nd[d] for d in range(ndim)]
    fs = np.meshgrid(*fs, indexing='ij')
    fs = [f.reshape((-1, 1), order='F') for f in fs]
    return np.hstack(fs)


def test_dtft_3d(verbose=False):
    """ function dtft_test() """
    Nd = (8, 12, 10)
    n_shift = np.asarray([1, 3, 2]).reshape(3, 1)
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(Nd)  # test signal

    # # test with uniform frequency locations
    om = _uniform_freqs(Nd)

    # DTFT result
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)

    # compare to FFT-based result
    Xf = np.fft.fftn(x)
    # add phase shift to the numpy fft
    Xf = Xf.ravel(order='F') * np.exp(1j * (np.dot(om, n_shift)))[:, 0]

    assert_allclose(np.squeeze(Xd), Xf, atol=1e-12)


def test_dtft_2d(verbose=False):
    """ function dtft_test() """
    Nd = (8, 12)
    n_shift = np.asarray([1, 3]).reshape(2, 1)
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(Nd)  # test signal
    # test with uniform frequency locations
    om = _uniform_freqs(Nd)

    # DTFT result
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)

    # compare to FFT-based result
    Xf = np.fft.fftn(x)
    # phase shift
    Xf = Xf.ravel(order='F') * np.exp(1j * (np.dot(om, n_shift)))[:, 0]

    assert_allclose(np.squeeze(Xd), Xf, atol=1e-12)
    if verbose:
        max_err = np.max(np.abs(np.squeeze(Xd)-np.squeeze(Xf)))
        print('max error = %g' % max_err)


def test_dtft_1d(verbose=False):
    """ function dtft_test() """
    Nd = (16, )
    n_shift = np.asarray([5, ]).reshape(1, 1)
    rstate = np.random.RandomState(1234)
    x = rstate.standard_normal(Nd)  # test signal
    # test with uniform frequency locations
    om = _uniform_freqs(Nd)

    # DTFT result
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)

    # compare to FFT-based result
    Xf = np.fft.fftn(x)
    # phase shift
    Xf = Xf.ravel(order='F') * np.exp(1j * (np.dot(om, n_shift)))[:, 0]

    assert_allclose(np.squeeze(Xd), Xf, atol=1e-12)
    if verbose:
        max_err = np.max(np.abs(np.squeeze(Xd)-np.squeeze(Xf)))
        print('max error = %g' % max_err)


def test_dtft_adj_3d(verbose=False, test_Cython=False):
    Nd = (32, 16, 2)
    n_shift = np.asarray([2, 1, 3]).reshape(3, 1)

    rstate = np.random.RandomState(1234)
    X = rstate.standard_normal(Nd)  # test signal
    X = X + 1j * rstate.standard_normal(Nd)

    # test with uniform frequency locations:
    om = _uniform_freqs(Nd)

    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, True)
    assert_allclose(xd, xl, atol=1e-7)

    Xp = np.exp(-1j * np.dot(om, n_shift))
    Xp = X * Xp.reshape(X.shape, order='F')
    xf = np.fft.ifftn(Xp) * np.prod(Nd)
    assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print('loop max %% difference = %g' % max_percent_diff(xl, xd))
        print('ifft max %% difference = %g' % max_percent_diff(xf, xd))

    if test_Cython:
        import time
        from pyir.nufft.cy_dtft import dtft_adj as cy_dtft_adj
        t_start = time.time()
        xc = cy_dtft_adj(X.ravel(order='F'), om, Nd, n_shift)
        print("duration (1 rep) = {}".format(time.time()-t_start))
        print('ifft max %% difference = %g' % max_percent_diff(xf, xc))

        X_16rep = np.tile(X.ravel(order='F')[:, None], (1, 16))
        t_start = time.time()
        xc16 = cy_dtft_adj(X_16rep, om, Nd, n_shift)
        print("duration (16 reps) = {}".format(time.time()-t_start))
        t_start = time.time()
        X_64rep = np.tile(X.ravel(order='F')[:, None], (1, 64))
        xc64 = cy_dtft_adj(X_64rep, om, Nd, n_shift)
        max_percent_diff(xf, xc64[..., -1])
        print("duration (64 reps) = {}".format(time.time()-t_start))
#        %timeit xd = dtft_adj(X_16rep, om, Nd, n_shift);

    return


def test_dtft_adj_2d(verbose=False):
    Nd = (4, 6)
    n_shift = np.asarray([2, 1]).reshape(2, 1)

    rstate = np.random.RandomState(1234)
    X = rstate.standard_normal(Nd)  # test signal
    X = X + 1j * rstate.standard_normal(Nd)

    # test with uniform frequency locations:
    om = _uniform_freqs(Nd)

    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, True)
    assert_allclose(xd, xl, atol=1e-7)

    Xp = np.exp(-1j * np.dot(om, n_shift))
    Xp = X * Xp.reshape(X.shape, order='F')
    xf = np.fft.ifftn(Xp) * np.prod(Nd)
    assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print('ifft max %% difference = %g' % max_percent_diff(xf, xd))
    return


def test_dtft_adj_1d(verbose=False):
    Nd = (16, )
    n_shift = np.asarray([2, ]).reshape(1, 1)

    rstate = np.random.RandomState(1234)
    X = rstate.standard_normal(Nd)  # test signal
    X = X + 1j * rstate.standard_normal(Nd)

    # test with uniform frequency locations:
    om = _uniform_freqs(Nd)

    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, True)
    assert_allclose(xd, xl, atol=1e-7)

    Xp = np.exp(-1j * np.dot(om, n_shift))
    Xp = X * Xp.reshape(X.shape, order='F')
    xf = np.fft.ifftn(Xp) * np.prod(Nd)
    assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print('ifft max %% difference = %g' % max_percent_diff(xf, xd))
    return


if __name__ == "__main__":
    run_module_suite()
