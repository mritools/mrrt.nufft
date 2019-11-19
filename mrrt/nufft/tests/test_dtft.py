import numpy as np
import pytest

from mrrt.nufft import dtft, dtft_adj
from mrrt.utils import config, max_percent_diff

if config.have_cupy:
    import cupy

    all_xp = [np, cupy]
else:
    all_xp = [np]


__all__ = [
    "test_dtft_1d",
    "test_dtft_2d",
    "test_dtft_3d",
    "test_dtft_adj_1d",
    "test_dtft_adj_2d",
    "test_dtft_adj_3d",
]


def _uniform_freqs(Nd, xp=np):
    """Generate a uniform cartesian frequency grid of shape Nd.

    (for testing DTFT routines vs. a standard Cartesian FFT)
    """
    if xp.isscalar(Nd):
        Nd = (Nd,)
    ndim = len(Nd)
    fs = [2 * np.pi * xp.arange(Nd[d]) / Nd[d] for d in range(ndim)]
    fs = xp.meshgrid(*fs, indexing="ij")
    fs = [f.reshape((-1, 1), order="F") for f in fs]
    return xp.hstack(fs)


@pytest.mark.parametrize("xp", all_xp)
def test_dtft_3d(xp, verbose=False):
    """ function dtft_test() """
    Nd = (8, 12, 10)
    n_shift = np.asarray([1, 3, 2]).reshape(3, 1)
    rstate = xp.random.RandomState(1234)
    x = rstate.standard_normal(Nd)  # test signal

    # # test with uniform frequency locations
    om = _uniform_freqs(Nd, xp=xp)

    # DTFT result
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)

    # compare to FFT-based result
    Xf = xp.fft.fftn(x)
    # add phase shift to the numpy fft
    Xf = (
        Xf.ravel(order="F")
        * xp.exp(1j * (xp.dot(om, xp.asarray(n_shift))))[:, 0]
    )

    xp.testing.assert_allclose(xp.squeeze(Xd), Xf, atol=1e-12)


@pytest.mark.parametrize("xp", all_xp)
def test_dtft_2d(xp, verbose=False):
    """ function dtft_test() """
    Nd = (8, 12)
    n_shift = np.asarray([1, 3]).reshape(2, 1)
    rstate = xp.random.RandomState(1234)
    x = rstate.standard_normal(Nd)  # test signal
    # test with uniform frequency locations
    om = _uniform_freqs(Nd, xp=xp)

    # DTFT result
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)

    # compare to FFT-based result
    Xf = xp.fft.fftn(x)
    # phase shift
    Xf = (
        Xf.ravel(order="F")
        * xp.exp(1j * (xp.dot(om, xp.asarray(n_shift))))[:, 0]
    )

    xp.testing.assert_allclose(xp.squeeze(Xd), Xf, atol=1e-12)
    if verbose:
        max_err = np.max(np.abs(xp.squeeze(Xd) - xp.squeeze(Xf)))
        print("max error = %g" % max_err)


@pytest.mark.parametrize("xp", all_xp)
def test_dtft_1d(xp, verbose=False):
    """ function dtft_test() """
    Nd = (16,)
    n_shift = np.asarray([5]).reshape(1, 1)
    rstate = xp.random.RandomState(1234)
    x = rstate.standard_normal(Nd)  # test signal
    # test with uniform frequency locations
    om = _uniform_freqs(Nd, xp=xp)

    # DTFT result
    Xd = dtft(x, om, n_shift=n_shift, useloop=False)

    # compare to FFT-based result
    Xf = xp.fft.fftn(x)
    # phase shift
    Xf = (
        Xf.ravel(order="F")
        * xp.exp(1j * (xp.dot(om, xp.asarray(n_shift))))[:, 0]
    )

    xp.testing.assert_allclose(xp.squeeze(Xd), Xf, atol=1e-12)
    if verbose:
        max_err = xp.max(xp.abs(xp.squeeze(Xd) - xp.squeeze(Xf)))
        print("max error = %g" % max_err)


@pytest.mark.parametrize("xp", all_xp)
def test_dtft_adj_3d(xp, verbose=False, test_Cython=False):
    Nd = (32, 16, 2)
    n_shift = np.asarray([2, 1, 3]).reshape(3, 1)

    rstate = xp.random.RandomState(1234)
    X = rstate.standard_normal(Nd)  # test signal
    X = X + 1j * rstate.standard_normal(Nd)

    # test with uniform frequency locations:
    om = _uniform_freqs(Nd, xp=xp)

    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, useloop=True)
    xp.testing.assert_allclose(xd, xl, atol=1e-7)

    Xp = xp.exp(-1j * xp.dot(om, xp.asarray(n_shift)))
    Xp = X * Xp.reshape(X.shape, order="F")
    xf = xp.fft.ifftn(Xp) * np.prod(Nd)
    xp.testing.assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print("loop max %% difference = %g" % max_percent_diff(xl, xd))
        print("ifft max %% difference = %g" % max_percent_diff(xf, xd))

    if test_Cython:
        import time
        from mrrt.nufft.cy_dtft import dtft_adj as cy_dtft_adj

        t_start = time.time()
        xc = cy_dtft_adj(X.ravel(order="F"), om, Nd, n_shift)
        print("duration (1 rep) = {}".format(time.time() - t_start))
        print("ifft max %% difference = %g" % max_percent_diff(xf, xc))

        X_16rep = xp.tile(X.ravel(order="F")[:, None], (1, 16))
        t_start = time.time()
        cy_dtft_adj(X_16rep, om, Nd, xp.asarray(n_shift))
        print("duration (16 reps) = {}".format(time.time() - t_start))
        t_start = time.time()
        X_64rep = xp.tile(X.ravel(order="F")[:, None], (1, 64))
        xc64 = cy_dtft_adj(X_64rep, om, Nd, xp.asarray(n_shift))
        max_percent_diff(xf, xc64[..., -1])
        print("duration (64 reps) = {}".format(time.time() - t_start))
    #        %timeit xd = dtft_adj(X_16rep, om, Nd, n_shift);

    return


@pytest.mark.parametrize("xp", all_xp)
def test_dtft_adj_2d(xp, verbose=False):
    Nd = (4, 6)
    n_shift = np.asarray([2, 1]).reshape(2, 1)

    rstate = xp.random.RandomState(1234)
    X = rstate.standard_normal(Nd)  # test signal
    X = X + 1j * rstate.standard_normal(Nd)

    # test with uniform frequency locations:
    om = _uniform_freqs(Nd, xp=xp)

    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, useloop=True)
    xp.testing.assert_allclose(xd, xl, atol=1e-7)

    Xp = xp.exp(-1j * xp.dot(om, xp.asarray(n_shift)))
    Xp = X * Xp.reshape(X.shape, order="F")
    xf = xp.fft.ifftn(Xp) * np.prod(Nd)
    xp.testing.assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print("ifft max %% difference = %g" % max_percent_diff(xf, xd))
    return


@pytest.mark.parametrize("xp", all_xp)
def test_dtft_adj_1d(xp, verbose=False):
    Nd = (16,)
    n_shift = np.asarray([2]).reshape(1, 1)

    rstate = xp.random.RandomState(1234)
    X = rstate.standard_normal(Nd)  # test signal
    X = X + 1j * rstate.standard_normal(Nd)

    # test with uniform frequency locations:
    om = _uniform_freqs(Nd, xp=xp)

    xd = dtft_adj(X, om, Nd, n_shift)
    xl = dtft_adj(X, om, Nd, n_shift, useloop=True)
    xp.testing.assert_allclose(xd, xl, atol=1e-7)

    Xp = xp.exp(-1j * xp.dot(om, xp.asarray(n_shift)))
    Xp = X * Xp.reshape(X.shape, order="F")
    xf = xp.fft.ifftn(Xp) * np.prod(Nd)
    xp.testing.assert_allclose(xd, xf, atol=1e-7)
    if verbose:
        print("ifft max %% difference = %g" % max_percent_diff(xf, xd))
    return
