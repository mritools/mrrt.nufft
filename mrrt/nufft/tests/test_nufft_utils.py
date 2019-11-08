import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_equal, assert_
import pytest

import mrrt.nufft
from mrrt.nufft import config
from mrrt.nufft.nufft_utils import (
    _nufft_interp_zn,
    _nufft_coef,
    _nufft_offset,
    _nufft_samples,
    max_percent_diff,
    reale,
)

from mrrt.nufft._simple_kernels import linear_kernel, nufft_diric, nufft_gauss
from scipy.special import diric

# some tests load results from Fessler's Matlab implementation for comparison

if config.have_cupy:
    import cupy
    all_xp = [np, cupy]
else:
    all_xp = [np]

# data generated using Jeff Fessler's Michigan image reconstruction toolbox
pkg_dir = os.path.dirname(os.path.realpath(mrrt.nufft.__file__))
data_dir = pjoin(pkg_dir, "tests", "data")


def test_nufft_offset():
    om = 0
    K = 128  # not used for the omega=0 case tested here
    assert_equal(_nufft_offset(om, 4, K), -2)
    assert_equal(_nufft_offset(om, 5, K), -3)
    assert_equal(_nufft_offset(om, 5.5, K), -3)
    assert_equal(_nufft_offset(om, 6, K), -3)
    assert_equal(_nufft_offset(np.asarray([om]), 6.01, K), np.asarray([-4]))


@pytest.mark.parametrize("xp", all_xp)
def test_nufft_coef(xp):
    J = 6
    if xp == np:
        c, arg = _nufft_coef(0, 6, 128, linear_kernel)
    else:
        c, arg = _nufft_coef(xp.asarray([0]), 6, 128, linear_kernel)
    # assert_equal(type(c), xp.ndarray)
    # assert_equal(type(arg), xp.ndarray)
    expected_arg = np.arange(2, -J // 2 - 1, -1, dtype=float).reshape(J, 1)
    expected_c = np.asarray([1 / 3, 2 / 3, 1, 2 / 3, 1 / 3, 0]).reshape(J, 1)
    xp.testing.assert_allclose(arg, expected_arg)
    xp.testing.assert_allclose(c, expected_c)


@pytest.mark.parametrize("xp", all_xp)
def test_nufft_gauss(xp, show_figure=False):
    # help(mfilename)
    N = 256
    K = 2 * N
    J = 4
    [kernel, kernel_ft] = nufft_gauss(J, xp=xp)

    # trick due to complex phase term
    n = xp.arange(0, N).T - (N - 1) / 2.0
    sn_ft = 1 / kernel_ft(n / float(K))
    sn_zn = reale(1 / _nufft_interp_zn(xp.array([0]), N, J, K, kernel))

    expected_results = np.load(pjoin(data_dir, "nufft_gauss.npz"))
    xp.testing.assert_array_almost_equal(sn_ft, expected_results["sn_ft"])
    xp.testing.assert_array_almost_equal(sn_zn, expected_results["sn_zn"])
    if show_figure and xp == np:  # skip show_figure on GPU
        from matplotlib import pyplot as plt

        k = np.linspace(-J / 2 - 1, J / 2 + 1, 101)
        plt.figure()
        plt.subplot(121)
        plt.plot(k, kernel(k, J))
        plt.axis("tight")
        plt.xlabel("k")
        plt.ylabel(r"$\psi(k)$")
        plt.title("Gaussian bell")
        plt.subplot(122)
        l1, l2 = plt.plot(n, sn_ft, "c-o", n, sn_zn, "y-")
        plt.axis("tight")
        plt.legend((l1, l2), ("1/FT", "1/zn"), loc="upper right")
        plt.xlabel("t")
        plt.ylabel(r"$\Psi(t)$")
        plt.title("Reciprocal of Fourier transform")
    return


@pytest.mark.parametrize("xp", all_xp)
def test_nufft_samples(xp, show_figure=False):
    om1d = _nufft_samples("epi", 32, xp=xp)
    assert_(len(om1d) == 32)
    om2d = _nufft_samples("epi", [32, 32], xp=xp)
    assert_equal(om2d.shape, (32 * 32, 2))
    if show_figure and xp == np:  # skip show_figure on GPU
        from matplotlib import pyplot as plt

        plt.figure(), plt.plot(om1d, np.zeros((om1d.size, 1)), "b-x")
        plt.title("1D EPI")
        plt.figure(), plt.plot(om2d[:, 0], om2d[:, 1], "b-x"), plt.show()
        plt.title("2D EPI")
        plt.show()
    return om2d


@pytest.mark.parametrize("xp", all_xp)
def test_nufft_interp_zn(xp, show_figure=False):
    alist = xp.arange(0, 20) / 20.0
    N = 2 ** 7
    K = 2 * N

    # linear kernel test
    J = 4
    z = _nufft_interp_zn(alist, N, J, K, linear_kernel)
    expected_results = np.load(pjoin(data_dir, "nufft_interp_zn.npz"))
    xp.testing.assert_allclose(z, expected_results["z"])

    if show_figure and xp == np:  # skip show_figure on GPU
        from matplotlib import pyplot as plt

        # plot interpolator
        k = np.linspace(-J / 2.0 - 1, J / 2.0 + 1, 101)
        plt.figure()
        plt.subplot(131)
        plt.plot(k, linear_kernel(k, J))
        plt.xlabel("k")
        plt.ylabel("F_0(k)")
        plt.axis("tight")

        plt.subplot(132)
        plt.plot(np.arange(1, N + 1), z.real)
        plt.axis("tight")
        plt.subplot(133)
        plt.plot(np.arange(1, N + 1), z.imag)
        plt.axis("tight")
    return


def test_nufft_diric(show_figure=False):
    kmax = 2 * (10 + 1 * 4)
    kf = np.linspace(-kmax, kmax, 201)
    ki = np.arange(-kmax, kmax + 1)
    N = 32
    K = 2 * N
    Nlist = np.array([8, 32, 7])
    Klist = 2 * Nlist
    Klist[-1] = Nlist[-1]
    for idx, N in enumerate(Nlist):
        K = Klist[idx]
        g = nufft_diric(kf, N, K, True)
        gi = nufft_diric(ki, N, K, True)
        s = nufft_diric(kf, N, K)
        dm = diric((2 * np.pi / K) * kf, N)
        # dm = diric2((2*np.pi/K)*kf,N)
        max_diff = max_percent_diff(g, dm)
        assert_(max_diff < 1e-7)
        if show_figure:
            from matplotlib import pyplot as plt

            plt.figure()
            l1, l2, l3, l4 = plt.plot(
                kf, g, "y", kf, s, "c-", kf, dm, "r--", ki, gi, "b."
            )
            plt.axis("tight")
            plt.legend(
                (l1, l2, l3),
                ("nufft diric", "sinc", "scipy.special.diric"),
                loc="upper right",
            )
            plt.xlabel("kf")
            plt.ylabel("diric(kf)")
            plt.title("N={}, K={}".format(N, K))
            print("max %% difference vs scipy.diric = %g" % max_diff)
    return
