""" Simple NUFFT kernels not recommended for general use.  Available mainly
for comparison purposes.
"""
import functools
from math import sqrt

import numpy as np


__all__ = ["nufft_gauss", "get_diric_kernel", "linear_kernel", "nufft_diric"]


def nufft_gauss(J=6, sig=None, xp=np):
    """ Gaussian bell kernel functions truncated to support [-J/2,J/2]
        for NUFFT interplation, with width parameter sig

    Parameters
    ----------
    ktype : {'string','inline'}
        specify whether string or function should be returned
    J : int
        interval width
    sig : float
        width parameter
    xp : {np, cupy}
        The array backend to use.

    Returns
    -------
    kernel : function or str
        kernel(k,J): Gaussian kernel
    kernel_ft : function or str
        kernel_ft(t):  Fourier transform of Gaussian kernel

    Notes
    -----
    Matlab vn: Copyright 2002-7-15, Jeff Fessler, The University of Michigan
    """
    if sig is None:
        sig = 0.78 * sqrt(J)

    def kernel(k, J=J, sig=sig):
        """Gaussian kernel"""
        return xp.exp(-((k / sig) ** 2) / 2.0) * (abs(k) < J / 2.0)

    def kernel_ft(t, sig=sig):
        """FT of Gaussian kernel"""
        tmp = sqrt(2 * xp.pi)
        return sig * tmp * xp.exp(-xp.pi * (t * sig * tmp) ** 2)

    return kernel, kernel_ft


def linear_kernel(k, J):
    return (1 - abs(k / (J / 2.0))) * (abs(k) < J / 2.0)


def _scale_tri(N, J, K, n_mid, xp=np):
    """
    scale factors when kernel is 'linear'
    tri(u/J) <-> J sinc^2(J x)
    """
    nc = xp.arange(N, dtype=np.float64) - n_mid

    def fun(x):
        return J * xp.sinc(J * x / K) ** 2

    cent = fun(nc)
    sn = 1 / cent

    # try the optimal formula
    tmp = 0
    LL = 3
    for ll in range(-LL, LL + 1):
        tmp += xp.abs(fun(nc - ll * K)) ** 2
    sn = cent / tmp
    return sn


def nufft_diric(k, N, K, use_true_diric=False, xp=np):
    """ "regular fourier" Dirichlet-function WITHOUT phase

    Parameters
    ----------
    k : array_like
        sample locations
    N : int
        signal length
    K : int
        DFT length
    use_true_diric : bool, optional
        if False, use sinc approximation

    Returns
    -------
    f : float
        functional values corresponding to k

    Notes
    -----
     diric(t) = sin(pi N t / K) / ( N * sin(pi t / K) )
        \approx sinc(t / (K/N))

    Matlab vers:  Copyright 2001-12-8, Jeff Fessler, The University of Michigan
    """
    if use_true_diric:
        # diric version
        t = (xp.pi / K) * k
        f = xp.sin(t)
        i = xp.abs(f) > 1e-12  # nonzero argument
        f[i] = xp.sin(N * t[i]) / (N * f[i])
        f[~i] = xp.sign(xp.cos(t[~i] * (N - 1)))
    else:
        # sinc version
        f = xp.sinc(k * N / K)
    return f


def _diric_kernel(k, J, N):
    d = nufft_diric(k, N, N, use_true_diric=True)
    return d * (abs(k) < J / 2.0)


def get_diric_kernel(N):
    """Return s a dirichlet kernel function for a specified grid size, N"""
    return functools.partial(_diric_kernel, N=N)
