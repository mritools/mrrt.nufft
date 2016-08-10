""" Simple NUFFT kernels not recommended for general use.  Available mainly
for comparison purposes.
"""
import functools
import numpy as np


__all__ = ['nufft_gauss',
           'nufft_best_gauss',
           'get_diric_kernel',
           'cos3diric_kernel',
           'linear_kernel',
           'nufft_diric']


def nufft_gauss(J=6, sig=None):
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
        sig = 0.78 * np.sqrt(J)

    if True:
        def kernel(k, J, sig=sig):
            """Gaussian kernel"""
            return np.exp(-(k/sig)**2/2.) * (abs(k) < J/2.)

        def kernel_ft(t, sig=sig):
            """FT of Gaussian kernel"""
            tmp = np.sqrt(2*np.pi)
            return sig*tmp*np.exp(-np.pi*(t*sig*tmp)**2)
    else:
        kernel = 'np.exp(-(k/%g)**2/2.) * (abs(k) < J/2.)' % sig
        kernel_ft = \
            '%g*np.sqrt(2*np.pi)*np.exp(-np.pi*(t*%g*np.sqrt(2*np.pi))**2)' % (
                sig, sig)
        kernel = eval('lambda k, J: ' + kernel)
        kernel_ft = eval('lambda t: ' + kernel_ft)
    return kernel, kernel_ft


def nufft_best_gauss(J, K_N=2, sn_type='ft'):
    """ Return "sigma" of best (truncated) gaussian for NUFFT with previously
    numerically-optimized width.

    Parameters
    ----------
    J : int
        # of neighbors used per frequency location
    K_N : float, optional
        K/N grid oversampling ratio
    sn_type : {'zn', 'ft'}
        method for calculating the rolloff prefilter ('ft' recommended)

    Returns
    -------
    sig : float
        best sigma
    kernel:
        string for inline kernel function, args (k,J)
    kernel_ft:
        string for Fourier transform function, arg: (t)

    Notes
    -----
    Matlab vn. Copyright 2002-4-11, Jeff Fessler, The University of Michigan
    """

    if K_N != 2:
        raise ValueError('ERROR in %s: only K/N=2 done')

    Jgauss2 = np.arange(2, 16)
    Sgauss2 = {}
    Sgauss2['zn'] = [
        0.4582,
        0.5854,
        0.6600,
        0.7424,
        0.8083,
        0.8784,
        0.9277,
        0.9840,
        1.0436,
        1.0945,
        1.1432,
        1.1898,
        1.2347,
        1.2781,
        1.3120]

    Sgauss2['ft'] = [
        0.4441,
        0.5508,
        0.6240,
        0.7245,
        0.7838,
        0.8519,
        0.9221,
        0.9660,
        1.0246,
        1.0812,
        1.1224,
        1.1826,
        1.2198,
        1.2626,
        1.3120]

    if np.sum(J == Jgauss2) != 1:
        print("user specified J = {}".format(J))
        raise ValueError('only J in the range [2-15] available')
    else:
        Jidx = np.where(J == Jgauss2)[0][0]

    sn_type = sn_type.lower()
    if sn_type == 'ft':
        sig = Sgauss2['ft'][Jidx]
    elif sn_type == 'zn':
        sig = Sgauss2['zn'][Jidx]
    else:
        raise ValueError('bad sn_type {}'.format(sn_type))

    [kernel, kernel_ft] = nufft_gauss(J, sig)
    return sig, kernel, kernel_ft


def linear_kernel(k, J):
    return (1 - abs(k/(J/2.))) * (abs(k) < J/2.)


def _scale_tri(N, J, K, Nmid):
    """
    scale factors when kernel is 'linear'
    tri(u/J) <-> J sinc^2(J x)
    """
    nc = np.arange(N, dtype=np.float64) - Nmid

    def fun(x):
        return J * np.sinc(J * x / K) ** 2

    cent = fun(nc)
    sn = 1 / cent

    # try the optimal formula
    tmp = 0
    LL = 3
    for ll in range(-LL, LL + 1):
        tmp += np.abs(fun(nc - ll * K)) ** 2
    sn = cent / tmp
    return sn


def cos3diric_kernel(k, J):
    from pyir.utils import diric
    tmp = 2*np.pi*k/J
    return diric(tmp, J) * np.cos(tmp/2.)**3


def nufft_diric(k, N, K, use_true_diric=False):
    ''' "regular fourier" Dirichlet-function WITHOUT phase

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
    '''
    if use_true_diric:
        # diric version
        t = (np.pi / K) * k
        f = np.sin(t)
        i = np.abs(f) > 1e-12  # nonzero argument
        f[i] = np.sin(N * t[i]) / (N * f[i])
        f[~i] = np.sign(np.cos(t[~i] * (N - 1)))
    else:
        # sinc version
        f = np.sinc(k * N / K)
    return f


def _diric_kernel(k, J, N):
    d = nufft_diric(k, N, N, use_true_diric=True)
    return d * (abs(k) < J/2.)


def get_diric_kernel(N):
    """Return s a dirichlet kernel function for a specified grid size, N"""
    return functools.partial(_diric_kernel, N=N)
