from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from pyir.utils._cupy import get_array_module
from pyir.utils import profile

__all__ = ['_nufft_samples',
           '_nufft_interp_zn',
           '_nufft_offset',
           'to_1d_int_array',
           ]


@profile
def _nufft_samples(stype, Nd=None, xp=np):
    """  default simple EPI sampling patterns

    Parameters
    ----------
    stype : str
        sampling pattern to generate
    Nd : array_like
        number of samples to generate

    Returns
    -------
    om : array_like
        k-space samples
    """
    if isinstance(Nd, int):
        Nd = np.array([Nd])  # convert to array

    pi = xp.pi
    twopi = 2 * pi
    if stype == 'epi':  # blipped echo-planar cartesian samples
        if len(Nd) == 1:
            Nd = Nd[0]  # convert to int
            om = (twopi / float(Nd)) * xp.arange(-Nd / 2., Nd / 2.)
        elif len(Nd) == 2:
            o1 = (twopi / float(Nd[0])) * xp.arange(-Nd[0] / 2., Nd[0] / 2.)
            o2 = (twopi / float(Nd[1])) * xp.arange(-Nd[1] / 2., Nd[1] / 2.)
            o1 = xp.tile(o1, (o2.shape[0], 1)).T
            o2 = xp.tile(o2, (o1.shape[0], 1))  # [o1 o2] = ndgrid(o1, o2)
            # [o1,o2]=xp.meshgrid(o2,o1)
            o1copy = o1.copy()
            # CANNOT DO THIS IN-PLACE, MUST USE THE COPY!!
            o1[:, 1::2] = xp.flipud(o1copy[:, 1::2])

            om = xp.zeros((o1.size, 2))
            om[:, 0] = o1.T.ravel()
            om[:, 1] = o2.T.ravel()
        else:
            raise ValueError('only 1d and 2d "epi" implemented')
    else:
        raise ValueError('unknown sampling type "%s"' % stype)
    return om


def _nufft_interp_zn(alist, N, J, K, func, n_mid=None, xp=None):
    """ compute the "zn" terms for a conventional "shift-invariant" interpolator
        as described in T-SP paper.  needed for error analysis and for user-
        defined kernels since I don't provide a means to put in an analytical
        Fourier transform for such kernels.

    Parameters
    ----------
    alist : array_like
        omega / gamma (fractions) in [0,1)
    N : int
    J : int
    K : int
    func : function
        func(k,J) support limited to [-J/2, J/2).  interpolator should not
        include the linaer phase term.  this routine provides it

    Returns
    -------
    zn : array_like
        [N,M]

    Matlab vers:  Copyright 2001-12-11 Jeff Fessler. The University of Michigan
    """
    if not n_mid:
        n_mid = (N - 1) / 2.  # default: old version

    xp, on_gpu = get_array_module(alist, xp)
    alist = xp.atleast_1d(alist)
    #
    # zn = \sum_{j=-J/2}^{J/2-1} exp(i gam (alf - j) * n) F1(alf - j)
    #    = \sum_{j=-J/2}^{J/2-1} exp(i gam (alf - j) * (n-n0)) F0(alf - j)
    #

    gam = 2 * np.pi / float(K)

    if (alist.min() < 0) or (alist.max() > 1):
        warnings.warn('value in alist exceeds [0,1]')

    # natural phase function. trick: force it to be 2pi periodic
    # Pfunc = inline('exp(-i * mod0(om,2*pi) * (N-1)/2)', 'om', 'N')

    if not np.remainder(J, 2):  # even
        jlist = xp.arange(-J / 2. + 1, J / 2. + 1)
    else:  # odd
        jlist = xp.arange(-(J - 1) / 2., (J - 1) / 2. + 1)
        alist[alist > 0.5] = 1 - alist[alist > 0.5]

    # n0 = (N-1)/2.;
    # nlist0 = np.arange(0,N) - n0;     # include effect of phase shift!
    n0 = xp.arange(0, N) - n_mid

    nn0, jj = xp.ogrid[n0[0]:n0[-1] + 1, jlist[0]:jlist[-1] + 1]

    # must initialize zn as complex
    zn = xp.zeros((N, len(alist)), dtype=np.complex64)

    for ia, alf in enumerate(alist):
        jarg = alf - jj         # [N,J]
        e = xp.exp(1j * gam * jarg * nn0)       # [N,J]
        # TODO: remove need for this try/except
        try:
            F = func(jarg, J)           # [N,J]
        except:
            F = func(jarg)           # [N,J]
        zn[:, ia] = xp.sum(F * e, axis=1)
    return zn


@profile
def _nufft_offset(om, J, K, xp=None):
    """ offset for NUFFT

    Parameters
    ----------
    om : array_like
        omega in [-pi, pi) (not essential!)
    J : int
        # of neighbors used
    K : int
        FFT size

    Returns
    -------
    k0 : array_like
        offset for NUFFT

    Notes
    -----
    Matlab version Copyright 2000-1-9, Jeff Fessler, The University of Michigan
    """
    if xp is None:
        xp, on_gpu = get_array_module(om)
    om = xp.asanyarray(om)
    gam = 2 * np.pi / K
    k0 = xp.floor(om / gam - J / 2.)  # new way

    return k0


@profile
def _nufft_coef(om, J, K, kernel, xp=None):
    """  Make NUFFT interpolation coefficient vector given kernel function.

    Parameters
    ----------
    om : array_like
        [M,1]   digital frequency omega in radians
    J : int
        # of neighbors used per frequency location
    K : int
        FFT size (should be >= N, the signal_length)
    kernel : function
        kernel function

    Returns
    -------
    coef : array_like
        [J,M]   coef vector for each frequency
    arg : array_like
        [J,M]   kernel argument

    Notes
    -----
    Matlab version Copyright 2002-4-11, Jeff Fessler, The University of
    Michigan.
    """
    xp, on_gpu = get_array_module(om, xp)

    om = xp.atleast_1d(xp.squeeze(om))
    if om.ndim > 1:
        raise ValueError("omega array must be 1D")
    # M = om.shape[0];
    gam = 2 * np.pi / K
    dk = om / gam - _nufft_offset(om, J, K, xp=xp)     # [M,1]

    # outer sum via broadcasting
    arg = -xp.arange(1, J + 1)[:, None] + dk[None, :]  # [J,M]
    try:
        # try calling kernel without J in case it is baked into the kernel
        coef = kernel(arg)
    except TypeError:
        # otherwise, provide J to the kernel
        coef = kernel(arg, J=J)

    return (coef, arg)


def to_1d_int_array(arr, n=None, dtype_out=np.intp, xp=None):
    """ convert to 1D integer array.  returns an error if the elements of arr
    aren't an integer type or arr has more than one non-singleton dimension.

    If `n` is specified, an error is raised if the array doesn't contain
    `n` elements.
    """
    if xp is None:
        xp, on_gpu = get_array_module(arr)
    arr = xp.atleast_1d(arr)
    if arr.ndim > 1:
        arr = xp.squeeze(arr)
        if arr.ndim > 1:
            raise ValueError("dimensions of arr cannot exceed 1")
        if arr.ndim == 0:
            arr = xp.atleast_1d(arr)
    if not issubclass(arr.dtype.type, np.integer):
        # float only OK if values are integers
        if not xp.all(xp.mod(arr, 1) == 0):
            print("arr = {}".format(arr))
            raise ValueError("arr contains non-integer values")
    if n is not None:
        if arr.size != n:
            if arr.size == 1:
                arr = xp.asarray([arr[0], ] * n)
            else:
                raise ValueError(
                    "array did not have the expected size of {}".format(n))

    return arr.astype(dtype_out)
