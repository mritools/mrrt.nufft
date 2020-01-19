import numpy as np

from mrrt.utils import get_array_module, profile


__all__ = []


"""
Notes
-----
Matlab code from which this was adapted was written by:
jeff Fessler, The University of Michigan.

Python port: Gregory R. Lee
"""


@profile
def _nufft_offset(om, j, k, xp=None):
    """ offset for NUFFT

    Parameters
    ----------
    om : array_like
        omega in [-pi, pi) (not essential!)
    j : int
        # of neighbors used
    k : int
        FFT size

    Returns
    -------
    k0 : array_like
        offset for NUFFT
    """
    if xp is None:
        xp, on_gpu = get_array_module(om)
    om = xp.asanyarray(om)
    gam = 2 * np.pi / k
    k0 = xp.floor(om / gam - j / 2.0)  # new way

    return k0


@profile
def _nufft_coef(om, j, k, kernel, xp=None):
    """  Make NUFFT interpolation coefficient vector given kernel function.

    Parameters
    ----------
    om : array_like
        [M,1]   digital frequency omega in radians
    j : int
        # of neighbors used per frequency location
    k : int
        FFT size (should be >= N, the signal_length)
    kernel : function
        kernel function

    Returns
    -------
    coef : array_like
        [j,M]   coef vector for each frequency
    arg : array_like
        [j,M]   kernel argument
    """
    xp, on_gpu = get_array_module(om, xp)

    om = xp.atleast_1d(xp.squeeze(om))
    if om.ndim > 1:
        raise ValueError("omega array must be 1D")
    # M = om.shape[0];
    gam = 2 * np.pi / k
    dk = om / gam - _nufft_offset(om, j, k, xp=xp)  # [M,1]

    # outer sum via broadcasting
    arg = -xp.arange(1, j + 1)[:, None] + dk[None, :]  # [j,M]
    try:
        # try calling kernel without j in case it is baked into the kernel
        coef = kernel(arg)
    except TypeError:
        # otherwise, provide j to the kernel
        coef = kernel(arg, j=j)

    return (coef, arg)


def _as_1d_ints(arr, n=None, xp=None):
    """Make sure arr is a 1D array of integers.

    Returns an error if the elements of ``arr`` aren't an integer type or if
    ``arr`` has more than one non-singleton dimension.

    Parameters
    ----------
    arr : array-like
        The array to check. If it is a scalar and ``n`` is specified, it will
        be broadcast to length ``n``.
    n : int, optional
        If specified, an error is raised if the array doesn't contain ``n``
        elements.
    xp : {numpy, cupy}
        The array module.

    Returns
    iarr : xp.ndarray
        ``arr`` cast to np.intp dtype.
    """
    if xp is None:
        xp, on_gpu = get_array_module(arr)
    arr = xp.atleast_1d(xp.squeeze(arr))
    if arr.ndim > 1:
        raise ValueError("arr must be scalar or 1d")
    if not issubclass(arr.dtype.type, np.integer):
        # float only OK if values are integers
        if not xp.all(xp.mod(arr, 1) == 0):
            raise ValueError("arr contains non-integer values")
    if n is not None:
        if arr.size != n:
            if arr.size == 1:
                arr = xp.asarray([arr[0]] * n)
            else:
                raise ValueError(
                    "array did not have the expected size of {}".format(n)
                )
    return arr.astype(np.intp)  # case to ints


def _as_tuple(seq, type=int, n=None):
    if np.isscalar(seq):
        if n is None:
            raise ValueError("for scalar, seq, n must be specified")
        return (type(seq),) * n
    elif n is not None and len(seq) != n:
        raise ValueError("array did not have the expected size of {}".format(n))
    return tuple(type(s) for s in seq)
