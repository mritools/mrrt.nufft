"""
Slow, brute force n-dimensional DTFT routines. These can be used to
validate the NUFFT on small problem sizes.

These are heavily modified (for n-dimensional support) from Matlab code
originally developed by Jeff Fessler and his students.
"""

import numpy as np

from mrrt.utils import get_array_module

__all__ = ["dtft", "dtft_adj"]


def dtft(x, omega, shape=None, n_shift=None, useloop=False, xp=None):
    """Compute exact (slow) n-dimensional non-uniform Fourier transform.

    This function is used as a reference for testing the NUFFT. It is not a
    fast transform.

    Parameters
    ----------
    x : ndarray
        The data to transform.
    omega : ndarray, optional
        Frequency locations (radians).
    shape : tuple of int, optional
        The shape of the transform. If not specified and
        ``x.ndim == omega.shape[1]`` all axes are transformed. Otherwise, all
        but the last axis is transformed. User-specified shape can be used to
        allow operator on raveled input, ``x``, but only if
        ``x.ravel(order='F')`` was used.
    n_shift : tuple of int, optional
        Spatial indices correspond to ``np.arange(n) - n_shift`` where ``n``
        is the size of x on a given axis.
    useloop : bool, optional
        If True, less memory is used (slower).
    xp : {numpy, cupy}
        The array module to use.

    Returns
    -------
    xk : array
        DTFT values

    Notes
    -----
    Requires enough memory to store M * prod(shape) size matrices
    (for testing only)

    Matlab version: Copyright 2001-9-17, Jeff Fessler,
                    The University of Michigan
    """
    xp, on_gpu = get_array_module(omega, xp)
    x = xp.asarray(x)
    omega = xp.asarray(omega)
    if omega.ndim != 2:
        raise ValueError("omega must be 2d")
    dd = omega.shape[1]
    if shape is None:
        if x.ndim == dd + 1:
            shape = x.shape[:-1]
        elif x.ndim == dd:
            shape = x.shape
        else:
            raise ValueError("shape must be specified")
    shape = xp.atleast_1d(shape)

    if len(shape) == dd:  # just one image
        x = x.ravel(order="F")
        x = x[:, xp.newaxis]
    elif len(shape) == dd + 1:  # multiple images
        shape = shape[:-1]
        x = xp.reshape(x, (xp.prod(shape), -1))  # [*shape,L]
    else:
        print("bad input signal size")

    if n_shift is None:
        n_shift = np.zeros(dd)
    n_shift = np.atleast_1d(np.squeeze(n_shift))
    if len(n_shift) != dd:
        raise ValueError("must specify one shift per axis")

    if np.any(n_shift != 0):
        nng = []
        for d in range(dd):
            nng.append(xp.arange(0, shape[d]) - n_shift[d])
        nng = xp.meshgrid(*nng, indexing="ij")
    else:
        nng = xp.indices(shape)

    if useloop:
        #
        # loop way: slower but less memory
        # Could make a numba version of this if desired
        #
        m = len(omega)
        xk = xp.zeros(
            (x.size // xp.prod(shape), m),
            dtype=xp.result_type(x.dtype, omega.dtype, xp.complex64),
        )  # [L,m]
        if omega.shape[1] < 3:
            # trick: make '3d'
            omega = xp.hstack((omega, xp.zeros(omega.shape[0])[:, xp.newaxis]))
        for d in range(dd):
            nng[d] = nng[d].ravel(order="F")
        for mm in range(0, m):
            tmp = omega[mm, 0] * nng[0]
            for d in range(1, dd):
                tmp += omega[mm, d] * nng[d]
            xk[:, mm] = xp.dot(xp.exp(-1j * tmp), x)
        xk = xk.T  # [m,L]
    else:
        xk = xp.outer(omega[:, 0], nng[0].ravel(order="F"))
        for d in range(1, dd):
            xk += xp.outer(omega[:, d], nng[d].ravel(order="F"))
        xk = xp.dot(xp.exp(-1j * xk), x)

    if xk.shape[-1] == 1:
        xk.shape = xk.shape[:-1]

    return xk


def dtft_adj(xk, omega, shape=None, n_shift=None, useloop=False, xp=None):
    """Compute adjoint of d-dim DTFT for spectrum xk at frequency locations
    omega.

    Parameters
    ----------
    xk : array
        DTFT values
    omega : array, optional
        frequency locations (radians)
    n_shift : array, optional
        indexed as range(0, N)-n_shift
    useloop : bool, optional
        True to reduce memory use (slower)

    Returns
    -------
    x : array
        signal values

    Requires enough memory to store m * (*shape) size matrices.
    (For testing only)
    """
    if xp is None:
        xp, on_gpu = get_array_module(omega)
    else:
        on_gpu = xp != np
    dd = omega.shape[1]
    if shape is None:
        if xk.ndim == dd + 1:
            shape = xk.shape[:-1]
        elif xk.ndim == dd:
            shape = xk.shape
        else:
            raise ValueError("shape must be specified")
    shape = xp.atleast_1d(shape)
    if len(shape) == dd:  # just one image
        xk = xk.ravel(order="F")
        xk = xk[:, xp.newaxis]
    elif len(shape) == dd + 1:  # multiple images
        shape = shape[:-1]
        xk = xp.reshape(xk, (xp.prod(shape), -1))  # [*shape,L]
    else:
        print("bad input signal size")

    if len(shape) != dd:
        raise ValueError(
            "length of shape must match number of columns in omega"
        )

    if n_shift is None:
        n_shift = (0,) * dd
    elif np.isscalar(n_shift):
        n_shift = (n_shift,) * dd
    if len(n_shift) != dd:
        raise ValueError("must specify one shift per axis")
    n_shift = xp.asarray(n_shift)

    if any(s != 0 for s in n_shift):
        nn = []
        for idx in range(dd):
            nn.append(xp.arange(0, shape[idx]) - n_shift[idx])
        nn = xp.meshgrid(*nn, indexing="ij")
    else:
        nn = xp.indices(shape)

    if on_gpu and isinstance(shape, xp.ndarray):
        shape = shape.get()
    shape = tuple(shape)

    if useloop:
        # slower, but low memory
        # Could make a numba version of this if desired
        m = omega.shape[0]
        x = xp.zeros(shape)  # [(shape), m]
        for mm in range(0, m):
            t = omega[mm, 0] * nn[0]
            for d in range(1, dd):
                t += omega[mm, d] * nn[d]
            x = x + xp.exp(1j * t) * xk[mm]
    else:
        x = xp.outer(nn[0].ravel(order="F"), omega[:, 0])
        for d in range(1, dd):
            x += xp.outer(nn[d].ravel(order="F"), omega[:, d])
        x = xp.dot(xp.exp(1j * x[:, xp.newaxis]), xk)  # [(*shape),L]
        x = xp.reshape(x, shape, order="F")  # [(shape),L]

    if x.shape[-1] == 1:
        x.shape = x.shape[:-1]

    return x
