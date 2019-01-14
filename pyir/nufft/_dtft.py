"""
Slow, brute force n-dimensional DTFT routines.  These can be used to
validate the NUFFT on small problem sizes.

These are heavily modified (for n-dimensional support) from Matlab code
originally developed by Jeff Fessler and his students.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from pyir.utils._cupy import get_array_module

__all__ = ['dtft', 'dtft_adj']


def dtft(x, omega, Nd=None, n_shift=None, useloop=False, xp=None):
    """  Compute d-dimensional DTFT of signal x at frequency locations omega

    Parameters
    ----------
    x : array
        signal values
    omega : array, optional
        frequency locations (radians)
    n_shift : array, optional
        indexed as range(0, N)-n_shift
    useloop : bool, optional
        True to reduce memory use (slower)

    Returns
    -------
    X : array
        DTFT values

    Requires enough memory to store M * prod(Nd) size matrices
    (for testing only)

    Matlab version: Copyright 2001-9-17, Jeff Fessler,
                    The University of Michigan
    """
    if xp is None:
        xp, on_gpu = get_array_module(omega)
    dd = omega.shape[1]
    if Nd is None:
        if x.ndim == dd+1:
            Nd = x.shape[:-1]
        elif x.ndim == dd:
            Nd = x.shape
        else:
            raise ValueError("Nd must be specified")
    Nd = xp.atleast_1d(Nd)

    if len(Nd) == dd:		# just one image
        x = x.ravel(order='F')
        x = x[:, xp.newaxis]
    elif len(Nd) == dd+1:  # multiple images
        Nd = Nd[:-1]
        x = xp.reshape(x, (xp.prod(Nd), -1))  # [*Nd,L]
    else:
        print('bad input signal size')

    if n_shift is None:
        n_shift = np.zeros(dd)
    n_shift = np.atleast_1d(np.squeeze(n_shift))
    if len(n_shift) != dd:
        raise ValueError("must specify one shift per axis")

    if np.any(n_shift != 0):
        nng = []
        for d in range(dd):
            nng.append(xp.arange(0, Nd[d]) - n_shift[d])
        nng = xp.meshgrid(*nng, indexing='ij')
    else:
        nng = xp.indices(Nd)

    if useloop:
        #
        # loop way: slower but less memory
        #
        M = len(omega)
        X = xp.zeros((x.size // xp.prod(Nd), M),
                     dtype=xp.result_type(x.dtype, omega.dtype,
                                          xp.complex64))  # [L,M]
        if omega.shape[1] < 3:
            # trick: make '3d'
            omega = xp.hstack((omega,
                               xp.zeros(omega.shape[0])[:, xp.newaxis]))
        for d in range(dd):
            nng[d] = nng[d].ravel(order='F')
        for mm in range(0, M):
            tmp = omega[mm, 0] * nng[0]
            for d in range(1, dd):
                tmp += omega[mm, d] * nng[d]
            X[:, mm] = xp.dot(xp.exp(-1j*tmp), x)
        X = X.T  # [M,L]
    else:
        X = xp.outer(omega[:, 0], nng[0].ravel(order='F'))
        for d in range(1, dd):
            X += xp.outer(omega[:, d], nng[d].ravel(order='F'))
        X = xp.dot(xp.exp(-1j*X), x)

    if X.shape[-1] == 1:
        X.shape = X.shape[:-1]

    return X


def dtft_adj(X, omega, Nd=None, n_shift=None, useloop=False, xp=None):
    """Compute adjoint of d-dim DTFT for spectrum X at frequency locations
    omega.

    Parameters
    ----------
    X : array
        DTFT values
    omega : array, optional
        frequency locations (radians)
    n_shift : array, optional
        indexed as range(0, N)-n_shift
    useloop : bool, optional
        True to reduce memory use (slower)

    Returns
    -------
    X : array
        signal values

    Requires enough memory to store M * (*Nd) size matrices.
    (For testing only)
    """
    if xp is None:
        xp, on_gpu = get_array_module(omega)
    else:
        on_gpu = (xp != np)
    dd = omega.shape[1]
    if Nd is None:
        if X.ndim == dd+1:
            Nd = X.shape[:-1]
        elif X.ndim == dd:
            Nd = X.shape
        else:
            raise ValueError("Nd must be specified")
    Nd = xp.atleast_1d(Nd)
    if len(Nd) == dd:		# just one image
        X = X.ravel(order='F')
        X = X[:, xp.newaxis]
    elif len(Nd) == dd+1:  # multiple images
        Nd = Nd[:-1]
        X = xp.reshape(X, (xp.prod(Nd), -1))  # [*Nd,L]
    else:
        print('bad input signal size')

    if len(Nd) != dd:
        raise ValueError("length of Nd must match number of columns in omega")

    if n_shift is None:
        n_shift = np.zeros(dd)
    n_shift = np.atleast_1d(np.squeeze(n_shift))
    if len(n_shift) != dd:
        raise ValueError("must specify one shift per axis")

    if np.any(n_shift != 0):
        nn = []
        for id in range(dd):
            nn.append(xp.arange(0, Nd[id]) - n_shift[id])
        nn = xp.meshgrid(*nn, indexing='ij')
    else:
        nn = xp.indices(Nd)

    if on_gpu:
        Nd = tuple(Nd.get())

    if useloop:
        # slower, but low memory
        M = omega.shape[0]
        x = xp.zeros(Nd)  # [(Nd), M]
        for mm in range(0, M):
            t = omega[mm, 0]*nn[0]
            for d in range(1, dd):
                t += omega[mm, d]*nn[d]
            x = x + xp.exp(1j*t) * X[mm]
    else:
        x = xp.outer(nn[0].ravel(order='F'), omega[:, 0])
        for d in range(1, dd):
            x += xp.outer(nn[d].ravel(order='F'), omega[:, d])
        x = xp.dot(xp.exp(1j*x[:, xp.newaxis]), X)  # [(*Nd),L]
        x = xp.reshape(x, Nd, order='F')  # [(Nd),L]

    if x.shape[-1] == 1:
        x.shape = x.shape[:-1]

    return x
