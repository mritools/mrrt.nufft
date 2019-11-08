"""
Try getting FFTs from pyFFTW, falling back to numpy if pyFFTW is unavailable.

Also defines various other utility functions for centered/unitary FFTs,
determining optimal transform shapes and planning pyFFTW FFTs.
"""
from __future__ import division, absolute_import, print_function

import multiprocessing
import functools
from functools import partial
from bisect import bisect_left

import numpy as np
from numpy.compat import integer_types

from .nufft_utils import complexify
from . import config

import warnings

if config.have_pyfftw:
    _default_lib = "pyfftw"
else:
    _default_lib = "numpy"


def define_if(condition, errmsg="requested function not available"):
    """Decorator for conditional definition of a function.

    Parameters
    ----------
    condition : bool
        If True the function is called normally.  If False, a
        ``NotImplementedError`` is raised when the function is called.
    errmsg: str, optional
        The error message to print when ``condition`` is False.

    Returns
    -------
    func : function wrapper
    """

    def decorator(func):
        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if condition:
                return func(*args, **kwargs)
            else:
                raise NotImplementedError(errmsg)

        return func_wrapper

    return decorator


try:
    import pyfftw

    has_pyfftw = True

    pyfftw_threads = config.pyfftw_config.NUM_THREADS

    # default effort for numpy interface routines
    pyfftw_planner_effort = config.pyfftw_config.PLANNER_EFFORT

    # default effort when calling pyFFTW builder routines
    pyfftw_builder_effort = config.pyfftw_config.PLANNER_EFFORT

    fftn = partial(
        pyfftw.interfaces.numpy_fft.fftn,
        planner_effort=pyfftw_planner_effort,
        threads=pyfftw_threads,
    )
    fftn.__doc__ = (
        "pyFFTW-based fftn with effort {} and {} " "threads"
    ).format(pyfftw_planner_effort, pyfftw_threads)

    ifftn = partial(
        pyfftw.interfaces.numpy_fft.ifftn,
        planner_effort=pyfftw_planner_effort,
        threads=pyfftw_threads,
    )
    ifftn.__doc__ = (
        "pyFFTW-based ifftn with effort {} and {} " "threads"
    ).format(pyfftw_planner_effort, pyfftw_threads)

    # shifting
    fftshift = pyfftw.interfaces.numpy_fft.fftshift
    ifftshift = pyfftw.interfaces.numpy_fft.ifftshift
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq

    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()

    # increase cache preservation time from default of 0.1 seconds
    pyfftw.interfaces.cache.set_keepalive_time(5)

except ImportError as e:
    has_pyfftw = False
    pyfftw_threads = None
    pyfftw_planner_effort = None
    pyfftw_builder_effort = None

    fftn = np.fft.fftn
    ifftn = np.fft.ifftn

    fftshift = np.fft.fftshift
    ifftshift = np.fft.ifftshift
    fftfreq = np.fft.fftfreq


__all__ = [
    "fftn",
    "ifftn",
    "fftshift",
    "fftfreq",
    "ifftshift",
    "fftnc",
    "ifftnc",
]


# centered versions of fftn & ifftn for convenience
def fftnc(
    a,
    s=None,
    axes=None,
    pre_shift_axes=None,
    post_shift_axes=None,
    norm=None,
    fftn_func=None,
):
    """Centered, n-dimensional forward FFT.

    Parameters
    ----------
    a : array_like
        The array to transform
    s : sequence of ints, optional
        shape of the transform
    axes : sequence of ints, optional
        Specify which axes to tranform.  The default is all axes.
    pre_shift_axes : sequence of ints, optional
        Which axes to ifftshift prior to calling fftn.  Default is equal to
        axes.
    post_shift_axes : sequence of ints, optional
        Which axes to fftshift after calling fftn.  Default is equal to
        axes.
    norm : {None, 'ortho'}, optional
        If None the forward transform is unscaled while the inverse is scaled
        by 1/N (where N is the product of the sizes of the transformed axes).
        If 'ortho', the forward and inverse transforms are scaled by 1/sqrt(N).
        In this case, the transform is unitary (preserves L2 norm).
    fftn_func : function or None, optional
        Can be used to pass in a pre-planned version of fftn.  If None, the
        default fftn function from this module is used.

    Notes
    -----
    if a pre-planned FFT is used via fftn_func, then ``s``, ``axes`` and
    ``norm`` arguments are ignored in favor of their pre-planned values.

    Returns
    -------
    y : array_like
        The transformed data
    """
    if pre_shift_axes is None:
        pre_shift_axes = axes
    if post_shift_axes is None:
        post_shift_axes = axes

    y = ifftshift(a, axes=pre_shift_axes)
    if config.have_pyfftw and isinstance(fftn_func, pyfftw.pyfftw.FFTW):
        # pre-planned so cannot pass s or axes
        y = fftn_func(y).copy()
        # copy may be needed in some cases with repeated calls
        # see, e.g.:  https://github.com/pyFFTW/pyFFTW/issues/139
        # this was definitely the case for ifftn_func in one application I
        # tested, but not sure about fftn_func.  copying anyway to be safe.
    else:
        if fftn_func is None:
            fftn_func = fftn
        y = fftn_func(y, s=s, axes=axes, norm=norm)
    return fftshift(y, axes=post_shift_axes)


def ifftnc(
    a,
    s=None,
    axes=None,
    pre_shift_axes=None,
    post_shift_axes=None,
    norm=None,
    ifftn_func=None,
):
    """Centered, n-dimensional inverse FFT.

    Parameters
    ----------
    a : array_like
        The array to transform
    s : sequence of ints, optional
        shape of the transform
    axes : sequence of ints, optional
        Specify which axes to tranform.  The default is all axes.
    pre_shift_axes : sequence of ints, optional
        Which axes to ifftshift prior to calling fftn.  Default is equal to
        axes.
    post_shift_axes : sequence of ints, optional
        Which axes to fftshift after calling fftn.  Default is equal to
        axes.
    norm : {None, 'ortho'}, optional
        If None the forward transform is unscaled while the inverse is scaled
        by 1/N (where N is the product of the sizes of the transformed axes).
        If 'ortho', the forward and inverse transforms are scaled by 1/sqrt(N).
        In this case, the transform is unitary (preserves L2 norm).
    ifftn_func : function or None, optional
        Can be used to pass in a pre-planned version of ifftn.  If None, the
        default ifftn from this module is used.

    Notes
    -----
    if a pre-planned IFFT is used via ifftn_func, then ``s``, ``axes`` and
    ``norm`` arguments are ignored in favor of their pre-planned values.

    Returns
    -------
    y : array_like
        The transformed data
    """
    if pre_shift_axes is None:
        pre_shift_axes = axes
    if post_shift_axes is None:
        post_shift_axes = axes

    y = ifftshift(a, axes=pre_shift_axes)
    if config.have_pyfftw and isinstance(ifftn_func, pyfftw.pyfftw.FFTW):
        # pre-planned so cannot pass s or axes
        y = ifftn_func(y).copy()
        # copy needed in some cases with repeated calls
        # see, e.g.:  https://github.com/pyFFTW/pyFFTW/issues/139
    else:
        if ifftn_func is None:
            ifftn_func = ifftn
        # recent numpy and pyFFTW support a "norm" argument
        y = ifftn(y, s=s, axes=axes, norm=norm)
    return fftshift(y, axes=post_shift_axes)
