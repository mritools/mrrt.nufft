"""Kaiser-Bessel function and it's Fourier transform.

The code in this module is a based on Matlab routines originally created by
Jeff Fessler and his students at the University of Michigan.  The original
license for the Matlab code is reproduced below.

 License

    You may freely use and distribute this software as long as you retain the
    author's name (myself and/or my students) with the software.
    It would also be courteous for you to cite the toolbox and any related
    publications in any papers that present results based on this software.
    UM and the authors make all the usual disclaimers about liability etc.

Python translation and GPU support was added by Gregory R. Lee

Notes:
1.) For speed, call ``i0`` and ``j0`` instead of ``iv``, ``jv`` when order = 0.
2.) CuPy-based GPU support added. If ``order != 0``, we currently have
to compute the kernels on the CPU due to lack of an ``iv`` or ``jv``
implementation there.

"""
import warnings

import numpy as np

from scipy.special import iv, jv, i0, j0
from mrrt.utils import config, get_array_module, profile, reale

if config.have_cupy:
    from cupyx.scipy.special import j0 as j0_cupy, i0 as i0_cupy

__all__ = ["kaiser_bessel", "kaiser_bessel_ft"]


def _i0(x, xp):
    """Wrapper for i0 that calls either the CPU or GPU implementation."""
    if xp is np:
        return i0(x)
    else:
        return i0_cupy(x)


def _j0(x, xp):
    """Wrapper for j0 that calls either the CPU or GPU implementation."""
    if xp is np:
        return j0(x)
    else:
        return j0_cupy(x)


def _iv(x1, x2, xp):
    """Wrapper for iv that can accept x1 and/or x2 as CuPy or numpy arrays."""
    if xp != np:
        # no iv function in CUDA math library, so must compute on the CPU
        if not xp.isscalar(x1):
            x1 = x1.get()
        if not xp.isscalar(x2):
            x2 = x2.get()
        return xp.asarray(iv(x1, x2))
    return iv(x1, x2)


def _jv(x1, x2, xp):
    """Wrapper for jv that can accept x1 and/or x2 as CuPy or numpy arrays."""
    if xp != np:
        # no jv function in CUDA math library, so must compute on the CPU
        if not xp.isscalar(x1):
            x1 = x1.get()
        if not xp.isscalar(x2):
            x2 = x2.get()
        return xp.asarray(jv(x1, x2))
    return jv(x1, x2)


@profile
def kaiser_bessel(x=None, J=6, alpha=None, m=0, K_N=None):
    """Generalized Kaiser-Bessel function for x in support [-J/2,J/2].

    Parameters
    ----------
    x : array_like or str
        arguments [M,1]
    J : int, optional
        kernel size in each dimension
    alpha : float, optional
        shape parameter (default 2.34 * J)
    m : float, optional
        order parameter
    K_N :
        grid oversampling factor (typically 1.25 < K_N <= 2)

    Returns
    -------
    kb : array_like or str or function
        [M,1] KB function values, if x is an array of numbers
        or string for kernel(k,J), if x is 'string'
        or inline function, if x is 'inline'

    Notes
    -----
    see (A1) in lewitt:90:mdi, JOSA-A, Oct. 1990

    Adapted from Matlab version:
        Copyright 2001-3-30, Jeff Fessler, The University of Michigan

    Modification 2002-10-29 by Samuel Matej
    - for Negative & NonInteger m the besseli() function has
      singular behavior at the boundaries - KB values shooting-up/down
      (worse for small alpha) leading to unacceptable interpolators
    - for real arguments and higher/reasonable values of alpha the
      besseli() gives similar values for positive and negative m
      except close to boundaries - tested for m=-2.35:0.05:2.35
      (besseli() gives exactly same values for integer +- m)
       => besseli(m,...) approximated by besseli(abs(m),...), which
      behaves well at the boundaries
      WARNING: it is not clear how correct the FT formula (JOSA) is
      for this approximation (for NonInteger Negative m)
      NOTE: Even for the original KB formula, the JOSA FT formula
      is derived only for m > -1 !
    """
    xp, on_gpu = get_array_module(x)
    if alpha is None:
        alpha = 2.34 * J

    """Warn about use of modified formula for negative m"""
    if (m < 0) and ((abs(round(m) - m)) > np.finfo(float).eps):
        wstr = "Negative NonInt m=%g\n" % (m)
        wstr += "\t- using modified definition of KB function\n"
        warnings.warn(wstr)
    m_bi = abs(m)  # modified "m" as described above
    ii = (2 * np.abs(x) < J).nonzero()
    tmp = 2 * x[ii] / J
    tmp *= tmp
    f = np.sqrt(1 - tmp)
    if m_bi != 0:
        denom = _iv(m_bi, alpha, xp=xp)
    else:
        denom = _i0(alpha, xp=xp)
    if denom == 0:
        print("m=%g alpha=%g" % (m, alpha))
    kb = xp.zeros_like(x)
    if m_bi != 0:
        kb[ii] = (f ** m * _iv(m_bi, alpha * f, xp=xp)) / float(denom)
    else:
        kb[ii] = _i0(alpha * f, xp=xp) / float(denom)
    kb = kb.real
    return kb


@profile
def kaiser_bessel_ft(u, J=6, alpha=None, m=0, d=1):
    """Fourier transform of generalized Kaiser-Bessel function, in dimension d.

    Parameters
    ----------
    u : array_like
        [M,1]   frequency arguments
    J : int, optional
        kernel size in each dimension
    alpha : float, optional
        shape parameter (default: 2.34 J)
    m : float, optional
        order parameter (default: 0)
    d : int, optional
        dimension (default: 1)

    Returns
    -------
    y : array_like
        [M,1] transform values if x is an array of numbers
        or string for kernel_ft(k,J), if x is 'string'
        or inline function, if x is 'inline'

    Notes
    -----
    See (A3) in lewitt:90:mdi, JOSA-A, Oct. 1990.

    Matlab ver.Copyright 2001-3-30, Jeff Fessler, The University of Michigan
    Python adaptation:  Gregory Lee
    """
    xp, on_gpu = get_array_module(u)
    if not alpha:
        alpha = 2.34 * J

    if m < -1:  # Check for validity of FT formula
        wstr = "m=%g < -1" % (m)
        wstr += " in kaiser_bessel_ft()\n"
        wstr += " - validity of FT formula uncertain for m < -1\n"
        warnings.warn(wstr)
    elif (m < 0) & ((np.abs(np.round(m) - m)) > np.finfo(float).eps):
        wstr = "\nNeg NonInt m=%g in " % (m)
        wstr += "kaiser_bessel_ft()\n\t- validity of FT formula uncertain\n"
        warnings.warn(wstr)

    # trick: scipy.special.jv can handle complex args
    tmp = (np.pi * J) * u
    tmp *= tmp
    tmp -= alpha * alpha
    if xp is np:
        # lib.scimath.sqrt gives complex value instead of NaN for negative
        # inputs
        z = np.lib.scimath.sqrt(tmp)
    else:
        # no cupy.lib.scimath.sqrt, but it is just equivalent to:
        # convert tmp to complex dtype before calling xp.sqrt
        tmp_cplx = tmp.astype(
            xp.result_type(tmp.dtype, xp.complex64), copy=False
        )
        z = xp.sqrt(tmp_cplx)

    nu = d / 2.0 + m
    const1 = (2 * np.pi) ** (d / 2.0) * (J / 2.0) ** d * alpha ** m
    if m == 0:
        const1 /= _i0(alpha, xp=xp)
    else:
        const1 /= _iv(m, alpha, xp=xp)
    if nu == 0:
        y = const1 * _j0(z, xp=xp)
        y /= z
    else:
        y = const1 * _jv(nu, z, xp=xp)
        y /= z ** nu
    y = reale(y.real)
    return y
