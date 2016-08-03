"""
The code in this module is a based on Matlab routines originally created by
Jeff Fessler and his students at the University of Michigan.  The original
license for the Matlab code is reproduced below.

 License

    You may freely use and distribute this software as long as you retain the
    author's name (myself and/or my students) with the software.
    It would also be courteous for you to cite the toolbox and any related
    publications in any papers that present results based on this software.
    UM and the authors make all the usual disclaimers about liability etc.

"""
from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
from scipy.special import iv, jv
from pyir.utils import reale, is_string_like

__all__ = ['kaiser_bessel', 'kaiser_bessel_ft']


def kaiser_bessel(x=None, J=6, alpha=None, kb_m=0, K_N=None):
    '''  generalized Kaiser-Bessel function for x in support [-J/2,J/2]

    Parameters
    ----------
    x : array_like or str
        arguments [M,1]
    J : int, optional
        kernel size in each dimension
    alpha : float, optional
        shape parameter (default 2.34 * J)
    kb_m : float, optional
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
    - for Negative & NonInteger kb_m the besseli() function has
      singular behavior at the boundaries - KB values shooting-up/down
      (worse for small alpha) leading to unacceptable interpolators
    - for real arguments and higher/reasonable values of alpha the
      besseli() gives similar values for positive and negative kb_m
      except close to boundaries - tested for kb_m=-2.35:0.05:2.35
      (besseli() gives exactly same values for integer +- kb_m)
       => besseli(kb_m,...) approximated by besseli(abs(kb_m),...), which
      behaves well at the boundaries
      WARNING: it is not clear how correct the FT formula (JOSA) is
      for this approximation (for NonInteger Negative kb_m)
      NOTE: Even for the original KB formula, the JOSA FT formula
      is derived only for m > -1 !
    '''

    if alpha is None:
        alpha = 2.34 * J

    if is_string_like(alpha):
        alpha, kb_m = _kaiser_bessel_params(alpha, J, K_N)

    """Warn about use of modified formula for negative kb_m"""
    if (kb_m < 0) and ((abs(round(kb_m) - kb_m)) > np.finfo(float).eps):
        wstr = 'Negative NonInt kb_m=%g\n' % (kb_m)
        wstr += '\t- using modified definition of KB function\n'
        warnings.warn(wstr)
    kb_m_bi = abs(kb_m)		# modified "kb_m" as described above
    ii = (2 * np.abs(x) < J).nonzero()
    tmp = (2 * x[ii] / J)
    f = np.sqrt(1 - tmp * tmp)
    denom = iv(kb_m_bi, alpha)
    if denom == 0:
        print('m=%g alpha=%g' % (kb_m, alpha))
    kb = np.zeros_like(x)
    kb[ii] = (f ** kb_m * iv(kb_m_bi, alpha * f)) / float(denom)
    kb = kb.real
    return kb


def kaiser_bessel_ft(u=None, J=6, alpha=None, kb_m=0, d=1):
    """ Fourier transform of generalized Kaiser-Bessel function, in dimension d

    Parameters
    ----------
    u : array_like
        [M,1]	frequency arguments
    J : int, optional
        kernel size in each dimension
    alpha : float, optional
        shape parameter (default: 2.34 J)
    kb_m : float, optional
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

    if not alpha:
        alpha = 2.34 * J

    # persistent warned  #TODO
    if (kb_m < -1):  # Check for validity of FT formula
        # if isempty(warned)	% only print this reminder the first time
        wstr = 'kb_m=%g < -1' % (kb_m)
        wstr += ' in kaiser_bessel_ft()\n'
        wstr += ' - validity of FT formula uncertain for kb_m < -1\n'
        warnings.warn(wstr)
    elif (kb_m < 0) & ((np.abs(np.round(kb_m) - kb_m)) > np.finfo(float).eps):
        # if isempty(warned)	% only print this reminder the first time
        wstr = '\nNeg NonInt kb_m=%g in ' % (kb_m)
        wstr += 'kaiser_bessel_ft()\n\t- validity of FT formula uncertain\n'
        warnings.warn(wstr)

    # trick: scipy.special.jv can handle complex args
    tmp = (np.pi * J * u)
    z = np.lib.scimath.sqrt(tmp * tmp - alpha * alpha)
    nu = d / 2. + kb_m
    y = (2 * np.pi) ** (d / 2.) * (J / 2.) ** d * alpha ** kb_m \
        / iv(kb_m, alpha) * jv(nu, z) / z ** nu
    y = reale(y.real)

    return y
