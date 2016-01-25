from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
from scipy.special import iv, jv
from grl_utils import reale, is_string_like

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
        shape parameter
    kb_m : float, optional
        order parameter (default: 0)
    K_N :
        grid oversampling factor (typically 1.25 < K_N <= 2)

    Returns
    -------
    kb : array_like or str or function
        [M,1] KB function values, if x is an array of numbers
        or string for kernel(k,J), if x is 'string'
        or inline function, if x is 'inline'
    alpha
        shape parameter
    kb_m
        order parameter

    Notes
    -----
    shape parameter "alpha" (default 2.34 J)
    order parameter "kb_m" (default 0)
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

    if isinstance(alpha, str):
        alpha, kb_m = _kaiser_bessel_params(alpha, J, K_N)
        # print "alpha = ", alpha

    if isinstance(x, str):
        if isinstance(alpha, str):
            if not K_N:
                raise ValueError('ERROR in %s:  K_N required' % __name__)
            kb = lambda k, J: kaiser_bessel(k, J, alpha, None, K_N)[0]
        else:
            kb = lambda k, J: kaiser_bessel(k, J, alpha, kb_m)[0]
        return kb, alpha, kb_m

    """Warn about use of modified formula for negative kb_m"""
    if (kb_m < 0) & ((abs(round(kb_m) - kb_m)) > np.finfo(float).eps):
        # persistent warned  #TODO
        # if isempty(warned)	# only print this reminder the first time
        wstr = 'Warning in %s: Negative NonInt kb_m=%g in ' % (__name__, kb_m)
        wstr += 'kaiser_bessel().\n\t- using modified definition of KB '
        wstr += 'function\n'
        warnings.warn(wstr)

    kb_m_bi = abs(kb_m)		# modified "kb_m" as described above
    ii = (2 * np.abs(x) < J).nonzero()
    tmp = (2 * x[ii] / J)
    f = np.sqrt(1 - tmp * tmp)
    denom = iv(kb_m_bi, alpha)  # denom = besseli(kb_m_bi,alpha)
    if denom == 0:
        print('%s:  m=%g alpha=%g' % (__name__, kb_m, alpha))
    kb = np.zeros_like(x)
    kb[ii] = (f ** kb_m * iv(kb_m_bi, alpha * f)) / float(denom)
    kb = kb.real  # TODD:  reale

    return kb, alpha, kb_m


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

    if is_string_like(u):
        kernel_ft = 'kaiser_bessel_ft(t, %d, %g, %g, 1)' % (J, alpha, kb_m)
        if u == 'string':
            y = kernel_ft
        elif u == 'inline':
            y = eval('lambda t: ' + kernel_ft)
        else:
            raise ValueError('ERROR in %s: bad argument for u' % __name__)
        return y

    # persistent warned  #TODO
    if (kb_m < -1):  # Check for validity of FT formula
        # if isempty(warned)	% only print this reminder the first time
        wstr = '\nWarning in %s: kb_m=%g < -1' % (__name__, kb_m)
        wstr += ' in kaiser_bessel_ft()\n'
        wstr += '\t- validity of FT formula uncertain for kb_m < -1\n'
        warnings.warn(wstr)
    elif (kb_m < 0) & ((abs(round(kb_m) - kb_m)) > np.finfo(float).eps):
        # if isempty(warned)	% only print this reminder the first time
        wstr = '\nWarning in %s: Neg NonInt kb_m=%g in ' % (__name__, kb_m)
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


def _kaiser_bessel_params(alpha='best', J=6, K_N=2):
    """ optimized shape and order parameters"""
    if alpha == 'best':
        if K_N == 2:
            kb_m = 0  # hardwired, because it was nearly the best!

            # manually replicate the file private/kaiser,m=0.mat
            Jlist = np.arange(2, 17)
            abest = np.array([2.5,
                              2.27,
                              2.31,
                              2.34,
                              2.32,
                              2.32,
                              2.35,
                              2.34,
                              2.34,
                              2.35,
                              2.34,
                              2.35,
                              2.35,
                              2.35,
                              2.33])

            ii = (J == Jlist)  # .nonzero()
            if(np.sum(ii) == 0):
                ii = np.abs(J - Jlist).argmin()
                warnings.warn(
                    'WARNING in %s:  J=%d not found, using %d' %
                    (__name__, J, int(
                        Jlist[ii])))
            alpha = J * abest[ii]
        else:
            wstr = ('WARNING in %s: kaiser_bessel optimized '
                    'only for K/N=2!\n'
                    '\tusing good defaults: m=0 and alpha = 2.34*J')
            warnings.warn(wstr)
            kb_m = 0
            alpha = 2.34 * J
        return alpha, kb_m
    elif alpha == 'beatty':
        # Eq. 5 of Beatty2005:  IEEE TMI 24(6):799:808, kb_m = 0
        alpha = np.pi * np.sqrt(J**2/K_N**2 * (K_N - 0.5)**2 - 0.8)
    else:
        raise ValueError('Error in %s:  unknown alpha mode' % __name__)
