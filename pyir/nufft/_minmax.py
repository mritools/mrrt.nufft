"""
The code in this module is a port of Matlab routines created by Jeff Fessler
and his students at the University of Michigan.  The original license for the
Matlab code is reproduced below.

 License

    You may freely use and distribute this software as long as you retain the
    author's name (myself and/or my students) with the software.
    It would also be courteous for you to cite the toolbox and any related
    publications in any papers that present results based on this software.
    UM and the authors make all the usual disclaimers about liability etc.

"""
from __future__ import division, print_function, absolute_import

import collections
import functools
import warnings

import numpy as np
import scipy.sparse

from pyir.nufft._kaiser_bessel import (kaiser_bessel,
                                       kaiser_bessel_ft,
                                       _kaiser_bessel_params)

from pyir.nufft.nufft_utils import _nufft_offset, _nufft_interp_zn

from pyir.nufft.simple_kernels import (linear_kernel,
                                       get_diric_kernel,
                                       cos3diric_kernel,
                                       nufft_best_gauss,
                                       nufft_diric)
from pyir.utils import is_string_like

# TODO: add linear and diric options as in newfft.m


__all__ = ['nufft_alpha_kb_fit',
           'nufft_best_alpha',
           'nufft1_err_mm',
           'nufft2_err_mm',
           'nufft1_error',
           'nufft_scale']


def _nufft_r(om, N, J, K, alpha=[1], beta=0.5, use_true_diric=False):
    """  make NUFFT "r" vector

    Parameters
    ----------
    om : array_like
        [M,1] digital frequency omega in radians
    N : int
        signal length
    J : int
        # of neighbors used per frequency location
    K : int
        FFT size (should be > N)
    alpha : array_like, optional
        [0:L] Fourier series coefficients of scaling factors
    beta : float, optional
        scale gamma=2pi/K by this in Fourier series. typically is
        K/N (Fessler) or 0.5 (Liu)

    Returns
    -------
    rr : array_like
        [J,M]   r vector for each frequency
    arg : array_like
        [J,M]   dirac argument for t=0

    Notes
    -----
    Matlab vn. Copyright 2001-12-13, Jeff Fessler, The University of Michigan

    """
    alpha = np.atleast_1d(alpha)
    om = np.atleast_1d(om)

    M = len(om)

    gam = 2 * np.pi / K
    dk = om / gam - _nufft_offset(om, J, K)     # [M,1]

    # outer sum via broadcasting
    arg = (-np.arange(1, J + 1))[:, None] + dk[None, :]  # [J,M]

    L = len(alpha) - 1
    if not np.isrealobj(alpha[0]):
        raise ValueError('need real alpha_0')
    if L > 0:
        rr = np.zeros((J, M))
        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = np.conj(alf)
            r1 = nufft_diric(arg + l1 * beta, N, K, use_true_diric)
            rr = rr + alf * r1  # [J,M]
    else:
        rr = nufft_diric(arg, N, K, use_true_diric)  # [J,M]

    return rr, arg


def _nufft_T(N, J, K, alpha=None, tol=1e-7, beta=0.5, use_true_diric=False):
    """  Precompute the matrix T = [C' S S' C]\inv used in NUFFT.

    Parameters
    ----------
    N : int
        signal length
    J : int
        # of neighbors
    K : int
        FFT length
    alpha : array_like, optional
        [L+1]   Fourier coefficient vector for scaling
    tol : float, optional
        tolerance for smallest eigenvalue
    beta : float, optional
        scale gamma=2*pi/K by this for Fourier series
    use_true_diric : boolean, optional

    Returns
    -------
    T : array_like
        [J,J]   precomputed matrix

    Notes
    -----
    This can be precomputed, being independent of frequency location.
    Matlab version Copyright 2000-1-9, Jeff Fessler, The University of Michigan
    """

    if N > K:
        raise ValueError('N > K not allowed')

    # default with unity scaling factors

    if not (alpha is None):
        alpha = np.atleast_1d(alpha)
        no_alpha = (len(alpha) == 0)
    else:
        no_alpha = True

    if no_alpha:
        # compute C'SS'C = C'C
        # [j1 j2] = ndgrid(1:J, 1:J)
        # j1=np.arange(1,J+1)   j1=np.tile(j1,(j1.shape[0],1))    j2=j1.T;
        j1, j2 = np.mgrid[1:J, 1:J]
        cssc = nufft_diric(j2 - j1, N, K, use_true_diric)
    else:
        # Fourier-series based scaling factors
        alpha = np.asarray(alpha)
        if not np.isrealobj(alpha[0]):
            raise ValueError('need real alpha_0')
        L = len(alpha) - 1  # L
        cssc = np.zeros((J, J))
        j1, j2 = np.mgrid[1:J + 1, 1:J + 1]
        for l1 in range(-L, L + 1):
            for l2 in range(-L, L + 1):
                alf1 = alpha[abs(l1)]
                if l1 < 0:
                    alf1 = np.conj(alf1)
                alf2 = alpha[abs(l2)]
                if l2 < 0:
                    alf2 = np.conj(alf2)
                tmp = j2 - j1 + beta * (l1 - l2)
                tmp = nufft_diric(tmp, N, K, use_true_diric)
                cssc = cssc + alf1 * np.conj(alf2) * tmp
                # print '%d %d %s %s' % (l1, l2, num2str(alf1), num2str(alf2))

    # Inverse, or, pseudo-inverse
    # smin = svds(cssc,1,0)
    U, s, Vh = np.linalg.svd(cssc)
    smin = min(s)
    s = np.diag(s)
    if smin < tol:  # smallest singular value
        warnings.warn(
            'WARNING in %s:  Poor conditioning %g => pinverse' %
            (__name__, smin))
        T = np.linalg.pinv(cssc, tol / 10.)
    else:
        T = np.linalg.inv(cssc)

    return T


def nufft_best_alpha(J, L=2, K_N=2):
    """ return previously numerically optimized alpha and beta
        use L=0 to return best available choice of any L

    Parameters
    ----------
    J : int
        # of neighbors
    L : int
    K_N : float
        grid oversampling ratio (K/N)

    Returns
    -------
    alpha : float
        numerically optimized alpha value
    beta : float
        numerically optimized beta value
    ok : bool


    Notes
    -----
    Matlab vn. Copyright 2001-12-17 Jeff Fessler. The University of Michigan

    used by kernel type:  'minmax:tuned'
    """

    Jlist1 = np.array([6, ])        # list of which J's
    alpha1 = np.array([1, -0.46, 0.19])   # last colum is best beta  #J=6

    Jlist2 = np.arange(2, 11)       # list of which J's
    alpha2 = np.array([
        [1, -0.200, -0.04, 0.34],
        [1, -0.485, 0.090, 0.48],
        [1, -0.470, 0.085, 0.56],
        [1, -0.4825, 0.12, 0.495],
        [1, -0.57, 0.14, 0.43],
        [1, -0.465, 0.07, 0.65],
        [1, -0.540, 0.16, 0.47],
        [1, -0.625, 0.14, 0.325],
        [1, -0.57, 0.185, 0.43]
    ])   # last colum is best beta

    Jlist3 = np.array([4, 6])       # list of which J's
    alpha3 = np.array([
        [1, -0.5319, 0.1522, -0.0199, 0.6339],
        [1, -0.6903, 0.2138, -0.0191, 0.2254]
    ])  # last colum is best beta

    if K_N == 2:
        if L == 0:
            if np.any(J == Jlist3):
                L = 3
            else:
                L = 2
            # current best
        if L == 1:
            alpha = alpha1
            Jlist = Jlist1
        elif L == 2:
            alpha = alpha2
            Jlist = Jlist2
        elif L == 3:
            alpha = alpha3
            Jlist = Jlist3
        else:
            # raise ValueError('L=%d not done' % L)
            warnings.warn('L=%d not done' % L)
            alpha = np.nan
            beta = np.nan
            ok = False
            return alpha, beta, ok
    else:
        # raise ValueError('K_N=%g not done' % K_N)
        warnings.warn('K_N=%g not done' % K_N)
        alpha = 1
        beta = 0.5
        ok = True
        return alpha, beta, ok

    if np.any(J == Jlist):
        j = (J == Jlist)
        beta = alpha[j, -1]
        alpha = alpha[j, 0:-1]
        ok = True
    else:
        ok = False
        alpha = np.nan
        beta = np.nan

    alpha = np.squeeze(alpha)
    return alpha, beta, ok


def nufft_alpha_kb_fit(N, J, K, L=None, beta=1, Nmid=None, verbose=False):
    """  return the alpha and beta corresponding to LS fit of L components
         to optimized Kaiser-Bessel scaling factors (m=0, alpha=2.34J).
         This is the best method I know currently for choosing alpha!

    Parameters
    ----------
    N : int
        signal size
    J : int
        # of neighbors
    K : int
        DFT size
    L : int
    beta : float
    Nmid : float
    verbose : bool

    Notes
    -----
    used by kernel_type 'minmax:kb'

    Matlab vn. copyright 2002-7-16, Jeff Fessler, The University of Michigan

    """
    # if nargin < 3, help(mfilename), error(mfilename), end
    if L is None:
        if N > 40:
            L = 13      # empirically found to be reasonable
        else:
            # a kludge to avoid "rank deficient" complaints
            L = np.ceil(N / 3.)

    # kb_alf = 2.34 * J;    # KB shape parameter
    # kb_m = 0;     # KB order

    if (Nmid is None):
        Nmid = (N - 1) / 2.

    nlist = np.arange(N, dtype=float) - Nmid

    if False:  # old way
        [tmp, sn_kaiser] = nufft1_error(0, N, J, K, 'kaiser', 'ft')[0:2]
        # sn_kaiser = reale(sn_kaiser) #TODO: reale
        sn_kaiser = sn_kaiser.real
    else:
        # kaiser-bessel with previously numerically-optimized shape
        K_N = K / float(N)
        kb_a, kb_m = _kaiser_bessel_params('best', J=J, K_N=K_N)
        kernel_ft = functools.partial(
            kaiser_bessel_ft, J=J, alpha=kb_a, kb_m=kb_m)
        sn_kaiser = 1 / kernel_ft(nlist / float(K))  # [N]

    # use regression to match NUFFT with BEST kaiser scaling's
    gam = 2 * np.pi / K
    nlist = nlist.reshape((-1, 1))        # np.asmatrix(nlist).T
    ar = np.arange(L+1).reshape((1, -1))  # np.asmatrix(np.arange(0, L + 1))
    X = np.cos(beta * gam * nlist * ar)  # [N,L]

    # TODO: pinv(X)*sn is not always equivalent to Matlab's mldivide()
    coef = np.dot(np.linalg.pinv(X), sn_kaiser.reshape((-1, 1)))
    coef = np.squeeze(np.asarray(coef))

    alphas = np.zeros(coef.shape)
    alphas[0] = coef[0].real
    alphas[1:] = coef[1:] / 2.

    if verbose:
        print(
            'condition # for LS fit to KB scale factors: %g' %
            np.linalg.cond(X))

    return alphas, beta


def nufft1_err_mm(om, N1, J1, K1, stype='sinc', alpha=[
                  1], beta=0.5, nargout=np.Inf, verbose=False):
    """ Compute worst-case error for each input frequency for min-max 1D NUFFT

    Parameters
    ----------
    om : array_like
        [M,2] digital frequency omega in radians
    N1 : int
        signal length
    J1 : int
        # of neighbors used per frequency location
    K1 : int
        FFT size (should be > N1)
    stype : {'sinc','diric','qr'}, optional
        kernel type
    alpha : array_like, optional
        [L,1] Fourier series coefficients of scaling factors
        trick: or, "sn" if len(alpha)==len(N1)
    beta : float, optional
        scale gamma=2pi/K by this in Fourier series
        typically is K/N (Fessler) or 0.5 (Liu)
    nargout : int, optional
        can be used in some cases to avoid computation of sn and/or T1 (will
        return None)

    Returns
    -------
    err : array_like
        [M,1] worst-case error over unit-norm signals
    sn : array_like
        [N,1] scaling factors corresponding to alpha,beta
    T1 : array_like
        [J,J] T matrix

    Notes
    -----
    Matlab vn. copyright 2001-12-7, Jeff Fessler, The University of Michigan
    """

    pi = np.pi

    use_qr = False
    if stype == 'sinc':
        use_true_diric = False
    elif stype == 'diric':
        use_true_diric = True
    elif stype == 'qr':
        use_qr = True
    else:
        raise ValueError('unknown stype: %s' % stype)

    if verbose:
        print('alpha={}'.format(alpha))

    # see if 'best' alpha is desired
    if isinstance(alpha, str):
        if alpha == 'uniform':
            alpha = [1]
            beta = 0.5
        else:
            if alpha == 'best':
                L = 0
            elif alpha == 'best,L=1':
                L = 1
            elif alpha == 'best,L=2':
                L = 2
            else:
                raise ValueError('unknown alpha argument: %s' % alpha)
            [alpha, beta, ok] = nufft_best_alpha(J1, L, K1 / float(N1))
            if not ok:
                warnings.warn('optimal alpha unknown for J=%d, K/N=%g, L=%d' %
                              (J1, K1 / float(N1), L))
                sn = np.ones(N1)
                err = np.nan
                return (err, sn, None)

    # if requested, return corresponding scaling factors too
    # alpha=np.squeeze(np.asarray(alpha))
    if isinstance(alpha, np.matrix):
        alpha = np.squeeze(np.asarray(alpha))
    if len(alpha) == N1:  # trick: special way to give explicit sn's
        sn = alpha
        if not use_qr:
            raise ValueError('give sn only allowed for QR version')
    elif nargout > 1 or use_qr:
        if isinstance(alpha, np.ndarray):
            alpha = alpha.squeeze()
        sn = nufft_scale(N1, K1, alpha, beta)
    else:
        sn = None

    # QR approach to error
    if use_qr:
        n = np.arange(0, N1).T - (N1 - 1) / 2.
        [nn, jj] = np.mgrid[n[0]:n[-1] + 1, 1:J1 + 1]
        gam1 = 2 * pi / K1

        C = np.exp(1j * gam1 * nn * jj) / np.sqrt(N1)
        S = scipy.sparse.spdiags(sn, diags=0, m=sn.size, n=sn.size)
        C = np.asmatrix(C)  # probably unnecessary
        A = S.H * C
        try:
            # [N,J] compact QR decomposition
            Q, R = np.linalg.qr(A, 'reduced')
        except ValueError:
            # numpy 1.7 or older
            Q, R = np.linalg.qr(A, 'full')

        do = (om - gam1 * _nufft_offset(om, J1, K1)).ravel()
        n = np.asmatrix(n)
        do = np.asmatrix(do)
        Db = np.exp(1j * n.T * do.conj()) / np.sqrt(N1)  # [N,M]
        err = np.asarray(Db - Q * (Q.conj().T * Db))
        err = np.lib.scimath.sqrt(np.sum(abs(err) ** 2, 0)).conj().T  # [M]
        return (err, sn, None)

    tol = 0
    T1 = _nufft_T(N1, J1, K1,
                  alpha=alpha,
                  tol=tol,
                  beta=beta,
                  use_true_diric=use_true_diric)  # [J,J]
    r1 = _nufft_r(om, N1, J1, K1,
                  alpha=alpha,
                  beta=beta,
                  use_true_diric=use_true_diric)[0]  # [J,M]

    # worst-case error at each frequency
    Tr1 = np.dot(T1, r1)        # [J,M]
    err = np.sum(r1.conj() * Tr1, 0).T  # [M,1]
    err[err.real > 1] = 1
    err = np.sqrt(1 - err)  # caution: this "1 -" may cause numerical error

    return (err, sn, T1)


def nufft2_err_mm(om, N1, N2, J1, J2, K1, K2, alpha=[1], beta=0.5):
    """  Compute worst-case error for each input frequency for min-max 2D NUFFT.

    Parameters
    ----------
    om : array_like
        [M,2] digital frequency omega in radians
    N1,N2 : int
        signal length
    J1,J2 : int
        # of neighbors used per frequency location
    K1,K2 :
        FFT size (should be > N1)
    alpha : array_like
        [L,1] Fourier series coefficients of scaling factors
    beta : float
        scale gamma=2pi/K by this in Fourier series
        typically is K/N (Fessler) or 0.5 (Liu)

    Returns
    -------
    err : array_like
        [M,1] worst-case error over unit-norm signals

    Notes
    -----
    Matlab vn. Copyright 2001-12-7, Jeff Fessler, The University of Michigan
    """

    pi = np.pi

    alpha1 = alpha
    alpha2 = alpha
    beta1 = beta
    beta2 = beta

    # trick to look at all relevant om's
    f1 = []
    if isinstance(om, str):
        if not om == 'all':
            raise ValueError('unknown om string: %s' % om)
        gam1 = 2 * pi / K1
        gam2 = 2 * pi / K2
        [f1, f2] = np.meshgrid(np.linspace(0, 1, 31), np.linspace(0, 1, 33),
                               indexing='ij')
        om = np.vstack((gam1 * f1.ravel(), gam2 * f2.ravel())).T

    # see if 'best' alpha is desired
    if isinstance(alpha, str):
        if not alpha == 'best':
            raise ValueError('unknown alpha string: %s' % alpha)
        [alpha1, beta1, ok] = nufft_best_alpha(J1, 0, K1 / N1)
        if not ok:
            raise ValueError('unknown J,K/N' % alpha)
        [alpha2, beta2, ok] = nufft_best_alpha(J2, 0, K2 / N2)
        if not ok:
            raise ValueError('unknown J,K/N' % alpha)

    tol = 0.

    T1 = _nufft_T(N1, J1, K1, alpha=alpha1, tol=tol, beta=beta1)  # [J,J]
    T2 = _nufft_T(N2, J2, K2, alpha=alpha2, tol=tol, beta=beta2)  # [J,J]
    r1 = _nufft_r(om[:, 0], N1, J1, K1, alpha=alpha1, beta=beta1)[0]  # [J,M]
    r2 = _nufft_r(om[:, 1], N2, J2, K2, alpha=alpha2, beta=beta2)[0]  # [J,M]
    T = np.kron(T2, T1)   # [J1*J2 x J1*J2]

    # kron for each om
    M = om.shape[0]
    r = np.zeros((J1 * J2, M))
    for ii in range(0, M):
        r[:, ii] = np.kron(r2[:, ii], r1[:, ii])

    # worst-case error at each frequency
    T = np.asmatrix(T)
    r = np.asmatrix(r)
    Tr = T * r          # [J,M]
    r = np.asarray(r)
    Tr = np.asarray(Tr)
    err = np.sum(r * Tr, 0).T  # [M,1]
    err = err.real  # TODO:  reale(err)
    err[err > 1] = 1
    err = np.sqrt(1 - err)

    if len(f1) > 0:  # isvar('f1')
        err = err.reshape(f1.shape)

    return err, f1, f2


def nufft1_error(om, N1, J1, K1, kernel, sn=[]):
    """ Compute worst-case error for each input frequency for 1D NUFFT
     using specified (inline) `ad hoc' kernel function (e.g., gaussian).
     This is worst-case for a unit-norm signal of length N1.

    Parameters
    ----------
    om : array_like
        [M,1] digital frequency omega in radians
    N1 : int
        signal length
    J1 : int
        # of neighbors used per frequency location
    K1 : int
        FFT size (should be > N1)
    kernel : function
        function taking args (k,J)
        (or choose from built-in examples - see code!)
    sn : array_like, optional
        optional scaling factors (otherwise do-no-harm)

    Returns
    -------
    err : array_like
        [M,1] worst-case error over unit-norm signals
    sn : array_like
        [N,1] scaling factors

    Notes
    ------
    examples for kernel:
      linear:       lambda k,J : (1 - abs(k/(J/2))) * (abs(k) < J/2)
      truncated diric:  lambda k,J : sinc(k) * (abs(k) < J/2)

    Matlab vn. Copyright 2001-12-7, Jeff Fessler, The University of Michigan
    """
    from pyir.nufft._minmax import nufft1_err_mm  # TODO: move?
    pi = np.pi

    # kernel selection
    if isinstance(kernel, str):
        # min-max interpolators (uniform, best, etc., see nufft1_err_mm.m)
        if kernel.startswith('minmax,'):
            alpha = kernel[7:]  # uniform or best or best,L=2 etc.
            [err, sn] = nufft1_err_mm(
                om, N1, J1, K1, stype='qr', alpha=alpha)[0:2]
            return err, sn, kernel

        # cos**3-tapered dirichlet
        elif kernel == 'cos3diric':
            kernel = cos3diric_kernel
        # Dirichlet (truncated)
        elif kernel == 'diric':
            kernel = get_diric_kernel(N1)

        # gaussian (truncated) with previously numerically-optimized width
        elif kernel == 'gauss':
            # if isvar('sn') & ischar(sn) & streq(sn, 'ft'):
            if sn == 'ft':
                stype = 'ft'
            else:
                stype = 'zn'
            [dummy, kernel, kernel_ft] = nufft_best_gauss(
                J1, K1 / float(N1), stype)

        # kaiser-bessel with previously numerically-optimized shape
        elif kernel == 'kaiser':
            K_N = K1 / float(N1)
            kb_a, kb_m = _kaiser_bessel_params('best', J=J1, K_N=K_N)
            kernel = functools.partial(
                kaiser_bessel, J=J1, alpha=kb_a, kb_m=kb_m, K_N=K_N)
            kernel_ft = functools.partial(
                kaiser_bessel_ft, J=J1, alpha=kb_a, kb_m=kb_m)

        # kaiser-bessel with previously numerically-optimized shape
        elif kernel == 'kb:beatty':
            K_N = K1 / N1
            # Eq. 5 for alpha
            alpha = \
                np.pi * np.sqrt(J1 ** 2 / K_N ** 2 * (K_N - 0.5) ** 2 - 0.8)
            kernel = functools.partial(kaiser_bessel,
                                       J=J1,
                                       alpha=alpha,
                                       kb_m=0)
            kernel_ft = functools.partial(kaiser_bessel_ft,
                                          J=J1,
                                          alpha=alpha,
                                          kb_m=0)
        # linear interpolation via triangular function (possibly "wide"!)
        elif kernel == 'linear':
            kernel = linear_kernel
        else:
            raise ValueError('unknown kernel: %s' % kernel)

        if not isinstance(kernel, collections.Callable):
            if is_string_like(kernel):
                kernel = eval('lambda k, J: ' + kernel)
            else:
                raise ValueError("kernel should be a string or callable")

    elif not isinstance(kernel, collections.Callable):
        raise ValueError('need callable kernel or recognized kernel string')

    gam = 2 * pi / K1

    # Compute scaling factors using the "do no harm" strategy.
    # This may not be optimal; analytical FT could also be reasonable.
    if len(sn) == 0:
        sn = 1 / _nufft_interp_zn([0, ], N1, J1, K1, kernel)  # [N]
        # sn = 1 ./ mean(_nufft_interp_zn([0 1/2], N1, J1, K1, kernel), 2) %
        # alt
    elif isinstance(sn, collections.Callable):
        Nmid = (N1 - 1) / 2.
        n = np.arange(0, N1).T - Nmid
        sn = 1 / sn(n / float(K1))      # [N]
    # trick to use Gaussian FT scaling factors
    # & isvar('kernel_ft') #TODO: isvar()
    elif is_string_like(sn) & (sn == 'ft'):
        Nmid = (N1 - 1) / 2.
        n = np.arange(0, N1).T - Nmid
        sn = 1 / kernel_ft(n / float(K1))       # [N]
        # if False and (J1 > 2):
        # sn_zn = 1 ./ _nufft_interp_zn(0, N1, J1, K1, kernel)  # [N]
        #    clf, plot(n, [sn reale(sn_zn)])
    # trick to use Kaiser-Bessel FT scaling factors
    # TODO:  dict or list?
    elif isinstance(sn, 'dict') & (sn['type'] == 'kaiser'):
        n = np.arange(0, N1).T - (N1 - 1) / 2.
        sn = 1 / kaiser_bessel_ft(n / float(K1), J1, sn.alpha, sn.m, 1)
    else:
        raise ValueError('unsupport scaling factors type')

    # interpolator worst-case error for each frequency (scaled by 1/sqrt(N)),
    # from equations (46)-(47) in Fessler&Sutton NUFFT paper, T-SP, Feb. 2003
    zn = _nufft_interp_zn(om / gam, N1, J1, K1, kernel)     # [N,M]
    sn = np.asmatrix(sn)
    if sn.shape[0] > sn.shape[1]:
        sn = sn.T
    S = scipy.sparse.spdiags(sn, 0, sn.size, sn.size)
    zn = np.asmatrix(zn)
    err = np.sqrt(np.mean(np.asarray(abs(S * zn - 1)) ** 2, 0)).conj().T  # [M]

    return (err, sn, kernel)


# TODO: check that this is correct.  should alpha be list or list of lists?
# not sure that all cases are functional here...
def nufft_scale(Nd, Kd, alpha, beta, Nmid=None, verbose=False):
    """  Compute Nd scaling factors for NUFFT

    Parameters
    ----------
    Nd : array_like of int
        signal dimensions
    Kd : array_like of int
        DFT dimensions
    alpha : list of lists
        array of alpha value arrays
    beta: list of float
        array of beta values
    Nmid : array_like of int, optional
        midpoint

    Returns
    -------
    sn : array_like
        [[Nd]] scaling factors

    Notes
    -----
    examples:

    N = 100; K = 2*N;
    sn = nufft_scale([N,N,N], [K,K,K], [[1],[1],[1]], [0.5, 0.5, 0.5])
    sn.shape  #(100,100,100)

    Matlab vn. Copyright 2004-7-8, Jeff Fessler, The University of Michigan
    """

    if isinstance(alpha, (int, float)):
        alpha = [alpha]  # convert scalar alpha to list

    Nd = np.atleast_1d(Nd)
    Kd = np.atleast_1d(Kd)
    dd = len(Nd)

    if verbose:
        print("type of alpha: {}".format(type(alpha)))
        print("alpha = {}".format(alpha))
        print("dd = {}".format(dd))

    if (Nmid is None):
        Nmid = (Nd - 1) / 2.
    Nmid = np.atleast_1d(Nmid)

    if dd != len(Kd):
        raise ValueError("length of Nd and Kd must match")
    if dd != len(Nmid):
        raise ValueError("length of Nd and Nmid must match")

    if (dd == 1) and (len(alpha) == 1):  # 1D case  # TODO: may be incorrect
        sn = _nufft_scale1(Nd[0], Kd, alpha, beta, Nmid)
    elif dd == 1:
        sn = _nufft_scale1(Nd[0], Kd, alpha, beta, Nmid)
    else:  # dd > 1
        # scaling factors: "outer product" of 1D vectors
        sn = np.array([1, ])
        for id in range(0, dd):
            tmp = _nufft_scale1(Nd[id], Kd[id], alpha[id], beta[id], Nmid[id])
            sn = sn.reshape((sn.size, 1))
            tmp = tmp.reshape((1, tmp.size))
            sn = sn * tmp
        sn = sn.reshape(tuple(Nd))

    return sn


def _nufft_scale1(N, K, alpha, beta, Nmid, verbose=False):
    """Compute scaling factors for 1D NUFFT (from Fourier series coefficients)

    Parameters
    ----------
        N : int
            signal size
        K : int
            DFT size
        alpha : float

        beta : float

        Nmid : int
            midpoint

     Returns
     -------
     sn : array_like
        [N] scaling factors (real 1D array)

    Matlab vn. Copyright 2004-7-8, Jeff Fessler, The University of Michigan

    """

    if not np.isrealobj(np.atleast_1d(alpha)[0]):
        raise ValueError('need real alpha_0')
    try:
        L = len(alpha) - 1
    except:
        L = 0

    # compute scaling factors from Fourier coefficients
    if L > 0:
        sn = np.zeros((1, N))
        n = np.arange(0, N)
        i_gam_n_n0 = 1.0j * (2 * np.pi / K) * (n - Nmid) * beta

        for l1 in range(-L, L + 1):
            alf = alpha[abs(l1)]
            if l1 < 0:
                alf = np.conj(alf)
            # print "l1 = %d" % l1
            # sn = sn.ravel() + alf * np.exp(i_gam_n_n0 * l1).ravel()
            sn = sn + alf * np.exp(i_gam_n_n0 * l1)
    else:
        sn = alpha * np.ones((N, 1))

    if verbose:
        print('range sn.real = %g,%g', np.min(sn.real), np.max(sn.real))
        print('range sn.imag = %g,%g', np.min(sn.imag), np.max(sn.imag))

    # TODO: reale.  #Note, not forced to real in the Matlab code
    sn = sn.real.ravel()

    return sn
