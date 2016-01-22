from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
import scipy.sparse

# from grl_utils import outer_sum
from PyIRT.nufft.kaiser_bessel import kaiser_bessel, kaiser_bessel_ft
from grl_utils import is_string_like
import collections

# TODO: add linear and diric options as in newfft.m
# newfft_scale_tri(Nd(id), Jd(id), Kd(id), st.Nmid)


def _nufft_samples(stype, Nd=None):
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

    pi = np.pi
    if stype == 'epi':  # blipped echo-planar cartesian samples
        if len(Nd) == 1:
            Nd = Nd[0]  # convert to int
            om = 2 * pi * np.arange(-Nd / 2., Nd / 2.) / float(Nd)
        elif len(Nd) == 2:
            o1 = 2 * pi * np.arange(-Nd[0] / 2., Nd[0] / 2.) / float(Nd[0])
            o2 = 2 * pi * np.arange(-Nd[1] / 2., Nd[1] / 2.) / float(Nd[1])
            o1 = np.tile(o1, (o2.shape[0], 1)).T
            o2 = np.tile(o2, (o1.shape[0], 1))  # [o1 o2] = ndgrid(o1, o2)
            # [o1,o2]=np.meshgrid(o2,o1)
            o1copy = o1.copy()
            # CANNOT DO THIS IN-PLACE, MUST USE THE COPY!!
            o1[:, 1::2] = np.flipud(o1copy[:, 1::2])

            om = np.zeros((o1.size, 2))
            om[:, 0] = o1.T.ravel()
            om[:, 1] = o2.T.ravel()
        else:
            raise ValueError('only 1d and 2d "epi" implemented')
    else:
        raise ValueError('unknown sampling type "%s"' % stype)
    return om


def _nufft_interp_zn(alist, N, J, K, func, Nmid=None):
    """ compute the "zn" terms for a conventional "shift-invariant" interpolator
        as described in T-SP paper.  needed for error analysis and for user-
        defined kernels since i don't provide a means to put in an analytical
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
    if not Nmid:
        Nmid = (N - 1) / 2.  # default: old version

    alist = np.atleast_1d(alist)
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
        jlist = np.arange(-J / 2. + 1, J / 2. + 1)
    else:  # odd
        jlist = np.arange(-(J - 1) / 2., (J - 1) / 2. + 1)
        alist[alist > 0.5] = 1 - alist[alist > 0.5]

    # n0 = (N-1)/2.;
    # nlist0 = np.arange(0,N) - n0;		# include effect of phase shift!
    n0 = np.arange(0, N) - Nmid
    nn0, jj = np.mgrid[n0[0]:n0[-1] + 1, jlist[0]:jlist[-1] + 1]
    # )*(1 + 1j) #NOTE:  must initialize zn as complex
    zn = np.zeros((N, len(alist)), dtype=np.complex64)

    for ia, alf in enumerate(alist):
        jarg = alf - jj			# [N,J]
        e = np.exp(1j * gam * jarg * nn0)		# [N,J]

        F = func(jarg, J)			# [N,J]
        # zn must be complex or result will be forced to real!
        zn[:, ia] = np.sum(F * e, 1)

    return zn


def _nufft_offset(om, J, K):
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
    om = np.asarray(om)
    gam = 2 * np.pi / K
    k0 = np.floor(om / gam - J / 2.)  # new way

    return k0


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
        f[i == False] = np.sign(np.cos(t[i == False] * (N - 1)))
    else:
        # sinc version
        f = np.sinc(k * N / K)

    return f


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
        [J,M]	r vector for each frequency
    arg : array_like
        [J,M]	dirac argument for t=0

    Notes
    -----
    Matlab vn. Copyright 2001-12-13, Jeff Fessler, The University of Michigan

    """
    alpha = np.atleast_1d(alpha)
    om = np.atleast_1d(om)

    M = len(om)

    gam = 2 * np.pi / K
    dk = om / gam - _nufft_offset(om, J, K)		# [M,1]

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
        [L+1]	Fourier coefficient vector for scaling
    tol : float, optional
        tolerance for smallest eigenvalue
    beta : float, optional
        scale gamma=2*pi/K by this for Fourier series
    use_true_diric : boolean, optional

    Returns
    -------
    T : array_like
        [J,J]	precomputed matrix

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
        L = len(alpha) - 1 	# L
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


def _nufft_coef(om, J, K, kernel):
    """  Make NUFFT interpolation coefficient vector given kernel function.

    Parameters
    ----------
    om : array_like
        [M,1]	digital frequency omega in radians
    J : int
        # of neighbors used per frequency location
    K : int
        FFT size (should be >= N, the signal_length)
    kernel : function
        kernel function

    Returns
    -------
    coef : array_like
        [J,M]	coef vector for each frequency
    arg : array_like
        [J,M]	kernel argument

    Notes
    -----
    Matlab vn. Copyright 2002-4-11, Jeff Fessler, The University of Michigan
    """

    om = np.squeeze(np.atleast_1d(om))
    if om.ndim > 1:
        raise ValueError("omega array must be 1D")
    # M = om.shape[0];
    gam = 2 * np.pi / K
    dk = om / gam - _nufft_offset(om, J, K)		# [M,1]

    # outer sum via broadcasting
    arg = -np.arange(1, J + 1)[:, None] + dk[None, :]  # [J,M]

    coef = kernel(arg, J)

    return (coef, arg)


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

    if (dd == 1) and (len(alpha) == 1):  # 1D case  #TODO: may be incorrect
        sn = _nufft_scale1(Nd, Kd, alpha, beta, Nmid)
    elif dd == 1:
        sn = _nufft_scale1(Nd, Kd, alpha, beta, Nmid)
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


def nufft_gauss(ktype='inline', J=6, sig=None):
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

    if isinstance(ktype, (np.ndarray, list, tuple)):
        if len(ktype) == 0:
            ktype = 'string'

    if sig is None:
        sig = 0.78 * np.sqrt(J)

    if not isinstance(ktype, str):
        raise ValueError('ktype must be a string')

    kernel = 'np.exp(-(k/%g)**2/2.) * (abs(k) < J/2.)' % sig
    kernel_ft = \
        '%g*np.sqrt(2*np.pi)*np.exp(-np.pi*(t*%g*np.sqrt(2*np.pi))**2)' % (
            sig, sig)

    if ktype == 'string':
        return kernel, kernel_ft
    elif ktype == 'inline':
        kernel = eval('lambda k,J: ' + kernel)
        kernel_ft = eval('lambda t: ' + kernel_ft)
    else:
        raise ValueError('ktype must be "inline" or "string"')

    return kernel, kernel_ft


def nufft_best_gauss(J, K_N=2, sn_type='ft'):
    """ Return "sigma" of best (truncated) gaussian for NUFFT
        with previously numerically-optimized width

    Parameters
    ----------
    J : int
        # of neighbors used per frequency location
    K_N : float, optional
        K/N grid oversampling ratio
    sn_type : {'zn', 'ft'}
        'ft' recommended

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

    # s = load('private/nufft_gauss2')
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
    # ii = find(J == s.Jgauss2)
    if np.sum(J == Jgauss2) != 1:
        print(Jgauss2)
        raise ValueError('only the J values listed above are done')

    if sn_type == 'ft':
        sig = Sgauss2['ft'][(J == Jgauss2).nonzero()[0]]
    elif sn_type == 'zn':
        sig = Sgauss2['zn'][(J == Jgauss2).nonzero()[0]]
    else:
        raise ValueError('bad sn_type %s' % sn_type)

    [kernel, kernel_ft] = nufft_gauss('string', J, sig)
    return sig, kernel, kernel_ft


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
    Matlab vn. Copyright 2001-12-17	Jeff Fessler. The University of Michigan
    """

    Jlist1 = np.array([6, ])		# list of which J's
    alpha1 = np.array([1, -0.46, 0.19])   # last colum is best beta  #J=6

    Jlist2 = np.arange(2, 11)		# list of which J's
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

    Jlist3 = np.array([4, 6])		# list of which J's
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
        Q, R = np.linalg.qr(A, 'full')  # [N,J] compact QR decomposition #TODO

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
    Tr1 = np.dot(T1, r1)		# [J,M]
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
    Tr = T * r			# [J,M]
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
      linear:		lambda k,J : (1 - abs(k/(J/2))) * (abs(k) < J/2)
      truncated diric:	lambda k,J : sinc(k) * (abs(k) < J/2)

    Matlab vn. Copyright 2001-12-7, Jeff Fessler, The University of Michigan
    """

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
            kernel = 'diric(2*np.pi*k/J, J) * cos((2*pi*k/J)/2.).^3'

        # Dirichlet (truncated)
        elif kernel == 'diric':
            kernel = 'nufft_diric(k,%d,%d,use_true_diric=True) * (abs(k) < J/2.)' % (
                N1, N1)

        # gaussian (truncated) with previously numerically-optimized width
        elif kernel == 'gauss':
            # if isvar('sn') & ischar(sn) & streq(sn, 'ft'):
            if sn == 'ft':
                stype = 'ft'
            else:
                stype = 'zn'
            [dummy, kernel, kernel_ft] = nufft_best_gauss(
                J1, K1 / float(N1), stype)
            kernel_ft = eval('lambda t: ' + kernel_ft)

        # kaiser-bessel with previously numerically-optimized shape
        elif kernel == 'kaiser':
            [kernel, kb_a, kb_m] = kaiser_bessel(
                'string', J1, 'best', 0, K1 / float(N1))
            kernel_ft = kaiser_bessel_ft('inline', J1, kb_a, kb_m, 1)

        # linear interpolation via triangular function (possibly "wide"!)
        elif kernel == 'linear':
            kernel = '(1 - abs(k/(J/2.))) * (abs(k) < J/2.)'

        else:
            raise ValueError('unknown kernel: %s' % kernel)

        if not isinstance(kernel, collections.Callable):  # convert string to kernel
            kernel = eval('lambda k,J: ' + kernel)
        # if not callable(kernel_ft): #convert string to kernel
        #    kernel_ft = eval('lambda t: ' + kernel_ft)

    elif not isinstance(kernel, collections.Callable):
        raise ValueError('need callable kernel or recognized kernel string')

    gam = 2 * pi / K1

    # if False:
    #    %	plot interpolator
    #    k = np.linspace(-J/2.-1,J/2.+1,101)
    #    figure(),subplot(221), plot(k, kernel(k, J))
    #    xlabel k, ylabel kernel(k), axis tight, grid

    # Compute scaling factors using the "do no harm" strategy.
    # This may not be optimal; analytical FT could also be reasonable.
    if len(sn) == 0:
        sn = 1 / _nufft_interp_zn([0, ], N1, J1, K1, kernel)  # [N]
        # sn = 1 ./ mean(_nufft_interp_zn([0 1/2], N1, J1, K1, kernel), 2) %
        # alt
    elif isinstance(sn, collections.Callable):
        Nmid = (N1 - 1) / 2.
        n = np.arange(0, N1).T - Nmid
        sn = 1 / sn(n / float(K1))		# [N]
    # trick to use Gaussian FT scaling factors
    # & isvar('kernel_ft') #TODO: isvar()
    elif is_string_like(sn) & (sn == 'ft'):
        Nmid = (N1 - 1) / 2.
        n = np.arange(0, N1).T - Nmid
        sn = 1 / kernel_ft(n / float(K1))		# [N]
        # if False and (J1 > 2):
        # sn_zn = 1 ./ _nufft_interp_zn(0, N1, J1, K1, kernel)	# [N]
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
    zn = _nufft_interp_zn(om / gam, N1, J1, K1, kernel)		# [N,M]
    sn = np.asmatrix(sn)
    if sn.shape[0] > sn.shape[1]:
        sn = sn.T
    S = scipy.sparse.spdiags(sn, 0, sn.size, sn.size)
    zn = np.asmatrix(zn)
    err = np.sqrt(np.mean(np.asarray(abs(S * zn - 1)) ** 2, 0)).conj().T  # [M]

    return (err, sn, kernel)


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
    Matlab vn. copyright 2002-7-16, Jeff Fessler, The University of Michigan
    """
    # if nargin < 3, help(mfilename), error(mfilename), end
    if L is None:
        if N > 40:
            L = 13		# empirically found to be reasonable
        else:
            L = np.ceil(
                N /
                3.)  # a kludge to avoid "rank deficient" complaints

    # kb_alf = 2.34 * J;	# KB shape parameter
    # kb_m = 0;		# KB order

    if (Nmid is None):
        Nmid = (N - 1) / 2.

    nlist = np.arange(N, dtype=float) - Nmid

    if False:  # old way
        [tmp, sn_kaiser] = nufft1_error(0, N, J, K, 'kaiser', 'ft')[0:2]
        # sn_kaiser = reale(sn_kaiser) #TODO: reale
        sn_kaiser = sn_kaiser.real
    else:
        # kaiser-bessel with previously numerically-optimized shape
        [kernel, kb_a, kb_m] = kaiser_bessel(
            'string', J, 'best', 0, K / float(N))
        kernel_ft = kaiser_bessel_ft('inline', J, kb_a, kb_m, 1)
        sn_kaiser = 1 / kernel_ft(nlist / float(K))  # [N]

    # use regression to match NUFFT with BEST kaiser scaling's
    gam = 2 * np.pi / K
    nlist = np.asmatrix(nlist).T
    ar = np.asmatrix(np.arange(0, L + 1))
    X = np.cos(beta * gam * nlist * ar)  # [N,L]
    # TODO:  how to do regress() in Python?
    # coef = (X \ sn_kaiser)';	% regress(sn_kaiser, X)';

    if sn_kaiser.ndim == 1:  # need 2D for matrix multiply below
        sn_kaiser = sn_kaiser[:, np.newaxis]
    # TODO: pinv(X)*sn is not always equivalent to Matlab's mldivide()
    coef = np.linalg.pinv(X) * sn_kaiser
    coef = np.squeeze(np.asarray(coef))

    alphas = np.zeros(coef.shape)
    alphas[0] = coef[0].real
    alphas[1:] = coef[1:] / 2.

    return alphas, beta

    if verbose:
        print(
            'condition # for LS fit to KB scale factors: %g' %
            np.linalg.cond(X))
