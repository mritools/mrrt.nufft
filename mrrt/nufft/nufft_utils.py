import sys
import warnings

import numpy as np

from . import config

if config.have_cupy:
    import cupy


__all__ = [
    "get_array_module",
    "profile",
    "reale",
    "is_string_like",
    "get_data_address",
    "complexify",
    "outer_sum",
    "max_percent_diff",
]


"""
@profile decorator that does nothing when line profiler is not active

see:
http://stackoverflow.com/questions/18229628/python-profiling-using-line-profiler-clever-way-to-remove-profile-statements
"""
try:
    if sys.version_info[0] >= 3:
        import builtins

        profile = builtins.__dict__["profile"]
    else:
        import __builtin__

        profile = __builtin__.profile
except (AttributeError, KeyError):
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


def is_string_like(obj):
    """Check if obj is string."""
    try:
        obj + ""
    except (TypeError, ValueError):
        return False
    return True


def get_array_module(arr, xp=None):
    """ Check if the array is a cupy GPU array and return the array module.

    Paramters
    ---------
    arr : numpy.ndarray or cupy.core.core.ndarray
        The array to check.

    Returns
    -------
    array_module : python module
        This will be cupy when on_gpu is True and numpy otherwise.
    on_gpu : bool
        Boolean indicating whether the array is on the GPU.
    """
    if xp is None:
        if config.have_cupy:
            xp = cupy.get_array_module(arr)
            return xp, (xp != np)
        else:
            return np, False
    else:
        return xp, (xp != np)


def get_data_address(x):
    """Returns memory address where a numpy or cupy array's data is stored."""
    if hasattr(x, "__array_interface__"):
        ptr_x = x.__array_interface__["data"][0]
    elif hasattr(x, "__cuda_array_interface__"):
        ptr_x = x.__cuda_array_interface__["data"][0]
    else:
        raise ValueError(
            "Input must have an __array_interface__ or "
            "__cuda_array_interface__ attribute."
        )
    return ptr_x


def complexify(x, complex_dtype=None, xp=None):
    """Promote to complex if real input was provided.

    Parameters
    ----------
    x : array-like
        The array to convert
    complex_dtype : np.complex64, np.complex128 or None
        The dtype to use.  If None, the dtype used will be the one returned by
        ``np.result_tupe(x.dtype, np.complex64)``.

    Returns
    -------
    xc : array-like
        Complex-valued x.
    """
    xp, on_gpu = get_array_module(x, xp)
    x = xp.asarray(x)

    if xp.iscomplexobj(x) and (x.dtype == complex_dtype):
        return x

    if complex_dtype is None:
        # determine complex datatype to use based on numpy's promotion rules
        complex_dtype = xp.result_type(x.dtype, xp.complex64)

    return xp.asarray(x, dtype=complex_dtype)


def outer_sum(
    xx=None, yy=None, conjugate_y=False, squeeze_output=True, xp=None
):
    """ outer_sum xx + yy.

    Parameters
    ----------
    xx : xp.ndarray
        n-dimensional array. will have a new axis appended.
    yy : xp.ndarray
        1d array.  will have xx.ndim new axes prepended
    conjugate_y : bool, optional
        If true, use the complex conjugate of yy.
    squeeze_output : bool, optional
        If true, squeeze any singleton dimensions in the output.

    Returns
    -------
    ss : xp.ndarray
        outer sum  (produced via numpy broadcating).
        ``ss.shape = xx.shape + (yy.size, )``
    """
    if xp is None:
        xp, on_gpu = get_array_module(xx)
    xx = xp.atleast_1d(xx)
    yy = xp.atleast_1d(yy)
    if conjugate_y:
        yy = yy.conj()
    if yy.ndim != 1:
        raise ValueError("yy must be a 1D vector")
    for d in range(xx.ndim):
        yy = yy[xp.newaxis, ...]
    ss = xx[..., xp.newaxis] + yy
    if squeeze_output:
        ss = xp.squeeze(ss)
    return ss


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
    if stype == "epi":  # blipped echo-planar cartesian samples
        if len(Nd) == 1:
            Nd = Nd[0]  # convert to int
            om = (twopi / float(Nd)) * xp.arange(-Nd / 2.0, Nd / 2.0)
        elif len(Nd) == 2:
            o1 = (twopi / float(Nd[0])) * xp.arange(-Nd[0] / 2.0, Nd[0] / 2.0)
            o2 = (twopi / float(Nd[1])) * xp.arange(-Nd[1] / 2.0, Nd[1] / 2.0)
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
        n_mid = (N - 1) / 2.0  # default: old version

    xp, on_gpu = get_array_module(alist, xp)
    alist = xp.atleast_1d(alist)
    #
    # zn = \sum_{j=-J/2}^{J/2-1} exp(i gam (alf - j) * n) F1(alf - j)
    #    = \sum_{j=-J/2}^{J/2-1} exp(i gam (alf - j) * (n-n0)) F0(alf - j)
    #

    gam = 2 * np.pi / float(K)

    if (alist.min() < 0) or (alist.max() > 1):
        warnings.warn("value in alist exceeds [0,1]")

    # natural phase function. trick: force it to be 2pi periodic
    # Pfunc = inline('exp(-i * mod0(om,2*pi) * (N-1)/2)', 'om', 'N')

    if not np.remainder(J, 2):  # even
        jlist = xp.arange(-J / 2.0 + 1, J / 2.0 + 1)
    else:  # odd
        jlist = xp.arange(-(J - 1) / 2.0, (J - 1) / 2.0 + 1)
        alist[alist > 0.5] = 1 - alist[alist > 0.5]

    # n0 = (N-1)/2.;
    # nlist0 = np.arange(0,N) - n0;     # include effect of phase shift!
    n0 = xp.arange(0, N) - n_mid

    nn0, jj = xp.ogrid[n0[0] : n0[-1] + 1, jlist[0] : jlist[-1] + 1]

    # must initialize zn as complex
    zn = xp.zeros((N, len(alist)), dtype=np.complex64)

    for ia, alf in enumerate(alist):
        jarg = alf - jj  # [N,J]
        e = xp.exp(1j * gam * jarg * nn0)  # [N,J]
        # TODO: remove need for this try/except
        try:
            F = func(jarg, J)  # [N,J]
        except:
            F = func(jarg)  # [N,J]
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
    k0 = xp.floor(om / gam - J / 2.0)  # new way

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
    dk = om / gam - _nufft_offset(om, J, K, xp=xp)  # [M,1]

    # outer sum via broadcasting
    arg = -xp.arange(1, J + 1)[:, None] + dk[None, :]  # [J,M]
    try:
        # try calling kernel without J in case it is baked into the kernel
        coef = kernel(arg)
    except TypeError:
        # otherwise, provide J to the kernel
        coef = kernel(arg, J=J)

    return (coef, arg)


def _as_1d_ints(arr, n=None, dtype_out=np.intp, xp=None):
    """Convert to a 1D array with dtype=dtype_out

    Returns an error if the elements of ``arr`` aren't an integer type or
    ``arr`` has more than one non-singleton dimension.

    If specified, an error is raised if the array doesn't contain ``n``
    elements.
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
                arr = xp.asarray([arr[0]] * n)
            else:
                raise ValueError(
                    "array did not have the expected size of {}".format(n)
                )

    return arr.astype(dtype_out)


def reale(x, com="error", tol=None, msg=None, xp=None):
    """Return real part of complex data (with error checking).

    Parameters
    ----------
    x : array-like
        The data to check.
    com : {'warn', 'error', 'display', 'report'}
        Control rather to raise a warning, an error, or to just display to the
        console.  If ``com == 'report'``, the relative magnitude of the
        imaginary component is printed to the console.
    tol : float or None
        Allow complex values below ``tol`` in magnitude.  If None, ``tol`` will
        be ``1000*eps``.
    msg : str or None
        Additional message to print upon encountering complex values.

    Notes
    -----
    based on Matlab routine by Jeff Fessler, University of Michigan
    """
    xp, on_gpu = get_array_module(x, xp)
    if not xp.iscomplexobj(x):
        return x

    if tol is None:
        tol = 1000 * xp.finfo(x.dtype).eps

    if com not in ["warn", "error", "display", "report"]:
        raise ValueError(
            (
                "Bad com: {}.  It must be one of {'warn', 'error', 'display', "
                "'report'}"
            ).format(com)
        )

    max_abs_x = xp.max(xp.abs(x))
    if max_abs_x == 0:
        if xp.any(xp.abs(xp.imag(x)) > 0):
            raise RuntimeError("max real 0, but imaginary!")
        else:
            return xp.real(x)

    frac = xp.max(xp.abs(x.imag)) / max_abs_x
    if com == "report":
        print("imaginary part %g%%" % frac * 100)

    if frac > tol:
        t = "imaginary fraction of x is %g (tol=%g)" % (frac, tol)
        if msg is not None:
            t += "\n" + msg
        if com == "display":
            print(t)
        elif com == "warn":
            warnings.warn(t)
        else:
            raise RuntimeError(t)

    return xp.real(x)


def max_percent_diff(s1, s2, use_both=False, doprint=False, xp=None):
    """Maximum percent difference between two signals.

    Parameters
    ----------
    s1, s2 : array-like
        The two signals to compare. These should have the same shape.
    use_both: bool, optional
        If True use the maximum across ``s1``, ``s2`` as the normalizer.
        Otherwise the maximum across ``s1`` is used.

    Returns
    -------
    d : float
        The maximum percent difference (in range [0, 100]).

    Notes
    -----
    Based on Matlab function of the same name
        Copyright 2000-9-16, Jeff Fessler, The University of Michigan
    Python port by Gregory Lee.
    """
    xp, on_gpu = get_array_module(s1, xp=xp)
    s1 = xp.squeeze(xp.asarray(s1))
    s2 = xp.squeeze(xp.asarray(s2))

    # first check that we have comparable signals!
    if s1.shape != s2.shape:
        raise ValueError("size mismatch")

    if xp.any(xp.isnan(s1)) | xp.any(xp.isnan(s2)):
        raise ValueError("NaN values found in input")

    if use_both:
        denom = xp.max(xp.abs(s1).max(), xp.abs(s2).max())
        if denom == 0:
            return 0
    else:
        denom = xp.abs(s1).max()
        if denom == 0:
            denom = xp.abs(s2).max()
        if denom == 0:
            return 0
    d = xp.max(xp.abs(s1 - s2)) / denom
    return d * 100
