from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
from pyir.utils import fft, ifft, fft2, ifft2, fftn, ifftn, complexify

from pyir.operators import LinearOperatorMulti, DiagonalOperator


def dsft_gram_hermitify(kern, show=False, verbose=False):
    """ Fixup Hermitian symmetry for an NUFFT kernel.

    When A is a Gdsft object and W is a Hermitian positive semidefinite matrix,
    e.g., when W is diagonal with real, nonnegative diagonal elements,
    the corresponding gram matrix A'WA is positive semidefinite and Toeplitz,
    and the corresponding kernel h[n] is Hermitian symmetric,
    i.e., h[-n] = np.conj(h[n]).

    This routine takes an input kernel that might be not quite Hermitian
    and uses averaging to force it to be Hermitian.
    It is needed because of NUFFT approximation error.

    Parameters
    ----------
    kern : ndarray
        1D, 2D, or 3D kernel or gram operator
    show : bool, optional
        make plots or images of the kernel

    Returns
    -------
    kern : ndarray
        kernel with Hermitian symmetry enforced

    Notes
    -----
    matlab version of this routine:
        Copyright 2012-06-06, Jeff Fessler, University of Michigan
    """
    kern = np.asarray(kern)

    # handle potential 2D shape in 1D case by temporarily removing size 1 axes
    shape_orig = kern.shape
    kern = np.squeeze(kern)

    # general n-dimensional case
    axes_n = tuple([np.arange(s) for s in kern.shape])
    axes_n_grid = np.meshgrid(*axes_n, indexing='ij', sparse=True)
    tmp1 = kern[axes_n_grid]
    axes_m = tuple([np.mod(-n, len(n)) for n in axes_n])
    axes_m_grid = np.meshgrid(*axes_m, indexing='ij', sparse=True)
    tmp2 = kern[axes_m_grid]
    np.conj(tmp2, out=tmp2)

    # force it to be Hermitian
    kern = _dsft_gram_hermitify_avg(tmp1, tmp2, verbose=verbose)

    if kern.shape != shape_orig:
        kern = kern.reshape(shape_orig)

    if show:
        if kern.ndim > 2:
            raise ValueError("show only currently supported for 1D, 2D")
        _dsft_gram_hermitify_show(tmp1, tmp2)

    return kern


def _dsft_gram_hermitify_avg(in1, in2, verbose=False):
    out = in1
    out += in2
    out *= 0.5
    if verbose:
        print('biggest relative change to make Hermitian: {}'.format(
            np.max(np.abs(out - in1)) / np.max(np.abs(in1))))
    return out


def _dsft_gram_hermitify_plot1(ax, data):
    n = numel(data)
    ax.plot(np.arange(-n/2, n/2), np.fft.fftshift(data), '-o')
    ax.set_xticks([-n/2, 0, n/2-1])
    return ax


def _dsft_gram_hermitify_show(tmp1, tmp2):
    from matplotlib import pyplot as plt
    if tmp1.ndim == 1:
        fun = _dsft_gram_hermitify_plot1
    else:
        def nd_fun(ax, d):
            ax.imshow(np.fft.fftshift(d),
                      interpolation='nearest',
                      cmap=plt.cm.gray)
        fun = nd_fun

    err = tmp2 - tmp1  # should be 0
    fig, axes = plt.subplots(3, 3)
    axes = axes.ravel()
    fun(axes[0], np.real(tmp1))
    fun(axes[1], np.real(tmp2))
    fun(axes[2], np.real(err))
    fun(axes[3], np.imag(tmp1))
    fun(axes[4], np.imag(tmp2))
    fun(axes[5], np.imag(err))
    fun(axes[6], np.abs(tmp1))
    fun(axes[7], np.abs(tmp2))
    fun(axes[8], np.abs(err))


def _test_dsft_gram_hermitify():
    # compare to result from Matlab implementation
    import scipy.io
    from numpy.testing import assert_allclose
    d = scipy.io.loadmat('kern2d.mat')
    kern_mat = d['kern']
    kernH_mat = d['kernH']
    kernH = dsft_gram_hermitify(kern_mat, show=False, verbose=False)
    assert_allclose(kernH, kernH_mat)


class ToeplitzOperator(LinearOperatorMulti):
    """A linear operator for Toeplitz NUFFT."""

    def __init__(self, mask, fftkern, nargin=None, nargout=None,
                 shape=None, **kwargs):
        if 'symmetric' in kwargs:
            kwargs.pop('symmetric')
        if 'hermitian' in kwargs:
            kwargs.pop('hermitian')
        if 'matvec' in kwargs:
            kwargs.pop('matvec')

        self.fftkern = fftkern
        self.mask = mask
        if mask is not None:
            if np.any(np.asarray(fftkern.shape) != 2*np.asarray(mask.shape)):
                raise ValueError("shape mismatch")
        squeeze_reps_in = kwargs.get('squeeze_reps_out', True)
        squeeze_reps_out = kwargs.get('squeeze_reps_out', True)

        if mask is not None:
            nargout = nargin = mask.sum()
        else:
            if nargin is None:
                raise ValueError("must provide nargin or mask")
            nargout = nargin
            self.mask = np.ones(np.asarray(fftkern.shape)//2, dtype=np.bool)

        shape_in = (nargin, )
        shape_out = (nargout, )

        super(ToeplitzOperator, self).__init__(
            nargin,
            nargout,
            shape_in=shape_in,
            shape_out=shape_out,
            symmetric=False,
            hermitian=True,
            matvec=self.forward,
            matvec_adj=self.forward,   # Hermitian symmetry
            mask_out=None,  # matvec() is handling the masking
            mask_in=None,  # matvec() is handling the masking
            nd_input=False,
            nd_output=False,
            squeeze_reps_in=squeeze_reps_in,
            squeeze_reps_out=squeeze_reps_out,
            **kwargs)

    def forward(self, x):
        # if (self.mask is not None) and (not self.nd_input):
        if self.mask is not None:
            x = embed(x, self.mask, order=self.order)

        y = Gnufft_gram_mult(x, self.fftkern)

        # if (self.mask is not None) and (not self.nd_output):
        if self.mask is not None:
            y = masker(y, self.mask, order=self.order)
        return y


def Gnufft_gram(G, W, reuse={}):
    """
    #|def [T, reuse] = Gnufft_gram(A, W, reuse)
    #|
    #| construct Toeplitz gram matrix object T = A'WA for a Gnufft object
    #| in
    #|  A   [M np]      Gnufft object (Fatrix or fatrix2)
    #|  W   [M M]       W = diag_sp(wi) (often simply "1" or [])
    #|              W = Gdiag() for fatrix2
    #|  reuse   struct      stuff from the last call that can be reused
    #|
    #| out
    #|  T   [np np]     fatrix2 or Fatrix object
    #|  reuse   struct      stuff that can be reused on future calls
    #|
    #| Copyright 2004-6-29, Jeff Fessler & Hugo Shi, University of Michigan
    """
    nk, npix = G.shape
    if W is None:
        W = np.ones(nk)  # the usual unweighted case
    elif np.isscalar(W):
        W = np.full(nk, float(W))  # the usual unweighted case

    if isinstance(W, DiagonalOperator):
        W = W.diag.diagonal()

    if not np.isrealobj(W):
        raise ValueError("not implemented for complex W.  see old version in"
                         "arch/ subfolder of irt Matlab package")

    fftkern, reuse = Gnufft_gram_init(G, W, reuse, verbose=False)

    if np.any(~G.mask):
        omask = G.mask.copy()  # need copy?
    else:
        omask = None

    T = ToeplitzOperator(mask=omask,
                         nargin=G.nargin,
                         fftkern=fftkern)
    return (T, reuse)


def Gnufft_gram_init(G, wi, reuse={}, show=False, verbose=False):
    """
    Construct kernel of circulant matrix into which T is embedded and take its
    DFT to prepare for multiplication.

    Parameters
    ----------
    TODO

    Returns
    -------
    fftkern : ndarray
        FFT of kernel of circulant matrix  (shape = 2*Nd)
    """
    ndim = G.ndim
    if ndim == 1:
        func = Gnufft_gram_init1
    elif ndim == 2:
        func = Gnufft_gram_init2
    elif ndim == 3:
        func = Gnufft_gram_init3
    else:
        raise ValueError(">3D not implemented")
    fftkern, reuse = func(G, wi=wi, reuse=reuse, show=show, verbose=verbose)
    return fftkern, reuse


# Gnufft_gram_init1()
# 1d filter for circulant matrix
# note: only toeplitz kernel values from -(N-1) to (N-1) are relevant
# so the value at +/- N does not matter so we set it to zero.
def Gnufft_gram_init1(G, wi, reuse={}, show=False, verbose=False):
    if 'G1' not in reuse:
        reuse['G1'] = Gnufft_gram_setup(G)

    block1 = reuse['G1'].H * complexify(wi.real)  # kludge

    # kernel of Toeplitz matrix from -N to N-1 but with fftshift
    # this is inherently Hermitian symmetric except for the middle [0] value
    # use 0 for the irrelevant value at -N
    err1 = np.abs(np.imag(block1[0])) / np.abs(block1[0])
    tol = 0
    if err1 > tol:
        print('removing imaginary h[0] part of relative size {}'.format(err1))
        block1[0] = block1[0].real
    # [2*N1]
    kern = np.concatenate((block1, [0, ], np.conj(block1[1:])[::-1]), axis=0)

    # force Hermitian symmetry:
    #     This fixes slight asymmetry due to finite NUFFT precision
    kern = dsft_gram_hermitify(kern, show, verbose=verbose)

    fftkern = fft(kern)  # [2*N1]
    return fftkern, reuse


# Gnufft_gram_init2()
def Gnufft_gram_init2(G, wi, reuse={}, show=False, verbose=False):
    if 'G2' not in reuse:
        reuse['G1'], reuse['G2'] = Gnufft_gram_setup(G)

    Nd = G.Nd
    N1, N2 = Nd

    # kludge
    block1 = np.reshape(reuse['G1'].H * complexify(wi.real), Nd, order=G.order)
    block2 = np.reshape(reuse['G2'].H * complexify(wi.real), Nd, order=G.order)

    # build the Hermitian NUFFT kernel (size 2*Nd)
    z1 = np.zeros((N1, 1))
    z2 = np.zeros((N1-1, 1))
    ktop = np.hstack(
        (block1,
         z1,
         np.conj(np.fliplr(np.vstack((block1[0:1, 1:], block2[1:, 1:]))))
         ))
    kmid = np.zeros((1, 2*N2))
    kbot = np.hstack(
        (np.flipud(block2[1:, :]),
         z2,
         np.fliplr(np.flipud(np.conj(block1[1:, 1:])))
         ))
    kern = np.vstack((ktop, kmid, kbot))  # [(2Nd)]

    # force Hermitian symmetry:
    #     This fixes slight asymmetry due to finite NUFFT precision
    kern = dsft_gram_hermitify(kern, show, verbose=verbose)

    fftkern = fftn(kern)
    return fftkern, reuse


def Gnufft_gram_init3(G, wi, reuse={}, show=False, verbose=False):
    if 'G4' not in reuse:
        reuse['G1'], reuse['G2'], reuse['G3'], reuse['G4'] = \
            Gnufft_gram_setup(G)

    Nd = G.Nd
    N1, N2, N3 = Nd

    # build the Hermitian NUFFT kernel (size 2*Nd)
    filtblk1 = np.reshape(reuse['G1'].H * wi.real, Nd, order=G.order)
    filtblk2 = np.reshape(reuse['G2'].H * wi.real, Nd, order=G.order)
    filtblk3 = np.reshape(reuse['G3'].H * wi.real, Nd, order=G.order)
    filtblk4 = np.reshape(reuse['G4'].H * wi.real, Nd, order=G.order)

    tblk1 = filtblk1
    tblk2 = filtblk2[1:, :, :]   # remove the duplicated part with filtblk1
    tblk3 = filtblk3[1:, 1:, :]  # remove the duplicated part with block 1,2,4
    tblk4 = filtblk4[:, 1:, :]   # remove the duplicated part with block 1

    z1 = np.zeros((N1, 1, N3))
    z2 = np.zeros((N1-1, 1, N3))

    # top half of the 3D filter
    kern_top = np.vstack(
        (hstack((tblk1, z1. tblk4[:, ::-1, :])),  # Upper block
         np.zeros((1, 2*N2, N3)),  # Zero padding in the middle
         hstack((tblk2[::-1, :, :], z2, tblk3[::-1, ::-1, :]))  # lower block
         ))

    # construct the bottom half now
    bblk1 = np.conj(filtblk3[:, :, 1:])[:, :, ::-1]
    bblk2 = np.conj(filtblk4[:, :, 1:])[:, :, ::-1]
    bblk2 = bblk2[1:, :, :]
    bblk3 = np.conj(filtblk1[:, :, 1:])[:, :, ::-1]
    bblk3 = bblk3[1:, 1:, :]
    bblk4 = np.conj(filtblk2[:, :, 1:])[:, :, ::-1]
    bblk4 = bblk4[:, 1:, :]

    z4 = np.zeros((N1, 1, N3-1))
    z5 = np.zeros((N1-1, 1, N3-1))
    kern_bottom = np.vstack(
        (np.hstack((bblk1, z4, bblk4[:, ::-1, :])),
         np.zeros((1, 2*N2, N3-1)),
         np.hstack((bblk2[::-1, :, :], z5, bblk3[::-1, ::-1, :]))
         ))

    kern = np.concatenate((kern_top,
                           np.zeros((2*N1, 2*N2, 1)),
                           kern_bottom), axis=2)

    # force Hermitian symmetry:
    #     This fixes slight asymmetry due to finite NUFFT precision
    kern = dsft_gram_hermitify(kern, show, verbose=verbose)

    fftkern = fftn(kern)
    return fftkern, reuse


def Gnufft_gram_setup(G):
    """modified versions of G (with full mask and no phase shifts)

    TODO:
    copy G and change mask to true and shifts to 0 instead of creating new
    operators.  Probably not a big deal as creation of the table-based
    operators used here is relatively fast.
    """
    if not isinstance(G, NUFFT_Operator):
        raise ValueError("NUFFT_Operator required")

    nufft_kwargs = dict(mask=np.ones_like(G.mask),
                        phasing='complex',  # required?
                        mode='table1',
                        n_shift=(0, ) * G.ndim,
                        Jd=G.Jd,
                        Kd=G.Kd,
                        Nd=G.Nd,
                        Ld=G.Ld,
                        kernel_type=G.kernel_type,
                        order=G.order)

    om = G.om
    G1 = NUFFT_Operator(om=om, **nufft_kwargs)

    if G.ndim > 3:
        raise ValueError("only NUFFT up to 3D done")

    if G.ndim == 1:
        return G1

    om2 = om.copy()
    om2[:, 0] = -om2[:, 0]  # negative kx relative to om
    G2 = NUFFT_Operator(om=om2, **nufft_kwargs)

    if G.ndim == 2:
        return G1, G2

    om3 = om2.copy()
    om3[:, 1] = -om3[:, 1]  # negative kx & ky relative to om
    G3 = NUFFT_Operator(om=om3, **nufft_kwargs)

    om4 = om3.copy()
    om4[:, 0] = -om4[:, 0]  # negative ky relative to om
    G4 = NUFFT_Operator(om=om4, **nufft_kwargs)

    # 3D case
    return G1, G2, G3, G4

if False:
    import numpy as np
    import scipy.io
    from pyir.operators_private import NUFFT_Operator
    from pyir.utils import embed, masker, complexify
    d = scipy.io.loadmat('st_attributes.mat')
    wi = np.squeeze(scipy.io.loadmat('wi.mat')['wi'])
    mask = np.squeeze(scipy.io.loadmat('mask.mat')['mask']).astype(np.bool)

    Nd = np.squeeze(d['Nd'])
    Kd = np.squeeze(d['Kd'])
    Ld = np.squeeze(d['Ld'])
    om = np.squeeze(d['om'])
    n_shift = np.squeeze(d['n_shift'])
    phase_shift = d['phase_shift']
    G = NUFFT_Operator(om=om,
                       Nd=Nd,
                       Kd=Kd,
                       Ld=Ld,
                       n_shift=n_shift,
                       mask=mask,
                       mode='table1')

    # wi = np.ones(np.prod(Nd))
    wi = complexify(wi)
    embed(G.H*wi, mask)

    G1, G2 = Gnufft_gram_setup(G)
    fftkern, reuse = Gnufft_gram_init2(G, wi, reuse={}, show=False,
                                       verbose=False)
    fftkern_mat = scipy.io.loadmat('fftkern.mat')['fftkern']

    # TODO: add this test for LinOp with mask_in, mask_out


# Gnufft_gram_mult()
# multiply an image x by a toeplitz matrix
# by embedding it into a circulant matrix of twice the size and using FFT.
# in
#   x   [*Nd 1] or [np 1]
#   fftkern [[2Nd]]
#   mask    [[Nd]]
# out
#   y   [*Nd 1] or [np 1]
def Gnufft_gram_mult(x, fftkern):
    fftkern = np.asarray(fftkern)
    N2 = fftkern.shape
    ndim = fftkern.ndim
    Nd = tuple(np.asarray(N2) // 2)

    if x.ndim > fftkern.ndim:
        LL = x.shape[-1]
    else:
        LL = 1
        x = x[..., np.newaxis]

    y = np.zeros(Nd + (LL, ))
    if ndim == 1:
        tmp = fft(x, N2[0], axis=0)  # [N2 L]
        tmp = tmp * fftkern[:, np.newaxis]
        y = ifft(tmp, axis=0)
        y = y[:Nd[0], :]  # [Nd L]
    elif ndim == 2:
        tmp = fftkern[..., np.newaxis] * fft2(x, N2, axes=(0, 1))
        tmp = ifft2(tmp, axes=(0, 1))
        y = tmp[:Nd[0], :Nd[1], :]
    elif ndim == 3:
        tmp = fftkern[..., np.newaxis] * fftn(x, N2, axes=(0, 1, 2))
        tmp = ifft2(tmp, axes=(0, 1, 2))
        y = tmp[:Nd[0], :Nd[1], :Nd[2], :]
    else:
        raise ValueError(">3D not implemented")
    return np.squeeze(y)



def assert_adjoint(A, nrep=1, tol=1e-5, do_complex=False, warn=False,
                   verbose=False):
    """Verifies that an operator A is adjoint by multiplying with random
    vectors.

    Set do_complex=True to test for Hermitian symmetry.
    """
    from pyir.utils import max_percent_diff
    rstate = np.random.RandomState(0)
    for ii in range(nrep):
        x = rstate.random_sample((A.shape[1], )) - 0.5
        y = rstate.random_sample((A.shape[0], )) - 0.5
        if do_complex:
            x = x + 1j * rstate.random_sample((A.shape[1], )) - 0.5
            y = y + 1j * rstate.random_sample((A.shape[0], )) - 0.5
        A.squeeze_reps_in = True
        A.squeeze_reps_out = True
        Ax = A * x
        if not np.isrealobj(Ax) and not do_complex:
            raise ValueError(
                'must test complex systems with "complex" option')

        v1 = np.sum(np.conj(y) * Ax)
        v2 = np.conj(np.sum(np.conj(x) * (A.H * y)))

        # expect v1 = v2 if symmetric (Hermetian)
        mpd = max_percent_diff(v1, v2)
        if mpd/100 > tol:
            print("{} {}".format(mpd/100, tol))
            if warn:
                warnings.warn('adjoint mismatch')
            else:
                raise ValueError('adjoint mismatch')
        elif verbose:
            print("v1={}".format(v1))
            print("v2={}".format(v2))
        return


def test_Gnufft_gram_1d():
    from numpy.testing import assert_allclose
    N = 16
    J = 6
    K = 2*N
    rstate = np.random.RandomState(0)
    M = 51
    omega = 2*np.pi*rstate.random_sample((M, 1))
    nufft_args = dict(Nd=N, Jd=J, Kd=K, n_shift=N/2, mode='table1', Ld=2**11,
                      kernel='minmax:kb')
    wi = np.arange(omega.shape[0])
    mask = np.ones((N, ), dtype=np.bool)
    mask[-3:] = 0
    A = NUFFT_Operator(om=omega, mask=mask, **nufft_args)
    T, reuse = Gnufft_gram(A, wi)
    # T = A.build_gram(wi)  # TODO

    assert_adjoint(T, do_complex=True)

    x = rstate.random_sample((A.shape[1], )) - 0.5
    x = x + 1j * rstate.random_sample((A.shape[1], )) - 0.5

    r1 = A.H * (DiagonalOperator(wi) * (A * x))
    r2 = T * x

    assert_allclose(r1, r2, rtol=1e-4)


# Gnufft_gram_test2()
# test this object and compare its speed to Gnufft approach
def test_Gnufft_gram_2d():
    from numpy.testing import assert_allclose
    from pyir.utils import ImageGeometry, ellipse_im
    from pyir.mri import mri_trajectory

    N = (16, 14)
    fov = N
    J = (6, 5)
    K = 2*np.asarray(N)
    ktype = 'spiral0'
    # 'voronoi')
    (kspace, omega, wi) = mri_trajectory(ktype, N=N, fov=fov, arg_wi=None)
    ig = ImageGeometry(nx=N[0], ny=N[1], dx=1)
    mask, junk = ellipse_im(ig, [0, 0, 14, 15, 0, 1], oversample=3)
    mask = mask > 0
    x, params = ellipse_im(ig, 'shepplogan-emis')  # , oversample=2, fov=250)
    x = complexify(x)

    nufft_args = dict(Nd=N, Jd=J, Kd=K, n_shift=np.asarray(N)/2, mode='table1',
                      Ld=2**12, kernel='kb:beatty')
    A = NUFFT_Operator(om=omega, mask=mask, **nufft_args)
    T, reuse = Gnufft_gram(A, wi)
    # T = A.build_gram(wi)  # TODO

    assert_adjoint(T, do_complex=True)

    r1 = A.H * (DiagonalOperator(wi) * (A * x))
    r2 = T * x
    assert_allclose(r1, r2, rtol=1e-4)

    A.prep_toeplitz()
    r1_unweighted = A.H * (A * x)
    r3 = masker(A.norm(x), mask)
    assert_allclose(r1_unweighted, r3, rtol=1e-3)




# if 1 # compare to exact Gdsft:
#     A = Gnufft(mask, [omega, nufft_args[:]])
#     G = Gdsft(omega, N, 'n_shift', N/2)
#     A = A[:, :]
#     G = G[:, :]
#     im plc 2 3
#     im(1, real(A)')
#     im(2, real(G)')
#     im(3, real(A - G)')
#     im(4, imag(A)')
#     im(5, imag(G)')
#     im(6, imag(A - G)')
#     equivs(A, G, 'thresh', 8e-4) # surprisingly large threshold needed?
# prompt
# end

# classes = ['fatrix2', 'Fatrix']
# for ic=1:numel(classes)
#     pr ic
#     A = Gnufft(classes[ic], mask, [omega, nufft_args[:]])
#     fatrix2_tests(A, 'complex', 1, 'tol_gram', 3e-5)

#     T = build_gram(A, diag_sp(wi))
#     fatrix2_tests(T, 'complex', 1)

#     tic
#     for ii=1:50, b1 = embed(A' * (wi * (A * x(mask))), mask); end
#     t1 = toc
#     tic
#     for ii=1:50, b2 = embed(T * x(mask), mask); end
#     t2 = toc
#     printm('time: A''Ax = #.3f, Tx=#.3f', t1, t2)

#     equivs(b1, b2, 'thresh', 7e-5) # bigger threshold due to NUFFT approx
#     if 0:
#         d = max_percent_diff(b1, b2)
#         printm('max percent diff between Tx and A''Ax = #g##', d)
#         im plc 2 1
#         im(1, np.abs(stackup(b1, b2)), 'A''A and T'), cbar
#         im(2, np.abs(b1-b2), 'diff'), cbar
#     end
# end

