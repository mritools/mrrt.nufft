import numpy as np
from pyir.utils import fftmod, fftscale

# static void decompose_dims(unsigned int N, long dims2[2 * N], long ostrs2[2 * N], long istrs2[2 * N],
#         const long factors[N], const long odims[N + 1], const long ostrs[N + 1], const long idims[N], const long istrs[N])
# {
#     long prod = 1;

#     for (unsigned int i = 0; i < N; i++) {

#         long f2 = idims[i] / factors[i];

#         assert(0 == idims[i] % factors[i]);
#         assert(odims[i] == idims[i] / factors[i]);

#         dims2[1 * N + i] = factors[i];
#         dims2[0 * N + i] = f2;

#         istrs2[0 * N + i] = istrs[i] * factors[i];
#         istrs2[1 * N + i] = istrs[i];

#         ostrs2[0 * N + i] = ostrs[i];
#         ostrs2[1 * N + i] = ostrs[N] * prod;

#         prod *= factors[i];
#     }

#     assert(odims[N] == prod);
# }

# void md_decompose2(unsigned int N, const long factors[N],
#         const long odims[N + 1], const long ostrs[N + 1], void* out,
#         const long idims[N], const long istrs[N], const void* in, size_t size)
# {
#     long dims2[2 * N];
#     long ostrs2[2 * N];
#     long istrs2[2 * N];

#     decompose_dims(N, dims2, ostrs2, istrs2, factors, odims, ostrs, idims, istrs);

#     md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
# }

# void md_decompose(unsigned int N, const long factors[N], const long odims[N + 1],
#         void* out, const long idims[N], const void* in, size_t size)
# {
#     long ostrs[N + 1];
#     md_calc_strides(N + 1, ostrs, odims, size);

#     long istrs[N];
#     md_calc_strides(N, istrs, idims, size);

#     md_decompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
# }



# void md_recompose2(unsigned int N, const long factors[N],
#         const long odims[N], const long ostrs[N], void* out,
#         const long idims[N + 1], const long istrs[N + 1], const void* in, size_t size)
# {
#     long dims2[2 * N];
#     long ostrs2[2 * N];
#     long istrs2[2 * N];

#     decompose_dims(N, dims2, istrs2, ostrs2, factors, idims, istrs, odims, ostrs);

#     md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
# }

# void md_recompose(unsigned int N, const long factors[N], const long odims[N],
#         void* out, const long idims[N + 1], const void* in, size_t size)
# {
#     long ostrs[N];
#     md_calc_strides(N, ostrs, odims, size);

#     long istrs[N + 1];
#     md_calc_strides(N + 1, istrs, idims, size);

#     md_recompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
# }




# def apply_linear_phase(img, pos):
#     """Compute the linear FFT phase corresponding to a spatial shift.

#     Parameters
#     ----------
#     dims : array-like
#         image shape
#     pos : array-like
#         shift along each image dimension

#     Returns
#     -------
#     linphase : ndarray
#         The complex phase in the Fourier domain corresponding to a shift by
#         ``pos``.
#     """
#     pos = np.asarray(pos)
#     dims = np.asarray(img.shape)
#     ndim = pos.size
#     if ndim != dims.size:
#         raise ValueError("size mismatch")
#     g = 2j * np.pi * pos / dims
#     for d in range(ndim):
#         if g[d] != 0:
#             # phase along a single axis
#             ph = np.exp(g[d] * np.arange(dims[d]))
#             # add singleton size along the other axes
#             shape = [1, ] * ndim
#             shape[d] = dims[d]
#             ph = ph.reshape(shape)  # add singleton axes
#             # net phase across all axes via broadcasting
#             img = img * ph
#     return img


def compute_linphases(img_dims, fft_axes):
    """
    # # for img_dims = (16, 16, 1)
    # array([[ 0. ,  0. ,  0. ],
    #        [-0.5,  0. ,  0. ],
    #        [ 0. , -0.5,  0. ],
    #        [-0.5, -0.5,  0. ],
    #        [-0.5, -0.5, -0.5],  # won't be used
    #        [ 0. ,  0. ,  0. ],
    #        [ 0. ,  0. ,  0. ],
    #        [ 0. ,  0. ,  0. ]])
    # # for img_dims = (16, 16, 16)
    # array([[ 0. ,  0. ,  0. ],
    #        [-0.5,  0. ,  0. ],
    #        [ 0. , -0.5,  0. ],
    #        [-0.5, -0.5,  0. ],
    #        [ 0. ,  0. , -0.5],
    #        [-0.5,  0. , -0.5],
    #        [ 0. , -0.5, -0.5],
    #        [-0.5, -0.5, -0.5]])
    # # for img_dims = (1, 16, 16)
    # array([[ 0. ,  0. ,  0. ],
    #        [ 0. , -0.5,  0. ],
    #        [ 0. ,  0. , -0.5],
    #        [ 0. , -0.5, -0.5],
    #        [-0.5, -0.5, -0.5],  # won't be used
    #        [ 0. ,  0. ,  0. ],
    #        [ 0. ,  0. ,  0. ],
    #        [ 0. ,  0. ,  0. ]])
    """
    from pyir.nufft._pruned_fft import _get_shifts, linear_phase
    if fft_axes is None:
        img_dims = np.asarray(img_dims)
    else:
        img_dims_orig = np.asarray(img_dims)
        # set img_dims to 1 along any axes not being transformed
        fft_axes = np.asarray(fft_axes)
        img_dims = np.ones(img_dims.size)
        img_dims[fft_axes] = img_dims_orig[fft_axes]
    shifts, slices = _get_shifts(img_dims)
    s = shifts.shape[0]
    linphase = np.zeros(tuple(img_dims) + (s, ), dtype=np.complex64)
    for i in range(s):
        linphase[..., i] = linear_phase(img_dims, shifts[i, :])
    return linphase


def _nufft_init_linphase(img_dims, fft_axes):
    """linphase will have 1 additional dimension appended to the end. """
    linphase = compute_linphases(img_dims)
    if not toeplitz:
        linphase = linphase * roll
    linphase = fftmod(linphase, axes=fft_axes)
    linphase = fftscale(linphase, axes=fft_axes)
    scale = 1.
    for i in range(3):
        if linphase.shape[i] > 1:
            scale *= 0.5
    linphase *= scale
    return linphase


def _nufft_init_fftmod(img_dims, fft_axes):
    fftm = np.ones(img_dims, dtype=np.complex64)
    fftm = fftmod(fftm, axes=fft_axes)


def _nufft_init_psf(linphases, traj, weights, toeplitz):
    if not toeplitz:
        return None
    psf = np.zeros_like(linphases)  # TODO: shape?
    raise ValueError("TODO: finish")
    return psf


def decompose(src):
    non_singleton_axes = np.where(src != 1)[0]
    ndim = len(non_singleton_axes)
    shifts, slices = _get_shifts(src[non_singleton_axes])
    if shifts.shape[-1] != ndim:
        raise ValueError("unexpected size mismatch")
    nsets = 2**ndim
    if shifts.shape[0] != nsets:
        raise ValueError("unexpected size mismatch")
    raise ValueError("TODO: basically reshapes as in pruned_fft demo below")


def recompose(src):
    raise ValueError("TODO: basically reshapes as in pruned_fft demo below")




def nufft_forward(self, src):
    return self * src


def nufft_apply_adjoint(self, src):
    return self.H * src


def nufft_apply_normal(src, toeplitz):
    if toeplitz:
        dst = toeplitz_mult(src)
    else:
        tmp_ksp = nufft_forward(src)
        dst = nufft_apply_adjoint(tmp_ksp)
    return dst

# /**
#  * Multiply the first complex array with the conjugate of the second complex array and add to output (with strides)
#  *
#  * optr = optr + iptr1 * conj(iptr2)
#  */
# void md_zfmacc2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)

# zmul2:  optr = iptr1 * iptr2
#MAKE_Z3OP(zmul, D, dim, ostr, optr, istr1, iptr1, istr2, iptr2);
def toeplitz_mult(self, src):
    # unsigned int ND = data->N + 3;

    # md_zmul2(ND, data->cml_dims, data->cml_strs, data->grid, data->cim_strs, src, data->lph_strs, data->linphase);

    # linop_forward(data->fft_op, ND, data->cml_dims, data->grid, ND, data->cml_dims, data->grid);
    # md_zmul2(ND, data->cml_dims, data->cml_strs, data->grid, data->cml_strs, data->grid, data->psf_strs, data->psf);
    # linop_adjoint(data->fft_op, ND, data->cml_dims, data->grid, ND, data->cml_dims, data->grid);

    # md_clear(ND, data->cim_dims, dst, CFL_SIZE);
    # md_zfmacc2(ND, data->cml_dims, data->cim_strs, dst, data->cml_strs, data->grid, data->lph_strs, data->linphase);
    grid = src * self.linphase
    grid = fft_forward(grid)
    grid = grid * self.psf
    grid = fft_adjoint(grid)
    dst = np.zeros(self.cim_dims)
    dst = dst + grid * np.conj(self.linphase)


def compute_psf2(kspace, img_dims, fov, weights, **nufft_kwargs):
    from pyir.utils import fftnc
    img_dims = np.atleast_1d(img_dims)
    non_singleton_dims = np.where(img_dims != 1)[0]
    img_dims2 = np.ones_like(img_dims, dtype=np.intp)
    img_dims2[non_singleton_dims] = 2 * img_dims[non_singleton_dims]

    kspace2 = 2 * kspace

    fov = np.atleast_1d(fov)
    fov2 = 2 * fov

    psft = compute_psf(kspace2, img_dims2, fov2, weights, **nufft_kwargs)
    psft = fftnc(psft, norm='ortho')
    scale = 4**len(non_singleton_dims)
    psft *= scale
    factors = np.ones(len(img_dims2))
    factors[non_singleton_dims] = 2

    # psf = md_decompose(N + 0, factors, psf_dims, img2_dims, psft, CFL_SIZE);


def test_psf(show_figures=False):
    kspace = np.arange(-128, 128, dtype=np.float32).reshape(-1, 1)
    fov = 1
    Nd = 256
    weights = None
    nufft_kwargs = dict(mode='table0', kernel='kb:beatty')
    psf = compute_psf(kspace, Nd, fov, weights, **nufft_kwargs)
    from numpy.testing import assert_almost_equal
    # should be a delta function
    assert_almost_equal(psf[Nd//2].real, Nd, decimal=2)
    assert_almost_equal(psf[Nd//2-1].real, 0, decimal=2)
    assert_almost_equal(psf[0].real, 0, decimal=2)
    assert_almost_equal(psf[-1].real, 0, decimal=2)
    assert_almost_equal(psf[Nd//2+1].real, 0, decimal=2)
    if show_figures:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(np.abs(psf))

    # double matrix size & FOV, but keep the same number of k-space samples
    psf2 = compute_psf(kspace*2, Nd*2, fov*2, weights, **nufft_kwargs)
    # should now get 3 aliased peaks
    assert_almost_equal(psf2[Nd//2].real, Nd, decimal=2)
    assert_almost_equal(psf2[Nd].real, Nd, decimal=2)
    assert_almost_equal(psf2[3*Nd//2].real, Nd, decimal=2)
    if show_figures:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(np.abs(psf2))


def compute_psf(kspace, Nd, fov, weights, Kd=None, Jd=None, **nufft_kwargs):
    from pyir.operators_private import MRI_Operator

    if np.isscalar(Nd):
        Nd = [Nd, ]
    if np.isscalar(fov):
        fov = [fov, ]
    Nd = np.asarray(Nd)
    if Kd is None:
        Kd = 2 * Nd
    if Jd is None:
        Jd = 6 * np.ones(len(Nd))
    mask = np.ones(Nd, dtype=np.bool)
    G = MRI_Operator(Nd=Nd,
                     Kd=Kd,
                     Jd=Jd,
                     fov=fov,
                     kspace=kspace,
                     mask=mask,
                     **nufft_kwargs)
    ones = np.ones(kspace.shape[0], dtype=np.complex64)
    if weights is not None:
        ones = weights * ones
        ones = np.conj(weights) * ones
    psft = G.H * ones
    return np.squeeze(psft)
