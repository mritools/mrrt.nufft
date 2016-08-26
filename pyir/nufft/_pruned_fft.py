import numpy as np
from pyir.utils import fftn, ifftn


__all__ = ['pruned_fftn',
           'pruned_ifftn',
           'pruned_fft_roundtrip']


def MD_BIT(x):
    """equivalent to the following C code:
        define MD_BIT(x) (1ul << (x))
    """
    return 1 << x


def MD_IS_SET(x, y):
    """equivalent to the following C code:
        define MD_IS_SET(x, y)  ((x) & MD_BIT(y))
    """
    return x & MD_BIT(y)


def _get_shifts(img_dims):
    """Compute FFT offsets for decompose/recompose."""
    img_dims = np.asarray(img_dims)
    ndim = img_dims.size
    if np.any(np.asarray(img_dims.shape) <= 1):
        raise ValueError("requires all dimensions to be non-singleton")
    shifts = np.zeros((2**ndim, ndim))
    slices = []
    for d in range(2*ndim):
        slices.append([slice(None), ] * ndim)
    s = 0
    for i in range(2**ndim):
        skip = False
        for j in range(ndim):
            shifts[s][j] = 0.
            slices[s][j] = slice(0, None, 2)
            if MD_IS_SET(i, j):
                skip = skip or (1 == img_dims[j])
                shifts[s][j] = -0.5
                slices[s][j] = slice(1, None, 2)
        if not skip:
            s += 1
    return shifts, slices


def linear_phase(dims, pos, dtype=np.complex64, sparse=False):
    """Compute the linear FFT phase corresponding to a spatial shift.

    Parameters
    ----------
    dims : array-like
        image shape
    pos : array-like
        shift along each image dimension

    Returns
    -------
    linphase : ndarray
        The complex phase in the Fourier domain corresponding to a shift by
        ``pos``.
    """
    pos = np.asarray(pos)
    dims = np.asarray(dims)
    ndim = pos.size
    if ndim != dims.size:
        raise ValueError("size mismatch")
    g = 2j * np.pi * pos / dims
    linphase = 1
    for d in range(ndim):
        if sparse:
            if d == 0:
                linphase = ()
        # phase along a single axis
        if g[d] == 0:
            if sparse:
                ph = 1
            else:
                ph = np.ones(dims[d], dtype=dtype)
        else:
            ph = np.exp(g[d] * np.arange(dims[d], dtype=dtype))
        if ph is not 1:
            # add singleton size along the other axes
            shape = [1, ] * ndim
            shape[d] = dims[d]
            ph = ph.reshape(shape)  # add singleton axes
        if sparse:
            if d == 0:
                linphase = [ph, ]
            else:
                linphase.append(ph)
        else:
            # net phase across all axes via broadcasting
            linphase = linphase * ph
    return linphase


def apply_linear_phase(img, linear_phase):
    """Compute the linear FFT phase corresponding to a spatial shift.

    Parameters
    ----------
    dims : array-like
        image shape
    pos : array-like
        shift along each image dimension

    Returns
    -------
    linphase : ndarray
        The complex phase in the Fourier domain corresponding to a shift by
        ``pos``.
    """
    img = np.asarray(img)
    if len(linear_phase) != img.ndim:
        raise ValueError("wrong number of dimensions.  expected a tuple")
    for d in range(img.ndim):
        lph = linear_phase[d]
        if lph is 1:
            continue
        img = img * lph
    return img


def pruned_fftn(img, linear_phases=None):
    """
    References
    ----------
    ..[1] Ong F, Uecker M, Jiang W, Lustig M.
        Fast Non-Cartesian Reconstruction with Pruned Fast Fourier Transform.
        Annual Meeting ISMRM, Toronto 2015, In: Proc Intl Soc Mag Reson Med 23;
        p. 3639.

    """
    shifts, slices = _get_shifts(img.shape)
    cplx_dtype = np.result_type(img.dtype, np.complex64)
    F = np.zeros(2*np.asarray(img.shape), dtype=cplx_dtype)
    for d, (shift, sl) in enumerate(zip(shifts, slices)):
        # tmp = img * linear_phase(img.shape, shift)
        if linear_phases is None:
            linph = linear_phase(img.shape, shift, sparse=True)
        else:
            linph = linear_phases[d]
        tmp = apply_linear_phase(img, linph)
        tmp = fftn(tmp)
        F[sl] = tmp
    return F


def pruned_ifftn(F, linear_phases=None):
    """
    References
    ----------
    ..[1] Ong F, Uecker M, Jiang W, Lustig M.
        Fast Non-Cartesian Reconstruction with Pruned Fast Fourier Transform.
        Annual Meeting ISMRM, Toronto 2015, In: Proc Intl Soc Mag Reson Med 23;
        p. 3639.

    """
    shape_in = np.asarray(F.shape)
    if np.any((shape_in % 2) > 0):
        raise ValueError("input array must have even size on all dimensions")
    img_shape = shape_in // 2
    shifts, slices = _get_shifts(img_shape)
    cplx_dtype = np.result_type(F.dtype, np.complex64)
    imgc = np.zeros(img_shape, dtype=cplx_dtype)
    for d, (shift, sl) in enumerate(zip(shifts, slices)):
        tmp = ifftn(F[sl])
        # tmp *= linear_phase(img.shape, -shift)
        if linear_phases is None:
            linph = linear_phase(img_shape, -shift, sparse=True)
        else:
            # assume linear_phases were generated for the Forward FFT
            # so need the conjugate here
            linph = np.conj(linear_phases[d])
        tmp = apply_linear_phase(tmp, linph)
        imgc += tmp
    sf = 2**F.ndim
    imgc /= sf
    return imgc


def pruned_fft_roundtrip(img, Q_pruned=None, linear_phases=None):
    """
    Round trip case saves memory vecause only Q_pruned is full size
    (oversampled by 2).  All FFTs and tmp are not oversampled.

    References
    ----------
    ..[1] Ong F, Uecker M, Jiang W, Lustig M.
        Fast Non-Cartesian Reconstruction with Pruned Fast Fourier Transform.
        Annual Meeting ISMRM, Toronto 2015, In: Proc Intl Soc Mag Reson Med 23;
        p. 3639.

    """
    shifts, slices = _get_shifts(img.shape)
    cplx_dtype = np.result_type(img.dtype, np.complex64)
    img2 = np.zeros(img.shape, dtype=cplx_dtype)
    sf_per_axis = 2
    for d, (shift, sl) in enumerate(zip(shifts, slices)):
        # tmp = img * linear_phase(img.shape, shift)
        if linear_phases is None:
            linph = linear_phase(img.shape, shift, sparse=True)
            # linph_conj = linear_phase(img.shape, -shift, sparse=True)
        else:
            linph = linear_phases[d]
        tmp = apply_linear_phase(img, linph)
        tmp = fftn(tmp)
        if Q_pruned is not None:
            if isinstance(Q_pruned, np.ndarray):
                tmp *= Q_pruned[sl]
            else:
                # list or tuple of already sliced Q_pruned
                tmp *= Q_pruned[d]
        tmp = ifftn(tmp)
        linph_conj = tuple([np.conj(l/sf_per_axis) for l in linph])
        img2 += apply_linear_phase(tmp, linph_conj)
    return img2


def split_for_pruned_fft(Q, aslist=True):
    """Reshape to: (Q.shape//2 + (2**Q.ndim, ))."""
    shifts, slices = _get_shifts(Q.shape)
    if aslist:
        # list of small arrays
        out = [Q[sl] for sl in slices]
    else:
        # create a single n-dimensional array
        qs = np.asarray(Q.shape)
        shape_out = tuple(qs//2) + (2**np.sum(qs > 1), )
        out = np.empty(shape_out, dtype=Q.dtype)
        for d, sl in enumerate(slices):
            out[..., d] = Q[sl]
    return out


def apply_linear_phase_gpu(img, linear_phase):
    """Compute the linear FFT phase corresponding to a spatial shift.

    Parameters
    ----------
    dims : array-like
        image shape
    pos : array-like
        shift along each image dimension

    Returns
    -------
    linphase : ndarray
        The complex phase in the Fourier domain corresponding to a shift by
        ``pos``.
    """
    # from skcuda.linalg import add_dot
    import skcuda.linalg as cuda_linalg

    # out = gpuarray.empty(img.shape, dtype=img.dtype)
    if len(linear_phase) != img.ndim:
        raise ValueError("wrong number of dimensions.  expected a tuple")
    for d in range(img.ndim):
        lph = linear_phase[d]
        if lph.ndim is 0:
            continue
        out = cuda_linalg.dot(img, lph)
    return out


def pruned_fft_roundtrip_cuda(img, Q_pruned, linear_phases=None, out=None):
    raise ValueError("TODO")
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    if out is None:
        out = gpuarray.empty(img.shape, dtype=cplx_dtype, order='F')
    else:
        if not isinstance(out, gpuarray):
            raise ValueError("expected out to be a gpuarray")

    asfort = np.asfortranarray
    if not isinstance(img, gpuarray):
        img = gpuarray.to_gpu(asfort(img))

    for n in range(len(Q_pruned)):
        if not isinstance(Q_pruned[n], gpuarray):
            Q_pruned[n] = gpuarray.to_gpu(asfort(Q_pruned[n]))

    sf_per_axis = 2
    shifts, slices = _get_shifts(img.shape)
    for d, (shift, sl) in enumerate(zip(shifts, slices)):
        if linear_phases is None:
            linph = linear_phase(img.shape, shift, sparse=True)
        else:
            linph = linear_phases[d]
        if not isinstance(linph, gpuarray):
            linph = gpuarray.to_gpu(asfort(linph))
        tmp = apply_linear_phase(img, linph)
        tmp = fftn(tmp)
        if Q_pruned is not None:
            if isinstance(Q_pruned, np.ndarray):
                tmp *= Q_pruned[sl]
            else:
                # list or tuple of already sliced Q_pruned
                tmp *= Q_pruned[d]
        tmp = ifftn(tmp)
        linph_conj = tuple([np.conj(l/sf_per_axis) for l in linph])
        out += apply_linear_phase(tmp, linph_conj)
    return out

def _cuda_demo():
    import numpy as np
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from skimage import img_as_float
    import skimage.data
    from pyir.nufft._pruned_fft import _get_shifts, linear_phase
    from pyir.cuda.autoinit import mulb, dev
    from pyvolplot import volshow
    from pyir.cuda.thrust_launch_config import default_block_configuration
    from pyir.cuda.cuda_utils import iDivUp
    img_cpu = img_as_float(skimage.data.camera())
    img = gpuarray.to_gpu(np.asfortranarray(img_cpu, dtype=np.complex64))

    shifts, slices = _get_shifts(img.shape)
    for d, (shift, sl) in enumerate(zip(shifts, slices)):
        print("shift={}".format(shift))

    linph = linear_phase(img.shape, [-0.5, -0.5], sparse=True)

    linph1_gpu = gpuarray.to_gpu(linph[1].ravel().astype(np.complex64))
    linph0_gpu = gpuarray.to_gpu(linph[0].ravel().astype(np.complex64))
    if False:
        tmp2 = img * linph1_gpu
        tmp2_cpu = img_cpu * linph[1]

        for r in range(img.shape[0]):
            img[r, :] = img[r, :] * linph0_gpu[:, 0]
        img = img.T
        for c in range(img.shape[1]):
            img[:, c] = img[:, c] * linph1_gpu[0, :]


    volshow(img.get())
    blocksize_mulb = (default_block_configuration(dev, mulb)[0], 1, 1)
    out = gpuarray.empty(img.shape, dtype=img.dtype, order='F')
    im_shape_gpu = gpuarray.to_gpu(np.asarray(img.shape, dtype=np.intp))
    mulb(np.intp(img.ndim),
         im_shape_gpu,
         img,
         linph1_gpu,
         out,
         np.intp(img.size),
         np.intp(1),
         grid=(iDivUp(img.size, blocksize_mulb[0]), 1, 1),
         block=blocksize_mulb)
