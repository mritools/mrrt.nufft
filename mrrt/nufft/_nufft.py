"""Non-uniform fast Fourier transform.

The main class here is NufftBase. For MRI, it is recommended to use the class
MRI_Operator in ``mrrt.operators`` instead (which is built on this object
internally).

The non-uniform FFT as implemented here is described in [1], although the
interpolation kernels used are not the min-max kernels defined there, but
instead the ones described by Beatty et. al. [2].

The "forward" transfrom as defined here is a transform from uniformly sampled
"spatial" domain to a non-uniformly sampled frequency domain.

The "adjoint" transfrom as defined here is a transform from non-Cartesian
frequency samples into a uniformly sampled spatial grid.

The transforms here are implemented in 1D, 2D and 3D for either single or
double precision.

References
----------
.. [1] Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using
    min-max interpolation. IEEE Trans Sig Proc. 2003 February;51(2):560–74.
    :DOI:10.1109/TSP.2002.807005
.. [2] Beatty PJ, Nishimura DG, Pauly JM. Rapid gridding reconstruction
    with a minimal overampling ratio. IEEE Trans. Medical Imaging. 2005;
    vol. 24(no 6):799–808.
    :DOI:10.1109/TMI.2005.848376
"""

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
from mrrt.utils import nullcontext, prod
from math import sqrt
from time import time
import warnings

from mrrt.utils import (
    complexify,
    config,
    fftn,
    get_array_module,
    get_data_address,
    ifftn,
    outer_sum,
    profile,
    reale,
)
import numpy as np
import scipy.sparse

from ._dtft import dtft, dtft_adj
from ._interp_table import (
    interp1_table,
    interp2_table,
    interp3_table,
    interp1_table_adj,
    interp2_table_adj,
    interp3_table_adj,
)
from ._kaiser_bessel import kaiser_bessel_ft
from ._kernels import BeattyKernel
from ._utils import _nufft_coef, _nufft_offset, _as_tuple

if config.have_cupy:
    import cupy
    import cupyx.scipy.sparse
    from ._cupy import default_device, get_1D_block_table_gridding
    from cupyx.scipy import fftpack as cupy_fftpack
    from mrrt.nufft.cuda.cupy import _get_gridding_funcs
    from cupyx.scipy.fftpack import get_fft_plan

    cupy_fftn = cupy_fftpack.fftn
    cupy_ifftn = cupy_fftpack.ifftn


__all__ = [
    "NufftBase",
    "nufft_adj",
    "nufft_forward",
    "nufft_adj_exact",
    "nufft_forward_exact",
]

supported_real_types = [np.float32, np.float64]
supported_cplx_types = [np.complex64, np.complex128]

# TODO: use more pythonic names for Kd, Jd, Ld, Nd, etc?
#     Nd -> shape
#     Kd -> gridded_shape
#     Ld -> table_sizes
#     Jd -> kernel_sizes


def _get_legend_text(ax):
    l = ax.get_legend()
    if l is None:
        return None
    else:
        return [t.get_text() for t in l.get_texts()]


def _block_outer_sum(x1, x2, xp=None):
    J1, M = x1.shape
    J2, M = x2.shape
    xx1 = x1.reshape((J1, 1, M))
    xx2 = x2.reshape((1, J2, M))
    # use numpy broadcasting
    return xx1 + xx2  # (J1, J2, M)


def _block_outer_prod(x1, x2, xp=None):
    J1, M = x1.shape
    J2, M = x2.shape
    xx1 = x1.reshape((J1, 1, M))
    xx2 = x2.reshape((1, J2, M))
    return xx1 * xx2  # (J1, J2, M)


class NufftBase(object):
    """Base NUFFT Class.

    Parameters
    ----------
    Nd : tuple of int
        The shape of the Cartesian grid in the spatial domain. This can be
        1d, 2d or 3d.
    omega : 2d ndarray
        Non-Cartesian sampling frequencies. The shape must be
        ``(num_samples, ndim)`` where ``ndim == len(Nd)``. The values must
        be in radians within the range [0, 2*np.pi].
    J : int or array-like, optional
        The desired size of the NUFFT interpolation kernel on each axis.
        For the GPU-based NUFFT, it is currently required to use the same size
        on each axis. If an integer is provided, the same kernel dimension is
        used on all axes.
    Kd : array-like, optional
        The size of the intermediate oversampled Cartesian grid.
        The default is ``int(1.5 * Nd)``.
    precision : {'single', 'double', 'auto'}, optional
        Precision of the NUFFT operations. If 'auto' the precision will be
        set to match the provided ``omega`` array.
    mode : {'sparse', 'table'}, optional
        The NUFFT implementation to use. ``sparse`` corresponds to
        precomputation of a sparse matrix corresponding to the operation.
        This requires a longer initialization and uses more memory, but is
        typically faster than the lookup-table based approaches.
        ``table`` uses a look-up table based interpolation.  This is
        fast to initialize and uses less memory.

    Additional Parameters
    ---------------------
    Ld : array-like, optional
        When ``mode != 'sparse'``, this controls the size of the kernel
        lookup table used. The total lookup table size will be ``J[i] * Ld[i]``
        for axis ``i``.
    n_shift : tuple of float, optional
        Controls the shift applied (e.g. default is an N // 2 shift like
        fftshift)
    ortho : bool, optional
        If True, an orthogonal FFT with normalization factor (1/sqrt(n)) is
        used in each direction. Otherwise normalization 1/n is applied
        during the inverse FFT and no scaling is applied to the forward
        transform.
    phasing : {'real', 'complex'}, optional
        If complex, the gridding kernel is complex-valued and the phase
        roll corresponding to ``n_shift`` is built into the kernel.
        If real, a real-valued kernel is used and the phase rolls are
        applied separately.  Performance is similar in either case.
    adjoint_scalefactor : float, optional
        A custom, constant multiplicative scaling factor for the
        adjoint operation. This is applied on top of an scaling related to
        ``ortho``.
    preplan_cufft : bool, optional
        If True the cuFFT plan is precomputed and stored as attribute
        `cufft_plan`. If the transform is not on the GPU, this parameter is
        ignored.

    Attributes
    ----------
    ndim : int
        The number of dimensions.
    kernel : ``mrrt.nufft.NufftKernelBase``
        The interpolation kernel object being used. Currently this will always
        be of type ``mrrt.nufft.BeattyKernel``.
    p : scipy.sparse.csc_matrix, cupyx.scipy.sparse.csc_matrix or None
        When ``mode == 'sparse'``, this contains the sparse matrix
        corresponding to the fully precomputed interpolation.
    tm : ndarray
        This is a version of ``omega`` normalized to the range ``[0, Kd[i]]``.

    Notes
    -----
    The non-uniform FFT as implemented here is described in [1], although the
    interpolation kernels used are not the min-max kernels defined there, but
    instead the ones described in [2].

    The package ``mrrt.mri`` provides a subclass that also inherits from
    mrrt.operators.LinearOperatorMulti.

    References
    ----------
    .. [1] Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using
        min-max interpolation. IEEE Trans Sig Proc. 2003 February;51(2):560–74.
        :DOI:10.1109/TSP.2002.807005
    .. [2] Beatty PJ, Nishimura DG, Pauly JM. Rapid gridding reconstruction
        with a minimal overampling ratio. IEEE Trans. Medical Imaging. 2005;
        vol. 24(no 6):799–808.
        :DOI:10.1109/TMI.2005.848376

    """

    @profile
    def __init__(
        self,
        Nd,
        omega,
        Jd=4,
        Kd=None,
        precision="single",
        mode="table",
        Ld=1024,
        ortho=False,
        n_shift=None,
        phasing="real",
        adjoint_scalefactor=1.0,
        preplan_cufft=True,
        order="F",
        verbose=False,
        on_gpu=False,
    ):
        """ Initialize NufftBase instance


        """
        self.verbose = verbose
        self.__init_complete = False  # will be set true after __init__()
        self.order = order

        self.__on_gpu = on_gpu
        # get the array module being used (based on state of self.on_gpu)

        # set the private version (__Nd not Nd) to avoid circular calls
        # also Nd, Kd, Jd, etc. should be on the CPU
        if np.isscalar(Nd):
            Nd = (Nd,)
        self.__Nd = _as_tuple(Nd, type=int)
        self.__phasing = phasing
        self.__omega = None  # will be set later below
        self._set_n_mid()
        self.ndim = len(self.Nd)  # number of dimensions
        self.__Jd = _as_tuple(Jd, type=int, n=self.ndim)

        if Kd is None:
            Kd = tuple([int(1.5 * n) for n in self.__Nd])
        self.__Kd = _as_tuple(Kd, type=int, n=self.ndim)

        # normalization for orthogonal FFT
        self.ortho = ortho
        self.scale_ortho = sqrt(prod(self.__Kd)) if self.ortho else 1

        # placeholders for phase_before/phase_after. phasing.setter
        self.phase_before = None
        self.phase_after = None

        # n_shift placeholder
        self.__n_shift = None

        # placeholders for dtypes:  will be set by precision.setter
        self._cplx_dtype = None
        self._real_dtype = None

        self._init_omega(omega)

        if precision == "auto":
            if hasattr(omega, "dtype"):
                if omega.dtype in [self.xp.float32]:
                    precision = "single"
                elif omega.dtype in [self.xp.float64]:
                    precision = "double"
            else:
                precision = "double"
        self.precision = precision
        self._forw = None
        self._adj = None
        self._init = None
        self.__mode = mode
        self.adjoint_scalefactor = adjoint_scalefactor

        # [M, *Kd]	sparse interpolation matrix (or empty if table-based)
        self.p = None

        # initialize the interpolation kernel
        self.kernel = BeattyKernel(
            shape=self.Jd, grid_shape=self.Nd, os_grid_shape=self.Kd
        )
        self._calc_scaling()

        self.M = 0
        if self.omega is not None:
            self.M = self.omega.shape[0]
        if n_shift is None:
            # not self.Nd // 2 because this is in addition to self.n_mid
            self.__n_shift = (0.0,) * self.ndim
        else:
            self.__n_shift = _as_tuple(n_shift, type=float, n=self.ndim)
        if (self.ndim != len(self.Jd)) or (self.ndim != len(self.Kd)):
            raise ValueError("Inconsistent Dimensions")

        # set the phase to be applied if self.phasing=='real'
        self._set_phase_funcs()

        self._update_array__precision()
        self._make_arrays_contiguous(order="F")

        # TODO: cleanup how initialization is done
        self.__sparse_format = None

        if self.mode == "sparse":
            self._init_sparsemat(highmem=True)  # create CSC matrix
            self.__sparse_format = "CSC"
            # Table-based gridding is still used during precomputation, so Ld
            # must be set.
            self.__Ld = Ld
        elif "table" in self.mode:
            self.__Ld = Ld

            odd_L = self.Ld % 2 == 1
            odd_J = np.mod(self.Jd, 2) == 1
            if odd_L and any(odd_J):
                warnings.warn(
                    "accuracy may be compromised when L and J are both odd"
                )
            self._init_table()

            tm = self.xp.zeros_like(self.omega)
            for d in range(0, self.ndim):
                gam = 2 * np.pi / self.Kd[d]
                tm[:, d] = self.omega[:, d] / gam  # t = omega / gamma
            self.tm = tm
            self.interp_table = _nufft_table_interp
            self.interp_table_adj = _nufft_table_adj
        elif self.mode == "exact":
            # TODO: wrap calls to dtft, dtft_adj
            raise ValueError("mode exact not implemented")
            pass
        else:
            raise ValueError("Invalid NUFFT mode: {}".format(self.mode))
        self.nargin1 = prod(self.Nd)
        self.nargout1 = self.omega.shape[0]
        self._update_array__precision()
        self._make_arrays_contiguous(order="F")
        self.__init_complete = True
        self.preplan_cufft = preplan_cufft

        if self.on_gpu:
            self._init_gpu()

    @profile
    def _init_gpu(self):
        M = self.omega.shape[0]
        if not len(np.unique(self.Jd)):
            raise ValueError(
                "GPU case requires same gridding kernel size on each axis."
            )

        if self.preplan_cufft:
            dummy = np.empty(self.Kd, dtype=self._cplx_dtype, order=self.order)
            self.cufft_plan = get_fft_plan(dummy, shape=dummy.shape)
            del dummy
        else:
            # Null context since self.cufft_plan is used as a context manager
            self.cufft_plan = nullcontext()

        # compile CUDA kernels corresponding to this operator
        self.kern_forward, self.kern_adj = _get_gridding_funcs(
            Kd=self.Kd,
            M=self.M,
            J=self.Jd[0],
            L=self.Ld,
            is_complex_kernel=(self.phasing == "complex"),
            precision=self.precision,
        )

        # store a block and grid configuration for use with the kernel
        self.block, self.grid = get_1D_block_table_gridding(
            M, dev=default_device, kernel=self.kern_adj
        )

    @property
    def on_gpu(self):
        """bool: Indicates whether the filterbank is a GPU filterbank."""
        return self.__on_gpu

    @on_gpu.setter
    def on_gpu(self, on_gpu):
        if on_gpu in [True, False]:
            self.__on_gpu = on_gpu
        else:
            raise ValueError("on_gpu must be True or False")

    @property
    def xp(self):
        """module: The ndarray backend being used."""
        # Note: The module is not kept as an attribute so pickling is possible.
        if self.__on_gpu:
            if not config.have_cupy:
                raise ValueError("CuPy is required for the GPU implementation.")
            return cupy
        else:
            return np

    def _swap_reps(self, x, narg):
        if x.size != narg:
            x = x.transpose(tuple(range(1, x.ndim)) + (0,))
        return x

    def _unswap_reps(self, x, narg):
        if x.size != narg:
            x = x.transpose((x.ndim - 1,) + tuple(range(x.ndim - 1)))
        return x

    def fft(self, x):
        """Perform the forward NUFFT (uniform spatial -> nonuniform freq.).

        Parameters
        ----------
        x : ndarray
            The uniform spatial domain data. This will have shape equal to
            ``self.Nd``.

        Returns
        -------
        k : ndarray
            The non-uniformly samples Fourier domain values.
        """
        # if self.order == "F":
        #     x = x.reshape(self.Nd + (-1,))
        # elif self.order == "C":
        #     x = x.reshape((-1,) + self.Nd)
        if self.order == "C":
            # functions expect reps at end, not start
            x = self._swap_reps(x, self.nargin1)
        if self.mode == "exact":
            k = nufft_forward_exact(self, x=x)
        else:
            k = nufft_forward(self, x=x)
        if self.order == "C":
            k = self._unswap_reps(k, self.nargout1)
        return k

    def adj(self, k):
        """Perform the adjoint NUFFT (nonuniform freq. -> uniform spatial).

        Parameters
        ----------
        k : ndarray
            This should contain Fourier domain values. It should have size
            equal to the number of frequency samples, ``self.omega.shape[0]``.

        Returns
        -------
        x : ndarray
            The uniform spatial domain data. This will have shape equal to
            ``self.Nd``.
        """
        # if self.order == "F":
        #     k = k.reshape((self.omega.shape[0], -1))
        # elif self.order == "C":
        #     k = k.reshape((-1, self.omega.shape[0]))
        if self.order == "C":
            # functions expect reps at end, not start
            k = self._swap_reps(k, self.nargout1)
        if self.mode == "exact":
            x = nufft_adj_exact(self, xk=k)
        else:
            x = nufft_adj(self, xk=k)
        if self.order == "C":
            x = self._unswap_reps(x, self.nargin1)
        return x

    def _set_phase_funcs(self):
        """Initialize the phasing used to realize the FFT shifts.

        When ``self.phasing == "real"``, this is an explict multiplication by
        the phase corresponding to the shift.

        When ``self.phasing == "complex"``, the phase shift is built into the
        NUFFT convolution kernel itself.

        """
        self._set_n_mid()
        if self.phasing == "real":
            self.phase_before = self._phase_before(self.Kd, self.n_mid)
            self.phase_after = self._phase_after(
                self.omega, self.n_mid, self.n_shift
            )
        elif self.phasing == "complex":
            # complex kernel incorporates the FFTshift phase
            self.phase_before = None
            self.phase_after = None
        else:
            raise ValueError(
                f"Invalid phasing: {self.phasing}. phasing must be 'real' or "
                "'complex'"
            )

    @property
    def sparse_format(self):
        """str: internal sparse matrix format (when ``mode=='sparse'``)."""
        return self.__sparse_format

    @property
    def precision(self):
        """str: The numerical precision of the transform (single or double)."""
        return self.__precision

    @precision.setter
    def precision(self, precision):
        """Set the desired numerical precision.

        Parameters
        ----------
        precision : {"single", "double"}
            The precision to use.

        Notes
        -----
        Setting this property results in properties such as ``omega``
        being converted to the specified precision.
        """
        # set corresponding real and complex types
        if precision == "single":
            self._cplx_dtype = np.dtype(np.complex64)
            self._real_dtype = np.dtype(np.float32)
        elif precision == "double":
            self._cplx_dtype = np.dtype(np.complex128)
            self._real_dtype = np.dtype(np.float64)
        else:
            raise ValueError("precision must be 'single', 'double' or 'auto'")
        self.__precision = precision
        if self.__init_complete:
            self._update_array__precision()

    @property
    def omega(self):
        """ndarray: The array of non-Cartesian sampling points."""
        return self.__omega

    def _init_omega(self, omega):
        """Initialize omega (called during __init__)."""
        if omega is not None:
            omega = self.xp.asarray(omega)
            if omega.ndim == 1:
                omega = omega[:, np.newaxis]
            if omega.shape[1] != self.ndim:
                raise ValueError("number of cols must match NUFFT dimension")
            if omega.dtype not in supported_real_types:
                raise ValueError(
                    "omega must be one of the following types: "
                    "{}".format(supported_real_types)
                )
            if self.ndim != omega.shape[1]:
                raise ValueError("omega needs {} columns".format(self.ndim))
        self.__omega = omega
        if isinstance(self.phase_before, np.ndarray):
            self.phase_after = self._phase_after(
                omega, self.n_mid, self.__n_shift
            )
        if self.__init_complete:
            self._reinitialize()

    def _reinitialize(self):
        """Utility to reinitialize the NUFFT object"""
        if self.mode == "sparse":
            self._init_sparsemat(highmem=True)
        elif "table" in "mode":
            self._init_table()

    @property
    def phasing(self):
        """str: The type of phasing used."""
        return self.__phasing

    @phasing.setter
    def phasing(self, phasing):
        self.__phasing = phasing
        self._set_n_mid()
        self._set_phase_funcs()

    @property
    def Nd(self):
        """tuple of int: The shape of the uniform spatial data to transform."""
        return self.__Nd

    @property
    def Jd(self):
        """tuple of int: The shape of the interpolation kernel."""
        return self.__Jd

    @property
    def Ld(self):
        """int or None: kernel lookup table size is ``self.Ld * self.Jd``."""
        return self.__Ld

    @property
    def Kd(self):
        """tuple of int: The shape of the oversampled spatial grid."""
        return self.__Kd

    @property
    def mode(self):
        """str: The NUFFT computation mode being used (table or sparse)."""
        return self.__mode

    @property
    def n_shift(self):
        """tuple of int: additional spatial shift factor."""
        return self.__n_shift

    def _set_n_mid(self):
        # midpoint of scaling factors
        if self.__phasing == "real":
            self.n_mid = tuple([n // 2 for n in self.Nd])
        else:
            self.n_mid = tuple([(n - 1) / 2.0 for n in self.Nd])
        if self.phasing == "real" and (self.__omega is not None):
            self.phase_after = self._phase_after(
                self.__omega, self.n_mid, self.__n_shift
            )
        if self.__init_complete:
            self._reinitialize()

    def _update_dtype(self, arr, mode=None, xp=np):
        if mode is None:
            if xp.iscomplexobj(arr):
                if arr.dtype != self._cplx_dtype:
                    arr = arr.astype(self._cplx_dtype)
            else:
                if arr.dtype != self._real_dtype:
                    arr = arr.astype(self._real_dtype)
        elif mode == "real":
            if arr.dtype != self._real_dtype:
                arr = arr.astype(self._real_dtype)
        elif mode == "complex":
            if arr.dtype != self._cplx_dtype:
                arr = arr.astype(self._cplx_dtype)
        else:
            raise ValueError("unrecognized mode")
        return arr

    def _update_array__precision(self):
        # update the data types of other members
        # TODO: warn if losing precision during conversion?
        xp = self.xp
        if isinstance(self.__omega, xp.ndarray):
            self.__omega = self._update_dtype(self.omega, "real", xp=xp)
        if isinstance(self.phase_before, xp.ndarray):
            self.phase_before = self._update_dtype(
                self.phase_before, "complex", xp=xp
            )
        if isinstance(self.phase_after, xp.ndarray):
            self.phase_after = self._update_dtype(
                self.phase_after, "complex", xp=xp
            )
        if hasattr(self, "sn") and isinstance(self.sn, xp.ndarray):
            self.sn = self._update_dtype(self.sn, xp=xp)
        if self.mode == "sparse":
            if hasattr(self, "p") and self.p is not None:
                self.p = self._update_dtype(self.p, self.phasing, xp=xp)
        elif "table" in self.mode:
            if hasattr(self, "h") and self.h is not None:
                for idx, h in enumerate(self.h):
                    self.h[idx] = self._update_dtype(h, self.phasing, xp=xp)
        else:
            raise ValueError("unknown mode")

    def _make_arrays_contiguous(self, order="F"):
        xp = self.xp
        # arrays potentially stored on the GPU use contig_func
        if order == "F":
            contig_func = xp.asfortranarray
        elif order == "C":
            contig_func = xp.ascontiguousarray
        else:
            raise ValueError("order must be 'F' or 'C'")
        self.__omega = contig_func(self.__omega)
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = contig_func(self.phase_before)
        if isinstance(self.phase_after, np.ndarray):
            self.phase_after = contig_func(self.phase_after)
        if hasattr(self, "sn") and self.sn is not None:
            self.sn = contig_func(self.sn)
        if self.mode == "sparse":
            pass
        if "table" in self.mode:
            if hasattr(self, "h") and self.h is not None:
                for h in self.h:
                    h = contig_func(h)

    def _phase_before(self, Kd, n_mid):
        """Needed to realize desired FFT shift for real-valued NUFFT kernel.
        """
        xp = self.xp
        rdt = self._real_dtype
        phase = (2 * np.pi / Kd[0] * n_mid[0]) * xp.arange(Kd[0], dtype=rdt)
        for d in range(1, self.ndim):
            tmp = (2 * np.pi / Kd[d] * n_mid[d]) * xp.arange(Kd[d], dtype=rdt)
            # fast outer sum via broadcasting
            phase = phase.reshape((phase.shape) + (1,)) + tmp.reshape(
                (1,) * d + (tmp.size,)
            )
        return xp.exp(1j * phase).astype(self._cplx_dtype, copy=False)

    def _phase_after(self, omega, n_mid, n_shift):
        """Needed to realize desired FFT shift for real-valued NUFFT kernel.
        """
        xp = self.xp
        rdt = self._real_dtype
        shift_vec = [(s - m) for s, m in zip(n_shift, n_mid)]
        phase = xp.exp(1j * xp.dot(omega, xp.asarray(shift_vec, dtype=rdt)))
        return phase.astype(self._cplx_dtype, copy=False)

    @profile
    def _calc_scaling(self):
        """image domain scaling factors to account for the finite NUFFT
        kernel.  (i.e. intensity rolloff correction)
        """
        xp = self.xp
        kernel = self.kernel
        Nd = self.Nd
        Kd = self.Kd
        Jd = self.Jd
        self.sn = xp.array([1.0])
        for d in range(self.ndim):
            # Note better to use NumPy than Cupy for these small 1d
            # arrays. Transfer tmp to the GPU after kaiser_bessel_ft
            start = -self.n_mid[d]
            nc = np.arange(start, start + Nd[d])
            tmp = 1 / kaiser_bessel_ft(
                nc / Kd[d], Jd[d], kernel.alpha[d], kernel.m[d], 1
            )

            tmp = reale(tmp)
            self.sn = xp.outer(self.sn.ravel(), xp.asarray(tmp.conj()))
        self.sn = self.sn.reshape(Nd)  # [(Nd)]

    @profile
    def _init_sparsemat(self, highmem=None):
        """Initialize structure for n-dimensional NUFFT using Sparse matrix
        multiplication.
        """
        tstart = time()
        ud = {}
        kd = {}
        omega = self.omega
        if omega.ndim == 1:
            omega = omega[:, np.newaxis]

        xp, on_gpu = get_array_module(omega)
        if on_gpu:
            coo_matrix = cupyx.scipy.sparse.coo_matrix
        else:
            coo_matrix = scipy.sparse.coo_matrix

        if self.phasing == "real":
            # call again just to be safe in case Kd, n_mid, etc changed?
            self._set_phase_funcs()

        for d in range(self.ndim):
            N = self.Nd[d]
            J = self.Jd[d]
            K = self.Kd[d]

            # callable kernel:  kaiser, linear, etc
            kernel_func = self.kernel.kernels[d]
            if not isinstance(kernel_func, Callable):
                raise ValueError("callable kernel function required")

            # [J?,M]
            [c, arg] = _nufft_coef(omega[:, d], J, K, kernel_func, xp=xp)

            # indices into oversampled FFT components
            #
            # [M,1] to leftmost near nbr
            koff = _nufft_offset(omega[:, d], J, K, xp=xp)

            # [J,M]
            kd[d] = xp.mod(outer_sum(xp.arange(1, J + 1), koff), K)

            if self.phasing == "complex":
                gam = 2 * np.pi / K
                phase_scale = 1j * gam * (N - 1) / 2.0
                phase = xp.exp(phase_scale * arg)  # [J,M] linear phase
            else:
                phase = 1.0
            # else:
            #     raise ValueError("Unknown phasing {}".format(self.phasing))

            ud[d] = phase * c  # [J?,M]

        tend1 = time()
        if self.verbose:
            print("Nd={}".format(self.Nd))
            print("Sparse init stage 1 duration = {} s".format(tstart - tend1))

        """
        build sparse matrix that is shape (M, *Kd)
        """
        M = self.omega.shape[0]
        kk = kd[0]  # [J1,M]
        uu = ud[0]  # [J1,M]

        for d in range(1, self.ndim):
            Jprod = prod(self.Jd[0 : d + 1])
            # trick: pre-convert these indices into offsets! (Fortran order)
            tmp = kd[d] * prod(self.Kd[:d])
            kk = _block_outer_sum(kk, tmp)  # outer sum of indices
            kk = kk.reshape((Jprod, M), order="F")
            uu = _block_outer_prod(uu, ud[d])  # outer product of coefficients
            uu = uu.reshape((Jprod, M), order="F")
        # now kk and uu are shape (*Jd, M)

        #
        # apply phase shift
        # pre-do Hermitian transpose of interpolation coefficients
        #
        if xp.iscomplexobj(uu):
            uu = uu.conj()

        if self.phasing == "complex":
            if any([s != 0 for s in self.n_shift]):
                phase = xp.exp(
                    1j * xp.dot(omega, xp.asarray(self.n_shift))
                )  # [1,M]
                phase = phase.reshape((1, -1), order="F")
                uu *= phase  # use broadcasting along first dimension
            sparse_dtype = self._cplx_dtype
        # elif self.phasing == 'real' or self.phasing is None:
        else:
            sparse_dtype = self._real_dtype
        # else:
        #    raise ValueError("Invalid phasing: {}".format(self.phasing))

        if self.verbose:
            RAM_GB = prod(self.Jd) * M * sparse_dtype.itemsize / 10 ** 9
            print(f"NUFFT sparse matrix storage will require {RAM_GB} GB")

        # shape (*Jd, M)
        mm = xp.tile(xp.arange(M), (prod(self.Jd), 1))

        self.p = coo_matrix(
            (uu.ravel(order="F"), (mm.ravel(order="F"), kk.ravel(order="F"))),
            shape=(M, int(prod(self.Kd))),
            dtype=sparse_dtype,
        )

        # return in CSC format by default
        if self.on_gpu:
            self.p = self.p.tocsc()
            if highmem is None:
                free_memory_bytes = cupy.cuda.device.Device().mem_info[0]
                # if "enough" memory left, keep a copy (factor of 2 is an
                # arbitrary choice)
                highmem = free_memory_bytes > 2 * self.p.data.nbytes
            if highmem:
                self.p_csr = self.p.tocsr()
            else:
                self.p_csr = None
        else:
            self.p = self.p.tocsc()

        tend2 = time()
        if self.verbose:
            print("Sparse init stage 2 duration = {} s".format(tend2 - tend1))

    # @profile
    def _init_table(self):
        """Initialize structure for n-dimensional NUFFT using table-based
        interpolator.
        """
        # for convenience
        ndim = self.ndim
        xp = self.xp
        # need to strip ndim, Nd, Jd, Kd from local copy of kernel_kwargs
        kernel_kwargs = self.kernel.params.copy()
        kernel_kwargs.pop("ndim", None)
        kernel_kwargs.pop("Nd", None)
        kernel_kwargs.pop("Jd", None)
        kernel_kwargs.pop("Kd", None)
        kernel_kwargs.pop("n_mid", None)
        # if ('kb:' in self.kernel.kernel_type):
        # how = 'ratio'  #currently a bug in ratio case for non-integer K/N
        #     else:
        how = "fast"
        if self.phasing == "complex" and any([s != 0 for s in self.n_shift]):
            self.phase_shift = xp.exp(
                1j * xp.dot(self.omega, xp.asarray(self.n_shift))
            )  # [M 1]
        else:
            self.phase_shift = None  # compute on-the-fly
        if self.Ld is None:
            self.Ld = 2 ** 10
        if ndim != len(self.Jd) or ndim != len(self.Kd):
            raise ValueError("inconsistent dimensions among ndim, Jd, Ld, Kd")
        if ndim != self.omega.shape[1]:
            raise ValueError("omega needs %d columns" % (ndim))

        self.h = []
        # build kernel lookup table (LUT) for each dimension
        for d in range(ndim):
            h, t0 = _nufft_table_make1(
                how=how,
                N=self.Nd[d],
                J=self.Jd[d],
                K=self.Kd[d],
                L=self.Ld,
                phasing=self.phasing,
                order=self.order,
                xp=xp,
            )

            if self.phasing == "complex":
                if xp.isrealobj(h):
                    warnings.warn("Real NUFFT kernel?")
                h = complexify(h, complex_dtype=self._cplx_dtype)
            # elif self.phasing in ['real', None]:
            else:
                try:
                    h = reale(h)
                except ValueError:
                    raise ValueError("expected real NUFFT kernel")
            self.h.append(h)

    def __str__(self):
        attribs = dir(self)
        attribs = [item for item in attribs if not item.startswith("__")]
        str = ""
        for attrib in attribs:
            val = getattr(self, attrib, None)
            if isinstance(val, np.ndarray):
                str += "{} = ndarray: dtype={}, shape={}\n".format(
                    attrib, val.dtype, val.shape
                )
            elif hasattr(val, "__cupy_array_interface__"):
                str += "{} = cupy array: dtype={}, shape={}\n".format(
                    attrib, val.dtype, val.shape
                )
            elif isinstance(val, scipy.sparse.data._data_matrix):
                str += "{} = {}\n".format(attrib, val.__repr__)
            else:
                str += "{} = {}\n".format(attrib, val)
            # print(str)
        return str

    def plot_kernels(self, with_phasing=False):
        from matplotlib import pyplot as plt

        """Plots the NUFFT gridding kernel for each axis of the NUFFT."""
        gridspec_kw = dict(hspace=0.1)
        fig, axes = plt.subplots(
            self.ndim, 1, sharex="col", gridspec_kw=gridspec_kw
        )
        for d in range(self.ndim):
            if self.mode != "sparse":
                x = np.linspace(-self.Jd[d] / 2, self.Jd[d] / 2, self.h[d].size)
                y = self.h[d]
                if self.on_gpu:
                    y = y.get()
            else:
                if self.kernel.kernel is not None:
                    if with_phasing:
                        raise ValueError(
                            "with_phasing option only supported for "
                            "table-based NUFFT"
                        )
                    x = np.linspace(-self.Jd[d] / 2, self.Jd[d] / 2, 1000)
                    y = self.kernel.kernel[d](x, self.Jd[d])
                else:
                    print(
                        "Kernel is not an inline function. will not be"
                        "plotted"
                    )
            axes[d].plot(x, np.abs(y), "k-", label="magnitude")
            if with_phasing:
                axes[d].plot(x, y.real, "k--", label="real")
                axes[d].plot(x, y.imag, "k:", label="imag")
                axes[d].legend()
            axes[d].set_ylabel("axis %d" % d)
            if d == self.ndim - 1:
                axes[d].set_xlabel("oversampled grid offset")
        return fig, axes


@profile
def _nufft_table_interp(obj, xk, omega=None, xp=None):
    """ Forward NUFFT based on kernel lookup table (image-space to k-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    xk : array
        DFT values (images) [npixels, ncoils*n_reps]
    copy_x : bool, optional
        make a copy of x internally to avoid potentially modifying the
        original array.

    Returns
    -------
    x : array
        DTFT coefficients (k-space)
    """
    if omega is None:
        omega = obj.omega

    ndim = len(obj.Kd)
    _xp, on_gpu = get_array_module(xk, xp)
    if obj.on_gpu and not on_gpu:
        # if GPU-based NUFFT and numpy input we have to transfer x to the GPU
        xk = obj.xp.asarray(xk)
        on_gpu = True
    elif on_gpu and not obj.on_gpu:
        # if CPU-based NUFFT and cupy input we have to transfer x from the GPU
        xk = xk.get()
        on_gpu = False
    xp = obj.xp

    tm = obj.tm
    if xk.ndim == 1:
        xk = xk[:, xp.newaxis]
    elif xk.shape[1] > xk.shape[0]:
        xk = xk.T
    nc = xk.shape[1]

    if xk.shape[0] != prod(obj.Kd):
        raise ValueError("xk size problem")

    xk = complexify(xk, complex_dtype=obj._cplx_dtype)  # force complex
    if not obj.on_gpu:
        arg = [obj.Jd, obj.Ld, tm]
        if ndim == 1:
            arg = [obj.Jd[0], obj.Ld, tm]
            x = interp1_table(xk, obj.h[0], *arg)
        elif ndim == 2:
            # Fortran ordering to match Matlab behavior
            xk = xp.reshape(xk, tuple(np.hstack((obj.Kd, nc))), order="F")
            x = interp2_table(xk, obj.h[0], obj.h[1], *arg)
        elif ndim == 3:
            # Fortran ordering to match Matlab behavior
            xk = xp.reshape(xk, tuple(np.hstack((obj.Kd, nc))), order="F")
            x = interp3_table(xk, obj.h[0], obj.h[1], obj.h[2], *arg)
        else:
            raise NotImplementedError("dimensions > 3 not implemented")
    else:
        x = xp.zeros((obj.M, xk.shape[-1]), dtype=obj._cplx_dtype, order="F")
        nreps = x.shape[-1]
        if nreps == 1:
            if ndim == 1:
                args = (xk, obj.h[0], tm, x)
            elif ndim == 2:
                args = (xk, obj.h[0], obj.h[1], tm, x)
            elif ndim == 3:
                args = (xk, obj.h[0], obj.h[1], obj.h[2], tm, x)
            obj.kern_forward(obj.grid, obj.block, args)
        else:
            for r in range(nreps):
                if ndim == 1:
                    args = (xk[..., r], obj.h[0], tm, x[..., r])
                elif ndim == 2:
                    args = (xk[..., r], obj.h[0], obj.h[1], tm, x[..., r])
                elif ndim == 3:
                    args = (
                        xk[..., r],
                        obj.h[0],
                        obj.h[1],
                        obj.h[2],
                        tm,
                        x[..., r],
                    )
                # obj.kern_forward(obj.block, obj.grid, args)
                obj.kern_forward(obj.grid, obj.block, args)

    # apply phase shift
    if hasattr(obj, "phase_shift"):
        if isinstance(obj.phase_shift, (xp.ndarray, list)):
            if len(obj.phase_shift) > 0:
                ph = obj.phase_shift
                if ph.ndim == 1:
                    ph = xp.reshape(ph, (-1, 1))
                if x.shape[0] != ph.shape[0] or x.ndim != ph.ndim:
                    raise RuntimeError("dimension mismatch")
                x *= ph

    return x.astype(xk.dtype, copy=False)


@profile
def _nufft_table_adj(obj, x, omega=None, xp=None):
    """ Adjoint NUFFT based on kernel lookup table (k-space to image-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    x : array
        DTFT values (k-space) [nsamples, ncoils*n_reps]
    omega : ndarray, optional
        Frequency locations corresponding to the samples.  By default this is
        obtained from obj.

    Returns
    -------
    xk : array
        DFT coefficients
    """
    if omega is None:
        omega = obj.omega
    _xp, on_gpu = get_array_module(x, xp)
    if obj.on_gpu and not on_gpu:
        # if GPU-based NUFFT and numpy input we have to transfer x to the GPU
        x = obj.xp.asarray(x)
        on_gpu = True
    elif on_gpu and not obj.on_gpu:
        # if CPU-based NUFFT and cupy input we have to transfer x from the GPU
        x = x.get()
        on_gpu = False
    xp = obj.xp

    ndim = len(obj.Kd)
    tm = obj.tm

    if x.shape[0] != omega.shape[0]:
        raise ValueError("x size problem")

    if x.ndim == 1:
        x = x[:, xp.newaxis]

    # adjoint of phase shift
    if hasattr(obj, "phase_shift"):
        if isinstance(obj.phase_shift, (xp.ndarray, list)):
            if len(obj.phase_shift) > 0:
                ph_conj = obj.phase_shift.conj()
                if ph_conj.ndim == 1:
                    ph_conj = xp.reshape(ph_conj, (-1, 1))
                if x.shape[0] != ph_conj.shape[0] or x.ndim != ph_conj.ndim:
                    raise RuntimeError("dimension mismatch")
                x *= ph_conj

    # force proper complex dtype
    x = complexify(x, complex_dtype=obj._cplx_dtype)

    if not obj.on_gpu:
        arg = [obj.Jd, obj.Ld, tm, obj.Kd[0:ndim]]
        if ndim == 1:
            arg = [obj.Jd[0], obj.Ld, tm, obj.Kd[0]]
            xk = interp1_table_adj(x, obj.h[0], *arg)
        elif ndim == 2:
            xk = interp2_table_adj(x, obj.h[0], obj.h[1], *arg)
        elif ndim == 3:
            xk = interp3_table_adj(x, obj.h[0], obj.h[1], obj.h[2], *arg)
        else:
            raise NotImplementedError("dimensions > 3 not implemented")
    else:
        # kern_forward(block, grid, (ck, h1, tm, fm))
        nreps = x.shape[-1]
        xk = xp.zeros((prod(obj.Kd), nreps), dtype=obj._cplx_dtype, order="F")
        if nreps == 1:
            if ndim == 1:
                args = (xk, obj.h[0], tm, x)
            elif ndim == 2:
                args = (xk, obj.h[0], obj.h[1], tm, x)
            elif ndim == 3:
                args = (xk, obj.h[0], obj.h[1], obj.h[2], tm, x)
            # obj.kern_adj(obj.block, obj.grid, args)
            obj.kern_adj(obj.grid, obj.block, args)
        else:
            # TODO: remove need for python-level loop here
            for r in range(nreps):
                if ndim == 1:
                    args = (xk[:, r], obj.h[0], tm, x[:, r])
                elif ndim == 2:
                    args = (xk[:, r], obj.h[0], obj.h[1], tm, x[:, r])
                elif ndim == 3:
                    args = (xk[:, r], obj.h[0], obj.h[1], obj.h[2], tm, x[:, r])
                # obj.kern_adj(obj.block, obj.grid, args)
                obj.kern_adj(obj.grid, obj.block, args)

    return xk.astype(x.dtype, copy=False)


# @profile
def _nufft_table_make1(how, N, J, K, L, phasing, order, debug=False, xp=np):
    """ make LUT for 1 dimension by creating a dummy 1D NUFFT object """
    # kernel_1d = BeattyKernel(shape=(J,), grid_shape=(N,), os_grid_shape=(K,))
    nufft_args = dict(
        Jd=J,
        n_shift=0,
        mode="sparse",
        phasing=phasing,
        order=order,
        on_gpu=(xp != np),
    )
    t0 = xp.arange(-J * L / 2.0, J * L / 2.0 + 1) / L  # [J*L+1]
    if not t0.size == (J * L + 1):
        raise ValueError("bad t0.size")
    pi = xp.pi
    if N % 2 == 0:
        # may be a slight symmetry error for odd N?
        # warnings.warn("odd N in _nufft_table_make1.  even N recommended.")
        pass

    # This is a slow and inefficient (but simple) way to get the table
    # because it builds a huge sparse matrix but only uses 1 column!
    if how == "slow":
        omega0 = t0 * 2 * pi / K  # gam
        s1 = NufftBase(omega=omega0, Nd=N, Kd=K, **nufft_args)
        h = xp.asarray(s1.p[:, 0].todense()).ravel()  # [J*L + 1]
    # This way is "J times faster" than the slow way, but still not ideal.
    # It works for any user-specified interpolator.
    elif how == "fast":
        # odd case set to match result of 'slow' above
        if N % 2 == 0:
            t1 = J / 2.0 - 1 + xp.arange(L) / L  # [L]
        else:
            t1 = J / 2.0 - 1 + xp.arange(1, L + 1) / L  # [L]
        omega1 = t1 * 2 * pi / K  # * gam
        s1 = NufftBase(omega=omega1, Nd=N, Kd=K, **nufft_args)
        try:
            h = xp.asarray(s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(
                order="F"
            )
        except ValueError:
            # fall back to alternate slicing for CuPy
            h = xp.asarray(s1.p[:, slice(0, J)].todense()[:, ::-1]).ravel(
                order="F"
            )
        if N % 2 == 0:
            h = xp.concatenate((h, xp.atleast_1d(h[0])), axis=0)  # [J*L+1,]
        else:
            h = xp.concatenate((xp.atleast_1d(h[-1]), h), axis=0)  # [J*L+1,]
    # This efficient way uses only "J" columns of sparse matrix!
    # The trick to this is to use fake small values for N and K,
    # which works for interpolators that depend only on the ratio K/N.
    elif how == "ratio":
        Nfake = J
        Kfake = Nfake * K / N
        if debug:
            print("N={},J={},K={}".format(N, J, K))
            print("Nfake={},Kfake={}".format(Nfake, Kfake))
        t1 = J / 2.0 - 1 + xp.arange(L) / L  # [L]
        omega1 = t1 * 2 * pi / Kfake  # gam
        s1 = NufftBase(omega=omega1, Nd=Nfake, Kd=Kfake, **nufft_args)
        try:
            h = xp.asarray(s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(
                order="F"
            )
        except ValueError:
            # fall back to alternate slicing for CuPy
            h = xp.asarray(s1.p[:, slice(0, J)].todense()[:, ::-1]).ravel(
                order="F"
            )
        # [J*L+1] assuming symmetric
        h = xp.concatenate((h, xp.asarray([h[0]])), axis=0)
        if phasing == "complex":
            h = h * xp.exp(1j * pi * t0 * (1 / K - 1 / Kfake))  # fix phase
    else:
        raise ValueError("Bad Type: {}".format(type))
    return h, t0


@profile
def nufft_forward(obj, x, copy_x=True, grid_only=False, xp=None):
    """ Forward NUFFT (image-space to k-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    x : array
        DFT values (images) [npixels, ncoils*n_reps]
    copy_x : bool, optional
        make a copy of x internally to avoid potentially modifying the
        original array.

    Returns
    -------
    xk : array
        DTFT coefficients (k-space)
    """
    Nd = obj.Nd
    Kd = obj.Kd
    xp, on_gpu = get_array_module(x, xp)

    if not x.flags.f_contiguous:
        x = xp.asfortranarray(x)
    try:
        # collapse all excess dimensions into just one
        # data_address_in = get_data_address(x)
        if grid_only:
            # x = x.reshape(list(Kd) + [-1, ], order='F')
            x = x.reshape((prod(Kd), -1), order="F")
        else:
            x = x.reshape(list(Nd) + [-1], order="F")
    except ValueError:
        print("Input signal has the wrong size.")
        raise

    # Promote to complex if real input was provided
    x = complexify(x, complex_dtype=obj._cplx_dtype)

    # if copy_x and (data_address_in == get_data_address(x)):
    #     # make sure original array isn't modified!
    #     x = x.copy(order="A")
    x = xp.array(x, order="A", copy=False)

    n_reps = x.shape[-1]

    if not grid_only:
        #
        # the usual case is where n_reps=1, i.e., there is just one input signal.
        #
        if obj.sn is not None:
            xk = x * obj.sn[..., xp.newaxis]  # scaling factors
        else:
            xk = x
        # TODO: cache FFT plans
        if xp is np:
            xk = fftn(xk, Kd, axes=range(x.ndim - 1))
        else:
            axes = tuple(range(x.ndim - 1))

            if hasattr(obj.cufft_plan, "shape"):
                try:
                    if obj.order == "C":
                        expected_shape = Kd
                    elif obj.order == "F":
                        expected_shape = Kd[::-1]
                    if obj.cufft_plan.shape == expected_shape:
                        with obj.cufft_plan:
                            if n_reps == 1:
                                xk = cupy_fftn(
                                    xk, Kd, axes=axes, overwrite_x=True
                                )
                            else:
                                xk_in = xk.copy(order=obj.order)
                                xk = cupy.empty(
                                    Kd + (n_reps,),
                                    dtype=xk_in.dtype,
                                    order=obj.order,
                                )
                                for rep in range(n_reps):
                                    xk[..., rep] = cupy_fftn(
                                        xk_in[..., rep],
                                        Kd,
                                        axes=axes,
                                        overwrite_x=True,
                                    )
                                del xk_in
                    else:
                        raise ValueError("unexpected pre-planned cuFFT shape")
                except ValueError:
                    # workaround until the following CuPy fix is merged:
                    #     https://github.com/cupy/cupy/pull/3034 is merged
                    xk = cupy_fftn(xk, Kd, axes=axes, overwrite_x=True)
            else:
                xk = cupy_fftn(xk, Kd, axes=axes, overwrite_x=True)
        if obj.phase_before is not None:
            xk *= obj.phase_before[..., xp.newaxis]
        xk = xk.reshape((prod(Kd), n_reps), order="F")

        if obj.ortho:
            xk /= obj.scale_ortho
    else:
        xk = x

    if "table" in obj.mode:
        # interpolate via tabulated interpolator
        x = obj.interp_table(obj, xk)
    else:
        # interpolate using precomputed sparse matrix
        if xp is np:
            x = obj.p * xk  # [M, n_reps]
        else:
            # cannot use the notation above because it seems there is a bug in
            # COO->CSC/CSR conversion that causes issues
            # can avoid by calling cusparse.csrmv directly

            # obj.p is CSC so use obj.p.T with transa=True to do a CSR multiplication
            if obj.p_csr is not None:
                if xk.ndim == 1:
                    x = cupy.cusparse.csrmv(
                        obj.p_csr, xk, transa=False
                    )  # [*Kd,  n_reps]
                else:
                    x = cupy.cusparse.csrmm(obj.p_csr, xk, transa=False)
            else:
                if xk.ndim == 1:
                    x = cupy.cusparse.csrmv(
                        obj.p.T, xk, transa=True
                    )  # [*Kd,  n_reps]
                else:
                    x = cupy.cusparse.csrmm(obj.p.T, xk, transa=True)

    x = xp.reshape(x, (obj.M, n_reps), order="F")
    if grid_only:
        return x

    if obj.phase_after is not None:
        x *= obj.phase_after[:, None]  # broadcast rather than np.tile

    remove_singleton = True
    if remove_singleton and n_reps == 1:
        x = x[..., 0]

    return x


# @profile
def nufft_forward_exact(obj, x, copy_x=True, xp=None):
    """ Brute-force forward DTFT (image-space to k-space).

    **Warning:** This is SLOW! Intended primarily for validating NUFFT in
    tests

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    x : array
        DFT values (images) [npixels, ncoils*n_reps]
    copy_x : bool, optional
        make a copy of x internally to avoid potentially modifying the
        original array.

    Returns
    -------
    xk : array
        DTFT coefficients (k-space)
    """
    Nd = obj.Nd
    xp, on_gpu = get_array_module(x, xp)

    if not x.flags.f_contiguous:
        x = xp.asfortranarray(x)

    try:
        # collapse all excess dimensions into just one
        data_address_in = get_data_address(x)
        x = x.reshape(list(Nd) + [-1], order="F")
    except ValueError:
        print("Input signal has the wrong size")
        raise

    # Promote to complex if real input was provided
    x = complexify(x, complex_dtype=obj._cplx_dtype)

    if copy_x and (data_address_in == get_data_address(x)):
        # make sure original array isn't modified!
        x = x.copy(order="A")

    L = x.shape[-1]

    xk = xp.empty((obj.M, L), dtype=x.dtype, order="F")
    for rep in range(L):
        xk[..., rep] = dtft(
            x[:, rep], omega=obj.omega, shape=Nd, n_shift=obj.n_shift, xp=xp
        )

    remove_singleton = True
    if remove_singleton and L == 1:
        xk = xk[..., 0]

    return xk


@profile
def nufft_adj(obj, xk, copy=True, return_psf=False, grid_only=False, xp=None):
    """Adjoint NUFFT (k-space to image-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    xk : array
        DTFT values (k-space) [nsamples, ncoils*n_reps]
    copy : bool, optional
        make a copy of ``xk`` internally to avoid potentially modifying the
        original array.
    return_psf : bool, optional
        EXPERIMENTAL

    Returns
    -------
    x : array
        DFT coefficients
    """
    Nd = obj.Nd
    Kd = obj.Kd
    xp, on_gpu = get_array_module(xk, xp)
    data_address_in = get_data_address(xk)
    if xk.size % obj.M != 0:
        raise ValueError("invalid size")
    if not xk.flags.f_contiguous:
        xk = xp.asfortranarray(xk)
    xk = complexify(xk, complex_dtype=obj._cplx_dtype)  # force complex
    xk = xp.reshape(xk, (obj.M, -1), order="F")  # [M,*L]
    if copy and (data_address_in == get_data_address(xk)):
        # ensure input array isn't modified by in-place operations below
        xk = xk.copy(order="A")

    n_reps = xk.shape[-1]

    if obj.phase_after is not None and not return_psf:
        # replaced np.tile() with broadcasting
        xk *= obj.phase_after.conj()[:, xp.newaxis]

    if "table" in obj.mode:
        # interpolate via tabulated interpolator
        xk = xk.astype(xp.result_type(obj.h[0], xk.dtype), copy=False)
        xk_all = obj.interp_table_adj(obj, xk)
    else:
        # interpolate using precomputed sparse matrix
        if xp is np:
            xk_all = obj.p.H * xk  # [*Kd, *L]
        else:
            # cannot use the notation above because it seems there is a bug in
            # COO->CSC/CSR conversion that causes issues
            # can avoid by calling cusparse.csrmv directly
            if xk.ndim == 1:
                # xk_all will have shape [*Kd, *L]
                xk_all = cupy.cusparse.csrmv(obj.p.H, xk, transa=False)
            else:
                xk_all = cupy.cusparse.csrmm(obj.p.H, xk, transa=False)

    if grid_only:
        # return raw gridded samples prior to FFT and truncation
        return xk_all

    # x = xp.zeros(tuple(Kd) + (n_reps,), dtype=xk.dtype)

    if xk_all.ndim == 1:
        xk_all = xk_all[:, None]

    xk_all = xk_all.reshape(Kd + (n_reps,), order="F")
    if return_psf:
        return xk_all[..., 0]
    if obj.phase_before is not None:
        xk_all *= obj.phase_before.conj()[..., xp.newaxis]
    # TODO: cache FFT plans
    axes = tuple(range(xk_all.ndim - 1))
    if xp is np:
        x = ifftn(xk_all, Kd, axes=axes)
    else:
        if hasattr(obj.cufft_plan, "shape"):
            try:
                if obj.order == "C":
                    expected_shape = Kd
                else:
                    # Plan object has reversed shape for order="F" case
                    expected_shape = Kd[::-1]
                if obj.cufft_plan.shape == expected_shape:
                    with obj.cufft_plan:
                        if n_reps == 1:
                            x = cupy_ifftn(
                                xk_all, Kd, axes=axes, overwrite_x=True
                            )
                        else:
                            x = cupy.empty(
                                Kd + (n_reps,),
                                dtype=xk_all.dtype,
                                order=obj.order,
                            )
                            for rep in range(n_reps):
                                x[..., rep] = cupy_ifftn(
                                    xk_all[..., rep],
                                    Kd,
                                    axes=axes,
                                    overwrite_x=True,
                                )
                else:
                    raise ValueError("unexpected pre-planned cuFFT shape")
            except ValueError:
                # workaround until the following CuPy fix is merged:
                #     https://github.com/cupy/cupy/pull/3034 is merged
                x = cupy_ifftn(xk_all, Kd, axes=axes, overwrite_x=True)
        else:
            x = cupy_ifftn(xk_all, Kd, axes=axes, overwrite_x=True)
    # eliminate zero padding from ends
    subset_slices = tuple([slice(d) for d in Nd] + [slice(None)])
    x = x[subset_slices]

    if obj.ortho:
        x *= obj.scale_ortho * obj.adjoint_scalefactor
    else:
        # undo default normalization of ifftn (as in matlab nufft_adj)
        x *= prod(Kd) * obj.adjoint_scalefactor

    # scaling factors
    if obj.sn is not None:
        x *= xp.conj(obj.sn)[..., xp.newaxis]

    remove_singleton = True
    if remove_singleton and n_reps == 1:
        x = x[..., 0]

    return x


# @profile
def nufft_adj_exact(obj, xk, copy=True, xp=None):
    """ Brute-force adjoint NUFFT (k-space to image-space).

    **Warning:** This is SLOW! Intended primarily for validating NUFFT in
    tests

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    xk : array
        DTFT values (k-space) [nsamples, ncoils*n_reps]
    copy : bool, optional
        make a copy of ``xk`` internally to avoid potentially modifying the
        original array.

    Returns
    -------
    x : array
        DFT coefficients
    """
    Nd = obj.Nd
    xp, on_gpu = get_array_module(xk, xp)

    if not xk.flags.f_contiguous:
        xk = xp.asfortranarray(xk)

    data_address_in = get_data_address(xk)
    if xk.size % obj.M != 0:
        raise ValueError("invalid size")
    xk = complexify(xk, complex_dtype=obj._cplx_dtype)  # force complex
    xk = xp.reshape(xk, (obj.M, -1), order="F")  # [M,*L]
    if copy and (data_address_in == get_data_address(xk)):
        # make sure the original array isn't modified!
        xk = xk.copy(order="A")

    n_reps = xk.shape[-1]

    x = xp.empty((prod(Nd), n_reps), dtype=xk.dtype, order="F")
    for rep in range(n_reps):
        x[..., rep] = dtft_adj(
            xk[:, rep], omega=obj.omega, shape=obj.Nd, n_shift=obj.n_shift
        )

    if obj.ortho:
        x *= obj.scale_ortho * obj.adjoint_scalefactor
    elif obj.adjoint_scalefactor != 1:
        x *= obj.adjoint_scalefactor

    remove_singleton = True
    if remove_singleton and n_reps == 1:
        x = x[..., 0]

    return x
