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

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
from math import sqrt
from time import time
import warnings

import numpy as np
import scipy.sparse

from ._dtft import dtft, dtft_adj
from ._fft_cpu import fftn, ifftn
from ._interp_table import (
    interp1_table,
    interp2_table,
    interp3_table,
    interp1_table_adj,
    interp2_table_adj,
    interp3_table_adj,
)
from ._kaiser_bessel import kaiser_bessel_ft
from ._kernels import NufftKernel
from .nufft_utils import (
    _nufft_samples,
    _nufft_interp_zn,
    _nufft_coef,
    _nufft_offset,
    complexify,
    get_array_module,
    get_data_address,
    is_string_like,
    outer_sum,
    profile,
    reale,
    _as_1d_ints,
)
from ._simple_kernels import _scale_tri
from . import config

if config.have_cupy:
    import cupy
    import cupyx.scipy.sparse
    from ._cupy import default_device, get_1D_block_table_gridding
    from cupyx.scipy import fftpack as cupy_fftpack

    cupy_fftn = cupy_fftpack.fftn
    cupy_ifftn = cupy_fftpack.ifftn
    try:
        cupy.fft.cache.enable()
    except AttributeError:
        # cache not implemented in official CuPy release
        pass


__all__ = [
    "NufftBase",
    "nufft_adj",
    "nufft_forward",
    "nufft_adj_exact",
    "nufft_forward_exact",
]

supported_real_types = [np.float32, np.float64]
supported_cplx_types = [np.complex64, np.complex128]

# TODO: use more pythonic names for Kd, Jd, Ld, Nd, etc
#       Nd -> shape
#       Kd -> gridded_shape
#       Ld -> table_sizes
#       Jd -> kernel_sizes
#       om -> omega


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


# TODO: change default n_shift to Nd/2?


class NufftBase(object):
    """Base NUFFT Class"""

    @profile
    def __init__(
        self,
        Nd,
        om,
        Jd=4,
        Kd=None,
        Ld=None,
        precision="single",
        mode="table",
        kernel_type="kb:beatty",
        ortho=False,
        n_shift=None,
        phasing="real",
        sparse_format="CSC",
        adjoint_scalefactor=1.0,
        kernel_kwargs={},
        verbose=False,
        on_gpu=False,
    ):
        """ Initialize NufftBase instance

        Parameters
        ----------
        Nd : array-like
            Shape of the Cartesian grid in the spatial domain.
        om : array
            Non-Cartesian sampling frequencies. (TODO: units)
        J : int or array-like, optional
            Size of the NUFFT kernel on each axis. For GPU-based NUFFT, it is
            currently required to use the same size on each axis.
        Kd : array-like, optional
            Oversampled cartesian grid size (default is ``1.5 * Nd``).
        Ld : array-like, optional
            When ``mode != 'sparse'``, this controls the size of the kernel
            lookup table used. The table size will be ``J[i] * Ld[i]`` for axis
            ``i``.
        precision : {'single', 'double', 'auto'}, optional
            Precision of the NUFFT operations. If 'auto' the precision will be
            set to match the provided ``om`` array.
        mode : {'sparse', 'table'}, optional
            The NUFFT implementation to use. ``sparse`` corresponds to
            precomputation of a sparse matrix corresponding to the operation.
            This requires a longer initialization and uses more memory, but is
            typically faster than the lookup-table based approaches.
            ``table`` uses a look-up table based interpolation.  This is
            fast to initialize and uses less memory.

        Additional Parameters
        ---------------------
        n_shift : array-like, optional
            Controls the shift applied (e.g. default is an N/2 shift like
            fftshift)
        ortho : bool, optional
            If True, an orthogonal FFT with normalization factor (1/sqrt(n)) is
            used in each direction. Otherwise normalization 1/n is applied
            during the inverse FFT. (TODO: double check this)
        phasing : {'real', 'complex'}, optional
            If complex, the gridding kernel is complex-valued and the phase
            roll corresponding to ``n_shift`` is built into the kernel.
            If real, a real-valued kernel is used and the phase rolls are
            applied separately.  Performance is similar in either case.
        sparse_format : {'CSC', 'CSR', 'COO', 'LIL', 'DOK'}, optional
            The sparse matrix format used by scipy.sparse for CPU-based
            implementation. The GPU implementation always uses CSR.
        adjoint_scalefactor : float, optional
            A custom, constant multiplicative scaling factor for the
            adjoint operation.
        kernel_type : {'kb:beatty', ...}, optional
            The type of gridding kernel to use.  'kb:beatty' is near optimal in
            most cases and works well for grid oversampling factors
             substantially less than 2.
        kernel_kwargs : dict
            Addtional kwargs to pass along to the NufftKernel object created.
        """
        self.verbose = verbose
        if self.verbose:
            print("Entering NufftBase init")
        self.__init_complete = False  # will be set true after __init__()

        self.__on_gpu = on_gpu
        # get the array module being used (based on state of self.on_gpu)

        # set the private version (__Nd not Nd) to avoid circular calls
        # also Nd, Kd, Jd, etc. should be on the CPU
        self.__Nd = _as_1d_ints(Nd, xp=np)
        self.kernel_type = kernel_type
        self.__phasing = phasing
        self.__om = None  # will be set later below
        self._set_n_mid()
        self.ndim = len(self.Nd)  # number of dimensions
        self._Jd = _as_1d_ints(Jd, n=self.ndim, xp=np)

        if Kd is None:
            Kd = 1.5 * self.__Nd
        self.__Kd = _as_1d_ints(Kd, n=self.ndim, xp=np)

        # normalization for orthogonal FFT
        self.ortho = ortho
        self.scale_ortho = sqrt(self.__Kd.prod()) if self.ortho else 1

        # placeholders for phase_before/phase_after.  phasing.setter
        self.phase_before = None
        self.phase_after = None

        # n_shift placeholder
        self.__n_shift = None

        # placeholders for dtypes:  will be set by precision.setter
        self._cplx_dtype = None
        self._real_dtype = None

        self._init_omega(
            om
        )  # TODO: force om to be an array. don't allow 'epi', etc.

        self.precision = precision
        self._forw = None
        self._adj = None
        self._init = None
        self.__mode = mode
        self.adjoint_scalefactor = adjoint_scalefactor
        kernel_type = kernel_type.lower()
        # [M, *Kd]	sparse interpolation matrix (or empty if table-based)
        self.p = None
        self.Jd = Jd
        self.kernel = NufftKernel(
            kernel_type,
            ndim=self.ndim,
            Nd=self.Nd,
            Jd=self.Jd,
            Kd=self.Kd,
            n_mid=self.n_mid,
            **kernel_kwargs,
        )
        self._calc_scaling()
        self.M = 0
        if self.om is not None:
            self.M = self.om.shape[0]
        if n_shift is None:
            self.__n_shift = (0,) * self.ndim  # TODO: set to self.Nd //2?
        else:
            self.__n_shift = n_shift
        if (self.ndim != len(self.Jd)) or (self.ndim != len(self.Kd)):
            raise ValueError("Inconsistent Dimensions")

        # set the phase to be applied if self.phasing=='real'
        self._set_phase_funcs()

        self.gram = None  # TODO
        self._update_array__precision()
        self._make_arrays_contiguous(order="F")

        # TODO: cleanup how initialization is done
        self.__sparse_format = None
        if Ld is None:
            Ld = 1024
        self.__Ld = _as_1d_ints(Ld, n=self.ndim, xp=np)
        if self.mode == "sparse":
            self._init_sparsemat()  # create COO matrix
            # convert to other format if specified
            if sparse_format is None:
                self.sparse_format = "COO"
            else:  # convert formats via setter if necessary
                self.sparse_format = sparse_format
        elif "table" in self.mode:
            # TODO: change name of Ld to table_oversampling
            self.Ld = _as_1d_ints(Ld, n=self.ndim, xp=np)
            odd_L = np.mod(self.Ld, 2) == 1
            odd_J = np.mod(self.Jd, 2) == 1
            if np.any(np.logical_and(odd_L, odd_J)):
                warnings.warn(
                    "accuracy may be compromised when L and J are both odd"
                )
            self._init_table()
            self.interp_table = _nufft_table_interp  # TODO: remove?
            self.interp_table_adj = _nufft_table_adj  # TODO: remove?
        elif self.mode == "exact":
            # TODO: wrap calls to dtft, dtft_adj
            raise ValueError("not implemented")
            pass
        else:
            raise ValueError("Invalid NUFFT mode: {}".format(self.mode))
        self.fft = self._nufft_forward
        self.adj = self._nufft_adj
        self._update_array__precision()
        self._make_arrays_contiguous(order="F")
        self.__init_complete = True  # TODO: currently unused
        if self.verbose:
            print("Exiting NufftBase init")
        if self.on_gpu:
            self._init_gpu()

    def _init_gpu(self):
        from mrrt.nufft.cuda.cupy import _get_gridding_funcs

        M = self.om.shape[0]
        if not len(np.unique(self.Jd)):
            raise ValueError(
                "GPU case requires same gridding kernel size on each axis."
            )
        if not len(np.unique(self.Ld)):
            raise ValueError(
                "GPU case requires same gridding table size on each axis."
            )

        # compile GPU kernels corresponding to this operator
        self.kern_forward, self.kern_adj = _get_gridding_funcs(
            Kd=self.Kd,
            M=self.M,
            J=self.Jd[0],
            L=self.Ld[0],
            is_complex_kernel=(self.phasing == "complex"),
            precision=self.precision,
        )

        # store a block and grid configuration for use with the kernel
        self.block, self.grid = get_1D_block_table_gridding(
            M, dev=default_device, kernel=self.kern_adj
        )

    @property
    def on_gpu(self):
        """Boolean indicating whether the filterbank is a GPU filterbank."""
        return self.__on_gpu

    @on_gpu.setter
    def on_gpu(self, on_gpu):
        if on_gpu in [True, False]:
            self.__on_gpu = on_gpu
        else:
            raise ValueError("on_gpu must be True or False")

    @property
    def xp(self):
        """Return the ndarray backend being used."""
        # Do not store the module as an attribute so pickling is possible.
        if self.__on_gpu:
            if not config.have_cupy:
                raise ValueError("CuPy is required for the GPU implementation.")
            return cupy
        else:
            return np

    def _nufft_forward(self, x):
        if self.mode == "exact":
            y = nufft_forward_exact(self, x=x)
        else:
            y = nufft_forward(self, x=x)
        return y

    def _nufft_adj(self, x):
        if self.mode == "exact":
            y = nufft_adj_exact(self, xk=x)
        else:
            y = nufft_adj(self, xk=x)
        return y

    def _set_phase_funcs(self):
        self._set_n_mid()
        if self.phasing == "real":
            self.phase_before = self._phase_before(self.Kd, self.n_mid)
            self.phase_after = self._phase_after(
                self.om, self.n_mid, self.n_shift
            )
        elif self.phasing == "complex":
            # complex kernel incorporates the FFTshift phase
            self.phase_before = None
            self.phase_after = None
        else:
            raise ValueError(
                "Invalid phasing: {}\n\t".format(self.phasing)
                + "must be 'real' or 'complex'"
            )

    @property
    def sparse_format(self):
        return self.__sparse_format

    @sparse_format.setter
    def sparse_format(self, sparse_format):
        """ convert sparse matrix to one of: {'CSC', 'CSR', 'COO', 'LIL',
        'DOK'} """
        sparse_format = sparse_format.upper()
        self.__sparse_format = sparse_format.upper()
        if self.p is not None:
            if sparse_format == "CSC":
                # works on GPU too now that cupy/cupy#2410 was merged
                self.p = self.p.tocsc()
            elif sparse_format == "CSR":
                # works on GPU too now that cupy/cupy#2410 was merged:
                self.p = self.p.tocsr()
            elif sparse_format == "COO":
                self.p = self.p.tocoo()
            elif sparse_format == "LIL":
                self.p = self.p.tolil()
            elif sparse_format == "DOK":
                self.p = self.p.todok()
            else:
                raise ValueError("unrecognized sparse format type")
        else:
            raise ValueError(
                "no sparse matrix exists.  cannot update sparse"
                + " format for mode: {}".format(self.mode)
            )

    @property
    def precision(self):
        return self.__precision

    @precision.setter
    def precision(self, precision):
        # default precision based on self.om
        if precision in [None, "auto"]:
            if isinstance(self.__om, np.ndarray):
                if self.__om.dtype in [np.float32]:
                    precision = "single"
                elif self.__om.dtype in [np.float64]:
                    precision = "double"
            else:
                precision = "double"
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
    def om(self):
        return self.__om

    # disallow setting om after object creation
    # @om.setter
    # def om(self, om):
    #     self._init_omega(om)

    def _init_omega(self, om):
        xp = self.xp
        if om is not None:
            if is_string_like(om):
                # special test cases of input sampling pattern
                om = _nufft_samples(om, self.Nd, xp=xp)
            om = self.xp.asarray(om)
            if om.ndim == 1:
                om = om[:, np.newaxis]
            if om.shape[1] != self.ndim:
                raise ValueError("number of cols must match NUFFT dimension")
            if om.dtype not in supported_real_types:
                raise ValueError(
                    "om must be one of the following types: "
                    "{}".format(supported_real_types)
                )
            if self.ndim != om.shape[1]:
                raise ValueError("omega needs {} columns".format(self.ndim))
        self.__om = om
        if isinstance(self.phase_before, np.ndarray):
            self.phase_after = self._phase_after(om, self.n_mid, self.__n_shift)
        if self.__init_complete:
            self._reinitialize()

    def _reinitialize(self):
        """utility to reinitialize the NUFFT object"""
        if self.mode == "sparse":
            self._init_sparsemat()
        elif "table" in "mode":
            self._init_table()

    @property
    def phasing(self):
        return self.__phasing

    @phasing.setter
    def phasing(self, phasing):
        self.__phasing = phasing
        self._set_n_mid()
        self._set_phase_funcs()

    @property
    def Nd(self):
        return self.__Nd

    @Nd.setter
    def Nd(self, Nd):
        K_N_ratio = self.__Kd / self.__Nd
        self.__Nd = _as_1d_ints(Nd, n=self.ndim, xp=np)
        self._set_n_mid()
        # update Kd to maintain approximately the same amount of oversampling
        self.__Kd = np.round(K_N_ratio * self.__Nd).astype(self.__Kd.dtype)
        if self.__init_complete:
            self._reinitialize()

    @property
    def Jd(self):
        return self._Jd

    @Jd.setter
    def Jd(self, Jd):
        self._Jd = _as_1d_ints(Jd, n=self.ndim, xp=np)
        if self.on_gpu and len(set(self._Jd)) > 1:
            raise ValueError("per-axis Jd not supported on the GPU")
        if self.__init_complete:
            self._reinitialize()

    @property
    def Ld(self):
        return self.__Ld

    @Ld.setter
    def Ld(self, Ld):
        self.__Ld = _as_1d_ints(Ld, n=self.ndim, xp=np)
        if "table" not in self.mode:
            warnings.warn("Ld is ignored for mode = {}".format(self.mode))
        elif self.on_gpu and len(set(self.__Ld)) > 1:
            raise ValueError("per-axis Ld not supported on the GPU")
        elif self.__init_complete:
            self._reinitialize()

    @property
    def Kd(self):
        return self.__Kd

    @Kd.setter
    def Kd(self, Kd):
        self.__Kd = _as_1d_ints(Kd, n=self.ndim, xp=np)
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = self._phase_before(Kd, self.n_mid)
        if self.__init_complete:
            self._reinitialize()

    @property
    def mode(self):
        return self.__mode

    # @mode.setter
    # def mode(self, mode):
    #     self.__mode = mode
    #     if self.__init_complete:
    #         self._reinitialize()

    @property
    def n_shift(self):
        return self.__n_shift

    @n_shift.setter
    def n_shift(self, n_shift):
        self.__n_shift = np.asarray(n_shift)
        if self.ndim != n_shift.size:
            raise ValueError("n_shift needs %d columns" % (self.ndim))
        self.phase_after = self._phase_after(self.__om, self.n_mid, n_shift)
        if self.__init_complete:
            self._reinitialize()

    def _set_n_mid(self):
        # midpoint of scaling factors
        if self.__phasing == "real":
            self.n_mid = np.floor(self.Nd / 2.0)
        else:
            self.n_mid = (self.Nd - 1) / 2.0
        if self.phasing == "real" and (self.__om is not None):
            self.phase_after = self._phase_after(
                self.__om, self.n_mid, self.__n_shift
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
        if isinstance(self.__om, xp.ndarray):
            self.__om = self._update_dtype(self.om, "real", xp=xp)
        if isinstance(self.__n_shift, np.ndarray):
            self.__n_shift = self._update_dtype(self.__n_shift, "real", xp=np)
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
        # arrays potentially stored on the GPU use contig_func_xp
        if order == "F":
            contig_func = np.asfortranarray
            contig_func_xp = xp.asfortranarray
        elif order == "C":
            contig_func = np.asfortranarray
            contig_func_xp = xp.ascontiguousarray
        else:
            raise ValueError("order must be 'F' or 'C'")
        self.__om = contig_func_xp(self.__om)
        self.__Kd = contig_func(self.__Kd)
        self.__Nd = contig_func(self.__Nd)
        self._Jd = contig_func(self._Jd)
        self.__n_shift = contig_func(self.__n_shift)
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = contig_func_xp(self.phase_before)
        if isinstance(self.phase_after, np.ndarray):
            self.phase_after = contig_func_xp(self.phase_after)
        if hasattr(self, "sn") and self.sn is not None:
            self.sn = contig_func_xp(self.sn)
        if self.mode == "sparse":
            pass
        if "table" in self.mode:
            if hasattr(self, "h") and self.h is not None:
                for h in self.h:
                    h = contig_func_xp(h)

    def _phase_before(self, Kd, n_mid):
        """Needed to realize desired FFT shift for real-valued NUFFT kernel.
        """
        xp = self.xp
        phase = 2 * np.pi * xp.arange(Kd[0]) / Kd[0] * n_mid[0]
        for d in range(1, Kd.size):
            tmp = 2 * np.pi * xp.arange(Kd[d]) / Kd[d] * n_mid[d]
            # fast outer sum via broadcasting
            phase = phase.reshape((phase.shape) + (1,)) + tmp.reshape(
                (1,) * d + (tmp.size,)
            )
        return xp.exp(1j * phase).astype(self._cplx_dtype)  # [(Kd)]

    def _phase_after(self, om, n_mid, n_shift):
        """Needed to realize desired FFT shift for real-valued NUFFT kernel.
        """
        xp = self.xp
        phase = xp.exp(
            1j * xp.dot(om, xp.asarray(n_shift - n_mid).reshape(-1, 1))
        )
        return xp.squeeze(phase).astype(self._cplx_dtype)  # [M,1]

    def _calc_scaling(self):
        """image domain scaling factors to account for the finite NUFFT
        kernel.  (i.e. intensity rolloff correction)
        """
        xp = self.xp
        kernel = self.kernel
        Nd = self.Nd
        Kd = self.Kd
        Jd = self.Jd
        ktype = kernel.kernel_type.lower()
        if ktype == "diric":
            self.sn = xp.ones(Nd)
        else:
            self.sn = xp.array([1.0])
            for d in range(self.ndim):
                if kernel.is_kaiser_scale:
                    # nc = np.arange(Nd[d])-(Nd[d]-1)/2.  #OLD WAY
                    nc = xp.arange(Nd[d]) - self.n_mid[d]
                    tmp = 1 / kaiser_bessel_ft(
                        nc / Kd[d], Jd[d], kernel.kb_alf[d], kernel.kb_m[d], 1
                    )
                elif ktype == "inline":
                    if self.phasing == "real":
                        warnings.warn(
                            "not sure if this is correct for real "
                            "phasing case (n_mid is set differently)"
                        )
                    tmp = 1 / _nufft_interp_zn(
                        0,
                        Nd[d],
                        Jd[d],
                        Kd[d],
                        kernel.kernel[d],
                        self.n_mid[d],
                        xp=xp,
                    )
                elif ktype == "linear":
                    # TODO: untested
                    tmp = _scale_tri(Nd[d], Jd[d], Kd[d], self.n_mid[d], xp=xp)
                else:
                    raise ValueError("Unsupported ktype: {}".format(ktype))
                # tmp = reale(tmp)  #TODO: reale?
                # TODO: replace outer with broadcasting?
                self.sn = xp.outer(self.sn.ravel(), tmp.conj())
        if len(Nd) > 1:
            self.sn = self.sn.reshape(tuple(Nd))  # [(Nd)]
        else:
            self.sn = self.sn.ravel()  # [(Nd)]

    @profile
    def _init_sparsemat(self, highmem=None):
        """Initialize structure for n-dimensional NUFFT using Sparse matrix
        multiplication.
        """
        tstart = time()
        ud = {}
        kd = {}
        om = self.om
        if om.ndim == 1:
            om = om[:, np.newaxis]

        # TODO: kernel() call below doesn't currently support CuPy
        xp, on_gpu = get_array_module(om)
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
            kernel_func = self.kernel.kernel[d]
            if not isinstance(kernel_func, Callable):
                raise ValueError("callable kernel function required")

            # [J?,M]
            [c, arg] = _nufft_coef(om[:, d], J, K, kernel_func, xp=xp)

            # indices into oversampled FFT components
            #
            # [M,1] to leftmost near nbr
            koff = _nufft_offset(om[:, d], J, K, xp=xp)

            # [J,M]
            kd[d] = xp.mod(outer_sum(xp.arange(1, J + 1), koff, xp=xp), K)

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
        M = self.om.shape[0]
        kk = kd[0]  # [J1,M]
        uu = ud[0]  # [J1,M]

        for d in range(1, self.ndim):
            Jprod = np.prod(self.Jd[0 : d + 1])
            # trick: pre-convert these indices into offsets! (Fortran order)
            tmp = kd[d] * np.prod(self.Kd[:d])
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
            if np.any(self.n_shift != 0):
                phase = xp.exp(
                    1j * xp.dot(om, xp.asarray(self.n_shift.ravel()))
                )  # [1,M]
                phase = phase.reshape((1, -1), order="F")
                uu *= phase  # use broadcasting along first dimension
            sparse_dtype = self._cplx_dtype
        # elif self.phasing == 'real' or self.phasing is None:
        else:
            sparse_dtype = self._real_dtype
        # else:
        #    raise ValueError("Invalid phasing: {}".format(self.phasing))

        if self.ndim >= 3:  # TODO: move elsewhere
            RAM_GB = self.Jd.prod() * M * sparse_dtype.itemsize / 10 ** 9
            if self.verbose:
                print(
                    "NUFFT sparse matrix storage will require "
                    + "%g GB" % (RAM_GB)
                )

        # shape (*Jd, M)
        mm = xp.tile(xp.arange(M), (int(np.prod(self.Jd)), 1))

        self.p = coo_matrix(
            (uu.ravel(order="F"), (mm.ravel(order="F"), kk.ravel(order="F"))),
            shape=(M, int(self.Kd.prod())),
            dtype=sparse_dtype,
        )

        # return in CSC format by default
        if self.on_gpu:
            self.p = self.p.tocsc()  # works on GPU after fix in cupy/cupy#2410
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
        if self.phasing == "complex" and np.any(np.asarray(self.n_shift) != 0):
            self.phase_shift = xp.exp(
                1j * xp.dot(self.om, xp.asarray(self.n_shift.ravel()))
            )  # [M 1]
        else:
            self.phase_shift = None  # compute on-the-fly
        if self.Ld is None:
            self.Ld = 2 ** 10
        if ndim != len(self.Jd) or ndim != len(self.Ld) or ndim != len(self.Kd):
            raise ValueError("inconsistent dimensions among ndim, Jd, Ld, Kd")
        if ndim != self.om.shape[1]:
            raise ValueError("omega needs %d columns" % (ndim))

        self.h = []
        # build kernel lookup table (LUT) for each dimension
        for d in range(ndim):
            if (
                "kb_alf" in kernel_kwargs
                and kernel_kwargs["kb_alf"] is not None
            ):
                kernel_kwargs["kb_alf"] = [self.kernel.params["kb_alf"][d]]
                kernel_kwargs["kb_m"] = [self.kernel.params["kb_m"][d]]

            h, t0 = _nufft_table_make1(
                how=how,
                N=self.Nd[d],
                J=self.Jd[d],
                K=self.Kd[d],
                L=self.Ld[d],
                phasing=self.phasing,
                kernel_type=self.kernel.kernel_type,
                kernel_kwargs=kernel_kwargs,
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
def _nufft_table_interp(obj, xk, om=None, xp=None):
    """ Forward NUFFT based on kernel lookup table (image-space to k-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    xk : array
        DFT values (images) [npixels, ncoils*nrepetitions]
    copy_x : bool, optional
        make a copy of x internally to avoid potentially modifying the
        original array.

    Returns
    -------
    x : array
        DTFT coefficients (k-space)
    """
    if om is None:
        om = obj.om

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

    tm = xp.zeros_like(om)
    pi = np.pi
    for d in range(0, ndim):
        gam = 2 * pi / obj.Kd[d]
        tm[:, d] = om[:, d] / gam  # t = omega / gamma

    if xk.ndim == 1:
        xk = xk[:, xp.newaxis]
    elif xk.shape[1] > xk.shape[0]:
        xk = xk.T
    nc = xk.shape[1]

    if xk.shape[0] != np.prod(obj.Kd):
        raise ValueError("xk size problem")

    xk = complexify(xk, complex_dtype=obj._cplx_dtype)  # force complex

    if not obj.on_gpu:
        arg = [obj.Jd, obj.Ld, tm]
        if ndim == 1:
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
            # obj.kern_forward(obj.block, obj.grid, args)
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

    return x.astype(xk.dtype)


@profile
def _nufft_table_adj(obj, x, om=None, xp=None):
    """ Adjoint NUFFT based on kernel lookup table (k-space to image-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    x : array
        DTFT values (k-space) [nsamples, ncoils*nrepetitions]
    om : ndarray, optional
        Frequency locations corresponding to the samples.  By default this is
        obtained from obj.

    Returns
    -------
    xk : array
        DFT coefficients
    """
    if om is None:
        om = obj.om
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

    tm = xp.zeros_like(om)
    pi = np.pi
    for d in range(0, ndim):
        gam = 2 * pi / obj.Kd[d]
        tm[:, d] = om[:, d] / gam  # t = omega / gamma

    if x.shape[0] != om.shape[0]:
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
        xk = xp.zeros(
            (int(np.prod(obj.Kd)), nreps), dtype=obj._cplx_dtype, order="F"
        )
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
                    args = (
                        xk[:, r],
                        obj.h[0],
                        obj.h[1],
                        obj.h[2],
                        tm,
                        x[:, r],
                    )
                # obj.kern_adj(obj.block, obj.grid, args)
                obj.kern_adj(obj.grid, obj.block, args)

    return xk.astype(x.dtype)


@profile
def _nufft_table_make1(
    how, N, J, K, L, kernel_type, phasing, debug=False, kernel_kwargs={}, xp=np
):
    """ make LUT for 1 dimension by creating a dummy 1D NUFFT object """
    nufft_args = dict(
        Jd=J,
        n_shift=0,
        kernel_type=kernel_type,
        kernel_kwargs=kernel_kwargs,
        mode="sparse",
        phasing=phasing,
        sparse_format="csc",
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
        om0 = t0 * 2 * pi / K  # gam
        s1 = NufftBase(om=om0, Nd=N, Kd=K, **nufft_args)
        h = xp.asarray(s1.p[:, 0].todense()).ravel()  # [J*L + 1]
    # This way is "J times faster" than the slow way, but still not ideal.
    # It works for any user-specified interpolator.
    elif how == "fast":
        # odd case set to match result of 'slow' above
        if N % 2 == 0:
            t1 = J / 2.0 - 1 + xp.arange(L) / L  # [L]
        else:
            t1 = J / 2.0 - 1 + xp.arange(1, L + 1) / L  # [L]
        om1 = t1 * 2 * pi / K  # * gam
        s1 = NufftBase(om=om1, Nd=N, Kd=K, **nufft_args)
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
        om1 = t1 * 2 * pi / Kfake  # gam
        s1 = NufftBase(om=om1, Nd=Nfake, Kd=Kfake, **nufft_args)
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
        DFT values (images) [npixels, ncoils*nrepetitions]
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
    try:
        # collapse all excess dimensions into just one
        data_address_in = get_data_address(x)
        if grid_only:
            # x = x.reshape(list(Kd) + [-1, ], order='F')
            x = x.reshape((np.prod(Kd), -1), order="F")
        else:
            x = x.reshape(list(Nd) + [-1], order="F")
    except ValueError:
        print("Input signal has the wrong size.")
        raise

    # Promote to complex if real input was provided
    x = complexify(x, complex_dtype=obj._cplx_dtype)

    if copy_x and (data_address_in == get_data_address(x)):
        # make sure original array isn't modified!
        x = x.copy(order="A")

    L = x.shape[-1]

    if not grid_only:
        #
        # the usual case is where L=1, i.e., there is just one input signal.
        #
        if obj.sn is not None:
            x *= obj.sn[..., xp.newaxis]  # scaling factors
        # TODO: cache FFT plans
        if xp == np:
            xk = fftn(x, tuple(Kd), axes=range(x.ndim - 1))
        else:
            xk = cupy_fftn(
                x, tuple(Kd), axes=tuple(range(x.ndim - 1)), overwrite_x=True
            )

        if obj.phase_before is not None:
            xk *= obj.phase_before[..., xp.newaxis]
        xk = xk.reshape((np.prod(Kd), L), order="F")

        if obj.ortho:
            xk /= obj.scale_ortho
    else:
        xk = x

    if "table" in obj.mode:
        # interpolate via tabulated interpolator
        x = obj.interp_table(obj, xk)
    else:
        # interpolate using precomputed sparse matrix
        if xp == np:
            x = obj.p * xk  # [M,*L]
        else:
            # cannot use the notation above because it seems there is a bug in
            # COO->CSC/CSR conversion that causes issues
            # can avoid by calling cusparse.csrmv directly

            # obj.p is CSC so use obj.p.T with transa=True to do a CSR multiplication
            if obj.p_csr is not None:
                if xk.ndim == 1:
                    x = cupy.cusparse.csrmv(
                        obj.p_csr, xk, transa=False
                    )  # [*Kd, *L]
                else:
                    x = cupy.cusparse.csrmm(obj.p_csr, xk, transa=False)
            else:
                if xk.ndim == 1:
                    x = cupy.cusparse.csrmv(
                        obj.p.T, xk, transa=True
                    )  # [*Kd, *L]
                else:
                    x = cupy.cusparse.csrmm(obj.p.T, xk, transa=True)

    x = xp.reshape(x, (obj.M, L), order="F")
    if grid_only:
        return x

    if obj.phase_after is not None:
        x *= obj.phase_after[:, None]  # broadcast rather than np.tile

    remove_singleton = True
    if remove_singleton and L == 1:
        x = x[..., 0]

    return x


@profile
def nufft_forward_exact(obj, x, copy_x=True, xp=None):
    """ Brute-force forward DTFT (image-space to k-space).

    **Warning:** This is SLOW! Intended primarily for validating NUFFT in
    tests

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    x : array
        DFT values (images) [npixels, ncoils*nrepetitions]
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
            x[:, rep], omega=obj.om, Nd=Nd, n_shift=obj.n_shift, xp=xp
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
        DTFT values (k-space) [nsamples, ncoils*nrepetitions]
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
    xk = complexify(xk, complex_dtype=obj._cplx_dtype)  # force complex
    xk = xp.reshape(xk, (obj.M, -1), order="F")  # [M,*L]
    if copy and (data_address_in == get_data_address(xk)):
        # ensure input array isn't modified by in-place operations below
        xk = xk.copy(order="A")

    nrepetitions = xk.shape[-1]

    if obj.phase_after is not None and not return_psf:
        # replaced np.tile() with broadcasting
        xk *= obj.phase_after.conj()[:, xp.newaxis]

    if "table" in obj.mode:
        # interpolate via tabulated interpolator
        xk = xk.astype(xp.result_type(obj.h[0], xk.dtype), copy=False)
        xk_all = obj.interp_table_adj(obj, xk)
    else:
        # interpolate using precomputed sparse matrix
        if xp == np:
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

    # x = xp.zeros(tuple(Kd) + (nrepetitions,), dtype=xk.dtype)

    if xk_all.ndim == 1:
        xk_all = xk_all[:, None]

    xk_all = xk_all.reshape(tuple(Kd) + (nrepetitions,), order="F")
    if return_psf:
        return xk_all[..., 0]
    if obj.phase_before is not None:
        xk_all *= obj.phase_before.conj()[..., xp.newaxis]
    # TODO: cache FFT plans
    if xp == np:
        x = ifftn(xk_all, axes=range(xk_all.ndim - 1))
    else:
        x = cupy_ifftn(
            xk_all, axes=tuple(range(xk_all.ndim - 1)), overwrite_x=True
        )

    # eliminate zero padding from ends
    subset_slices = tuple([slice(d) for d in Nd] + [slice(None)])
    x = x[subset_slices]

    if obj.ortho:
        x *= obj.scale_ortho * obj.adjoint_scalefactor
    else:
        # undo default normalization of ifftn (as in matlab nufft_adj)
        x *= np.prod(Kd) * obj.adjoint_scalefactor

    # scaling factors
    if obj.sn is not None:
        x *= xp.conj(obj.sn)[..., xp.newaxis]

    remove_singleton = True
    if remove_singleton and nrepetitions == 1:
        x = x[..., 0]

    return x


def nufft_adj_exact(obj, xk, copy=True, xp=None):
    """ Brute-force adjoint NUFFT (k-space to image-space).

    **Warning:** This is SLOW! Intended primarily for validating NUFFT in
    tests

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    xk : array
        DTFT values (k-space) [nsamples, ncoils*nrepetitions]
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

    data_address_in = get_data_address(xk)
    if xk.size % obj.M != 0:
        raise ValueError("invalid size")
    xk = complexify(xk, complex_dtype=obj._cplx_dtype)  # force complex
    xk = xp.reshape(xk, (obj.M, -1), order="F")  # [M,*L]
    if copy and (data_address_in == get_data_address(xk)):
        # make sure the original array isn't modified!
        xk = xk.copy(order="A")

    nrepetitions = xk.shape[-1]

    x = xp.empty((np.prod(Nd), nrepetitions), dtype=xk.dtype, order="F")
    for rep in range(nrepetitions):
        x[..., rep] = dtft_adj(
            xk[:, rep], omega=obj.om, Nd=obj.Nd, n_shift=obj.n_shift
        )

    if obj.ortho:
        x *= obj.scale_ortho * obj.adjoint_scalefactor
    elif obj.adjoint_scalefactor != 1:
        x *= obj.adjoint_scalefactor

    remove_singleton = True
    if remove_singleton and nrepetitions == 1:
        x = x[..., 0]

    return x
