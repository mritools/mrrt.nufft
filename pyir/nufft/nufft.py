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

import collections
import warnings

from time import time
import numpy as np
from numpy.testing import assert_

import scipy.sparse
from scipy.sparse import coo_matrix

from pyir.nufft.nufft_utils import (_nufft_samples,
                                    _nufft_interp_zn,
                                    _nufft_coef,
                                    _nufft_offset,
                                    to_1d_int_array
                                    )

from pyir.nufft._minmax import (_nufft_r,
                                _nufft_T,
                                nufft_scale)

from pyir.nufft._dtft import dtft, dtft_adj

from pyir.nufft._kaiser_bessel import kaiser_bessel_ft

from pyir.nufft.interp_table import (interp1_table,
                                     interp2_table,
                                     interp3_table,
                                     interp1_table_adj,
                                     interp2_table_adj,
                                     interp3_table_adj)

from pyir.nufft.simple_kernels import _scale_tri

from pyir.utils import (fftn,
                        ifftn,
                        fast_fft_shape,
                        outer_sum,
                        complexify,
                        is_string_like,
                        reale,
                        profile,
                        get_data_address)

from ._kernels import NufftKernel


__all__ = ['NufftBase', 'compute_Q', 'nufft_adj', 'nufft_forward']

supported_real_types = [np.float32, np.float64]
supported_cplx_types = [np.complex64, np.complex128]


def _get_legend_text(ax):
    l = ax.get_legend()
    if l is None:
        return None
    else:
        return [t.get_text() for t in l.get_texts()]


def _block_outer_sum(x1, x2):
    J1, M = x1.shape
    J2, M = x2.shape
    xx1 = np.reshape(x1, (J1, 1, M))
    xx2 = np.reshape(x2, (1, J2, M))
    # use numpy broadcasting
    y = xx1 + xx2			# (J1, J2, M)
    return y


def _block_outer_prod(x1, x2):
    J1, M = x1.shape
    J2, M = x2.shape
    xx1 = np.reshape(x1, (J1, 1, M))
    xx2 = np.reshape(x2, (1, J2, M))
    # use numpy broadcasting
    y = xx1 * xx2			# (J1, J2, M)
    return y

# TODO: change name of NufftBase to NFFT_Base
# TODO: change default n_shift to Nd/2?


class NufftBase(object):
    """Base NUFFT Class"""
    @profile
    def __init__(self, Nd, om, Jd=6, Kd=None, Ld=2048, precision='single',
                 kernel_type='kb:beatty', mode='table0', ortho=False,
                 n_shift=None, phasing='real', sparse_format='CSC',
                 adjoint_scalefactor=1., kernel_kwargs={}, verbose=False):
        """ Initialize NufftBase instance

        Parameters
        ----------
        Nd : array-like
            Shape of the Cartesian grid in the spatial domain.
        om : array
            non-Cartesian sampling frequencies
        J : int or array-like, optional
            size of the NUFFT kernel on each axis
        Kd : array-like, optional
            Oversampled cartesian grid size (default = 2*Nd)
        Ld : array-like, optional
            When ``mode != 'sparse'``, this controls the size of the kernel
            lookup table used.   (size will be ``J[i]*Ld[i]`` for axis i).
        precision : {'single', 'double'}, optional
            precision of the NUFFT operations
        kernel_type : {'kb:beatty', ...}, optional
            The type of gridding kernel to use.  'kb:beatty' is near optimal in
            most cases and works well for grid oversampling factors
            substantially less than 2.
        mode : {'sparse', 'table1', 'table0'}, optional
            The NUFFT implementation to use.  ``sparse`` corresponds to
            precomputation of a sparse matrix corresponding to the operation.
            This requires a longer initialization and uses more memory, but is
            typically faster than the lookup-table based approaches.
            ``table*`` modes use a look-up table based interpolation.  This is
            fast to initialize and uses less memory.
            ``table1`` and ``table0`` correspond to first-order and
            nearest-neighbor interpolation from the gridding lookup table.
        ortho : bool, optional
            TODO

        Additional Parameters
        ---------------------
        n_shift : array-like, optional
            controls the shift applied (e.g. default is like N/2 shift like
            fftshift)
        phasing : {'real', 'complex'}, optional
            If complex, the gridding kernel is complex-valued and the phase
            roll corresponding to ``n_shift`` is built into the kernel.
            If real, a real-valued kernel is used and the phase rolls are
            applied separately.  Performance is similar in either case.
        sparse_format : {'CSC', 'CSR', 'COO', 'LIL', 'DOK'}, optional
            The sparse matrix format used by scipy.sparse for CPU-based
            implementation.  The GPU implementation always uses CSR.
        adjoint_scalefactor : float, optional
            A custom, constant multiplicative scaling factor for the
            adjoint operation.
        kernel_kwargs : dict
            Addtional kwargs to pass along to the NufftKernel object created.
        """
        self.verbose = verbose
        if self.verbose:
            print("Entering NufftBase init")
        self.__init_complete = False  # will be set true after __init__()

        # must set the __ version of these to avoid circular calls by the
        # setters
        self.__Nd = to_1d_int_array(Nd)
        if self.verbose:
            print("Nd={}".format(Nd))
            print("self.__Nd={}".format(self.__Nd))
            print("self.Nd={}".format(self.Nd))

        self.kernel_type = kernel_type
        self.__phasing = phasing
        self.__om = None  # will be set later below
        self._set_Nmid()
        self.ndim = len(self.Nd)  # number of dimensions
        self._Jd = to_1d_int_array(Jd, nelem=self.ndim)

        if Kd is None:
            Kd = 2 * self.__Nd
        self.__Kd = to_1d_int_array(Kd, nelem=self.ndim)

        self.ortho = ortho  # normalization for orthogonal FFT
        if self.ortho:
            self.scale_ortho = np.sqrt(self.__Kd.prod())
        else:
            self.scale_ortho = 1

        # placeholders for phase_before/phase_after.  phasing.setter
        self.phase_before = None
        self.phase_after = None

        # n_shift placeholder
        self.__n_shift = None

        # placeholders for dtypes:  will be set by precision.setter
        self._cplx_dtype = None
        self._real_dtype = None

        self.om = om
        self.precision = precision
        self.__mode = None  # set below by mode.setter()
        self._forw = None
        self._adj = None
        self._init = None
        self.mode = mode  # {'table', 'sparse', 'exact'}
        self.adjoint_scalefactor = adjoint_scalefactor
        kernel_type = kernel_type.lower()
        # [M, *Kd]	sparse interpolation matrix (or empty if table-based)
        self.p = None
        self.Jd = Jd
        if 'mols' in kernel_type:
            if 'table' not in self.mode:
                raise ValueError(
                    'MOLS NUFFT kernel, requires a table-based mode')
            if np.any(np.asarray(Jd) % 2):
                raise ValueError("MOLS only currently working for even J")
            if self.phasing != 'real':
                raise ValueError(
                    'MOLS NUFFT kernel, requires a real phasing')

        self.kernel = NufftKernel(kernel_type,
                                  ndim=self.ndim,
                                  Nd=self.Nd,
                                  Jd=self.Jd,
                                  Kd=self.Kd,
                                  Nmid=self.Nmid,
                                  **kernel_kwargs)
        self._calc_scaling()  # [(Nd)]  scaling factors
        self.M = 0
        if self.om is not None:
            self.M = self.om.shape[0]
        if n_shift is None:
            self.__n_shift = (0,) * self.ndim
        else:
            self.__n_shift = n_shift
        if (self.ndim != len(self.Jd)) or (self.ndim != len(self.Kd)):
            raise ValueError("Inconsistent Dimensions")
        # set the phase to be applied if self.phasing=='real'
        self._set_phase_funcs()
        self.gram = None  # TODO
        self._update_array__precision()
        self._make_arrays_contiguous(order='F')
        # TODO: cleanup how initialization is done
        self.__sparse_format = None
        self.__Ld = None
        if self.mode == 'sparse':
            self._init_sparsemat()  # create COO matrix
            # convert to other format if specified
            if sparse_format is None:
                self.sparse_format = 'COO'
            else:  # convert formats via setter if necessary
                self.sparse_format = sparse_format
        elif 'table' in self.mode:
            # TODO: change name of Ld to table_oversampling
            self.Ld = to_1d_int_array(Ld, nelem=self.ndim)
            odd_L = np.mod(self.Ld, 2) == 1
            odd_J = np.mod(self.Jd, 2) == 1
            if np.any(np.logical_and(odd_L, odd_J)):
                warnings.warn(
                    "accuracy may be compromised when L and J are both odd")
            if self.mode == 'table0':
                self.table_order = 0  # just order in newfft
            elif self.mode == 'table1' or self.mode == 'table':
                self.table_order = 1  # just order in newfft
            else:
                raise ValueError("Invalid NUFFT mode: {}".format(self.mode))
            self._init_table()
            self.interp_table = _nufft_table_interp  # TODO: remove?
            self.interp_table_adj = _nufft_table_adj  # TODO: remove?
        elif self.mode == 'exact':

            # TODO: wrap calls to dtft, dtft_adj
            raise ValueError("not implemented")
            pass
        else:
            raise ValueError("Invalid NUFFT mode: {}".format(self.mode))
        self.fft = self._nufft_forward
        self.adj = self._nufft_adj
        self._update_array__precision()
        self._make_arrays_contiguous(order='F')
        self.__init_complete = True  # TODO: currently unused
        if self.verbose:
            print("Exiting NufftBase init")

    def _nufft_forward(self, x):
        if self.mode == 'exact':
            y = nufft_forward_exact(self, x=x)
        else:
            y = nufft_forward(self, x=x)
        return y

    def _nufft_adj(self, X):
        if self.mode == 'exact':
            y = nufft_adjoint_exact(self, X=X)
        else:
            y = nufft_adj(self, X=X)
        return y

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
            if sparse_format == 'CSC':
                self.p = self.p.tocsc()
            elif sparse_format == 'CSR':
                self.p = self.p.tocsr()
            elif sparse_format == 'COO':
                self.p = self.p.tocoo()
            elif sparse_format == 'LIL':
                self.p = self.p.tolil()
            elif sparse_format == 'DOK':
                self.p = self.p.todok()
            else:
                raise ValueError("unrecognized sparse format type")
        else:
            raise ValueError(
                "no sparse matrix exists.  cannot update sparse" +
                " format for mode: {}".format(self.mode))

    @property
    def precision(self):
        return self.__precision

    @precision.setter
    def precision(self, precision):
        # default precision based on self.om
        if precision in [None, 'auto']:
            if isinstance(self.__om, np.ndarray):
                if self.__om.dtype in [np.float32]:
                    precision = 'single'
                elif self.__om.dtype in [np.float64]:
                    precision = 'double'
            else:
                precision = 'double'
        # set corresponding real and complex types
        if precision == 'single':
            self._cplx_dtype = np.dtype(np.complex64)
            self._real_dtype = np.dtype(np.float32)
        elif precision == 'double':
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

    @om.setter
    def om(self, om):
        if om is not None:
            if is_string_like(om):
                # special test cases of input sampling pattern
                om = _nufft_samples(om, self.Nd)
            om = np.asarray(om)
            if om.ndim == 1:
                om = om[:, np.newaxis]
            if om.shape[1] != self.ndim:
                raise ValueError("number of cols must match NUFFT dimension")
            if om.dtype not in supported_real_types:
                raise ValueError("om must be one of the following types: "
                                 "{}".format(supported_real_types))
            if self.ndim != om.shape[1]:
                raise ValueError('omega needs {} columns'.format(self.ndim))
        self.__om = om
        if isinstance(self.phase_before, np.ndarray):
            self.phase_after = self._phase_after(om, self.Nmid, self.__n_shift)
        if self.__init_complete:
            self._reinitialize()

    def _reinitialize(self):
        """utility to reinitialize the NUFFT object"""
        if self.mode == 'sparse':
            self._init_sparsemat()
        elif 'table' in 'mode':
            self._init_table()

    @property
    def phasing(self):
        return self.__phasing

    @phasing.setter
    def phasing(self, phasing):
        self.__phasing = phasing
        self._set_Nmid()
        self._set_phase_funcs()

    @property
    def Nd(self):
        return self.__Nd

    @Nd.setter
    def Nd(self, Nd):
        K_N_ratio = self.__Kd / self.__Nd
        self.__Nd = to_1d_int_array(Nd, nelem=self.ndim)
        self._set_Nmid()
        # update Kd to maintain approximately the same amount of oversampling
        self.__Kd = np.round(K_N_ratio * self.__Nd).astype(self.__Kd.dtype)
        if self.__init_complete:
            self._reinitialize()

    @property
    def Jd(self):
        return self._Jd

    @Jd.setter
    def Jd(self, Jd):
        self._Jd = to_1d_int_array(Jd, nelem=self.ndim)
        if self.__init_complete:
            self._reinitialize()

    @property
    def Ld(self):
        return self.__Ld

    @Ld.setter
    def Ld(self, Ld):
        self.__Ld = to_1d_int_array(Ld, nelem=self.ndim)
        if 'table' not in self.mode:
            warnings.warn("Ld is ignored for mode = {}".format(self.mode))
        elif self.__init_complete:
            self._reinitialize()

    @property
    def Kd(self):
        return self.__Kd

    @Kd.setter
    def Kd(self, Kd):
        self.__Kd = to_1d_int_array(Kd, nelem=self.ndim)
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = self._phase_before(Kd, self.Nmid)
        if self.__init_complete:
            self._reinitialize()

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        self.__mode = mode
        # TODO: allow changing mode

    @property
    def n_shift(self):
        return self.__n_shift

    @n_shift.setter
    def n_shift(self, n_shift):
        self.__n_shift = np.asarray(n_shift)
        if self.ndim != n_shift.size:
            raise ValueError('n_shift needs %d columns' % (self.ndim))
        self.phase_after = self._phase_after(self.__om, self.Nmid, n_shift)
        if self.__init_complete:
            self._reinitialize()

    def _set_Nmid(self):
        # midpoint of scaling factors
        if self.__phasing == 'real':
            self.Nmid = np.floor(self.Nd / 2.)
        else:
            self.Nmid = (self.Nd - 1) / 2.
        if self.phasing == 'real'and (self.__om is not None):
            self.phase_after = self._phase_after(self.__om,
                                                 self.Nmid, self.__n_shift)
        if self.__init_complete:
            self._reinitialize()

    def _update_dtype(self, arr, mode=None):
        if mode is None:
            if np.iscomplexobj(arr):
                if arr.dtype != self._cplx_dtype:
                    arr = arr.astype(self._cplx_dtype)
            else:
                if arr.dtype != self._real_dtype:
                    arr = arr.astype(self._real_dtype)
        elif mode == 'real':
            if arr.dtype != self._real_dtype:
                arr = arr.astype(self._real_dtype)
        elif mode == 'complex':
            if arr.dtype != self._cplx_dtype:
                arr = arr.astype(self._cplx_dtype)
        else:
            raise ValueError("unrecognized mode")
        return arr

    def _update_array__precision(self):
        # update the data types of other members
        # TODO: warn if losing precision during conversion?
        if isinstance(self.__om, np.ndarray):
            self.__om = self._update_dtype(self.om, 'real')
        if isinstance(self.__n_shift, np.ndarray):
            self.__n_shift = self._update_dtype(self.__n_shift, 'real')
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = self._update_dtype(
                self.phase_before, 'complex')
        if isinstance(self.phase_after, np.ndarray):
            self.phase_after = self._update_dtype(self.phase_after, 'complex')
        if hasattr(self, 'sn') and isinstance(self.sn, np.ndarray):
            self.sn = self._update_dtype(self.sn)
        if self.mode == 'sparse':
            if hasattr(self, 'p') and self.p is not None:
                self.p = self._update_dtype(self.p, self.phasing)
        elif 'table' in self.mode:
            if hasattr(self, 'h') and self.h is not None:
                for idx, h in enumerate(self.h):
                    self.h[idx] = self._update_dtype(h, self.phasing)
        else:
            raise ValueError("unknown mode")

    def _make_arrays_contiguous(self, order='F'):
        if order == 'F':
            contig_func = np.asfortranarray
        elif order == 'C':
            contig_func = np.ascontiguousarray
        else:
            raise ValueError("order must be 'F' or 'C'")
        self.__om = contig_func(self.__om)
        self.__Kd = contig_func(self.__Kd)
        self.__Nd = contig_func(self.__Nd)
        self._Jd = contig_func(self._Jd)
        self.__n_shift = contig_func(self.__n_shift)
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = contig_func(self.phase_before)
        if isinstance(self.phase_after, np.ndarray):
            self.phase_after = contig_func(self.phase_after)
        if hasattr(self, 'sn') and self.sn is not None:
            self.sn = contig_func(self.sn)
        if self.mode == 'sparse':
            pass
        if 'table' in self.mode:
            if hasattr(self, 'h') and self.h is not None:
                for h in self.h:
                    h = contig_func(h)

    def _set_phase_funcs(self):
        if self.phasing == 'real':
            # TODO: fix 'mols' kernel in complex case to incorporate this?
            self.phase_before = self._phase_before(self.Kd, self.Nmid)
            self.phase_after = self._phase_after(self.om,
                                                 self.Nmid,
                                                 self.n_shift)
        elif self.phasing == 'complex':
            # complex kernel incorporates the FFTshift phase
            self.phase_before = None
            self.phase_after = None
        else:
            raise ValueError("Invalid phasing: {}\n\t".format(self.phasing) +
                             "must be 'real' or 'complex'")

    def _phase_before(self, Kd, Nmid):
        """Needed to realize desired FFT shift for real-valued NUFFT kernel.
        """
        phase = 2 * np.pi * np.arange(Kd[0]) / Kd[0] * Nmid[0]
        for d in range(1, Kd.size):
            tmp = 2 * np.pi * np.arange(Kd[d]) / Kd[d] * Nmid[d]
            # fast outer sum via broadcasting
            phase = phase.reshape(
                (phase.shape) + (1,)) + tmp.reshape((1,) * d + (tmp.size,))
        return np.exp(1j * phase).astype(self._cplx_dtype)  # [(Kd)]

    def _phase_after(self, om, Nmid, n_shift):
        """Needed to realize desired FFT shift for real-valued NUFFT kernel.
        """
        phase = np.exp(1j * np.dot(om, (n_shift - Nmid).reshape(-1, 1)))
        return np.squeeze(phase).astype(self._cplx_dtype)  # [M,1]

    def _calc_scaling(self):
        """image domain scaling factors to account for the finite NUFFT
        kernel.  (i.e. intensity rolloff correction)
        """
        kernel = self.kernel
        Nd = self.Nd
        Kd = self.Kd
        Jd = self.Jd
        ktype = kernel.kernel_type.lower()
        if ktype == 'diric':
            self.sn = np.ones(Nd)
        elif 'minmax:' in ktype:
            self.sn = nufft_scale(Nd, Kd, kernel.alpha, kernel.beta, self.Nmid)
        elif 'mols' in ktype:
            self.sn = None
            # will get set later during _init_table()
        else:
            self.sn = np.array([1.])
            for d in range(self.ndim):
                if kernel.is_kaiser_scale:
                    # nc = np.arange(Nd[d])-(Nd[d]-1)/2.  #OLD WAY
                    nc = np.arange(Nd[d]) - self.Nmid[d]
                    tmp = 1 / kaiser_bessel_ft(nc / Kd[d], Jd[d],
                                               kernel.kb_alf[d],
                                               kernel.kb_m[d], 1)
                elif ktype == 'inline':
                    if self.phasing == 'real':
                        warnings.warn("not sure if this is correct for real "
                                      "phasing case (Nmid is set differently)")
                    tmp = 1 / _nufft_interp_zn(0, Nd[d], Jd[d], Kd[d],
                                               kernel.kernel[d],
                                               self.Nmid[d])
                elif ktype == 'linear':
                    # TODO: untested
                    tmp = _scale_tri(Nd[d], Jd[d], Kd[d], self.Nmid[d])
                else:
                    raise ValueError("Unsupported ktype: {}".format(ktype))
                # tmp = reale(tmp)  #TODO: reale?
                # TODO: replace outer with broadcasting?
                self.sn = np.outer(self.sn.ravel(), tmp.conj())
        if 'mols' not in ktype:
            if len(Nd) > 1:
                self.sn = self.sn.reshape(tuple(Nd))  # [(Nd)]
            else:
                self.sn = self.sn.ravel()  # [(Nd)]

    @profile
    def _init_sparsemat(self):
        """Initialize structure for n-dimensional NUFFT using Sparse matrix
        multiplication.
        """
        tstart = time()
        ud = {}
        kd = {}
        om = self.om
        if om.ndim == 1:
            om = om[:, np.newaxis]

        if self.phasing == 'real':
            # call again just to be safe in case Kd, Nmid, etc changed?
            self._set_phase_funcs()

        for d in range(self.ndim):
            N = self.Nd[d]
            J = self.Jd[d]
            K = self.Kd[d]

            # callable kernel:  kaiser, linear, etc
            if (self.kernel.kernel is not None):
                kernel_func = self.kernel.kernel[d]
                if not isinstance(kernel_func, collections.Callable):
                    raise ValueError("callable kernel function required")
                # [J?,M]
                [c, arg] = _nufft_coef(om[:, d], J, K, kernel_func)
            else:  # minmax:
                alpha = self.kernel.alpha[d]
                beta = self.kernel.beta[d]
                # [J?,J?]  TODO: move .tol into kernel object
                T = _nufft_T(N, J, K, tol=1e-7, alpha=alpha, beta=beta)
                [r, arg] = _nufft_r(
                    om[:, d], N, J, K, alpha=alpha, beta=beta)  # [J?,M]
                # c = T * r  clear T r
                c = np.dot(T, r)
            #
            # indices into oversampled FFT components
            #
            # [M,1] to leftmost near nbr
            koff = _nufft_offset(om[:, d], J, K)

            # [J,M]
            kd[d] = np.mod(outer_sum(np.arange(1, J + 1), koff), K)

            if self.phasing == 'complex':
                gam = 2 * np.pi / K
                phase_scale = 1j * gam * (N - 1) / 2.
                phase = np.exp(phase_scale * arg)   # [J,M] linear phase
            else:
                phase = 1.
            # else:
            #     raise ValueError("Unknown phasing {}".format(self.phasing))

            ud[d] = phase * c      # [J?,M]

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
            Jprod = np.prod(self.Jd[0:d + 1])
            # trick: pre-convert these indices into offsets! (Fortran order)
            tmp = kd[d] * np.prod(self.Kd[:d])
            kk = _block_outer_sum(kk, tmp)  # outer sum of indices
            kk = kk.reshape(Jprod, M, order='F')
            uu = _block_outer_prod(uu, ud[d])  # outer product of coefficients
            uu = uu.reshape(Jprod, M, order='F')
        # now kk and uu are shape (*Jd, M)

        #
        # apply phase shift
        # pre-do Hermitian transpose of interpolation coefficients
        #
        if np.iscomplexobj(uu):
            uu = uu.conj()

        if self.phasing == 'complex':
            if np.any(self.n_shift != 0):
                phase = np.exp(1j * np.dot(om, self.n_shift.ravel()))			# [1,M]
                phase = phase.reshape((1, -1), order='F')
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
                print('NUFFT sparse matrix storage will require ' +
                      '%g GB' % (RAM_GB))

        # shape (*Jd, M)
        mm = np.tile(np.arange(M), (np.product(self.Jd), 1))

        self.p = coo_matrix((uu.ravel(order='F'),
                             (mm.ravel(order='F'), kk.ravel(order='F'))),
                            shape=(M, self.Kd.prod()), dtype=sparse_dtype)
        tend2 = time()
        if self.verbose:
            print("Sparse init stage 2 duration = {} s".format(tend2-tend1))

    def _init_table(self):
        """Initialize structure for n-dimensional NUFFT using table-based
        interpolator.
        """
        # for convenience
        ndim = self.ndim
        # need to strip ndim, Nd, Jd, Kd from local copy of kernel_kwargs
        kernel_kwargs = self.kernel.params.copy()
        kernel_kwargs.pop('ndim', None)
        kernel_kwargs.pop('Nd', None)
        kernel_kwargs.pop('Jd', None)
        kernel_kwargs.pop('Kd', None)
        kernel_kwargs.pop('Nmid', None)
        # if ('kb:' in self.kernel.kernel_type):
        # how = 'ratio'  #currently a bug in ratio case for non-integer K/N
        #     else:
        how = 'fast'
        if self.phasing == 'complex' and \
                np.any(np.asarray(self.n_shift) != 0):
            self.phase_shift = np.exp(
                1j * np.dot(self.om, self.n_shift.ravel()))  # [M 1]
        else:
            self.phase_shift = None  # compute on-the-fly
        if self.Ld is None:
            if self.table_order == 0:
                self.Ld = 2 ** 11
            elif self.table_order == 1:
                self.Ld = 2 ** 9
            else:
                raise ValueError("Bad table mode")
        if ndim != len(self.Jd) or ndim != len(self.Ld) or \
                ndim != len(self.Kd):
            raise ValueError('inconsistent dimensions among ndim, Jd, Ld, Kd')
        if ndim != self.om.shape[1]:
            raise ValueError('omega needs %d columns' % (ndim))

        self.h = []
        # build kernel lookup table (LUT) for each dimension
        for d in range(ndim):
            if 'alpha' in kernel_kwargs:
                kernel_kwargs['alpha'] = [self.kernel.params['alpha'][d], ]
                kernel_kwargs['beta'] = [self.kernel.params['beta'][d], ]
            if 'kb_alf' in kernel_kwargs:
                kernel_kwargs['kb_alf'] = [self.kernel.params['kb_alf'][d], ]
                kernel_kwargs['kb_m'] = [self.kernel.params['kb_m'][d], ]

            if 'mols' in self.kernel.kernel_type:
                if d == 0:
                    # dict to cache previously generated kernels
                    MOLS_generated = {}

                # TODO: test this IOWA case
                from pyir.nufft._iowa_MOLSkernel import PreNUFFT_fm
                if self.Ld[d] % 2 == 0:
                    raise ValueError("mols requires odd Ld")
                key = (self.Jd[d], self.Nd[d], self.Ld[d], self.Kd[d])

                if key in MOLS_generated:
                    pre, h = MOLS_generated[key]
                else:
                    pre, h, junk1, err1 = PreNUFFT_fm(
                        J=self.Jd[d], N=self.Nd[d], Ofactor=self.Ld[d],
                        K=self.Kd[d], Order=2, H=np.ones(self.Nd[d]),
                        degree=self.Jd[d]-1, realkernel=True)
                    # store for reuse in case other dimensions are the same
                    MOLS_generated[key] = (pre, h)

                if d == ndim - 1:
                    del MOLS_generated

                if len(h) != (self.Jd[d] * self.Ld[d] + 1):
                    raise ValueError("unexpected kernel size")
                # scale factor computation
                if d == 0:
                    self.sn = 1
                pre_shape = np.ones(self.ndim, dtype=np.intp)
                pre_shape[d] = len(pre)
                # if negligable imaginary component, keep only the real part
                try:
                    pre = reale(pre)
                except ValueError:
                    pass
                self.sn = self.sn * pre.reshape(pre_shape, order='F')
                h = np.abs(h)  # TODO: take abs here?
                # h /= h.max() # can't renormalize unless prefilter also scaled

#                self.sn = np.outer(self.sn.ravel(), pre.conj())
#                if d == ndim - 1:
#                    if len(Nd) > 1:
#                        self.sn = self.sn.reshape(tuple(Nd))  # [(Nd)]
#                    else:
#                        self.sn = self.sn.ravel()  # [(Nd)]

            else:
                if self.kernel.kernel_type in ['kb:minmax', 'kb:beatty']:
                    # avoid warnings in kernel calls during _nufft_table_make1
                    kernel_kwargs.pop('kb_m', None)
                    kernel_kwargs.pop('kb_alf', None)
                h, t0 = _nufft_table_make1(how=how, N=self.Nd[d], J=self.Jd[d],
                                           K=self.Kd[d], L=self.Ld[d],
                                           phasing=self.phasing,
                                           kernel_type=self.kernel.kernel_type,
                                           kernel_kwargs=kernel_kwargs)

            if self.phasing == 'complex':
                if np.isrealobj(h):
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
        attribs = [item for item in attribs if not item.startswith('__')]
        str = ''
        for attrib in attribs:
            val = getattr(self, attrib, None)
            if isinstance(val, np.ndarray):
                str += "{} = ndarray: dtype={}, shape={}\n".format(
                    attrib,
                    val.dtype,
                    val.shape)
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
        fig, axes = plt.subplots(self.ndim, 1, sharex='col',
                                 gridspec_kw=gridspec_kw)
        for d in range(self.ndim):
            if self.mode != 'sparse':
                x = np.linspace(-self.Jd[d]/2, self.Jd[d]/2, self.h[d].size)
                y = self.h[d]
            else:
                if self.kernel.kernel is not None:
                    if with_phasing:
                        raise ValueError(
                            "with_phasing option only supported for "
                            "table-based NUFFT")
                    x = np.linspace(-self.Jd[d]/2, self.Jd[d]/2, 1000)
                    y = self.kernel.kernel[d](x, self.Jd[d])
                else:
                    print("Kernel is not an inline function. will not be"
                          "plotted")
            axes[d].plot(x, np.abs(y), 'k-', label='magnitude')
            if with_phasing:
                axes[d].plot(x, y.real, 'k--', label='real')
                axes[d].plot(x, y.imag, 'k:', label='imag')
                axes[d].legend()
            axes[d].set_ylabel('axis %d' % d)
            if d == self.ndim-1:
                axes[d].set_xlabel('oversampled grid offset')
        return fig, axes


@profile
def _nufft_table_interp(obj, Xk, om=None):
    """ Forward NUFFT based on kernel lookup table (image-space to k-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    Xk : array
        DFT values (images) [npixels, ncoils*nrepetitions]
    copy_x : bool, optional
        make a copy of x internally to avoid potentially modifying the
        original array.

    Returns
    -------
    X : array
        DTFT coefficients (k-space)
    """
    order = obj.table_order

    if om is None:
        om = obj.om

    ndim = len(obj.Kd)

    tm = np.zeros_like(om)
    pi = np.pi
    for d in range(0, ndim):
        gam = 2 * pi / obj.Kd[d]
        tm[:, d] = om[:, d] / gam  # t = omega / gamma

    if Xk.ndim == 1:
        Xk = Xk[:, np.newaxis]
    elif Xk.shape[1] > Xk.shape[0]:
        Xk = Xk.T
    nc = Xk.shape[1]

    if Xk.shape[0] != np.product(obj.Kd):
        raise ValueError('Xk size problem')

    Xk = complexify(Xk, complex_dtype=obj._cplx_dtype)  # force complex

    arg = [obj.Jd, obj.Ld, tm, order]
    if ndim == 1:
        X = interp1_table(Xk, obj.h[0], *arg)
    elif ndim == 2:
        # Fortran ordering to match Matlab behavior
        Xk = np.reshape(Xk, np.hstack((obj.Kd, nc)), order='F')
        X = interp2_table(Xk, obj.h[0], obj.h[1], *arg)
    elif ndim == 3:
        # Fortran ordering to match Matlab behavior
        Xk = np.reshape(Xk, np.hstack((obj.Kd, nc)), order='F')
        X = interp3_table(Xk, obj.h[0], obj.h[1], obj.h[2], *arg)
    else:
        raise ValueError('dimensions > 3d not done')

    # apply phase shift
    if hasattr(obj, 'phase_shift'):
        if isinstance(obj.phase_shift, (np.ndarray, list)):
            if len(obj.phase_shift) > 0:
                ph = obj.phase_shift
                if ph.ndim == 1:
                    ph = np.reshape(ph, (-1, 1))
                if X.shape[0] != ph.shape[0] or X.ndim != ph.ndim:
                    raise RuntimeError("dimension mismatch")
                X *= ph

    return X.astype(Xk.dtype)


@profile
def _nufft_table_adj(obj, X, om=None):
    """ Adjoint NUFFT based on kernel lookup table (k-space to image-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    X : array
        DTFT values (k-space) [nsamples, ncoils*nrepetitions]
    om : ndarray, optional
        Frequency locations corresponding to the samples.  By default this is
        obtained from obj.

    Returns
    -------
    Xk : array
        DFT coefficients
    """
    order = obj.table_order
    if om is None:
        om = obj.om

    ndim = len(obj.Kd)

    tm = np.zeros_like(om)
    pi = np.pi
    for d in range(0, ndim):
        gam = 2 * pi / obj.Kd[d]
        tm[:, d] = om[:, d] / gam  # t = omega / gamma

    if X.shape[0] != om.shape[0]:
        raise ValueError('X size problem')

    if X.ndim is 1:
        X = X[:, np.newaxis]

    # adjoint of phase shift
    if hasattr(obj, 'phase_shift'):
        if isinstance(obj.phase_shift, (np.ndarray, list)):
            if len(obj.phase_shift) > 0:
                ph_conj = obj.phase_shift.conj()
                if ph_conj.ndim == 1:
                    ph_conj = np.reshape(ph_conj, (-1, 1))
                if X.shape[0] != ph_conj.shape[0] or X.ndim != ph_conj.ndim:
                    raise RuntimeError("dimension mismatch")
                X *= ph_conj

    # force proper complex dtype
    X = complexify(X, complex_dtype=obj._cplx_dtype)

    arg = [obj.Jd, obj.Ld, tm, obj.Kd[0:ndim], order]

    if ndim == 1:
        Xk = interp1_table_adj(X, obj.h[0], *arg)
    elif ndim == 2:
        Xk = interp2_table_adj(X, obj.h[0], obj.h[1], *arg)
    elif ndim == 3:
        Xk = interp3_table_adj(X, obj.h[0], obj.h[1], obj.h[2], *arg)
    else:
        raise ValueError('> 3d not done')

    return Xk.astype(X.dtype)


@profile
def _nufft_table_make1(
        how, N, J, K, L, kernel_type, phasing, debug=False, kernel_kwargs={}):
    """ make LUT for 1 dimension by creating a dummy 1D NUFFT object """
    nufft_args = dict(Jd=J, n_shift=0, kernel_type=kernel_type,
                      kernel_kwargs=kernel_kwargs,
                      mode='sparse',
                      phasing=phasing,
                      sparse_format='csc')
    t0 = np.arange(-J * L / 2., J * L / 2. + 1) / L  # [J*L+1]
    assert_(t0.size == (J*L + 1))
    pi = np.pi
    if N % 2 == 0:
        # may be a slight symmetry error for odd N?
        # warnings.warn("odd N in _nufft_table_make1.  even N recommended.")
        pass

    # This is a slow and inefficient (but simple) way to get the table
    # because it builds a huge sparse matrix but only uses 1 column!
    if how == 'slow':
        om0 = t0 * 2 * pi / K  # gam
        s1 = NufftBase(om=om0, Nd=N, Kd=K, **nufft_args)
        h = np.asarray(s1.p[:, 0].todense()).ravel()  # [J*L + 1]
    # This way is "J times faster" than the slow way, but still not ideal.
    # It works for any user-specified interpolator.
    elif how == 'fast':
        # odd case set to match result of 'slow' above
        if N % 2 == 0:
            t1 = J / 2. - 1 + np.arange(L) / L  # [L]
        else:
            t1 = J / 2. - 1 + np.arange(1, L+1) / L  # [L]
        om1 = t1 * 2 * pi / K		# * gam
        s1 = NufftBase(om=om1, Nd=N, Kd=K, **nufft_args)
        h = np.asarray(
            s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(order='F')
        if N % 2 == 0:
            h = np.concatenate((h, np.asarray([h[0], ])), axis=0)  # [J*L+1,]
        else:
            h = np.concatenate((np.asarray([h[-1], ]), h), axis=0)  # [J*L+1,]
    # This efficient way uses only "J" columns of sparse matrix!
    # The trick to this is to use fake small values for N and K,
    # which works for interpolators that depend only on the ratio K/N.
    elif how == 'ratio':  # e.g., 'minmax:kb' | 'kb:*'
        Nfake = J
        Kfake = Nfake * K / N
        if debug:
            print("N={},J={},K={}".format(N, J, K))
            print("Nfake={},Kfake={}".format(Nfake, Kfake))
        t1 = J / 2. - 1 + np.arange(L) / L  # [L]
        om1 = t1 * 2 * pi / Kfake		# gam
        s1 = NufftBase(om=om1, Nd=Nfake, Kd=Kfake, **nufft_args)
        h = np.asarray(
            s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(order='F')
        # [J*L+1] assuming symmetric
        h = np.concatenate((h, np.asarray([h[0], ])), axis=0)
        if phasing == 'complex':
            # TODO: fix 'mols' case
            h = h * np.exp(1j * pi * t0 * (1 / K - 1 / Kfake))  # fix phase
    else:
        raise ValueError("Bad Type: {}".format(type))
    return h, t0


@profile
def nufft_forward(obj, x, copy_x=True):
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
    X : array
        DTFT coefficients (k-space)
    """
    Nd = obj.Nd
    Kd = obj.Kd
    try:
        # collapse all excess dimensions into just one
        data_address_in = get_data_address(x)
        x = x.reshape(list(Nd) + [-1, ], order='F')
    except:
        raise ValueError('input signal has wrong size')

    # Promote to complex if real input was provided
    x = complexify(x, complex_dtype=obj._cplx_dtype)

    if copy_x and (data_address_in == get_data_address(x)):
        # make sure original array isn't modified!
        x = x.copy()

    L = x.shape[-1]
    #
    # the usual case is where L=1, i.e., there is just one input signal.
    #
    if obj.sn is not None:
        x *= obj.sn[..., np.newaxis]		# scaling factors
    Xk = fftn(x, Kd, axes=range(x.ndim-1))
    if obj.phase_before is not None:
        Xk *= obj.phase_before[..., np.newaxis]
    Xk = Xk.reshape((np.product(Kd), L), order='F')

    if obj.ortho:
        Xk /= obj.scale_ortho

    if 'table' in obj.mode:
        # interpolate via tabulated interpolator
        X = obj.interp_table(obj, Xk)
    else:
        # interpolate using precomputed sparse matrix
        X = obj.p * Xk  # [M,*L]

    X = np.reshape(X, (obj.M, L), order='F')

    if obj.phase_after is not None:
        X *= obj.phase_after[:, None]  # broadcast rather than np.tile

    remove_singleton = True
    if remove_singleton and L == 1:
        X = X[..., 0]

    return X


@profile
def nufft_forward_exact(obj, x, copy_x=True):
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
    X : array
        DTFT coefficients (k-space)
    """
    Nd = obj.Nd

    try:
        # collapse all excess dimensions into just one
        data_address_in = get_data_address(x)
        x = x.reshape(list(Nd) + [-1, ], order='F')
    except:
        raise ValueError('input signal has wrong size')

    # Promote to complex if real input was provided
    x = complexify(x, complex_dtype=obj._cplx_dtype)

    if copy_x and (data_address_in == get_data_address(x)):
        # make sure original array isn't modified!
        x = x.copy()

    L = x.shape[-1]

    X = np.empty((obj.M, L), dtype=x.dtype, order='F')
    for rep in range(L):
        X[..., rep] = dtft(
            x[:, rep], omega=obj.om, Nd=Nd, n_shift=obj.n_shift)

    remove_singleton = True
    if remove_singleton and L == 1:
        X = X[..., 0]

    return X


@profile
def nufft_adj(obj, X, copy_X=True, return_psf=False):
    """Adjoint NUFFT (k-space to image-space).

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    X : array
        DTFT values (k-space) [nsamples, ncoils*nrepetitions]
    copy_X : bool, optional
        make a copy of x internally to avoid potentially modifying the
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
    data_address_in = get_data_address(X)
    if X.size % obj.M != 0:
        raise ValueError("invalid size")
    X = complexify(X, complex_dtype=obj._cplx_dtype)  # force complex
    X = np.reshape(X, (obj.M, -1), order='F')  # [M,*L]
    if copy_X and (data_address_in == get_data_address(X)):
        # ensure input array isn't modified by in-place operations below
        X = X.copy()

    nrepetitions = X.shape[-1]

    if obj.phase_after is not None and not return_psf:
        # replaced np.tile() with broadcasting
        X *= obj.phase_after.conj()[:, np.newaxis]

    if 'table' in obj.mode:
        # interpolate via tabulated interpolator
        X = X.astype(np.result_type(obj.h[0], X.dtype), copy=False)
        Xk_all = obj.interp_table_adj(obj, X)
    else:
        # interpolate using precomputed sparse matrix
        Xk_all = (obj.p.H * X)  # [*Kd,*L]

    x = np.zeros(tuple(Kd) + (nrepetitions,), dtype=X.dtype)

    if Xk_all.ndim == 1:
        Xk_all = Xk_all[:, None]

    Xk_all = Xk_all.reshape(tuple(Kd) + (nrepetitions, ), order='F')
    if return_psf:
        return Xk_all[..., 0]
    if obj.phase_before is not None:
        Xk_all *= obj.phase_before.conj()[..., np.newaxis]
    x = ifftn(Xk_all, axes=range(Xk_all.ndim-1))

    # eliminate zero padding from ends
    subset_slices = [slice(d) for d in Nd] + [slice(None), ]
    x = x[subset_slices]

    if obj.ortho:
        # TODO: even if obj.ortho?
        x *= (obj.scale_ortho * obj.adjoint_scalefactor)
    else:
        # undo default normalization of ifftn (as in matlab nufft_adj)
        x *= (np.product(Kd) * obj.adjoint_scalefactor)

    # scaling factors
    if obj.sn is not None:
        x *= np.conj(obj.sn)[..., np.newaxis]

    remove_singleton = True
    if remove_singleton and nrepetitions == 1:
        x = x[..., 0]

    return x


def nufft_adjoint_exact(obj, X, copy_X=True):
    """ Brute-force adjoint NUFFT (k-space to image-space).

    **Warning:** This is SLOW! Intended primarily for validating NUFFT in
    tests

    Parameters
    ----------
    obj : NufftBase object
        instance of NufftBase (contains k-space locations, kernel, etc.)
    X : array
        DTFT values (k-space) [nsamples, ncoils*nrepetitions]
    copy_x : bool, optional
        make a copy of x internally to avoid potentially modifying the
        original array.

    Returns
    -------
    x : array
        DFT coefficients
    """
    # extract attributes from structure
    Nd = obj.Nd

    data_address_in = get_data_address(X)
    if X.size % obj.M != 0:
        raise ValueError("invalid size")
    X = complexify(X, complex_dtype=obj._cplx_dtype)  # force complex
    X = np.reshape(X, (obj.M, -1), order='F')  # [M,*L]
    if copy_X and (data_address_in == get_data_address(X)):
        # make sure the original array isn't modified!
        X = X.copy()

    nrepetitions = X.shape[-1]

    x = np.empty((np.prod(Nd), nrepetitions), dtype=X.dtype, order='F')
    for rep in range(nrepetitions):
        x[..., rep] = dtft_adj(X[:, rep], omega=obj.om, Nd=obj.Nd,
                               n_shift=obj.n_shift)

    if obj.ortho:
        x *= (obj.scale_ortho * obj.adjoint_scalefactor)
    elif obj.adjoint_scalefactor != 1:
        x *= obj.adjoint_scalefactor

    remove_singleton = True
    if remove_singleton and nrepetitions == 1:
        x = x[..., 0]

    return x


def compute_Q(G, wi=None, Nd_os=2, Kd_os=1.35, J=5, **extra_nufft_kwargs):
    """compute Q such that IFFT(Q*FFT(x)) = (G.H * G * x).

    Notes
    -----
    requires that G.Kd ~= 2*G.Nd for good accuracy.
    can get away with Kd_os < substantially less than 2

    References
    ----------
    ..[1] Wajer FTAW, Pruessmann KP. Major Speedup of Reconstruction for
    Sensitivity Encoding with Arbitrary Trajectories.
    Proc. Intl. Soc. Mag. Reson. Med. 9 (2001), p.767.

    ..[2] Eggers H, Boernert P, Boesiger P.  Comparison of Gridding- and
    Convolution-Based Iterative Reconstruction Algorithms For
    Sensitivity-Encoded Non-Cartesian Acquisitions.
    Proc. Intl. Soc. Mag. Reson. Med. 10 (2002)

    ..[3] Liu C, Moseley ME, Bammer R.  Fast SENSE Reconstruction Using Linear
    System Transfer Function.
    Proc. Intl. Soc. Mag. Reson. Med. 13 (2005), p.689.
    """
    from pyir.operators_private import MRI_Operator, NUFFT_Operator

    if isinstance(G, NUFFT_Operator):
        Gnufft_op = G
    elif isinstance(G, MRI_Operator):
        Gnufft_op = G.Gnufft
    else:
        raise ValueError("G must be an NUFFT_Operator or MRI_Operator")

    # need reasonably accurate gridding onto a 2x oversampled grid
    Nd = (Nd_os * Gnufft_op.Nd).astype(np.intp)
    Kd = fast_fft_shape((Kd_os*Nd).astype(np.intp))
    if np.any(G.Kd < 2*G.Nd):
        warnings.warn("Q operator unlikely to be accurate.  Recommend using G "
                      "with a grid oversampling factor of 2")
    if Nd_os != 2:
        warnings.warn("recommend keeping Nd_os=2")

    G2 = NUFFT_Operator(om=Gnufft_op.om,
                        Nd=Nd,
                        Kd=Kd,
                        Jd=(J, )*len(Nd),
                        Ld=Gnufft_op.Ld,
                        n_shift=Nd/2,
                        kernel_type=Gnufft_op.kernel.kernel_type,
                        mode=Gnufft_op.mode,
                        phasing='real',  # ONLY WORKS IF THIS IS REAL!
                        **extra_nufft_kwargs)

    if wi is None:
        wi = np.ones(Gnufft_op.om.shape[0], dtype=Gnufft_op._cplx_dtype)

    psft = G2.H * wi
    # TODO: allow DiagonalOperator too for weights
    psft = np.fft.fftshift(psft.reshape(G2.Nd, order=G2.order))
    return fftn(psft)


def compute_Q_v2(G, copy_X=True):
    """Alternative version of compute_Q.

    experimental:  not recommended over compute_Q()
    """
    from pyir.nufft.nufft import nufft_adj
    ones = np.ones(G.kspace.shape[0], G.Gnufft._cplx_dtype)
    sf = np.sqrt(np.prod(G.Gnufft.Kd))
    return sf * fftn(nufft_adj(G.Gnufft, ones, copy_X=True, return_psf=True))
