from __future__ import division, print_function, absolute_import

from time import time
import warnings
import numpy as np
import collections
try:
    from matplotlib import pyplot as plt
    from pyvolplot import subplot_stack
except:
    warnings.warn("matplotlib not found.  won't be able to plot")

try:
    import scipy.sparse
except:
    # most cases don't need scipy
    pass

try:
    import pyfftw
    has_pyfftw = True
    fftn = pyfftw.interfaces.numpy_fft.fftn
    ifftn = pyfftw.interfaces.numpy_fft.ifftn
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError as e:
    try:
        warnings.warn("pyFFTW not found.  will try to use mklfft instead.")
        has_pyfftw = False
        import mklfft
        fftn = mklfft.fftpack.fftn
        ifftn = mklfft.fftpack.ifftn
    except ImportError as e:
        warnings.warn("pyFFTW & mklfft not found.  using numpy FFTs instead.")
        # Numpy's n-dimensional FFT routines may be using MKL, so prefered over scipy
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn


# TODO: move this to a global configuration

from scipy.sparse import coo_matrix
# import scipy.fftpack  #faster than numpy.fft

from PyIRT.nufft.nufft_utils import (_nufft_samples,
                                     nufft_alpha_kb_fit,
                                     nufft_best_alpha,
                                     nufft_scale,
                                     _nufft_interp_zn,
                                     _nufft_coef,
                                     _nufft_r,
                                     _nufft_T,
                                     _nufft_offset,
                                     nufft_diric,
                                     )

from grl_utils import (outer_sum, complexify)

from PyIRT.nufft.kaiser_bessel import kaiser_bessel, kaiser_bessel_ft


from PyIRT.nufft.interp_table import (interp1_table,
                                      interp2_table,
                                      interp3_table)

from PyIRT.nufft.interp_table_adj import (interp1_table_adj,
                                          interp2_table_adj,
                                          interp3_table_adj)

from grl_utils import is_string_like


def _to_1d_int_array(arr, nelem=None, dtype_out=np.intp):
    """ convert to 1D integer array.  returns an error if the elements of arr
    aren't an integer type or arr has more than one non-singleton dimension.
    """
    arr = np.atleast_1d(arr)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            raise ValueError("dimensions of arr cannot exceed 1")
    # arr.dtype.kind in ['i','u']
    if not issubclass(arr.dtype.type, np.integer):
        # float only OK if values are integers
        if not np.all(np.mod(arr, 1) == 0):
            print("arr = {}".format(arr))
            raise ValueError("arr contains non-integer values")
    if nelem is not None:
        if arr.size != nelem:
            if arr.size == 1:
                arr = np.asarray([arr[0], ] * nelem)
            else:
                raise ValueError(
                    "array did not have the expected size of {}".format(nelem))

    return arr.astype(dtype_out)


def _scale_tri(N, J, K, Nmid):
    """
    scale factors when kernel is 'linear'
    tri(u/J) <-> J sinc^2(J x)
    """
    # TODO: test this one
    nc = np.arange(N, dtype=np.float64) - Nmid
    fun = lambda x: J * np.sinc(J * x / K) ** 2
    cent = fun(nc)
    sn = 1 / cent

    # try the optimal formula
    tmp = 0
    LL = 3
    for ll in range(-LL, LL + 1):
        tmp += np.abs(fun(nc - ll * K)) ** 2
    sn = cent / tmp
    return sn

supported_real_types = [np.float32, np.float64]
supported_cplx_types = [np.complex64, np.float128]

"""
See:
http://stackoverflow.com/questions/7019643/overriding-properties-in-python

And then subclasses can override a single setter/getter like this:

class C(object):
    def __init__(self):
        self._x = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    def printx(self):  #just to test calling function from parent class
        print(self.x)

class C2(C):
    @C.x.setter
    def x(self, x):
        C.printx(self)
        C.x.fset(self, x * 2)
        C.printx(self)



c2=C2(); c2.x=5; c2.x


This is a little warty because overriding multiple methods seems to require
you to do something like:

class C3(C):
    @C.x.getter
    def x(self):
        return self._x * -1
    # C3 now has an x property with a modified getter
    # so modify its setter rather than C.x's setter.
    @x.setter
    def x(self, value):
        self._x = value * 2

Of course at the point that you're overriding getter, setter, and deleter you
can probably just redefine the property for C3.


"""

# class NufftExact(Nufft):
#     def __init__(self,**kwargs):
#         super(NufftExact, self).__init__(**kwargs)

# class NufftSparse(Nufft):
#     def __init__(self,**kwargs):
#         super(NufftSparse, self).__init__(**kwargs)
#         self.p = None

# class NufftTable(Nufft):
#     def __init__(self,**kwargs):
#         super(NufftTable, self).__init__(**kwargs)
# @Nufft.Kd.setter
# def x(self, Kd):
# Nufft.Kd.fset(self, Kd)


def _get_legend_text(ax):
    l = ax.get_legend()
    if l is None:
        return None
    else:
        return [t.get_text() for t in l.get_texts()]



class NufftKernel(object):
    """ Interpolation kernel for use in the gridding stage of the NUFFT. """

    def __init__(self, kernel_type='minmax:kb', **kwargs):
        self.kernel = None
        self.is_kaiser_scale = False
        self.params = kwargs.copy()
        self.kernel_type = kernel_type

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type):
        if is_string_like(kernel_type):
            self._kernel_type = kernel_type
            self._initialize_kernel(kernel_type)
        elif isinstance(kernel_type, (list, tuple, set)):
            kernel_type = list(kernel_type)
            if isinstance(kernel_type[0], collections.Callable):
                if len(kernel_type) != self.ndim:
                    raise ValueError(
                        'wrong # of kernels specified in list')
                # list of kernels was already passed in
                self.kernel = kernel_type
            else:
                raise ValueError('kernel_type list must contain a series of ' +
                                 'callable kernel functions')
            self._kernel_type = 'inline'
        # if single kernel was passed, replicate over all dimensions
        elif isinstance(kernel_type, collections.Callable):
            # replicate to fill list for each dim
            self.kernel = [kernel_type, ] * self.ndim
            self._kernel_type = 'inline'
        else:
            raise ValueError('invalid type for kernel_type: {}'.format(
                type(kernel_type)))

    def _initialize_kernel(self, kernel_type):

        params = self.params.copy()

        # if no dimensions specified, using longest among Kd, Jd, Nd
        if params.get('ndim', None) is None:
            max_len = 0
            for k, v in list(params.items()):
                if k in ['Kd', 'Jd', 'Nd']:
                    params[k] = _to_1d_int_array(v)
                    max_len = len(params[k])
            self.ndim = max_len
        else:
            self.ndim = params.pop('ndim')

        # replicate any that were length one to ndim array
        for k, v in list(params.items()):
            if k in ['Kd', 'Jd', 'Nd']:
                params[k] = _to_1d_int_array(v, self.ndim)
            # Nmid is not necessarily an integer, so handle it manually
            if 'Nmid' in params and len(params['Nmid']) < self.ndim:
                if len(params['Nmid']) > 1:
                    raise ValueError("Nmid dimension mismatch")
                else:
                    params['Nmid'] = np.asarray(
                        [params['Nmid'][0], ] * self.ndim)

        ndim = self.ndim  # number of dimensions

        Kd = params.get('Kd', None)  # oversampled image size
        Jd = params.get('Jd', None)  # kernel size
        Nd = params.get('Nd', None)  # image size
        Nmid = params.get('Nmid', None)
        kb_alf = params.get('kb_alf', None)  # alpha for kb:* cases
        kb_m = params.get('kb_m', None)  # m for kb:* cases
        alpha = params.get('alpha', None)  # alpha for minmax:* cases
        beta = params.get('beta', None)  # beta for minmax:* cases

        # linear interpolator straw man
        if kernel_type == 'linear':
            kernel_type = 'inline'
            kernel = lambda k, J: (1 - abs(k / (J / 2.))) * (abs(k) < J / 2.)
            self.kernel = [kernel, ] * ndim

        elif kernel_type == 'diric':   # exact interpolator
            if (Kd is None) or (Nd is None):
                raise ValueError("kwargs must contain Kd, Nd for diric case")

            if not np.all(np.equal(Jd, Kd)):
                warnings.warn('diric inexact unless Jd=Kd')

            self.kernel = []
            for id in range(ndim):
                N = Nd[id]
                K = Kd[id]
                if self.params.get('phasing', None) == 'real':
                    N = 2 * np.floor((K + 1) / 2.) - 1  # trick
                self.kernel.append(lambda k, J: (
                    N / K * nufft_diric(k, N, K, True)))
        elif kernel_type == 'kb:beatty':  # KB with Beatty et al parameters
            self.is_kaiser_scale = True

            # TODO: could take K_N directly instead
            if (Kd is None) or (Nd is None) or (Jd is None):
                raise ValueError("kwargs must contain Kd, Nd, Jd for " +
                                 "{} case".format(kernel_type))

            # warn if user specified specific alpha, m
            if (kb_m is not None) or (kb_alf is not None):
                warnings.warn('user supplied kb_alf and kb_m ignored')

            K_N = Kd / Nd
            params['kb_alf'] = \
                np.pi * np.sqrt(Jd ** 2 / K_N ** 2 * (K_N - 0.5) ** 2 - 0.8)
            params['kb_m'] = np.zeros((ndim,))
            self.kernel = []
            for id in range(ndim):
                self.kernel.append(kaiser_bessel('inline',
                                                 Jd[id],
                                                 params['kb_alf'][id],
                                                 params['kb_m'][id])[0])
        # alpha = pi * sqrt(J^2/K_N^2 * (K_N - 0.5)^2 - 0.8);  %Eq. 5 of
        # Beatty2005:  IEEETMI 24(6):799:808

        # KB with minmax-optimized parameters
        elif kernel_type == 'kb:minmax':
            self.is_kaiser_scale = True

            if (Jd is None):
                raise ValueError("kwargs must contain Jd for " +
                                 "{} case".format(kernel_type))

            # warn if user specified specific alpha, m
            if (kb_m is not None) or (kb_alf is not None):
                warnings.warn('user supplied kb_alf and kb_m ignored')

            self.kernel = []
            params['kb_alf'] = []
            params['kb_m'] = []
            for id in range(ndim):
                k, alf, m = kaiser_bessel('inline', Jd[id])
                self.kernel.append(k)
                params['kb_alf'].append(alf)
                params['kb_m'].append(m)

        elif kernel_type == 'kb:user':  # KB with Beatty et al parameters
            self.is_kaiser_scale = True

            if (Jd is None) or (kb_m is None) or (kb_alf is None):
                raise ValueError("kwargs must contain Jd, kb_m, kb_alf for" +
                                 "{} case".format(kernel_type))

            self.kernel = []
            for id in range(ndim):
                self.kernel.append(kaiser_bessel('inline',
                                                 J=Jd[id],
                                                 alpha=kb_alf[id],
                                                 kb_m=kb_m[id])[0])

        # minmax interpolator with KB scaling factors (recommended default)
        elif kernel_type == 'minmax:kb':
            if (Kd is None) or (Nd is None) or (Jd is None) or (Nmid is None):
                raise ValueError("kwargs must contain Kd, Nd, Jd, Nmid for " +
                                 "{} case".format(kernel_type))
            params['alpha'] = []
            params['beta'] = []
            for id in range(ndim):
                [al, be] = nufft_alpha_kb_fit(N=Nd[id], J=Jd[id], K=Kd[id],
                                              Nmid=Nmid[id])
                params['alpha'].append(al)
                params['beta'].append(be)

        # minmax interpolator with numerically "tuned" scaling factors
        elif kernel_type == 'minmax:tuned':  # TODO
            if (Kd is None) or (Nd is None) or (Jd is None):
                raise ValueError("kwargs must contain Kd, Nd, Jd for " +
                                 "{} case".format(kernel_type))
            params['alpha'] = []
            params['beta'] = []
            for id in range(ndim):
                [al, be, ok] = nufft_best_alpha(J=Jd[id], L=0,
                                                K_N=Kd[id] / Nd[id])
                params['alpha'].append(al)
                params['beta'].append(be)
                if not ok:
                    raise ValueError('unknown J,K/N')

        # minmax interpolator with user-provided scaling factors
        elif kernel_type == 'minmax:user':
            if (alpha is None) or (beta is None):
                raise ValueError("user must provide alpha, beta for " +
                                 "{} case".format(kernel_type))
            if len(alpha) != ndim or len(beta) != ndim:
                print("alpha={}".format(alpha))
                print("beta={}".format(beta))
                print("ndim={}".format(ndim))
                raise ValueError('alpha/beta size mismatch')

        elif kernel_type == 'minmax:unif':
            params['alpha'] = []
            params['beta'] = []
            for id in range(ndim):
                params['alpha'].append(1.)
                params['beta'].append(0.)
        else:
            raise ValueError('unknown kernel type')

        if 'alpha' in params:
            self.alpha = params['alpha']
        if 'beta' in params:
            self.beta = params['beta']
        if 'kb_m' in params:
            self.kb_m = params['kb_m']
        if 'kb_alf' in params:
            self.kb_alf = params['kb_alf']

        self.params = params

    def plot(self, axes=None):
        """ plot the (separable) kernel for each axis """
        legend_text = 'type: {}'.format(self.kernel_type)
        if axes is None:
            f, axes = plt.subplots(self.ndim, 1, sharex=True)
            axes = np.atleast_1d(axes)
            legend_labels = [legend_text, ]
        else:
            legend_labels = _get_legend_text(axes[0])
            if legend_labels is None:
                legend_labels = [legend_text, ]
            else:
                legend_labels.append(legend_text)
        for d in range(self.ndim):
            if 'Jd' in self.params:
                J = self.params['Jd'][d]
            else:
                J = 1
            x = np.linspace(-J/2, J/2, 1001)

            if 'minmax:kb' in self.kernel_type:
                y = _nufft_table_make1('fast', N=self.params['Nd'][d],
                                       K=self.params['Kd'][d],
                                       J=self.params['Jd'][d],
                                       L=250, kernel_type=self.kernel_type,
                                       phasing='real')[0]
            else:
                y = self.kernel[d](x, J)
            axes[d].plot(x, y)
            if d == self.ndim - 1:
                axes[d].xaxis.set_ticks([-J/2, J/2])
                axes[d].xaxis.set_ticklabels(['-J/2', 'J/2'])
            axes[d].set_ylabel('kernel amplitude, axis {}'.format(d))
            axes[d].legend(legend_labels)
        plt.draw()
        return axes

    def __repr__(self):
        repstr = "kernel type: {}\n".format(self.kernel_type)
        repstr += "kernel dimensions: {}\n".format(self.ndim)
        if 'kb:' in self.kernel_type:
            repstr += "Kaiser Bessel params:\n"
            for d in range(self.ndim):
                repstr += "    alpha[{}], m[{}] = {}, {}\n".format(
                    d, d, self.kb_alf[d], self.kb_m[d])
        elif 'minmax:' in self.kernel_type:
            repstr += "Minmax params:\n"
            for d in range(self.ndim):
                repstr += "    alpha[{}], beta[{}] = {}, {}".format(
                    d, d, self.alpha[d], self.beta[d])
        return repstr





# change name of NufftBase to NFFT_Base
# Note: must have object here to get a new-style class!
# TODO: change default n_shift to Nd/2?
class NufftBase(object):

    def __init__(self, Nd, om, Jd=6, Kd=None, p=None, sn=None, Ld=2048,
                 tol=1e-7, precision=None, kernel_type='kb:beatty',
                 n_shift=None, kernel_kwargs={}, phasing='real',
                 mode='table0', sparse_format='CSC', verbose=False,
                 ortho=False,
                 **kwargs):

        self.verbose = verbose
        if self.verbose:
            print("Entering NufftBase init")
        self.__init_complete = False  # will be set true after __init__()

        # must set the __ version of these to avoid circular calls by the
        # setters
        self.__Nd = _to_1d_int_array(Nd)
        if self.verbose:
            print("Nd={}".format(Nd))
            print("self.__Nd={}".format(self.__Nd))
            print("self.Nd={}".format(self.Nd))

        self.__phasing = phasing
        # TODO: lowmem functionality not currently implemented
        self._lowmem = False  # if True don't prestore phase values
        self.__om = None  # will be set later below
        self._set_Nmid()
        self.ndim = len(self.Nd)  # number of dimensions
        if self.ndim == 1:
            # TODO: only double precision routines in 1D case
            if precision is not None and precision == 'single':
                warnings.warn("Forcing double precision for 1D case")
            precision = 'double'
        self._Jd = _to_1d_int_array(Jd, nelem=self.ndim)

        if Kd is None:
            Kd = 2 * self.__Nd
        self.__Kd = _to_1d_int_array(Kd, nelem=self.ndim)

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
        # [M, *Kd]	sparse interpolation matrix (or empty if table-based)
        self.p = None
        self.Jd = Jd
        self.kernel = NufftKernel(kernel_type,
                                  ndim=self.ndim,
                                  Nd=self.Nd,
                                  Jd=self.Jd,
                                  Kd=self.Kd,
                                  Nmid=self.Nmid,
                                  **kernel_kwargs)
        self._calc_scaling()  # [(Nd)]		scaling factors
        self.tol = tol
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
        if self.mode == 'sparse':
            self._init_sparsemat()  # create COO matrix
            # convert to other format if specified
            if sparse_format is None:
                self.sparse_format = 'COO'
            else:  # convert formats via setter if necessary
                self.sparse_format = sparse_format
            self.__Ld = None
        elif 'table' in self.mode:
            # TODO: change name of Ld to table_oversampling
            self.Ld = _to_1d_int_array(Ld, nelem=self.ndim)
            if self.mode == 'table0':
                self.table_order = 0  # just order in newfft
            elif self.mode == 'table1':
                self.table_order = 1  # just order in newfft
            else:
                raise ValueError("Invalid NUFFT mode: {}".format(self.mode))
            self._init_table()
            self.interp_table = _nufft_table_interp  # TODO: remove?
            self.interp_table_adj = _nufft_table_adj  # TODO: remove?
        elif self.mode == 'exact':
            self.__Ld = None
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
        #super(NufftBase, self).__init__(**kwargs)

    def _nufft_forward(self, x):
        y = nufft_forward(self, x=x)
        return y

    def _nufft_adj(self, X):
        y = nufft_adj(self, X=X)
        return y

#    TODO:
#    def _init_pyfftw(self, X):
#        a_b = pyfftw.n_byte_align_empty(4, 16, dtype='complex128')
#        self.pyfftw_fftn = yfftw.builders.fftn(a_b, threads=nthreads,
#                                  overwrite_input=False,
#                                  planner_effort=planning_flag)

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
                self.p = self.p.tocoo()
            elif sparse_format == 'DOK':
                self.p = self.p.tocoo()
            else:
                raise ValueError("unrecognized sparse format type")
        else:
            raise ValueError("no sparse matrix exists.  cannot update sparse" +
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
            if om.ndim == 1:
                om = om[:, np.newaxis]
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
        self.__Nd = _to_1d_int_array(Nd, nelem=self.ndim)
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
        self._Jd = _to_1d_int_array(Jd, nelem=self.ndim)
        if self.__init_complete:
            self._reinitialize()

    @property
    def Ld(self):
        return self.__Ld

    @Ld.setter
    def Ld(self, Ld):
        self.__Ld = _to_1d_int_array(Ld, nelem=self.ndim)
        if 'table' not in self.mode:
            warnings.warn("Ld is ignored for mode = {}".format(self.mode))
        elif self.__init_complete:
            self._reinitialize()

    @property
    def Kd(self):
        return self.__Kd

    @Kd.setter
    def Kd(self, Kd):
        self.__Kd = _to_1d_int_array(Kd, nelem=self.ndim)
        # if (not self._lowmem) and (self.phasing == 'real'):
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
        if (not self._lowmem) and (
                self.phasing == 'real') and (self.__om is not None):
            self.phase_after = self._phase_after(self.__om,
                                                 self.Nmid, self.__n_shift)
        if self.__init_complete:
            self._reinitialize()

    def _update_array__precision(self):
        # update the data types of other members
        # TODO: warn if losing precision during conversion?
        if isinstance(self.__om, np.ndarray):
            self.__om = self.__om.astype(self._real_dtype)
        if isinstance(self.__n_shift, np.ndarray):
            self.__n_shift = self.__n_shift.astype(self._real_dtype)
        if isinstance(self.phase_before, np.ndarray):
            self.phase_before = self.phase_before.astype(self._cplx_dtype)
        if isinstance(self.phase_after, np.ndarray):
            self.phase_after = self.phase_after.astype(self._cplx_dtype)
        if hasattr(self, 'sn') and isinstance(self.sn, np.ndarray):
            if np.iscomplexobj(self.sn):
                self.sn = self.sn.astype(self._cplx_dtype)
            else:
                self.sn = self.sn.astype(self._real_dtype)
        if self.mode == 'sparse':
            if hasattr(self, 'p') and self.p is not None:
                if self.phasing == 'complex':
                    self.p = self.p.astype(self._cplx_dtype)
                else:
                    self.p = self.p.astype(self._real_dtype)
        elif 'table' in self.mode:
            if hasattr(self, 'h') and self.h is not None:
                for idx, h in enumerate(self.h):
                    if self.phasing == 'complex':
                        self.h[idx] = h.astype(self._cplx_dtype)
                    else:
                        self.h[idx] = h.astype(self._real_dtype)

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
            self.phase_before = self._phase_before(self.Kd, self.Nmid)
            self.phase_after = self._phase_after(self.om,
                                                 self.Nmid,
                                                 self.n_shift)
        # complex kernel incorporates the phase
        elif self.phasing == 'complex':
            self.phase_before = None
            self.phase_after = None
        else:
            raise ValueError("Invalid phasing: {}\n\t".format(self.phasing) +
                             "must be 'real' or 'complex'")

    def _phase_before(self, Kd, Nmid):
        phase = 2 * np.pi * np.arange(Kd[0]) / Kd[0] * Nmid[0]
        for id in range(1, Kd.size):
            tmp = 2 * np.pi * np.arange(Kd[id]) / Kd[id] * Nmid[id]
            # fast outer sum via broadcasting
            phase = phase.reshape(
                (phase.shape) + (1,)) + tmp.reshape((1,) * id + (tmp.size,))
        return np.exp(1j * phase).astype(self._cplx_dtype)  # [(Kd)]

    def _phase_after(self, om, Nmid, n_shift):
        phase = np.exp(1j * np.dot(om, (n_shift - Nmid).reshape(-1, 1)))
        return np.squeeze(phase).astype(self._cplx_dtype)  # [M,1]

    def _calc_scaling(self):
        """
        # scaling factors: "outer product" of 1D vectors
        """
        kernel = self.kernel
        Nd = self.Nd
        Kd = self.Kd
        Jd = self.Jd
        ktype = kernel.kernel_type
        if ktype == 'diric':
            self.sn = np.ones(Nd)
        elif 'minmax:' in ktype:
            self.sn = nufft_scale(Nd, Kd, kernel.alpha,
                                  kernel.beta, self.Nmid)
        else:
            self.sn = np.array([1.])
            for id in range(self.ndim):
                if kernel.is_kaiser_scale:
                    # nc = np.arange(Nd[id])-(Nd[id]-1)/2.  #OLD WAY
                    nc = np.arange(Nd[id]) - self.Nmid[id]
                    tmp = 1 / kaiser_bessel_ft(nc / Kd[id], Jd[id],
                                               kernel.kb_alf[id],
                                               kernel.kb_m[id], 1)
                elif ktype == 'inline':
                    tmp = 1 / _nufft_interp_zn(0, Nd[id], Jd[id], Kd[id],
                                               kernel.kernel[id],
                                               self.Nmid[id])
                elif ktype == 'linear':
                    raise ValueError("Not Implemented")
                    # TODO: _scale_tri
                    tmp = _scale_tri(Nd[id], Jd[id], Kd[id], self.Nmid[id])
#                elif 'minmax:' in ktype:
#                    tmp = nufft_scale(Nd[id], Kd[id], kernel.alpha[id],
#                                      kernel.beta[id], self.Nmid[id])
                else:
                    raise ValueError("Unsupported ktype: {}".format(ktype))
                # tmp = reale(tmp)  #TODO: reale?
                # TODO: replace outer with broadcasting?
                self.sn = np.outer(self.sn.ravel(), tmp.conj())
        if len(Nd) > 1:
            self.sn = self.sn.reshape(tuple(Nd))  # [(Nd)]
        else:
            self.sn = self.sn.ravel()  # [(Nd)]

    def _init_sparsemat(self):
        """  [J?,M] interpolation coefficient vectors.  will need kron of these
        later
        """
        tstart = time()
        ud = {}
        kd = {}
        om = self.om
        if om.ndim == 1:
            om = om[:, np.newaxis]

        if self.phasing == 'real':
            # recall just to be safe in case Kd, Nmid, etc changed?
            self._set_phase_funcs()

        for id in range(self.ndim):
            N = self.Nd[id]
            J = self.Jd[id]
            K = self.Kd[id]

            # callable kernel:  kaiser, linear, etc
            if (self.kernel.kernel is not None):
                kernel_func = self.kernel.kernel[id]
                if not isinstance(kernel_func, collections.Callable):
                    raise ValueError("callable kernel function required")
                # [J?,M]
                [c, arg] = _nufft_coef(om[:, id], J, K, kernel_func)
            else:  # minmax:
                alpha = self.kernel.alpha[id]
                beta = self.kernel.beta[id]
                # [J?,J?]  TODO: move .tol into kernel object
                T = _nufft_T(N, J, K, tol=self.tol, alpha=alpha, beta=beta)
                [r, arg] = _nufft_r(
                    om[:, id], N, J, K, alpha=alpha, beta=beta)  # [J?,M]
                # c = T * r  clear T r
                c = np.dot(T, r)
            #
            # indices into oversampled FFT components
            #
            # [M,1] to leftmost near nbr
            koff = _nufft_offset(om[:, id], J, K)

            # [J,M]
            kd[id] = np.mod(outer_sum(np.arange(1, J + 1), koff), K).T

            if self.phasing == 'complex':
                gam = 2 * np.pi / K
                phase_scale = 1j * gam * (N - 1) / 2.
                phase = np.exp(phase_scale * arg)   # [J?,M] linear phase
            elif self.phasing in ['real', None]:
                phase = 1.
            else:
                raise ValueError("Unknown phasing {}".format(self.phasing))

            ud[id] = phase * c      # [J?,M]


        tend1 = time()
        if self.verbose:
            print("Nd={}".format(Nd))


        """
        build sparse matrix that is [M,*Kd]
        """
        M = self.om.shape[0]
        kk = kd[0]  # [J1,M]
        uu = ud[0]  # [J1,M]

        for id in range(1, self.ndim):
            Jprod = np.prod(self.Jd[0:id + 1])
            # trick: pre-convert these indices into offsets! (Fortran order)
            tmp = kd[id] * np.prod(self.Kd[:id])
            kk = _block_outer_sum(kk, tmp)  # outer sum of indices
            kk = kk.reshape(Jprod, M, order='F')
            uu = _block_outer_prod(uu, ud[id])  # outer product of coefficients
            uu = uu.reshape(Jprod, M, order='F')
        # now kk and uu are [*Jd, M]

        #
        # apply phase shift
        # pre-do Hermitian transpose of interpolation coefficients
        #
        if np.iscomplexobj(uu):
            uu = uu.conj()

        if self.phasing == 'complex':
            phase = np.exp(1j * np.dot(om, self.n_shift.ravel()))			# [1,M]
            phase = phase.reshape((1, -1), order='F')  # =(phase.shape[0],1)
            uu *= phase  # use broadcasting along first dimension
            sparse_dtype = self._cplx_dtype
        elif self.phasing == 'real' or self.phasing is None:
            sparse_dtype = self._real_dtype
        else:
            raise ValueError("Invalid phasing: {}".format(self.phasing))

        if self.ndim >= 3:  # TODO: move elsewhere
            RAM_GB = self.Jd.prod() * M * sparse_dtype.itemsize / 10 ** 9
            if self.verbose:
                print('NUFFT sparse matrix storage will require %g GB' % (RAM_GB))

        # [*Jd,M]
        mm = np.tile(np.arange(M), (np.product(self.Jd), 1))

        self.p = coo_matrix((uu.ravel(order='F'),
                             (mm.ravel(order='F'), kk.ravel(order='F'))),
                            shape=(M, self.Kd.prod()), dtype=sparse_dtype)
        tend2 = time()
        if self.verbose:
            print("Sparse init stage 2 duration = {} s".format(tend2-tend1))

    def _init_table(self):
        """ Initialize structure for d-dimension NUFFT using table-based
        interpolator

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
        self.phase_shift = None  # compute on-the-fly
        if self.Ld is None:
            if self.table_order == 0:
                self.Ld = 2 ** 11
            elif self.table_order == 1:
                self.Ld = 2 ** 9
            else:
                raise ValueError("Bad table mode")
        if ndim != len(self.Jd) or ndim != len(self.Ld) or ndim != len(self.Kd):
            raise ValueError('inconsistent dimensions among ndim, Jd, Ld, Kd')
        if ndim != self.om.shape[1]:
            raise ValueError('omega needs %d columns' % (ndim))

        self.h = []
        # build kernel lookup table (LUT) for each dimension
        for id in range(ndim):
            if 'alpha' in kernel_kwargs:
                kernel_kwargs['alpha'] = [self.kernel.params['alpha'][id], ]
                kernel_kwargs['beta'] = [self.kernel.params['beta'][id], ]
            if 'kb_alf' in kernel_kwargs:
                kernel_kwargs['kb_alf'] = [self.kernel.params['kb_alf'][id], ]
                kernel_kwargs['kb_m'] = [self.kernel.params['kb_m'][id], ]

            h, t0 = _nufft_table_make1(how=how, N=self.Nd[id], J=self.Jd[id],
                                       K=self.Kd[id], L=self.Ld[id],
                                       phasing=self.phasing,
                                       kernel_type=self.kernel.kernel_type,
                                       kernel_kwargs=kernel_kwargs)
            if self.phasing == 'complex':
                if np.isrealobj(h):
                    warnings.warn("Real NUFFT kernel?")
            elif self.phasing in ['real', None]:
                if not np.isrealobj(h):
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

    def plot_kernels(self):
        """may want to change to subplot_stack"""
        f = plt.figure()
        plt.hold('on')
        if self.kernel.kernel is not None:
            legend_labels = []
            x = np.zeros((1000, self.ndim), dtype=self._real_dtype)
            if self.phasing == 'complex':
                Y = np.zeros((1000, self.ndim), dtype=self._cplx_dtype)
            else:
                Y = np.zeros((1000, self.ndim), dtype=self._real_dtype)
            d = 0
            while d < self.ndim:
                x[:, d] = np.linspace(-self.Jd[d] / 2, self.Jd[d] / 2, 1000)
                Y[:, d] = self.kernel.kernel[d](x[:, d], self.Jd[d])
                legend_labels.append('axis %d' % d)
                d += 1
            subplot_stack(x, Y, ncols=1,
                          title='NUFFT kernel shapes (type: {})'.format(
                              self.kernel.kernel_type),
                          legends=legend_labels,
                          use_yticks=True)
        else:
            print("Kernel is not an inline function. will not be plotted")
        return f


def _nufft_table_interp(st, Xk, om=None):
    """ table-based nufft
     in
        st	structure	formed by nufft_init (through nufft_init_table)
        Xk	[*Kd,nc]	over-sampled DFT coefficients
        om	[M,1]		frequency locations, overriding st.om
     out
        X	[M,nc]		NUFFT values
    Matlab version copyright 2004-3-30, Jeff Fessler and Yingying Zhang, University of Michigan

    Note: should not call this directly, but via nufft_forward()
    """

    order = st.table_order
    flips = np.zeros(st.Nd.shape)  # TODO: remove

    if om is None:
        om = st.om

    ndim = len(st.Kd)

    tm = np.zeros_like(om)
    pi = np.pi
    for id in range(0, ndim):
        gam = 2 * pi / st.Kd[id]
        tm[:, id] = om[:, id] / gam  # t = omega / gamma

    if Xk.ndim == 1:
        Xk = Xk[:, np.newaxis]
    elif Xk.shape[1] > Xk.shape[0]:
        Xk = Xk.T
    nc = Xk.shape[1]

    if Xk.shape[0] != np.product(st.Kd):
        raise ValueError('Xk size problem')

    Xk = complexify(Xk)  # force complex

    # X = np.zeros((om.shape[0], nc),dtype=Xk.dtype)
    arg = [st.Jd, st.Ld, tm, order, flips]

    if ndim == 1:
        X = interp1_table(Xk, st.h[0], *arg)
    elif ndim == 2:
        # Fortran ordering to match Matlab behavior
        Xk = np.reshape(Xk, np.hstack((st.Kd, nc)), order='F')
        X = interp2_table(Xk, st.h[0], st.h[1], *arg)
    elif ndim == 3:
        # Xk = np.asarray(Xk)
        # Fortran ordering to match Matlab behavior
        Xk = np.reshape(Xk, np.hstack((st.Kd, nc)), order='F')
        X = interp3_table(Xk, st.h[0], st.h[1], st.h[2], *arg)
    else:
        raise ValueError('dimensions > 3d not done')

    # apply phase shift
    if hasattr(st, 'phase_shift'):
        if isinstance(st.phase_shift, (np.ndarray, list)):
            if len(st.phase_shift) > 0:
                ph = np.tile(st.phase_shift, (1, nc))  # TODO: change to broadcasting instead
                ph.shape = X.shape  # ensure same size
                X = X * ph  # for arrays, * is elementwise multiplication
    return X.astype(Xk.dtype)


def _nufft_table_adj(st, X, om=None):
    """  adjoint of table-based nufft interpolation.
     in
        st		structure from nufft_init
        X [M,nc]	DTFT values (usually nc=1)
        om [M,1]	optional (default st.om)
     out
        Xk [*Kd,nc]	DFT coefficients
    Matlab version copyright 2004-3-30, Jeff Fessler and Yingying Zhang, University of Michigan
    """

    order = st.table_order
    flips = np.zeros(st.Nd.shape)  # TODO: remove

    if om is None:
        om = st.om

    ndim = len(st.Kd)

    tm = np.zeros_like(om)
    pi = np.pi
    for id in range(0, ndim):
        gam = 2 * pi / st.Kd[id]
        tm[:, id] = om[:, id] / gam  # t = omega / gamma

    if X.shape[0] != om.shape[0]:
        raise ValueError('X size problem')

    if X.ndim is 1:
        X = X[:, np.newaxis]
    nc = X.shape[1]

    X = complexify(X)  # force complex

    # adjoint of phase shift
    if hasattr(st, 'phase_shift'):
        if isinstance(st.phase_shift, (np.ndarray, list)):
            if len(st.phase_shift) > 0:
                ph = np.tile(st.phase_shift.conj(), (1, nc))
                ph.shape = X.shape
                # elementwise multiplication
                X = np.asarray(X) * np.asarray(ph)

    arg = [st.Jd, st.Ld, tm, st.Kd[0:ndim], order, flips]

    # Xk = np.zeros((np.product(st.Kd), nc))
    if ndim == 1:
        Xk = interp1_table_adj(X, st.h[0], *arg)
    elif ndim == 2:
        Xk = interp2_table_adj(X, st.h[0], st.h[1], *arg)
    elif ndim == 3:
        Xk = interp3_table_adj(X, st.h[0], st.h[1], st.h[2], *arg)
    else:
        raise ValueError('> 3d not done')

    return Xk.astype(X.dtype)


def nufft_init():  # TODO: remove this
    pass


def nufft():  # TODO: remove this
    pass


def _nufft_table_make1(
        how, N, J, K, L, kernel_type, phasing, kernel_kwargs={}):
    """ make LUT for 1 dimension by creating a dummy 1D NUFFT object """
    nufft_args = {'Jd': J, 'n_shift': 0,
                  'kernel_type': kernel_type, 'kernel_kwargs': kernel_kwargs,
                  'mode': 'sparse', 'phasing': phasing, 'sparse_format': 'csc'}
    t0 = np.arange(-J * L / 2., J * L / 2. + 1) / L  # [J*L+1]
    pi = np.pi
    # This is a slow and inefficient (but simple) way to get the table
    # because it builds a huge sparse matrix but only uses 1 column!
    if how == 'slow':
        om0 = t0 * 2 * pi / K  # gam
        s1 = NufftBase(om=om0, Nd=N, Kd=K, **nufft_args)
        h = np.asarray(s1.p[:, 0].todense()).ravel()  # [J*L + 1]
    # This way is "J times faster" than the slow way, but still not ideal.
    # It works for any user-specified interpolator.
    elif how == 'fast':
        t1 = J / 2. - 1 + np.arange(L) / L  # [L]
        om1 = t1 * 2 * pi / K		# * gam
        s1 = NufftBase(om=om1, Nd=N, Kd=K, **nufft_args)
        h = np.asarray(
            s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(order='F')
        h = np.concatenate((h, np.asarray([h[0], ])), axis=0)  # [J*L+1,]
    # This efficient way uses only "J" columns of sparse matrix!
    # The trick to this is to use fake small values for N and K,
    # which works for interpolators that depend only on the ratio K/N.
    elif how == 'ratio':  # e.g., 'minmax:kb' | 'kb:*'
        Nfake = J
        print("N={},J={},K={}".format(N, J, K))
        Kfake = Nfake * K / N
        print("Nfake={},Kfake={}".format(Nfake, Kfake))
        t1 = J / 2. - 1 + np.arange(L) / L  # [L]
        om1 = t1 * 2 * pi / Kfake		# gam
        s1 = NufftBase(om=om1, Nd=Nfake, Kd=Kfake, **nufft_args)
        h = np.asarray(
            s1.p[:, np.arange(J - 1, -1, -1)].todense()).ravel(order='F')
        # [J*L+1] assuming symmetric
        h = np.concatenate((h, np.asarray([h[0], ])), axis=0)
        if phasing == 'complex':
            h = h * np.exp(1j * pi * t0 * (1 / K - 1 / Kfake))  # fix phase
    else:
        raise ValueError("Bad Type: {}".format(type))
    return h, t0


def _block_outer_sum(x1, x2):
    """#function y = _block_outer_sum(x1, x2)"""
    [J1, M] = x1.shape
    [J2, M] = x2.shape
    xx1 = np.reshape(x1, (J1, 1, M))  # [J1,1,M] from [J1,M]
    xx2 = np.reshape(x2, (1, J2, M))  # [1,J2,M] from [J2,M]
    # use numpy broadcasting
    y = xx1 + xx2			# [J1,J2,M]
    return y


def _block_outer_prod(x1, x2):
    """#function y = _block_outer_prod(x1, x2)"""
    [J1, M] = x1.shape
    [J2, M] = x2.shape
    xx1 = np.reshape(x1, (J1, 1, M))  # [J1,1,M] from [J1,M]
    xx2 = np.reshape(x2, (1, J2, M))  # [1,J2,M] from [J2,M]
    # use numpy broadcasting
    y = xx1 * xx2			# [J1,J2,M]
    return y


def nufft_forward(st, x, copy_x=True):
    """
    %function X = nufft(x, st)
    % Compute d-dimensional NUFFT of signal/image x
    % in
    %	x	[(L),N1,N2,...,Nd]	L input image(s) of size
    %						N1 x N2 x ... x Nd
    %	st	structure		precomputed by nufft_init()
    % out
    %	X	[M,(L)]			output spectra
    %
    """

    Nd = st.Nd
    Kd = st.Kd

    ndim = len(Nd)

    if copy_x:  # make sure original array isn't modified!
        x = x.copy()

    try:  # collapse all excess dimensions into just one
        x = x.reshape(list(Nd) + [-1, ],)
    except:
        raise ValueError('input signal has wrong size')

    # Promote to complex if real input was provided
    x = complexify(x)

    L = x.shape[-1]
    # x=np.squeeze(x)

    #
    # the usual case is where L=1, i.e., there is just one input signal.
    #
    Xk = np.zeros((np.product(Kd), L), dtype=x.dtype)			# [*L,*Kd]
    for ll in range(L):
        xl = x[..., ll] * st.sn		# scaling factors
        # Fortran order to match Matlab's behavior
        Xk[:, ll] = fftn(xl, Kd).ravel(order='F')
        if st.phase_before is not None:
            # TODO: store already raveled?
            Xk[:, ll] *= st.phase_before.ravel(order='F')

    if st.ortho:
        Xk /= st.scale_ortho
    # interpolate using precomputed sparse matrix
    # or with tabulated interpolator
    if 'table' in st.mode:
        # Xk = Xk.astype(st.h[0].dtype)
        X = st.interp_table(st, Xk)
    else:
        X = st.p * Xk					# [M,*L]

    if x.ndim > ndim:
        X = np.reshape(X, (st.M, L))

    if st.phase_after is not None:
        X *= st.phase_after[:, None]  # broadcast rather than np.tile

    return X







def nufft_adj(st, X, copy_X=True):
    """
    function x = nufft_adj(X, st)
     Apply adjoint of d-dimensional NUFFT to spectrum vector(s) X
     in
        X	[M,(L)]
        st			structure precomputed by nufft_init()
     out
        x	[(Nd),(L)]	signal(s)/image(s)

     Matlab vers. copyright 2003-6-1, Jeff Fessler, The University of Michigan
    """
    # extract attributes from structure
    Nd = st.Nd
    Kd = st.Kd
    dims = X.shape
    if dims[0] != st.M:
        raise ValueError('size mismatch')

    #
    # adjoint of interpolator using precomputed sparse matrix
    #
    try:
        not_1d = dims[1] > 1
    except:
        not_1d = False

    if copy_X:  # make sure original array isn't modified!
        X = X.copy()

    if (len(dims) > 2) or (not_1d):
        Lprod = np.product(dims[1:])
        X.shape = (st.M, Lprod)  # [M,*L]
        # X = reshape(X, (st.M, Lprod))	#
    else:
        X.shape = (st.M, 1)
        Lprod = 1  # the usual case

    X = complexify(X)  # force complex

    if st.phase_after is not None:
        # replaced np.tile() with broadcasting
        X *= st.phase_after.conj()[:, None]

    if 'table' in st.mode:
        X = X.astype(np.result_type(st.h[0], X.dtype), copy=False)
        Xk_all = st.interp_table_adj(st, X)
    else:
        Xk_all = (st.p.H * X)  # [*Kd,*L]

    x = np.zeros((np.product(Kd), Lprod), dtype=X.dtype)  # [*Kd,*L]

    if Xk_all.ndim == 1:
        Xk_all = Xk_all[:, None]

    for ll in range(Lprod):
        Xk = np.reshape(Xk_all[:, ll], Kd, order='F')  # [(Kd)]
        if st.phase_before is not None:
            Xk *= st.phase_before.conj()
        x[:, ll] = np.product(Kd) * ifftn(Xk).ravel(order='F')

    if st.ortho:
        x *= st.scale_ortho

    x = x.reshape(tuple(Kd) + (Lprod,), order='F')  # [(Kd),*L]

    # eliminate zero padding from ends  fix: need generic method
    if len(Nd) == 1:
        x = x[0:Nd[0], :]  # [N1,*L]
    elif len(Nd) == 2:
        x = x[0:Nd[0], 0:Nd[1], :]  # [N2,N1,*L]
    elif len(Nd) == 3:
        x = x[0:Nd[0], 0:Nd[1], 0:Nd[2], :]
    else:
        raise ValueError('only up to 3D implemented currently')

    # scaling factors
    x *= st.sn.conj()[..., None]
    # x = np.squeeze(x) #remove singleton dimension(s)

    return x
