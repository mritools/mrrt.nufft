"""Convolution kernel to be used for Non-Uniform FFT.

"""
from abc import ABC, abstractmethod

import functools
from math import sqrt

import numpy as np

from ._kaiser_bessel import kaiser_bessel
from .nufft_utils import _as_1d_ints


__all__ = [
    "BeattyKernel",
    "LinearKernel",
    "KaiserBesselKernel",
    "NufftKernelBase",
]


class NufftKernelBase(ABC):

    shape = ()  # The shape of the kernel (tuple).
    kernels = []  # List of 1d functions for generating a (separable) nd kernel
    kernel_type = ""
    params = {}  # TODO: remove params ?

    @abstractmethod
    def __init__(self):
        pass

    @property
    def ndim(self):
        return len(self.shape)

    @abstractmethod
    def _initialize_kernel(self):
        """Initalize the kernels attribute.

        self.kernels must be a list containing one function per axis in
        ``self.shape``.

        Each function takes a single array of sampling points as input and
        returns the evaluated kernel at those locations. Values where the
        kernel is to be evaluated should fall in the range ``[-J/2, J/2]``
        where ``J == self.shape[axis]``.
        """

    def plot(self, real_imag=False, axes=None):
        from matplotlib import pyplot as plt

        """plot the (separable) kernel for each axis."""
        title_text = "type: {}".format(self.kernel_type)
        if axes is None:
            f, axes = plt.subplots(self.ndim, 1, sharex=True)
            axes = np.atleast_1d(axes)
        for d in range(self.ndim):
            if self.shape is not None:
                j = self.shape[d]
            else:
                j = 1
            x = np.linspace(-j / 2, j / 2, 1001)

            y = self.kernels[d](x)
            if real_imag:
                axes[d].plot(x, y.real, "k-", label="real")
                axes[d].plot(x, y.imag, "r--", label="imaginary")
            else:
                axes[d].plot(x, np.abs(y), "k-", label="magnitude")
            if d == self.ndim - 1:
                axes[d].xaxis.set_ticks([-j / 2, 0, j / 2])
                axes[d].xaxis.set_ticklabels([f"-{j/2}", 0, f"{j/2}"])
            axes[d].set_ylabel("kernel amplitude, axis {}".format(d))
            axes[d].set_title(title_text)
            axes[d].grid(True)
            axes[d].legend()
        plt.draw()
        return axes

    def __str__(self):
        repstr = "kernel type: {}\n".format(self.kernel_type)
        repstr += "kernel shape: {}\n".format(self.shape)
        return repstr

    # TODO: add __repr__


kernel_docstr = """
Interpolation kernel.

Parameters
----------
k : ndarray
    Coordinates at which to evalate the kernel. Should be within
    the range [-J/2, J/2] where J is the kernel size.
"""


def _linear_kernel(k, J):
    return (1 - abs(k / (J / 2.0))) * (abs(k) < J / 2.0)


class LinearKernel(NufftKernelBase):
    def __init__(self, shape):
        if np.isscalar(shape):
            shape = (shape,)
        self.shape = tuple(shape)
        self.kernel_type = "linear"
        self.kernels = self._initialize_kernel()

    def _initialize_kernel(self):
        # linear interpolator straw man
        kernels = []
        for d in range(self.ndim):
            kernels.append(functools.partial(_linear_kernel, J=self.shape[d]))
            kernels[-1].__doc__ = kernel_docstr
        return kernels


class KaiserBesselKernel(NufftKernelBase):

    is_kaiser_scale = None  # set during init

    def __init__(self, shape, alpha=None, m=None):
        self.kernel_type = "kb:user"
        self.params = {}

        if np.isscalar(shape):
            shape = (shape,)
        self.shape = tuple(shape)
        alpha = _as_1d_ints(alpha, self.ndim)
        m = _as_1d_ints(m, self.ndim)
        self.kernels = self._initialize_kernel(alpha, m)

    def _initialize_kernel(self, alpha, m):
        self.is_kaiser_scale = True
        if m is None or alpha is None:
            raise ValueError(
                "kwargs must contain shape, m, alpha for"
                + f"{self.kernel_type} case"
            )
        self.alpha = alpha
        self.m = m
        kernels = []
        for j, alpha, m in zip(self.shape, alpha, m):
            kernels.append(
                functools.partial(kaiser_bessel, J=j, alpha=alpha, m=m),
            )
        return kernels


class BeattyKernel(KaiserBesselKernel):

    is_kaiser_scale = None  # set during init

    def __init__(self, shape, grid_shape, os_grid_shape):
        self.kernel_type = "kb:beatty"

        if np.isscalar(shape):
            shape = (shape,)
        self.shape = tuple(shape)

        self.grid_shape = _as_1d_ints(grid_shape, self.ndim)
        self.os_grid_shape = _as_1d_ints(os_grid_shape, self.ndim)
        alpha, m = self._get_beatty_kb_params()
        self.kernels = super()._initialize_kernel(alpha, m)

    def _get_beatty_kb_params(self):
        # KB with Beatty et al parameters
        # Beatty2005:  IEEETMI 24(6):799:808
        # TODO: add proper citation
        self.is_kaiser_scale = True

        def get_alpha(j, k, n):
            # Beatty, IEEE TMI
            k_n = k / n
            return np.pi * sqrt(j ** 2 / k_n ** 2 * (k_n - 0.5) ** 2 - 0.8)

        # Eq. 5 for alpha
        alphas = [
            get_alpha(j, k, n)
            for j, k, n in zip(self.shape, self.os_grid_shape, self.grid_shape)
        ]
        m = [0] * self.ndim
        return alphas, m


# # TODO: remove DiricKernel
# class DiricKernel(NufftKernelBase):

#     def __init__(self, shape, grid_shape, os_grid_shape, phasing=None):
#         self.kernel_type = 'diric'

#         if shape is None:
#             shape = os_grid_shape
#         if np.isscalar(shape):
#             shape = (shape, )
#         self.shape = tuple(shape)
#         self.grid_shape = tuple(grid_shape)
#         self.os_grid_shape = tuple(os_grid_shape)
#         self.phasing = phasing
#         self.kernels = self._initialize_kernel()

#     def _initialize_kernel(self):
#         if not self.shape == tuple(self.os_grid_shape):
#             warnings.warn("diric inexact unless shape=os_grid_shape")

#         kernels = []
#         for n, k in zip(self.grid_shape, self.os_grid_shape):
#             if self.phasing == "real":
#                 n = 2 * ((k + 1) // 2) - 1  # trick
#             kernels.append(
#                 lambda x: (n / k * nufft_diric(x, n, k, True))
#             )
#         return kernels


# # TODO: remove this NufftKernel class
# class NufftKernel(object):
#     """ Interpolation kernel for use in the gridding stage of the NUFFT. """

#     def __init__(self, kernel_type="kb:beatty", shape=None, **kwargs):
#         self.kernel = None
#         self.is_kaiser_scale = False
#         self.params = kwargs.copy()
#         if np.isscalar(shape):
#             shape = (shape, )
#         self.shape = tuple(shape)
#         if self.shape is not None:
#             self.ndim = len(self.shape)
#         self.kernel_type = kernel_type

#     @property
#     def kernel_type(self):
#         return self._kernel_type

#     @kernel_type.setter
#     def kernel_type(self, kernel_type):
#         if isinstance(kernel_type, str):
#             self._kernel_type = kernel_type
#             self._initialize_kernel(kernel_type)
#         elif isinstance(kernel_type, (list, tuple, set)):
#             # list containing 1 kernel per dimension
#             kernel_type = list(kernel_type)
#             if isinstance(kernel_type[0], collections.Callable):
#                 if len(kernel_type) != self.ndim:
#                     raise ValueError("wrong # of kernels specified in list")
#                 self.kernels = kernel_type
#             else:
#                 raise ValueError(
#                     "kernel_type list must contain a series of "
#                     + "callable kernel functions"
#                 )
#             self._kernel_type = "inline"
#         elif isinstance(kernel_type, collections.Callable):
#             # replicate to fill list for each dim
#             self.kernels = [kernel_type] * self.ndim
#             self._kernel_type = "inline"
#         else:
#             raise ValueError(
#                 "invalid type for kernel_type: {}".format(type(kernel_type))
#             )

#     def _initialize_kernel(self, kernel_type):

#         params = self.params.copy()

#         # replicate any that were length one to ndim array
#         for k, v in list(params.items()):
#             if k in ["Kd", "Jd", "Nd"]:
#                 params[k] = _as_1d_ints(v, self.ndim)

#         ndim = self.ndim  # number of dimensions
#         Kd = params.get("Kd", None)  # oversampled image size
#         Nd = params.get("Nd", None)  # image size
#         alpha = params.get("alpha", None)
#         m = params.get("m", None)

#         kernel_type = kernel_type.lower()
#         if kernel_type == "linear":
#             # linear interpolator straw man
#             kernel_type = "inline"

#             def kernel_linear(k, J):
#                 return (1 - abs(k / (J / 2.0))) * (abs(k) < J / 2.0)

#             self.kernels = []
#             for d in range(ndim):
#                 self.kernels.append(functools.partial(kernel_linear, J=self.shape[d]))
#         elif kernel_type == "diric":  # exact interpolator
#             if (Kd is None) or (Nd is None):
#                 raise ValueError("kwargs must contain Kd, Nd for diric case")
#             if not np.all(np.equal(self.shape, Kd)):
#                 warnings.warn("diric inexact unless shape=Kd")
#             self.kernels = []
#             for d in range(ndim):
#                 N = Nd[d]
#                 K = Kd[d]
#                 if self.params.get("phasing", None) == "real":
#                     N = 2 * np.floor((K + 1) / 2.0) - 1  # trick
#                 self.kernels.append(
#                     lambda k: (N / K * nufft_diric(k, N, K, True))
#                 )
#         elif kernel_type == "kb:beatty":
#             # KB with Beatty et al parameters
#             # Beatty2005:  IEEETMI 24(6):799:808
#             self.is_kaiser_scale = True
#             if (Kd is None) or (Nd is None) or (self.shape is None):
#                 raise ValueError(
#                     "kwargs must contain Kd, Nd, shape for "
#                     + "{} case".format(kernel_type)
#                 )

#             K_N = Kd / Nd
#             # Eq. 5 for alpha
#             shape_sq = np.asarray([j * j for j in self.shape])
#             params["alpha"] = np.pi * np.sqrt(
#                 shape_sq / K_N ** 2 * (K_N - 0.5) ** 2 - 0.8
#             )
#             params["m"] = np.zeros(ndim)
#             self.kernels = []
#             for d in range(ndim):
#                 self.kernels.append(
#                     functools.partial(
#                         kaiser_bessel,
#                         J=self.shape[d],
#                         alpha=params["alpha"][d],
#                         m=params["m"][d],
#                     )
#                 )

#         elif kernel_type == "kb:user":
#             self.is_kaiser_scale = True

#             if (self.shape is None) or (m is None) or (alpha is None):
#                 raise ValueError(
#                     "kwargs must contain shape, m, alpha for"
#                     + "{} case".format(kernel_type)
#                 )

#             self.kernels = []
#             for d in range(ndim):
#                 self.kernels.append(
#                     functools.partial(
#                         kaiser_bessel, J=self.shape[d], alpha=alpha[d], m=m[d]
#                     )
#                 )
#         else:
#             raise ValueError("unknown kernel type")

#         if "alpha" in params:
#             self.alpha = params["alpha"]
#         if "beta" in params:
#             self.beta = params["beta"]
#         if "m" in params:
#             self.m = params["m"]
#         if "alpha" in params:
#             self.alpha = params["alpha"]

#         self.params = params

#     def plot(self, real_imag=False, axes=None):
#         from matplotlib import pyplot as plt

#         """plot the (separable) kernel for each axis."""
#         title_text = "type: {}".format(self.kernel_type)
#         if axes is None:
#             f, axes = plt.subplots(self.ndim, 1, sharex=True)
#             axes = np.atleast_1d(axes)
#         for d in range(self.ndim):
#             if self.shape is not None:
#                 j = self.shape[d]
#             else:
#                 j = 1
#             x = np.linspace(-j / 2, j / 2, 1001)

#             y = self.kernels[d](x)
#             if real_imag:
#                 axes[d].plot(x, y.real, "k-", label="real")
#                 axes[d].plot(x, y.imag, "r--", label="imaginary")
#             else:
#                 axes[d].plot(x, np.abs(y), "k-", label="magnitude")
#             if d == self.ndim - 1:
#                 axes[d].xaxis.set_ticks([-j / 2, j / 2])
#                 axes[d].xaxis.set_ticklabels(["-J/2", "J/2"])
#             axes[d].set_ylabel("kernel amplitude, axis {}".format(d))
#             axes[d].set_title(title_text)
#             axes[d].legend()
#         plt.draw()
#         return axes

#     def __repr__(self):
#         repstr = "NufftKernel({}, ".format(self.kernel_type)
#         for k, v in self.params.items():
#             repstr += ", {}={}".format(k, v)
#         repstr += ")"
#         return repstr

#     def __str__(self):
#         repstr = "kernel type: {}\n".format(self.kernel_type)
#         repstr += "kernel dimensions: {}\n".format(self.ndim)
#         if "kb:" in self.kernel_type:
#             repstr += "Kaiser Bessel params:\n"
#             for d in range(self.ndim):
#                 repstr += "    alpha[{}], m[{}] = {}, {}\n".format(
#                     d, d, self.alpha[d], self.m[d]
#                 )
#         return repstr
