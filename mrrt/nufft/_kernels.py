"""Convolution kernel to be used for Non-Uniform FFT.

"""
from abc import ABC, abstractmethod

import functools
from math import sqrt

import numpy as np

from ._kaiser_bessel import kaiser_bessel
from ._utils import _as_1d_ints


__all__ = ["BeattyKernel", "KaiserBesselKernel", "NufftKernelBase"]


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
                functools.partial(kaiser_bessel, J=j, alpha=alpha, m=m)
            )
            kernels[-1].__doc__ = kernel_docstr
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
