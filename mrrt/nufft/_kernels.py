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
import collections
import warnings
import functools

import numpy as np

from ._kaiser_bessel import kaiser_bessel
from ._simple_kernels import nufft_diric
from .nufft_utils import _as_1d_ints


__all__ = ["NufftKernel"]

kernel_types = ["linear", "diric", "kb:beatty", "kb:user"]


class NufftKernel(object):
    """ Interpolation kernel for use in the gridding stage of the NUFFT. """

    def __init__(self, kernel_type="kb:beatty", **kwargs):
        self.kernel = None
        self.is_kaiser_scale = False
        self.params = kwargs.copy()
        self.kernel_type = kernel_type

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, kernel_type):
        if isinstance(kernel_type, str):
            self._kernel_type = kernel_type
            self._initialize_kernel(kernel_type)
        elif isinstance(kernel_type, (list, tuple, set)):
            # list containing 1 kernel per dimension
            kernel_type = list(kernel_type)
            if isinstance(kernel_type[0], collections.Callable):
                if len(kernel_type) != self.ndim:
                    raise ValueError("wrong # of kernels specified in list")
                self.kernel = kernel_type
            else:
                raise ValueError(
                    "kernel_type list must contain a series of "
                    + "callable kernel functions"
                )
            self._kernel_type = "inline"
        elif isinstance(kernel_type, collections.Callable):
            # replicate to fill list for each dim
            self.kernel = [kernel_type] * self.ndim
            self._kernel_type = "inline"
        else:
            raise ValueError(
                "invalid type for kernel_type: {}".format(type(kernel_type))
            )

    def _initialize_kernel(self, kernel_type):

        params = self.params.copy()

        # if no dimensions specified, using longest among Kd, Jd, Nd
        if params.get("ndim", None) is None:
            max_len = 0
            for k, v in list(params.items()):
                if k in ["Kd", "Jd", "Nd"]:
                    params[k] = _as_1d_ints(v)
                    plen = len(params[k])
                    if plen > max_len:
                        max_len = plen
            self.ndim = max_len
        else:
            self.ndim = params.pop("ndim")

        # replicate any that were length one to ndim array
        for k, v in list(params.items()):
            if k in ["Kd", "Jd", "Nd"]:
                params[k] = _as_1d_ints(v, self.ndim)
            # n_mid is not necessarily an integer, so handle it manually
            if "n_mid" in params and len(params["n_mid"]) < self.ndim:
                if len(params["n_mid"]) > 1:
                    raise ValueError("n_mid dimension mismatch")
                else:
                    params["n_mid"] = np.asarray(
                        [params["n_mid"][0]] * self.ndim
                    )

        ndim = self.ndim  # number of dimensions
        Kd = params.get("Kd", None)  # oversampled image size
        Jd = params.get("Jd", None)  # kernel size
        Nd = params.get("Nd", None)  # image size
        kb_alf = params.get("kb_alf", None)
        kb_m = params.get("kb_m", None)

        # linear interpolator straw man
        kernel_type = kernel_type.lower()
        if kernel_type == "linear":
            kernel_type = "inline"

            def kernel_linear(k, J):
                return (1 - abs(k / (J / 2.0))) * (abs(k) < J / 2.0)

            self.kernel = []
            for d in range(ndim):
                self.kernel.append(functools.partial(kernel_linear, J=Jd[d]))
        elif kernel_type == "diric":  # exact interpolator
            if (Kd is None) or (Nd is None):
                raise ValueError("kwargs must contain Kd, Nd for diric case")
            if not np.all(np.equal(Jd, Kd)):
                warnings.warn("diric inexact unless Jd=Kd")
            self.kernel = []
            for d in range(ndim):
                N = Nd[d]
                K = Kd[d]
                if self.params.get("phasing", None) == "real":
                    N = 2 * np.floor((K + 1) / 2.0) - 1  # trick
                self.kernel.append(
                    lambda k: (N / K * nufft_diric(k, N, K, True))
                )
        elif kernel_type == "kb:beatty":
            # KB with Beatty et al parameters
            # Beatty2005:  IEEETMI 24(6):799:808
            self.is_kaiser_scale = True
            if (Kd is None) or (Nd is None) or (Jd is None):
                raise ValueError(
                    "kwargs must contain Kd, Nd, Jd for "
                    + "{} case".format(kernel_type)
                )

            K_N = Kd / Nd
            # Eq. 5 for alpha
            params["kb_alf"] = np.pi * np.sqrt(
                Jd ** 2 / K_N ** 2 * (K_N - 0.5) ** 2 - 0.8
            )
            params["kb_m"] = np.zeros(ndim)
            self.kernel = []
            for d in range(ndim):
                self.kernel.append(
                    functools.partial(
                        kaiser_bessel,
                        J=Jd[d],
                        alpha=params["kb_alf"][d],
                        kb_m=params["kb_m"][d],
                    )
                )

        elif kernel_type == "kb:user":
            self.is_kaiser_scale = True

            if (Jd is None) or (kb_m is None) or (kb_alf is None):
                raise ValueError(
                    "kwargs must contain Jd, kb_m, kb_alf for"
                    + "{} case".format(kernel_type)
                )

            self.kernel = []
            for d in range(ndim):
                self.kernel.append(
                    functools.partial(
                        kaiser_bessel, J=Jd[d], alpha=kb_alf[d], kb_m=kb_m[d]
                    )
                )
        else:
            raise ValueError("unknown kernel type")

        if "alpha" in params:
            self.alpha = params["alpha"]
        if "beta" in params:
            self.beta = params["beta"]
        if "kb_m" in params:
            self.kb_m = params["kb_m"]
        if "kb_alf" in params:
            self.kb_alf = params["kb_alf"]

        self.params = params

    def plot(self, real_imag=False, axes=None):
        from matplotlib import pyplot as plt

        """plot the (separable) kernel for each axis."""
        title_text = "type: {}".format(self.kernel_type)
        if axes is None:
            f, axes = plt.subplots(self.ndim, 1, sharex=True)
            axes = np.atleast_1d(axes)
        for d in range(self.ndim):
            if "Jd" in self.params:
                J = self.params["Jd"][d]
            else:
                J = 1
            x = np.linspace(-J / 2, J / 2, 1001)

            y = self.kernel[d](x)
            if real_imag:
                axes[d].plot(x, y.real, "k-", label="real")
                axes[d].plot(x, y.imag, "r--", label="imaginary")
            else:
                axes[d].plot(x, np.abs(y), "k-", label="magnitude")
            if d == self.ndim - 1:
                axes[d].xaxis.set_ticks([-J / 2, J / 2])
                axes[d].xaxis.set_ticklabels(["-J/2", "J/2"])
            axes[d].set_ylabel("kernel amplitude, axis {}".format(d))
            axes[d].set_title(title_text)
            axes[d].legend()
        plt.draw()
        return axes

    def __repr__(self):
        repstr = "NufftKernel({}, ".format(self.kernel_type)
        for k, v in self.params.items():
            repstr += ", {}={}".format(k, v)
        repstr += ")"
        return repstr

    def __str__(self):
        repstr = "kernel type: {}\n".format(self.kernel_type)
        repstr += "kernel dimensions: {}\n".format(self.ndim)
        if "kb:" in self.kernel_type:
            repstr += "Kaiser Bessel params:\n"
            for d in range(self.ndim):
                repstr += "    alpha[{}], m[{}] = {}, {}\n".format(
                    d, d, self.kb_alf[d], self.kb_m[d]
                )
        return repstr
