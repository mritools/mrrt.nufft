# -*- coding: utf-8 -*-

import collections
import warnings
import functools

import numpy as np

from pyir.nufft.kaiser_bessel import kaiser_bessel

from pyir.nufft.nufft_utils import (nufft_alpha_kb_fit,
                                    nufft_best_alpha,
                                    nufft_diric,
                                    to_1d_int_array)

from pyir.utils import is_string_like

__all__ = ['NufftKernel', ]

kernel_types = ['minmax:kb',
                'linear',
                'diric',
                'kb:beatty'
                'kb:minmax'
                'minmax:unif',
                'minmax:tuned',
                ]


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
            # list containing 1 kernel per dimension
            kernel_type = list(kernel_type)
            if isinstance(kernel_type[0], collections.Callable):
                if len(kernel_type) != self.ndim:
                    raise ValueError(
                        'wrong # of kernels specified in list')
                self.kernel = kernel_type
            else:
                raise ValueError('kernel_type list must contain a series of ' +
                                 'callable kernel functions')
            self._kernel_type = 'inline'
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
                    params[k] = to_1d_int_array(v)
                    plen = len(params[k])
                    if plen > max_len:
                        max_len = plen
            self.ndim = max_len
        else:
            self.ndim = params.pop('ndim')

        # replicate any that were length one to ndim array
        for k, v in list(params.items()):
            if k in ['Kd', 'Jd', 'Nd']:
                params[k] = to_1d_int_array(v, self.ndim)
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
        kernel_type = kernel_type.lower()
        if kernel_type == 'linear':
            kernel_type = 'inline'

            def kernel_linear(k, J):
                return (1 - abs(k / (J / 2.))) * (abs(k) < J / 2.)
            self.kernel = []
            for d in range(ndim):
                self.kernel.append(functools.partial(kernel_linear,
                                                     J=Jd[d]))
        elif kernel_type == 'diric':   # exact interpolator
            if (Kd is None) or (Nd is None):
                raise ValueError("kwargs must contain Kd, Nd for diric case")
            if not np.all(np.equal(Jd, Kd)):
                warnings.warn('diric inexact unless Jd=Kd')
            self.kernel = []
            for d in range(ndim):
                N = Nd[d]
                K = Kd[d]
                if self.params.get('phasing', None) == 'real':
                    N = 2 * np.floor((K + 1) / 2.) - 1  # trick
                self.kernel.append(lambda k: (
                    N / K * nufft_diric(k, N, K, True)))
        elif kernel_type == 'kb:beatty':
            # KB with Beatty et al parameters
            # Beatty2005:  IEEETMI 24(6):799:808
            self.is_kaiser_scale = True
            if (Kd is None) or (Nd is None) or (Jd is None):
                raise ValueError("kwargs must contain Kd, Nd, Jd for " +
                                 "{} case".format(kernel_type))

            # warn if user specified specific alpha, m
            if (kb_m is not None) or (kb_alf is not None):
                warnings.warn(
                    'kb:beatty:  user supplied kb_alf and kb_m ignored')

            K_N = Kd / Nd
            # Eq. 5 for alpha
            params['kb_alf'] = \
                np.pi * np.sqrt(Jd ** 2 / K_N ** 2 * (K_N - 0.5) ** 2 - 0.8)
            params['kb_m'] = np.zeros(ndim)
            self.kernel = []
            for d in range(ndim):
                self.kernel.append(
                    functools.partial(kaiser_bessel,
                                      J=Jd[d],
                                      alpha=params['kb_alf'][d],
                                      kb_m=params['kb_m'][d]))

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
            for d in range(ndim):
                alf = 2.34 * Jd[d]
                m = 0
                self.kernel.append(
                    functools.partial(kaiser_bessel, J=Jd[d], alpha=alf,
                                      kb_m=m))
                params['kb_alf'].append(alf)
                params['kb_m'].append(0)

        elif kernel_type == 'kb:user':  # KB with Beatty et al parameters
            self.is_kaiser_scale = True

            if (Jd is None) or (kb_m is None) or (kb_alf is None):
                raise ValueError("kwargs must contain Jd, kb_m, kb_alf for" +
                                 "{} case".format(kernel_type))

            self.kernel = []
            for d in range(ndim):
                self.kernel.append(functools.partial(kaiser_bessel,
                                                     J=Jd[d],
                                                     alpha=kb_alf[d],
                                                     kb_m=kb_m[d]))

        # minmax interpolator with KB scaling factors (recommended default)
        elif kernel_type == 'minmax:kb':
            if (Kd is None) or (Nd is None) or (Jd is None) or (Nmid is None):
                raise ValueError("kwargs must contain Kd, Nd, Jd, Nmid for " +
                                 "{} case".format(kernel_type))
            params['alpha'] = []
            params['beta'] = []
            for d in range(ndim):
                [al, be] = nufft_alpha_kb_fit(N=Nd[d], J=Jd[d], K=Kd[d],
                                              Nmid=Nmid[d])
                params['alpha'].append(al)
                params['beta'].append(be)

        # minmax interpolator with numerically "tuned" scaling factors
        elif kernel_type == 'minmax:tuned':  # TODO
            if (Kd is None) or (Nd is None) or (Jd is None):
                raise ValueError("kwargs must contain Kd, Nd, Jd for " +
                                 "{} case".format(kernel_type))
            params['alpha'] = []
            params['beta'] = []
            for d in range(ndim):
                [al, be, ok] = nufft_best_alpha(J=Jd[d], L=0,
                                                K_N=Kd[d] / Nd[d])
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
            for d in range(ndim):
                params['alpha'].append(1.)
                params['beta'].append(0.)
        elif 'mols' in kernel_type:
            pass
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
        from matplotlib import pyplot as plt
        from pyir.nufft.nufft import _nufft_table_make1
        """plot the (separable) kernel for each axis."""
        title_text = 'type: {}'.format(self.kernel_type)
        if axes is None:
            f, axes = plt.subplots(self.ndim, 1, sharex=True)
            axes = np.atleast_1d(axes)
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
                y = self.kernel[d](x)
            axes[d].plot(x, np.abs(y), 'k-', label='magnitude')
            if d == self.ndim - 1:
                axes[d].xaxis.set_ticks([-J/2, J/2])
                axes[d].xaxis.set_ticklabels(['-J/2', 'J/2'])
            axes[d].set_ylabel('kernel amplitude, axis {}'.format(d))
            axes[d].set_title(title_text)
            axes[d].legend()
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
