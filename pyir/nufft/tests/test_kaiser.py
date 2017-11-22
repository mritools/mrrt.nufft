# -*- coding: utf-8 -*-
import functools

import numpy as np
from numpy.fft import fftshift, fft
from numpy.testing import dec, assert_array_equal, run_module_suite

from pyir.nufft._kaiser_bessel import kaiser_bessel, kaiser_bessel_ft


def test_kaiser_bessel(verbose=False):
    J = 8
    alpha = 2.34 * J
    x = np.linspace(-(J + 1) / 2.0, (J + 1) / 2.0, 1001)
    mlist = [-4, 0, 2, 7]
    leg = []
    yy = np.zeros((len(x), len(mlist)))
    for i, kb_m in enumerate(mlist):
        yy[:, i] = kaiser_bessel(x, J, alpha, kb_m)
        leg.append('m = %d' % kb_m)
        func = functools.partial(kaiser_bessel, alpha=alpha, kb_m=kb_m)
        yf = func(x, J)
        assert_array_equal(yf, yy[:, i])

    if verbose:
        from matplotlib import pyplot as plt
        plt.figure()
        l1, l2, l3, l4 = plt.plot(x, yy[:, 0], 'c-',
                                  x, yy[:, 1], 'y-',
                                  x, yy[:, 2], 'm-',
                                  x, yy[:, 3], 'g-')
        plt.legend((l1, l2, l3, l4), leg, loc='upper right')
        plt.xlabel(r'$\kappa$')
        plt.ylabel(r'F($\kappa$)')
        plt.title(r'KB functions: J=%g $\alpha$=%g' % (J, alpha))
        plt.axis('tight')
        plt.show()


def test_kaiser_bessel_ft(verbose=False):
    J = 5
    alpha = 6.8
    N = 2 ** 10
    x = np.arange(-N / 2., N / 2.) / float(N) * (J + 3) / 2.
    dx = x[1] - x[0]
    du = 1 / float(N) / float(dx)
    u = np.arange(-N / 2., N / 2.) * du
    uu = 1.5 * np.linspace(-1, 1, 201)

    mlist = [-2, 0, 2, 7]
    leg = []
    yy = np.zeros((len(x), len(mlist)))
    Yf = np.zeros_like(yy)
    Y = np.zeros_like(yy)
    Yu = np.zeros((len(uu), len(mlist)))
    for ii, kb_m in enumerate(mlist):
        kb_m = mlist[ii]
        yy[:, ii] = kaiser_bessel(x, J, alpha, kb_m)
        Yf[:, ii] = np.real(fftshift(fft(fftshift(yy[:, ii])))) * dx
        Y[:, ii] = kaiser_bessel_ft(u, J, alpha, kb_m, 1)
        Yu[:, ii] = kaiser_bessel_ft(uu, J, alpha, kb_m, 1)
        leg.append('m=%d' % kb_m)

    if verbose:
        from matplotlib import pyplot as plt
        if False:
            plt.figure()
            l1, l2, l3 = plt.plot(
                u, Yf[
                    :, 2], 'cx', u, Y[
                    :, 2], 'yo', uu, Yu[
                    :, 2], 'y-')
            plt.legend(
                (l1, l2, l3), [
                    'FFT', 'FT coarse', 'FT fine'], loc='upper right')
            plt.axis('tight')
            plt.grid(True)  # , axisx(range(uu)), grid

            plt.figure()
            plt.plot(x, yy[:, 0], 'c-', x, yy[:, 1], 'y-',
                     x, yy[:, 2], 'm-', x, yy[:, 3], 'g-')
        plt.figure()
        l1, l2, l3, l4 = plt.plot(uu, Yu[:, 0], 'c-', uu, Yu[:, 1], 'y-',
                                  uu, Yu[:, 2], 'm-', uu, Yu[:, 3], 'g-')

        plt.axis('tight')
        plt.legend((l1, l2, l3, l4), leg, loc='upper right')
        plt.hold('on')
        plt.plot(u, Yf[:, 1], 'y.')
        plt.hold('off')
        plt.xlabel('u')
        plt.ylabel('Y(u)')
        plt.title(r'KB FT: $\alpha$=%g' % alpha)

    return

if __name__ == "__main__":
    run_module_suite()
