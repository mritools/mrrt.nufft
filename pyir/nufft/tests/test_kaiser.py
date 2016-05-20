# -*- coding: utf-8 -*-
import functools
import numpy as np
from numpy.testing import dec


#def _test_kaiser(compare_to_matlab=False):
#    print("\n\nRunning _kaiser_bessel_test: ")
#    _kaiser_bessel_test()
#    print("\n\nRunning _kaiser_bessel_ft_test: ")
#    _kaiser_bessel_ft_test()
#    if compare_to_matlab:
#        print("\n\nRunning _kaiser_matlab_compare: ")
#        _kaiser_matlab_compare()
#


def test_kaiser_bessel(verbose=False):
    from pyir.nufft.kaiser_bessel import kaiser_bessel
    J = 8
    alpha = 2.34 * J
    x = np.linspace(-(J + 1) / 2.0, (J + 1) / 2.0, 1001)
    mlist = [-4, 0, 2, 7]
    leg = []
    yy = np.zeros((len(x), len(mlist)))
    for i, kb_m in enumerate(mlist):
        yy[:, i] = kaiser_bessel(x, J, alpha, kb_m)
        leg.append('m = %d' % kb_m)
        func = functools.partial(kaiser_bessel, 0, alpha, kb_m)
        yf = func(x, J)
        if (yf != yy[:, i]).any():
            print('ERROR in %s:  bug' % __name__)

    yb = kaiser_bessel(x, J, 'best', [], 2)
    leg.append('best')
    if verbose:
        from matplotlib import pyplot as plt
        plt.figure()
        l1, l2, l3, l4, l5 = plt.plot(x, yy[:, 0], 'c-',
                                      x, yy[:, 1], 'y-',
                                      x, yy[:, 2], 'm-',
                                      x, yy[:, 3], 'g-',
                                      x, yb, 'r--')
        plt.legend((l1, l2, l3, l4, l5), leg, loc='upper right')
        plt.xlabel(r'$\kappa$')
        plt.ylabel(r'F($\kappa$)')
        plt.title(r'KB functions: J=%g $\alpha$=%g' % (J, alpha))
        plt.axis('tight')
        plt.show()


def test_kaiser_bessel_ft(verbose=False):
    from numpy.fft import fftshift, fft
    from pyir.nufft.kaiser_bessel import kaiser_bessel, kaiser_bessel_ft
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


@dec.skipif(True)  # skip this as it requires Matlab
def _kaiser_matlab_compare(show_figures=True):
    from pymatbridge import Matlab
    from pyir.nufft.kaiser_bessel import kaiser_bessel, kaiser_bessel_ft
    if show_figures:
        from matplotlib import pyplot as plt

    mlab = Matlab(executable='/usr/local/bin/matlab')
    mlab.start()

    J = 5
    alpha = 6.8
    N = 2 ** 10
    x = np.arange(-N / 2., N / 2.) / float(N) * (J + 3) / 2.
    dx = x[1] - x[0]
    du = 1 / float(N) / float(dx)
    u = np.arange(-N / 2., N / 2.) * du
    uu = 1.5 * np.linspace(-1, 1, 201)
    mlist = [-2, 0, 2, 7]
    for ii, kb_m in enumerate(mlist):
        mcmd = "J = %d; alpha = %g; N = %d; kb_m=%d; " % (J, alpha, N, kb_m)
        mcmd += "x = [-N/2:N/2-1]'/N * (J+3)/2; "
        mcmd += "dx = x(2) - x(1); "
        mcmd += "du = 1 / N / dx; "
        mcmd += "u = [-N/2:N/2-1]' * du; "
        mcmd += "uu = 1.5*linspace(-1,1,201)';"
        mcmd += "k=kaiser_bessel(x, J, alpha, kb_m);"
        mcmd += "kft=kaiser_bessel_ft(u, J, alpha, kb_m, 1);"
        mcmd += "kft_uu=kaiser_bessel_ft(uu, J, alpha, kb_m, 1);"
        res = mlab.run_code(mcmd)
        k = []
        kft = []
        kft_uu = []
        exec('k=np.array(%s)' % mlab.get_variable('k'))
        exec('kft=np.array(%s)' % mlab.get_variable('kft'))
        exec('kft_uu=np.array(%s)' % mlab.get_variable('kft_uu'))
        k_py = kaiser_bessel(x, J, alpha, kb_m)
        kft_py = kaiser_bessel_ft(u, J, alpha, kb_m, 1)
        kft_uu_py = kaiser_bessel_ft(uu, J, alpha, kb_m, 1)
        diff_k = np.max(k_py - k)
        diff_kft = np.max(kft_py - kft)
        diff_kft_uu = np.max(kft_uu_py - kft_uu)
        assert(diff_k < 1e-8)
        assert(diff_kft < 1e-8)
        assert(diff_kft_uu < 1e-8)
        if show_figures:
            plt.figure(
                1 + 3 * ii), plt.plot(
                x, k, 'k.', x, k_py, 'k'), plt.title(
                'm = %d, max diff = %g' %
                (kb_m, diff_k))
            plt.figure(
                2 + 3 * ii), plt.plot(
                u, kft, 'k.', u, kft_py, 'k'), plt.title(
                'm = %d, max diff = %g' %
                (kb_m, diff_kft))
            plt.figure(
                3 + 3 * ii), plt.plot(
                uu, kft_uu, 'k.', uu, kft_uu_py, 'k'), plt.title(
                'm = %d, max diff = %g' %
                (kb_m, diff_kft_uu))
    # res=mlab.run_func('kaiser_bessel.m',{'x' : x.tolist(), 'J' : J,
    #                   'alpha' : alpha, 'kb_m' : 0, 'K_N' : 2})
    mlab.stop()

    # res=mlab.run_code(mcmd)
    # ip=get_ipython()
    # pymat.load_ipython_extension(ip)
    return


