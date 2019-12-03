import functools

import numpy as np
import pytest

from mrrt.nufft._kaiser_bessel import kaiser_bessel, kaiser_bessel_ft
from mrrt.utils import config

if config.have_cupy:
    import cupy

    all_xp = [np, cupy]
else:
    all_xp = [np]


@pytest.mark.parametrize("xp", all_xp)
def test_kaiser_bessel(xp, show_figure=False):
    J = 8
    alpha = 2.34 * J
    x = xp.linspace(-(J + 1) / 2.0, (J + 1) / 2.0, 1001)
    mlist = [-4, 0, 2, 7]
    leg = []
    yy = xp.zeros((len(x), len(mlist)))
    for i, m in enumerate(mlist):
        yy[:, i] = kaiser_bessel(x, J, alpha, m)
        leg.append("m = %d" % m)
        func = functools.partial(kaiser_bessel, alpha=alpha, m=m)
        yf = func(x, J)
        xp.testing.assert_array_equal(yf, yy[:, i])

    if show_figure and xp is np:  # skip plotting if arrays are on the GPU
        # create plots similar to those in Fessler's matlab toolbox
        from matplotlib import pyplot as plt

        plt.figure()
        l1, l2, l3, l4 = plt.plot(
            x,
            yy[:, 0],
            "c-",
            x,
            yy[:, 1],
            "y-",
            x,
            yy[:, 2],
            "m-",
            x,
            yy[:, 3],
            "g-",
        )
        plt.legend((l1, l2, l3, l4), leg, loc="upper right")
        plt.xlabel(r"$\kappa$")
        plt.ylabel(r"F($\kappa$)")
        plt.title(r"KB functions: J=%g $\alpha$=%g" % (J, alpha))
        plt.axis("tight")
        plt.show()


@pytest.mark.parametrize("xp", all_xp)
def test_kaiser_bessel_ft(xp, show_figure=False):
    J = 5
    alpha = 6.8
    N = 2 ** 10
    x = xp.arange(-N / 2.0, N / 2.0) / float(N) * (J + 3) / 2.0
    dx = x[1] - x[0]
    du = 1 / float(N) / float(dx)
    u = xp.arange(-N / 2.0, N / 2.0) * du
    uu = 1.5 * xp.linspace(-1, 1, 201)

    mlist = [-2, 0, 2, 7]
    leg = []
    yy = xp.zeros((len(x), len(mlist)))
    Yf = xp.zeros_like(yy)
    Y = xp.zeros_like(yy)
    Yu = xp.zeros((len(uu), len(mlist)))
    fftshift = xp.fft.fftshift
    fft = xp.fft.fft
    for ii, m in enumerate(mlist):
        m = mlist[ii]
        yy[:, ii] = kaiser_bessel(x, J, alpha, m)
        Yf[:, ii] = xp.real(fftshift(fft(fftshift(yy[:, ii])))) * dx
        if m < -1:
            with pytest.warns(UserWarning):
                Y[:, ii] = kaiser_bessel_ft(u, J, alpha, m, 1)
                Yu[:, ii] = kaiser_bessel_ft(uu, J, alpha, m, 1)
        else:
            Y[:, ii] = kaiser_bessel_ft(u, J, alpha, m, 1)
            Yu[:, ii] = kaiser_bessel_ft(uu, J, alpha, m, 1)
        leg.append("m=%d" % m)

    if show_figure and xp is np:  # skip plotting if arrays are on the GPU
        # create plots similar to those in Fessler's matlab toolbox
        from matplotlib import pyplot as plt

        if False:
            plt.figure()
            l1, l2, l3 = plt.plot(
                u, Yf[:, 2], "cx", u, Y[:, 2], "yo", uu, Yu[:, 2], "y-"
            )
            plt.legend(
                (l1, l2, l3), ["FFT", "FT coarse", "FT fine"], loc="upper right"
            )
            plt.axis("tight")
            plt.grid(True)  # , axisx(range(uu)), grid

            plt.figure()
            plt.plot(
                x,
                yy[:, 0],
                "c-",
                x,
                yy[:, 1],
                "y-",
                x,
                yy[:, 2],
                "m-",
                x,
                yy[:, 3],
                "g-",
            )
        plt.figure()
        l1, l2, l3, l4 = plt.plot(
            uu,
            Yu[:, 0],
            "c-",
            uu,
            Yu[:, 1],
            "y-",
            uu,
            Yu[:, 2],
            "m-",
            uu,
            Yu[:, 3],
            "g-",
        )

        plt.axis("tight")
        plt.legend((l1, l2, l3, l4), leg, loc="upper right")
        plt.plot(u, Yf[:, 1], "y.")
        plt.xlabel("u")
        plt.ylabel("Y(u)")
        plt.title(r"KB FT: $\alpha$=%g" % alpha)

    return
