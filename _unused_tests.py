import os
from os.path import join as pjoin

import numpy as np

import pyir.nufft
from pyir.nufft import dtft
from pyir.nufft.nufft import NufftBase, nufft_forward
from pyir.utils import max_percent_diff

pkg_dir = os.path.dirname(pyir.nufft.__file__)
data_dir = pjoin(pkg_dir, "data", "mat_files")


def _nufft_testdata(test3d=False, initialize_from_Matlab=False, random_seed=0):
    from pyir.nufft._minmax import nufft2_err_mm

    if test3d:
        Jd = np.array([5, 4, 4])
        Nd = np.array([23, 15, 19])
        alpha_user = [1, 1, 1]  # default alpha, beta
        beta_user = [0.5, 0.5, 0.5]
    else:
        Jd = np.array([5, 6])
        # Nd = np.array([60, 75])
        Nd = np.array([60, 76])
        alpha_user = [1, 1]
        beta_user = [0.5, 0.5]

    Kd = 2 * Nd
    gam = 2 * np.pi / Kd
    n_shift = np.zeros(Nd.shape)

    if True:
        print(
            "err alf1 %g best %g"
            % (
                nufft2_err_mm(
                    "all", Nd[0], Nd[1], Jd[0], Jd[1], Kd[0], Kd[1], [1]
                )[0].max(),
                nufft2_err_mm(
                    "all", Nd[0], Nd[1], Jd[0], Jd[1], Kd[0], Kd[1], "best"
                )[0].max(),
            )
        )

    if initialize_from_Matlab:
        from os.path import join as pjoin
    else:
        rstate = np.random.RandomState(random_seed)
        x = rstate.standard_normal(tuple(Nd))
        # nufft_dir = os.path.dirname(nufft_forward.__globals__['__file__'])

    # TODO: fix so don't have to convert to complex manually
    # x = np.asarray(x,dtype=np.complex64)  #TODO: Need to fix

    if len(Nd) == 3:  # nonuniform frequencies
        [o1, o2, o3] = np.meshgrid(
            np.linspace(0, gam[0], 11),
            np.linspace(0, gam[1], 13),
            np.linspace(0, gam[2], 5),
            indexing="ij",
        )

        om1 = np.array(list(o1.ravel(order="F")) + [0, 7.2, 2.6, 3.3])
        om2 = np.array(list(o2.ravel(order="F")) + [0, 4.2, -1, 5.5])
        om3 = np.array(list(o3.ravel(order="F")) + [0, 1.1, -2, 3.4])

        om = np.hstack(
            (om1[:, np.newaxis], om2[:, np.newaxis], om3[:, np.newaxis])
        )

        # ignore x, om from above and load exact ones from Matlab for
        # comparison
        if initialize_from_Matlab:
            from scipy.io import loadmat

            f = loadmat(pjoin(data_dir, "nufft_test3D.mat"))
            # get same random vector & om as generated in Matlab
            x = f["x"]

    else:
        o1 = np.linspace(-3 * gam[0], 3 * gam[0], 41)
        o2 = np.linspace(-2 * gam[1], gam[1], 31)
        [o1, o2] = np.meshgrid(o1, o2, indexing="ij")
        om1 = np.array(list(o1.ravel(order="F")) + [0, 7.2, 2.6, 3.3])
        om2 = np.array(list(o2.ravel(order="F")) + [0, 4.2, -1, 5.5])
        om = np.hstack((om1[:, np.newaxis], om2[:, np.newaxis]))

        # ignore x, om from above and load exact ones from Matlab for
        # comparison
        if initialize_from_Matlab:
            from scipy.io import loadmat

            # f = loadmat('nufft.mat') #get same random vector & om as
            # generated in Matlab
            f = loadmat(pjoin(data_dir, "nufft_test2D.mat"))
            # get same random vector & om as generated in Matlab
            x = f["x"]
    return om, x, Nd, Jd, Kd, n_shift


def _nufft_test(
    test3d=False, initialize_from_Matlab=False, make_fig=False, random_seed=0
):
    # from numpy.fft import fft2
    s = {}
    Y = {}
    # x = randn(Nd)

    om, x, Nd, Jd, Kd, n_shift = _nufft_testdata(
        test3d=test3d,
        initialize_from_Matlab=initialize_from_Matlab,
        random_seed=random_seed,
    )

    Y["d"] = dtft(x, om, n_shift=n_shift)

    try:
        s["tab"] = NufftBase(
            om=om,
            Nd=Nd,
            Jd=Jd,
            Kd=Kd,
            n_shift=n_shift,
            mode="table1",
            Ld=2 ** 12,
            kernel_type="minmax:kb",
        )  # TODO:  'table' case
        Y["tab"] = nufft_forward(s["tab"], x)
        print(
            "table0        max%%diff = %g" % max_percent_diff(Y["d"], Y["tab"])
        )
    except:
        # warnings.warn('table-based NUFFT failed')
        raise ValueError("table-based NUFFT failed")

    s["mmkb"] = NufftBase(
        om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift, kernel_type="minmax:kb"
    )
    Y["mmkb"] = nufft_forward(s["mmkb"], x)
    print("minmax:kb    max%%diff = %g" % max_percent_diff(Y["d"], Y["mmkb"]))

    if True:
        # test multiple input case
        x3 = x[..., np.newaxis]
        x3 = np.concatenate((x3, x3, x3), axis=x.ndim)

        Y3 = nufft_forward(s["mmkb"], x3)
        print(
            "multi    max%%diff = %g" % max_percent_diff(Y["mmkb"], Y3[..., -1])
        )

    # kaiser with minmax best alpha,m
    s["kb"] = NufftBase(
        om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift, kernel_type="kb:minmax"
    )
    Y["kb"] = nufft_forward(s["kb"], x)
    print("kaiser        max%%diff = %g" % max_percent_diff(Y["d"], Y["kb"]))

    # kaiser with user-specified supoptimal alpha,m for comparison
    kernel_kwargs = {}
    kernel_kwargs["kb_alf"] = s["kb"].kernel.params["kb_alf"] + 0.1 * np.ones(
        np.size(s["kb"].kernel.params["kb_alf"])
    )
    kernel_kwargs["kb_m"] = s["kb"].kernel.params["kb_m"]
    s["ku"] = NufftBase(
        om=om,
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        kernel_type="kb:user",
        kernel_kwargs=kernel_kwargs,
    )
    Y["ku"] = nufft_forward(s["ku"], x)
    print("kaiser-user    max%%diff = %g" % max_percent_diff(Y["d"], Y["ku"]))

    s["mmtu"] = NufftBase(
        om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift, kernel_type="minmax:tuned"
    )
    Y["mmtu"] = nufft_forward(s["mmtu"], x)
    print(
        "minmax:tuned    max%%diff = %g" % max_percent_diff(Y["d"], Y["mmtu"])
    )

    kernel_kwargs = {}
    kernel_kwargs["alpha"] = alpha_user
    kernel_kwargs["beta"] = beta_user
    s["mm"] = NufftBase(
        om=om,
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        kernel_type="minmax:user",
        kernel_kwargs=kernel_kwargs,
    )
    Y["mm"] = nufft_forward(s["mm"], x)
    print("minmax:user    max%%diff = %g" % max_percent_diff(Y["d"], Y["mm"]))

    if make_fig:
        from matplotlib import pyplot as plt

        plt.close("all")
        plt.figure(), plt.plot(np.abs(Y["d"]))
        plt.figure(), plt.plot(np.abs(Y["mmkb"]))
        plt.figure(), plt.plot(np.abs(Y["kb"]))
        plt.figure(), plt.plot(np.abs(Y["ku"]))
        plt.figure(), plt.plot(np.abs(Y["mmtu"]))
        plt.figure(), plt.plot(np.abs(Y["mm"]))

    if False:  # 'linear'
        # TODO: broken
        s["lin"] = NufftBase(
            om=om, Nd=Nd, Jd=Jd, Kd=Kd, n_shift=n_shift, kernel_type="linear"
        )
        Y["lin"] = nufft_forward(s["lin"], x)
        print("user-linear max%%diff = %g" % max_percent_diff(Y["d"], Y["lin"]))

    if False:  # test 'diric'
        # Note: for diric case, Jd = Kd
        # TODO: broken
        s["diric"] = NufftBase(
            om=om, Nd=Nd, Jd=Kd, Kd=Kd, n_shift=n_shift, kernel_type="diric"
        )
        Y["diric"] = nufft_forward(s["diric"], x)
        print(
            "user-linear max%%diff = %g" % max_percent_diff(Y["d"], Y["diric"])
        )

    return Y
