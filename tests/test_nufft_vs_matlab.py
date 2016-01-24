# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 18:21:05 2014

@author: lee8rx
"""
from __future__ import division, print_function, absolute_import

import time
import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import scipy.io

from matplotlib import pyplot as plt

import PyIRT.nufft
from PyIRT.nufft.nufft import NufftKernel, NufftBase, nufft_forward, nufft_adj
from grl_utils import max_percent_diff

data_dir = pjoin(os.path.dirname(PyIRT.nufft.__file__),
                 'tests', 'data', 'mat_files')


def test_nufft_table_make1():
    from PyIRT.nufft.nufft import _nufft_table_make1
    from numpy.testing import assert_almost_equal
    for N in [33, 64]:
        for phasing in ['real', 'complex']:
            h0, t0 = _nufft_table_make1(how='slow', N=N, J=6, K=2 * N, L=2048,
                                        phasing=phasing,
                                        kernel_type='kb:beatty',
                                        kernel_kwargs={})
            h1, t1 = _nufft_table_make1(how='fast', N=N, J=6, K=2 * N, L=2048,
                                        phasing=phasing,
                                        kernel_type='kb:beatty',
                                        kernel_kwargs={})
            h2, t2 = _nufft_table_make1(how='ratio', N=N, J=6, K=2 * N, L=2048,
                                        phasing=phasing,
                                        kernel_type='kb:beatty',
                                        kernel_kwargs={})
            assert_almost_equal(h0, h1)
            assert_almost_equal(h0, h2)
            assert_almost_equal(t0, t1)
            assert_almost_equal(t0, t2)


def NufftBase_tests():
    from numpy.testing import assert_equal

    Kd = [128, 128]
    Nd = [64, 64]
    Jd = [6, 6]
    om = np.ones((1000, 2))
    p = NufftBase(om=om, Nd=Nd, Jd=Jd, Kd=Kd, kernel_type='kb:beatty',
                  phasing='complex')
    assert_equal(p.Nmid, [31.5, 31.5])
    p.phasing = 'real'
    assert_equal(p.Nmid, [32.0, 32.0])
    p.phasing = 'complex'
    assert_equal(p.Nmid, [31.5, 31.5])
    p.Nd = [96, 96]
    assert_equal(p.Nmid, [47.5, 47.5])
    p.Nd = [96, ]  # auto broadcast 1D
    assert_equal(p.Nd, [96, 96])
    p.Nd = 96  # auto broadcast scalar
    assert_equal(p.Nd, [96, 96])
    p.Nd = 196  # auto-update Kd
    assert_equal(p.Kd, p.Nd * (np.asarray(Kd) / np.asarray(Nd)))


def kernel_tests():
    from matplotlib import pyplot as plt
    from PyIRT.nufft.nufft_utils import _nufft_coef

    ktype_tests = [{'kernel_type': 'linear',
                    'kw_args': {'dd': 2,
                                'Jd': np.ones((2,))}
                    },

                   {'kernel_type': 'minmax:unif',
                    'kw_args': {}
                    },

                   {'kernel_type': 'minmax:tuned',
                    'kw_args': {'Jd': 6,
                                'Kd': 128,
                                'Nd': 64
                                }
                    },

                   {'kernel_type': 'kb:minmax',
                    'kw_args': {'dd': 1,
                                'Jd': 6}
                    },

                   {'kernel_type': 'kb:beatty',
                    'kw_args': {'dd': 1,
                                'Jd': 2,
                                'Kd': 96,
                                'Nd': [64, ],
                                }
                    },
                   {'kernel_type': 'kb:beatty',
                    'kw_args': {'dd': 1,
                                'Jd': 3,
                                'Kd': 96,
                                'Nd': [64, ],
                                }
                    },
                   {'kernel_type': 'kb:beatty',
                    'kw_args': {'dd': 1,
                                'Jd': 4,
                                'Kd': 96,
                                'Nd': [64, ],
                                }
                    },
                   {'kernel_type': 'kb:beatty',
                    'kw_args': {'dd': 1,
                                'Jd': 6,
                                'Kd': 96,
                                'Nd': [64, ],
                                }
                    },
                   {'kernel_type': 'kb:beatty',
                    'kw_args': {'dd': 1,
                                'Jd': 6,
                                'Kd': 128,
                                'Nd': [64, ],
                                }
                    },
                   {'kernel_type': 'kb:user',
                    'kw_args': {'Jd': [6, 6],
                                'kb_m': [0, 0],
                                'kb_alf': [2.34 * 6, 2.34 * 6],
                                }
                    },
                   {'kernel_type': 'kb:beatty',
                    'kw_args': {'dd': 2,
                                'Jd': [6, 6],
                                'Kd': [128, 128],
                                'Nd': [64, 64],
                                }
                    },
                   ]

    for kt in ktype_tests:
        k = NufftKernel(kt['kernel_type'],
                        **kt['kw_args'])
        if k.kernel is not None:
            plt.figure()
            J = kt['kw_args']['Jd']
            if isinstance(J, (list, tuple, set, np.ndarray)):
                J = J[0]
            plt.plot(k.kernel[0](np.linspace(-J / 2., J / 2., 1000), J))
            plt.title(k.kernel_type + ', ' + ', '.join(
                ['{}:{}'.format(key, v) for key, v in list(
                    kt['kw_args'].items())]))

    if False:  # error test
        from PyIRT.nufft.nufft_utils import nufft1_error
        K = 84  # 128
        N = 64
        J = 4
        kt = {'kernel_type': 'kb:beatty',
                             'kw_args': {'dd': 1,
                                         'Jd': J,
                                         'Kd': K,
                                         'Nd': [N, ],
                                         }
              }
        kt2 = {'kernel_type': 'kb:minmax',
               'kw_args': {'dd': 1,
                           'Jd': J}
               }

        k = NufftKernel(kt['kernel_type'],
                        **kt['kw_args'])
        k2 = NufftKernel(kt2['kernel_type'],
                         **kt2['kw_args'])

        gam = 2 * np.pi / K
        om = gam * np.linspace(0, 1, 101)
        nufft_errs = nufft1_error(om, N, J, K, kernel=k.kernel[0])[0]

        print(
            "mean=%0.3g, median=%0.3g, max=%0.3g" %
            (np.mean(nufft_errs),
             np.median(nufft_errs),
             np.max(nufft_errs)))
        nufft_errs2 = nufft1_error(om, N, J, K, kernel=k2.kernel[0])[0]
        print(
            "mean=%0.3g, median=%0.3g, max=%0.3g" %
            (np.mean(nufft_errs2),
             np.median(nufft_errs2),
             np.max(nufft_errs2)))

    #om = _nufft_samples('epi',96)[:,None]
    om = 2 * np.pi * np.arange(-96 / 2., 96 / 2.) / float(96)  # [M,]
    om = om[:, None]  # [M,1]

    [c, arg] = _nufft_coef(om[:, 0], 6, 96, k.kernel[0])  # [J,M]

    # DIRIC runs -K/2:K/2, not normalized to -1, 1 like the others
    kt = {'kernel_type': 'diric',
                         'kw_args': {'dd': 3,
                                     'Jd': [64, 64, 64],
                                     'Nd': [64, 64, 64],
                                     'Kd': [64, 64, 64],
                                     }
          }
    k = NufftKernel(kt['kernel_type'],
                    **kt['kw_args'])
    plt.figure()
#    plt.plot(k.kernel[0](np.linspace(-64,64,1000),2))
    # plt.plot(k.kernel[0](np.linspace(-1,1,1000),2))
    plt.plot(k.kernel[0](np.linspace(-
                                     kt['kw_args']['Jd'][0] /
                                     2, kt['kw_args']['Jd'][0] /
                                     2 -
                                     1, 1000), 2))
    plt.title('diric')

    plt.figure()
    for kt in ktype_tests:
        k = NufftKernel(kt['kernel_type'],
                        **kt['kw_args'])
        if k.kernel is not None:
            plt.plot(k.kernel[0](np.linspace(-1, 1, 1000), 2))


def load_matlab_newfft(filename):
#    data = scipy.io.loadmat('/media/Data1/src_repositories/my_git/pyrecon/PyIRT/nufft/mat_files/newwfft_test_3D_kb_beatty_sparse_complex.mat')
    data = scipy.io.loadmat(pjoin(data_dir, 'newwfft_test_2D_kb_beatty_table1_real.mat'))

    dd = data['data']['dd'][0, 0][0, 0]

    om = data['data']['om'][0, 0].astype(np.float64)
    om.flags  # F_CONTIGUOUS = True, WRITEABLE = True

    a = data['data']['alpha'][0, 0]
    b = data['data']['beta'][0, 0]
    ka = data['data']['kb_alf'][0, 0]
    km = data['data']['kb_m'][0, 0]
    alpha = []
    beta = []
    kb_alf = []
    kb_m = []
    for d in range(a.shape[1]):
        alpha.append(np.squeeze(a[0, d]))
        beta.append(float(b[0, d]))
    for d in range(ka.shape[1]):
        kb_alf.append(float(ka[0, d]))
        kb_m.append(float(km[0, d]))

    ktype = data['data']['ktype'][0, 0][0]
    mode = data['data']['mode'][0, 0][0]
    phasing = data['data']['phasing'][0, 0][0]
    if data['data']['phase_before'][0, 0].size > 0:
        phase_before = np.squeeze(data['data']['phase_before'][0, 0])
        phase_after = np.squeeze(data['data']['phase_after'][0, 0])

    # need astype(int) to fix uints
    n_shift = data['data']['n_shift'][0, 0].astype(int)
    dd = int(data['data']['dd'][0, 0][0])
    Nd = data['data']['Nd'][0, 0][0].astype(int)
    Jd = data['data']['Jd'][0, 0][0].astype(int)
    Kd = data['data']['Kd'][0, 0][0].astype(int)
    if 'table' in mode:
        Ld = data['data']['oversample'][0, 0][0].astype(int)
    else:
        Ld = None
    oversample = Ld
    if mode in ['table0', 'table1']:
        Ld = data['data']['oversample'][0, 0][0]
        h_data = data['data']['h'][0, 0]
        h = []
        for d in range(dd):
            h.append(np.squeeze(h_data[0, d]))

    if mode in ['sparse', ]:
        p = data['data']['p'][0, 0]
    M = om.shape[0]
    Nmid = data['data']['Nmid'][0, 0][0]
    sn = np.squeeze(data['data']['sn'][0, 0])
    is_kaiser_scale = int(data['data']['is_kaiser_scale'][0, 0][0])

    if dd > 1:
        plt.figure()
        plt.subplot(121)
        plt.imshow(montager(np.abs(sn)))
        plt.subplot(122)
        plt.imshow(montager(np.angle(sn)))

    if mode in ['table0', 'table1']:
        plt.figure()
        plt.plot(np.real(h[0]))
        plt.plot(np.imag(h[0]))
        plt.legend(('real', 'complex'))

    kernel_kwargs = {}
    if np.any(kb_alf):
        kernel_kwargs['kb_alf'] = kb_alf
        kernel_kwargs['kb_m'] = kb_m
    if alpha != []:
        kernel_kwargs['alpha'] = alpha
        kernel_kwargs['beta'] = beta
    # now call python version using inputs from matlab
    st = NufftBase(om=om,
                     Nd=Nd,
                     Jd=Jd,
                     Kd=Kd,
                     Ld=Ld,
                     n_shift=n_shift,
                     kernel_type=ktype,
                     phasing=phasing,
                     mode=mode,
                     kernel_kwargs=kernel_kwargs,
                     )
    # st.init_sparsemat()
    if mode == 'sparse':
        spmat = np.asarray(st.p.todense())

    f = st.plot_kernels()

    if (dd > 1) and (st.phase_before is not None):
        plt.figure()
        plt.subplot(121)
        plt.imshow(montager(np.squeeze(np.angle(st.phase_before))))
        plt.subplot(122)
        plt.plot(montager(np.angle(st.phase_after)))


    print(("sn max diff: %0.4g" % max_percent_diff(st.sn, sn)))
    print(("om max diff: %0.4g" % max_percent_diff(st.om, om)))
    print(("Nd max diff: %0.4g" % max_percent_diff(st.Nd, Nd)))
    print(("Kd max diff: %0.4g" % max_percent_diff(st.Kd, Kd)))
    print(("Jd max diff: %0.4g" % max_percent_diff(st.Jd, Jd)))
    print(("Jd max diff: %0.4g" % max_percent_diff(st.Nmid, Nmid)))
    if 'minmax:' in ktype:
        print((
            "alpha max diff: %0.4g" %
            max_percent_diff(
                st.kernel.alpha[0],
                alpha[0])))
        print((
            "alpha max diff: %0.4g" %
            max_percent_diff(
                st.kernel.beta[0],
                beta[0])))
    # assert_equal(st.phase_before.shape,st.Kd)
    if st.phase_before is not None:
        print((
            "phase_before max diff: %0.4g" %
            max_percent_diff(
                st.phase_before,
                phase_before)))
    if st.phase_after is not None:
        print((
            "phase_after max diff: %0.4g" %
            max_percent_diff(
                st.phase_after,
                phase_after)))
    if mode == 'sparse':
        print(("p max diff: %0.4g" % max_percent_diff(spmat, p)))
    elif 'table' in mode:
        for d in range(len(h)):
            print((
                "h[%d] max diff: %0.4g" %
                (d,
                 max_percent_diff(
                     h[d],
                     st.h[d]))))

    assert_almost_equal(st.sn, sn)
    assert_almost_equal(st.om, om)
    assert_equal(st.Nd, Nd)
    assert_equal(st.Kd, Kd)
    assert_equal(st.Jd, Jd)
    assert_equal(st.Nmid, Nmid)
    if 'minmax:' in ktype:
        assert_almost_equal(st.kernel.alpha[0], alpha[0])
        assert_almost_equal(st.kernel.beta[0], beta[0])
    if st.phase_before is not None:
        assert_almost_equal(st.phase_before, phase_before)
    if st.phase_after is not None:
        assert_almost_equal(st.phase_after, phase_after)
    if mode == 'sparse':
        assert_almost_equal(spmat, p)
    elif 'table' in mode:
        for d in range(len(h)):
            assert_almost_equal(h[d], st.h[d])

    x0 = data['x0'].copy()
    xe = data['xe'].copy()
    Xe = data['Xe'].copy()
    xs = data['xs'].copy()
    Xs = data['Xs'].copy()

    t_start_for = time.time()
    Xs_py = nufft_forward(st, x0)
    tau_for = time.time() - t_start_for
    print(("max diff: %0.4g" % max_percent_diff(Xs, Xs_py)))

    t_start_adj = time.time()
    xs_py = nufft_adj(st, Xe)
    tau_adj = time.time() - t_start_adj
    print(("max diff: %0.4g" % max_percent_diff(xs, xs_py)))

    assert_almost_equal(np.squeeze(Xs), np.squeeze(Xs_py), decimal=5)
    assert_almost_equal(np.squeeze(xs), np.squeeze(xs_py), decimal=5)

    # Run multiple simultaneous repetitions
    t_start_32for = time.time()
    Xs32_py = nufft_forward(
        np.tile(x0[..., None], ((1,) * x0.ndim + (32,))), st)
    tau_32for = time.time() - t_start_32for
    assert_equal(np.sum(np.abs(np.diff(Xs32_py, axis=-1))), 0)

    t_start_adj32 = time.time()
    xs32_py = nufft_adj(np.tile(Xe, (1, 32)), st)
    tau_32adj = time.time() - t_start_adj32
    # assure all repetitions gave an identical result
    assert_equal(np.sum(np.abs(np.diff(xs32_py, axis=-1))), 0)

    # repeat comparison, but at single precision
    st = NufftBase(om=om,
                   Nd=Nd,
                   Jd=Jd,
                   Kd=Kd,
                   Ld=Ld,
                   n_shift=n_shift,
                   kernel_type=ktype,
                   phasing=phasing,
                   mode=mode,
                   kernel_kwargs=kernel_kwargs,
                   precision='single'
                   )
    t_start_for = time.time()
    Xs_py = nufft_forward(st, x0)
    tau_for = time.time() - t_start_for
    print(("max diff: %0.4g" % max_percent_diff(Xs, Xs_py)))

    t_start_adj = time.time()
    xs_py = nufft_adj(st, x0)
    tau_adj = time.time() - t_start_adj
    print(("max diff: %0.4g" % max_percent_diff(xs, xs_py)))
    assert_almost_equal(np.squeeze(Xs), np.squeeze(Xs_py), decimal=2)
    assert_almost_equal(np.squeeze(xs), np.squeeze(xs_py), decimal=2)

    # plt.figure();
    # plt.subplot(121); plt.imshow(montager(np.abs(xs)))
    # plt.subplot(122); plt.imshow(montager(np.abs(xs_py)))


def load_matlab_old_nufft(filename):
    import scipy.io
    data = scipy.io.loadmat(pjoin(data_dir, 'OLD_Gtest_minmax_table0.mat'))

    om = data['data']['om'][0, 0]
    om.flags  # F_CONTIGUOUS = True, WRITEABLE = True
    dd = om.shape[1]

    try:
        a = data['data']['alpha'][0, 0]
        b = data['data']['beta'][0, 0]
        alpha = []
        beta = []
        for d in range(a.shape[1]):
            alpha.append(np.squeeze(a[0, d]))
            beta.append(float(b[0, d]))
        kb_alf = None
        kb_m = None
    except:
        ka = data['data']['kb_alf'][0, 0]
        km = data['data']['kb_m'][0, 0]
        kb_alf = []
        kb_m = []
        for d in range(ka.shape[1]):
            kb_alf.append(float(ka[0, d]))
            kb_m.append(float(km[0, d]))
        alpha = None
        beta = None

    try:
        h_data = data['data']['h'][0, 0]
        h = []
        for d in range(dd):
            h.append(np.squeeze(h_data[0, d]))
    except:
        h = None

    sn = data['data']['sn'][0, 0]

    plt.figure()
    plt.imshow(np.abs(sn))
    plt.figure()
    plt.imshow(np.angle(sn))

    plt.figure()
    plt.subplot(121)
    plt.plot(np.abs(h[0]))
    if np.any(np.iscomplex(h[0])):
        plt.subplot(122)
        plt.plot(np.angle(h[0]))



