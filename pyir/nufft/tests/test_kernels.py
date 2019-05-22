from __future__ import division, print_function, absolute_import
from itertools import product

import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_
import pytest

from pyir.nufft.nufft import NufftKernel
from pyir.nufft import config

if config.have_cupy:
    import cupy

    all_xp = [np, cupy]
else:
    all_xp = [np]

kernel_types = ["kb:beatty", "linear", "diric"]


@pytest.mark.parametrize("xp, dtype", product(all_xp, [np.float32, np.float64]))
def test_kernel(xp, dtype, show_figure=False):

    J = 4
    x = xp.linspace(-J / 2, J / 2, 1001, dtype=dtype)

    # can call with mixtures of integer, list or array input types
    d1 = NufftKernel("kb:beatty", Kd=[32, 32], Jd=[J, J], Nd=[24, 16])
    assert_equal(len(d1.kernel), 2)
    y = d1.kernel[0](x)
    assert_equal(x.dtype, y.dtype)

    d2 = NufftKernel("linear", ndim=2, Jd=J)
    assert_equal(len(d2.kernel), 2)
    y = d2.kernel[0](x)
    assert_equal(x.dtype, y.dtype)

    d3 = NufftKernel("diric", ndim=1, Kd=32, Jd=32, Nd=16)
    x = xp.linspace(-16, 16, 1001, dtype=dtype)
    assert_equal(len(d3.kernel), 1)
    y = d3.kernel[0](x)
    assert_equal(x.dtype, y.dtype)

    # invalid kernel raises ValueError
    assert_raises(ValueError, NufftKernel, "foo", ndim=2, Jd=4)

    if show_figure:
        d1.plot()
        d2.plot()
        d3.plot()


def test_kernel_range(show_figure=False):
    Jd = np.asarray([3, 4])
    kernel_types = ["kb:beatty"]
    for ktype in kernel_types:
        kernel = NufftKernel(
            ktype, ndim=2, Nd=[64, 64], Jd=Jd, Kd=[128, 128], Nmid=[32, 32]
        )

        for d in range(kernel.ndim):
            # non-zero within extent of J
            x = np.linspace(-Jd[d] / 2 + 0.001, Jd[d] / 2 - 0.001, 100)
            assert_(np.all(kernel.kernel[d](x) > 0))

            # 0 outside range of J
            assert_(kernel.kernel[d](np.asarray([Jd[d] / 2]))[0] == 0)
            assert_(kernel.kernel[d](np.asarray([-Jd[d] / 2]))[0] == 0)
