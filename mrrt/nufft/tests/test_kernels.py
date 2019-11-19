from itertools import product

import numpy as np
import pytest

from mrrt.utils import config
from mrrt.nufft._kernels import BeattyKernel

if config.have_cupy:
    import cupy

    all_xp = [np, cupy]
else:
    all_xp = [np]


@pytest.mark.parametrize("xp, dtype", product(all_xp, [np.float32, np.float64]))
def test_kernel(xp, dtype, show_figure=False):

    j = 4
    x = xp.linspace(-j / 2, j / 2, 1001, dtype=dtype)

    # can call with mixtures of integer, list or array input types
    d1 = BeattyKernel(shape=(j, j), grid_shape=(24, 16), os_grid_shape=(32, 32))
    assert len(d1.kernels) == 2
    y = d1.kernels[0](x)
    assert x.dtype == y.dtype

    # invalid kernel raises ValueError
    with pytest.raises(ValueError):
        BeattyKernel(shape=(j, j), grid_shape=(8, 8, 8), os_grid_shape=(32, 32))
    with pytest.raises(ValueError):
        BeattyKernel(
            shape=(j, j), grid_shape=(8, 8), os_grid_shape=(32, 32, 16)
        )

    if show_figure:
        d1.plot()

    # 1d case: can call with integers
    BeattyKernel(shape=j, grid_shape=24, os_grid_shape=32)


def test_kernel_range(show_figure=False):
    shape = np.asarray([3, 4])
    kernel = BeattyKernel(
        shape=shape, grid_shape=(64, 64), os_grid_shape=(128, 128)
    )
    for d in range(kernel.ndim):
        # non-zero within extent of J
        x = np.linspace(-shape[d] / 2 + 0.001, shape[d] / 2 - 0.001, 100)
        assert np.all(kernel.kernels[d](x) > 0)

        # 0 outside range of J
        assert kernel.kernels[d](np.asarray([shape[d] / 2]))[0] == 0
        assert kernel.kernels[d](np.asarray([shape[d] / 2 + 10]))[0] == 0
        assert kernel.kernels[d](np.asarray([-shape[d] / 2]))[0] == 0
        assert kernel.kernels[d](np.asarray([-shape[d] / 2 - 10]))[0] == 0
