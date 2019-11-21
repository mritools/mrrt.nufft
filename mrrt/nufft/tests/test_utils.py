import numpy as np
from numpy.testing import assert_equal

from mrrt.nufft._utils import _nufft_offset
from mrrt.utils import config

if config.have_cupy:
    import cupy

    all_xp = [np, cupy]
else:
    all_xp = [np]


def test_nufft_offset():
    om = 0
    k = 128  # not used for the omega=0 case tested here
    assert_equal(_nufft_offset(om, 4, k), -2)
    assert_equal(_nufft_offset(om, 5, k), -3)
    assert_equal(_nufft_offset(om, 5.5, k), -3)
    assert_equal(_nufft_offset(om, 6, k), -3)
    assert_equal(_nufft_offset(np.asarray([om]), 6.01, k), np.asarray([-4]))
