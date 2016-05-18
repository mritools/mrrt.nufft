from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import run_module_suite, assert_raises

from PyIRT.nufft.nufft import NufftKernel

def test_kernel(show_figure=False):
    
    # can call with mixtures of integer, list or array input types
    d2 = NufftKernel('kb:beatty', Kd=[32, 32], Jd=[4, 4], Nd=[24, 16])
    d3 = NufftKernel('kb:minmax', Kd=[32, 32], Jd=4, Nd=np.asarray([24, 16]))
    d4 = NufftKernel('minmax:kb',
                     Kd=[32, 32], Jd=4, Nd=[24, 16], Nmid=[12, 8])
    d1 = NufftKernel('linear', ndim=2, Jd=4)
    d5 = NufftKernel('diric', ndim=1, Kd=32, Jd=32, Nd=16)
    
    # invalid kernel raises ValueError
    assert_raises(ValueError, NufftKernel, 'foo', ndim=2, Jd=4)

    if show_figure:
        axes=d2.plot()
        d3.plot(axes)
        d4.plot(axes)    
        d1.plot(axes)
        d5.plot()


if __name__ == '__main__':
    run_module_suite()