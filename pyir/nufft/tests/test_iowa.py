import os
from os.path import join as pjoin

import numpy as np
from numpy.testing import (run_module_suite, assert_allclose)

import pyir.nufft
from pyir.nufft._iowa import giveLSInterpolator

pkg_dir = os.path.dirname(pyir.nufft.__file__)
data_dir = pjoin(pkg_dir, 'tests', 'data')


def test_giveLSInterpolator():
    # compare result to output of Matlab implementation
    data = np.load(pjoin(data_dir, 'giveLSInterpolator_testdata.npz'))
    J, K, N, Ofactor = data['J'], data['K'], data['N'], data['Ofactor']

    prefilter, interpolator = giveLSInterpolator(N, K, Ofactor, J)

    assert_allclose(interpolator, data['interpolator_expected'])
    assert_allclose(prefilter, data['prefilter_expected'])


if __name__ == "__main__":
    run_module_suite()
