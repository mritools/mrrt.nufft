import numpy as np
from numpy.testing import assert_allclose, assert_equal, run_module_suite

from pyir.utils import fftn
from pyir.nufft._pruned_fft import (pruned_ifftn,
                                    pruned_fftn,
                                    pruned_fft_roundtrip,
                                    split_for_pruned_fft)


def test_pruned_fft():
    rstate = np.random.RandomState(1234)

    img_shape = (16, 16)
    rtol = 1e-7
    atol = 1e-5
    for dtype in [np.complex64, np.complex128]:
        img = rstate.standard_normal(img_shape).astype(dtype)
        img += 1j * rstate.standard_normal(img_shape).astype(dtype)

        F = fftn(img, s=(2*img.shape[0], 2*img.shape[1]))

        imgc = pruned_ifftn(F)
        assert_allclose(img, imgc, rtol=rtol, atol=atol)
        assert_allclose(F, pruned_fftn(img), rtol=rtol, atol=atol)

        img2 = pruned_ifftn(pruned_fftn(img))
        assert_allclose(img, img2, rtol=rtol, atol=atol)

        img3 = pruned_fft_roundtrip(img)
        assert_allclose(img, img3, rtol=rtol, atol=atol)

        Qp = split_for_pruned_fft(F, aslist=False)
        assert_equal(Qp.shape, img_shape + (2**len(img_shape), ))

        Qp = split_for_pruned_fft(F, aslist=True)
        assert_equal(len(Qp), 2**len(img_shape))
        assert_equal(np.all([Q.shape == img_shape for Q in Qp]), True)


if __name__ == '__main__':
    run_module_suite()
