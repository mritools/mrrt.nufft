import numpy as np
from numpy.testing import assert_almost_equal, run_module_suite

from pyir.nufft._toeplitz import compute_psf
show_figures = False


def test_psf(show_figures=show_figures):
    kspace = np.arange(-128, 128, dtype=np.float32).reshape(-1, 1)
    fov = 1
    Nd = 256
    weights = None
    nufft_kwargs = dict(mode='table0', kernel_type='kb:beatty',
                        pixel_basis='dirac')
    psf = compute_psf(kspace, Nd, fov, weights, **nufft_kwargs)
    # should be a delta function
    assert_almost_equal(psf[Nd//2].real, Nd, decimal=2)
    assert_almost_equal(psf[Nd//2-1].real, 0, decimal=2)
    assert_almost_equal(psf[0].real, 0, decimal=2)
    assert_almost_equal(psf[-1].real, 0, decimal=2)
    assert_almost_equal(psf[Nd//2+1].real, 0, decimal=2)
    if show_figures:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(np.abs(psf), 'k.-')

    # double matrix size & FOV, but keep the same number of k-space samples
    psf2 = compute_psf(kspace*2, Nd*2, fov*2, weights, **nufft_kwargs)
    # should now get 3 aliased peaks
    assert_almost_equal(psf2[Nd//2].real, Nd, decimal=2)
    assert_almost_equal(psf2[Nd].real, Nd, decimal=2)
    assert_almost_equal(psf2[3*Nd//2].real, Nd, decimal=2)
    if show_figures:
        axes[1].plot(np.abs(psf2))


if __name__ == '__main__':
    run_module_suite()
