from itertools import product

import numpy as np
from numpy.testing import assert_equal
import pytest

from mrrt.nufft import dtft, dtft_adj
from mrrt.nufft._nufft import NufftBase, _nufft_table_make1
from mrrt.utils import config, max_percent_diff
from mrrt.nufft.tests.test_dtft import _uniform_freqs

if config.have_cupy:
    import cupy

    all_xp = [np, cupy]
else:
    all_xp = [np]


def _perturbed_gridpoints(Nd, rel_std=0.5, seed=1234, xp=np):
    """Generate a uniform cartesian frequency grid of shape Nd and then
    perturb each point's position by an amount between [0, rel_std) of the
    grid spacing along each axis.

    (for testing NUFFT routines vs. the DTFT)
    """
    rstate = np.random.RandomState(seed)
    if np.isscalar(Nd):
        Nd = (Nd,)
    Nd = np.asarray(Nd)
    if np.any(Nd < 2):
        raise ValueError("must be at least size 2 on all dimensions")
    omega = _uniform_freqs(Nd)  # [npoints, ndim]
    df = 2 * np.pi / Nd
    npoints = omega.shape[0]
    for d in range(len(Nd)):
        # don't randomize edge points so values will still fall within
        # a 2*pi range
        omega[:, d] += df[d] * rel_std * rstate.rand(npoints)
        # rescale to keep values within a 2*pi range
        omega[:, d] += np.min(omega[:, d])
        omega[:, d] *= (2 * np.pi) / np.max(omega[:, d])

    return xp.asarray(omega)


def _randomized_gridpoints(Nd, rel_std=0.5, seed=1234, xp=np):
    """Generate a uniform cartesian frequency grid of shape Nd and then
    perturb each point's position by an amount between [0, rel_std) of the
    grid spacing along each axis.

    (for testing NUFFT routines vs. the DTFT)
    """
    rstate = xp.random.RandomState(seed)
    ndim = len(Nd)
    omegas = rstate.uniform(size=(tuple(Nd) + (ndim,)), low=0, high=2 * np.pi)
    return omegas.reshape((-1, ndim), order="F")


@pytest.mark.parametrize(
    "xp, mode, phasing",
    product(all_xp, ["sparse", "table"], ["real", "complex"]),
)
def test_nufft_adj(xp, mode, phasing, verbose=False):
    """ test nufft_adj() """
    N1 = 4
    N2 = 8
    n_shift = [2.7, 3.1]  # random shifts to stress it
    o1 = 2 * np.pi * xp.array([0.0, 0.1, 0.3, 0.4, 0.7, 0.9])
    o2 = o1[::-1].copy()
    omega = xp.stack((o1, o2), axis=-1)
    st = NufftBase(
        omega=omega,
        Nd=(N1, N2),
        Jd=[8, 8],
        Kd=2 * np.array([N1, N2]),
        Ld=1024,
        n_shift=n_shift,
        phasing=phasing,
        mode=mode,
        on_gpu=xp != np,
    )

    data = xp.arange(1, o1.size + 1).ravel() ** 2  # test spectrum
    xd = dtft_adj(data, omega, shape=(N1, N2), n_shift=n_shift, xp=xp)

    data_3reps = xp.stack((data,) * 3, axis=-1)
    xn = st.adj(data_3reps)
    if verbose:
        print(
            "nufft vs dtft max%%diff = %g" % max_percent_diff(xd, xn[:, :, -1])
        )
    xp.testing.assert_array_almost_equal(
        xp.squeeze(xd), xp.squeeze(xn[:, :, -1]), decimal=4
    )
    return


@pytest.mark.parametrize(
    "xp, mode, precision, phasing, order",
    product(
        all_xp,
        ["sparse", "table"],
        ["single", "double"],
        ["real", "complex"],
        ["C", "F"],
    ),
)
def test_nufft_1d(xp, mode, precision, phasing, order, verbose=False):
    Nd = 64
    Kd = 128
    Jd = 6
    Ld = 1024
    n_shift = Nd // 2
    omega = _perturbed_gridpoints(Nd, xp=xp)
    rstate = xp.random.RandomState(1234)

    rtol = 1e-3
    atol = 1e-5
    A = NufftBase(
        omega=omega,
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        mode=mode,
        Ld=Ld,
        precision=precision,
        phasing=phasing,
        order=order,
        on_gpu=xp != np,
    )

    x = rstate.randn(Nd) + 1j * rstate.randn(Nd)

    # forward
    y = A.fft(x)
    y_true = dtft(x, omega=omega, shape=Nd, n_shift=n_shift)
    xp.testing.assert_allclose(y, y_true, rtol=rtol, atol=atol)

    # multi-repetition forward
    if order == "C":
        x_reps = xp.stack((x,) * 2, axis=0)
        sl1 = (0, Ellipsis)
    else:
        x_reps = xp.stack((x,) * 2, axis=-1)
        sl1 = (Ellipsis, 0)
    y_reps = A.fft(x_reps)
    xp.testing.assert_allclose(y_reps[sl1], y_true, rtol=rtol, atol=atol)

    # adjoint
    x_adj = A.adj(y)
    x_adj_true = dtft_adj(y, omega=omega, shape=Nd, n_shift=n_shift)
    xp.testing.assert_allclose(x_adj, x_adj_true, rtol=rtol, atol=atol)

    # multi-repetition adjoint
    if order == "C":
        y_reps = xp.stack((y,) * 2, axis=0)
    else:
        y_reps = xp.stack((y,) * 2, axis=-1)
    x_adj = A.adj(y_reps)
    xp.testing.assert_allclose(x_adj[sl1], x_adj_true, rtol=rtol, atol=atol)

    if verbose:
        print(mode, precision, phasing)
        print(
            f"\t{max_percent_diff(y, y_true)} "
            f"{max_percent_diff(x_adj, x_adj_true)}"
        )


@pytest.mark.parametrize(
    "xp, mode, precision, phasing, Kd, Jd, order",
    product(
        all_xp,
        ["sparse", "table"],
        ["single", "double"],
        ["real", "complex"],
        [(32, 32), (33, 31)],  # test both even and odd
        [6, 7],  # test both even and odd
        ["F", "C"],
    ),
)
def test_nufft_2d(xp, mode, precision, phasing, Kd, Jd, order, verbose=False):
    Nd = (16, 16)
    Ld = 1024
    n_shift = np.asarray(Nd) / 2
    omega = _perturbed_gridpoints(Nd, xp=xp)
    rstate = xp.random.RandomState(1234)

    rtol = 1e-3
    atol = 1e-5
    A = NufftBase(
        omega=omega,
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        mode=mode,
        Ld=Ld,
        precision=precision,
        phasing=phasing,
        on_gpu=xp != np,
        order=order,
    )
    x = rstate.standard_normal(Nd)
    x = x + 1j * rstate.standard_normal(Nd)

    # forward
    y = A.fft(x)
    y_true = dtft(x, omega=omega, shape=Nd, n_shift=n_shift)
    xp.testing.assert_allclose(y, y_true, rtol=rtol, atol=atol)

    # multi-repetition forward
    if order == "F":
        x_reps = xp.stack((x,) * 2, axis=-1)
        sl1 = (Ellipsis, 0)
    else:
        x_reps = xp.stack((x,) * 2, axis=0)
        sl1 = (0, Ellipsis)
    y_reps = A.fft(x_reps)
    xp.testing.assert_allclose(y_reps[sl1], y_true, rtol=rtol, atol=atol)

    # adjoint
    x_adj = A.adj(y)
    x_adj_true = dtft_adj(y, omega=omega, shape=Nd, n_shift=n_shift)
    xp.testing.assert_allclose(x_adj, x_adj_true, rtol=rtol, atol=atol)

    # multi-repetition adjoint
    if order == "F":
        y_reps = xp.stack((y,) * 2, axis=-1)
        sl1 = (Ellipsis, 0)
    else:
        y_reps = xp.stack((y,) * 2, axis=0)
        sl1 = (0, Ellipsis)
    x_adj = A.adj(y_reps)
    xp.testing.assert_allclose(x_adj[sl1], x_adj_true, rtol=rtol, atol=atol)

    if verbose:
        print(mode, precision, phasing)
        print(
            f"\t{max_percent_diff(y, y_true)} "
            f"{max_percent_diff(x_adj, x_adj_true)}"
        )


@pytest.mark.parametrize(
    "xp, mode, precision, phasing, order",
    product(
        all_xp,
        ["sparse", "table"],
        ["single", "double"],
        ["real", "complex"],
        ["F", "C"],
    ),
)
def test_nufft_3d(xp, mode, precision, phasing, order, verbose=False):
    ndim = 3
    Nd = [8] * ndim
    Kd = [16] * ndim
    Jd = [6] * ndim  # use odd kernel for variety (even in 1D, 2D tests)
    Ld = 1024
    n_shift = np.asarray(Nd) / 2
    omega = _perturbed_gridpoints(Nd, xp=xp)

    rtol = 1e-2
    atol = 1e-4
    rstate = xp.random.RandomState(1234)
    A = NufftBase(
        omega=omega,
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        mode=mode,
        Ld=Ld,
        precision=precision,
        phasing=phasing,
        order=order,
        on_gpu=xp != np,
    )
    x = rstate.standard_normal(Nd)
    x = x + 1j * rstate.standard_normal(Nd)

    # forward
    y = A.fft(x)
    y_true = dtft(x, omega=omega, shape=Nd, n_shift=n_shift)
    xp.testing.assert_allclose(y, y_true, rtol=rtol, atol=atol)

    # TODO: fix case with multiple additional axes at start or end
    # (multi-repetition forward with 2 additional axes at start or end)
    if order == "C":
        # x_reps = xp.stack((x,) * 4, axis=0).reshape((2, 2) + x.shape)
        x_reps = xp.stack((x,) * 4, axis=0).reshape((4,) + x.shape)
        sl1 = (0, Ellipsis)
    else:
        # x_reps = xp.stack((x,) * 4, axis=-1).reshape(x.shape + (2, 2))
        x_reps = xp.stack((x,) * 4, axis=-1).reshape(x.shape + (4,))
        sl1 = (Ellipsis, 0)
    y_reps = A.fft(x_reps)
    xp.testing.assert_allclose(y_reps[sl1], y_true, rtol=rtol, atol=atol)

    # adjoint
    x_adj = A.adj(y)
    x_adj_true = dtft_adj(y, omega=omega, shape=Nd, n_shift=n_shift)
    xp.testing.assert_allclose(x_adj, x_adj_true, rtol=rtol, atol=atol)

    # TODO: fix case with multiple additional axes at start or end
    # multi-repetition adjoint with 2 additional axes at start or end
    if order == "C":
        # y_reps = xp.stack((y,) * 4, axis=0).reshape((2, 2) + y.shape)
        y_reps = xp.stack((y,) * 4, axis=0).reshape((4,) + y.shape)
    else:
        # y_reps = xp.stack((y,) * 4, axis=-1).reshape(y.shape + (2, 2))
        y_reps = xp.stack((y,) * 4, axis=-1).reshape(y.shape + (4,))
    x_adj = A.adj(y_reps)
    xp.testing.assert_allclose(x_adj[sl1], x_adj_true, rtol=rtol, atol=atol)

    if verbose:
        print(mode, precision, phasing)
        print(
            f"\t{max_percent_diff(y, y_true)} "
            f"{max_percent_diff(x_adj, x_adj_true)}"
        )


@pytest.mark.parametrize("precision, xp", product(["single", "double"], [np]))
def test_nufft_dtypes(precision, xp):
    Nd = 64
    Kd = 128
    Jd = 6
    Ld = 4096
    n_shift = Nd // 2
    omega = _perturbed_gridpoints(Nd, xp=xp)

    mode = "table"
    A = NufftBase(
        omega=omega,
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        mode=mode,
        Ld=Ld,
        precision=precision,
        on_gpu=xp != np,
    )

    if precision == "single":
        assert_equal(A._cplx_dtype, np.complex64)
        assert_equal(A._real_dtype, np.float32)
    else:
        assert_equal(A._cplx_dtype, np.complex128)
        assert_equal(A._real_dtype, np.float64)

    # set based on precision of omega rather than the precision argument
    A2 = NufftBase(
        omega=omega.astype(A._real_dtype),
        Nd=Nd,
        Jd=Jd,
        Kd=Kd,
        n_shift=n_shift,
        mode=mode,
        Ld=Ld,
        precision="auto",
        on_gpu=xp != np,
    )

    if precision == "single":
        assert_equal(A2._cplx_dtype, np.complex64)
        assert_equal(A2._real_dtype, np.float32)
    else:
        assert_equal(A2._cplx_dtype, np.complex128)
        assert_equal(A2._real_dtype, np.float64)

    x = xp.random.randn(Nd) + 1j * xp.random.randn(Nd)

    # output matches operator dtype regardless of input dtype
    y = A.fft(x.astype(np.complex64))
    assert_equal(y.dtype, A._cplx_dtype)
    y = A.fft(x.astype(np.complex128))
    assert_equal(y.dtype, A._cplx_dtype)

    # real input also gives complex output
    y = A.fft(x.real.astype(np.float32))
    assert_equal(y.dtype, A._cplx_dtype)
    y = A.fft(x.real.astype(np.float64))
    assert_equal(y.dtype, A._cplx_dtype)


@pytest.mark.parametrize(
    "xp, n, phasing",
    product(all_xp, [64, 65], ["real", "complex"]),  # test both odd and even
)
def test_nufft_table_make1(xp, n, phasing):
    decimal = 6
    h0, t0 = _nufft_table_make1(
        how="slow", N=n, J=6, K=2 * n, L=2048, phasing=phasing, order="F"
    )
    h1, t1 = _nufft_table_make1(
        how="fast", N=n, J=6, K=2 * n, L=2048, phasing=phasing, order="F"
    )
    h2, t2 = _nufft_table_make1(
        how="ratio", N=n, J=6, K=2 * n, L=2048, phasing=phasing, order="F"
    )
    xp.testing.assert_array_almost_equal(h0, h1, decimal=decimal)
    xp.testing.assert_array_almost_equal(h0, h2, decimal=decimal)
    xp.testing.assert_array_almost_equal(t0, t1, decimal=decimal)
    xp.testing.assert_array_almost_equal(t0, t2, decimal=decimal)
