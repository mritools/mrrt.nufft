import warnings

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.signal import bspline
from pyir.utils import rowF, colF
from numpy.testing import assert_almost_equal

from pyir.utils import fftnc, ifftnc  # centered FFT


def sinc_new(x):
    indicator = np.abs(x) < 1e-6
    x = x + indicator
    out = np.sin(x) / x
    out = out*np.logical_not(indicator) + indicator
    return out


def calcKernelDiscretemod_fm(DFTMtx, fn, K, N, Ofactor, Order, H):
    # in Matlab:  fn & H are column vectors.  DFTMtx is a 2D matrix
    vector = np.arange(-K*Ofactor/2, K*Ofactor/2)
    abeta = fftnc(bspline(vector, 2*(Order)-1))

    # FT from -2*pi*Ofactor/2 to 2*pi*Ofactor/2
    vector = 2*np.pi*vector/K
    # 1/Ofactor*sinc_new(omega/(2 Ofactor)^(Order+1)
    BsplineFourier = sinc_new(vector/(2*Ofactor))**(2*Order)  # TODO: colF?

    index = (Ofactor - 1)*K//2
    bbeta = abeta.copy()
    bbeta[index:index+K] = abeta[index:index+K] - BsplineFourier[index:index+K]

    FN = np.squeeze(np.dot(DFTMtx, colF(fn)))
    FNabs2 = np.abs(FN)**2

    FN_den = FNabs2 * abeta
    FN_num = FNabs2 * bbeta

    Den = np.zeros(K, dtype=FN.dtype)
    Num = np.zeros(K, dtype=FN.dtype)
    for i in range(Ofactor):
        Den = Den + FN_den[i*K:(i+1)*K]
        Num = Num + FN_num[i*K:(i+1)*K]
    Kernel = Num/Den

    subset_idx = np.concatenate(
        (np.arange(K/2-N/2, dtype=np.intp),
         np.arange(K/2+N/2+1, Kernel.size, dtype=np.intp)))
    Kernel[subset_idx] = 0

    # new change
    H2 = np.zeros(K, dtype=FN.dtype)
    H2[K//2-N//2:-K//2+N//2] = H
    error = np.abs(np.mean(H2*Kernel))
    calcweight = H2/Den  # rowF

    calcweight[subset_idx] = 0
    CentralIndex = (Ofactor-1)/2*K
    if CentralIndex % 2 != 0:
        raise ValueError("expected an integer")
    else:
        CentralIndex = int(CentralIndex)
    calcweight = np.tile(calcweight, Ofactor)
    calcweight[CentralIndex:CentralIndex+K] = 0
    return (Kernel, calcweight, error)


def giveOptStepDiscrete_fm(DFTMtx, fn, K, N, Ofactor, Olderror, Oldfn, a,
                           Order, H, tol=None):
    steps = 0.5**(np.arange(31))
    steps = np.concatenate((steps, np.array([0, ])))
    # if tol is None:
    #     tol = 100 * np.finfo(fn.dtype).eps
    for step in steps:
        fn = step*a + (1 - step)*Oldfn
        Kernel, calcweight, error = calcKernelDiscretemod_fm(
            DFTMtx, fn, K, N, Ofactor, Order, H)
        # if (error - Olderror) < tol:
        if (Olderror - error) > 0:
            break
    return (Kernel, calcweight, error, fn, step)


def giveSymmetricDiscrete2_fm(J, K, N, Ofactor, Order, H, degree):
    if False:
        # direct matlab translation
        x = np.linspace(0, np.ceil(J/2), np.ceil(J/2)*Ofactor+1)
        assert_almost_equal(x[2] - x[1], 1/Ofactor)
        x = x[np.where(x <= J/2)]
        centreindex = len(x)
        x = np.concatenate((-x[1:][::-1], x))
    else:
        # differs from the Matlab translation above slightly when J=odd

        x = np.arange(-J * Ofactor / 2., J * Ofactor / 2. + 1) / Ofactor
        if x.size % 2 == 1:
            centreindex = int(np.ceil(x.size/2))
        else:
            centreindex = None


    # K samples: ranges from -2*pi*(Ofactor-1)/2 to 2*pi*(Ofactor-1)/2
    Nsamples = Ofactor*K
    k = np.arange(-Nsamples/2, Nsamples/2)
    DFTMtx = -1j*2*np.pi*np.dot(k[:, np.newaxis]/K, x[np.newaxis, :])
    DFTMtx = np.exp(DFTMtx, out=DFTMtx)

    vector = np.concatenate((np.arange(K*Ofactor/2),
                             np.arange(-K*Ofactor/2, 0)))
    abeta = K*Ofactor*ifftnc(bspline(vector, 3))
    Weight = scipy.sparse.diags(abeta)
    B = np.dot(np.conj(DFTMtx.T), np.asarray(Weight * np.asmatrix(DFTMtx)))
    if centreindex is not None:
        fn = bspline(x[centreindex-1:], degree)
        fn2 = fn[1:][::-1]
        fn = np.concatenate((fn2, fn))
    else:
        fn = bspline(x, degree)
    Kernel, current_weight, error = calcKernelDiscretemod_fm(
        DFTMtx, fn, K, N, Ofactor, Order, H)
    oldfn = fn
    olderror = error
    e = error

    # Start of iteration
    for itr in range(100):
        print("itr = {}, error={}".format(itr, error))
        weight = current_weight
        Weight = scipy.sparse.diags(weight)
        A = np.dot(np.conj(DFTMtx.T), np.asarray(Weight * np.asmatrix(DFTMtx)))
        evals, evecs = scipy.linalg.eig(A, B)
        newfn = evecs[:, np.argmin(np.abs(evals))]
        newfn = newfn/np.sum(newfn)
        if len(oldfn) == 0:
            fn = newfn
            oldfn = fn
            Kernel, current_weight, error = calcKernelDiscretemod_fm(
                DFTMtx, fn, K, N, Ofactor, Order, H)
            step = 1
        else:
            Kernel, current_weight, error, fn, step = giveOptStepDiscrete_fm(
                DFTMtx, fn, K, N, Ofactor, olderror, oldfn, newfn, Order, H)
            oldfn = fn
        e = error
        olderror = error
        oldfn = fn
        if step == 0:
            break
    return (fn, Kernel, e)


def givePrefilterNew_fm(fn, J, K, Ofactor, Order):
    # x values over the range of the interpolation coefficients
    if False:
        # direct matlab translation
        x = np.linspace(0, np.ceil(J/2), np.ceil(J/2)*Ofactor+1)
        assert_almost_equal(x[2] - x[1], 1/Ofactor)
        x = x[np.where(x <= J/2)]
        x = np.concatenate((-x[1:][::-1], x))
    else:
        # differs from the Matlab translation above slightly when J=odd
        x = np.arange(-J * Ofactor / 2., J * Ofactor / 2. + 1) / Ofactor

    Nsamples = Ofactor*K

    # K samples: ranges from -2*pi*(Ofactor-1)/2 to 2*pi*(Ofactor-1)/2
    k = np.arange(-Nsamples/2, Nsamples/2)
    DFTMtx = -1j*2*np.pi*np.dot(colF(k)/K, rowF(x))
    DFTMtx = np.exp(DFTMtx, out=DFTMtx)

    # DFT of the interpolation coefficients
    FN = np.dot(DFTMtx, fn)

    # Continuous Fourier transform of the bspline function
    BsplineFourier = np.sinc(np.pi*k/Nsamples)**(Order)/Ofactor

    k = np.arange(-49*Nsamples/2, 49*Nsamples/2)
    junk = np.sinc(np.pi*k/Nsamples)**(Order)/Ofactor
    BSPCorr = np.zeros_like(BsplineFourier)
    for i in range(49):
        sl = slice(i*Nsamples, i*Nsamples+Nsamples)
        BSPCorr = BSPCorr + np.abs(junk[sl])**2

    # Evaluating the denominator
    BSPCorr = BSPCorr*np.abs(FN)**2

    Den = np.zeros(K)
    for i in range(Ofactor):
        sl = slice(i*K, i*K + K)
        Den = Den + np.abs(BSPCorr[sl])

    midindex = (Ofactor - 1)//2
    sl = slice(midindex*K, midindex*K+K)
    prefilter = np.real(FN[sl])*BsplineFourier[sl]
    prefilter = prefilter/Den
    return prefilter


if False:
    J = 6
    N = 128
    K = 130
    Ofactor = 151
    Order = 2
    H = np.ones(N)
    degree = J - 1

    import numpy as np
    from pyir.nufft._iowa_MOLSkernel import PreNUFFT_fm
    from matplotlib import pyplot as plt
    J = 4
    N = 64
    K = 96
    Ofactor = 151
    Order = 2
    H = np.ones(N)
    degree = J - 1
    pre, fnn1, kernel, error1 = PreNUFFT_fm(J, N, Ofactor, K, Order=Order, H=H, degree=degree)
    plt.figure()
    plt.plot(np.arange(fnn1.size), fnn1.real,
             np.arange(fnn1.size), fnn1.imag)
#    kernel = NufftKernel('kb:beatty',
#                         ndim=1,
#                         Nd=[64, ],
#                         Jd=6,
#                         Kd=[96, ],
#                         Nmid=[32, ])


def PreNUFFT_fm(J, N, Ofactor, K, Order=2, H=None, degree=None,
                compute_prefilter=True):
    """Function to computer MOL interpolators and scale factors for MOLS

    Parameters
    ----------
    J : int
        interpolator size
    N : int
        size of image
    K : int
        oversampled size of image
    Ofactor : int
        oversampling factor of interpolator
    Order : TODO
    H : array
        energy distribution of the image
    degree : TODO

    Returns
    -------
    prefilter_fm : array
        MOLS prefilter (scale factors)
    fnn1 : array
        MOLS interpolator
    Kernel : TODO
    error1 : TODO

    References
    ----------
    Z. Yang, M. Jacob, "Mean square optimal NUFFT approximation for non-Cartesian MRI reconstruction", J Magn Reson, vol. 242, pp. 126-135, 2014
    M. Jacob,"Optimized least square non uniform fast Fourier transform (OLS-NUFFT)" , IEEE Transactions of Signal Processing, vol. 57, issue 6, pp. 2165-2177, 2009

    """
    if H is None:
        H = np.ones(N)
    if degree is None:
        degree = J - 1
    if Ofactor % 2 != 1:
        # TODO: does K*Ofactor need to be even ?
        Ofactor += 1
        warnings.warn("increasing Ofactor by 1:  (the current implementation "
                      "seems to require an odd Ofactor)")

    if (N % 2 != 0) or (K % 2 != 0):
        raise ValueError("odd N or K unsupported")

    if (K*Ofactor) % 2 != 0:
        warnings.warn("odd K*Ofactor untested")

    (fnn1, Kernel, error1) = giveSymmetricDiscrete2_fm(
        J, K, N, Ofactor, Order, H, degree)
    if compute_prefilter:
        prefilter_fm = givePrefilterNew_fm(fnn1, J, K, Ofactor, Order)

        # keep central size N block
        pre_nborder = (K - N) / 2
        if pre_nborder % 2 != 0:
            raise ValueError("uneven prefilter shape")
        else:
            pre_nborder = int(pre_nborder)
        prefilter_fm = prefilter_fm[slice(pre_nborder, -pre_nborder)]
    else:
        prefilter_fm = None

    return (prefilter_fm, fnn1, Kernel, error1)
