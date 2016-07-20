import numpy as np
from scipy.special import i0
from pyir.utils import rowF, colF

if False:
    N = 128
    K = 130
    Ofactor = 151
    J = 6


def giveLSInterpolator(N, K, Ofactor, J):
    """
    % function to compute LS-KB scalefactor(prefilter) and interpolator
    % interpolator---LS-KB interpolators
    % scalefactor---scale factors
    % J---interpolator size
    % N---size of image
    % K---oversampled size of image
    % Ofactor--oversampling factor of interpolator
    % H---energy distribution of the image
    """
    m = J/2
    Samples = np.linspace(0, 1, Ofactor + 1)
    k = np.linspace(-N/2, N/2-1, N)
    alpha = K/N
    a1 = np.pi*(2 - 1/alpha)
    aaa = m * np.sqrt(a1*a1-(2*np.pi*(k+0.5)/K)**2)
    pre = 1/(K*i0(aaa))
    l_full = np.arange(-m, m+1, dtype=np.float64)
    D = np.diag(pre)
    q = []
    interpolator = np.zeros((int(J*Ofactor), ))  # , dtype=np.complex64)
    for j in range(len(Samples)):

        q = Samples[j] + l_full
        mask = np.where(np.abs(q) < m)
        l = l_full[mask]
        q = q[mask]

        l = rowF(l)
        k = colF(k)

        T = np.exp(-2*np.pi*1j*(k + 0.5)*l/K)
        tmp = np.dot(np.conj(T).T, D)
        tmp = np.dot(tmp, D)
        TDT = np.dot(tmp, T)
        TDTi = np.linalg.inv(TDT)

        Tt = T.T
        E = np.exp(-2*np.pi*1j*rowF(k)*Samples[j]/K)

        bb = np.dot(TDTi, Tt)
        bb = np.dot(bb, D)
        bb = np.dot(bb, E.T)
        interpolator[np.round((q+m)*Ofactor).astype(np.intp)] = bb.real.ravel()

    interpolator = interpolator.real
    if np.mod(m*2, 2) == 0:
        interpolator = interpolator[1:]

    q = np.linspace(-K/2, K/2-1, K)
    minindex = np.where(q == (-N/2))[0][0]
    maxindex = np.where(q == (N/2))[0][0]

    scalefactor = np.zeros(K)
    scalefactor[minindex:maxindex] = pre

    iindex = np.ceil(m*Ofactor).astype(np.intp)
    scalefactor *= interpolator[iindex]
    interpolator /= interpolator[iindex]
    return (scalefactor, interpolator)
