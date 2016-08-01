import numpy as np
from scipy.special import i0
from pyir.utils import rowF, colF

__all__ = ['giveLSInterpolator', ]

if False:
    N = 128
    K = 130
    Ofactor = 151
    J = 6


def giveLSInterpolator(N, K, Ofactor, J):
    """Compute LS-KB scalefactor(prefilter) and interpolator.

    Parameters
    ----------
    N : int
        size of image
    K : int
        oversampled size of image
    Ofactor : int
        oversampling factor of interpolator
    J : int
        kernel extent

    Returns
    -------
    scalefactor : array
        LS prefilter (scale factors)
    interpolator : array
        LS interpolation kernel

    References
    ----------

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
    interpolator = np.zeros((int(np.ceil(J*Ofactor)), ))
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

    if np.mod(m*2, 2) == 0:
        interpolator = interpolator[1:]

    q = np.linspace(-K/2, K/2-1, K)
    minindex = np.where(q == (-N/2))[0][0]
    maxindex = np.where(q == (N/2))[0][0]

    scalefactor = np.zeros(K)
    scalefactor[minindex:maxindex] = pre

    iindex = np.ceil(m*Ofactor).astype(np.intp) - 1
    scalefactor *= interpolator[iindex]
    interpolator /= interpolator[iindex]
    return (scalefactor, interpolator)
