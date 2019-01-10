"""
The underlying C code for these routines are adapted from C code originally
developed by Jeff Fessler and his students at the University of Michigan.

OpenMP support and the Cython wrappers were added by Gregory R. Lee
(Cincinnati Childrens Hospital Medical Center).

Note:  For simplicity the adjoint NUFFT is only parallelized across multiple
coils and/or repetitions.  This was done for simplicity to avoid any potential
thread conflicts.
"""
import numpy as np

from ._extensions._nufft_table import (_interp1_table_per,
                                       _interp2_table_per,
                                       _interp3_table_per,
                                       _interp1_table_adj,
                                       _interp2_table_adj,
                                       _interp3_table_adj)


def interp1_table(ck, h1, J1, L1, tm, order=0):
    K1 = ck.shape[0]
    N = ck.shape[1]
    M = tm.shape[0]
    if h1.ndim == 1:
        h1 = h1[:, None]

    if (h1.shape[0] != J1 * L1 + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (J1, L1, h1.shape[0]))
        raise ValueError("h1 size problem")

    if tm.ndim == 1:
        tm = tm[:, None]
    if not tm.shape == (M, 1):  # (M != tm.shape[0]) | (1 != tm.shape[1]):
        raise ValueError("tm must be Mx1 col vector")

    J1 = int(J1)
    L1 = int(L1)
    fm = _interp1_table_per(ck, K1, h1, J1, L1, tm, M, N, order)
    return fm


def interp1_table_adj(fm, h1, J1, L1, tm, K1, order=None):

    M = fm.shape[0]
    N = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, None]

    if (h1.shape[0] != J1 * L1 + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (J1, L1, h1.shape[0]))
        raise ValueError("h size problem")

    if tm.ndim == 1:
        tm = tm[:, None]

    if not tm.shape == (M, 1):  # (M != tm.shape[0]) | (1 != tm.shape[1]):
        raise ValueError("tm must be Mx1 col vector")

    J1 = int(J1)
    L1 = int(L1)
    ck = _interp1_table_adj(fm, K1, h1, J1, L1, tm, M, N, order)
    return ck


def interp2_table(ck, h1, h2, Jd, Ld, tm, order=0):
    Kd = ck.shape
    if(ck.ndim == 2):
        N = 1
    elif(ck.ndim == 3):
        N = ck.shape[2]
        Kd = Kd[:-1]
    if h1.ndim == 1:
        h1 = h1[:, None]
    if h2.ndim == 1:
        h2 = h2[:, None]

    M = tm.shape[0]
    Jd = np.asanyarray(Jd).astype(np.int32)
    Ld = np.asanyarray(Ld).astype(np.int32)

    if not ((len(Jd) == 2) & (len(Ld) == 2) & (len(Kd) == 2)):
        print("Error:  J, K and L must all be length 2")

    if (h1.shape[0] != Jd[0] * Ld[0] + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[0], Ld[0], h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != Jd[1] * Ld[1] + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[1], Ld[1], h2.shape[0]))
        raise ValueError("h2 size problem")

    if not tm.shape == (M, 2):  # (M != tm.shape[0]) | (2 != tm.shape[1]):
        raise ValueError("tm must be Mx2")

    fm = _interp2_table_per(ck, Kd, h1, h2, Jd, Ld, tm, M, N, order)
    return fm


def interp2_table_adj(fm, h1, h2, Jd, Ld, tm, Kd, order=None):

    M = fm.shape[0]
    N = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, None]
    if h2.ndim == 1:
        h2 = h2[:, None]

    Jd = np.asanyarray(Jd).astype(np.int32)
    Ld = np.asanyarray(Ld).astype(np.int32)

    if not ((len(Jd) == 2) & (len(Ld) == 2) & (len(Kd) == 2)):
        raise ValueError("Error:  J, K and L must all be length 2")

    if (h1.shape[0] != Jd[0] * Ld[0] + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[0], Ld[0], h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != Jd[1] * Ld[1] + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[1], Ld[1], h2.shape[0]))
        raise ValueError("h2 size problem")

    if not tm.shape == (M, 2):  # (M != tm.shape[0]) | (2 != tm.shape[1]):
        raise ValueError("tm must be Mx2")

    ck = _interp2_table_adj(fm, Kd, h1, h2, Jd, Ld, tm, M, N, order)
    return ck


def interp3_table(ck, h1, h2, h3, Jd, Ld, tm, order=0):
    Kd = ck.shape
    if(ck.ndim == 3):
        N = 1
    elif(ck.ndim == 4):
        N = ck.shape[3]
        Kd = Kd[:-1]
    if h1.ndim == 1:
        h1 = h1[:, None]
    if h2.ndim == 1:
        h2 = h2[:, None]
    if h3.ndim == 1:
        h3 = h3[:, None]

    M = tm.shape[0]
    Jd = np.asanyarray(Jd).astype(np.int32)
    Ld = np.asanyarray(Ld).astype(np.int32)

    if not ((len(Jd) == 3) & (len(Ld) == 3) & (len(Kd) == 3)):
        print("Error:  J, K and L must all be length 3")

    if (h1.shape[0] != Jd[0] * Ld[0] + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[0], Ld[0], h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != Jd[1] * Ld[1] + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[1], Ld[1], h2.shape[0]))
        raise ValueError("h2 size problem")

    if (h3.shape[0] != Jd[2] * Ld[2] + 1) | (h3.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[2], Ld[2], h3.shape[0]))
        raise ValueError("h3 size problem")

    if not tm.shape == (M, 3):  # (M != tm.shape[0]) | (2 != tm.shape[1]):
        raise ValueError("tm must be Mx3")

    fm = _interp3_table_per(ck, Kd, h1, h2, h3, Jd, Ld, tm, M, N, order)
    return fm


def interp3_table_adj(fm, h1, h2, h3, Jd, Ld, tm, Kd, order=None):

    M = fm.shape[0]
    N = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, None]
    if h2.ndim == 1:
        h2 = h2[:, None]
    if h3.ndim == 1:
        h3 = h3[:, None]

    Jd = np.asanyarray(Jd).astype(np.int32)
    Ld = np.asanyarray(Ld).astype(np.int32)

    if not ((len(Jd) == 3) & (len(Ld) == 3) & (len(Kd) == 3)):
        raise ValueError("Error:  J, K and L must all be length 3")

    if (h1.shape[0] != Jd[0] * Ld[0] + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[0], Ld[0], h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != Jd[1] * Ld[1] + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[1], Ld[1], h2.shape[0]))
        raise ValueError("h2 size problem")

    if (h3.shape[0] != Jd[2] * Ld[2] + 1) | (h3.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (Jd[2], Ld[2], h3.shape[0]))
        raise ValueError("h3 size problem")

    if not tm.shape == (M, 3):  # (M != tm.shape[0]) | (2 != tm.shape[1]):
        raise ValueError("tm must be Mx2")

    ck = _interp3_table_adj(fm, Kd, h1, h2, h3, Jd, Ld, tm, M, N, order)
    return ck
