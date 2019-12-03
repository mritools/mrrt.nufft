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

from ._extensions._nufft_table import (
    _interp1_table_forward,
    _interp2_table_forward,
    _interp3_table_forward,
    _interp1_table_adj,
    _interp2_table_adj,
    _interp3_table_adj,
)

__all__ = []


def interp1_table(ck, h1, j1, os_table, tm):
    k1 = ck.shape[0]
    n = ck.shape[1]
    m = tm.shape[0]
    if h1.ndim == 1:
        h1 = h1[:, np.newaxis]

    if (h1.shape[0] != j1 * os_table + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (j1, os_table, h1.shape[0]))
        raise ValueError("h1 size problem")

    if tm.ndim == 1:
        tm = tm[:, np.newaxis]
    if tm.ndim != 2 or tm.shape[-1] != 1:
        raise ValueError("tm must be a column vector")

    fm = _interp1_table_forward(ck, k1, h1, j1, os_table, tm, m, n)
    return fm


def interp1_table_adj(fm, h1, j1, os_table, tm, k1):
    m = fm.shape[0]
    n = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, np.newaxis]

    if (h1.shape[0] != j1 * os_table + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (j1, os_table, h1.shape[0]))
        raise ValueError("h size problem")

    if tm.ndim == 1:
        tm = tm[:, np.newaxis]

    if tm.ndim != 2 or tm.shape[-1] != 1:
        raise ValueError("tm must be a column vector")

    ck = _interp1_table_adj(fm, k1, h1, j1, os_table, tm, m, n)
    return ck


def interp2_table(ck, h1, h2, jd, os_table, tm):
    kd = ck.shape
    if ck.ndim == 2:
        n = 1
    elif ck.ndim == 3:
        n = ck.shape[2]
        kd = kd[:-1]
    if h1.ndim == 1:
        h1 = h1[:, np.newaxis]
    if h2.ndim == 1:
        h2 = h2[:, np.newaxis]

    m = tm.shape[0]

    if not ((len(jd) == 2) & (len(kd) == 2)):
        print("Error:  J, k and L must all be length 2")

    if (h1.shape[0] != jd[0] * os_table + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[0], os_table, h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != jd[1] * os_table + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[1], os_table, h2.shape[0]))
        raise ValueError("h2 size problem")

    if tm.ndim != 2 or tm.shape[-1] != 2:
        raise ValueError("tm must be size 2 on the second dimension")

    fm = _interp2_table_forward(ck, kd, h1, h2, jd, os_table, tm, m, n)
    return fm


def interp2_table_adj(fm, h1, h2, jd, os_table, tm, kd):
    m = fm.shape[0]
    n = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, np.newaxis]
    if h2.ndim == 1:
        h2 = h2[:, np.newaxis]

    if not ((len(jd) == 2) & (len(kd) == 2)):
        raise ValueError("Error:  J, k and L must all be length 2")

    if (h1.shape[0] != jd[0] * os_table + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[0], os_table, h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != jd[1] * os_table + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[1], os_table, h2.shape[0]))
        raise ValueError("h2 size problem")

    if tm.ndim != 2 or tm.shape[-1] != 2:
        raise ValueError("tm must be size 2 on the second dimension")

    ck = _interp2_table_adj(fm, kd, h1, h2, jd, os_table, tm, m, n)
    return ck


def interp3_table(ck, h1, h2, h3, jd, os_table, tm):
    kd = ck.shape
    if ck.ndim == 3:
        n = 1
    elif ck.ndim == 4:
        n = ck.shape[3]
        kd = kd[:-1]
    if h1.ndim == 1:
        h1 = h1[:, np.newaxis]
    if h2.ndim == 1:
        h2 = h2[:, np.newaxis]
    if h3.ndim == 1:
        h3 = h3[:, np.newaxis]

    m = tm.shape[0]

    if not ((len(jd) == 3) & (len(kd) == 3)):
        print("Error:  J, k and L must all be length 3")

    if (h1.shape[0] != jd[0] * os_table + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[0], os_table, h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != jd[1] * os_table + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[1], os_table, h2.shape[0]))
        raise ValueError("h2 size problem")

    if (h3.shape[0] != jd[2] * os_table + 1) | (h3.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[2], os_table, h3.shape[0]))
        raise ValueError("h3 size problem")

    if tm.ndim != 2 or tm.shape[1] != 3:
        raise ValueError("tm must be size 3 on the third dimension")

    fm = _interp3_table_forward(ck, kd, h1, h2, h3, jd, os_table, tm, m, n)
    return fm


def interp3_table_adj(fm, h1, h2, h3, jd, os_table, tm, kd):
    m = fm.shape[0]
    n = fm.shape[1]
    if h1.ndim == 1:
        h1 = h1[:, np.newaxis]
    if h2.ndim == 1:
        h2 = h2[:, np.newaxis]
    if h3.ndim == 1:
        h3 = h3[:, np.newaxis]

    if not ((len(jd) == 3) & (len(kd) == 3)):
        raise ValueError("Error:  J, k and L must all be length 3")

    if (h1.shape[0] != jd[0] * os_table + 1) | (h1.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[0], os_table, h1.shape[0]))
        raise ValueError("h1 size problem")

    if (h2.shape[0] != jd[1] * os_table + 1) | (h2.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[1], os_table, h2.shape[0]))
        raise ValueError("h2 size problem")

    if (h3.shape[0] != jd[2] * os_table + 1) | (h3.shape[1] != 1):
        print("J=%d, L=%d, tablelength=%d" % (jd[2], os_table, h3.shape[0]))
        raise ValueError("h3 size problem")

    if tm.ndim != 2 or tm.shape[-1] != 3:
        raise ValueError("tm must be size 3 on the third dimension")

    ck = _interp3_table_adj(fm, kd, h1, h2, h3, jd, os_table, tm, m, n)
    return ck
