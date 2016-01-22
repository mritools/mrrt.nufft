# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange


cimport cython.operator
from cython.operator cimport(address,
                             dereference,
                             preincrement,
                             predecrement,
                             postincrement,
                             postdecrement)


from libc.math cimport sqrt, exp, fmax, floor

# apparently libc.math only has the double precision variants.
# manually import the single-precision ones here:
cdef extern from "math.h" nogil:
    float sqrtf(float x)
    float expf(float x)
    float fmaxf(float x, float y)

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

ctypedef cython.doublecomplex dcomplex

# cplx_floating = cython.fused_type(cython.floatcomplex, cython.doublecomplex)
floating = cython.fused_type(
    cython.float,
    cython.double,
    cython.floatcomplex,
    cython.doublecomplex)
cplx_type = cython.fused_type(cython.floatcomplex, cython.doublecomplex)
real_type = cython.fused_type(cython.float, cython.double)

cdef extern from "math.h" nogil:
    float floorf(float x)


@cython.cdivision(True)  # need to avoid python-based checks for divide by zero
cdef cnp.npy_intp mymod(cnp.npy_intp k, cnp.npy_intp K) nogil:
    return < cnp.npy_intp > (k - K * floor(k / (< double > K)))

# cdef cnp.npy_intp iround(double x) nogil:
#    return <cnp.npy_intp> floor(x + 0.5)

cdef cnp.npy_intp iround(floating x) nogil:
    return < cnp.npy_intp > floor(< double > x + 0.5)


# define mymod(k,K) ((k) - (K) * floor((k) / (double) (K)))
# define iround(x) floor(x + 0.5)

# cdef extern from "macros.h":
#    cdef cnp.npy_intp mymod(cnp.npy_intp k, cnp.npy_intp K) nogil
#    cdef double iround(double x) nogil


def interp3_table(ck, h1, h2, h3, J, L, tm, M, order=0):

    ck = np.asfortranarray(ck)
    h1 = np.asfortranarray(h1)
    h2 = np.asfortranarray(h2)
    h3 = np.asfortranarray(h3)
    tm = np.asfortranarray(tm)

    M = tm.shape[0]

    if tm.shape[1] != 3:
        raise ValueError("Wrong shape for tm")

    if(ck.ndim == 3):
        N = 1
        ck = ck[:, :, :, np.newaxis]
    elif(ck.ndim == 4):
        N = ck.shape[3]

    J = nd.asarray(J, dtype=np.intp)
    L = nd.asarray(L, dtype=np.intp)

    fm = np.asfortranarray(np.zeros((M, N), dtype=kernel_dtype, order='F'))

    if len(J) != 3:
        raise ValueError("J must be length 3")
    J1, J2, J3 = J
    if len(L) != 3:
        raise ValueError("L must be length 3")
    L1, L2, L3 = L
    # fm = np.asfortranarray(fm)

    for n in range(N):
        if order == 0:
            interp3_table0_per(
                ck[:, :, :, n], h1, h2, h3, J1, J2, J3, L1, L2, L3, tm, M, fm)
            # interp3_table0_per(ck[:,:,:,n], h1, h2, h3, *J, *L, tm, M, fm)
        else:
            raise ValueError("Only 0th order implemented")

    return fm


"""
 interp3_table0_complex_per()
 3D, 0th order, complex, periodic
"""


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void interp3_table0_per(const floating[::1, :, :] ck,  # [K1,K2,K3]
                             const floating[::1] h1,  # [J1*L1+1,1]
                             const floating[::1] h2,
                             const floating[::1] h3,
                             const cnp.npy_intp J1,
                             const cnp.npy_intp J2,
                             const cnp.npy_intp J3,
                             const cnp.npy_intp L1,
                             const cnp.npy_intp L2,
                             const cnp.npy_intp L3,
                             # [M,3],  values in range [-K/2 K/2]
                             const double[::1, :] ptm,
                             const int M,
                             floating[::1, :, :] fm):  # [M,1]
    cdef:
        cnp.npy_intp K1 = ck.shape[0]
        cnp.npy_intp K2 = ck.shape[1]
        cnp.npy_intp K3 = ck.shape[2]
        cnp.npy_intp mm
        cnp.npy_intp ncenter1 = < cnp.npy_intp > floor(J1 * L1 / 2)
        cnp.npy_intp ncenter2 = < cnp.npy_intp > floor(J2 * L2 / 2)
        cnp.npy_intp ncenter3 = < cnp.npy_intp > floor(J3 * L3 / 2)
        floating * p_h1 = &h1[0]
        floating * p_h2 = &h2[0]
        floating * p_h3 = &h3[0]
        floating * p_ck = &ck[0, 0, 0]
        # dcomplex *p_ptm = &p_m[0,0]
        floating * p_fm = &fm[0, 0, 0]
        double t1, t2, t3
        cnp.npy_intp jj1, jj2, jj3
        cnp.npy_intp koff1, koff2
        cnp.npy_intp k1, k2, k3
        double p1, p2, p3
        cnp.npy_intp n1, n2, n3
        floating sum1, sum2, sum3
        floating coef1, coef2, coef3
        cnp.npy_intp k3mod, k2mod, k23mod, k1mod, kk

    # trick: shift table pointer to center
    p_h1 += ncenter1
    p_h2 += ncenter2
    p_h3 += ncenter3

    # interp
    with nogil, parallel():
        for mm in prange(M):
            #        t3 = p_ptm[2*M+mm]
            #        t2 = p_ptm[M+mm]
            #        t1 = p_ptm[mm]
            t3 = ptm[mm, 2]
            t2 = ptm[mm, 1]
            t1 = ptm[mm, 0]
            sum3 = 0
            koff1 = 1 + <cnp.npy_intp > floor(t1 - J1 / 2.)
            koff2 = 1 + <cnp.npy_intp > floor(t2 - J2 / 2.)
            k3 = 1 + <cnp.npy_intp > floor(t3 - J3 / 2.)

            for jj3 in range(J3):  # (jj3=0; jj3 < J3; jj3++, k3++)
                k3 = k3 + 1
                p3 = (t3 - k3) * L3
                n3 = iround(p3)  # <cnp.npy_intp> floor(p3+0.5)
                coef3 = p_h3[n3]
                k3mod = mymod(k3, K3)
                sum2 = 0
                k2 = koff2

                for jj2 in range(J2):  # (jj2=0; jj2 < J2; jj2++, k2++)
                    k2 = k2 + 1
                    p2 = (t2 - k2) * L2
                    n2 = iround(p2)  # <cnp.npy_intp> floor(p2+0.5)
                    coef2 = p_h2[n2]
                    k2mod = mymod(k2, K2)
                    k23mod = (k3mod * K2 + k2mod) * K1
                    sum1 = 0
                    k1 = koff1

                    for jj1 in range(J1):  # (jj1=0; jj1 < J1; jj1++, k1++)
                        k1 = k1 + 1
                        p1 = (t1 - k1) * L1
                        n1 = iround(p1)  # <cnp.npy_intp> floor(p1+0.5)
                        coef1 = p_h1[n1]
                        k1mod = mymod(k1, K1)
                        kk = k23mod + k1mod  # 3D array index */
                        sum1 = sum1 + coef1 * p_ck[kk]

                    sum2 = sum2 + coef2 * sum1

                sum3 = sum3 + coef3 * sum2

            p_fm[mm] = sum3


# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
# cdef void interp3_table0_complex_per(const dcomplex [::1, :, :] ck,
# const dcomplex [::1] h1,       #[J1*L1+1,1]
#                                     const dcomplex [::1] h2,
#                                     const dcomplex [::1] h3,
#                                     const cnp.npy_intp J1,
#                                     const cnp.npy_intp J2,
#                                     const cnp.npy_intp J3,
#                                     const cnp.npy_intp L1,
#                                     const cnp.npy_intp L2,
#                                     const cnp.npy_intp L3,
# const double [::1, :] ptm,       #[M,3],  values in range [-K/2 K/2]
#                                     const int M,
# dcomplex [::1,:,:] fm):   #[M,1]
#    cdef:
#        cnp.npy_intp K1 = ck.shape[0]
#        cnp.npy_intp K2 = ck.shape[1]
#        cnp.npy_intp K3 = ck.shape[2]
#        cnp.npy_intp mm
#        cnp.npy_intp ncenter1 = <cnp.npy_intp> floor(J1 * L1/2)
#        cnp.npy_intp ncenter2 = <cnp.npy_intp> floor(J2 * L2/2)
#        cnp.npy_intp ncenter3 = <cnp.npy_intp> floor(J3 * L3/2)
#        dcomplex *p_h1 = &h1[0]
#        dcomplex *p_h2 = &h2[0]
#        dcomplex *p_h3 = &h3[0]
#        dcomplex *p_ck = &ck[0,0,0]
# dcomplex *p_ptm = &p_m[0,0]
#        dcomplex *p_fm = &fm[0,0,0]
#        double t1, t2 ,t3
#        cnp.npy_intp jj1, jj2, jj3
#        cnp.npy_intp koff1, koff2
#        cnp.npy_intp k1, k2, k3
#        double p1, p2, p3
#        cnp.npy_intp n1, n2, n3
#        dcomplex sum1, sum2, sum3
#        dcomplex coef1, coef2, coef3
#        cnp.npy_intp k3mod, k2mod, k23mod, k1mod, kk
#
# trick: shift table pointer to center
#    p_h1 += ncenter1
#    p_h2 += ncenter2
#    p_h3 += ncenter3
#
# interp
#    with nogil, parallel():
#        for mm in prange(M):
# t3 = p_ptm[2*M+mm]
# t2 = p_ptm[M+mm]
# t1 = p_ptm[mm]
#            t3 = ptm[mm, 2]
#            t2 = ptm[mm, 1]
#            t1 = ptm[mm, 0]
#            sum3 = 0
#            koff1 = 1 + <cnp.npy_intp> floor(t1 - J1 / 2.)
#            koff2 = 1 + <cnp.npy_intp> floor(t2 - J2 / 2.)
#            k3 = 1 + <cnp.npy_intp> floor(t3 - J3 / 2.)
#
# for jj3 in range(J3):  #(jj3=0; jj3 < J3; jj3++, k3++)
#                k3 = k3 + 1
#                p3 = (t3 - k3) * L3
# n3 = iround(p3) #<cnp.npy_intp> floor(p3+0.5)
#                coef3 = p_h3[n3]
#                k3mod = mymod(k3, K3)
#                sum2 = 0
#                k2 = koff2
#
# for jj2 in range(J2): #(jj2=0; jj2 < J2; jj2++, k2++)
#                    k2 = k2 + 1
#                    p2 = (t2 - k2) * L2
# n2 = iround(p2) #<cnp.npy_intp> floor(p2+0.5)
#                    coef2 = p_h2[n2]
#                    k2mod = mymod(k2, K2)
#                    k23mod = (k3mod * K2 + k2mod) * K1
#                    sum1 = 0
#                    k1 = koff1
#
# for jj1 in range(J1): #(jj1=0; jj1 < J1; jj1++, k1++)
#                        k1 = k1 + 1
#                        p1 = (t1 - k1) * L1
# n1 = iround(p1) #<cnp.npy_intp> floor(p1+0.5)
#                        coef1 = p_h1[n1]
#                        k1mod = mymod(k1, K1)
# kk = k23mod + k1mod # 3D array index */
#                        sum1 = sum1 + coef1 * p_ck[kk]
#
#                    sum2 = sum2 + coef2 * sum1
#
#                sum3 = sum3 + coef3 * sum2
#
#            p_fm[mm] = sum3
#
