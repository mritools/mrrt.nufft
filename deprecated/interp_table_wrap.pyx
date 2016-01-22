# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython

cdef extern: # from "interp_table.cpp":
    void interp1_table0_complex_per(
        const double *r_ck, # [K1,1] in */
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in */
        const double *i_h1, # imaginary part of complex interpolator */
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in */
        const int M,
        double *r_fm,       # [M,1] out */
        double *i_fm)
    void interp1_table1_complex_per(
        const double *r_ck, # [K1,1] in */
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in */
        const double *i_h1, # imaginary part of complex interpolator */
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in */
        const int M,
        double *r_fm,       # [M,1] out */
        double *i_fm)
    void interp1_table0_real_per(
        const double *r_ck, # [K1,1] in */
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in */
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in */
        const int M,
        double *r_fm,       # [M,1] out */
        double *i_fm)
    void interp1_table1_real_per(
        const double *r_ck, # [K1,1] in */
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in */
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in */
        const int M,
        double *r_fm,       # [M,1] out */
        double *i_fm)

def py_interp1_per(double [::1] r_ck,
                   double [::1] i_ck,
                   int K1,
                   double [::1] r_h1,
                   double [::1] i_h1,
                   int J1,
                   int L1,
                   double [::1] p_tm,
                   int M,
                   double [::1] r_fm,
                   double [::1] i_fm,
                   int iscomplex=0,
                   int order=0):
    if iscomplex:
        if order == 0:
            interp1_table0_complex_per(&r_ck[0], &i_ck[0], K1, &r_h1[0],
                                       &i_h1[0], J1, L1, &p_tm[0], M,
                                       &r_fm[0], &i_fm[0])
        elif order == 1:
            interp1_table1_complex_per(&r_ck[0], &i_ck[0], K1, &r_h1[0],
                                       &i_h1[0], J1, L1, &p_tm[0], M,
                                       &r_fm[0], &i_fm[0])
    else:
        if order == 0:
            interp1_table0_real_per(&r_ck[0], &i_ck[0], K1, &r_h1[0],
                                       J1, L1, &p_tm[0], M,
                                       &r_fm[0], &i_fm[0])
        elif order == 1:
            interp1_table1_real_per(&r_ck[0], &i_ck[0], K1, &r_h1[0],
                                       J1, L1, &p_tm[0], M,
                                       &r_fm[0], &i_fm[0])
