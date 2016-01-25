
DEF PROVIDE_FLIP = 1
# MISSING:
#   double_interp1_table1_complex_per
#   double_interp1_table1_complex_per_adj
#   float_interp1_table1_complex_per
#   float_interp1_table1_complex_per_adj

cdef extern from "c/nufft_table.h":
    # Cython does not know the 'restrict' keyword

    cdef void double_interp1_table0_complex_per(
        const double *r_ck, # [K1,1] in
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1, # imaginary part of complex interpolator
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp1_table0_real_per(
        const double *r_ck, # [K,1] in
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in (real)
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp1_table1_real_per(
        const double *r_ck, # [K,1] in
        const double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in (real)
        const int J1,
        const int L1,
        const double *p_tm, # [M,1] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp1_table0_complex_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const int J1,
        const int L1,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)


    cdef void double_interp1_table0_real_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in
        const int J1,
        const int L1,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp1_table1_real_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const double *r_h1, # [J1*L1+1,1] in
        const int J1,
        const int L1,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp2_table0_complex_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp2_table0_real_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp2_table1_real_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp2_table1_complex_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp2_table0_complex_per(
        const double *r_ck, # [K1,K2] in
        const double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp2_table0_real_per(
        const double *r_ck, # [K1,K2] in
        const double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp2_table1_real_per(
        const double *r_ck, # [K1,K2] in
        const double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp2_table1_complex_per(
        const double *r_ck, # [K1,K2] in
        const double *i_ck,
        const int K1,
        const int K2,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const double *p_tm, # [M,2] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp3_table0_complex_per(
        const double *r_ck, # [K1,K2,K3] in
        const double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const double *r_h3, # [J3*L3+1,1] in
        const double *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,3] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)


    cdef void double_interp3_table1_complex_per(
        const double *r_ck, # [K1,K2,K3] in
        const double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const double *r_h3, # [J3*L3+1,1] in
        const double *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,3] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp3_table0_real_per(
        const double *r_ck, # [K1,K2,K3] in
        const double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const double *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,3] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)


    cdef void double_interp3_table1_real_per(
        const double *r_ck, # [K1,K2,K3] in
        const double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const double *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,3] in
        const int M,
        double *r_fm,       # [M,1] out
        double *i_fm)

    cdef void double_interp3_table0_complex_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const double *r_h3, # [J3*L3+1,1] in
        const double *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp3_table1_complex_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *i_h1,
        const double *r_h2, # [J2*L2+1,1] in
        const double *i_h2,
        const double *r_h3, # [J3*L3+1,1] in
        const double *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp3_table0_real_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const double *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void double_interp3_table1_real_per_adj(
        double *r_ck,   # [K1,K2] in
        double *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const double *r_h1, # [J1*L1+1,1] in
        const double *r_h2, # [J2*L2+1,1] in
        const double *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const double *p_tm, # [M,2] in
        const int M,
        const double *r_fm,     # [M,1] out
        const double *i_fm,
        const int N)

    cdef void float_interp1_table0_complex_per(
        const float *r_ck, # [K1,1] in
        const float *i_ck,
        const int K1,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1, # imaginary part of complex interpolator
        const int J1,
        const int L1,
        const float *p_tm, # [M,1] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp1_table0_real_per(
        const float *r_ck, # [K,1] in
        const float *i_ck,
        const int K1,
        const float *r_h1, # [J1*L1+1,1] in (real)
        const int J1,
        const int L1,
        const float *p_tm, # [M,1] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp1_table1_real_per(
        const float *r_ck, # [K,1] in
        const float *i_ck,
        const int K1,
        const float *r_h1, # [J1*L1+1,1] in (real)
        const int J1,
        const int L1,
        const float *p_tm, # [M,1] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp1_table0_complex_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const int J1,
        const int L1,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)


    cdef void float_interp1_table0_real_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const float *r_h1, # [J1*L1+1,1] in
        const int J1,
        const int L1,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp1_table1_real_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const float *r_h1, # [J1*L1+1,1] in
        const int J1,
        const int L1,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp2_table0_complex_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp2_table0_real_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp2_table1_real_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp2_table1_complex_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp2_table0_complex_per(
        const float *r_ck, # [K1,K2] in
        const float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp2_table0_real_per(
        const float *r_ck, # [K1,K2] in
        const float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp2_table1_real_per(
        const float *r_ck, # [K1,K2] in
        const float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp2_table1_complex_per(
        const float *r_ck, # [K1,K2] in
        const float *i_ck,
        const int K1,
        const int K2,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const int J1,
        const int J2,
        const int L1,
        const int L2,
        const float *p_tm, # [M,2] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp3_table0_complex_per(
        const float *r_ck, # [K1,K2,K3] in
        const float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const float *r_h3, # [J3*L3+1,1] in
        const float *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,3] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)


    cdef void float_interp3_table1_complex_per(
        const float *r_ck, # [K1,K2,K3] in
        const float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const float *r_h3, # [J3*L3+1,1] in
        const float *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,3] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp3_table0_real_per(
        const float *r_ck, # [K1,K2,K3] in
        const float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const float *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,3] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)


    cdef void float_interp3_table1_real_per(
        const float *r_ck, # [K1,K2,K3] in
        const float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const float *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,3] in
        const int M,
        float *r_fm,       # [M,1] out
        float *i_fm)

    cdef void float_interp3_table0_complex_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const float *r_h3, # [J3*L3+1,1] in
        const float *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp3_table1_complex_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *i_h1,
        const float *r_h2, # [J2*L2+1,1] in
        const float *i_h2,
        const float *r_h3, # [J3*L3+1,1] in
        const float *i_h3,
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp3_table0_real_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const float *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)

    cdef void float_interp3_table1_real_per_adj(
        float *r_ck,   # [K1,K2] in
        float *i_ck,
        const int K1,
        const int K2,
        const int K3,
        const float *r_h1, # [J1*L1+1,1] in
        const float *r_h2, # [J2*L2+1,1] in
        const float *r_h3, # [J3*L3+1,1] in
        const int J1,
        const int J2,
        const int J3,
        const int L1,
        const int L2,
        const int L3,
        const float *p_tm, # [M,2] in
        const int M,
        const float *r_fm,     # [M,1] out
        const float *i_fm,
        const int N)


ctypedef void float_interp1_per_cplx(const float *, const float *,
                                     const int,
                                     const float *, const float *,
                                     const int,
                                     const int,
                                     const float *, const int,
                                     float *, float *)
ctypedef void float_interp1_per_real(const float *, const float *,
                                     const int,
                                     const float *,
                                     const int,
                                     const int,
                                     const float *, const int,
                                     float *, float *)
ctypedef void double_interp1_per_cplx(const double *, const double *,
                                      const int,
                                      const double *, const double *,
                                      const int,
                                      const int,
                                      const double *, const int,
                                      double *, double *)
ctypedef void double_interp1_per_real(const double *, const double *,
                                      const int,
                                      const double *,
                                      const int,
                                      const int,
                                      const double *, const int,
                                      double *, double *)
ctypedef float_interp1_per_cplx* float_interp1_per_cplx_t
ctypedef float_interp1_per_real* float_interp1_per_real_t
ctypedef double_interp1_per_cplx* double_interp1_per_cplx_t
ctypedef double_interp1_per_real* double_interp1_per_real_t


ctypedef void float_interp2_per_cplx(const float *, const float *,
                                     const int, const int,
                                     const float *, const float *,
                                     const float *, const float *,
                                     const int, const int,
                                     const int, const int,
                                     const float *, const int,
                                     float *, float *)
ctypedef void float_interp2_per_real(const float *, const float *,
                                     const int, const int,
                                     const float *, const float *,
                                     const int, const int,
                                     const int, const int,
                                     const float *, const int,
                                     float *, float *)
ctypedef void double_interp2_per_cplx(const double *, const double *,
                                      const int, const int,
                                      const double *, const double *,
                                      const double *, const double *,
                                      const int, const int,
                                      const int, const int,
                                      const double *, const int,
                                      double *, double *)
ctypedef void double_interp2_per_real(const double *, const double *,
                                      const int, const int,
                                      const double *, const double *,
                                      const int, const int,
                                      const int, const int,
                                      const double *, const int,
                                      double *, double *)
ctypedef float_interp2_per_cplx* float_interp2_per_cplx_t
ctypedef float_interp2_per_real* float_interp2_per_real_t
ctypedef double_interp2_per_cplx* double_interp2_per_cplx_t
ctypedef double_interp2_per_real* double_interp2_per_real_t


ctypedef void float_interp3_per_cplx(const float *, const float *,
                                     const int, const int, const int,
                                     const float *, const float *,
                                     const float *, const float *,
                                     const float *, const float *,
                                     const int, const int, const int,
                                     const int, const int, const int,
                                     const float *, const int,
                                     float *, float *)
ctypedef void float_interp3_per_real(const float *, const float *,
                                     const int, const int, const int,
                                     const float *, const float *,
                                     const float *,
                                     const int, const int, const int,
                                     const int, const int, const int,
                                     const float *, const int,
                                     float *, float *)
ctypedef void double_interp3_per_cplx(const double *, const double *,
                                      const int, const int, const int,
                                      const double *, const double *,
                                      const double *, const double *,
                                      const double *, const double *,
                                      const int, const int, const int,
                                      const int, const int, const int,
                                      const double *, const int,
                                      double *, double *)
ctypedef void double_interp3_per_real(const double *, const double *,
                                      const int, const int, const int,
                                      const double *, const double *,
                                      const double *,
                                      const int, const int, const int,
                                      const int, const int, const int,
                                      const double *, const int,
                                      double *, double *)
ctypedef float_interp3_per_cplx* float_interp3_per_cplx_t
ctypedef float_interp3_per_real* float_interp3_per_real_t
ctypedef double_interp3_per_cplx* double_interp3_per_cplx_t
ctypedef double_interp3_per_real* double_interp3_per_real_t


ctypedef void float_interp1_per_adj_cplx(float *, float *,
                                     const int,
                                     const float *, const float *,
                                     const int,
                                     const int,
                                     const float *, const int,
                                     const float *, const float *,
                                     const int)
ctypedef void float_interp1_per_adj_real(float *, float *,
                                     const int,
                                     const float *,
                                     const int,
                                     const int,
                                     const float *, const int,
                                     const float *, const float *,
                                     const int)
ctypedef void double_interp1_per_adj_cplx(double *, double *,
                                      const int,
                                      const double *, const double *,
                                      const int,
                                      const int,
                                      const double *, const int,
                                      const double *, const double *,
                                      const int)
ctypedef void double_interp1_per_adj_real(double *, double *,
                                      const int,
                                      const double *,
                                      const int,
                                      const int,
                                      const double *, const int,
                                      const double *, const double *,
                                      const int)
ctypedef float_interp1_per_adj_cplx* float_interp1_per_adj_cplx_t
ctypedef float_interp1_per_adj_real* float_interp1_per_adj_real_t
ctypedef double_interp1_per_adj_cplx* double_interp1_per_adj_cplx_t
ctypedef double_interp1_per_adj_real* double_interp1_per_adj_real_t


ctypedef void float_interp2_per_adj_cplx(float *, float *,
                                     const int, const int,
                                     const float *, const float *,
                                     const float *, const float *,
                                     const int, const int,
                                     const int, const int,
                                     const float *, const int,
                                     const float *, const float *,
                                     const int)
ctypedef void float_interp2_per_adj_real(float *, float *,
                                     const int, const int,
                                     const float *, const float *,
                                     const int, const int,
                                     const int, const int,
                                     const float *, const int,
                                     const float *, const float *,
                                     const int)
ctypedef void double_interp2_per_adj_cplx(double *, double *,
                                      const int, const int,
                                      const double *, const double *,
                                      const double *, const double *,
                                      const int, const int,
                                      const int, const int,
                                      const double *, const int,
                                      const double *, const double *,
                                      const int)
ctypedef void double_interp2_per_adj_real(double *, double *,
                                      const int, const int,
                                      const double *, const double *,
                                      const int, const int,
                                      const int, const int,
                                      const double *, const int,
                                      const double *, const double *,
                                      const int)
ctypedef float_interp2_per_adj_cplx* float_interp2_per_adj_cplx_t
ctypedef float_interp2_per_adj_real* float_interp2_per_adj_real_t
ctypedef double_interp2_per_adj_cplx* double_interp2_per_adj_cplx_t
ctypedef double_interp2_per_adj_real* double_interp2_per_adj_real_t


ctypedef void float_interp3_per_adj_cplx(float *, float *,
                                     const int, const int, const int,
                                     const float *, const float *,
                                     const float *, const float *,
                                     const float *, const float *,
                                     const int, const int, const int,
                                     const int, const int, const int,
                                     const float *, const int,
                                     const float *, const float *,
                                     const int)
ctypedef void float_interp3_per_adj_real(float *, float *,
                                     const int, const int, const int,
                                     const float *, const float *,
                                     const float *,
                                     const int, const int, const int,
                                     const int, const int, const int,
                                     const float *, const int,
                                     const float *, const float *,
                                     const int)
ctypedef void double_interp3_per_adj_cplx(double *, double *,
                                      const int, const int, const int,
                                      const double *, const double *,
                                      const double *, const double *,
                                      const double *, const double *,
                                      const int, const int, const int,
                                      const int, const int, const int,
                                      const double *, const int,
                                      const double *, const double *,
                                      const int)
ctypedef void double_interp3_per_adj_real(double *, double *,
                                      const int, const int, const int,
                                      const double *, const double *,
                                      const double *,
                                      const int, const int, const int,
                                      const int, const int, const int,
                                      const double *, const int,
                                      const double *, const double *,
                                      const int)
ctypedef float_interp3_per_adj_cplx* float_interp3_per_adj_cplx_t
ctypedef float_interp3_per_adj_real* float_interp3_per_adj_real_t
ctypedef double_interp3_per_adj_cplx* double_interp3_per_adj_cplx_t
ctypedef double_interp3_per_adj_real* double_interp3_per_adj_real_t


