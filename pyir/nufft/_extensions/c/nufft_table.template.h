/* Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/> */
/* See COPYING for license details. */

#include "templating.h"


#ifndef TYPE
#error TYPE must be defined here.
#else

//#include "common.h"

#if defined _MSC_VER
#define restrict __restrict
#elif defined __GNUC__
#define restrict __restrict__
#endif



/*
* interp1_table0_complex_per()
* 1D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp1_table0_complex_per)(
    const TYPE *r_ck, /* [K1,1] in */
    const TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1, /* imaginary part of complex interpolator */
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,1] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);

/*
* interp1_table1_complex_per()
* 1D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp1_table1_complex_per)(
    const TYPE *r_ck, /* [K1,1] in */
    const TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1, /* imaginary part of complex interpolator */
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,1] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);

/*
* interp1_table0_complex_per()
* 1D, 0th order, real, periodic
*/
void CAT(TYPE, _interp1_table0_real_per)(
    const TYPE *r_ck, /* [K,1] in */
    const TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in (real) */
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,1] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);

/*
* interp1_table1_real_per()
* 1D, 1st-order, real, periodic
*/
void CAT(TYPE, _interp1_table1_real_per)(
    const TYPE *r_ck, /* [K,1] in */
    const TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in (real) */
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,1] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp1_table0_complex_per_adj()
* 1D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp1_table0_complex_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp1_table1_complex_per_adj()
* 1D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp1_table1_complex_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp1_table0_real_per_adj()
* 1D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp1_table0_real_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp1_table1_real_per_adj()
* 1D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp1_table1_real_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const int J1,
    const int L1,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp2_table0_complex_per_adj()
* 2D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp2_table0_complex_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp2_table0_real_per_adj()
* 2D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp2_table0_real_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);

/*
* interp2_table1_real_per_adj()
* 2D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp2_table1_real_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp2_table1_complex_per_adj()
* 2D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp2_table1_complex_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp2_table0_complex_per()
* 2D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp2_table0_complex_per)(
    const TYPE *r_ck, /* [K1,K2] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp2_table0_real_per()
* 2D, 0th-order, real, periodic
*/
void CAT(TYPE, _interp2_table0_real_per)(
    const TYPE *r_ck, /* [K1,K2] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp2_table1_real_per()
* 2D, 1st-order, real, periodic
*/
void CAT(TYPE, _interp2_table1_real_per)(
    const TYPE *r_ck, /* [K1,K2] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp2_table1_complex_per()
* 2D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp2_table1_complex_per)(
    const TYPE *r_ck, /* [K1,K2] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const int J1,
    const int J2,
    const int L1,
    const int L2,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp3_table0_complex_per()
* 3D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp3_table0_complex_per)(
const TYPE *r_ck, /* [K1,K2,K3] in */
const TYPE *i_ck,
const int K1,
const int K2,
const int K3,
const TYPE *r_h1, /* [J1*L1+1,1] in */
const TYPE *i_h1,
const TYPE *r_h2, /* [J2*L2+1,1] in */
const TYPE *i_h2,
const TYPE *r_h3, /* [J3*L3+1,1] in */
const TYPE *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE *p_tm, /* [M,3] in */
const int M,
TYPE *r_fm,       /* [M,1] out */
TYPE *i_fm);

/*
* interp3_table1_complex_per()
* 3D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp3_table1_complex_per)(
    const TYPE *r_ck, /* [K1,K2,K3] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const TYPE *i_h3,
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,3] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);



/*
* interp3_table0_real_per()
* 3D, 0th-order, real, periodic
*/
void CAT(TYPE, _interp3_table0_real_per)(
    const TYPE *r_ck, /* [K1,K2,K3] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,3] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp3_table1_real_per()
* 3D, 1st-order, real, periodic
*/
void CAT(TYPE, _interp3_table1_real_per)(
    const TYPE *r_ck, /* [K1,K2,K3] in */
    const TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,3] in */
    const int M,
    TYPE *r_fm,       /* [M,1] out */
    TYPE *i_fm);


/*
* interp3_table0_complex_per_adj()
* 3D, 0th order, complex, periodic
*/
void CAT(TYPE, _interp3_table0_complex_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const TYPE *i_h3,
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp3_table1_complex_per_adj()
* 3D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp3_table1_complex_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *i_h1,
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *i_h2,
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const TYPE *i_h3,
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp3_table0_real_per_adj()
* 3D, 0th order, real, periodic
*/
void CAT(TYPE, _interp3_table0_real_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


/*
* interp3_table1_real_per_adj()
* 3D, 1st order, real, periodic
*/
void CAT(TYPE, _interp3_table1_real_per_adj)(
    TYPE *r_ck,   /* [K1,K2] in */
    TYPE *i_ck,
    const int K1,
    const int K2,
    const int K3,
    const TYPE *r_h1, /* [J1*L1+1,1] in */
    const TYPE *r_h2, /* [J2*L2+1,1] in */
    const TYPE *r_h3, /* [J3*L3+1,1] in */
    const int J1,
    const int J2,
    const int J3,
    const int L1,
    const int L2,
    const int L3,
    const TYPE *p_tm, /* [M,2] in */
    const int M,
    const TYPE *r_fm,     /* [M,1] out */
    const TYPE *i_fm,
    const int N);


// int CAT(TYPE, _downsampling_convolution)(const TYPE * const restrict input, const size_t N,
//                                          const TYPE * const restrict filter, const size_t F,
//                                          TYPE * const restrict output, const size_t step,
//                                          MODE mode);


#undef restrict
#endif /* TYPE */
