/*
The underlying C code for these routines are adapted from C code originally
developed by Jeff Fessler and his students at the University of Michigan.

OpenMP support and the Cython wrappers were added by Gregory R. Lee
(Cincinnati Childrens Hospital Medical Center).

Note:  For simplicity the adjoint NUFFT is only parallelized across multiple
coils and/or repetitions.  This was done for simplicity to avoid any potential
thread conflicts.

The C templating used here is based on the implementation by Kai Wohlfahrt as
developed for the BSD-licensed PyWavelets project.
*/
#include "templating.h"

#ifndef TYPE
#error TYPE must be defined here.
#else

#include <math.h>
#include <stdio.h>
//#include "def,interp,table.h"

#include <string.h> /* for memset */

#include "nufft_table.h"

#define mymod(k,K) ((k) - (K) * floor((k) / (double) (K)))
#define iround(x) floor(x + 0.5)

#if defined _MSC_VER
#define restrict __restrict
#elif defined __GNUC__
#define restrict __restrict__
#endif


/*
* interp1_table1_complex_forward()
* 1D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp1_table1_complex_forward)(
const TYPE * restrict r_ck, /* [K1,1] in */
const TYPE * restrict i_ck,
const int K1,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1, /* imaginary part of complex interpolator */
const int J1,
const int L1,
const TYPE * restrict p_tm, /* [M,1] in */
const int M,
TYPE * restrict r_fm,       /* [M,1] out */
TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor((double) (J1 * L1/2.));
    r_h1 += ncenter1;
    i_h1 += ncenter1;
    }

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1;
        const TYPE t1 = *p_tm++;
        register TYPE sum1r = 0;
        register TYPE sum1i = 0;
        int k1 = 1 + floor(t1 - J1 / 2.);

        for (jj1=0; jj1 < J1; jj1++, k1++) {
            const TYPE p1 = (t1 - k1) * L1;
            const int n1 = floor(p1);
            const TYPE alf1 = p1 - n1;
            register const TYPE * restrict ph1r = r_h1 + n1;
            register const TYPE * restrict ph1i = i_h1 + n1;
            register TYPE coef1r = (1 - alf1) * *ph1r + alf1 * *(ph1r+1);
            register TYPE coef1i = (1 - alf1) * *ph1i + alf1 * *(ph1i+1);
            const int k1mod = mymod(k1, K1);

            /* sum1 += coef1 * ck */
            sum1r += coef1r * r_ck[k1mod] - coef1i * i_ck[k1mod];
            sum1i += coef1r * i_ck[k1mod] + coef1i * r_ck[k1mod];
        }
        *r_fm++ = sum1r;
        *i_fm++ = sum1i;
    }
}


/*
* interp1_table1_real_forward()
* 1D, 1st-order, real, periodic
*/
void CAT(TYPE, _interp1_table1_real_forward)(
const TYPE * restrict r_ck, /* [K,1] in */
const TYPE * restrict i_ck,
const int K1,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in (real) */
const int J1,
const int L1,
const TYPE * restrict p_tm, /* [M,1] in */
const int M,
TYPE * restrict r_fm,       /* [M,1] out */
TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor((double) (J1 * L1/2.));
    r_h1 += ncenter1;
    }

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1;
        const TYPE t1 = *p_tm++;
        register TYPE sum1r = 0;
        register TYPE sum1i = 0;
        int k1 = 1 + floor(t1 - J1 / 2.);

        for (jj1=0; jj1 < J1; jj1++, k1++) {
            const TYPE p1 = (t1 - k1) * L1;
            const int n1 = floor(p1);
            const TYPE alf1 = p1 - n1;
            register const TYPE * restrict ph1 = r_h1 + n1;
            register TYPE coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
            const int wrap1 = floor(k1 / (TYPE) K1);
            const int k1mod = k1 - K1 * wrap1;

            /* sum1 += coef1 * ck */
            sum1r += coef1r * r_ck[k1mod];
            sum1i += coef1r * i_ck[k1mod];
        }
        *r_fm++ = sum1r;
        *i_fm++ = sum1i;
    }
}


void CAT(TYPE, _interp1_table1_complex_adj_inner)(
TYPE * restrict r_ck,       /* [K1,1] out */
TYPE * restrict i_ck,
const int K1,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1, /* imaginary part of complex interpolator */
const int J1,
const int L1,
const TYPE * restrict p_tm, /* [M,1] in */
const int M,
const TYPE * restrict r_fm, /* [M,1] in */
const TYPE * restrict i_fm)
{
    int mm;

    /* initialize output to zero */
    (void) memset((void *) r_ck, 0, K1*sizeof(*r_ck));
    (void) memset((void *) i_ck, 0, K1*sizeof(*i_ck));

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor((double) (J1 * L1/2.));
    r_h1 += ncenter1;
    i_h1 += ncenter1;
    }

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1;
        const TYPE t1 = *p_tm++;
        const TYPE fmr = *r_fm++;
        const TYPE fmi = *i_fm++;
        int k1 = 1 + floor(t1 - J1 / 2.);

        for (jj1=0; jj1 < J1; jj1++, k1++) {
            const TYPE p1 = (t1 - k1) * L1;
            const int n1 = floor(p1);
            const TYPE alf1 = p1 - n1;
            register const TYPE * restrict ph1r = r_h1 + n1;
            register const TYPE * restrict ph1i = i_h1 + n1;
            register TYPE coef1r = (1 - alf1) * *ph1r + alf1 * *(ph1r+1);
            register TYPE coef1i = (1 - alf1) * *ph1i + alf1 * *(ph1i+1);
            const int k1mod = mymod(k1, K1);

            /* instead of f = h c, we have c += h^* f */
            r_ck[k1mod] += coef1r * fmr + coef1i * fmi;
            i_ck[k1mod] += coef1r * fmi - coef1i * fmr;
        }
    }
}

/*
* interp1_table1_complex_adj()
* 1D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp1_table1_complex_adj)(
TYPE * restrict r_ck,   /* [K1,K2] in */
TYPE * restrict i_ck,
const int K1,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const int J1,
const int L1,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm,     /* [M,1] out */
const TYPE * restrict i_fm,
const int N)
{
    int nn;
    int K = K1;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
       CAT(TYPE, _interp1_table1_complex_adj_inner)(&r_ck[nn*K], &i_ck[nn*K],
                        K1,
                        r_h1, i_h1,
                        J1, L1,
                        p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


void CAT(TYPE, _interp1_table1_real_adj_inner)(
TYPE * restrict r_ck,       /* [K1,1] out */
TYPE * restrict i_ck,
const int K1,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in (real) */
const int J1,
const int L1,
const TYPE * restrict p_tm, /* [M,1] in */
const int M,
const TYPE * restrict r_fm, /* [M,1] in */
const TYPE * restrict i_fm)
{
    int mm;

    /* initialize output to zero */
    (void) memset((void *) r_ck, 0, K1*sizeof(*r_ck));
    (void) memset((void *) i_ck, 0, K1*sizeof(*i_ck));

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor((double) (J1 * L1/2.));
    r_h1 += ncenter1;
    }

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1;
        const TYPE t1 = *p_tm++;
        const TYPE fmr = *r_fm++;
        const TYPE fmi = *i_fm++;
        int k1 = 1 + floor(t1 - J1 / 2.);

        for (jj1=0; jj1 < J1; jj1++, k1++) {
            const TYPE p1 = (t1 - k1) * L1;
            const int n1 = floor(p1);
            const TYPE alf1 = p1 - n1;
            register const TYPE * restrict ph1 = r_h1 + n1;
            register TYPE coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
            const int wrap1 = floor(k1 / (TYPE) K1);
            const int k1mod = k1 - K1 * wrap1;

            /* instead of f = h c, we have c += h^* f */
            r_ck[k1mod] += coef1r * fmr;
            i_ck[k1mod] += coef1r * fmi;
        }
    }
}

/*
* interp1_table1_real_adj()
* 1D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp1_table1_real_adj)(
TYPE * restrict r_ck,   /* [K1,K2] in */
TYPE * restrict i_ck,
const int K1,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const int J1,
const int L1,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm,     /* [M,1] out */
const TYPE * restrict i_fm,
const int N)
{
    int nn;
    int K = K1;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
       CAT(TYPE, _interp1_table1_real_adj_inner)(&r_ck[nn*K], &i_ck[nn*K],
                        K1,
                        r_h1,
                        J1, L1,
                        p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


void CAT(TYPE, _interp2_table1_real_adj_inner)(
TYPE * restrict r_ck,       /* [K1,K2] out */
TYPE * restrict i_ck,
const int K1,
const int K2,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm, /* [M,1] in */
const TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor((double) (J1 * L1/2.));
    r_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor((double) (J2 * L2/2.));
    r_h2 += ncenter2;
    }

    /* initialize output to zero */
    (void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
    (void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1, jj2;
        const TYPE t2 = p_tm[M];
        const TYPE t1 = *p_tm++;
        const TYPE fmr = *r_fm++;
        const TYPE fmi = *i_fm++;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        int k2 = 1 + floor(t2 - J2 / 2.);

        for (jj2=0; jj2 < J2; jj2++, k2++) {
            const TYPE p2 = (t2 - k2) * L2;
            const int n2 = floor(p2);
            const TYPE alf2 = p2 - n2;
            register const TYPE * restrict ph2 = r_h2 + n2;
            TYPE coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
            const int wrap2 = floor(k2 / (TYPE) K2);
            const int k2mod = k2 - K2 * wrap2;
            const int k12mod = k2mod * K1;


            const TYPE v2r = coef2r * fmr;
            const TYPE v2i = coef2r * fmi;
            int k1 = koff1;

            for (jj1=0; jj1 < J1; jj1++, k1++) {
                const TYPE p1 = (t1 - k1) * L1;
                const int n1 = floor(p1);
                const TYPE alf1 = p1 - n1;
                register const TYPE * restrict ph1 = r_h1 + n1;
                register TYPE coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
                const int wrap1 = floor(k1 / (TYPE) K1);
                const int k1mod = k1 - K1 * wrap1;
                const int kk = k12mod + k1mod; /* 2D array index */


                r_ck[kk] += coef1r * v2r;
                i_ck[kk] += coef1r * v2i;
            } /* j1 */
        } /* j2 */
    }
}

/*
* interp2_table1_real_adj()
* 2D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp2_table1_real_adj)(
TYPE * restrict r_ck,   /* [K1,K2] in */
TYPE * restrict i_ck,
const int K1,
const int K2,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm,     /* [M,1] out */
const TYPE * restrict i_fm,
const int N)
{
    int nn;
    int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
       CAT(TYPE, _interp2_table1_real_adj_inner)(&r_ck[nn*K], &i_ck[nn*K],
                        K1, K2,
                        r_h1, r_h2,
                        J1, J2, L1, L2,
                        p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


void CAT(TYPE, _interp2_table1_complex_adj_inner)(
TYPE * restrict r_ck,       /* [K1,K2] out */
TYPE * restrict i_ck,
const int K1,
const int K2,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm, /* [M,1] in */
const TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor(J1 * L1/2);
    r_h1 += ncenter1;
    i_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor(J2 * L2/2);
    r_h2 += ncenter2;
    i_h2 += ncenter2;
    }

    /* initialize output to zero */
    (void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
    (void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1, jj2;
        const TYPE t2 = p_tm[M];
        const TYPE t1 = *p_tm++;
        const TYPE fmr = *r_fm++;
        const TYPE fmi = *i_fm++;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        int k2 = 1 + floor(t2 - J2 / 2.);

        for (jj2=0; jj2 < J2; jj2++, k2++) {
            const TYPE p2 = (t2 - k2) * L2;
            const int n2 = floor(p2);
            const TYPE alf2 = p2 - n2;
            register const TYPE * restrict r_ph2 = r_h2 + n2;
            register const TYPE * restrict i_ph2 = i_h2 + n2;
            /* const TYPE coef2r = r_h2[n2]; */
            /* const TYPE coef2i = i_h2[n2]; */
            TYPE coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
            TYPE coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
            const int k2mod = mymod(k2, K2);
            const int k12mod = k2mod * K1;

            const TYPE v2r = coef2r * fmr + coef2i * fmi;
            const TYPE v2i = coef2r * fmi - coef2i * fmr;
            int k1 = koff1;

            for (jj1=0; jj1 < J1; jj1++, k1++) {
                const TYPE p1 = (t1 - k1) * L1;
                const int n1 = floor(p1);
                const TYPE alf1 = p1 - n1;
                register const TYPE * restrict r_ph1 = r_h1 + n1;
                register const TYPE * restrict i_ph1 = i_h1 + n1;
                TYPE coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                TYPE coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);
                const int k1mod = mymod(k1, K1);
                const int kk = k12mod + k1mod; /* 2D array index */

                r_ck[kk] += coef1r * v2r + coef1i * v2i;
                i_ck[kk] += coef1r * v2i - coef1i * v2r;
            } /* j1 */
        } /* j2 */
    }
}

/*
* interp2_table1_complex_adj()
* 2D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp2_table1_complex_adj)(
TYPE * restrict r_ck,   /* [K1,K2] in */
TYPE * restrict i_ck,
const int K1,
const int K2,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm,     /* [M,1] out */
const TYPE * restrict i_fm,
const int N)
{
    int nn;
    int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
       CAT(TYPE, _interp2_table1_complex_adj_inner)(&r_ck[nn*K], &i_ck[nn*K],
                        K1, K2,
                        r_h1, i_h1, r_h2, i_h2,
                        J1, J2, L1, L2,
                        p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp2_table1_real_forward()
* 2D, 1st-order, real, periodic
*/
void CAT(TYPE, _interp2_table1_real_forward)(
const TYPE * restrict r_ck, /* [K1,K2] in */
const TYPE * restrict i_ck,
const int K1,
const int K2,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
TYPE * restrict r_fm,       /* [M,1] out */
TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor((double) (J1 * L1/2.));
    r_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor((double) (J2 * L2/2.));
    r_h2 += ncenter2;
    }

    /* interp */

    #pragma omp parallel for private(mm) schedule(dynamic,1000)  /* use omp even in 2D case? */
    for (mm=0; mm < M; mm++) {
        int jj1, jj2;
        const TYPE t2 = p_tm[M+mm];
        const TYPE t1 = p_tm[mm];

        // const TYPE t2 = p_tm[M;
        //const TYPE t1 = *p_tm++;
        TYPE sum2r = 0;
        TYPE sum2i = 0;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        int k2 = 1 + floor(t2 - J2 / 2.);

        for (jj2=0; jj2 < J2; jj2++, k2++) {
            const TYPE p2 = (t2 - k2) * L2;
            const int n2 = floor(p2);
            const TYPE alf2 = p2 - n2;
            register const TYPE * restrict ph2 = r_h2 + n2;
            TYPE coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
            const int wrap2 = floor(k2 / (TYPE) K2);
            const int k2mod = k2 - K2 * wrap2;
            const int k12mod = k2mod * K1;

            register TYPE sum1r = 0;
            register TYPE sum1i = 0;
            int k1 = koff1;

            for (jj1=0; jj1 < J1; jj1++, k1++) {
                const TYPE p1 = (t1 - k1) * L1;
                const int n1 = floor(p1);
                const TYPE alf1 = p1 - n1;
                register const TYPE * restrict ph1 = r_h1 + n1;
                register TYPE coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
                const int wrap1 = floor(k1 / (TYPE) K1);
                const int k1mod = k1 - K1 * wrap1;
                const int kk = k12mod + k1mod; /* 2D array index */

                /* sum1 += coef1 * ck */
                sum1r += coef1r * r_ck[kk];
                sum1i += coef1r * i_ck[kk];
            } /* j1 */

            /* sum2 += coef2 * sum1 */
            sum2r += coef2r * sum1r;
            sum2i += coef2r * sum1i;
        } /* j2 */
        r_fm[mm] = sum2r;
        i_fm[mm] = sum2i;
    }
}


/*
* interp2_table1_complex_forward()
* 2D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp2_table1_complex_forward)(
const TYPE * restrict r_ck, /* [K1,K2] in */
const TYPE * restrict i_ck,
const int K1,
const int K2,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
TYPE * restrict r_fm,       /* [M,1] out */
TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor(J1 * L1/2);
    r_h1 += ncenter1;
    i_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor(J2 * L2/2);
    r_h2 += ncenter2;
    i_h2 += ncenter2;
    }

    /* interp */
    #pragma omp parallel for private(mm) schedule(dynamic,1000)  /* use omp even in 2D case? */
    for (mm=0; mm < M; mm++) {
        int jj1, jj2;
        const TYPE t2 = p_tm[M+mm];
        const TYPE t1 = p_tm[mm];
        TYPE sum2r = 0;
        TYPE sum2i = 0;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        int k2 = 1 + floor(t2 - J2 / 2.);

        for (jj2=0; jj2 < J2; jj2++, k2++) {
            const TYPE p2 = (t2 - k2) * L2;
            const int n2 = floor(p2);
                    const TYPE alf2 = p2 - n2;
                    register const TYPE * restrict r_ph2 = r_h2 + n2;
                    register const TYPE * restrict i_ph2 = i_h2 + n2;
                    TYPE coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
                    TYPE coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
                const int k2mod = mymod(k2, K2);
            const int k12mod = k2mod * K1;

            register TYPE sum1r = 0;
            register TYPE sum1i = 0;
            int k1 = koff1;

            for (jj1=0; jj1 < J1; jj1++, k1++) {
                const TYPE p1 = (t1 - k1) * L1;
                const int n1 = floor(p1);
                        const TYPE alf1 = p1 - n1;
                        register const TYPE * restrict r_ph1 = r_h1 + n1;
                        register const TYPE * restrict i_ph1 = i_h1 + n1;
                    TYPE coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                        TYPE coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);
                    const int k1mod = mymod(k1, K1);
                const int kk = k12mod + k1mod; /* 2D array index */

                /* sum1 += coef1 * ck */
                sum1r += coef1r * r_ck[kk] - coef1i * i_ck[kk];
                sum1i += coef1r * i_ck[kk] + coef1i * r_ck[kk];
            } /* j1 */

            /* sum2 += coef2 * sum1 */
            sum2r += coef2r * sum1r - coef2i * sum1i;
            sum2i += coef2r * sum1i + coef2i * sum1r;
        } /* j2 */
        r_fm[mm] = sum2r;
        i_fm[mm] = sum2i;
    }
}


/*
* interp3_table1_complex_forward()
* 3D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp3_table1_complex_forward)(
const TYPE * restrict r_ck, /* [K1,K2,K3] in */
const TYPE * restrict i_ck,
const int K1,
const int K2,
const int K3,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict i_h2,
const TYPE * restrict r_h3, /* [J3*L3+1,1] in */
const TYPE * restrict i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE * restrict p_tm, /* [M,3] in */
const int M,
TYPE * restrict r_fm,       /* [M,1] out */
TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor(J1 * L1/2);
    r_h1 += ncenter1;
    i_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor(J2 * L2/2);
    r_h2 += ncenter2;
    i_h2 += ncenter2;
    }
    {
    const int ncenter3 = floor(J3 * L3/2);
    r_h3 += ncenter3;
    i_h3 += ncenter3;
    }

    /* interp */
    int jj1, jj2, jj3;
    #pragma omp parallel for private(mm, jj1, jj2, jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
        const TYPE t3 = p_tm[2*M+mm];
        const TYPE t2 = p_tm[M+mm];
        const TYPE t1 = p_tm[mm];
        TYPE sum3r = 0;
        TYPE sum3i = 0;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        const int koff2 = 1 + floor(t2 - J2 / 2.);
        int k3 = 1 + floor(t3 - J3 / 2.);

        for (jj3=0; jj3 < J3; jj3++, k3++) {
            const TYPE p3 = (t3 - k3) * L3;
            const int n3 = floor(p3);
            const TYPE alf3 = p3 - n3;
            TYPE coef3r = (1 - alf3) * r_h3[n3] + alf3 * r_h3[n3+1];
            TYPE coef3i = (1 - alf3) * i_h3[n3] + alf3 * i_h3[n3+1];
            const int k3mod = mymod(k3, K3);

            TYPE sum2r = 0;
            TYPE sum2i = 0;
            int k2 = koff2;

            for (jj2=0; jj2 < J2; jj2++, k2++) {
                const TYPE p2 = (t2 - k2) * L2;
                const int n2 = floor(p2);
                const TYPE alf2 = p2 - n2;
                TYPE coef2r = (1 - alf2) * r_h2[n2] + alf2 * r_h2[n2+1];
                TYPE coef2i = (1 - alf2) * i_h2[n2] + alf2 * i_h2[n2+1];
                const int k2mod = mymod(k2, K2);
                const int k23mod = (k3mod * K2 + k2mod) * K1;

                register TYPE sum1r = 0;
                register TYPE sum1i = 0;
                int k1 = koff1;

                for (jj1=0; jj1 < J1; jj1++, k1++) {
                    const TYPE p1 = (t1 - k1) * L1;
                    const int n1 = floor(p1);
                    const TYPE alf1 = p1 - n1;
                    TYPE coef1r = (1 - alf1) * r_h1[n1] + alf1 * r_h1[n1+1];
                    TYPE coef1i = (1 - alf1) * i_h1[n1] + alf1 * i_h1[n1+1];
                    const int k1mod = mymod(k1, K1);
                    const int kk = k23mod + k1mod; /* 3D array index */

                    /* sum1 += coef1 * ck */
                    sum1r += coef1r * r_ck[kk] - coef1i * i_ck[kk];
                    sum1i += coef1r * i_ck[kk] + coef1i * r_ck[kk];
                } /* j1 */

                /* sum2 += coef2 * sum1 */
                sum2r += coef2r * sum1r - coef2i * sum1i;
                sum2i += coef2r * sum1i + coef2i * sum1r;
            } /* j2 */

            /* sum3 += coef3 * sum2 */
            sum3r += coef3r * sum2r - coef3i * sum2i;
            sum3i += coef3r * sum2i + coef3i * sum2r;
        } /* j3 */

    r_fm[mm] = sum3r;
    i_fm[mm] = sum3i;
    }
}


/*
* interp3_table1_real_forward()
* 3D, 1st-order, real, periodic
*/
void CAT(TYPE, _interp3_table1_real_forward)(
const TYPE * restrict r_ck, /* [K1,K2,K3] in */
const TYPE * restrict i_ck,
const int K1,
const int K2,
const int K3,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict r_h3, /* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE * restrict p_tm, /* [M,3] in */
const int M,
TYPE * restrict r_fm,       /* [M,1] out */
TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor(J1 * L1/2);
    r_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor(J2 * L2/2);
    r_h2 += ncenter2;
    }
    {
    const int ncenter3 = floor(J3 * L3/2);
    r_h3 += ncenter3;
    }

    /* interp */
    int jj1, jj2, jj3;
    #pragma omp parallel for private(mm, jj1, jj2, jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
        const TYPE t3 = p_tm[2*M+mm];
        const TYPE t2 = p_tm[M+mm];
        const TYPE t1 = p_tm[mm];
        TYPE sum3r = 0;
        TYPE sum3i = 0;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        const int koff2 = 1 + floor(t2 - J2 / 2.);
        int k3 = 1 + floor(t3 - J3 / 2.);

        for (jj3=0; jj3 < J3; jj3++, k3++) {
            const TYPE p3 = (t3 - k3) * L3;
            const int n3 = floor(p3);
            const TYPE alf3 = p3 - n3;
            TYPE coef3r = (1 - alf3) * r_h3[n3] + alf3 * r_h3[n3+1];
            const int wrap3 = floor(k3 / (TYPE) K3);
            const int k3mod = k3 - K3 * wrap3;

            register TYPE sum2r = 0;
            register TYPE sum2i = 0;
            int k2 = koff2;

            for (jj2=0; jj2 < J2; jj2++, k2++) {
                const TYPE p2 = (t2 - k2) * L2;
                const int n2 = floor(p2);
                const TYPE alf2 = p2 - n2;
                TYPE coef2r = (1 - alf2) * r_h2[n2] + alf2 * r_h2[n2+1];
                const int wrap2 = floor(k2 / (TYPE) K2);
                const int k2mod = k2 - K2 * wrap2;
                const int k23mod = (k3mod * K2 + k2mod) * K1;

                register TYPE sum1r = 0;
                register TYPE sum1i = 0;
                int k1 = koff1;

                for (jj1=0; jj1 < J1; jj1++, k1++) {
                    const TYPE p1 = (t1 - k1) * L1;
                    const int n1 = floor(p1);
                    const TYPE alf1 = p1 - n1;
                    register TYPE coef1r = (1 - alf1) * r_h1[n1] + alf1 * r_h1[n1+1];
                    const int wrap1 = floor(k1 / (TYPE) K1);
                    const int k1mod = k1 - K1 * wrap1;
                    const int kk = k23mod + k1mod;

                    /* sum1 += coef1 * ck */
                    sum1r += coef1r * r_ck[kk];
                    sum1i += coef1r * i_ck[kk];
                } /* j1 */

                /* sum2 += coef2 * sum1 */
                sum2r += coef2r * sum1r;
                sum2i += coef2r * sum1i;
            } /* j2 */

            /* sum3 += coef3 * sum2 */
            sum3r += coef3r * sum2r;
            sum3i += coef3r * sum2i;
        } /* j3 */

        r_fm[mm] = sum3r;
        i_fm[mm] = sum3i;
    }
}


void CAT(TYPE, _interp3_table1_complex_adj_inner)(
TYPE * restrict r_ck,       /* [K1,K2,K3] out */
TYPE * restrict i_ck,
const int K1,
const int K2,
const int K3,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict i_h2,
const TYPE * restrict r_h3, /* [J3*L3+1,1] in */
const TYPE * restrict i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE * restrict p_tm, /* [M,3] in */
const int M,
const TYPE * restrict r_fm, /* [M,1] in */
const TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor(J1 * L1/2);
    r_h1 += ncenter1;
    i_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor(J2 * L2/2);
    r_h2 += ncenter2;
    i_h2 += ncenter2;
    }
    {
    const int ncenter3 = floor(J3 * L3/2);
    r_h3 += ncenter3;
    i_h3 += ncenter3;
    }

    /* initialize output to zero */
    (void) memset((void *) r_ck, 0, K1*K2*K3*sizeof(*r_ck));
    (void) memset((void *) i_ck, 0, K1*K2*K3*sizeof(*i_ck));

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1, jj2, jj3;
        const TYPE t3 = p_tm[2*M];
        const TYPE t2 = p_tm[M];
        const TYPE t1 = *p_tm++;
        const TYPE fmr = *r_fm++;
        const TYPE fmi = *i_fm++;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        const int koff2 = 1 + floor(t2 - J2 / 2.);
        int k3 = 1 + floor(t3 - J3 / 2.);

        for (jj3=0; jj3 < J3; jj3++, k3++) {
            const TYPE p3 = (t3 - k3) * L3;
            const int n3 = floor(p3);
            const TYPE alf3 = p3 - n3;
            register const TYPE * restrict r_ph3 = r_h3 + n3;
                    register const TYPE * restrict i_ph3 = i_h3 + n3;
            TYPE coef3r = (1 - alf3) * *r_ph3 + alf3 * *(r_ph3+1);
                    TYPE coef3i = (1 - alf3) * *i_ph3 + alf3 * *(i_ph3+1);
            const int k3mod = mymod(k3, K3);

            const TYPE v3r = coef3r * fmr + coef3i * fmi;
            const TYPE v3i = coef3r * fmi - coef3i * fmr;
            int k2 = koff2;

            for (jj2=0; jj2 < J2; jj2++, k2++) {
                const TYPE p2 = (t2 - k2) * L2;
                const int n2 = floor(p2);
                const TYPE alf2 = p2 - n2;
                        register const TYPE * restrict r_ph2 = r_h2 + n2;
                        register const TYPE * restrict i_ph2 = i_h2 + n2;
                TYPE coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
                        TYPE coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
                const int k2mod = mymod(k2, K2);
                const int k23mod = (k3mod * K2 + k2mod) * K1;

                const TYPE v2r = coef2r * v3r + coef2i * v3i;
                const TYPE v2i = coef2r * v3i - coef2i * v3r;
                int k1 = koff1;

                for (jj1=0; jj1 < J1; jj1++, k1++) {
                    const TYPE p1 = (t1 - k1) * L1;
                    const int n1 = floor(p1);
                            const TYPE alf1 = p1 - n1;
                            register const TYPE * restrict r_ph1 = r_h1 + n1;
                            register const TYPE * restrict i_ph1 = i_h1 + n1;
                    TYPE coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                            TYPE coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);

                    const int k1mod = mymod(k1, K1);
                    const int kk = k23mod + k1mod; /* 3D array index */

                    r_ck[kk] += coef1r * v2r + coef1i * v2i;
                    i_ck[kk] += coef1r * v2i - coef1i * v2r;

                } /* j1 */
                } /* j2 */
        } /* j3 */
    }
}


/*
* interp3_table1_complex_adj()
* 3D, 1st order, complex, periodic
*/
void CAT(TYPE, _interp3_table1_complex_adj)(
TYPE * restrict r_ck,   /* [K1,K2] in */
TYPE * restrict i_ck,
const int K1,
const int K2,
const int K3,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict i_h1,
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict i_h2,
const TYPE * restrict r_h3, /* [J3*L3+1,1] in */
const TYPE * restrict i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm,     /* [M,1] out */
const TYPE * restrict i_fm,
const int N)
{
    int nn;
    int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
       CAT(TYPE, _interp3_table1_complex_adj_inner)(&r_ck[nn*K], &i_ck[nn*K],
                        K1, K2, K3,
                        r_h1, i_h1, r_h2, i_h2, r_h3, i_h3,
                        J1, J2, J3, L1, L2, L3,
                        p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


void CAT(TYPE, _interp3_table1_real_adj_inner)(
TYPE * restrict r_ck,       /* [K1,K2,K3] out */
TYPE * restrict i_ck,
const int K1,
const int K2,
const int K3,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict r_h3, /* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE * restrict p_tm, /* [M,3] in */
const int M,
const TYPE * restrict r_fm, /* [M,1] in */
const TYPE * restrict i_fm)
{
    int mm;

    /* trick: shift table pointer to center */
    {
    const int ncenter1 = floor(J1 * L1/2);
    r_h1 += ncenter1;
    }
    {
    const int ncenter2 = floor(J2 * L2/2);
    r_h2 += ncenter2;
    }
    {
    const int ncenter3 = floor(J3 * L3/2);
    r_h3 += ncenter3;
    }

    /* initialize output to zero */
    (void) memset((void *) r_ck, 0, K1*K2*K3*sizeof(*r_ck));
    (void) memset((void *) i_ck, 0, K1*K2*K3*sizeof(*i_ck));

    /* interp */
    for (mm=0; mm < M; mm++) {
        int jj1, jj2, jj3;
        const TYPE t3 = p_tm[2*M];
        const TYPE t2 = p_tm[M];
        const TYPE t1 = *p_tm++;
        const TYPE fmr = *r_fm++;
        const TYPE fmi = *i_fm++;
        const int koff1 = 1 + floor(t1 - J1 / 2.);
        const int koff2 = 1 + floor(t2 - J2 / 2.);
        int k3 = 1 + floor(t3 - J3 / 2.);

        for (jj3=0; jj3 < J3; jj3++, k3++) {
            const TYPE p3 = (t3 - k3) * L3;
            const int n3 = floor(p3);
            const TYPE alf3 = p3 - n3;
            register const TYPE * restrict ph3 = r_h3 + n3;
            TYPE coef3r = (1 - alf3) * *ph3 + alf3 * *(ph3+1);
            const int wrap3 = floor(k3 / (TYPE) K3);
            const int k3mod = k3 - K3 * wrap3;

            const TYPE v3r = coef3r * fmr;
            const TYPE v3i = coef3r * fmi;
            int k2 = koff2;

            for (jj2=0; jj2 < J2; jj2++, k2++) {
                const TYPE p2 = (t2 - k2) * L2;
                const int n2 = floor(p2);
                const TYPE alf2 = p2 - n2;
                register const TYPE * restrict ph2 = r_h2 + n2;
                TYPE coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
                const int wrap2 = floor(k2 / (TYPE) K2);
                const int k2mod = k2 - K2 * wrap2;
                const int k23mod = (k3mod * K2 + k2mod) * K1;

                const TYPE v2r = coef2r * v3r;
                const TYPE v2i = coef2r * v3i;
                int k1 = koff1;

                for (jj1=0; jj1 < J1; jj1++, k1++) {
                    const TYPE p1 = (t1 - k1) * L1;
                    const int n1 = floor(p1);
                    const TYPE alf1 = p1 - n1;
                    register const TYPE * restrict ph1 = r_h1 + n1;
                    register TYPE coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
                    const int wrap1 = floor(k1 / (TYPE) K1);
                    const int k1mod = k1 - K1 * wrap1;
                    const int kk = k23mod + k1mod;

                    r_ck[kk] += coef1r * v2r;
                    i_ck[kk] += coef1r * v2i;
                } /* j1 */
            } /* j2 */
        } /* j3 */
    }
}

/*
* interp3_table1_real_adj()
* 3D, 1st order, real, periodic
*/
void CAT(TYPE, _interp3_table1_real_adj)(
TYPE * restrict r_ck,   /* [K1,K2] in */
TYPE * restrict i_ck,
const int K1,
const int K2,
const int K3,
const TYPE * restrict r_h1, /* [J1*L1+1,1] in */
const TYPE * restrict r_h2, /* [J2*L2+1,1] in */
const TYPE * restrict r_h3, /* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const TYPE * restrict p_tm, /* [M,2] in */
const int M,
const TYPE * restrict r_fm,     /* [M,1] out */
const TYPE * restrict i_fm,
const int N)
{
    int nn;
    int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
       CAT(TYPE, _interp3_table1_real_adj_inner)(&r_ck[nn*K], &i_ck[nn*K],
                        K1, K2, K3,
                        r_h1, r_h2, r_h3,
                        J1, J2, J3, L1, L2, L3,
                        p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


#undef restrict
#endif /* TYPE */
