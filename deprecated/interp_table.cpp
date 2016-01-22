/*
*
* Copyright 2004-3 Yingying Zhang and Jeff Fessler, University of Michigan
* 
*/
#include <math.h>
#include <stdio.h>
//#include "def,interp,table.h"

#include <string.h> /* for memset */

#define mymod(k,K) ((k) - (K) * floor((k) / (double) (K)))
#define iround(x) floor(x + 0.5)

#ifdef _WIN32 //_WIN32 also exists on 64-bit windows machines so this should work for both
#define EXPORTED __declspec(dllexport)
#else
#define EXPORTED
#endif


/*
* interp1_table0_complex_per()
* 1D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp1_table0_complex_per(
const double *r_ck,	/* [K1,1] in */
const double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,	/* imaginary part of complex interpolator */
const int J1,
const int L1,
const double *p_tm,	/* [M,1] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	i_h1 += ncenter1;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1;
	const double t1 = *p_tm++;
	register double sum1r = 0;
	register double sum1i = 0;
	int k1 = 1 + floor(t1 - J1 / 2.);

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const double coef1r = r_h1[n1];
		register const double coef1i = i_h1[n1];
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
* interp1_table0_real_per()
* 1D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp1_table0_real_per(
const double *r_ck,	/* [K,1] in */
const double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in (real) */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
#endif
const int J1,
const int L1,
const double *p_tm,	/* [M,1] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1;
	const double t1 = *p_tm++;
	register double sum1r = 0;
	register double sum1i = 0;
	int k1 = 1 + floor(t1 - J1 / 2.);

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register double coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* sum1 += coef1 * ck */
		sum1r += coef1r * r_ck[k1mod];
		sum1i += coef1r * i_ck[k1mod];
	}
	*r_fm++ = sum1r;
	*i_fm++ = sum1i;
    }
}


/*
* interp1_table1_real_per()
* 1D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp1_table1_real_per(
const double *r_ck,	/* [K,1] in */
const double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in (real) */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
#endif
const int J1,
const int L1,
const double *p_tm,	/* [M,1] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1;
	const double t1 = *p_tm++;
	register double sum1r = 0;
	register double sum1i = 0;
	int k1 = 1 + floor(t1 - J1 / 2.);

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const double alf1 = p1 - n1;
		register const double *ph1 = r_h1 + n1;
		register double coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* sum1 += coef1 * ck */
		sum1r += coef1r * r_ck[k1mod];
		sum1i += coef1r * i_ck[k1mod];
	}
	*r_fm++ = sum1r;
	*i_fm++ = sum1i;
    }
}


/*
* interp1_table0_complex_per_adj_inner()
* 1D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp1_table0_complex_per_adj_inner(
double *r_ck,		/* [K1,1] out */
double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,	/* imaginary part of complex interpolator */
const int J1,
const int L1,
const double *p_tm,	/* [M,1] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
{
	int mm;

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*sizeof(*i_ck));

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	i_h1 += ncenter1;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1;
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	int k1 = 1 + floor(t1 - J1 / 2.);

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const double coef1r = r_h1[n1];
		register const double coef1i = i_h1[n1];
		const int k1mod = mymod(k1, K1);

		/* instead of f = h c, we have c += h^* f */
		r_ck[k1mod] += coef1r * fmr + coef1i * fmi;
		i_ck[k1mod] += coef1r * fmi - coef1i * fmr;
	}
    }
}

/*
* interp1_table0_complex_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp1_table0_complex_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const int J1,
const int L1,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp1_table0_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1,
        				r_h1, i_h1, 
        				J1, L1, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp1_table0_real_per()
* 1D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp1_table0_real_per_adj_inner(
double *r_ck,		/* [K1,1] out */
double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in (real) */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
#endif
const int J1,
const int L1,
const double *p_tm,	/* [M,1] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
{
	int mm;

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*sizeof(*i_ck));

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1;
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	int k1 = 1 + floor(t1 - J1 / 2.);

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register double coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* instead of f = h c, we have c += h^* f */
		r_ck[k1mod] += coef1r * fmr;
		i_ck[k1mod] += coef1r * fmi;
	}
    }
}


/*
* interp1_table0_real_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp1_table0_real_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in */
const int J1,
const int L1,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp1_table0_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1,
        				r_h1,  
        				J1, L1, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}

/*
* interp1_table1_real_per_adj_inner()
* 1D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp1_table1_real_per_adj_inner(
double *r_ck,		/* [K1,1] out */
double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in (real) */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
#endif
const int J1,
const int L1,
const double *p_tm,	/* [M,1] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
{
	int mm;

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*sizeof(*i_ck));

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1;
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	int k1 = 1 + floor(t1 - J1 / 2.);

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const double alf1 = p1 - n1;
		register const double *ph1 = r_h1 + n1;
		register double coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* instead of f = h c, we have c += h^* f */
		r_ck[k1mod] += coef1r * fmr;
		i_ck[k1mod] += coef1r * fmi;
	}
    }
}

/*
* interp1_table1_real_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp1_table1_real_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const double *r_h1,	/* [J1*L1+1,1] in */
const int J1,
const int L1,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp1_table1_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1,
        				r_h1, 
        				J1, L1, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp2_table0_complex_per_adj_inner()
*/
extern "C" EXPORTED void interp2_table0_complex_per_adj_inner(
double *r_ck,		/* [K1,K2] out */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	i_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	i_h2 += ncenter2;
	}

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const double coef2r = r_h2[n2];
		const double coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		const double v2r = coef2r * fmr + coef2i * fmi;
		const double v2i = coef2r * fmi - coef2i * fmr;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const double coef1r = r_h1[n1];
		register const double coef1i = i_h1[n1];
		const int k1mod = mymod(k1, K1);
		const int kk = k12mod + k1mod; /* 2D array index */

		r_ck[kk] += coef1r * v2r + coef1i * v2i;
		i_ck[kk] += coef1r * v2i - coef1i * v2r;
	} /* j1 */
	} /* j2 */
    }
}

/*
* interp2_table0_complex_per()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp2_table0_complex_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2_table0_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, i_h1, r_h2, i_h2,
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp2_table0_real_per_adj_inner()
* 2D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp2_table0_real_per_adj_inner(
double *r_ck,		/* [K1,K2] out */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		double coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const double v2r = coef2r * fmr;
		const double v2i = coef2r * fmi;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register double coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
    }
}

/*
* interp2_table0_real_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp2_table0_real_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2_table0_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, r_h2, 
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}

/*
* interp2_table1_real_per_adj_inner()
* 2D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp2_table1_real_per_adj_inner(
double *r_ck,		/* [K1,K2] out */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const double alf2 = p2 - n2;
		register const double *ph2 = r_h2 + n2;
		double coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const double v2r = coef2r * fmr;
		const double v2i = coef2r * fmi;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const double alf1 = p1 - n1;
		register const double *ph1 = r_h1 + n1;
		register double coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
    }
}

/*
* interp2_table1_real_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp2_table1_real_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2_table1_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, r_h2, 
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp2_table1_complex_per_adj_inner()
* 2D, 1st-order, complex, periodic
*/
extern "C" EXPORTED void interp2_table1_complex_per_adj_inner(
double *r_ck,		/* [K1,K2] out */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
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
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
        const double alf2 = p2 - n2;
        register const double *r_ph2 = r_h2 + n2;
        register const double *i_ph2 = i_h2 + n2;
		/* const double coef2r = r_h2[n2]; */
		/* const double coef2i = i_h2[n2]; */
        double coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
        double coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
		const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		const double v2r = coef2r * fmr + coef2i * fmi;
		const double v2i = coef2r * fmi - coef2i * fmr;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
        const int n1 = floor(p1);
        const double alf1 = p1 - n1;
        register const double *r_ph1 = r_h1 + n1;
        register const double *i_ph1 = i_h1 + n1;
		double coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
        double coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);
		const int k1mod = mymod(k1, K1);
		const int kk = k12mod + k1mod; /* 2D array index */

		r_ck[kk] += coef1r * v2r + coef1i * v2i;
		i_ck[kk] += coef1r * v2i - coef1i * v2r;
	} /* j1 */
	} /* j2 */
    }
}

/*
* interp2_table1_complex_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp2_table1_complex_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2_table1_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, i_h1, r_h2, i_h2,
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}



extern "C" EXPORTED void interp2_table0_complex_per(
const double *r_ck,	/* [K1,K2] in */
const double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	i_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	i_h2 += ncenter2;
	}

    
    /* interp */
    for (mm=0; mm < M; mm++) {
    	int jj1, jj2;
    	const double t2 = p_tm[M];
    	const double t1 = *p_tm++;
    	double sum2r = 0;
    	double sum2i = 0;
    	const int koff1 = 1 + floor(t1 - J1 / 2.);
    	int k2 = 1 + floor(t2 - J2 / 2.);
    
    	for (jj2=0; jj2 < J2; jj2++, k2++) {
    		const double p2 = (t2 - k2) * L2;
    		const int n2 = /* ncenter2 + */ iround(p2);
    		const double coef2r = r_h2[n2];
    		const double coef2i = i_h2[n2];
    		const int k2mod = mymod(k2, K2);
    		const int k12mod = k2mod * K1;
    
    		register double sum1r = 0;
    		register double sum1i = 0;
    		int k1 = koff1;
    
    	for (jj1=0; jj1 < J1; jj1++, k1++) {
    		const double p1 = (t1 - k1) * L1;
    		const int n1 = /* ncenter1 + */ iround(p1);
    		register const double coef1r = r_h1[n1];
    		register const double coef1i = i_h1[n1];
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
    
    	*r_fm++ = sum2r;
    	*i_fm++ = sum2i;
    }

}


/*
* interp2_table0_real_per()
* 2D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp2_table0_real_per(
const double *r_ck,	/* [K1,K2] in */
const double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	double sum2r = 0;
	double sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		double coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register double coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* sum1 += coef1 * ck */
		sum1r += coef1r * r_ck[kk];
		sum1i += coef1r * i_ck[kk];
	} /* j1 */

		/* sum2 += coef2 * sum1 */
		sum2r += coef2r * sum1r;
		sum2i += coef2r * sum1i;
	} /* j2 */

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}

/*
* interp2_table1_real_per()
* 2D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp2_table1_real_per(
const double *r_ck,	/* [K1,K2] in */
const double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	double sum2r = 0;
	double sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const double alf2 = p2 - n2;
		register const double *ph2 = r_h2 + n2;
		double coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const double alf1 = p1 - n1;
		register const double *ph1 = r_h1 + n1;
		register double coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* sum1 += coef1 * ck */
		sum1r += coef1r * r_ck[kk];
		sum1i += coef1r * i_ck[kk];
	} /* j1 */

		/* sum2 += coef2 * sum1 */
		sum2r += coef2r * sum1r;
		sum2i += coef2r * sum1i;
	} /* j2 */

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}


/*
* interp2_table1_complex_per()
* 2D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp2_table1_complex_per(
const double *r_ck,	/* [K1,K2] in */
const double *i_ck,
const int K1,
const int K2,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const double *p_tm,	/* [M,2] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
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
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	double sum2r = 0;
	double sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
                const double alf2 = p2 - n2;
                register const double *r_ph2 = r_h2 + n2;
                register const double *i_ph2 = i_h2 + n2;
                double coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
                double coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
        	const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
                const double alf1 = p1 - n1;
                register const double *r_ph1 = r_h1 + n1;
                register const double *i_ph1 = i_h1 + n1;
        	double coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                double coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);
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

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}


/*
* interp2_table0_complex_per_adj_inner()
*/
extern "C" EXPORTED void interp2f_table0_complex_per_adj_inner(
float *r_ck,		/* [K1,K2] out */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	i_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	i_h2 += ncenter2;
	}

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const float coef2r = r_h2[n2];
		const float coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		const float v2r = coef2r * fmr + coef2i * fmi;
		const float v2i = coef2r * fmi - coef2i * fmr;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const float coef1r = r_h1[n1];
		register const float coef1i = i_h1[n1];
		const int k1mod = mymod(k1, K1);
		const int kk = k12mod + k1mod; /* 2D array index */

		r_ck[kk] += coef1r * v2r + coef1i * v2i;
		i_ck[kk] += coef1r * v2i - coef1i * v2r;
	} /* j1 */
	} /* j2 */
    }
}

/*
* interp2f_table0_complex_per_adj()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp2f_table0_complex_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2f_table0_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, i_h1, r_h2, i_h2,
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp2_table0_real_per_adj_inner()
* 2D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp2f_table0_real_per_adj_inner(
float *r_ck,		/* [K1,K2] out */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		float coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const float v2r = coef2r * fmr;
		const float v2i = coef2r * fmi;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register float coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
    }
}


/*
* interp2f_table0_real_per_adj()
* 2D, 0th order, real, periodic
*/
extern "C" EXPORTED void interp2f_table0_real_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2f_table0_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, r_h2,
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}

/*
* interp2_table1_real_per_adj_inner()
* 2D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp2f_table1_real_per_adj_inner(
float *r_ck,		/* [K1,K2] out */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* initialize output to zero */
	(void) memset((void *) r_ck, 0, K1*K2*sizeof(*r_ck));
	(void) memset((void *) i_ck, 0, K1*K2*sizeof(*i_ck));

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const float alf2 = p2 - n2;
		register const float *ph2 = r_h2 + n2;
		float coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const float v2r = coef2r * fmr;
		const float v2i = coef2r * fmi;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const float alf1 = p1 - n1;
		register const float *ph1 = r_h1 + n1;
		register float coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
    }
}

/*
* interp2f_table1_real_per_adj()
* 2D, 1st order, real, periodic
*/
extern "C" EXPORTED void interp2f_table1_real_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2f_table1_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, r_h2, 
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}

/*
* interp2_table1_complex_per_adj_inner()
* 2D, 1st-order, complex, periodic
*/
extern "C" EXPORTED void interp2f_table1_complex_per_adj_inner(
float *r_ck,		/* [K1,K2] out */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
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
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
        const float alf2 = p2 - n2;
        register const float *r_ph2 = r_h2 + n2;
        register const float *i_ph2 = i_h2 + n2;
		/* const float coef2r = r_h2[n2]; */
		/* const float coef2i = i_h2[n2]; */
        float coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
        float coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
		const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		const float v2r = coef2r * fmr + coef2i * fmi;
		const float v2i = coef2r * fmi - coef2i * fmr;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
        const int n1 = floor(p1);
        const float alf1 = p1 - n1;
        register const float *r_ph1 = r_h1 + n1;
        register const float *i_ph1 = i_h1 + n1;
		float coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
        float coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);
		const int k1mod = mymod(k1, K1);
		const int kk = k12mod + k1mod; /* 2D array index */

		r_ck[kk] += coef1r * v2r + coef1i * v2i;
		i_ck[kk] += coef1r * v2i - coef1i * v2r;
	} /* j1 */
	} /* j2 */
    }
}


/*
* interp2f_table1_complex_per_adj()
* 2D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp2f_table1_complex_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp2f_table1_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, 
        				r_h1, i_h1, r_h2, i_h2,
        				J1, J2, L1, L2, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}

/*
* interp2_table0_complex_per()
* 2D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp2f_table0_complex_per(
const float *r_ck,	/* [K1,K2] in */
const float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	i_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	i_h2 += ncenter2;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	float sum2r = 0;
	float sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const float coef2r = r_h2[n2];
		const float coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const float coef1r = r_h1[n1];
		register const float coef1i = i_h1[n1];
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

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}


/*
* interp2_table0_real_per()
* 2D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp2f_table0_real_per(
const float *r_ck,	/* [K1,K2] in */
const float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	float sum2r = 0;
	float sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		float coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register float coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* sum1 += coef1 * ck */
		sum1r += coef1r * r_ck[kk];
		sum1i += coef1r * i_ck[kk];
	} /* j1 */

		/* sum2 += coef2 * sum1 */
		sum2r += coef2r * sum1r;
		sum2i += coef2r * sum1i;
	} /* j2 */

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}

/*
* interp2_table1_real_per()
* 2D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp2f_table1_real_per(
const float *r_ck,	/* [K1,K2] in */
const float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
#endif
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
{
	int mm;

	/* trick: shift table pointer to center */
	{
	const int ncenter1 = floor(double(J1 * L1/2.));
	r_h1 += ncenter1;
	}
	{
	const int ncenter2 = floor(double(J2 * L2/2.));
	r_h2 += ncenter2;
	}

	/* interp */
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	float sum2r = 0;
	float sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const float alf2 = p2 - n2;
		register const float *ph2 = r_h2 + n2;
		float coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k12mod = k2mod * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const float alf1 = p1 - n1;
		register const float *ph1 = r_h1 + n1;
		register float coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k12mod + k1mod; /* 2D array index */

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		/* sum1 += coef1 * ck */
		sum1r += coef1r * r_ck[kk];
		sum1i += coef1r * i_ck[kk];
	} /* j1 */

		/* sum2 += coef2 * sum1 */
		sum2r += coef2r * sum1r;
		sum2i += coef2r * sum1i;
	} /* j2 */

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}


/*
* interp2_table1_complex_per()
* 2D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp2f_table1_complex_per(
const float *r_ck,	/* [K1,K2] in */
const float *i_ck,
const int K1,
const int K2,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const int J1,
const int J2,
const int L1,
const int L2,
const float *p_tm,	/* [M,2] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
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
    for (mm=0; mm < M; mm++) {
	int jj1, jj2;
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	float sum2r = 0;
	float sum2i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	int k2 = 1 + floor(t2 - J2 / 2.);

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
                const float alf2 = p2 - n2;
                register const float *r_ph2 = r_h2 + n2;
                register const float *i_ph2 = i_h2 + n2;
                float coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
                float coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
        	const int k2mod = mymod(k2, K2);
		const int k12mod = k2mod * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
                const float alf1 = p1 - n1;
                register const float *r_ph1 = r_h1 + n1;
                register const float *i_ph1 = i_h1 + n1;
        	float coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                float coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);
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

	*r_fm++ = sum2r;
	*i_fm++ = sum2i;
    }
}

/*
* interp3f_table0_complex_per()
* 3D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp3f_table0_complex_per(
const float *r_ck,	/* [K1,K2,K3] in */
const float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const float *r_h3,	/* [J3*L3+1,1] in */
const float *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
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
    #pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const float t3 = p_tm[2*M+mm];
	const float t2 = p_tm[M+mm];
	const float t1 = p_tm[mm];
	float sum3r = 0;
	float sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		const float coef3r = r_h3[n3];
		const float coef3i = i_h3[n3];
		const int k3mod = mymod(k3, K3);

		float sum2r = 0;
		float sum2i = 0;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const float coef2r = r_h2[n2];
		const float coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const float coef1r = r_h1[n1];
		register const float coef1i = i_h1[n1];
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
* interp3f_table1_complex_per()
* 3D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp3f_table1_complex_per(
const float *r_ck,	/* [K1,K2,K3] in */
const float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const float *r_h3,	/* [J3*L3+1,1] in */
const float *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const float t3 = p_tm[2*M+mm];
	const float t2 = p_tm[M+mm];
	const float t1 = p_tm[mm];
	float sum3r = 0;
	float sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const float alf3 = p3 - n3;
		float coef3r = (1 - alf3) * r_h3[n3] + alf3 * r_h3[n3+1];
        float coef3i = (1 - alf3) * i_h3[n3] + alf3 * i_h3[n3+1];
		const int k3mod = mymod(k3, K3);

		float sum2r = 0;
		float sum2i = 0;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
        const int n2 = floor(p2);
		const float alf2 = p2 - n2;
		float coef2r = (1 - alf2) * r_h2[n2] + alf2 * r_h2[n2+1];
        float coef2i = (1 - alf2) * i_h2[n2] + alf2 * i_h2[n2+1];
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
        const float alf1 = p1 - n1;
		float coef1r = (1 - alf1) * r_h1[n1] + alf1 * r_h1[n1+1];
        float coef1i = (1 - alf1) * i_h1[n1] + alf1 * i_h1[n1+1];
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
* interp3f_table0_real_per()
* 3D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp3f_table0_real_per(
const float *r_ck,	/* [K1,K2,K3] in */
const float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const float *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const float t3 = p_tm[2*M+mm];
	const float t2 = p_tm[M+mm];
	const float t1 = p_tm[mm];
	float sum3r = 0;
	float sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		float coef3r = r_h3[n3];
		const int wrap3 = floor(k3 / (float) K3);
		const int k3mod = k3 - K3 * wrap3;

		register float sum2r = 0;
		register float sum2i = 0;
		int k2 = koff2;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		float coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register float coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

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


/*
* interp3f_table1_real_per()
* 3D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp3f_table1_real_per(
const float *r_ck,	/* [K1,K2,K3] in */
const float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const float *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
float *r_fm,		/* [M,1] out */
float *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const float t3 = p_tm[2*M+mm];
	const float t2 = p_tm[M+mm];
	const float t1 = p_tm[mm];
	float sum3r = 0;
	float sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const float alf3 = p3 - n3;
		float coef3r = (1 - alf3) * r_h3[n3] + alf3 * r_h3[n3+1];
		const int wrap3 = floor(k3 / (float) K3);
		const int k3mod = k3 - K3 * wrap3;

		register float sum2r = 0;
		register float sum2i = 0;
		int k2 = koff2;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const float alf2 = p2 - n2;
		float coef2r = (1 - alf2) * r_h2[n2] + alf2 * r_h2[n2+1];
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register float sum1r = 0;
		register float sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const float alf1 = p1 - n1;
		register float coef1r = (1 - alf1) * r_h1[n1] + alf1 * r_h1[n1+1];
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

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


/*
* interp3_table0_complex_per()
* 3D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp3_table0_complex_per(
const double *r_ck,	/* [K1,K2,K3] in */
const double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const double *r_h3,	/* [J3*L3+1,1] in */
const double *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const double t3 = p_tm[2*M+mm];
	const double t2 = p_tm[M+mm];
	const double t1 = p_tm[mm];
	double sum3r = 0;
	double sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		const double coef3r = r_h3[n3];
		const double coef3i = i_h3[n3];
		const int k3mod = mymod(k3, K3);

		double sum2r = 0;
		double sum2i = 0;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const double coef2r = r_h2[n2];
		const double coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const double coef1r = r_h1[n1];
		register const double coef1i = i_h1[n1];
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
* interp3_table1_complex_per()
* 3D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp3_table1_complex_per(
const double *r_ck,	/* [K1,K2,K3] in */
const double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const double *r_h3,	/* [J3*L3+1,1] in */
const double *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const double t3 = p_tm[2*M+mm];
	const double t2 = p_tm[M+mm];
	const double t1 = p_tm[mm];
	double sum3r = 0;
	double sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const double alf3 = p3 - n3;
		double coef3r = (1 - alf3) * r_h3[n3] + alf3 * r_h3[n3+1];
        double coef3i = (1 - alf3) * i_h3[n3] + alf3 * i_h3[n3+1];
		const int k3mod = mymod(k3, K3);

		double sum2r = 0;
		double sum2i = 0;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
        const int n2 = floor(p2);
		const double alf2 = p2 - n2;
		double coef2r = (1 - alf2) * r_h2[n2] + alf2 * r_h2[n2+1];
        double coef2i = (1 - alf2) * i_h2[n2] + alf2 * i_h2[n2+1];
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
        const double alf1 = p1 - n1;
		double coef1r = (1 - alf1) * r_h1[n1+1] + alf1 * r_h1[n1+1];
        double coef1i = (1 - alf1) * i_h1[n1] + alf1 * i_h1[n1+1];
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
* interp3_table0_real_per()
* 3D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp3_table0_real_per(
const double *r_ck,	/* [K1,K2,K3] in */
const double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const double *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const double t3 = p_tm[2*M+mm];
	const double t2 = p_tm[M+mm];
	const double t1 = p_tm[mm];
	double sum3r = 0;
	double sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		double coef3r = r_h3[n3];
		const int wrap3 = floor(k3 / (double) K3);
		const int k3mod = k3 - K3 * wrap3;

		register double sum2r = 0;
		register double sum2i = 0;
		int k2 = koff2;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		double coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register double coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

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


/*
* interp3_table1_real_per()
* 3D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp3_table1_real_per(
const double *r_ck,	/* [K1,K2,K3] in */
const double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const double *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
double *r_fm,		/* [M,1] out */
double *i_fm)
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
	#pragma omp parallel for private(mm, jj1, jj2,jj3) schedule(dynamic,1000)  /* Result is identical to the case without openmp*/
    for (mm=0; mm < M; mm++) {
	const double t3 = p_tm[2*M+mm];
	const double t2 = p_tm[M+mm];
	const double t1 = p_tm[mm];
	double sum3r = 0;
	double sum3i = 0;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const double alf3 = p3 - n3;
		double coef3r = (1 - alf3) * r_h3[n3] + alf3 * r_h3[n3+1];
		const int wrap3 = floor(k3 / (double) K3);
		const int k3mod = k3 - K3 * wrap3;

		register double sum2r = 0;
		register double sum2i = 0;
		int k2 = koff2;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const double alf2 = p2 - n2;
		double coef2r = (1 - alf2) * r_h2[n2] + alf2 * r_h2[n2+1];
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		register double sum1r = 0;
		register double sum1i = 0;
		int k1 = koff1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const double alf1 = p1 - n1;
		register double coef1r = (1 - alf1) * r_h1[n1] + alf1 * r_h1[n1+1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

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

/*
* interp3_table0_complex_per_adj_inner()
*/
extern "C" EXPORTED void interp3_table0_complex_per_adj_inner(
double *r_ck,		/* [K1,K2,K3] out */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const double *r_h3,	/* [J3*L3+1,1] in */
const double *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
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
	const double t3 = p_tm[2*M];
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		const double coef3r = r_h3[n3];
		const double coef3i = i_h3[n3];
		const int k3mod = mymod(k3, K3);

		const double v3r = coef3r * fmr + coef3i * fmi;
		const double v3i = coef3r * fmi - coef3i * fmr;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const double coef2r = r_h2[n2];
		const double coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		const double v2r = coef2r * v3r + coef2i * v3i;
		const double v2i = coef2r * v3i - coef2i * v3r;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const double coef1r = r_h1[n1];
		register const double coef1i = i_h1[n1];
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
* interp3_table0_complex_per_adj()
* 3D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp3_table0_complex_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const double *r_h3,	/* [J3*L3+1,1] in */
const double *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3_table0_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, i_h1, r_h2, i_h2, r_h3, i_h3,
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp3_table1_complex_per_adj_inner()
* 3D, 1st-order, complex, periodic
*/
extern "C" EXPORTED void interp3_table1_complex_per_adj_inner(
double *r_ck,		/* [K1,K2,K3] out */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const double *r_h3,	/* [J3*L3+1,1] in */
const double *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
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
	const double t3 = p_tm[2*M];
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const double alf3 = p3 - n3;
		register const double *r_ph3 = r_h3 + n3;
                register const double *i_ph3 = i_h3 + n3;
		double coef3r = (1 - alf3) * *r_ph3 + alf3 * *(r_ph3+1);
                double coef3i = (1 - alf3) * *i_ph3 + alf3 * *(i_ph3+1);
		const int k3mod = mymod(k3, K3);

		const double v3r = coef3r * fmr + coef3i * fmi;
		const double v3i = coef3r * fmi - coef3i * fmr;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2); 
		const double alf2 = p2 - n2;
                register const double *r_ph2 = r_h2 + n2;
                register const double *i_ph2 = i_h2 + n2;
		double coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
                double coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		const double v2r = coef2r * v3r + coef2i * v3i;
		const double v2i = coef2r * v3i - coef2i * v3r;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
                const double alf1 = p1 - n1;
                register const double *r_ph1 = r_h1 + n1;
                register const double *i_ph1 = i_h1 + n1;
		double coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                double coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);

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
* interp3_table1_complex_per_adj()
* 3D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp3_table1_complex_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *i_h1,
const double *r_h2,	/* [J2*L2+1,1] in */
const double *i_h2,
const double *r_h3,	/* [J3*L3+1,1] in */
const double *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3_table1_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, i_h1, r_h2, i_h2, r_h3, i_h3,
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp3_table0_real_per_adj_inner()
* 3D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp3_table0_real_per_adj_inner(
double *r_ck,		/* [K1,K2,K3] out */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const double *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
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
	const double t3 = p_tm[2*M];
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		double coef3r = r_h3[n3];
		const int wrap3 = floor(k3 / (double) K3);
		const int k3mod = k3 - K3 * wrap3;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

		const double v3r = coef3r * fmr;
		const double v3i = coef3r * fmi;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		double coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const double v2r = coef2r * v3r;
		const double v2i = coef2r * v3i;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register double coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
	} /* j3 */
    }
}

/*
* interp3_table0_real_per_adj()
* 3D, 0th order, real, periodic
*/
extern "C" EXPORTED void interp3_table0_real_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const double *r_h3,	/* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3_table0_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, r_h2, r_h3, 
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}
/*
* interp3_table1_real_per_adj_inner()
* 3D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp3_table1_real_per_adj_inner(
double *r_ck,		/* [K1,K2,K3] out */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const double *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,3] in */
const int M,
const double *r_fm,	/* [M,1] in */
const double *i_fm)
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
	const double t3 = p_tm[2*M];
	const double t2 = p_tm[M];
	const double t1 = *p_tm++;
	const double fmr = *r_fm++;
	const double fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const double p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const double alf3 = p3 - n3;
		register const double *ph3 = r_h3 + n3;
		double coef3r = (1 - alf3) * *ph3 + alf3 * *(ph3+1);
		const int wrap3 = floor(k3 / (double) K3);
		const int k3mod = k3 - K3 * wrap3;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

		const double v3r = coef3r * fmr;
		const double v3i = coef3r * fmi;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const double p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const double alf2 = p2 - n2;
		register const double *ph2 = r_h2 + n2;
		double coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
		const int wrap2 = floor(k2 / (double) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const double v2r = coef2r * v3r;
		const double v2i = coef2r * v3i;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const double p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const double alf1 = p1 - n1;
		register const double *ph1 = r_h1 + n1;
		register double coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (double) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
	} /* j3 */
    }
}

/*
* interp3_table1_real_per_adj()
* 3D, 1st order, real, periodic
*/
extern "C" EXPORTED void interp3_table1_real_per_adj(
double *r_ck,	/* [K1,K2] in */
double *i_ck,
const int K1,
const int K2,
const int K3,
const double *r_h1,	/* [J1*L1+1,1] in */
const double *r_h2,	/* [J2*L2+1,1] in */
const double *r_h3,	/* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const double *p_tm,	/* [M,2] in */
const int M,
const double *r_fm,		/* [M,1] out */
const double *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3_table1_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, r_h2, r_h3, 
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}

/*
* interp3_table0f_complex_per_adj_inner()
*/
extern "C" EXPORTED void interp3f_table0_complex_per_adj_inner(
float *r_ck,		/* [K1,K2,K3] out */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const float *r_h3,	/* [J3*L3+1,1] in */
const float *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
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
	const float t3 = p_tm[2*M];
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		const float coef3r = r_h3[n3];
		const float coef3i = i_h3[n3];
		const int k3mod = mymod(k3, K3);

		const float v3r = coef3r * fmr + coef3i * fmi;
		const float v3i = coef3r * fmi - coef3i * fmr;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		const float coef2r = r_h2[n2];
		const float coef2i = i_h2[n2];
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		const float v2r = coef2r * v3r + coef2i * v3i;
		const float v2i = coef2r * v3i - coef2i * v3r;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register const float coef1r = r_h1[n1];
		register const float coef1i = i_h1[n1];
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
* interp3f_table0_complex_per_adj()
* 3D, 0th order, complex, periodic
*/
extern "C" EXPORTED void interp3f_table0_complex_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const float *r_h3,	/* [J3*L3+1,1] in */
const float *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3f_table0_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, i_h1, r_h2, i_h2, r_h3, i_h3,
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp3f_table1_complex_per_adj_inner()
* 3D, 1st-order, complex, periodic
*/
extern "C" EXPORTED void interp3f_table1_complex_per_adj_inner(
float *r_ck,		/* [K1,K2,K3] out */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const float *r_h3,	/* [J3*L3+1,1] in */
const float *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
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
	const float t3 = p_tm[2*M];
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const float alf3 = p3 - n3;
		register const float *r_ph3 = r_h3 + n3;
                register const float *i_ph3 = i_h3 + n3;
		float coef3r = (1 - alf3) * *r_ph3 + alf3 * *(r_ph3+1);
                float coef3i = (1 - alf3) * *i_ph3 + alf3 * *(i_ph3+1);
		const int k3mod = mymod(k3, K3);

		const float v3r = coef3r * fmr + coef3i * fmi;
		const float v3i = coef3r * fmi - coef3i * fmr;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2); 
		const float alf2 = p2 - n2;
                register const float *r_ph2 = r_h2 + n2;
                register const float *i_ph2 = i_h2 + n2;
		float coef2r = (1 - alf2) * *r_ph2 + alf2 * *(r_ph2+1);
                float coef2i = (1 - alf2) * *i_ph2 + alf2 * *(i_ph2+1);
		const int k2mod = mymod(k2, K2);
		const int k23mod = (k3mod * K2 + k2mod) * K1;

		const float v2r = coef2r * v3r + coef2i * v3i;
		const float v2i = coef2r * v3i - coef2i * v3r;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
                const float alf1 = p1 - n1;
                register const float *r_ph1 = r_h1 + n1;
                register const float *i_ph1 = i_h1 + n1;
		float coef1r = (1 - alf1) * *r_ph1 + alf1 * *(r_ph1+1);
                float coef1i = (1 - alf1) * *i_ph1 + alf1 * *(i_ph1+1);

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
* interp3f_table1_complex_per_adj()
* 3D, 1st order, complex, periodic
*/
extern "C" EXPORTED void interp3f_table1_complex_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *i_h1,
const float *r_h2,	/* [J2*L2+1,1] in */
const float *i_h2,
const float *r_h3,	/* [J3*L3+1,1] in */
const float *i_h3,
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3f_table1_complex_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, i_h1, r_h2, i_h2, r_h3, i_h3,
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


/*
* interp3f_table0_real_per_adj_inner()
* 3D, 0th-order, real, periodic
*/
extern "C" EXPORTED void interp3f_table0_real_per_adj_inner(
float *r_ck,		/* [K1,K2,K3] out */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const float *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
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
	const float t3 = p_tm[2*M];
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = /* ncenter3 + */ iround(p3);
		float coef3r = r_h3[n3];
		const int wrap3 = floor(k3 / (float) K3);
		const int k3mod = k3 - K3 * wrap3;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

		const float v3r = coef3r * fmr;
		const float v3i = coef3r * fmi;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = /* ncenter2 + */ iround(p2);
		float coef2r = r_h2[n2];
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const float v2r = coef2r * v3r;
		const float v2i = coef2r * v3i;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = /* ncenter1 + */ iround(p1);
		register float coef1r = r_h1[n1];
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
	} /* j3 */
    }
}

/*
* interp3f_table0_real_per_adj()
* 3D, 0th order, real, periodic
*/
extern "C" EXPORTED void interp3f_table0_real_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const float *r_h3,	/* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3f_table0_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, r_h2, r_h3, 
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}
/*
* interp3f_table1_real_per_adj_inner()
* 3D, 1st-order, real, periodic
*/
extern "C" EXPORTED void interp3f_table1_real_per_adj_inner(
float *r_ck,		/* [K1,K2,K3] out */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const float *r_h3,	/* [J3*L3+1,1] in */
#ifdef Provide_flip
const int flip1,	/* sign flips every K? */
const int flip2,
const int flip3,
#endif
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,3] in */
const int M,
const float *r_fm,	/* [M,1] in */
const float *i_fm)
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
	const float t3 = p_tm[2*M];
	const float t2 = p_tm[M];
	const float t1 = *p_tm++;
	const float fmr = *r_fm++;
	const float fmi = *i_fm++;
	const int koff1 = 1 + floor(t1 - J1 / 2.);
	const int koff2 = 1 + floor(t2 - J2 / 2.);
	int k3 = 1 + floor(t3 - J3 / 2.);

	for (jj3=0; jj3 < J3; jj3++, k3++) {
		const float p3 = (t3 - k3) * L3;
		const int n3 = floor(p3);
		const float alf3 = p3 - n3;
		register const float *ph3 = r_h3 + n3;
		float coef3r = (1 - alf3) * *ph3 + alf3 * *(ph3+1);
		const int wrap3 = floor(k3 / (float) K3);
		const int k3mod = k3 - K3 * wrap3;

#ifdef Provide_flip
		if (flip3 && (wrap3 % 2))
			coef3r = -coef3r; /* trick: sign flip */
#endif

		const float v3r = coef3r * fmr;
		const float v3i = coef3r * fmi;
		int k2 = koff2;

	for (jj2=0; jj2 < J2; jj2++, k2++) {
		const float p2 = (t2 - k2) * L2;
		const int n2 = floor(p2);
		const float alf2 = p2 - n2;
		register const float *ph2 = r_h2 + n2;
		float coef2r = (1 - alf2) * *ph2 + alf2 * *(ph2+1);
		const int wrap2 = floor(k2 / (float) K2);
		const int k2mod = k2 - K2 * wrap2;
		const int k23mod = (k3mod * K2 + k2mod) * K1;

#ifdef Provide_flip
		if (flip2 && (wrap2 % 2))
			coef2r = -coef2r; /* trick: sign flip */
#endif

		const float v2r = coef2r * v3r;
		const float v2i = coef2r * v3i;
		int k1 = koff1;

	for (jj1=0; jj1 < J1; jj1++, k1++) {
		const float p1 = (t1 - k1) * L1;
		const int n1 = floor(p1);
		const float alf1 = p1 - n1;
		register const float *ph1 = r_h1 + n1;
		register float coef1r = (1 - alf1) * *ph1 + alf1 * *(ph1+1);
		const int wrap1 = floor(k1 / (float) K1);
		const int k1mod = k1 - K1 * wrap1;
		const int kk = k23mod + k1mod;

#ifdef Provide_flip
		if (flip1 && (wrap1 % 2))
			coef1r = -coef1r; /* trick: sign flip */
#endif

		r_ck[kk] += coef1r * v2r;
		i_ck[kk] += coef1r * v2i;
	} /* j1 */
	} /* j2 */
	} /* j3 */
    }
}

/*
* interp3f_table1_real_per_adj()
* 3D, 1st order, real, periodic
*/
extern "C" EXPORTED void interp3f_table1_real_per_adj(
float *r_ck,	/* [K1,K2] in */
float *i_ck,
const int K1,
const int K2,
const int K3,
const float *r_h1,	/* [J1*L1+1,1] in */
const float *r_h2,	/* [J2*L2+1,1] in */
const float *r_h3,	/* [J3*L3+1,1] in */
const int J1,
const int J2,
const int J3,
const int L1,
const int L2,
const int L3,
const float *p_tm,	/* [M,2] in */
const int M,
const float *r_fm,		/* [M,1] out */
const float *i_fm,
const int N)
{
    int nn;
    static int K = K1*K2*K3;

    #pragma omp parallel for private(nn)
    for (nn=0; nn<N; nn++) {
        interp3f_table1_real_per_adj_inner(&r_ck[nn*K], &i_ck[nn*K],
        				K1, K2, K3, 
        				r_h1, r_h2, r_h3, 
        				J1, J2, J3, L1, L2, L3, 
        				p_tm, M, &r_fm[nn*M], &i_fm[nn*M]);
    }

}


extern "C" EXPORTED int main()
{
  return 0;
}


