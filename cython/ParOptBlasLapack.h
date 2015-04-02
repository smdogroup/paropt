#ifndef PAR_OPT_BLAS_LAPACK_H
#define PAR_OPT_BLAS_LAPACK_H

/*
  Copyright (c) 2014-2015 Graeme Kennedy. All rights reserved
*/

// The following are the definitions required for BLAS/LAPACK
#define BLASddot     ddot_
#define BLASdnrm2    dnrm2_
#define BLASdaxpy    daxpy_
#define BLASdscal    dscal_
#define LAPACKdgetrf dgetrf_
#define LAPACKdgetrs dgetrs_
#define LAPACKdpptrf dpptrf_
#define LAPACKdpptrs dpptrs_

extern "C" {
  extern double BLASddot( int *n, double *x, int *incx, 
			      double *y, int *incy );
  extern double BLASdnrm2( int *n, double *x, int *incx );
  extern void BLASdaxpy( int *n, double *a, double *x, int *incx, 
			 double *y, int *incy );
  extern void BLASdscal( int *n, double *a, double *x, int *incx );
  
  // General factorization routines
  extern void LAPACKdgetrf( int *m, int *n, 
			    double *a, int *lda, int *ipiv, int * info );
  extern void LAPACKdgetrs( const char *c, int *n, int *nrhs, 
			    double *a, int *lda, int *ipiv, 
			    double *b, int *ldb, int *info );
  
  // Factorization of packed-storage matrices
  extern void LAPACKdpptrf( const char *c, int *n, double *ap, int *info );
  extern void LAPACKdpptrs( const char *c, int *n, int *nrhs,
			    double *ap, double *rhs, int *ldrhs, int *info );
}

#endif
