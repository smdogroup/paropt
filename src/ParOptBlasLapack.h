#ifndef PAR_OPT_BLAS_LAPACK_H
#define PAR_OPT_BLAS_LAPACK_H

// The following are the definitions required for BLAS/LAPACK
#define BLASddot     ddot_
#define BLASdnrm2    dnrm2_
#define BLASdaxpy    daxpy_
#define BLASdscal    dscal_
#define LAPACKdgetrf dgetrf_
#define LAPACKdgetrs dgetrs_

extern "C" {
  extern TacsScalar BLASddot( int *n, TacsScalar *x, int *incx, 
			      TacsScalar *y, int *incy );
  extern double BLASdnrm2( int *n, TacsScalar *x, int *incx );
  extern void BLASdaxpy( int *n, TacsScalar *a, TacsScalar *x, int *incx, 
			 TacsScalar *y, int *incy );
  extern void BLASscal( int *n, TacsScalar *a, TacsScalar *x, int *incx );
  // Compute an LU factorization of a matrix
  extern void LAPACKgetrf( int *m, int *n, 
			   TacsScalar *a, int *lda, int *ipiv, int * info );

  // This routine solves a system of equations with a factored matrix
  extern void LAPACKgetrs( const char *c, int *n, int *nrhs, 
			   TacsScalar *a, int *lda, int *ipiv, 
			   TacsScalar *b, int *ldb, int *info );
}

#endif
