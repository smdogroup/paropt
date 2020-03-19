#ifndef PAR_OPT_BLAS_LAPACK_H
#define PAR_OPT_BLAS_LAPACK_H

#include "ParOptVec.h"

// The following are the definitions required for BLAS/LAPACK
#ifdef PAROPT_USE_COMPLEX
#define BLASddot     zdotu_
#define BLASdnrm2    dznrm2_
#define BLASdaxpy    zaxpy_
#define BLASdscal    zscal_
#define LAPACKdgetrf zgetrf_
#define LAPACKdgetrs zgetrs_
#define LAPACKdpptrf zpptrf_
#define LAPACKdpptrs zpptrs_
#else
#define BLASddot     ddot_
#define BLASdnrm2    dnrm2_
#define BLASdaxpy    daxpy_
#define BLASdscal    dscal_
#define LAPACKdgetrf dgetrf_
#define LAPACKdgetrs dgetrs_
#define LAPACKdpptrf dpptrf_
#define LAPACKdpptrs dpptrs_
#endif // PAROPT_USE_COMPLEX

extern "C" {
  extern ParOptScalar BLASddot( int *n, ParOptScalar *x, int *incx,
                                ParOptScalar *y, int *incy );
  extern double BLASdnrm2( int *n, ParOptScalar *x, int *incx );
  extern void BLASdaxpy( int *n, ParOptScalar *a, ParOptScalar *x, int *incx,
                         ParOptScalar *y, int *incy );
  extern void BLASdscal( int *n, ParOptScalar *a, ParOptScalar *x, int *incx );

  // General factorization routines
  extern void LAPACKdgetrf( int *m, int *n,
                            ParOptScalar *a, int *lda, int *ipiv, int * info );
  extern void LAPACKdgetrs( const char *c, int *n, int *nrhs,
                            ParOptScalar *a, int *lda, int *ipiv,
                            ParOptScalar *b, int *ldb, int *info );

  // Factorization of packed-storage matrices
  extern void LAPACKdpptrf( const char *c, int *n, ParOptScalar *ap, int *info );
  extern void LAPACKdpptrs( const char *c, int *n, int *nrhs,
                            ParOptScalar *ap, ParOptScalar *rhs,
                            int *ldrhs, int *info );
}

#endif
