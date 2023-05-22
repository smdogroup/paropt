#ifndef PAR_OPT_SPARSE_UTILS_H
#define PAR_OPT_SPARSE_UTILS_H

#include "ParOptComplexStep.h"
#include "ParOptVec.h"

// Compute y = alpha * A * x + beta * y
void ParOptCSRMatVec(double alpha, int nrows, const int *rowp, const int *cols,
                     const ParOptScalar *Avals, const ParOptScalar *x,
                     double beta, ParOptScalar *y);

// Compute A * x -> y
void ParOptCSCMatVec(double alpha, int nrows, int ncols, const int *colp,
                     const int *rows, const ParOptScalar *Avals,
                     const ParOptScalar *x, double beta, ParOptScalar *y);

// Based on the pattern of A, compute A^{T}. The numerical values are optional
void ParOptSparseTranspose(int nrows, int ncols, const int *rowp,
                           const int *cols, const ParOptScalar *Avals,
                           int *colp, int *rows, ParOptScalar *ATvals);

// Compute the number of non-zeros in the matrix product A * A^{T}
int ParOptMatMatTransSymbolic(int nrows, int ncols, const int *rowp,
                              const int *cols, const int *colp, const int *rows,
                              int *Bcolp, int *flag);

// Compute the matrix-matrix product
void ParOptMatMatTransNumeric(int nrows, int ncols, const int *rowp,
                              const int *cols, const ParOptScalar *Avals,
                              const int *colp, const int *rows,
                              const ParOptScalar *ATvals, const int *Bcolp,
                              int *Brows, ParOptScalar *Bvals, int *flag,
                              ParOptScalar *tmp);

#endif  // PAR_OPT_SPARSE_UTILS_H