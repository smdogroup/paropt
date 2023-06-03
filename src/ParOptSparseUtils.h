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

// Compute the matrix-matrix product A * A^{T}
void ParOptMatMatTransNumeric(int nrows, int ncols, const int *rowp,
                              const int *cols, const ParOptScalar *Avals,
                              const int *colp, const int *rows,
                              const ParOptScalar *ATvals, const int *Bcolp,
                              int *Brows, ParOptScalar *Bvals, int *flag,
                              ParOptScalar *tmp);

// Compute the result C + A * D * A^{T}, where C and D are diagonal
void ParOptMatMatTransNumeric(int nrows, int ncols, const ParOptScalar *cvals,
                              const int *rowp, const int *cols,
                              const ParOptScalar *Avals,
                              const ParOptScalar *dvals, const int *colp,
                              const int *rows, const ParOptScalar *ATvals,
                              const int *Bcolp, int *Brows, ParOptScalar *Bvals,
                              int *flag, ParOptScalar *tmp);

// Remove duplicates from a list
int ParOptRemoveDuplicates(int *array, int len, int exclude = -1);

// Sort and make the data structure unique - remove diagonal
void ParOptSortAndRemoveDuplicates(int nvars, int *rowp, int *cols,
                                   int remove_diagonal = 0);

#endif  // PAR_OPT_SPARSE_UTILS_H