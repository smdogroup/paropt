#include "ParOptSparseMat.h"

#include <algorithm>

#include "ParOptComplexStep.h"
#include "ParOptSparseCholesky.h"
#include "ParOptSparseUtils.h"

ParOptQuasiDefBlockMat::ParOptQuasiDefBlockMat(ParOptProblem *prob0,
                                               int _nwblock) {
  nwblock = _nwblock;

  prob = prob0;
  prob->incref();
  prob->getProblemSizes(&nvars, NULL, &nwcon);

  x = NULL;
  Dinv = NULL;
  C = NULL;

  // Allocate space for the block-diagonal matrix
  Cw = new ParOptScalar[nwcon * (nwblock + 1) / 2];
}

ParOptQuasiDefBlockMat::~ParOptQuasiDefBlockMat() {
  if (x) {
    x->decref();
  }
  if (Dinv) {
    Dinv->decref();
  }
  if (C) {
    C->decref();
  }
  prob->decref();
  delete[] Cw;
}

int ParOptQuasiDefBlockMat::factor(ParOptVec *x0, ParOptVec *Dinv0,
                                   ParOptVec *C0) {
  // Hang on to the x, D and C vectors
  x0->incref();
  if (x) {
    x->decref();
  }
  x = x0;
  Dinv0->incref();
  if (Dinv) {
    Dinv->decref();
  }
  Dinv = Dinv0;
  C0->incref();
  if (C) {
    C->decref();
  }
  C = C0;

  // Set the values in the Cw diagonal matrix
  memset(Cw, 0, nwcon * (nwblock + 1) / 2 * sizeof(ParOptScalar));

  ParOptScalar *cvals;
  C->getArray(&cvals);

  // Set the pointer and the increment for the
  // block-diagonal matrix
  ParOptScalar *cw = Cw;
  const int incr = ((nwblock + 1) * nwblock) / 2;

  // Iterate over each block matrix
  for (int i = 0; i < nwcon; i += nwblock) {
    // Index into each block
    for (int j = 0, k = 0; j < nwblock; j++, k += j + 1) {
      cw[k] = cvals[i + j];
    }

    // Increment the pointer to the next block
    cw += incr;
  }

  // Next, complete the evaluation of Cw by adding the following
  // contribution to the matrix
  // Cw += Aw*D^{-1}*Aw^{T}
  if (nwcon > 0) {
    prob->addSparseInnerProduct(1.0, x, Dinv, Cw);
  }

  if (nwblock == 1) {
    for (int i = 0; i < nwcon; i++) {
      // Compute and store Cw^{-1}
      if (Cw[i] == 0.0) {
        return 1;
      } else {
        Cw[i] = 1.0 / Cw[i];
      }
    }
  } else {
    ParOptScalar *cw = Cw;
    const int incr = ((nwblock + 1) * nwblock) / 2;
    for (int i = 0; i < nwcon; i += nwblock) {
      // Factor the matrix using the Cholesky factorization
      // for upper-triangular packed storage
      int info = 0;
      LAPACKdpptrf("U", &nwblock, cw, &info);

      if (info) {
        return i + info;
      }
      cw += incr;
    }
  }

  return 0;
}

/*
  Solve the quasi-definite system of equations

  Here bx is unmodified. Note the negative sign on the yw variables.
*/
void ParOptQuasiDefBlockMat::apply(ParOptVec *bx, ParOptVec *yx,
                                   ParOptVec *yw) {
  // Compute the right-hand-side for the Schur complement
  // yw = Aw * D^{-1} * bx.
  // First compute yx = D^{-1} * bx
  ParOptScalar *yxvals, *bxvals, *dvals;
  yx->getArray(&yxvals);
  bx->getArray(&bxvals);
  Dinv->getArray(&dvals);
  for (int i = 0; i < nvars; i++) {
    yxvals[i] = dvals[i] * bxvals[i];
  }

  // Compute yw = -Aw * yx
  yw->zeroEntries();
  if (nwcon > 0) {
    prob->addSparseJacobian(-1.0, x, yx, yw);
  }

  // Apply the factorization to the yw entries
  applyFactor(yw);

  // Compute yx = D^{-1}(bx + Aw^{T} * yw)
  yx->copyValues(bx);
  if (nwcon > 0) {
    prob->addSparseJacobianTranspose(1.0, x, yw, yx);
  }
  for (int i = 0; i < nvars; i++) {
    yxvals[i] *= dvals[i];
  }
}

/*
  Solve the quasi-definite system of equations

  In the call bx and bw must remain unmodified. Note the negative sign on the
  yw variables.
*/
void ParOptQuasiDefBlockMat::apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx,
                                   ParOptVec *yw) {
  // Compute the right-hand-side for the Schur complement
  // yw = Aw * D^{-1} * bx.
  // First compute yx = D^{-1} * bx
  ParOptScalar *yxvals, *bxvals, *dvals;
  yx->getArray(&yxvals);
  bx->getArray(&bxvals);
  Dinv->getArray(&dvals);
  for (int i = 0; i < nvars; i++) {
    yxvals[i] = bxvals[i] * dvals[i];
  }

  // Compute yw = bw - Aw * yx
  yw->copyValues(bw);
  if (nwcon > 0) {
    prob->addSparseJacobian(-1.0, x, yx, yw);
  }

  // Apply the factorization to the yw entries
  applyFactor(yw);

  // Compute yx = D^{-1}(bx + Aw^{T} * yw)
  yx->copyValues(bx);
  if (nwcon > 0) {
    prob->addSparseJacobianTranspose(1.0, x, yw, yx);
  }
  for (int i = 0; i < nvars; i++) {
    yxvals[i] *= dvals[i];
  }
}

/*
  Apply the factored Cw-matrix that is stored as a series of block-symmetric
  matrices.
*/
int ParOptQuasiDefBlockMat::applyFactor(ParOptVec *vec) {
  ParOptScalar *rhs;
  vec->getArray(&rhs);

  if (nwblock == 1) {
    for (int i = 0; i < nwcon; i++) {
      rhs[i] *= Cw[i];
    }
  } else {
    ParOptScalar *cw = Cw;
    const int incr = ((nwblock + 1) * nwblock) / 2;
    for (int i = 0; i < nwcon; i += nwblock) {
      // Factor the matrix using the Cholesky factorization
      // for the upper-triangular packed storage format
      int info = 0, one = 1;
      LAPACKdpptrs("U", &nwblock, &one, cw, rhs, &nwblock, &info);

      if (info) {
        return i + info;
      }

      // Increment the pointers to the next block
      rhs += nwblock;
      cw += incr;
    }
  }

  return 0;
}

const char *ParOptQuasiDefBlockMat::getFactorInfo() {
  snprintf(info, sizeof(info), "nblock: %d", nwblock);
  return info;
}

/*
  A simple serial LDL sparse matrix factorization
*/
ParOptQuasiDefSparseMat::ParOptQuasiDefSparseMat(ParOptSparseProblem *problem) {
  prob = problem;
  prob->incref();

  prob->getProblemSizes(&nvars, NULL, &nwcon);

  // Get the sparse Jacobian information in CSR format
  const int *rowp = NULL, *cols = NULL;
  prob->getSparseJacobianData(&rowp, &cols, NULL);

  // Compute the sparse matrix transpose
  colp = new int[nvars + 1];
  rows = new int[rowp[nwcon]];
  ParOptSparseTranspose(nwcon, nvars, rowp, cols, NULL, colp, rows, NULL);

  // Count up the number of dense columns
  ndense = 0;
  for (int i = 0; i < nvars; i++) {
    if ((colp[i + 1] - colp[i] > 0.5 * nwcon)) {
      ndense++;
    }
  }

  int rhs_size = nwcon;
  if (nvars > nwcon) {
    rhs_size = nvars;
  }
  rhs = new ParOptScalar[rhs_size];

  chol = NULL;
  Dinv = NULL;
  Atvals = NULL;
  Kcolp = NULL;
  Krows = NULL;
  Kvals = NULL;
}

ParOptQuasiDefSparseMat::~ParOptQuasiDefSparseMat() {
  prob->decref();

  if (Dinv) {
    Dinv->decref();
  }

  // Delete the transpose constraint Jacobian data
  delete[] colp;
  delete[] rows;
  if (Atvals) {
    delete[] Atvals;
  }

  // Delete the Schur complement matrix data
  if (Kcolp) {
    delete[] Kcolp;
  }
  if (Krows) {
    delete[] Krows;
  }
  if (Kvals) {
    delete[] Kvals;
  }

  // Free the right-hand-side
  delete[] rhs;
}

/*
  Compute the elements and factor the sparse matrix
*/
int ParOptQuasiDefSparseMat::factor(ParOptVec *x, ParOptVec *Dinv0,
                                    ParOptVec *C) {
  Dinv0->incref();
  if (Dinv) {
    Dinv->decref();
  }
  Dinv = Dinv0;

  ParOptScalar *dvals, *cvals;
  Dinv->getArray(&dvals);
  C->getArray(&cvals);

  // Get the sparse Jacobian information in CSR format
  const int *rowp = NULL, *cols = NULL;
  const ParOptScalar *data;
  prob->getSparseJacobianData(&rowp, &cols, &data);

  // Compute the transpose of the constraint Jacobian
  if (!Atvals) {
    Atvals = new ParOptScalar[rowp[nwcon]];
  }
  ParOptSparseTranspose(nwcon, nvars, rowp, cols, data, colp, rows, Atvals);

  if (!chol) {
    // Compute the non-zero pattern of the full matrix
    int *flag = new int[nwcon];
    Kcolp = new int[nwcon + 1];
    int nnz = ParOptMatMatTransSymbolic(nwcon, nvars, rowp, cols, colp, rows,
                                        Kcolp, flag);

    // Compute the values of the matrix
    Krows = new int[nnz];
    Kvals = new ParOptScalar[nnz];
    ParOptMatMatTransNumeric(nwcon, nvars, cvals, rowp, cols, data, dvals, colp,
                             rows, Atvals, Kcolp, Krows, Kvals, flag, rhs);

    delete[] flag;

    // Allocate the sparse Cholesky factorization
    int use_nd_order = 1;  // Use the nested-disection ordering
    chol = new ParOptSparseCholesky(nwcon, Kcolp, Krows, use_nd_order);
  } else {
    int *flag = new int[nwcon];
    ParOptMatMatTransNumeric(nwcon, nvars, cvals, rowp, cols, data, dvals, colp,
                             rows, Atvals, Kcolp, Krows, Kvals, flag, rhs);
    delete[] flag;
  }

  chol->setValues(nwcon, Kcolp, Krows, Kvals);
  int fail = chol->factor();

  return fail;
}

void ParOptQuasiDefSparseMat::apply(ParOptVec *bx, ParOptVec *yx,
                                    ParOptVec *yw) {
  ParOptScalar *bx_array, *dvals;
  bx->getArray(&bx_array);
  Dinv->getArray(&dvals);

  // Get the solution array
  ParOptScalar *yw_array;
  yw->getArray(&yw_array);

  for (int i = 0; i < nvars; i++) {
    rhs[i] = dvals[i] * bx_array[i];
  }

  // Get the sparse Jacobian information in CSR format
  const int *rowp = NULL, *cols = NULL;
  const ParOptScalar *data;
  prob->getSparseJacobianData(&rowp, &cols, &data);

  // Compute yw = - A * D^{-1} * bx
  ParOptCSRMatVec(-1.0, nwcon, rowp, cols, data, rhs, 0.0, yw_array);

  // Solve the problem for (C + A * D * A^{T}) * yw = bw - A * D^{-1} * bx
  chol->solve(yw_array);

  // Compute yx = D^{-1} * (bx + A^{T} * yw)
  ParOptCSCMatVec(1.0, nvars, nwcon, rowp, cols, data, yw_array, 0.0, rhs);

  ParOptScalar *yx_array;
  yx->getArray(&yx_array);
  for (int i = 0; i < nvars; i++) {
    yx_array[i] = dvals[i] * (bx_array[i] + rhs[i]);
  }
}

void ParOptQuasiDefSparseMat::apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx,
                                    ParOptVec *yw) {
  ParOptScalar *bx_array, *bw_array, *dvals;
  bx->getArray(&bx_array);
  bw->getArray(&bw_array);
  Dinv->getArray(&dvals);

  // Get the solution array
  ParOptScalar *yw_array;
  yw->getArray(&yw_array);

  for (int i = 0; i < nvars; i++) {
    rhs[i] = dvals[i] * bx_array[i];
  }
  for (int i = 0; i < nwcon; i++) {
    yw_array[i] = bw_array[i];
  }

  // Get the sparse Jacobian information in CSR format
  const int *rowp = NULL, *cols = NULL;
  const ParOptScalar *data;
  prob->getSparseJacobianData(&rowp, &cols, &data);

  // Compute yw = bw - A * D^{-1} * bx
  ParOptCSRMatVec(-1.0, nwcon, rowp, cols, data, rhs, 1.0, yw_array);

  // Solve the problem for (C + A * D * A^{T}) * yw = bw - A * D^{-1} * bx
  chol->solve(yw_array);

  // Compute rhs = A^{T} * yw
  ParOptCSCMatVec(1.0, nvars, nwcon, rowp, cols, data, yw_array, 0.0, rhs);

  // Compute yx = D^{-1} * (bx + A^{T} * yw)
  ParOptScalar *yx_array;
  yx->getArray(&yx_array);
  for (int i = 0; i < nvars; i++) {
    yx_array[i] = dvals[i] * (bx_array[i] + rhs[i]);
  }
}

const char *ParOptQuasiDefSparseMat::getFactorInfo() {
  if (Kcolp && chol) {
    // Only count the non-zeros in the symmetric part of the matrix
    int nnzK = (Kcolp[nwcon] + nwcon) / 2;

    // Get information from the factorization
    int n, num_snodes, nnzL;
    chol->getInfo(&n, &num_snodes, &nnzL);

    snprintf(info, sizeof(info),
             "n %5d nsnodes %5d ndense %3d nnz(K) %7d nnz(L) %7d nnz(L) / "
             "nnz(K) %8.4f sparsity(L) %8.2e",
             nwcon, num_snodes, ndense, nnzK, nnzL, 1.0 * nnzL / nnzK,
             1.0 * nnzL / (nwcon * (nwcon + 1) / 2));

    return info;
  }
  return NULL;
}
