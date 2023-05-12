#include "ParOptSparseMat.h"

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

/*
  A simple serial LDL sparse matrix factorization
*/

ParOptQuasiDefSparseMat::ParOptQuasiDefSparseMat(int _nvars, int _nwcon,
                                                 const int *rowp,
                                                 const int *cols) {}

ParOptQuasiDefSparseMat::~ParOptQuasiDefSparseMat() {}

/*
  Compute the elements and factor the sparse matrix
*/
int ParOptQuasiDefSparseMat::factor(ParOptVec *x, ParOptVec *Dinv,
                                    ParOptVec *C) {
  return 0;
}
void ParOptQuasiDefSparseMat::apply(ParOptVec *bx, ParOptVec *yx,
                                    ParOptVec *yw) {}
void ParOptQuasiDefSparseMat::apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx,
                                    ParOptVec *yw) {}