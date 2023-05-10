#ifndef PAR_OPT_SPARSE_MAT_H
#define PAR_OPT_SPARSE_MAT_H

#include "ParOptBlasLapack.h"
#include "ParOptProblem.h"
#include "ParOptVec.h"

/*
  Interface for the quasi-definite matrix

  [ D   Aw^{T} ]
  [ Aw   -C    ]

*/
class ParOptQuasiDefMat : public ParOptBase {
 public:
  ParOptQuasiDefMat(ParOptProblem *prob0) {
    prob = prob0;
    prob->incref();
    prob->getProblemSizes(&nvars, NULL, NULL, &nwcon, &nwblock);

    x = NULL;
    Dinv = NULL;
    C = NULL;

    // Allocate space for the block-diagonal matrix
    Cw = new ParOptScalar[nwcon * (nwblock + 1) / 2];
  }
  ~ParOptQuasiDefMat() {
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

  int factor(ParOptVec *x0, ParOptVec *Dinv0, ParOptVec *C0) {
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
    Solve the system of equations

    [ D   Aw^{T} ][  yx ] = [ bx ]
    [ Aw    - C  ][ -yw ] = [ 0  ]

    Here bx is unmodified. Note the negative sign on the yw variables.
  */
  void apply(ParOptVec *bx, ParOptVec *yx, ParOptVec *yw) {
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
    Solve the equations

    [ D   Aw^{T} ][  yx ] = [ bx ]
    [ Aw    - C  ][ -yw ] = [ bw ]

    Here bx and bw remain unmodified
  */
  void apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx, ParOptVec *yw) {
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

 private:
  /*
    Apply the factored Cw-matrix that is stored as a series of block-symmetric
    matrices.
  */
  int applyFactor(ParOptVec *vec) {
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

  // Problem data
  ParOptProblem *prob;

  // Vectors that point to the input data
  ParOptVec *x, *Dinv, *C;

  // The data for the block-diagonal matrix
  int nvars;         // The number of variables
  int nwcon;         // The number of sparse constraints
  int nwblock;       // The nuber of constraints per block
  ParOptScalar *Cw;  // Block diagonal matrix
};

#endif  //  PAR_OPT_SPARSE_MAT_H