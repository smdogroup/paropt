#include "ParOptSparseMat.h"

#include <algorithm>

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
ParOptQuasiDefSparseMat::ParOptQuasiDefSparseMat(ParOptSparseProblem *problem) {
  prob = problem;
  prob->incref();

  prob->getProblemSizes(&nvars, NULL, &nwcon);

  // Set the size of the matrix
  size = nvars + nwcon;

  // Get the sparse Jacobian information in CSR format
  const int *rowp = NULL, *cols = NULL;
  prob->getSparseJacobianData(&rowp, &cols, NULL);

  // Compute the number of non-zeros in the K matrix
  Knnz = size + rowp[nwcon];
  Kcolp = new int[size + 1];
  Krows = new int[Knnz];
  computeCSCSymbolic(nvars, nwcon, rowp, cols, Kcolp, Krows);

  // Set the permutation and inverse permutation
  perm = iperm = NULL;

  // Set the permutation
  if (perm) {
    for (int i = 0; i < size; i++) {
      iperm[perm[i]] = i;
    }
  }

  // Allocate temporary variables
  int *temp = new int[4 * size];
  int *list = &temp[0];
  int *parent = &temp[size];
  int *first = &temp[2 * size];
  int *flag = &temp[3 * size];

  // Perform the symbolic Choleksy factorization - first figure out the size of
  // the factorization
  lcolp = new int[size + 1];
  lnnz = factorSymbolic(size, Kcolp, Krows, list, parent, first, flag, lcolp,
                        NULL);

  // Compute the non-zero pattern of the factorization
  lrows = new int[lnnz];
  factorSymbolic(size, Kcolp, Krows, list, parent, first, flag, lcolp, lrows);
  delete[] temp;

  // Allocate arrays for storing the numerical data
  Kvals = new ParOptScalar[Knnz];
  Kdiag = new ParOptScalar[size];
  ldiag = new ParOptScalar[size];
  lvals = new ParOptScalar[lnnz];
  rhs = new ParOptScalar[size];
}

ParOptQuasiDefSparseMat::~ParOptQuasiDefSparseMat() {
  prob->decref();

  delete[] Kcolp;
  delete[] Krows;
  delete[] Kvals;
  delete[] Kdiag;

  delete[] lcolp;
  delete[] lrows;
  delete[] ldiag;
  delete[] lvals;

  delete[] rhs;
}

/**
  Compute the lower CSC non-zero pattern for the symmetric matrix

  K = [ D   A^{T} ]
  .   [ A    - C  ]

  @param nvars0 Number of variables (size of D)
  @param nwcon0 Number of constraints (sizeof C)
  @param rowp Pointer into the rows of the constraint Jacobian
  @param cols Column indices of the constraint Jacobian entries
  @param colp Output pointer to the columns of K
  @param rows Output the row indices of the columns of K
*/
void ParOptQuasiDefSparseMat::computeCSCSymbolic(const int nvars0,
                                                 const int nwcon0,
                                                 const int *rowp,
                                                 const int *cols, int *colp,
                                                 int *rows) {
  int size0 = nvars0 + nwcon0;
  for (int j = 0; j < size0 + 1; j++) {
    colp[j] = 0;
  }

  // Count up the number of rows in each column and store in colp
  for (int i = 0; i < nwcon0; i++) {
    int jp_end = rowp[i + 1];
    for (int jp = rowp[i]; jp < jp_end; jp++) {
      int j = cols[jp];
      colp[j + 1]++;
    }
  }

  // Set the colp array to be a pointer into each row
  for (int j = 0; j < size0; j++) {
    colp[j + 1] += colp[j];
  }

  // Now, add the rows indices
  for (int i = 0; i < nwcon0; i++) {
    int jp_end = rowp[i + 1];
    for (int jp = rowp[i]; jp < jp_end; jp++) {
      int j = cols[jp];
      rows[colp[j]] = i + nvars0;
      colp[j]++;
    }
  }

  // Reset the colp array
  for (int j = size0 - 1; j >= 0; j--) {
    colp[j + 1] = colp[j];
  }
  colp[0] = 0;
}

/**
  Compute the lower CSC non-zero pattern for the symmetric matrix

  K = [ D   A^{T} ]
  .   [ A    - C  ]

  @param nvars0 Number of variables (size of D)
  @param nwcon0 Number of constraints (sizeof C)
  @param dinv Inverse of the D diagonal matrix
  @param cvals Diagonal C matrix values
  @param rowp Pointer into the rows of the constraint Jacobian
  @param cols Column indices of the constraint Jacobian entries
  @param avals Constraint Jacobian entry values
  @param kdiag Diagonal matrix entries
  @param colp Pointer to the columns of K (modified then reset internally)
  @param rows Output the row indices of the columns of K
  @param kvals Output entries of the K matrix
*/
void ParOptQuasiDefSparseMat::setCSCNumeric(
    const int nvars0, const int nwcon0, const ParOptScalar *dinv,
    const ParOptScalar *cvals, const int *rowp, const int *cols,
    const ParOptScalar *avals, ParOptScalar *kdiag, int *colp, const int *rows,
    ParOptScalar *kvals) {
  int size0 = nvars0 + nwcon0;

  // Set the diagonal elements
  for (int j = 0; j < nvars0; j++) {
    kdiag[j] = 1.0 / dinv[j];
  }
  for (int j = 0; j < nwcon0; j++) {
    kdiag[j + nvars0] = -cvals[j];
  }

  // Now, add the rows indices
  for (int i = 0; i < nwcon0; i++) {
    int jp_end = rowp[i + 1];
    for (int jp = rowp[i]; jp < jp_end; jp++) {
      int j = cols[jp];
      kvals[colp[j]] = avals[jp];
      colp[j]++;
    }
  }

  // Reset the colp array
  for (int j = size0 - 1; j >= 0; j--) {
    colp[j + 1] = colp[j];
  }
  colp[0] = 0;
}

/**
  Perform the symbolic computation for sparse Cholesky

  This makes several assumptions

  (1) The input data represent a strict lower triangular matrix
  (2) The intput row indices in each column are sorted
  (3) If Lrows is an input, the Lcolp must be an input generated from a prior
  call to the function

  @param n Size of the symmetric matrix
  @param colp Pointer into the columns of the matrix
  @param rows Row indices for the matrix
  @param list Linked list pointing to a row of the matrix
  @param first Number of non-zeros for each column output
  @param parent The elimination tree
  @param flag Temporary flags to track visited dof
  @param Lcolp Pointer into each column
  @param Lrows Sorted row indices for each column
*/
int ParOptQuasiDefSparseMat::factorSymbolic(const int n, const int *colp,
                                            const int *rows, int *list,
                                            int *first, int *parent, int *flag,
                                            int *Lcolp, int *Lrows) {
  // Initialize the linked list
  for (int j = 0; j < n; j++) {
    list[j] = -1;
  }

  if (!Lrows) {
    for (int j = 0; j < n; j++) {
      Lcolp[j] = 0;
    }
  }

  int nnz = 0;  // Total number of non-zero entries
  for (int j = 0; j < n; j++) {
    parent[j] = -1;  // Elimination tree data structure
    flag[j] = j;     // Flag to keep track of whether we've visited this dof

    // Loop over non-zeros L[k, j]
    int k = list[j];
    while (k != -1) {
      int next_k = list[k];
      int ip_start = first[k];
      int ip_end = colp[k + 1];

      if (ip_start + 1 < ip_end) {
        first[k] = ip_start + 1;
        list[k] = list[rows[ip_start + 1]];
        list[rows[ip_start + 1]] = k;
      }

      // Now the entry L[k, j] is non-zero - add up the entries from the
      // corresponding row
      int i = k;
      while (flag[i] != j) {
        if (parent[i] == -1) {
          parent[i] = j;
        }

        // Add the non-zero pattern L[j, i] is non-zero
        if (Lrows) {
          Lrows[Lcolp[i]] = j;
        }
        Lcolp[i]++;

        nnz += 1;
        flag[i] = j;
        i = parent[i];
      }

      // Move to the next k
      k = next_k;
    }

    // Initialize new members of the list
    if (colp[j] < colp[j + 1]) {
      first[j] = colp[j];
      list[j] = list[rows[colp[j]]];
      list[rows[colp[j]]] = j;
    }
  }

  if (Lrows) {
    for (int j = 0; j < n; j++) {
      std::sort(&Lrows[Lcolp[j]], &Lrows[Lcolp[j + 1]]);
    }

    // Reset the Lcolp array
    for (int j = n - 1; j >= 0; j--) {
      Lcolp[j + 1] = Lcolp[j];
    }
    Lcolp[0] = 0;
  } else {
    int count = 0;
    for (int j = 0; j < n; j++) {
      int tmp = Lcolp[j];
      Lcolp[j] = count;
      count += tmp;
    }
    Lcolp[n] = count;
  }

  return nnz;
}

/*
  Perform the LDL^T factorization
*/
int ParOptQuasiDefSparseMat::factorNumeric(
    const int n, const ParOptScalar *Adiag, const int *colp, const int *rows,
    const ParOptScalar *Avals, int *list, int *first, ParOptScalar *column,
    ParOptScalar *Ldiag, const int *Lcolp, const int *Lrows,
    ParOptScalar *Lvals) {
  // Initialize the linked list and copy the diagonal values
  for (int j = 0; j < size; j++) {
    list[j] = -1;
    Ldiag[j] = Adiag[j];
  }

  for (int j = 0; j < n; j++) {
    // Zero entries that will be set
    int lp_start = lcolp[j];
    int lp_end = lcolp[j + 1];
    for (int lp = lp_start; lp < lp_end; lp++) {
      int l = Lrows[lp];
      column[l] = 0.0;
    }

    // Copy the column into the factorization
    int kp_start = colp[j];
    int kp_end = colp[j + 1];
    for (int kp = kp_start; kp < kp_end; kp++) {
      int k = rows[kp];
      column[k] = Avals[kp];
    }

    int k = list[j];
    while (k != -1) {
      int next_k = list[k];
      int ip_start = first[k];
      int ip_end = Lcolp[k + 1];

      if (ip_start + 1 < ip_end) {
        first[k] = ip_start + 1;
        list[k] = list[Lrows[ip_start + 1]];
        list[Lrows[ip_start + 1]] = k;
      }

      ParOptScalar l0 = Lvals[ip_start] * Ldiag[k];

      for (int ip = ip_start + 1; ip < ip_end; ip++) {
        int i = Lrows[ip];
        column[i] -= l0 * Lvals[ip];
      }

      // Move to the next k non-zero
      k = next_k;
    }

    // Update the diagonal entries
    int ip_start = Lcolp[j];
    int ip_end = Lcolp[j + 1];
    for (int ip = ip_start; ip < ip_end; ip++) {
      int i = Lrows[ip];
      Lvals[ip] = column[i] / Ldiag[j];
    }

    for (int ip = ip_start; ip < ip_end; ip++) {
      int i = Lrows[ip];
      Ldiag[i] = Ldiag[i] - Ldiag[j] * Lvals[ip] * Lvals[ip];
    }

    // Update the list
    if (Lcolp[j] < Lcolp[j + 1]) {
      first[j] = Lcolp[j];
      list[j] = list[Lrows[Lcolp[j]]];
      list[Lrows[Lcolp[j]]] = j;
    }
  }

  // No failure
  return 0;
}

/*
  Compute the elements and factor the sparse matrix
*/
int ParOptQuasiDefSparseMat::factor(ParOptVec *x, ParOptVec *Dinv,
                                    ParOptVec *C) {
  ParOptScalar *dvals, *cvals;
  Dinv->getArray(&dvals);
  C->getArray(&cvals);

  // Get the sparse Jacobian
  // information in CSR format
  const int *rowp = NULL, *cols = NULL;
  const ParOptScalar *data;
  prob->getSparseJacobianData(&rowp, &cols, &data);

  // Set the matrix values
  setCSCNumeric(nvars, nwcon, dvals, cvals, rowp, cols, data, Kdiag, Kcolp,
                Krows, Kvals);

  // Factor the matrix
  int *temp = new int[2 * size];
  int *list = &temp[0];
  int *first = &temp[size];
  int fail = factorNumeric(size, Kdiag, Kcolp, Krows, Kvals, list, first, rhs,
                           ldiag, lcolp, lrows, lvals);
  delete[] temp;

  return fail;
}

/**
  Compute the solution error for the right-hand-side b = K * 1 so that x = 1

  @return The error associated with the known solution x = 1
*/
ParOptScalar ParOptQuasiDefSparseMat::computeSolutionError() {
  // Compute rhs = K * e
  for (int i = 0; i < size; i++) {
    rhs[i] = Kdiag[i];
  }
  for (int i = 0; i < size; i++) {
    for (int jp = Kcolp[i]; jp < Kcolp[i + 1]; jp++) {
      int j = Krows[jp];

      rhs[i] += Kvals[jp];
      rhs[j] += Kvals[jp];
    }
  }

  solve(rhs);

  ParOptScalar dot = 0.0;
  for (int i = 0; i < size; i++) {
    dot += (rhs[i] - 1.0) * (rhs[i] - 1.0);
  }

  return sqrt(dot);
}

void ParOptQuasiDefSparseMat::apply(ParOptVec *bx, ParOptVec *yx,
                                    ParOptVec *yw) {
  ParOptScalar *bx_array;
  bx->getArray(&bx_array);

  for (int i = 0; i < nvars; i++) {
    rhs[i] = bx_array[i];
  }
  for (int i = 0; i < nwcon; i++) {
    rhs[i + nvars] = 0.0;
  }

  solve(rhs);

  ParOptScalar *yx_array, *yw_array;
  yx->getArray(&yx_array);
  yw->getArray(&yw_array);
  for (int i = 0; i < nvars; i++) {
    yx_array[i] = rhs[i];
  }
  for (int i = 0; i < nwcon; i++) {
    yw_array[i] = -rhs[i + nvars];
  }
}

void ParOptQuasiDefSparseMat::apply(ParOptVec *bx, ParOptVec *bw, ParOptVec *yx,
                                    ParOptVec *yw) {
  ParOptScalar *bx_array, *bw_array;
  bx->getArray(&bx_array);
  bw->getArray(&bw_array);

  for (int i = 0; i < nvars; i++) {
    rhs[i] = bx_array[i];
  }
  for (int i = 0; i < nwcon; i++) {
    rhs[i + nvars] = bw_array[i];
  }

  solve(rhs);

  ParOptScalar *yx_array, *yw_array;
  yx->getArray(&yx_array);
  yw->getArray(&yw_array);
  for (int i = 0; i < nvars; i++) {
    yx_array[i] = rhs[i];
  }
  for (int i = 0; i < nwcon; i++) {
    yw_array[i] = -rhs[i + nvars];
  }
}

void ParOptQuasiDefSparseMat::solve(ParOptScalar *x) {
  for (int j = 0; j < size; j++) {
    int ip_end = lcolp[j + 1];
    for (int ip = lcolp[j]; ip < ip_end; ip++) {
      int i = lrows[ip];
      x[i] -= lvals[ip] * x[j];
    }
  }

  for (int i = 0; i < size; i++) {
    x[i] /= ldiag[i];
  }

  for (int j = size - 1; j >= 0; j--) {
    int ip_end = lcolp[j + 1];
    for (int ip = lcolp[j]; ip < ip_end; ip++) {
      int i = lrows[ip];
      x[j] -= lvals[ip] * x[i];
    }
  }
}
