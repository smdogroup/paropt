#include "ParOptSparseCholesky.h"

#include "ParOptAMD.h"
#include "ParOptBlasLapack.h"
#include "ParOptSparseUtils.h"

// Include METIS
extern "C" {
#include "metis.h"
}

ParOptSparseCholesky::ParOptSparseCholesky(int _size, const int *Acolp,
                                           const int *Arows,
                                           ParOptOrderingType order,
                                           const int *_perm) {
  // Set the size of the sparse matrix
  size = _size;

  perm = NULL;
  iperm = NULL;
  temp = NULL;

  if (order == PAROPT_AMD_ORDER) {
    int *copy_Acolp = new int[size + 1];
    for (int i = 0; i < size + 1; i++) {
      copy_Acolp[i] = Acolp[i];
    }
    int nnz = Acolp[size];
    int *copy_Arows = new int[nnz];
    for (int i = 0; i < nnz; i++) {
      copy_Arows[i] = Arows[i];
    }

    // Set up the matrix for reordering - remove the diagonal entry
    ParOptSortAndRemoveDuplicates(size, copy_Acolp, copy_Arows);

    perm = new int[size];
    iperm = new int[size];

    // Compute the permutation using approximate AMD implemented here...
    int use_exact_degree = 0;
    ParOptAMD(size, copy_Acolp, copy_Arows, perm, use_exact_degree);

    for (int i = 0; i < size; i++) {
      iperm[perm[i]] = i;
    }

    delete[] copy_Acolp;
    delete[] copy_Arows;
  } else if (order == PAROPT_ND_ORDER) {
    int *copy_Acolp = new int[size + 1];
    for (int i = 0; i < size + 1; i++) {
      copy_Acolp[i] = Acolp[i];
    }
    int nnz = Acolp[size];
    int *copy_Arows = new int[nnz];
    for (int i = 0; i < nnz; i++) {
      copy_Arows[i] = Arows[i];
    }

    // Set up the matrix for reordering - remove the diagonal entry
    int remove_diagonal = 1;
    ParOptSortAndRemoveDuplicates(size, copy_Acolp, copy_Arows,
                                  remove_diagonal);

    // Compute the permutation using approximate AMD implemented here...
    perm = new int[size];
    iperm = new int[size];

    // Set the default options in METIS
    int options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    // Use 0-based numbering
    options[METIS_OPTION_NUMBERING] = 0;

    int n = size;
    METIS_NodeND(&n, copy_Acolp, copy_Arows, NULL, options, perm, iperm);

    delete[] copy_Acolp;
    delete[] copy_Arows;
  } else {  // order == PAROPT_NATURAL_ORDER
    // Store the re-ordering
    if (_perm) {
      perm = new int[size];
      for (int i = 0; i < size; i++) {
        perm[i] = _perm[i];
      }
      iperm = new int[size];
      for (int i = 0; i < size; i++) {
        iperm[perm[i]] = i;
      }
    }
  }

  if (perm) {
    temp = new ParOptScalar[size];
  }

  // Perform a symbolic analysis to determine the size of the factorization
  int *parent = new int[size];  // Space for the etree
  int *Lnz = new int[size];     // Nonzeros below the diagonal
  buildForest(Acolp, Arows, parent, Lnz);

  // Find the supernodes in the matrix
  var_to_snode = new int[size];
  num_snodes = initSupernodes(parent, Lnz, var_to_snode);

  // Set the remainder of the data based on the var to snode data
  snode_size = new int[num_snodes];
  snode_to_first_var = new int[num_snodes];
  for (int i = 0, j = 0; i < num_snodes; i++) {
    int jstart = j;
    while (j < size && var_to_snode[j] == i) {
      j++;
    }
    snode_size[i] = j - jstart;
    snode_to_first_var[i] = jstart;
  }

  // Set the data for the entries in the matrix
  colp = new int[num_snodes + 1];
  data_ptr = new int[num_snodes + 1];

  // Compute the non-zeros in the list
  colp[0] = 0;
  data_ptr[0] = 0;
  for (int i = 0; i < num_snodes; i++) {
    int ssize = snode_size[i];
    int var = snode_to_first_var[i];
    int n = 1 + Lnz[var] - ssize;

    data_ptr[i + 1] = data_ptr[i] + (ssize * (ssize + 1) / 2) + n * ssize;
    colp[i + 1] = colp[i] + n;
  }

  rows = new int[colp[num_snodes]];

  // Now we can build the non-zero pattern
  buildNonzeroPattern(Acolp, Arows, parent, Lnz);
  delete[] parent;
  delete[] Lnz;

  data = new ParOptScalar[data_ptr[num_snodes]];

  // Compute the work size
  work_size = 0;
  for (int i = 0; i < num_snodes; i++) {
    int col_size = snode_size[i] * (colp[i + 1] - colp[i]);
    if (col_size > work_size) {
      work_size = col_size;
    }
    int diag_size = snode_size[i] * snode_size[i];
    if (diag_size > work_size) {
      work_size = diag_size;
    }
  }
}

ParOptSparseCholesky::~ParOptSparseCholesky() {
  delete[] snode_size;
  delete[] var_to_snode;
  delete[] snode_to_first_var;
  delete[] data_ptr;
  delete[] data;

  if (perm) {
    delete[] perm;
    delete[] iperm;
    delete[] temp;
  }
}

/**
  Get information about the sparse Cholesky factorization

  @param size The dimension of the matrix
  @param num_snodes The number of supernodes
  @param nnzA The number of non-zeros in the original matrix
  @param nnzL The number of non-zeros in the factorized matrix
*/
void ParOptSparseCholesky::getInfo(int *_size, int *_num_snodes, int *_nnzL) {
  if (_size) {
    *_size = size;
  }
  if (_num_snodes) {
    *_num_snodes = num_snodes;
  }
  if (_nnzL) {
    *_nnzL = data_ptr[num_snodes];
  }
}

/**
  Set the values into the matrix.

  This assumes that the inputs are in the original ordering - not the permuted
  ordering. Contributions are added, so duplicates get summed.

  @param n The number of columns in the input
  @param Acolp Pointer into the columns
  @param Arows Row indices of the nonzero entries
  @param Avals The numerical values
*/
void ParOptSparseCholesky::setValues(int n, const int Acolp[],
                                     const int Arows[],
                                     const ParOptScalar Avals[]) {
  for (int i = 0; i < data_ptr[num_snodes]; i++) {
    data[i] = 0.0;
  }

  for (int j = 0; j < n; j++) {
    int ipj = j;
    if (iperm) {
      ipj = iperm[j];
    }
    int sj = var_to_snode[ipj];
    int jfirst = snode_to_first_var[sj];
    int jsize = snode_size[sj];

    int ip_end = Acolp[j + 1];
    for (int ip = Acolp[j]; ip < ip_end; ip++) {
      int ipi = Arows[ip];
      if (iperm) {
        ipi = iperm[ipi];
      }

      if (ipi >= ipj) {
        // Check if this is a diagonal element
        if (ipi < jfirst + jsize) {
          int jj = ipj - jfirst;
          int ii = ipi - jfirst;

          ParOptScalar *D = get_diag_pointer(sj);
          D[get_diag_index(ii, jj)] += Avals[ip];
        } else {
          int jj = ipj - jfirst;

          // Look for the row
          for (int kp = colp[sj]; kp < colp[sj + 1]; kp++) {
            if (rows[kp] == ipi) {
              ParOptScalar *L = get_factor_pointer(sj, jsize, kp);
              L[jj] += Avals[ip];

              break;
            }
          }
        }
      }
    }
  }
}

/**
  Build the elimination tree/forest and compute the number of non-zeros in each
  column.

  @param Acolp The pointer into each column
  @param Arows The row indices for each matrix entry
  @param parent The elimination tree/forest
  @param Lnz The number of non-zeros in each column
*/
void ParOptSparseCholesky::buildForest(const int Acolp[], const int Arows[],
                                       int parent[], int Lnz[]) {
  int *flag = new int[size];

  for (int k = 0; k < size; k++) {
    parent[k] = -1;
    flag[k] = k;
    Lnz[k] = 0;

    int pk = k;
    if (perm) {
      pk = perm[k];
    }

    // Loop over the k-th column of the original matrix
    int ip_end = Acolp[pk + 1];
    for (int ip = Acolp[pk]; ip < ip_end; ip++) {
      int i = Arows[ip];
      if (iperm) {
        i = iperm[i];
      }

      if (i < k) {
        // Scan up the etree
        for (; flag[i] != k; i = parent[i]) {
          if (parent[i] == -1) {
            parent[i] = k;
          }

          // L[k, i] is non-zero
          Lnz[i]++;
          flag[i] = k;
        }
      }
    }
  }

  delete[] flag;
}

/**
  Initialize the supernodes in the matrix

  The supernodes share the same column non-zero pattern

  @param parent The elimination tree data
  @param Lnz The number of non-zeros per variable
  @param vtn The array of supernodes for each variable
*/
int ParOptSparseCholesky::initSupernodes(const int parent[], const int Lnz[],
                                         int vtn[]) {
  int snode = 0;
  for (int i = 0; i < size;) {
    vtn[i] = snode;
    i++;

    while (i < size && (parent[i - 1] == i) && Lnz[i] == Lnz[i - 1] - 1) {
      vtn[i] = snode;
      i++;
    }
    snode++;
  }

  return snode;
}

/**
  Build the non-zero pattern for the supernodes in the matrix

  This follows a similar logic to buildForest(), but uses the parent data.
  This must be called after the supernodes are constructed.

  @param Acolp The pointer into each column
  @param Arows The row indices for each matrix entry
  @param parent The elimination tree/forest
  @param Lnz The number of non-zeros in each column
*/
void ParOptSparseCholesky::buildNonzeroPattern(const int Acolp[],
                                               const int Arows[],
                                               const int parent[], int Lnz[]) {
  int *flag = new int[size];

  for (int k = 0; k < size; k++) {
    flag[k] = k;
    Lnz[k] = 0;

    int pk = k;
    if (perm) {
      pk = perm[k];
    }

    // Loop over the k-th column
    int ip_end = Acolp[pk + 1];
    for (int ip = Acolp[pk]; ip < ip_end; ip++) {
      int i = Arows[ip];
      if (iperm) {
        i = iperm[i];
      }

      if (i < k) {
        // Scan up the etree
        for (; flag[i] != k; i = parent[i]) {
          int si = var_to_snode[i];
          int ivar = snode_to_first_var[si];
          if (i == ivar) {
            int isize = snode_size[si];
            if (k >= ivar + isize) {
              rows[colp[si] + Lnz[i]] = k;
              Lnz[i]++;
            }
          }

          flag[i] = k;
        }
      }
    }
  }

  delete[] flag;
}

/**
  Add the diagonal update

  Update the entries of the diagonal matrix

  D <- D - L * L^{T}
*/
void ParOptSparseCholesky::updateDiag(const int lsize, const int nlrows,
                                      const int lfirst_var, const int *lrows,
                                      ParOptScalar *L, const int diag_size,
                                      ParOptScalar *diag, ParOptScalar *work) {
  // Compute L * L^{T}
  int n = nlrows;
  int k = lsize;
  ParOptScalar alpha = 1.0, beta = 0.0;
  BLASsyrk("L", "T", &n, &k, &alpha, L, &k, &beta, work, &n);

  // Add D <- D - L * L^{T}
  for (int jj = 0; jj < nlrows; jj++) {
    int j = lrows[jj] - lfirst_var;
    for (int ii = 0; ii < jj + 1; ii++) {
      int i = lrows[ii] - lfirst_var;
      diag[i + j * (j + 1) / 2] -= work[jj + nlrows * ii];
    }
  }
}

/**
  Perform a column update into the work column.

  The column T has dimensions of the number of non-zero rows in L32 by the
  number of non-zero rows in L21.

  Given the current factorization at step *, where the matrix is of the form

  [ L11  0 ]
  [ L21  * ]
  [ L31  T ]

  Compute the result

  T = L31 * L21^{T}

  @param lwidth The number of columns in L21 and L32
  @param n21rows The number of non-zero rows in L21
  @param L21 The numerical values of L21 in row-major order
  @param n31rows The number of non-zero rows in L32
  @param L31 The numerical values of L32 in row-major order
  @param T The temporary vector
*/
void ParOptSparseCholesky::updateWorkColumn(int lwidth, int n21rows,
                                            ParOptScalar *L21, int n31rows,
                                            ParOptScalar *L31,
                                            ParOptScalar *T) {
  // These matrices are stored in row-major order. To compute the result we
  // use LAPACK with the computation: T^{T} = L21 * L31^{T}
  // dimension of T^{T} is n21rows X n32rows
  // dimension of L21 is n21rows X lwidth
  // dimension of L31^{T} is lwidth X n31rows
  ParOptScalar alpha = 1.0, beta = 0.0;
  BLASgemm("T", "N", &n21rows, &n31rows, &lwidth, &alpha, L21, &lwidth, L31,
           &lwidth, &beta, T, &n21rows);
}

/**
  Subtract a sparse column from another sparse column

  L32 = L32 - T

  The row indices in the input brows must be a subset of the rows in arows.
  Both the input arows and brows must be sorted. All indices in arows must
  exist in brows.

  @param lwidth The width of the L32 column
  @param nrows The number of the rows to update
  @param lrows The indices of the L32 column
  @param A The A values of the column
  @param brows The indices of the B column
  @param B The B values of the column
*/
void ParOptSparseCholesky::updateColumn(const int lwidth, const int nlcols,
                                        const int lfirst_var, const int *lrows,
                                        int nrows, const int *arows,
                                        const ParOptScalar *A, const int *brows,
                                        ParOptScalar *B) {
  for (int i = 0, bi = 0; i < nrows; i++) {
    while (brows[bi] < arows[i]) {
      bi++;
      B += lwidth;
    }

    for (int jj = 0; jj < nlcols; jj++) {
      int j = lrows[jj] - lfirst_var;
      B[j] -= A[jj];
    }
    A += nlcols;
  }
}

/*
  Perform the dense Cholesky factorization of the diagonal components
*/
int ParOptSparseCholesky::factorDiag(const int diag_size, ParOptScalar *D) {
  int n = diag_size, info;
  LAPACKdpptrf("U", &n, D, &info);
  return info;
}

/*
  Solve L * y = x and output x = y
*/
void ParOptSparseCholesky::solveDiag(int diag_size, ParOptScalar *L, int nrhs,
                                     ParOptScalar *x) {
  int incr = 1;
  for (int k = 0; k < nrhs; k++) {
    BLAStpsv("U", "T", "N", &diag_size, L, x, &incr);
    x += diag_size;
  }
}

/*
  Solve L^{T} * y = x and output x = y
*/
void ParOptSparseCholesky::solveDiagTranspose(int diag_size, ParOptScalar *L,
                                              int nrhs, ParOptScalar *x) {
  int incr = 1;
  for (int k = 0; k < nrhs; k++) {
    BLAStpsv("U", "N", "N", &diag_size, L, x, &incr);
    x += diag_size;
  }
}

/*
  Compute the Cholesky decomposition using a left-looking supernode approach

  At the current iteration L11, L21 and L31 are computed. We want to compute
  the update to L22 and L32. This leads to the relationship

  [ L11   0    0  ][ L11^T  L21^T  L31^T ] = [ x         ]
  [ L21  L22   0  ][  0     L22^T  L32^T ] = [ x  A22    ]
  [ L31  L32  L33 ][  0       0    L33^T ] = [ x  A32  x ]

  The diagonal block gives A22 = L22 * L22^{T} + L21 * L21^{T}, which leads
  to the factorization

  L22 * L22^{T} = A22 - L21 * L21^{T}

  The A32 block gives A32 = L31 * L21^{T} + L32 * L22^{T}

  L32 = (A32 - L32 * L21^{T}) * L22^{-T}

  This leads to the following steps in the algorithm:

  (1) Compute the diagonal update: A22 <- A22 - L21 * L21^{T}

  (2) Compute the column update: A32 <- A32 - L32 * L21. This is a two
  step process whereby we first accumulate the numerical results in a
  temporary work vector then apply them to the A32/L32 data.

  After all columns in the row have completed

  (3) Factor the diagonal to obtain L22

  (4) Apply the factor to the column L32 <- (A32 - L32 * L21) * L22^{-T}
*/
int ParOptSparseCholesky::factor() {
  int *list = new int[num_snodes];  // List pointer
  int *first = new int[num_snodes];

  // Temporary numeric workspace for stuff
  ParOptScalar *work_temp = new ParOptScalar[work_size];

  // Initialize the linked list and copy the diagonal values
  for (int j = 0; j < num_snodes; j++) {
    list[j] = -1;
  }

  for (int j = 0; j < num_snodes; j++) {
    // Keep track of the size of the supernode on the diagonal
    int diag_size = snode_size[j];
    ParOptScalar *diag = get_diag_pointer(j);

    // First variable associated with this supernode
    int jfirst_var = snode_to_first_var[j];

    // Set the pointer to the current column indices
    const int *jrows = &rows[colp[j]];
    ParOptScalar *jptr = get_factor_pointer(j, diag_size);

    // Go through the linked list of supernodes to find the super node columns
    // k with non-zero entries in row j
    int k = list[j];
    while (k != -1) {
      // Store the next supernode that we will visit
      int next_k = list[k];

      // Width of this supernode
      int ksize = snode_size[k];

      // The array "first" is indexed by supernode and points into the row
      // indices such that rows[first[list[j]]] = j
      int ip_start = first[k];
      int ip_end = colp[k + 1];

      // Find the extent of the variables associated with this supernode
      int ip_next = ip_start + 1;
      while (ip_next < ip_end && var_to_snode[rows[ip_next]] == j) {
        ip_next++;
      }

      // Set the value for the next column in the list
      if (ip_next < ip_end) {
        int snode = var_to_snode[rows[ip_next]];

        // Update the first/list data structure
        first[k] = ip_next;
        list[k] = list[snode];
        list[snode] = k;
      }

      // The number of rows in L21
      int nkrows = ip_next - ip_start;
      const int *krows = &rows[ip_start];
      ParOptScalar *kvals = get_factor_pointer(k, ksize, ip_start);

      // Perform the update to the diagonal by computing
      // diag <- diag - L21 * L21^{T}
      updateDiag(ksize, nkrows, jfirst_var, krows, kvals, diag_size, diag,
                 work_temp);

      // Perform the update for the column by computing
      // work_temp = L31 * L21^{T}
      int iremain = ip_end - ip_next;
      updateWorkColumn(ksize, nkrows, get_factor_pointer(k, ksize, ip_start),
                       iremain, get_factor_pointer(k, ksize, ip_next),
                       work_temp);

      // Add the temporary column to the remainder
      // updateColumn(nkrows, iremain, &rows[ip_next], work_temp, jrows, jptr);
      updateColumn(diag_size, nkrows, jfirst_var, krows, iremain,
                   &rows[ip_next], work_temp, jrows, jptr);

      // Move to the next k non-zero
      k = next_k;
    }

    // Facgtor the diagonal and copy the entries back to the diagonal
    factorDiag(diag_size, diag);

    // Compute (A32 - L32 * L21 ) * L21^{-T}
    int nrhs = colp[j + 1] - colp[j];
    solveDiag(diag_size, diag, nrhs, jptr);

    // Update the list for this column
    if (colp[j] < colp[j + 1]) {
      int snode = var_to_snode[rows[colp[j]]];
      first[j] = colp[j];
      list[j] = list[snode];
      list[snode] = j;
    }
  }

  delete[] list;
  delete[] first;
  delete[] work_temp;

  return 0;
}

/*
  Solve the system of equations with the Cholesky factorization
*/
void ParOptSparseCholesky::solve(ParOptScalar *x) {
  ParOptScalar *xt = x;

  // Compute temp = P * x
  if (perm) {
    for (int i = 0; i < size; i++) {
      temp[i] = x[perm[i]];
    }
    xt = temp;
  }

  // Solve L * x = x
  for (int j = 0; j < num_snodes; j++) {
    const int jsize = snode_size[j];
    ParOptScalar *D = get_diag_pointer(j);
    ParOptScalar *y = &xt[snode_to_first_var[j]];
    solveDiag(jsize, D, 1, y);

    // Apply the update from the whole column
    const ParOptScalar *L = get_factor_pointer(j, jsize);

    int ip_end = colp[j + 1];
    for (int ip = colp[j]; ip < ip_end; ip++) {
      ParOptScalar val = 0.0;
      for (int ii = 0; ii < jsize; ii++) {
        val += L[ii] * y[ii];
      }
      xt[rows[ip]] -= val;
      L += jsize;
    }
  }

  // Solve L^{T} * x = x
  for (int j = num_snodes - 1; j >= 0; j--) {
    const int jsize = snode_size[j];
    ParOptScalar *y = &xt[snode_to_first_var[j]];
    ParOptScalar *L = get_factor_pointer(j, jsize);

    int ip_end = colp[j + 1];
    for (int ip = colp[j]; ip < ip_end; ip++) {
      for (int ii = 0; ii < jsize; ii++) {
        y[ii] -= L[ii] * xt[rows[ip]];
      }
      L += jsize;
    }

    ParOptScalar *D = get_diag_pointer(j);
    solveDiagTranspose(jsize, D, 1, y);
  }

  // Compute x = P^{T} * temp
  if (perm) {
    for (int i = 0; i < size; i++) {
      x[perm[i]] = temp[i];
    }
  }
}