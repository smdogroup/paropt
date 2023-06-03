#include "ParOptSparseUtils.h"

#include <algorithm>

// Compute y = alpha A * x + beta * y
void ParOptCSRMatVec(double alpha, int nrows, const int *rowp, const int *cols,
                     const ParOptScalar *Avals, const ParOptScalar *x,
                     double beta, ParOptScalar *y) {
  if (alpha == 1.0 && beta == 0.0) {
    for (int i = 0; i < nrows; i++) {
      int jp_end = rowp[i + 1];
      ParOptScalar value = 0.0;
      for (int jp = rowp[i]; jp < jp_end; jp++) {
        int j = cols[jp];
        value += Avals[jp] * x[j];
      }
      y[i] = value;
    }
  } else if (alpha == -1.0 && beta == 0.0) {
    for (int i = 0; i < nrows; i++) {
      int jp_end = rowp[i + 1];
      ParOptScalar value = 0.0;
      for (int jp = rowp[i]; jp < jp_end; jp++) {
        int j = cols[jp];
        value += Avals[jp] * x[j];
      }
      y[i] = -value;
    }
  } else {
    if (beta == 0.0) {
      for (int i = 0; i < nrows; i++) {
        int jp_end = rowp[i + 1];
        ParOptScalar value = 0.0;
        for (int jp = rowp[i]; jp < jp_end; jp++) {
          int j = cols[jp];
          value += Avals[jp] * x[j];
        }
        y[i] = alpha * value;
      }
    } else {
      for (int i = 0; i < nrows; i++) {
        int jp_end = rowp[i + 1];
        ParOptScalar value = 0.0;
        for (int jp = rowp[i]; jp < jp_end; jp++) {
          int j = cols[jp];
          value += Avals[jp] * x[j];
        }
        y[i] = beta * y[i] + alpha * value;
      }
    }
  }
}

// Compute y = alpha A * x + beta * y
void ParOptCSCMatVec(double alpha, int nrows, int ncols, const int *colp,
                     const int *rows, const ParOptScalar *Avals,
                     const ParOptScalar *x, double beta, ParOptScalar *y) {
  if (beta == 0.0) {
    for (int i = 0; i < nrows; i++) {
      y[i] = 0.0;
    }
  } else {
    for (int i = 0; i < nrows; i++) {
      y[i] = beta * y[i];
    }
  }

  if (alpha == 1.0) {
    for (int i = 0; i < ncols; i++) {
      int jp_end = colp[i + 1];
      ParOptScalar xi = x[i];
      for (int jp = colp[i]; jp < jp_end; jp++) {
        int j = rows[jp];
        y[j] += Avals[jp] * xi;
      }
    }
  } else if (alpha == -1.0) {
    for (int i = 0; i < ncols; i++) {
      int jp_end = colp[i + 1];
      ParOptScalar xi = x[i];
      for (int jp = colp[i]; jp < jp_end; jp++) {
        int j = rows[jp];
        y[j] -= Avals[jp] * xi;
      }
    }
  } else {
    for (int i = 0; i < ncols; i++) {
      int jp_end = colp[i + 1];
      ParOptScalar xi = alpha * x[i];
      for (int jp = colp[i]; jp < jp_end; jp++) {
        int j = rows[jp];
        y[j] += Avals[jp] * xi;
      }
    }
  }
}

void ParOptSparseTranspose(int nrows, int ncols, const int *rowp,
                           const int *cols, const ParOptScalar *Avals,
                           int *colp, int *rows, ParOptScalar *ATvals) {
  for (int j = 0; j < ncols + 1; j++) {
    colp[j] = 0;
  }

  for (int i = 0; i < nrows; i++) {
    int jp_end = rowp[i + 1];
    for (int jp = rowp[i]; jp < jp_end; jp++) {
      int j = cols[jp];
      colp[j + 1]++;
    }
  }

  // Set the colp array to be a pointer into each row
  for (int j = 0; j < ncols; j++) {
    colp[j + 1] += colp[j];
  }

  // Now, add the rows indices
  for (int i = 0; i < nrows; i++) {
    int jp_end = rowp[i + 1];
    for (int jp = rowp[i]; jp < jp_end; jp++) {
      int j = cols[jp];
      rows[colp[j]] = i;
      if (Avals) {
        ATvals[colp[j]] = Avals[jp];
      }
      colp[j]++;
    }
  }

  // Reset the colp array
  for (int j = ncols - 1; j >= 0; j--) {
    colp[j + 1] = colp[j];
  }
  colp[0] = 0;
}

// Compute the number of entries in the matrix product A * A^{T}
int ParOptMatMatTransSymbolic(int nrows, int ncols, const int *rowp,
                              const int *cols, const int *colp, const int *rows,
                              int *Bcolp, int *flag) {
  for (int i = 0; i < nrows; i++) {
    Bcolp[i] = 0;
    flag[i] = -1;
  }

  // P_{*j} = A_{*k} * A_{jk}
  for (int j = 0; j < nrows; j++) {
    int nz = 0;

    // Loop over the non-zero columns
    int kp_end = rowp[j + 1];
    for (int kp = rowp[j]; kp < kp_end; kp++) {
      int k = cols[kp];

      // Add the non-zero pattern from column k
      int ip_end = colp[k + 1];
      for (int ip = colp[k]; ip < ip_end; ip++) {
        int i = rows[ip];

        if (flag[i] != j) {
          flag[i] = j;
          nz++;
        }
      }
    }

    Bcolp[j] = nz;
  }

  int nnz = 0;
  for (int j = 0; j < nrows; j++) {
    int tmp = Bcolp[j];
    Bcolp[j] = nnz;
    nnz += tmp;
  }
  Bcolp[nrows] = nnz;

  return nnz;
}

// Compute the number of entries in the matrix product A * A^{T}
void ParOptMatMatTransNumeric(int nrows, int ncols, const int *rowp,
                              const int *cols, const ParOptScalar *Avals,
                              const int *colp, const int *rows,
                              const ParOptScalar *ATvals, const int *Bcolp,
                              int *Brows, ParOptScalar *Bvals, int *flag,
                              ParOptScalar *tmp) {
  for (int i = 0; i < nrows; i++) {
    flag[i] = -1;
  }

  // P_{*j} = A_{*k} * A_{jk}
  for (int j = 0; j < nrows; j++) {
    int nz = 0;

    // Loop over the non-zero columns
    int kp_end = rowp[j + 1];
    for (int kp = rowp[j]; kp < kp_end; kp++) {
      ParOptScalar Ajk = Avals[kp];
      int k = cols[kp];

      // Add the non-zero pattern from column k
      int ip_end = colp[k + 1];
      for (int ip = colp[k]; ip < ip_end; ip++) {
        int i = rows[ip];

        if (flag[i] != j) {
          flag[i] = j;
          tmp[i] = ATvals[ip] * Ajk;
          Brows[Bcolp[j] + nz] = i;
          nz++;
        } else {
          tmp[i] += ATvals[ip] * Ajk;
        }
      }
    }

    // Copy the values from the temporary column
    for (int k = 0; k < nz; k++) {
      Bvals[Bcolp[j] + k] = tmp[Brows[Bcolp[j] + k]];
    }
  }
}

// Compute the matrix C + A * D * A^{T}
void ParOptMatMatTransNumeric(int nrows, int ncols, const ParOptScalar *cvals,
                              const int *rowp, const int *cols,
                              const ParOptScalar *Avals,
                              const ParOptScalar *dvals, const int *colp,
                              const int *rows, const ParOptScalar *ATvals,
                              const int *Bcolp, int *Brows, ParOptScalar *Bvals,
                              int *flag, ParOptScalar *tmp) {
  for (int i = 0; i < nrows; i++) {
    flag[i] = -1;
  }

  // P_{*j} = A_{*k} * A_{jk}
  for (int j = 0; j < nrows; j++) {
    int nz = 0;

    // Loop over the non-zero columns
    int kp_end = rowp[j + 1];
    for (int kp = rowp[j]; kp < kp_end; kp++) {
      int k = cols[kp];
      ParOptScalar dAjk = dvals[k] * Avals[kp];

      // Add the non-zero pattern from column k
      int ip_end = colp[k + 1];
      for (int ip = colp[k]; ip < ip_end; ip++) {
        int i = rows[ip];

        if (flag[i] != j) {
          flag[i] = j;
          if (i == j) {
            tmp[i] = cvals[i] + ATvals[ip] * dAjk;
          } else {
            tmp[i] = ATvals[ip] * dAjk;
          }
          Brows[Bcolp[j] + nz] = i;
          nz++;
        } else {
          tmp[i] += ATvals[ip] * dAjk;
        }
      }
    }

    // Copy the values from the temporary column
    for (int k = 0; k < nz; k++) {
      Bvals[Bcolp[j] + k] = tmp[Brows[Bcolp[j] + k]];
    }
  }
}

/*
  Sort an array of length len, then remove duplicate entries and
  entries with values -1.
*/
int ParOptRemoveDuplicates(int *array, int len, int exclude) {
  std::sort(array, array + len);

  // Remove any negative numbers
  int i = 0;  // location to take entries from
  int j = 0;  // location to place entries

  while (i < len && array[i] < 0) i++;

  if (exclude >= 0) {
    for (; i < len; i++, j++) {
      while ((i < len - 1) && (array[i] == array[i + 1])) i++;

      if (array[i] == exclude) {
        j--;
      } else if (i != j) {
        array[j] = array[i];
      }
    }
  } else {
    for (; i < len; i++, j++) {
      while ((i < len - 1) && (array[i] == array[i + 1])) i++;

      if (i != j) {
        array[j] = array[i];
      }
    }
  }

  return j;  // The new length of the array
}

/*
  Sort the CSR data and remove duplicates
*/
void ParOptSortAndRemoveDuplicates(int nvars, int *rowp, int *cols,
                                   int remove_diagonal) {
  int begin = rowp[0];
  for (int i = 0; i < nvars; i++) {
    int len = rowp[i + 1] - begin;
    int new_len = -1;
    if (remove_diagonal) {
      new_len = ParOptRemoveDuplicates(&cols[begin], len, i);
    } else {
      new_len = ParOptRemoveDuplicates(&cols[begin], len);
    }

    if (begin != rowp[i]) {
      for (int k = 0; k < new_len; k++) {
        cols[rowp[i] + k] = cols[begin + k];
      }
    }

    begin = rowp[i + 1];
    rowp[i + 1] = rowp[i] + new_len;
  }
}