
#include "ParOptAMD.h"

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cstring>

#include "ParOptSparseUtils.h"

/*
  Data structure for storing updating the minimum or approximate minimum degree
*/
class ParOptMinDegreeList {
 public:
  ParOptMinDegreeList(int size, const int rowp[]) : size(size) {
    degree = new int[size];
    first = new int[size];
    next = new int[size];
    prev = new int[size];

    // Set the initial degree for each node
    min_degree = 0;
    for (int i = 0; i < size; i++) {
      degree[i] = rowp[i + 1] - rowp[i];
      if (degree[i] < min_degree) {
        min_degree = degree[i];
      }
      first[i] = -1;
      next[i] = -1;
      prev[i] = -1;
    }

    // Set the intial data structure
    for (int i = 0; i < size; i++) {
      int d = degree[i];

      // Initial list is empty
      if (first[d] == -1) {
        first[d] = i;
      } else {
        // Update the list
        prev[first[d]] = i;
        next[i] = first[d];
        first[d] = i;
      }
    }
  }
  ~ParOptMinDegreeList() {
    delete[] degree;
    delete[] first;
    delete[] next;
    delete[] prev;
  }

  int get_degree(int i) { return degree[i]; }

  // Remove the variable from the list structure
  void remove(int i) {
    int d = degree[i];

    // If this was the first variable of the given degree, increment to the next
    // element
    if (i == first[d]) {
      first[d] = next[i];
    }
    if (next[i] != -1) {
      prev[next[i]] = prev[i];
    }
    if (prev[i] != -1) {
      next[prev[i]] = next[i];
    }
    prev[i] = -1;
    next[i] = -1;
  }

  // Update the degree of the specified element to the new one
  void update_degree(int i, int d) {
    if (d < min_degree) {
      min_degree = d;
    }

    // Remove i from the old list
    remove(i);

    if (first[d] == -1) {
      first[d] = i;
    } else {
      // Update the list
      prev[first[d]] = i;
      next[i] = first[d];
      first[d] = i;
    }
    degree[i] = d;
  }

  // Retrieve the variable with the minimum degree
  int get_min_degree_var() {
    for (int d = min_degree; d < size; d++) {
      if (first[d] != -1) {
        return first[d];
      }
    }

    return -1;
  }

  int size;

  // Keep track of the minimum degree via updates
  int min_degree;

  // The degree associated with each
  int *degree;

  // For each degree, which is the first variable
  int *first;

  // For each variable what is the next variable with the same degree
  int *next;

  // For each variable what is the previous variable with the same degree
  int *prev;
};

/*
  Check the formatting of the AMD data structure to ensure that things
  are still ordered correctly.

  If there is a problem with one of the rows of the data structure,
  return row+1, otherwise, return 0;
*/
static int ParOptCheckFormat(int nvars, int *rowp, int *cols, int *elen,
                             int *alen) {
  int flag = 0;
  for (int i = 0; i < nvars; i++) {
    for (int j = rowp[i]; j < rowp[i] + elen[i] - 1; j++) {
      if (cols[j] < 0 || cols[j + 1] < 0 || cols[j + 1] <= cols[j]) {
        flag = i + 1;
        break;
      }
    }
    if (flag) {
      break;
    }

    for (int j = rowp[i] + elen[i]; j < rowp[i] + alen[i] - 1; j++) {
      if (cols[j] < 0 || cols[j + 1] < 0 || cols[j + 1] <= cols[j]) {
        flag = i + 1;
        break;
      }
    }
    if (flag) {
      break;
    }
  }

  return flag;
}

/*
  Find 'required_space' extra locations for the row entry by moving
  the entries in the array to different positions.

  Input:
  nvars, rowp, cols, alen: The current entries in the AMD data structure

  Procedure:

  First scan through the rows starting from r, and proceeding to 0,
  counting up the space that would be freed by compressing them.
  Next, copy those rows, and modifying rowp/cols such that the bound
  alen is tight - note that elen/alen values are not modified.  If
  that isn't enough space, count up the space from r, to nvars-1.
  Compress the required number of rows to free up the required amount
  of space.
*/
static void ParOptAMDAddSpace(int nvars, int *rowp, int *cols, const int *alen,
                              const int r, int required_space) {
  // First, try and collect the space required from the rows preceeding r
  int new_space = 0;

  if (r > 0) {
    int start = r - 1;
    // Count up the space freed by compressing rows start through r
    for (; new_space < required_space && start >= 0; start--) {
      new_space += (rowp[start + 1] - rowp[start]) - alen[start];
    }
    start++;

    int j = rowp[start] + alen[start];  // Where new entries will be placed
    for (int i = start + 1; i <= r; i++) {
      int k = rowp[i];
      int new_rowp = j;
      int end = k + alen[i];
      for (; k < end; k++, j++) {
        cols[j] = cols[k];
      }
      rowp[i] = new_rowp;
    }
  }

  // If not enough space has been found, use the remaining columns
  if (new_space < required_space) {
    int start = r + 1;
    for (; new_space < required_space && start < nvars; start++) {
      new_space += (rowp[start + 1] - rowp[start]) - alen[start];
    }
    start--;

    // Cannot exceed the size of the matrix - print an error here
    if (start >= nvars) {
      start = nvars - 1;
      fprintf(stderr, "Error, not enough memory found\n");
    }

    // Start from the end of the entries
    int j = rowp[start + 1] - 1;  // Where new entries will be placed
    for (int i = start; i > r; i--) {
      int end = rowp[i];
      int k = end + alen[i] - 1;
      for (; k >= end; k--, j--) {
        cols[j] = cols[k];
      }
      rowp[i] = rowp[i + 1] - alen[i];  // Tight bound on the entries for i
    }
  }
}

/*
  Compare two variables to see if they are indistinguishable

  This checks to see if the variables i and j form the same adjacency
  structure. If so, then they will be used to form a supervariable.
  This function first checks if the nodes are the same, and then
  checks that the { adj[j], i } is equal to { adj[i], j }. This makes
  things a bit more interesting.

  Input:
  i, j: the nodes to be compared

  elen, alen: The length of the elements, and total number variables
  and nodes respectively

  rowp, cols: The quotient graph data structure.
*/
static int ParOptCompareVariable(const int i, const int j, const int *elen,
                                 const int *alen, const int *rowp,
                                 const int *cols) {
  // First, check if they are the same length
  if (i == j) {
    return 0;  // The same node, this should be avoided
  } else if ((alen[i] != alen[j]) ||
             (elen[i] != elen[j])) {  // The lengths must be the same
    return 0;
  } else {
    // Compare cols[rowp[i] ... rowp[i] + alen[i]-1] and
    //         cols[rowp[j] ... rowp[j] + alen[j]-1]
    int size = alen[i];
    int ip = rowp[i], iend = ip + alen[i];
    int jp = rowp[j], jend = jp + alen[j];

    for (int k = 0; k < size; k++, jp++, ip++) {
      if (cols[ip] == j) {
        ip++;
      }
      if (cols[jp] == i) {
        jp++;
      }
      if (cols[ip] != cols[jp]) {
        break;
      }
    }

    return (ip == iend && jp == jend);
  }
}

/*
  Remove the references to a variable from the data structure.

  This is used when removing an indistinguishable variable.
*/
void ParOptRemoveVariable(int var, int *elen, int *alen, int *rowp, int *cols,
                          int nvars) {
  int i = rowp[var];

  // First, visit all the elements pointed to by var
  for (; i < rowp[var] + elen[var]; i++) {
    int e = cols[i];  // The element number

    int j = rowp[e];
    int jend = (rowp[e] + alen[e]) - 1;
    for (; j < jend; j++) {
      if (cols[j] == var) {
        break;
      }
    }

    // cols[j] == var: This should always be true
    for (; j < jend; j++) {
      cols[j] = cols[j + 1];
    }
    alen[e]--;
  }

  // Remove the variable from the reference
  for (; i < rowp[var] + alen[var]; i++) {
    int v = cols[i];

    int j = rowp[v] + elen[v];
    int jend = (rowp[v] + alen[v]) - 1;
    for (; j < jend; j++) {
      if (cols[j] == var) {
        break;
      }
    }

    // cols[j] == var: This should always be true
    for (; j < jend; j++) {
      cols[j] = cols[j + 1];
    }
    alen[v]--;
  }

  elen[var] = 0;
  alen[var] = 0;
}

/*
  The following function performs an approximate minimum degree
  reordering of the input matrix.  This code follows the description
  of (Amestoy, Davis and Duff, An Approximate Minimum Degree
  Reordering Algorithm, SIAM J. Mat. Anl. 1996).

  This uses a quotient graph representation of the elimination graph.
  The quotient graph representation uses a known amount of memory -
  less than or equal to the storage of A itself. The implementation
  below uses the arrays provided and as a result, the contents of
  these arrays are destroyed during the call.

  Input data description:
  -----------------------

  nvars: The dimensions of the square input matrix

  rowp: An integer that points to the beginnning entry for each row in
  cols

  cols: An array of the entries in A. During the computation this is
  modified. The output of the matrix is meaningless. All entries are
  destroyed.

  Output:
  -------

  perm: The permutation array for the new ordering.

  Data allocated within the function:
  -----------------------------------

  Lp: A temporary array of size nvars used to collect the variables
  associated pivot

  degrees: An integer array storing the current degree estimates

  elem_degrees: An integer array used during the aprpoximation of
  the variable degrees - represents |Le \ Lp|

  state: An integer array denoting the current status of a variable.
  The states are: interface variable (cannot be selected as a
  pivot), variable (not yet eliminated), element (an eliminated
  variable) and inactive element (no longer required by the algorithm)

  elen: The number of elements in the row. The elements are stored
  first within the row in ascending order.

  alen: The number of elements + variables in the row. The variables
  are stored immediately after the elements.

  slen: The length of the supernode!

  hash: The hash table values used to compare nodes to see if they
  are indistinguishable

  As a result:
  cols[rowp[i] ... rowp[i] + elen[i]] == The elements in the row

  cols[rowp[i] + elen[i] ... rowp[i] + alen[i]] == The
  variables in each row

  On initialization elen[:] = 0, alen[i] = rowp[i+1] - rowp[i]
*/
void ParOptAMD(int nvars, int *rowp, int *cols, int *perm,
               int use_exact_degree) {
  int *alen = new int[nvars];  // Number of entries in a row
  int *elen = new int[nvars];  // Number of elements in a row
  int *slen = new int[nvars];  // The length of each supernode

  int *degree = new int[nvars];       // Degree (exact or estimate) of the node
  int *elem_degree = new int[nvars];  // The degree of the element
  int *Lp = new int[nvars];           // A set of variables
  int *state = new int[nvars];        // Are we a variable, or element?

  for (int i = 0; i < nvars; i++) {
    perm[i] = i;
    degree[i] = rowp[i + 1] - rowp[i];
    alen[i] = rowp[i + 1] - rowp[i];
    elen[i] = 0;
    state[i] = 1;  // Set everything to variables
    slen[i] = 1;
  }

  if (ParOptCheckFormat(nvars, rowp, cols, elen, alen) != 0) {
    fprintf(stderr,
            "ParOptAMD: Detected problem with initial data structure\n");
    return;
  }

  ParOptMinDegreeList deglist(nvars, rowp);

  // Perform the elimination
  int nsvars = 0;  // Keep track of the number of supervariables
  for (int i = 0; i < nvars; nsvars++) {
    // Select the pivot
    int piv = deglist.get_min_degree_var();
    deglist.remove(piv);

    perm[nsvars] = piv;
    state[piv] = 0;  // Eliminated variable

    // Determine the set of variables Lp U { piv } - the non-zeros
    // in the column of L
    int lenlp = 0;
    Lp[lenlp] = piv;
    lenlp++;

    // Add the contributions from the row piv
    for (int j = rowp[piv] + elen[piv]; j < rowp[piv] + alen[piv]; j++) {
      if (piv != cols[j]) {
        Lp[lenlp] = cols[j];
        lenlp++;
      }
    }

    // Add the non-zero pattern to Lp
    // First add the contributions from the elements
    for (int j = rowp[piv]; j < rowp[piv] + elen[piv]; j++) {
      int e = cols[j];
      for (int k = rowp[e] + elen[e]; k < rowp[e] + alen[e]; k++) {
        if (lenlp >= nvars) {
          // Remove duplicates - this has to free up enough space for Lp
          lenlp = ParOptRemoveDuplicates(Lp, lenlp);
        }
        if (cols[k] != piv) {
          Lp[lenlp] = cols[k];
          lenlp++;
        }
      }
    }

    // Sort and remove any duplicates from the list Lp
    lenlp = ParOptRemoveDuplicates(Lp, lenlp);

    // Update the non-zero pattern in cols
    for (int j = 0; j < lenlp; j++) {
      int lj = Lp[j];

      if (lj == piv) {
        continue;
      }

      // Absorb elements that are no longer required
      int nre = 0;  // Number of removed elements
      int end = rowp[piv] + elen[piv];
      for (int k = rowp[piv], ck = rowp[lj];
           ((k < end) && (ck < rowp[lj] + elen[lj])); ck++) {
        while ((k < end) && (cols[k] < cols[ck])) {
          k++;
        }
        if (k < end && cols[k] == cols[ck]) {
          cols[ck] = -1;  // Remove the element from the list
          nre++;
        }
      }

      // Remove Lp[j], Lp[k] if it exists in cols
      int nrv = 0;  // Number of removed variables
      // Remove Lp U { piv } from the columns
      for (int k = 0, ck = rowp[lj] + elen[lj];
           (k < lenlp) && (ck < rowp[lj] + alen[lj]); ck++) {
        while ((k < lenlp) && (Lp[k] < cols[ck])) {
          k++;
        }
        if (k < lenlp && cols[ck] == Lp[k]) {
          cols[ck] = -1;  // Remove it from the list
          nrv++;
        }
      }

      // Remove negative entries from the element list
      if (nre > 0 || nrv > 0) {
        int end = rowp[lj] + alen[lj];
        int k = rowp[lj];
        int nk = k;
        for (; k < end; k++, nk++) {
          while (k < end && cols[k] == -1) {
            k++;
          }
          if (k < end) {
            cols[nk] = cols[k];
          }
        }

        // Adjust the number of variables and elements within the list
        elen[lj] = elen[lj] - nre;
        alen[lj] = alen[lj] - nre - nrv;
      }
    }

    // Now, add piv to the elements in rows Lp \ {piv}
    for (int j = 0; j < lenlp; j++) {
      int lj = Lp[j];

      if (lj == piv) {
        continue;
      }

      if (alen[lj] == rowp[lj + 1] - rowp[lj]) {
        ParOptAMDAddSpace(nvars, rowp, cols, alen, lj, 1);
      }

      // Now, find where piv should go in [rowp[lj] ... rowp[lj] + elen[lj]-1]
      int k = rowp[lj];
      int end = k + elen[lj];
      while (k < end && cols[k] < piv) {
        k++;
      }

      if (cols[k] != piv) {
        int p = rowp[lj] + alen[lj] - 1;
        while (p >= k) {
          cols[p + 1] = cols[p];
          p--;
        }
        cols[k] = piv;

        alen[lj] += 1;
        elen[lj] += 1;
      }
    }

    // Remove the rows associated with the elements of piv
    for (int j = rowp[piv]; j < rowp[piv] + elen[piv]; j++) {
      int e = cols[j];
      // Remove row e
      alen[e] = 0;
      elen[e] = 0;
      state[e] = -1;  // This element has been entirely removed
    }

    // Copy Lp to the row piv
    int rsize = rowp[piv + 1] - rowp[piv];

    // Test to see if new space is requried
    if (lenlp - 1 > rsize) {
      ParOptAMDAddSpace(nvars, rowp, cols, alen, piv,
                        lenlp - 1 - (rowp[piv + 1] - rowp[piv]));
    }

    elen[piv] = 0;
    alen[piv] = lenlp - 1;
    for (int j = rowp[piv], k = 0; k < lenlp; k++, j++) {
      if (Lp[k] == piv) {
        k++;
        if (k == lenlp) {
          break;
        }
      }
      cols[j] = Lp[k];
    }

    if (use_exact_degree) {
      // Update the degrees for each variable in the row piv
      for (int j = rowp[piv]; j < rowp[piv] + alen[piv]; j++) {
        // This row should be entirely variables by definition
        if (state[cols[j]] <= 0) {
          fprintf(stderr,
                  "Error, the pivot row %d should contain only variables, not "
                  "element %d\n",
                  piv, cols[j]);
        }

        lenlp = 0;
        int var = cols[j];

        // Add the contributions from A
        for (int k = rowp[var] + elen[var]; k < rowp[var] + alen[var]; k++) {
          Lp[lenlp] = cols[k];
          lenlp++;
        }

        // Add the sets from the elements in this row
        for (int k = rowp[var]; k < rowp[var] + elen[var]; k++) {
          int e = cols[k];  // Take the elements belonging to var

          // Now, add in the variables corresponding to the element e
          for (int p = rowp[e] + elen[e]; p < rowp[e] + alen[e]; p++) {
            if (cols[p] != var) {
              if (lenlp >= nvars) {
                lenlp = ParOptRemoveDuplicates(Lp, lenlp);
              }
              Lp[lenlp] = cols[p];
              lenlp++;
            }
          }
        }

        lenlp = ParOptRemoveDuplicates(Lp, lenlp);

        int deg = 0;
        for (int k = 0; k < lenlp; k++) {
          deg += slen[Lp[k]];
        }
        deglist.update_degree(var, deg);
        // degree[var] = deg;
      }
    } else {  // Approximate degree
      // The worst cases are:
      // The trailing submatrix is dense: degree = N-i
      // All nodes in k, result in new fill in: d^{i+1} = d^{i} + lenlp
      // The approximate degree is:
      //          |A_{i} \ i| + |L_p \ i| + sum_{e} | L_{e} \ L_p|

      // For each supervariable in Lp
      for (int j = 0; j < lenlp; j++) {
        int lj = Lp[j];
        int end = rowp[lj] + elen[lj];
        // For all elements pointed to by row Lp[j] = lj
        for (int k = rowp[lj]; k < end; k++) {
          // Find all the elements
          int e = cols[k];
          if (elem_degree[e] < 0) {
            // Calculate the element degree
            elem_degree[e] = 0;
            for (int p = rowp[e]; p < rowp[e] + alen[e]; p++) {
              elem_degree[e] += slen[cols[p]];
            }
          }
          elem_degree[e] -= slen[lj];
        }
      }

      // Compute | Lp |
      int deg_Lp = 0;
      for (int j = 0; j < lenlp; j++) {
        if (Lp[j] != piv) {
          deg_Lp += slen[Lp[j]];
        }
      }

      for (int j = 0; j < lenlp; j++) {
        int lj = Lp[j];

        if (lj == piv) {
          continue;
        }

        // Add the contributions from A
        int deg_estimate = deg_Lp;
        for (int k = rowp[lj] + elen[lj]; k < rowp[lj] + alen[lj]; k++) {
          deg_estimate += slen[cols[k]];
        }

        // If lj is in A
        int start = rowp[lj] + elen[lj];
        int len = alen[lj] - elen[lj];
        if (len > 0) {
          if (std::binary_search(&cols[start], &cols[start] + len, lj)) {
            deg_estimate -= slen[lj];
          }
        }

        // If lj is in Lp U piv
        if (std::binary_search(Lp, Lp + lenlp, lj)) {  //
          deg_estimate -= slen[lj];
        }

        for (int k = rowp[lj]; k < rowp[lj] + elen[lj]; k++) {
          int e = cols[k];
          if (e == piv) {
            continue;
          }

          if (elem_degree[e] < 0) {
            elem_degree[e] = 0;
            for (int p = rowp[e]; p < rowp[e] + alen[e]; p++) {
              elem_degree[e] += slen[cols[p]];
            }
            deg_estimate += elem_degree[e];
          } else {
            deg_estimate += elem_degree[e];
          }
        }

        // Now, compute the degree estimate
        deg_estimate = (deg_estimate < nvars - i ? deg_estimate : nvars - i);

        // Update the degree
        int d = deglist.get_degree(lj);
        d = (deg_estimate < d + deg_Lp ? deg_estimate : d + deg_Lp);
        deglist.update_degree(lj, d);

        for (int j = 0; j < lenlp; j++) {
          int lj = Lp[j];
          int end = rowp[lj] + elen[lj];
          // For all elements pointed to by row Lp[j] = lj
          for (int k = rowp[lj]; k < end; k++) {
            // Find all the elements
            int e = cols[k];
            elem_degree[e] = -1;
          }
        }
      }
    }

    // Supervariable detection and construction
    for (int j = 0; j < lenlp; j++) {
      int lj = Lp[j];

      if (lj == piv || slen[lj] < 0) {
        continue;
      }

      for (int k = j + 1; k < lenlp; k++) {
        int lk = Lp[k];

        if (lk == piv || slen[lk] < 0) {
          continue;
        }

        // Quick check to see if the nodes are the same
        if (ParOptCompareVariable(lj, lk, elen, alen, rowp, cols)) {
          // Merge lk into lj
          slen[lj] += slen[lk];
          // degree[lj] -= slen[lk];
          int d = deglist.get_degree(lj);
          d -= slen[lk];
          deglist.update_degree(lj, d);
          slen[lk] = -(lj + 1);
          state[lk] = -1;

          // Remove lj from the quotient graph
          ParOptRemoveVariable(lk, elen, alen, rowp, cols, nvars);
          deglist.remove(lk);
        }
      }
    }

    // Reduce the number of variables to eliminate
    i = i + slen[piv];
  }

  // We have the following situation...
  // perm[0:nsvars] = piv, contains the supernodal pivots
  // slen[i] > 0 means i is a principal supervariable
  // slen[i] < 0 means variable i was collapsed into supervariable (slen[i]-1)

  // First arrange the principal supervariables in the permutation array
  int end = nvars;
  for (int j = nsvars - 1; j >= 0; j--) {
    int sv = perm[j];     // Get the supervariable selected last
    end -= slen[sv];      // Figure out where it should start
    perm[end] = sv;       // Place the principal variable at the beginning
    state[sv] = end + 1;  // Increment the pointer into the array
  }

  // Fill in the non-principal variables
  for (int k = 0; k < nvars; k++) {
    if (slen[k] < 0) {  // non-principal variable
      // Back-track until we find a princiapl supervariable
      int j = k;
      while (slen[j] < 0) {  //
        j = -(slen[j] + 1);  // j was eliminated by -(slen[j]+1) and so on...
      }
      // j should now be a principal supervariable
      perm[state[j]] = k;
      state[j]++;
    }
  }

  delete[] alen;
  delete[] elen;
  delete[] degree;
  delete[] elem_degree;
  delete[] Lp;
  delete[] state;
  delete[] slen;
}
