#ifndef PAR_OPT_AMD_H
#define PAR_OPT_AMD_H

/*
  Comptute the AMD reordering of the variables.

  Note: cols is overwritten
*/
void ParOptAMD(int nvars, int *rowp, int *cols, int *perm,
               int use_exact_degree);

#endif  // PAR_OPT_AMD_H