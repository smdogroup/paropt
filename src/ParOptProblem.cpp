#include "ParOptProblem.h"

#include <string.h>

#include "ParOptComplexStep.h"

ParOptProblem::ParOptProblem(MPI_Comm _comm) {
  comm = _comm;
  nvars = ncon = nwcon = 0;
  ninequality = nwinequality = -1;
}

ParOptProblem::~ParOptProblem() {}

/**
  Create a new distributed design vector

  @return a new distributed design vector
*/
ParOptVec *ParOptProblem::createDesignVec() {
  return new ParOptBasicVec(comm, nvars);
}

/**
  Create a new distributed sparse constraint vector

  @return a new distributed sparse constraint vector
*/
ParOptVec *ParOptProblem::createConstraintVec() {
  return new ParOptBasicVec(comm, nwcon);
}

/**
  Get the communicator for the problem

  @return the MPI communicator for the problem
*/
MPI_Comm ParOptProblem::getMPIComm() { return comm; }

/**
  Set the problem size

  @param _nvars the number of local design variables
  @param _ncon the global number of dense constraints
  @param _nwcon the local number of sparse separable constraints
*/
void ParOptProblem::setProblemSizes(int _nvars, int _ncon, int _nwcon) {
  nvars = _nvars;
  ncon = _ncon;
  nwcon = _nwcon;

  // By default, all the constraints are treated as inequalities
  if (ninequality < 0) {
    ninequality = ncon;
  }
  if (nwinequality < 0) {
    nwinequality = nwcon;
  }
}

/**
  Set the number of sparse or dense inequalities

  @param _ninequality the number of inequality constraints
  @param _nwinequality the block size of the separable constraints
*/
void ParOptProblem::setNumInequalities(int _ninequality, int _nwinequality) {
  ninequality = _ninequality;
  nwinequality = _nwinequality;
}

/**
  Get the problem size

  @param _nvars the number of local design variables
  @param _ncon the global number of dense constraints
  @param _nwcon the local number of sparse separable constraints
*/
void ParOptProblem::getProblemSizes(int *_nvars, int *_ncon, int *_nwcon) {
  if (_nvars) {
    *_nvars = nvars;
  }
  if (_ncon) {
    *_ncon = ncon;
  }
  if (_nwcon) {
    *_nwcon = nwcon;
  }
}

/**
  Get the number of inequalities

  @param _ninequality the number of dense inequality constraints
  @param _nwinequality the block size of the separable constraints
*/
void ParOptProblem::getNumInequalities(int *_ninequality, int *_nwinequality) {
  if (_ninequality) {
    *_ninequality = ninequality;
  }
  if (_nwinequality) {
    *_nwinequality = nwinequality;
  }
}

/**
  Are the dense constraints inequalities? Default is true.

  @return flag indicating if the dense constraints are inequalities
*/
int ParOptProblem::isSparseInequality() { return 1; }

/**
  Indicate whether to use the lower variable bounds. Default is true.

  @return flag indicating whether to use lower variable bound.
*/
int ParOptProblem::useLowerBounds() { return 1; }

/**
   Indicate whether to use the upper variable bounds. Default is true.

   @return flag indicating whether to use upper variable bounds.
 */
int ParOptProblem::useUpperBounds() { return 1; }

/**
  Evaluate the product of the Hessian with a given vector.

  This function is only called if Hessian-vector products are requested
  by the optimizer. By default, no implementation is provided.

  @param x is the design variable vector
  @param z is the array of multipliers for the dense constraints
  @param zw is the vector of multipliers for the sparse constraints
  @param px is the direction vector
  @param hvec is the output vector hvec = H(x,z,zw)*px
  @return zero on success, non-zero flag on error
*/
int ParOptProblem::evalHvecProduct(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                                   ParOptVec *px, ParOptVec *hvec) {
  return 0;
}

/**
  Evaluate the diagonal of the Hessian.

  This is only used by MMA.
*/
int ParOptProblem::evalHessianDiag(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                                   ParOptVec *hdiag) {
  return 0;
}

/**
  Evaluate the sparse constraints.

  Give the design variable vector x, compute the sparse constraints.

  @param x is the design variable vector
  @param out is the sparse constraint vector
*/
void ParOptProblem::evalSparseCon(ParOptVec *x, ParOptVec *out) {}

/**
  Compute the Jacobian-vector product of the sparse constraints.

  This code computes the action of the Jacobian of the sparse constraint
  matrix on the input vector px, to compute out = alpha*J(x)*px.

  @param alpha is a scalar factor
  @param x is the design variable vector
  @param px is the input direction vector
  @param out is the sparse product vector
*/
void ParOptProblem::addSparseJacobian(ParOptScalar alpha, ParOptVec *x,
                                      ParOptVec *px, ParOptVec *out) {}

/**
  Compute the tranpose Jacobian-vector product of the sparse constraints.

  This code computes the action of the transpose Jacobian of the sparse
  constraint matrix on the input vector pzw, to compute
  out = alpha*J(x)^{T}*pzw.

  @param alpha is a scalar factor
  @param x is the design variable vector
  @param pzw is the input direction vector
  @param out is the sparse product vector
*/
void ParOptProblem::addSparseJacobianTranspose(ParOptScalar alpha, ParOptVec *x,
                                               ParOptVec *pzw, ParOptVec *out) {
}

/**
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix

  @param alpha is a scalar factor
  @param x is the design variable vector
  @param cvec are input components of the diagonal matrix
  @param A is the output block-diagonal matrix
*/
void ParOptProblem::addSparseInnerProduct(ParOptScalar alpha, ParOptVec *x,
                                          ParOptVec *cvec, ParOptScalar *A) {}
/**
  Compute a correction or modification of the quasi-Newton update.

  The vectors s and y represent the step and gradient difference,
  respectively between iterations. By default, no correction or
  modification is performed. However, some problems may benefit by
  modifying the gradient difference term.

  @param x is the design variable vector
  @param z is the array of multipliers for the dense constraints
  @param zw is the vector of multipliers for the sparse constraints
  @param s The step in the quasi-Newton update
  @param y The gradient difference in the quasi-Newton update
*/
void ParOptProblem::computeQuasiNewtonUpdateCorrection(
    ParOptVec *x, ParOptScalar *z, ParOptVec *zw, ParOptVec *s, ParOptVec *y) {}

void ParOptProblem::writeOutput(int iter, ParOptVec *x) {}

void ParOptProblem::checkGradients(double dh, ParOptVec *xvec,
                                   int check_hvec_product) {
  ParOptVec *x = xvec;
  if (!xvec) {
    x = createDesignVec();
    x->incref();
  }

  // Create the perturbation vector
  ParOptVec *px = createDesignVec();
  px->incref();

  // Allocate space for constraints
  ParOptScalar fobj = 0.0;
  ParOptScalar *c = new ParOptScalar[ncon];

  // Allocate objective gradient/constraint gradient
  ParOptVec *g = createDesignVec();
  g->incref();

  ParOptVec **Ac = new ParOptVec *[ncon];
  for (int i = 0; i < ncon; i++) {
    Ac[i] = createDesignVec();
    Ac[i]->incref();
  }

  // Get the design variable values. Note that the values in g and px are
  // discarded.
  if (!xvec) {
    getVarsAndBounds(x, g, px);
  }

  // Evaluate the objective/constraint and gradients
  evalObjCon(x, &fobj, c);
  evalObjConGradient(x, g, Ac);

  ParOptScalar *pxvals, *gvals;
  px->getArray(&pxvals);
  g->getArray(&gvals);
  for (int i = 0; i < nvars; i++) {
    if (ParOptRealPart(gvals[i]) >= 0.0) {
      pxvals[i] = 1.0;
    } else {
      pxvals[i] = -1.0;
    }
  }

  // Compute the projected derivative of the objective
  ParOptScalar pobj = g->dot(px);

  // Compute the projected derivative of the constraints
  ParOptScalar *Apx = new ParOptScalar[ncon];
  px->mdot(Ac, ncon, Apx);

  // Set the step direction in the sparse Lagrange multipliers
  // to an initial vector
  ParOptVec *zw = NULL;
  if (nwcon > 0) {
    zw = createConstraintVec();
    zw->incref();

    // Set a value for the zw array
    ParOptScalar *zwvals;
    zw->getArray(&zwvals);
    for (int i = 0; i < nwcon; i++) {
      zwvals[i] = 1.05 + 0.25 * (i % 21);
    }
  }

  // Evaluate the Hessian-vector product
  ParOptScalar *ztemp = NULL;
  ParOptVec *hvec = NULL, *htemp = NULL;

  if (check_hvec_product) {
    ztemp = new ParOptScalar[ncon];
    for (int i = 0; i < ncon; i++) {
      ztemp[i] = 2.3 - 0.15 * (i % 5);
    }

    // Add the contribution to gradient of the Lagrangian
    // from the sparse constraints
    if (nwcon > 0) {
      addSparseJacobianTranspose(-1.0, x, zw, g);
    }

    for (int i = 0; i < ncon; i++) {
      g->axpy(-ztemp[i], Ac[i]);
    }

    // Evaluate the Hessian-vector product
    hvec = createDesignVec();
    hvec->incref();
    evalHvecProduct(x, ztemp, zw, px, hvec);

    // Check that multiple calls to the Hvec code
    // produce the same result
    htemp = createDesignVec();
    htemp->incref();
    evalHvecProduct(x, ztemp, zw, px, htemp);

    int rank;
    MPI_Comm_rank(comm, &rank);
    htemp->axpy(-1.0, hvec);
    double diff_nrm = htemp->norm();

    if (rank == 0) {
      printf("Hvec code reproducibility test\n");
      printf("Difference between multiple calls: %15.8e\n\n", diff_nrm);
    }
  }

  // Compute the point xt = x + dh*px
  ParOptVec *xt = createDesignVec();
  xt->incref();

  // Copy values into the constraint
  xt->copyValues(x);

#ifdef PAROPT_USE_COMPLEX
  xt->axpy(ParOptScalar(0.0, dh), px);
#else
  xt->axpy(dh, px);
#endif  // PAROPT_USE_COMPLEX

  // Compute the finite-difference product
  ParOptScalar fobj2;
  ParOptScalar *ctemp = new ParOptScalar[ncon];
  evalObjCon(xt, &fobj2, ctemp);

#ifdef PAROPT_USE_COMPLEX
  ParOptScalar pfd = ParOptImagPart(fobj2) / dh;
#else
  ParOptScalar pfd = (fobj2 - fobj) / dh;
#endif  // PAROPT_USE_COMPLEX

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == 0) {
    printf("Objective gradient test\n");
    printf(
        "Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  "
        "Rel err: %8.2e\n",
        ParOptRealPart(pfd), ParOptRealPart(pobj),
        fabs(ParOptRealPart(pobj - pfd)),
        fabs(ParOptRealPart((pobj - pfd) / pfd)));

    printf("\nConstraint gradient test\n");
    for (int i = 0; i < ncon; i++) {
#ifdef PAROPT_USE_COMPLEX
      ParOptScalar fd = ParOptImagPart(ctemp[i]) / dh;
#else
      ParOptScalar fd = (ctemp[i] - c[i]) / dh;
#endif  // PAROPT_USE_COMPLEX

      printf(
          "Con[%3d]  FD: %15.8e  Actual: %15.8e  Err: %8.2e  "
          "Rel err: %8.2e\n",
          i, ParOptRealPart(fd), ParOptRealPart(Apx[i]),
          fabs(ParOptRealPart(fd - Apx[i])),
          fabs(ParOptRealPart((fd - Apx[i]) / fd)));
    }
  }

  if (check_hvec_product) {
    // Evaluate the objective/constraints
    ParOptVec *g2 = createDesignVec();
    g2->incref();

    ParOptVec **Ac2 = new ParOptVec *[ncon];
    for (int i = 0; i < ncon; i++) {
      Ac2[i] = createDesignVec();
      Ac2[i]->incref();
    }

    // Evaluate the gradient at the perturbed point and add the
    // contribution from the sparse constraints to the Hessian
    evalObjConGradient(xt, g2, Ac2);
    if (nwcon > 0) {
      addSparseJacobianTranspose(-1.0, xt, zw, g2);
    }

    // Add the contribution from the dense constraints
    for (int i = 0; i < ncon; i++) {
      g2->axpy(-ztemp[i], Ac2[i]);
    }

#ifdef PAROPT_USE_COMPLEX
    // Evaluate the real part
    ParOptScalar *gvals;
    int gsize = g2->getArray(&gvals);

    for (int i = 0; i < gsize; i++) {
      gvals[i] = ParOptImagPart(gvals[i]) / dh;
    }
#else
    // Compute the difference
    g2->axpy(-1.0, g);
    g2->scale(1.0 / dh);
#endif  // TACS_USE_COMPLEX

    // Compute the norm of the finite-difference approximation and
    // actual Hessian-vector products
    double fdnorm = g2->norm();
    double hnorm = hvec->norm();

    // Compute the max error between the two
    g2->axpy(-1.0, hvec);
    double herr = g2->norm();

    if (rank == 0) {
      printf("\nHessian product test\n");
      printf(
          "Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  "
          "Rel err: %8.2e\n",
          fdnorm, hnorm, herr, herr / hnorm);
    }

    // Clean up the allocated data
    g2->decref();
    for (int i = 0; i < ncon; i++) {
      Ac2[i]->decref();
    }
    delete[] Ac2;
  }

  // Now, perform a check of the sparse constraints (if any)
  ParOptVec *cw = NULL, *cwtemp = NULL;
  if (nwcon > 0) {
    cw = createConstraintVec();
    cw->incref();

    // Allocate the constraint vectors
    cwtemp = createConstraintVec();
    cwtemp->incref();

#ifdef PAROPT_USE_COMPLEX
    // Check that the Jacobian is the derivative of the constraints
    xt->copyValues(x);
    xt->axpy(ParOptScalar(0.0, dh), px);
    evalSparseCon(xt, cwtemp);

    ParOptScalar *cwcvals;
    int cwsize = cwtemp->getArray(&cwcvals);
    for (int i = 0; i < cwsize; i++) {
      cwcvals[i] = ParOptImagPart(cwcvals[i]) / dh;
    }
#else
    // Check that the Jacobian is the derivative of the constraints
    evalSparseCon(x, cw);

    xt->copyValues(x);
    xt->axpy(dh, px);
    evalSparseCon(xt, cwtemp);

    // Compute rcw = (cw(x + dh*px) - cw(x))/dh
    cwtemp->axpy(-1.0, cw);
    cwtemp->scale(1.0 / dh);
#endif
    // Compute the Jacobian-vector product
    cw->zeroEntries();
    addSparseJacobian(1.0, x, px, cw);

    // Compute the difference between the vectors
    cwtemp->axpy(-1.0, cw);

    // Compute the relative difference
    double cw_error = cwtemp->maxabs();

    if (rank == 0) {
      printf("\nSparse constraint checks\n");
      printf("||(cw(x + h*px) - cw(x))/h - J(x)*px||: %8.2e\n", cw_error);
    }

    // Check the that the matrix-multiplication and its transpose are
    // equivalent by computing the inner product with two vectors
    // from either side
    cw->zeroEntries();
    addSparseJacobian(1.0, x, px, cw);

    g->zeroEntries();
    addSparseJacobianTranspose(1.0, x, zw, g);

    ParOptScalar d1 = zw->dot(cw);
    ParOptScalar d2 = g->dot(px);

    if (rank == 0) {
      printf("\nTranspose-equivalence\n");
      printf(
          "x^{T}*(J(x)*p): %8.2e  p*(J(x)^{T}*x): %8.2e  "
          "Err: %8.2e  Rel Err: %8.2e\n",
          ParOptRealPart(d1), ParOptRealPart(d2), fabs(ParOptRealPart(d1 - d2)),
          fabs(ParOptRealPart((d1 - d2) / d2)));
    }

    // ParOptVec *Cvec = createDesignVec();
    // Cvec->incref();

    // // Set Cvec to something more-or-less random
    // ParOptScalar *cvals;
    // Cvec->getArray(&cvals);
    // for (int i = 0; i < nvars; i++) {
    //   cvals[i] = 0.05 + 0.25 * (i % 37);
    // }

    // // Check the inner product zw^{T}*J(x)*cvec*J(x)^{T}*zw against the
    // // matrix Cw
    // ParOptScalar *Cw = new ParOptScalar[nwcon * (nwblock + 1) / 2];
    // memset(Cw, 0, nwcon * (nwblock + 1) / 2 * sizeof(ParOptScalar));
    // addSparseInnerProduct(1.0, x, Cvec, Cw);

    // // Compute the vector product using the Jacobians
    // px->zeroEntries();
    // addSparseJacobianTranspose(1.0, x, zw, px);

    // // Multiply component-wise
    // for (int i = 0; i < nvars; i++) {
    //   pxvals[i] *= cvals[i];
    // }
    // cw->zeroEntries();
    // addSparseJacobian(1.0, x, px, cw);
    // d1 = cw->dot(zw);

    // // Set the pointer into the Cw
    // d2 = 0.0;
    // ParOptScalar *cwvals = Cw;

    // ParOptScalar *zwvals;
    // zw->getArray(&zwvals);

    // // Iterate over each block matrix
    // for (int i = 0; i < nwcon; i += nwblock) {
    //   // Index into each block
    //   for (int j = 0; j < nwblock; j++) {
    //     for (int k = 0; k < j; k++) {
    //       d2 += 2.0 * cwvals[0] * zwvals[i + j] * zwvals[i + k];
    //       cwvals++;
    //     }

    //     d2 += cwvals[0] * zwvals[i + j] * zwvals[i + j];
    //     cwvals++;
    //   }
    // }

    // // Add the result across all processors
    // ParOptScalar temp = d2;
    // MPI_Reduce(&temp, &d2, 1, PAROPT_MPI_TYPE, MPI_SUM, 0, comm);

    // if (rank == 0) {
    //   printf("\nJ(x)*C^{-1}*J(x)^{T} test: \n");
    //   printf("Product: %8.2e  Matrix: %8.2e  Err: %8.2e  Rel Err: %8.2e\n",
    //          ParOptRealPart(d1), ParOptRealPart(d2),
    //          fabs(ParOptRealPart(d1 - d2)),
    //          fabs(ParOptRealPart((d1 - d2) / d2)));
    // }

    // delete[] Cw;
    // Cvec->decref();
  }

  // Deallocate vector
  delete[] c;
  delete[] ctemp;
  delete[] Apx;
  if (!xvec) {
    x->decref();
  }
  px->decref();
  g->decref();
  for (int i = 0; i < ncon; i++) {
    Ac[i]->decref();
  }
  delete[] Ac;
  xt->decref();

  if (ztemp) {
    delete[] ztemp;
  }
  if (zw) {
    zw->decref();
  }
  if (hvec) {
    hvec->decref();
  }
  if (htemp) {
    htemp->decref();
  }
  if (cw) {
    cw->decref();
  }
  if (cwtemp) {
    cwtemp->decref();
  }
}

ParOptSparseProblem::ParOptSparseProblem(MPI_Comm comm) : ParOptProblem(comm) {
  rowp = NULL;
  cols = NULL;
  data = NULL;
  cw = NULL;
  nnz = 0;
}

void ParOptSparseProblem::setSparseJacobianData(const int *_rowp,
                                                const int *_cols) {
  if (rowp) {
    delete[] rowp;
  }
  if (cols) {
    delete[] cols;
  }
  if (data) {
    delete[] data;
  }
  if (cw) {
    cw->decref();
  }

  // Sparse constraint data
  cw = createConstraintVec();
  cw->incref();

  // Copy the constraint Jacobian non-zero pattern
  rowp = new int[nwcon + 1];
  memcpy(rowp, _rowp, (nwcon + 1) * sizeof(int));

  nnz = rowp[nwcon];

  cols = new int[nnz];
  memcpy(cols, _cols, nnz * sizeof(int));

  // Check to make sure that the sparse matrix entries are correct
  int count = 0;
  for (int i = 0; i < nnz; i++) {
    if (cols[i] < 0 || cols[i] > nvars) {
      count++;
    }
  }
  if (count > 0) {
    fprintf(
        stderr,
        "ParOptSparseProblem: %d columns out of range in sparse Jacobian.\n",
        count);
  }

  data = new ParOptScalar[nnz];
  memset(data, 0, nnz * sizeof(ParOptScalar));
}

ParOptSparseProblem::~ParOptSparseProblem() {
  cw->decref();
  delete[] rowp;
  delete[] cols;
  delete[] data;
}

/*
  Get the sparse constraint Jacobian data
*/
void ParOptSparseProblem::getSparseJacobianData(const int **_rowp,
                                                const int **_cols,
                                                const ParOptScalar **_data) {
  if (_rowp) {
    *_rowp = rowp;
  }
  if (_cols) {
    *_cols = cols;
  }
  if (_data) {
    *_data = data;
  }
}

/**
  Create a new quasi-definite matrix object

  @return a new quasi-definite matrix object
*/
ParOptQuasiDefMat *ParOptSparseProblem::createQuasiDefMat() {
  return new ParOptQuasiDefSparseMat(this);
}

/**
  Evaluate the objective and constraints.

  This makes a call to the sparse constraint implementation.

  @param x is the design variable vector
  @param fobj is the objective value at x
  @param cons is the array of constraint vaules at x
  @return zero on success, non-zero fail flag on error
*/
int ParOptSparseProblem::evalObjCon(ParOptVec *x, ParOptScalar *fobj,
                                    ParOptScalar *cons) {
  return evalSparseObjCon(x, fobj, cons, cw);
}

/**
  Evaluate the objective and constraint gradients.

  This makes a call to the sparse constraint Jacobian implementation.

  @param x is the design variable vector
  @param g is the gradient of the objective at x
  @param Ac are the gradients of the dense constraints at x
  @return zero on success, non-zero fail flag on error
*/
int ParOptSparseProblem::evalObjConGradient(ParOptVec *x, ParOptVec *g,
                                            ParOptVec **Ac) {
  return evalObjConSparseGradient(x, g, Ac, data);
}

/**
  Evaluate the sparse constraints.

  This copies the values of the sparse constraints into the output

  @param x is the design variable vector
  @param out is the sparse constraint vector
*/
void ParOptSparseProblem::evalSparseCon(ParOptVec *x, ParOptVec *out) {
  out->copyValues(cw);
}

/**
  Compute the Jacobian-vector product of the sparse constraints.

  This code computes the sparse matrix-Jacobian product uing the stored data.

  @param alpha is a scalar factor
  @param x is the design variable vector
  @param px is the input direction vector
  @param out is the sparse product vector
*/
void ParOptSparseProblem::addSparseJacobian(ParOptScalar alpha, ParOptVec *x,
                                            ParOptVec *px, ParOptVec *out) {
  ParOptScalar *px_array, *out_array;
  px->getArray(&px_array);
  out->getArray(&out_array);

  const ParOptScalar *vals = data;
  for (int i = 0; i < nwcon; i++) {
    int jp = rowp[i];
    int jp_end = rowp[i + 1];
    const int *j = &cols[jp];

    ParOptScalar out_val = 0.0;
    for (; jp < jp_end; jp++, j++, vals++) {
      out_val += vals[0] * px_array[j[0]];
    }
    out_array[0] += alpha * out_val;
    out_array++;
  }
}

/**
  Compute the tranpose Jacobian-vector product of the sparse constraints.

  This code computes the transpose of the sparse

  @param alpha is a scalar factor
  @param x is the design variable vector
  @param pzw is the input direction vector
  @param out is the sparse product vector
*/
void ParOptSparseProblem::addSparseJacobianTranspose(ParOptScalar alpha,
                                                     ParOptVec *x,
                                                     ParOptVec *pzw,
                                                     ParOptVec *out) {
  ParOptScalar *pzw_array, *out_array;
  pzw->getArray(&pzw_array);
  out->getArray(&out_array);

  const ParOptScalar *vals = data;
  for (int i = 0; i < nwcon; i++) {
    int jp = rowp[i];
    int jp_end = rowp[i + 1];
    const int *j = &cols[jp];

    for (; jp < jp_end; jp++, j++, vals++) {
      out_array[j[0]] += alpha * vals[0] * pzw_array[0];
    }
    pzw_array++;
  }
}