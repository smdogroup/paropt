#include "CyParOptProblem.h"

#include <string.h>

#include "ParOptSparseMat.h"

/**
  The constructor for the ParOptProblem wrapper.

  This wrapper is used when generating a Python instance of a
  ParOptProblem class. The wrapper is instantiated by Cython
  and callbacks are used to implement the required API.

  @param _comm the MPI communicator
  @param _nvars the local number of variables
  @param _ncon the number of global constraints
  @param _nwcon the number of sparse constraints
  @param _nwblock the size of the sparse constraint block
*/
CyParOptBlockProblem::CyParOptBlockProblem(MPI_Comm _comm, int _nwblock)
    : ParOptProblem(_comm) {
  nwblock = _nwblock;

  // Set the default options
  useLower = 1;
  useUpper = 1;

  // Set the initial values for the callbacks
  self = NULL;
  getvarsandbounds = NULL;
  evalobjcon = NULL;
  evalobjcongradient = NULL;
  evalhvecproduct = NULL;
  evalhessiandiag = NULL;
  evalsparsecon = NULL;
  addsparsejacobian = NULL;
  addsparsejacobiantranspose = NULL;
  addsparseinnerproduct = NULL;
  computequasinewtonupdatecorrection = NULL;
}

CyParOptBlockProblem::~CyParOptBlockProblem() {}

/*
  Return the quasi-def block matrix instance associated with this problem
*/
ParOptQuasiDefMat *CyParOptBlockProblem::createQuasiDefMat() {
  // Check that the block size makes sense
  if (nwcon > 0 && nwcon % nwblock != 0) {
    fprintf(stderr, "ParOpt: Weighted block size inconsistent\n");
  }

  return new ParOptQuasiDefBlockMat(this, nwblock);
}

/**
  Set options associated with the inequality constraints

  @param _useLower indicates whether to use the lower bounds
  @param _useUpper indicates whether to use the upper boundss
*/
void CyParOptBlockProblem::setVarBoundOptions(int _useLower, int _useUpper) {
  useLower = _useLower;
  useUpper = _useUpper;
}

/*
  Function to indicate the type of sparse constraints
*/
int CyParOptBlockProblem::useLowerBounds() { return useLower; }
int CyParOptBlockProblem::useUpperBounds() { return useUpper; }

/*
  Set the member callback functions that are required
*/
void CyParOptBlockProblem::setSelfPointer(void *_self) { self = _self; }

void CyParOptBlockProblem::setGetVarsAndBounds(
    void (*func)(void *, int, ParOptVec *, ParOptVec *, ParOptVec *)) {
  getvarsandbounds = func;
}

void CyParOptBlockProblem::setEvalObjCon(int (*func)(
    void *, int, int, ParOptVec *, ParOptScalar *, ParOptScalar *)) {
  evalobjcon = func;
}

void CyParOptBlockProblem::setEvalObjConGradient(
    int (*func)(void *, int, int, ParOptVec *, ParOptVec *, ParOptVec **)) {
  evalobjcongradient = func;
}

void CyParOptBlockProblem::setEvalHvecProduct(
    int (*func)(void *, int, int, int, ParOptVec *, ParOptScalar *, ParOptVec *,
                ParOptVec *, ParOptVec *)) {
  evalhvecproduct = func;
}

void CyParOptBlockProblem::setEvalHessianDiag(
    int (*func)(void *, int, int, int, ParOptVec *, ParOptScalar *, ParOptVec *,
                ParOptVec *)) {
  evalhessiandiag = func;
}

void CyParOptBlockProblem::setComputeQuasiNewtonUpdateCorrection(
    void (*func)(void *, int, int, ParOptVec *, ParOptScalar *, ParOptVec *,
                 ParOptVec *, ParOptVec *)) {
  computequasinewtonupdatecorrection = func;
}

void CyParOptBlockProblem::setEvalSparseCon(void (*func)(void *, int, int,
                                                         ParOptVec *,
                                                         ParOptVec *)) {
  evalsparsecon = func;
}

void CyParOptBlockProblem::setAddSparseJacobian(void (*func)(
    void *, int, int, ParOptScalar, ParOptVec *, ParOptVec *, ParOptVec *)) {
  addsparsejacobian = func;
}

void CyParOptBlockProblem::setAddSparseJacobianTranspose(void (*func)(
    void *, int, int, ParOptScalar, ParOptVec *, ParOptVec *, ParOptVec *)) {
  addsparsejacobiantranspose = func;
}

void CyParOptBlockProblem::setAddSparseInnerProduct(
    void (*func)(void *, int, int, int, ParOptScalar, ParOptVec *, ParOptVec *,
                 ParOptScalar *)) {
  addsparseinnerproduct = func;
}

/*
  Get the variables and bounds from the problem
*/
void CyParOptBlockProblem::getVarsAndBounds(ParOptVec *x, ParOptVec *lb,
                                            ParOptVec *ub) {
  if (!getvarsandbounds) {
    fprintf(stderr, "getvarsandbounds callback not defined\n");
    return;
  }
  getvarsandbounds(self, nvars, x, lb, ub);
}

/*
  Evaluate the objective and constraints
*/
int CyParOptBlockProblem::evalObjCon(ParOptVec *x, ParOptScalar *fobj,
                                     ParOptScalar *cons) {
  if (!evalobjcon) {
    fprintf(stderr, "evalobjcon callback not defined\n");
    return 1;
  }

  // Evaluate the objective and constraints
  int fail = evalobjcon(self, nvars, ncon, x, fobj, cons);

  return fail;
}

/*
  Evaluate the objective and constraint gradients
*/
int CyParOptBlockProblem::evalObjConGradient(ParOptVec *x, ParOptVec *g,
                                             ParOptVec **Ac) {
  if (!evalobjcongradient) {
    fprintf(stderr, "evalobjcongradient callback not defined\n");
    return 1;
  }

  // Evaluate the objective/constraint gradient
  int fail = evalobjcongradient(self, nvars, ncon, x, g, Ac);

  return fail;
}

/*
  Evaluate the product of the Hessian with a given vector
*/
int CyParOptBlockProblem::evalHvecProduct(ParOptVec *x, ParOptScalar *z,
                                          ParOptVec *zw, ParOptVec *px,
                                          ParOptVec *hvec) {
  if (!evalhvecproduct) {
    fprintf(stderr, "evalhvecproduct callback not defined\n");
    return 1;
  }

  // Evaluate the Hessian-vector callback
  int fail = evalhvecproduct(self, nvars, ncon, nwcon, x, z, zw, px, hvec);
  return fail;
}

/*
  Evaluate the diagonal of the Hessian
*/
int CyParOptBlockProblem::evalHessianDiag(ParOptVec *x, ParOptScalar *z,
                                          ParOptVec *zw, ParOptVec *hdiag) {
  if (!evalhessiandiag) {
    fprintf(stderr, "evalhessiandiag callback not defined\n");
    return 1;
  }

  // Evaluate the Hessian-vector callback
  int fail = evalhessiandiag(self, nvars, ncon, nwcon, x, z, zw, hdiag);
  return fail;
}

/*
  Apply the quasi-Newton update correction
*/
void CyParOptBlockProblem::computeQuasiNewtonUpdateCorrection(
    ParOptVec *x, ParOptScalar *z, ParOptVec *zw, ParOptVec *s, ParOptVec *y) {
  if (!computequasinewtonupdatecorrection) {
    fprintf(stderr,
            "computequasinewtonupdatecorrection callback not defined\n");
  }

  // Evaluate the Hessian-vector callback
  computequasinewtonupdatecorrection(self, nvars, ncon, x, z, zw, s, y);
}

/*
  Evaluate the constraints
*/
void CyParOptBlockProblem::evalSparseCon(ParOptVec *x, ParOptVec *out) {
  if (!evalsparsecon) {
    fprintf(stderr, "evalsparsecon callback not defined\n");
    return;
  }

  // Evaluate the Hessian-vector callback
  evalsparsecon(self, nvars, nwcon, x, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void CyParOptBlockProblem::addSparseJacobian(ParOptScalar alpha, ParOptVec *x,
                                             ParOptVec *px, ParOptVec *out) {
  if (!addsparsejacobian) {
    fprintf(stderr, "addsparsejacobian callback not defined\n");
    return;
  }

  // Evaluate the sparse Jacobian output
  addsparsejacobian(self, nvars, nwcon, alpha, x, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void CyParOptBlockProblem::addSparseJacobianTranspose(ParOptScalar alpha,
                                                      ParOptVec *x,
                                                      ParOptVec *pzw,
                                                      ParOptVec *out) {
  if (!addsparsejacobiantranspose) {
    fprintf(stderr, "addsparsejacobiantranspose callback not defined\n");
    return;
  }

  // Evaluate the sparse Jacobian output
  addsparsejacobiantranspose(self, nvars, nwcon, alpha, x, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such that
  A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void CyParOptBlockProblem::addSparseInnerProduct(ParOptScalar alpha,
                                                 ParOptVec *x, ParOptVec *cvec,
                                                 ParOptScalar *A) {
  if (!addsparseinnerproduct) {
    fprintf(stderr, "addsparseinnerproduct callback not defined\n");
    return;
  }

  // Evaluate the sparse Jacobian output
  addsparseinnerproduct(self, nvars, nwcon, nwblock, alpha, x, cvec, A);
}

/**
  The constructor for the ParOptProblem wrapper.

  This wrapper is used when generating a Python instance of a
  ParOptProblem class. The wrapper is instantiated by Cython
  and callbacks are used to implement the required API.

  @param _comm the MPI communicator
  @param _nvars the local number of variables
  @param _ncon the number of global constraints
  @param _nwcon the number of sparse constraints
*/
CyParOptSparseProblem::CyParOptSparseProblem(MPI_Comm _comm)
    : ParOptSparseProblem(_comm) {
  // Set the default options
  useLower = 1;
  useUpper = 1;

  // Set the initial values for the callbacks
  self = NULL;
  getvarsandbounds = NULL;
  evalobjcon = NULL;
  evalobjcongradient = NULL;
  evalhvecproduct = NULL;
  evalhessiandiag = NULL;
  computequasinewtonupdatecorrection = NULL;
}

CyParOptSparseProblem::~CyParOptSparseProblem() {}

/**
  Set options associated with the inequality constraints

  @param _useLower indicates whether to use the lower bounds
  @param _useUpper indicates whether to use the upper boundss
*/
void CyParOptSparseProblem::setVarBoundOptions(int _useLower, int _useUpper) {
  useLower = _useLower;
  useUpper = _useUpper;
}

/*
  Function to indicate the type of sparse constraints
*/
int CyParOptSparseProblem::useLowerBounds() { return useLower; }
int CyParOptSparseProblem::useUpperBounds() { return useUpper; }

/*
  Set the member callback functions that are required
*/
void CyParOptSparseProblem::setSelfPointer(void *_self) { self = _self; }

void CyParOptSparseProblem::setGetVarsAndBounds(
    void (*func)(void *, int, ParOptVec *, ParOptVec *, ParOptVec *)) {
  getvarsandbounds = func;
}

void CyParOptSparseProblem::setEvalObjCon(
    int (*func)(void *, int, int, int, ParOptVec *, ParOptScalar *,
                ParOptScalar *, ParOptVec *)) {
  evalobjcon = func;
}

void CyParOptSparseProblem::setEvalObjConGradient(
    int (*func)(void *, int, int, int, ParOptVec *, ParOptVec *, ParOptVec **,
                int, ParOptScalar *)) {
  evalobjcongradient = func;
}

void CyParOptSparseProblem::setEvalHvecProduct(
    int (*func)(void *, int, int, int, ParOptVec *, ParOptScalar *, ParOptVec *,
                ParOptVec *, ParOptVec *)) {
  evalhvecproduct = func;
}

void CyParOptSparseProblem::setEvalHessianDiag(
    int (*func)(void *, int, int, int, ParOptVec *, ParOptScalar *, ParOptVec *,
                ParOptVec *)) {
  evalhessiandiag = func;
}

void CyParOptSparseProblem::setComputeQuasiNewtonUpdateCorrection(
    void (*func)(void *, int, int, ParOptVec *, ParOptScalar *, ParOptVec *,
                 ParOptVec *, ParOptVec *)) {
  computequasinewtonupdatecorrection = func;
}

/*
  Get the variables and bounds from the problem
*/
void CyParOptSparseProblem::getVarsAndBounds(ParOptVec *x, ParOptVec *lb,
                                             ParOptVec *ub) {
  if (!getvarsandbounds) {
    fprintf(stderr, "getvarsandbounds callback not defined\n");
    return;
  }
  getvarsandbounds(self, nvars, x, lb, ub);
}

/*
  Evaluate the objective and constraints
*/
int CyParOptSparseProblem::evalSparseObjCon(ParOptVec *x, ParOptScalar *fobj,
                                            ParOptScalar *cons,
                                            ParOptVec *sparse_con) {
  if (!evalobjcon) {
    fprintf(stderr, "evalobjcon callback not defined\n");
    return 1;
  }

  // Evaluate the objective and constraints
  int fail = evalobjcon(self, nvars, ncon, nwcon, x, fobj, cons, sparse_con);

  return fail;
}

/*
  Evaluate the objective and constraint gradients
*/
int CyParOptSparseProblem::evalSparseObjConGradient(ParOptVec *x, ParOptVec *g,
                                                    ParOptVec **Ac,
                                                    ParOptScalar *data) {
  if (!evalobjcongradient) {
    fprintf(stderr, "evalobjcongradient callback not defined\n");
    return 1;
  }

  // Get the number of sparse Jacobian entries
  int nnz = getSparseJacobianData(NULL, NULL, NULL);

  // Evaluate the objective/constraint gradient
  int fail = evalobjcongradient(self, nvars, ncon, nwcon, x, g, Ac, nnz, data);

  return fail;
}

/*
  Evaluate the product of the Hessian with a given vector
*/
int CyParOptSparseProblem::evalHvecProduct(ParOptVec *x, ParOptScalar *z,
                                           ParOptVec *zw, ParOptVec *px,
                                           ParOptVec *hvec) {
  if (!evalhvecproduct) {
    fprintf(stderr, "evalhvecproduct callback not defined\n");
    return 1;
  }

  // Evaluate the Hessian-vector callback
  int fail = evalhvecproduct(self, nvars, ncon, nwcon, x, z, zw, px, hvec);
  return fail;
}

/*
  Evaluate the diagonal of the Hessian
*/
int CyParOptSparseProblem::evalHessianDiag(ParOptVec *x, ParOptScalar *z,
                                           ParOptVec *zw, ParOptVec *hdiag) {
  if (!evalhessiandiag) {
    fprintf(stderr, "evalhessiandiag callback not defined\n");
    return 1;
  }

  // Evaluate the Hessian-vector callback
  int fail = evalhessiandiag(self, nvars, ncon, nwcon, x, z, zw, hdiag);
  return fail;
}

/*
  Apply the quasi-Newton update correction
*/
void CyParOptSparseProblem::computeQuasiNewtonUpdateCorrection(
    ParOptVec *x, ParOptScalar *z, ParOptVec *zw, ParOptVec *s, ParOptVec *y) {
  if (!computequasinewtonupdatecorrection) {
    fprintf(stderr,
            "computequasinewtonupdatecorrection callback not defined\n");
  }

  // Evaluate the Hessian-vector callback
  computequasinewtonupdatecorrection(self, nvars, ncon, x, z, zw, s, y);
}
