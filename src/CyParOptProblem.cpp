#include <string.h>
#include "CyParOptProblem.h"

/**
  The constructor for the ParOptProblem wrapper.

  This wrapper is used when generating a Python instance of a
  ParOptProblem class. The wrapper is instantiated by Cython
  and callbacks are used to implement the required API.

  @param _comm the MPI communicator
  @param _nvars the local number of variables
  @param _ncon the number of global constraints
  @param _ninequality the number of global inequality constraints
  @param _nwcon the number of sparse constraints
  @param _nwblock the size of the sparse constraint block
*/
CyParOptProblem::CyParOptProblem( MPI_Comm _comm,
                                  int _nvars, int _ncon,
                                  int _ninequality,
                                  int _nwcon, int _nwblock ):
ParOptProblem(_comm, _nvars, _ncon, _ninequality, _nwcon, _nwblock){
  // Set the default options
  isSparseInequal = 1;
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

CyParOptProblem::~CyParOptProblem(){}

/**
  Set options associated with the inequality constraints

  @param _isSparseInequal indicates the type of sparse constraint
  @param _useLower indicates whether to use the lower bounds
  @param _useUpper indicates whether to use the upper boundss
*/
void CyParOptProblem::setInequalityOptions( int _isSparseInequal,
                                            int _useLower,
                                            int _useUpper ){
  isSparseInequal = _isSparseInequal;
  useLower = _useLower;
  useUpper = _useUpper;
}

/*
  Function to indicate the type of sparse constraints
*/
int CyParOptProblem::isSparseInequality(){
  return isSparseInequal;
}
int CyParOptProblem::useLowerBounds(){
  return useLower;
}
int CyParOptProblem::useUpperBounds(){
  return useUpper;
}

/*
  Set the member callback functions that are required
*/
void CyParOptProblem::setSelfPointer( void *_self ){
  self = _self;
}

void CyParOptProblem::setGetVarsAndBounds( void (*func)(void *, int,
                                                        ParOptVec*,
                                                        ParOptVec*,
                                                        ParOptVec*) ){
  getvarsandbounds = func;
}

void CyParOptProblem::setEvalObjCon( int (*func)(void*, int, int,
                                                 ParOptVec*,
                                                 ParOptScalar*,
                                                 ParOptScalar*) ){
  evalobjcon = func;
}

void CyParOptProblem::setEvalObjConGradient( int (*func)(void*, int, int,
                                                         ParOptVec*,
                                                         ParOptVec*,
                                                         ParOptVec**) ){
  evalobjcongradient = func;
}

void CyParOptProblem::setEvalHvecProduct( int (*func)(void *, int, int, int,
                                                      ParOptVec*,
                                                      ParOptScalar*,
                                                      ParOptVec*,
                                                      ParOptVec*,
                                                      ParOptVec*) ){
  evalhvecproduct = func;
}

void CyParOptProblem::setEvalHessianDiag( int (*func)(void *, int, int, int,
                                                      ParOptVec*,
                                                      ParOptScalar*,
                                                      ParOptVec*,
                                                      ParOptVec*) ){
  evalhessiandiag = func;
}

void CyParOptProblem::setComputeQuasiNewtonUpdateCorrection( void (*func)(void*, int, int,
                                                                          ParOptVec*,
                                                                          ParOptScalar*,
                                                                          ParOptVec*,
                                                                          ParOptVec*,
                                                                          ParOptVec*) ){
  computequasinewtonupdatecorrection = func;
}

void CyParOptProblem::setEvalSparseCon( void (*func)(void *, int, int,
                                                     ParOptVec*,
                                                     ParOptVec*) ){
  evalsparsecon = func;
}

void CyParOptProblem::setAddSparseJacobian( void (*func)(void *, int, int,
                                                         ParOptScalar,
                                                         ParOptVec*,
                                                         ParOptVec*,
                                                         ParOptVec*) ){
  addsparsejacobian = func;
}

void CyParOptProblem::setAddSparseJacobianTranspose( void (*func)(void *,
                                                                  int, int,
                                                                  ParOptScalar,
                                                                  ParOptVec*,
                                                                  ParOptVec*,
                                                                  ParOptVec*) ){
  addsparsejacobiantranspose = func;
}

void CyParOptProblem::setAddSparseInnerProduct( void (*func)(void *,
                                                             int, int, int,
                                                             ParOptScalar,
                                                             ParOptVec*,
                                                             ParOptVec*,
                                                             ParOptScalar*) ){
  addsparseinnerproduct = func;
}

/*
  Get the variables and bounds from the problem
*/
void CyParOptProblem::getVarsAndBounds( ParOptVec *x,
                                        ParOptVec *lb,
                                        ParOptVec *ub ){
  if (!getvarsandbounds){
    fprintf(stderr, "getvarsandbounds callback not defined\n");
    return;
  }
  getvarsandbounds(self, nvars, x, lb, ub);
}

/*
  Evaluate the objective and constraints
*/
int CyParOptProblem::evalObjCon( ParOptVec *x,
                                 ParOptScalar *fobj,
                                 ParOptScalar *cons ){
  if (!evalobjcon){
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
int CyParOptProblem::evalObjConGradient( ParOptVec *x,
                                         ParOptVec *g,
                                         ParOptVec **Ac ){
  if (!evalobjcongradient){
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
int CyParOptProblem::evalHvecProduct( ParOptVec *x,
                                      ParOptScalar *z,
                                      ParOptVec *zw,
                                      ParOptVec *px,
                                      ParOptVec *hvec ){
  if (!evalhvecproduct){
    fprintf(stderr, "evalhvecproduct callback not defined\n");
    return 1;
  }

  // Evaluate the Hessian-vector callback
  int fail = evalhvecproduct(self, nvars, ncon, nwcon,
                             x, z, zw, px, hvec);
  return fail;
}

/*
  Evaluate the diagonal of the Hessian
*/
int CyParOptProblem::evalHessianDiag( ParOptVec *x,
                                      ParOptScalar *z, ParOptVec *zw,
                                      ParOptVec *hdiag ){
  if (!evalhessiandiag){
    fprintf(stderr, "evalhessiandiag callback not defined\n");
    return 1;
  }

  // Evaluate the Hessian-vector callback
  int fail = evalhessiandiag(self, nvars, ncon, nwcon,
                             x, z, zw, hdiag);
  return fail;
}

/*
  Apply the quasi-Newton update correction
*/
void CyParOptProblem::computeQuasiNewtonUpdateCorrection( ParOptVec *x,
                                                          ParOptScalar *z,
                                                          ParOptVec *zw,
                                                          ParOptVec *s,
                                                          ParOptVec *y ){
  if (!computequasinewtonupdatecorrection){
    fprintf(stderr, "computequasinewtonupdatecorrection callback not defined\n");
  }

  // Evaluate the Hessian-vector callback
  computequasinewtonupdatecorrection(self, nvars, ncon, x, z, zw, s, y);
}

/*
  Evaluate the constraints
*/
void CyParOptProblem::evalSparseCon( ParOptVec *x,
                                     ParOptVec *out ){
  if (!evalsparsecon){
    fprintf(stderr, "evalsparsecon callback not defined\n");
    return;
  }

  // Evaluate the Hessian-vector callback
  evalsparsecon(self, nvars, nwcon, x, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void CyParOptProblem::addSparseJacobian( ParOptScalar alpha,
                                         ParOptVec *x,
                                         ParOptVec *px,
                                         ParOptVec *out ){
  if (!addsparsejacobian){
    fprintf(stderr, "addsparsejacobian callback not defined\n");
    return;
  }

  // Evaluate the sparse Jacobian output
  addsparsejacobian(self, nvars, nwcon, alpha, x, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void CyParOptProblem::addSparseJacobianTranspose( ParOptScalar alpha,
                                                  ParOptVec *x,
                                                  ParOptVec *pzw,
                                                  ParOptVec *out ){
  if (!addsparsejacobiantranspose){
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
void CyParOptProblem::addSparseInnerProduct( ParOptScalar alpha,
                                             ParOptVec *x,
                                             ParOptVec *cvec,
                                             ParOptScalar *A ){
  if (!addsparseinnerproduct){
    fprintf(stderr, "addsparseinnerproduct callback not defined\n");
    return;
  }

  // Evaluate the sparse Jacobian output
  addsparseinnerproduct(self, nvars, nwcon, nwblock, alpha, x, cvec, A);
}
