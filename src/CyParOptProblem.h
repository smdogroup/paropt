#ifndef PAR_OPT_CYTHON_PROBLEM_H
#define PAR_OPT_CYTHON_PROBLEM_H

#include "ParOptProblem.h"

/**
  This code implements a simplifed interface for the ParOptProblem
  that can be wrapped using Cython. The class uses function pointers
  as an intermediate layer between the ParOptProblem class and the
  python layer. These callback functions can then be wrapped and set
  from python.

  Note: The class is currently restricted to utilize only the
  objective and gradient calculations and not the Hessian code. In
  addition, the sparse constraints are not implemented. This class
  then solves the (distributed) optimization problem:

  min    f(x)
  w.r.t. lb <= x <= ub
  s.t.   c(x) >= 0
         cw(x) >= 0
*/
class CyParOptProblem : public ParOptProblem {
 public:
  CyParOptProblem( MPI_Comm _comm,
                   int _nvars, int _ncon,
                   int _ninequality,
                   int _nwcon, int _nwblock );
  ~CyParOptProblem();

  // Set options associated with the inequality constraints
  // ------------------------------------------------------
  void setInequalityOptions( int _isSparseInequal,
                             int _useLower, int _useUpper );

  // Function to indicate the type of sparse constraints
  // ---------------------------------------------------
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Set the member callback functions that are required
  // ---------------------------------------------------
  void setSelfPointer( void *_self );
  void setGetVarsAndBounds( void (*func)(void*, int, ParOptVec*,
                                         ParOptVec*, ParOptVec*) );
  void setEvalObjCon( int (*func)(void*, int, int, ParOptVec*,
                                  ParOptScalar*, ParOptScalar*) );
  void setEvalObjConGradient( int (*func)(void*, int, int, ParOptVec*,
                                          ParOptVec*, ParOptVec**) );
  void setEvalHvecProduct( int (*func)(void*, int, int, int,
                                       ParOptVec*, ParOptScalar*,
                                       ParOptVec*, ParOptVec*,
                                       ParOptVec*) );
  void setEvalHessianDiag( int (*func)(void*, int, int, int, ParOptVec*,
                                       ParOptScalar*, ParOptVec*,
                                       ParOptVec*) );
  void setComputeQuasiNewtonUpdateCorrection( void (*func)(void*, int, int,
                                                           ParOptVec*,
                                                           ParOptScalar*,
                                                           ParOptVec*,
                                                           ParOptVec*,
                                                           ParOptVec*) );
  void setEvalSparseCon( void (*func)(void*, int, int,
                                      ParOptVec*, ParOptVec*) );
  void setAddSparseJacobian( void (*func)(void*, int, int,
                                          ParOptScalar, ParOptVec*,
                                          ParOptVec*, ParOptVec*) );
  void setAddSparseJacobianTranspose( void (*func)(void*, int, int,
                                                   ParOptScalar,
                                                   ParOptVec*,
                                                   ParOptVec*,
                                                   ParOptVec*) );
  void setAddSparseInnerProduct( void (*func)(void*, int, int, int,
                                              ParOptScalar,
                                              ParOptVec*,
                                              ParOptVec*,
                                              ParOptScalar*) );

  // Get the variables and bounds from the problem
  // ---------------------------------------------
  void getVarsAndBounds( ParOptVec *x, ParOptVec *lb,
                         ParOptVec *ub );

  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *x, ParOptScalar *fobj,
                  ParOptScalar *cons );

  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *x,
                          ParOptVec *g, ParOptVec **Ac );

  // Evaluate the product of the Hessian with a given vector
  // -------------------------------------------------------
  int evalHvecProduct( ParOptVec *x,
                       ParOptScalar *z, ParOptVec *zw,
                       ParOptVec *px, ParOptVec *hvec );

  // Evaluate the diagonal of the Hessian
  // ------------------------------------
  int evalHessianDiag( ParOptVec *x,
                       ParOptScalar *z, ParOptVec *zw,
                       ParOptVec *hdiag );

  // Compute a quasi-Newton update correction/modification
  // -----------------------------------------------------
  void computeQuasiNewtonUpdateCorrection( ParOptVec *x,
                                           ParOptScalar *z, ParOptVec *zw,
                                           ParOptVec *s, ParOptVec*y );

  // Evaluate the constraints
  // ------------------------
  void evalSparseCon( ParOptVec *x, ParOptVec *out );

  // Compute the Jacobian-vector product out = J(x)*px
  // --------------------------------------------------
  void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                          ParOptVec *px, ParOptVec *out );

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                   ParOptVec *pzw, ParOptVec *out );

  // Add the inner product of the constraints to the matrix such
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  // ------------------------------------------------------------
  void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                              ParOptVec *cvec, ParOptScalar *A );

 private:
  // Public member function pointers to the callbacks that are
  // required before class can be used
  // ---------------------------------------------------------
  void *self;
  void (*getvarsandbounds)( void *self, int nvars,
                            ParOptVec *x, ParOptVec *lb,
                            ParOptVec *ub );
  int (*evalobjcon)( void *self, int nvars, int ncon,
                     ParOptVec *x, ParOptScalar *fobj,
                     ParOptScalar *cons );
  int (*evalobjcongradient)( void *self, int nvars, int ncon,
                             ParOptVec *x, ParOptVec *gobj,
                             ParOptVec **A );
  int (*evalhvecproduct)( void *self, int nvars, int ncon, int nwcon,
                          ParOptVec *x, ParOptScalar *z,
                          ParOptVec *zw, ParOptVec *px,
                          ParOptVec *hvec );
  int (*evalhessiandiag)( void *self, int nvars, int ncon, int nwcon,
                          ParOptVec *x, ParOptScalar *z,
                          ParOptVec *zw, ParOptVec *hdiag );
  void (*computequasinewtonupdatecorrection)( void *self, int, int,
                                              ParOptVec *x, ParOptScalar *z,
                                              ParOptVec *zw,
                                              ParOptVec *s, ParOptVec*y );
  void (*evalsparsecon)( void *self, int nvars, int nwcon,
                         ParOptVec *x, ParOptVec *out );
  void (*addsparsejacobian)( void *self, int nvars, int nwcon,
                             ParOptScalar alpha, ParOptVec *x,
                             ParOptVec *px, ParOptVec *out );
  void (*addsparsejacobiantranspose)( void *self, int nvars, int nwcon,
                                      ParOptScalar alpha, ParOptVec *x,
                                      ParOptVec *px, ParOptVec *out );
  void (*addsparseinnerproduct)( void *self,
                                 int nvars, int nwcon, int nwblock,
                                 ParOptScalar alpha, ParOptVec *x,
                                 ParOptVec *c, ParOptScalar *A );

  // Store information about the type of problem to solve
  int isSparseInequal;
  int useLower;
  int useUpper;
};

#endif // PAR_OPT_CYTHON_PROBLEM_H
