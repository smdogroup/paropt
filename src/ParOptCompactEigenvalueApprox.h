#ifndef PAROPT_COMPACT_EIGENVALUE_APPROX_H
#define PAROPT_COMPACT_EIGENVALUE_APPROX_H

#include "ParOptTrustRegion.h"
#include "ParOptQuasiNewton.h"

class ParOptCompactEigenApprox : public ParOptBase {
 public:
  ParOptCompactEigenApprox( ParOptProblem *problem,
                            int _N );
  ~ParOptCompactEigenApprox();

  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y );
  void getApproximation( ParOptScalar **_c0, ParOptVec **_g0,
                         int *_N, ParOptScalar **_M, ParOptScalar **_Minv,
                         ParOptVec ***_hvecs );
  ParOptScalar evalApproximation( ParOptVec *s, ParOptVec *t );
  void evalApproximationGradient( ParOptVec *s, ParOptVec *grad );

 private:
  // The constraint value and gradient
  ParOptScalar c0;
  ParOptVec *g0;

  // The Hessian approximation of the constraint
  int N;
  ParOptScalar *M;
  ParOptScalar *Minv;
  ParOptVec **hvecs;

  // Temporary vector for matrix-vector products
  ParOptScalar *tmp;
};

class ParOptEigenQuasiNewton : public ParOptCompactQuasiNewton {
 public:
  ParOptEigenQuasiNewton( ParOptCompactQuasiNewton *_qn,
                          ParOptCompactEigenApprox *_eigh,
                          int _index=0 );
  ~ParOptEigenQuasiNewton();

  // Reset the internal data
  void reset();

  // In this case, the quasi-Newton update is not performed here.
  // The quasi-Newton update will be performed directly on the
  // quasi-Newton object itself.
  int update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw,
              ParOptVec *s, ParOptVec *y );
  int update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw );

  // Perform a matrix-vector multiplication
  void mult( ParOptVec *x, ParOptVec *y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y );

  // Get the compact representation of the quasi-Newton method
  int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                     const ParOptScalar **_M, ParOptVec ***Z );

  // Get the maximum size of the limited-memory BFGS
  int getMaxLimitedMemorySize();

  // Get the compact eigenvalue approximation
  ParOptCompactQuasiNewton *getCompactQuasiNewton();
  ParOptCompactEigenApprox *getCompactEigenApprox();
  int getMultiplierIndex();

 private:
  // The two contributions to the Hessian of the Lagrangian
  int index;
  ParOptScalar z0;
  ParOptCompactQuasiNewton *qn;
  ParOptCompactEigenApprox *eigh;

  // Objects to store the vectors
  int max_vecs;
  ParOptScalar *M, *d;
  ParOptVec **Z;
};

class ParOptEigenSubproblem : public ParOptTrustRegionSubproblem {
 public:
  ParOptEigenSubproblem( ParOptProblem *_problem,
                         ParOptEigenQuasiNewton *_qn );
  ~ParOptEigenSubproblem();

  // Set the update function for the eigenvalue approximation
  void setEigenModelUpdate( void *data,
                            void (*update)(void*, ParOptVec*,
                                           ParOptCompactEigenApprox*) );

  // Implementation for the trust-region specific functions
  ParOptCompactQuasiNewton* getQuasiNewton();
  void initModelAndBounds( double tr_size );
  void setTrustRegionBounds( double tr_size );
  int evalTrialPointAndUpdate( ParOptVec *xt, const ParOptScalar *z,
                               ParOptVec *zw,
                               ParOptScalar *fobj, ParOptScalar *cons );
  int acceptTrialPoint( ParOptVec *xt, const ParOptScalar *z, ParOptVec *zw );
  void rejectTrialPoint();
  int getQuasiNewtonUpdateType();

  // Create the design vectors
  ParOptVec *createDesignVec();
  ParOptVec *createConstraintVec();

  // Get the communicator for the problem
  MPI_Comm getMPIComm();

  // Function to indicate the type of sparse constraints
  int isDenseInequality();
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Get the variables and bounds from the problem
  void getVarsAndBounds( ParOptVec *x, ParOptVec *lb, ParOptVec *ub );

  // Evaluate the objective and constraints
  int evalObjCon( ParOptVec *x, ParOptScalar *fobj, ParOptScalar *cons );

  // Evaluate the objective and constraint gradients
  int evalObjConGradient( ParOptVec *x, ParOptVec *g, ParOptVec **Ac );

  // Evaluate the product of the Hessian with a given vector
  int evalHvecProduct( ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                       ParOptVec *px, ParOptVec *hvec ){
    return 0;
  }

  // Evaluate the diagonal Hessian
  int evalHessianDiag( ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                       ParOptVec *hdiag ){
    return 0;
  }

  // Evaluate the constraints
  void evalSparseCon( ParOptVec *x, ParOptVec *out );

  // Compute the Jacobian-vector product out = J(x)*px
  void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                          ParOptVec *px, ParOptVec *out );

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                   ParOptVec *pzw, ParOptVec *out );

  // Add the inner product of the constraints to the matrix such
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                              ParOptVec *cvec, ParOptScalar *A );

  // Over-write this function if you'd like to print out
  // something with the same frequency as the output files
  void writeOutput( int iter, ParOptVec *x );

  int getLinearModel( ParOptVec **_xk=NULL, ParOptVec **_gk=NULL,
                      ParOptVec ***_Ak=NULL,
                      ParOptVec **_lb=NULL, ParOptVec **_ub=NULL );

 private:
  void *data;
  void (*updateEigenModel)( void*, ParOptVec*, ParOptCompactEigenApprox* );

  // Pointer to the optimization problem
  ParOptProblem *prob;

  // Set the quadratic model parameters for this problem
  ParOptEigenQuasiNewton *approx;
  int qn_update_type;

  int n; // The number of design variables (local)
  int m; // The number of dense constraints (global)

  // Lower/upper bounds for the original problem
  ParOptVec *lb, *ub;

  // Lower/upper bounds for the trust region problem (these lie within
  // the problem lower/upper bounds)
  ParOptVec *lk, *uk;

  // Current design point
  ParOptVec *xk;

  // Current objective and constraint values and gradients
  ParOptScalar fk, *ck;
  ParOptVec *gk;
  ParOptVec **Ak;

  // Temporary objective/constraint values and their gradients
  ParOptScalar ft, *ct;
  ParOptVec *gt;
  ParOptVec **At;

  // Temporary vectors
  ParOptVec *s, *t;
};

#endif // PAROPT_COMPACT_EIGENVALUE_APPROX_H
