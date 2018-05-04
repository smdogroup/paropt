#ifndef PAR_OPT_TRUST_REGION_H
#define PAR_OPT_TRUST_REGION_H

#include "ParOpt.h"

/*
  Trust Region method
*/
class ParOptTrustRegion : public ParOptProblem {
 public:
  ParOptTrustRegion( ParOptProblem *_prob, 
                     ParOptCompactQuasiNewton *_qn, double _tr_size,
                     double _tr_min_size, double _tr_max_size,
                     double _eta=0.25, double penalty_value=10.0,
                     double _bound_relax=1e-4 );
  ~ParOptTrustRegion();

  // Initialize the subproblem
  void initialize();

  // Update the problem
  void update( ParOptVec *xt, const ParOptScalar *z, ParOptVec *zw,
               double *infeas, double *l1, double *linfty );

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
  void writeOutput( int iter, ParOptVec *x ){}

 private:
  // Set the trust region bounds
  void setTrustRegionBounds( double tr, ParOptVec *x,
                             ParOptVec *l, ParOptVec *u,
                             ParOptVec *ltr, ParOptVec *utr );

  // Compute the KKT error in the solution
  void computeKKTError( ParOptVec *xt, ParOptVec *g, ParOptVec **A,
                        const ParOptScalar *z, ParOptVec *zw,
                        double *l1, double *linfty );

  // File pointer for the summary file - depending on the settings
  // FILE *fp;
  // int first_print;

  int n; // The number of design variables (local)
  int m; // The number of dense constraints (global)

  // Set the parameters
  double tr_size;
  double tr_min_size, tr_max_size;
  double eta;
  double penalty_value;
  double bound_relax;

  // Pointer to the optimization problem
  ParOptProblem *prob;

  // Communicator for this problem
  MPI_Comm comm;

  // Set the quadratic model parameters for this problem
  ParOptCompactQuasiNewton *qn;

  // Lower/upper bounds for the original problem
  ParOptVec *lb, *ub;

  // Lower/upper bounds for the trust region problem (these lie within
  // the problem lower/upper bounds)
  ParOptVec *lk, *uk;

  // Current design point
  ParOptVec *xk;

  // The objective/constraint values
  ParOptScalar fk, *ck;

  // Current objective and constraint gradients
  ParOptVec *gk;
  ParOptVec **Ak;

  // Temporary objective/constraint values and their gradients
  ParOptScalar ft, *ct;
  ParOptVec *gt;
  ParOptVec **At;

  // Temporary vectors
  ParOptVec *s, *t;
};

#endif // PAR_OPT_TRUST_REGION_H
