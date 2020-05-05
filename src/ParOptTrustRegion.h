#ifndef PAR_OPT_TRUST_REGION_H
#define PAR_OPT_TRUST_REGION_H

#include "ParOptInteriorPoint.h"

/*
  This class defines the trust region subproblem interface.

  The subproblem class is a ParOptProblem class that has additional
  member functions that enable it to be updated from point to point
  within the trust-region method.
*/
class ParOptTrustRegionSubproblem : public ParOptProblem {
 public:
  ParOptTrustRegionSubproblem( MPI_Comm comm ):
    ParOptProblem(comm){}

  /**
    Return the compact quasi-Newton Hessian approximation

    @return The compact quasi-Newton object
  */
  virtual ParOptCompactQuasiNewton* getQuasiNewton() = 0;

  /**
    Initialize the sub-problem at the problem starting point

    @param tr_size The trust-region radius at the starting point
  */
  virtual void initModelAndBounds( double tr_size ) = 0;

  /**
    Set the trust region radius about the current point

    @param tr_size The trust-region radius at the starting point
  */
  virtual void setTrustRegionBounds( double tr_size ) = 0;

  /**
    Evaluate the objective and constraints (and often their gradients)
    at the specified trial point and update the model.

    @param xt The trial point location
    @param z The multipliers for the dense constraints
    @param zw The multipliers for the sparse constraints
    @param fobj The objective value at the trial point
    @param cons The dense constraint values at the trial point
    @return Flag indicating whether the objective evaluation failed
  */
  virtual int evalTrialPointAndUpdate( ParOptVec *xt,
                                       const ParOptScalar *z,
                                       ParOptVec *zw,
                                       ParOptScalar *fobj,
                                       ParOptScalar *cons ) = 0;

  /**
    Accept the trial point and use this point as the base point
    for the next model step

    @param xt The trial point location
    @param z The multipliers for the dense constraints
    @param zw The multipliers for the sparse constraints
    @return Flag indicating whether the objective evaluation failed
  */
  virtual int acceptTrialPoint( ParOptVec *xt,
                                const ParOptScalar *z,
                                ParOptVec *zw ) = 0;

  /**
    The trial step is rejected.
  */
  virtual void rejectTrialPoint() = 0;

  /**
    Get access to a linearization of the model

    @param _xk The base point
    @param _gk The gradient of the objective
    @param _Ak The gradient of the constraints
    @return The number of constraints
  */
  virtual int getLinearModel( ParOptVec **_xk=NULL, ParOptVec **_gk=NULL,
                              ParOptVec ***_Ak=NULL,
                              ParOptVec **_lb=NULL, ParOptVec **_ub=NULL ) = 0;
};

/*
  This class defines the quadratic subproblem for the trust-region method
*/
class ParOptQuadraticSubproblem : public ParOptTrustRegionSubproblem {
 public:
  ParOptQuadraticSubproblem( ParOptProblem *_problem,
                             ParOptCompactQuasiNewton *_qn );
  ~ParOptQuadraticSubproblem();

  // Implementation for the trust-region specific functions
  ParOptCompactQuasiNewton* getQuasiNewton();
  void initModelAndBounds( double tr_size );
  void setTrustRegionBounds( double tr_size );
  int evalTrialPointAndUpdate( ParOptVec *xt, const ParOptScalar *z,
                               ParOptVec *zw,
                               ParOptScalar *fobj, ParOptScalar *cons );
  int acceptTrialPoint( ParOptVec *xt, const ParOptScalar *z, ParOptVec *zw );
  void rejectTrialPoint();

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
  void writeOutput( int iter, ParOptVec *x ){
    prob->writeOutput(iter, x);
  }

  int getLinearModel( ParOptVec **_xk=NULL, ParOptVec **_gk=NULL,
                      ParOptVec ***_Ak=NULL,
                      ParOptVec **_lb=NULL, ParOptVec **_ub=NULL );

 private:
  // Pointer to the optimization problem
  ParOptProblem *prob;

  // Set the quadratic model parameters for this problem
  ParOptCompactQuasiNewton *qn;

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

/*
  ParOptTrustRegion implements a trust-region method
*/
class ParOptTrustRegion : public ParOptBase {
 public:
  ParOptTrustRegion( ParOptTrustRegionSubproblem *_subproblem,
                     double _tr_size,
                     double _tr_min_size, double _tr_max_size,
                     double _eta=0.25, double penalty_value=10.0,
                     double _bound_relax=1e-4 );
  ~ParOptTrustRegion();

  // Initialize the subproblem
  void initialize();

  // Update the problem
  void update( ParOptVec *xt, const ParOptScalar *z, ParOptVec *zw,
               double *infeas, double *l1, double *linfty );

  // Set parameters for the trust region method
  void setAdaptiveGammaUpdate( int truth );
  void setMaxTrustRegionIterations( int max_iters );
  void setTrustRegionTolerances( double _infeas_tol,
                                 double _l1_tol, double _linfty_tol );
  void setPenaltyGamma( double gamma );
  void setPenaltyGamma( const double *gamma );
  int getPenaltyGamma( const double **gamma );
  void setPenaltyGammaMax( double _gamma_max );
  void setOutputFrequency( int _write_output_frequency );

  // Optimization loop using the trust region subproblem
  void optimize( ParOptInteriorPoint *optimize );

  // Set the output file (only on the root proc)
  void setOutputFile( const char *filename );
  void setPrintLevel( int _print_level );

 protected:
  ParOptTrustRegionSubproblem *subproblem;

  // Compute the KKT error in the solution
  void computeKKTError( const ParOptScalar *z, ParOptVec *zw,
                        double *l1, double *linfty );

 private:
  // Print the options summary
  void printOptionSummary( FILE *fp );

  // File pointer for the summary file - depending on the settings
  FILE *fp;
  int iter_count;
  int print_level;

  int n; // The number of design variables (local)
  int m; // The number of dense constraints (global)
  int nwcon; // The number of sparse constraints
  int nwblock; // The block size

  // Set the parameters
  double tr_size;
  double tr_min_size, tr_max_size;
  double eta;
  double bound_relax;

  // Store the function precision
  double function_precision;

  // Control the adaptive penalty update
  int adaptive_gamma_update;
  double *penalty_gamma;
  double penalty_gamma_max;

  // Set the output parameters
  int write_output_frequency;

  // Set the trust region solution parameters
  int max_tr_iterations;
  double l1_tol, linfty_tol;
  double infeas_tol;

  // Temporary vectors
  ParOptVec *t;
};

#endif // PAR_OPT_TRUST_REGION_H
