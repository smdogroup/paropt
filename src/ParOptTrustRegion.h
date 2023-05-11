#ifndef PAR_OPT_TRUST_REGION_H
#define PAR_OPT_TRUST_REGION_H

#include <list>

#include "ParOptInteriorPoint.h"

/*
  This class defines the trust region subproblem interface.

  The subproblem class is a ParOptProblem class that has additional
  member functions that enable it to be updated from point to point
  within the trust-region method.
*/
class ParOptTrustRegionSubproblem : public ParOptProblem {
 public:
  ParOptTrustRegionSubproblem(MPI_Comm comm) : ParOptProblem(comm) {}

  /**
    Return the compact quasi-Newton Hessian approximation

    @return The compact quasi-Newton object
  */
  virtual ParOptCompactQuasiNewton *getQuasiNewton() = 0;

  /**
    Initialize the sub-problem at the problem starting point

    @param tr_size The trust-region radius at the starting point
  */
  virtual void initModelAndBounds(double tr_size) = 0;

  /**
    Set the trust region radius about the current point

    @param tr_size The trust-region radius at the starting point
  */
  virtual void setTrustRegionBounds(double tr_size) = 0;

  /**
    Evaluate the objective and constraints (and often their gradients)
    at the specified trial point and update the model.

    @param update_flag Perform a quasi-Newton update (or model update) or not
    @param step The trial step
    @param z The multipliers for the dense constraints
    @param zw The multipliers for the sparse constraints
    @param fobj The objective value at the trial point
    @param cons The dense constraint values at the trial point
    @return Flag indicating whether the objective evaluation failed
  */
  virtual int evalTrialStepAndUpdate(int update_flag, ParOptVec *step,
                                     ParOptScalar *z, ParOptVec *zw,
                                     ParOptScalar *fobj,
                                     ParOptScalar *cons) = 0;

  /**
    Accept the trial point and use this point as the base point
    for the next model step

    @param step The trial step
    @param z The multipliers for the dense constraints
    @param zw The multipliers for the sparse constraints
    @return Flag indicating whether the objective evaluation failed
  */
  virtual int acceptTrialStep(ParOptVec *xt, ParOptScalar *z,
                              ParOptVec *zw) = 0;

  /**
    The trial step is rejected.
  */
  virtual void rejectTrialStep() = 0;

  /**
    Get the Hessian update type from the most recent update

    @return The quasi-Newton update type
  */
  virtual int getQuasiNewtonUpdateType() { return 0; }

  /**
    Get access to a linearization of the model

    @param xk The base point
    @param fk The objective function value
    @param gk The gradient of the objective
    @param ck The constraint values
    @param Ak The gradient of the constraints
    @param lb The lower bounds
    @param ub The upper bounds
    @return The number of constraints
  */
  virtual int getLinearModel(ParOptVec **_xk = NULL, ParOptScalar *fk = NULL,
                             ParOptVec **gk = NULL,
                             const ParOptScalar **ck = NULL,
                             ParOptVec ***Ak = NULL, ParOptVec **lb = NULL,
                             ParOptVec **ub = NULL) = 0;

  /**
    Switch on second order correction. This function sets flag is_soc_step to 1,
    and needs to be called right before performing optimization for second order
    correction.
  */
  virtual void startSecondOrderCorrection() { return; }

  /**
    Switch off second order correction. This function sets flag is_soc_step to
    0, and needs to be called right after performing optimization for second
    order correction.
  */
  virtual void endSecondOrderCorrection() { return; }

  /**
    Update second order correction constraints. When performing second order
    correction, a modified quadratic programming problem is solved by re-using
    the trust region subproblem optimizer object. This helper function updates
    the problem formulation to prepare for the second order correction step.

    @param xt [in] The candidate design point
    @param ct [in] The constraint values at candidate design point
  */
  virtual void updateSocCon(ParOptVec *xt, ParOptScalar *ct) { return; }

  /**
    Evaluate (f, h) at the second order correction trial point xt

    @param xt [in] The candidate design point
    @param soc_use_quad_model [in] decide whether to use quadratic model or
    original problem for function and constraint evaluation
    @param f [out] function value at candidate design point
    @param h [out] constraint violation value at candidate design point
  */
  virtual int evalSocTrialPoint(ParOptVec *xt, int soc_use_quad_model,
                                ParOptScalar *f, ParOptScalar *h) {
    return 0;
  }

  /**
    Evaluate gradients at the second order correction trial point xt

    @param xt [in] The candidate design point
    @param soc_use_quad_model [in] decide whether to use quadratic model or
    original problem for function and constraint evaluation
  */
  virtual int evalSocTrialGrad(ParOptVec *xt, int soc_use_quad_model) {
    return 0;
  }
};

/*
  This class defines the quadratic subproblem for the trust-region method
*/
class ParOptQuadraticSubproblem : public ParOptTrustRegionSubproblem {
 public:
  ParOptQuadraticSubproblem(ParOptProblem *_problem,
                            ParOptCompactQuasiNewton *_qn);
  ~ParOptQuadraticSubproblem();

  // Implementation for the trust-region specific functions
  ParOptCompactQuasiNewton *getQuasiNewton();
  void initModelAndBounds(double tr_size);
  void setTrustRegionBounds(double tr_size);
  int evalTrialStepAndUpdate(int update_flag, ParOptVec *step, ParOptScalar *z,
                             ParOptVec *zw, ParOptScalar *fobj,
                             ParOptScalar *cons);
  int acceptTrialStep(ParOptVec *xt, ParOptScalar *z, ParOptVec *zw);
  void rejectTrialStep();
  int getQuasiNewtonUpdateType();

  // Create the design vectors
  ParOptVec *createDesignVec();
  ParOptVec *createConstraintVec();
  ParOptQuasiDefMat *createQuasiDefMat();

  // Get the communicator for the problem
  MPI_Comm getMPIComm();

  // Function to indicate the type of sparse constraints
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Get the variables and bounds from the problem
  void getVarsAndBounds(ParOptVec *x, ParOptVec *lb, ParOptVec *ub);

  // Evaluate the objective and constraints
  int evalObjCon(ParOptVec *x, ParOptScalar *fobj, ParOptScalar *cons);

  // Evaluate the objective and constraint gradients
  int evalObjConGradient(ParOptVec *x, ParOptVec *g, ParOptVec **Ac);

  // Evaluate the product of the Hessian with a given vector
  int evalHvecProduct(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *px, ParOptVec *hvec) {
    return 0;
  }

  // Evaluate the diagonal Hessian
  int evalHessianDiag(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *hdiag) {
    return 0;
  }

  // Evaluate the constraints
  void evalSparseCon(ParOptVec *x, ParOptVec *out);

  // Compute the Jacobian-vector product out = J(x)*px
  void addSparseJacobian(ParOptScalar alpha, ParOptVec *x, ParOptVec *px,
                         ParOptVec *out);

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  void addSparseJacobianTranspose(ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *pzw, ParOptVec *out);

  // Add the inner product of the constraints to the matrix such
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  void addSparseInnerProduct(ParOptScalar alpha, ParOptVec *x, ParOptVec *cvec,
                             ParOptScalar *A);

  // Over-write this function if you'd like to print out
  // something with the same frequency as the output files
  void writeOutput(int iter, ParOptVec *x) { prob->writeOutput(iter, x); }

  int getLinearModel(ParOptVec **_xk = NULL, ParOptScalar *_fk = NULL,
                     ParOptVec **_gk = NULL, const ParOptScalar **_ck = NULL,
                     ParOptVec ***_Ak = NULL, ParOptVec **_lb = NULL,
                     ParOptVec **_ub = NULL);

  // Call this function before soc iteration loop
  void startSecondOrderCorrection() { is_soc_step = 1; }

  // Call this function after soc iteration loop
  void endSecondOrderCorrection() { is_soc_step = 0; }

  // Update second order correction constraints
  void updateSocCon(ParOptVec *step, ParOptScalar *ct) {
    for (int i = 0; i < m; i++) {
      c_soc[i] = ct[i] - Ak[i]->dot(step);
    }
  }

  // Evaluate (f, h) at the second order correction trial point xt
  int evalSocTrialPoint(ParOptVec *xt, int soc_use_quad_model, ParOptScalar *f,
                        ParOptScalar *h);

  // Evaluate gradients at the second order correction trial point xt
  int evalSocTrialGrad(ParOptVec *xt, int soc_use_quad_model);

 private:
  // Pointer to the optimization problem
  ParOptProblem *prob;

  // Set the quadratic model parameters for this problem
  ParOptCompactQuasiNewton *qn;
  int qn_update_type;

  int n;  // The number of design variables (local)
  int m;  // The number of dense constraints (global)

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

  // Temoprary constraint values used for second order correction
  ParOptScalar *c_soc;

  // Temporary vectors
  ParOptVec *t, *xtemp;

  // Variable indicating if second order correction is used
  int is_soc_step;
};

/*
  Infeasible subproblem class
*/
class ParOptInfeasSubproblem : public ParOptProblem {
 public:
  // Set the objective type
  static const int PAROPT_SUBPROBLEM_OBJECTIVE = 1;
  static const int PAROPT_LINEAR_OBJECTIVE = 2;
  static const int PAROPT_CONSTANT_OBJECTIVE = 3;

  // Set the constraint type
  static const int PAROPT_SUBPROBLEM_CONSTRAINT = 1;
  static const int PAROPT_LINEAR_CONSTRAINT = 2;

  ParOptInfeasSubproblem(ParOptTrustRegionSubproblem *_prob,
                         int subproblem_objective, int subproblem_constraint);
  ~ParOptInfeasSubproblem();

  // Set the objective scaling
  void setObjectiveScaling(ParOptScalar _scale) { obj_scale = _scale; }

  // Create the design vectors
  ParOptVec *createDesignVec();
  ParOptVec *createConstraintVec();
  ParOptQuasiDefMat *createQuasiDefMat();

  // Get the communicator for the problem
  MPI_Comm getMPIComm();

  // Function to indicate the type of sparse constraints
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Get the variables and bounds from the problem
  void getVarsAndBounds(ParOptVec *x, ParOptVec *lb, ParOptVec *ub);

  // Evaluate the objective and constraints
  int evalObjCon(ParOptVec *x, ParOptScalar *fobj, ParOptScalar *cons);

  // Evaluate the objective and constraint gradients
  int evalObjConGradient(ParOptVec *x, ParOptVec *g, ParOptVec **Ac);

  // Evaluate the product of the Hessian with a given vector
  int evalHvecProduct(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *px, ParOptVec *hvec) {
    return 0;
  }

  // Evaluate the diagonal Hessian
  int evalHessianDiag(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *hdiag) {
    return 0;
  }

  // Evaluate the constraints
  void evalSparseCon(ParOptVec *x, ParOptVec *out);

  // Compute the Jacobian-vector product out = J(x)*px
  void addSparseJacobian(ParOptScalar alpha, ParOptVec *x, ParOptVec *px,
                         ParOptVec *out);

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  void addSparseJacobianTranspose(ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *pzw, ParOptVec *out);

  // Add the inner product of the constraints to the matrix such
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  void addSparseInnerProduct(ParOptScalar alpha, ParOptVec *x, ParOptVec *cvec,
                             ParOptScalar *A);

 private:
  ParOptScalar obj_scale;  // Objective function scaling factor

  int n;  // The number of design variables (local)
  int m;  // The number of dense constraints (global)
  ParOptTrustRegionSubproblem *prob;

  // Set the type of subproblem to solve
  int subproblem_objective;   // The subproblem objective type
  int subproblem_constraint;  // The constraint type
};

/*
  ParOptTrustRegion implements a trust-region method
*/
class ParOptTrustRegion : public ParOptBase {
 public:
  ParOptTrustRegion(ParOptTrustRegionSubproblem *_subproblem,
                    ParOptOptions *_options = NULL);
  ~ParOptTrustRegion();

  // Get the default option values
  static void addDefaultOptions(ParOptOptions *options);
  ParOptOptions *getOptions();

  // Initialize the subproblem
  void initialize();

  // Set parameters for the trust region method
  void setPenaltyGamma(double gamma);
  void setPenaltyGamma(const double *gamma);
  int getPenaltyGamma(const double **gamma);
  void setPenaltyGammaMax(double _gamma_max);
  void setPenaltyGammaMin(double _gamma_min);

  // Optimization loop using the trust region subproblem
  void optimize(ParOptInteriorPoint *optimizer);

  // Get the optimized point
  void getOptimizedPoint(ParOptVec **_x);

 private:
  // The trust region optimization subproblem
  ParOptTrustRegionSubproblem *subproblem;

  // The options object for the trust-region method
  ParOptOptions *options;

  // Solve the subproblem using SL1QP method
  void sl1qpUpdate(ParOptVec *step, ParOptScalar *z, ParOptVec *zw,
                   double *infeas, double *l1, double *linfty);

  // Optimization-specific code for the filter and sl1qp strategies
  void filterOptimize(ParOptInteriorPoint *optimizer);
  void sl1qpOptimize(ParOptInteriorPoint *optimizer);

  // Minimize the infeasibility, this can be used for either
  // adaptive penalty gamma update or in the filterSQP method
  void minimizeInfeas(ParOptInteriorPoint *optimizer,
                      ParOptInfeasSubproblem *infeas_problem,
                      ParOptOptions *ip_options, ParOptVec *step,
                      ParOptScalar *best_con_infeas,
                      int infeas_objective_type_flag,
                      int infeas_constraint_type_flag);

  // Perform second order correction if trial step is rejected,
  // note that this is only useful for filterOptimize function
  int isAcceptedBySoc(ParOptInteriorPoint *optimizer, ParOptVec *step,
                      ParOptScalar *fobj_trial, ParOptScalar *con_trial,
                      int *niters, ParOptScalar *r);

  // Test that if (f_new, h_new) is acceptable by pair (f_old, h_old)
  int acceptableByPair(ParOptScalar f_new, ParOptScalar h_new,
                       ParOptScalar f_old, ParOptScalar h_old);

  // Test that if (f, h) is acceptable by the current filter set
  int acceptableByFilter(ParOptScalar f, ParOptScalar h);

  // Add pair (f, h) to the filter set, meanwhile remove dominated pairs
  void addToFilter(ParOptScalar f, ParOptScalar h);

  // Clear the blocking elements from the filter
  // void clearBlockingFilter( ParOptScalar f, ParOptScalar h,
  //                           double q, double mu );

  // Print filter entries
  void printFilter();

  // Set the output file
  void setOutputFile(const char *filename);

  // Compute the KKT error in the solution
  void computeKKTError(const ParOptScalar *z, ParOptVec *zw, double *l1,
                       double *linfty);

  // Print the options summary
  void printOptionSummary(FILE *fp);

  // Add info statement
  void addToInfo(size_t info_size, char *info, const char *format, ...);

  // File pointer for the summary file - depending on the settings
  FILE *outfp;
  int iter_count;                 // Iteration counter
  int subproblem_iters;           // Subproblem iteration counter
  int adaptive_subproblem_iters;  // Subproblem iteration counter

  int n;       // The number of design variables (local)
  int m;       // The number of dense constraints (global)
  int nineq;   // The number of inequality constraints
  int nwcon;   // The number of sparse constraints
  int nwineq;  // The number of sparse inequality constraints

  double tr_size;         // The trust region size
  double *penalty_gamma;  // Penalty parameters

  // Temporary vectors
  ParOptVec *t;
  ParOptVec *best_step;

  // Filter, set element is the filter pair (fi, hi, qi, mui)
  class FilterElement {
   public:
    FilterElement(ParOptScalar fk, ParOptScalar hk) {
      f = fk;
      h = hk;
    }
    ParOptScalar f, h;
  };
  std::list<FilterElement> filter;
};

#endif  // PAR_OPT_TRUST_REGION_H
