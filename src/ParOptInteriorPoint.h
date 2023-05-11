#ifndef PAR_OPT_INTERIOR_POINT_H
#define PAR_OPT_INTERIOR_POINT_H

#include <stdio.h>

#include "ParOptOptions.h"
#include "ParOptProblem.h"
#include "ParOptQuasiNewton.h"
#include "ParOptScaledQuasiNewton.h"
#include "ParOptSparseMat.h"
#include "ParOptVec.h"

/*
  Different options for use within ParOpt
*/
enum ParOptNormType { PAROPT_INFTY_NORM, PAROPT_L1_NORM, PAROPT_L2_NORM };

enum ParOptQuasiNewtonType {
  PAROPT_BFGS,
  PAROPT_SR1,
  PAROPT_NO_HESSIAN_APPROX
};

enum ParOptBarrierStrategy {
  PAROPT_MONOTONE,
  PAROPT_MEHROTRA,
  PAROPT_MEHROTRA_PREDICTOR_CORRECTOR,
  PAROPT_COMPLEMENTARITY_FRACTION
};

enum ParOptStartingPointStrategy {
  PAROPT_NO_START_STRATEGY,
  PAROPT_LEAST_SQUARES_MULTIPLIERS,
  PAROPT_AFFINE_STEP
};

/*
  ParOptInteriorPoint is a parallel interior-point optimizer
  implemented in C++ for large-scale constrained optimization.

  This code uses an interior-point method to perform gradient-based
  design optimization. The KKT system is solved using a bordered
  solution technique that may suffer from numerical precision issues
  under some circumstances, but is well-suited for large-scale
  applications.

  The optimization problem is formulated as follows:

  min f(x)
  s.t.  c(x) >= 0
  s.t.  cw(x) >= 0
  s.t.  lb <= x <= ub

  where c(x) is a small (< 100) vector of constraints, cw(x) is a
  (possibly) nonlinear constraint with a special sparse Jacobian
  structure and lb/ub are the lower and upper bounds,
  respectively. The perturbed KKT conditions for this problem are:

  g(x) - A(x)^{T}*z - Aw^{T}*zw - zl + zu = 0
  gamma_s + z - zs = 0
  gamma_t - z - zt = 0
  c(x) - s + t = 0
  cw(x) - sw = 0
  S*z - mu*e = 0
  T*zt - mu*e = 0
  Sw*zw - mu*e = 0
  (X - Xl)*zl - mu*e = 0
  (Xu - X)*zu - mu*e = 0

  where g = grad f(x), A(x) = grad c(x) and Aw(x) = grad cw(x). The
  Lagrange multipliers are z, zw, zl, and zu, respectively.

  At each step of the optimization, we compute a solution to the
  linear system above, using:

  ||K(q)*p + r(q)|| <= eta*||r(q)||

  where q are all the optimization variables, r(q) are the perturbed
  KKT residuals and K(q) is either an approximate or exact
  linearization of r(q). The parameter eta is a forcing term that
  controls how tightly the linearization is solved. The inexact
  solution p is a search direction that is subsequently used in a line
  search.

  During the early stages of the optimization, we use a quasi-Newton
  Hessian approximations based either on compact limited-memory BFGS
  or SR1 updates. In this case, we can compute an exact solution to
  the update equations using the Sherman-Morrison-Woodbury formula.
  When these formula are used, we can represent the update formula as
  follows:

  B = b0*I - Z*M*Z^{T}

  where b0 is a scalar, M is a small matrix and Z is a matrix with
  small column dimension that is stored as a series of vectors. The
  form of these matrices depends on whether the limited-memory BFGS or
  SR1 technique is used.

  After certain transition criteria are met, we employ an exact
  Hessian, accessible through Hessian-vector products, and instead
  solve exact linearization inexactly using a forcing parameter such
  that eta > 0. We use this technique because the Hessian-vector
  products are costly to compute and may not provide a benefit early
  in the optimization.

  In the inexact phase, we select the forcing parameter based on the
  work of Eisenstat and Walker as follows:

  eta = gamma*(||r(q_{k})||_{infty}/||r(q_{k-1})||_{infty})^{alpha}

  where gamma and alpha are parameters such that 0 < gamma <= 1.0, and
  1 < alpha <= 2.  The transition from the approximate to the inexact
  optimization phase depends on two factors:

  1. The KKT residuals measured in the infinity norm ||r(q)||_{infty}
  must be reduced below a specified tolerance

  2. The eta parameter predicted by the Eisenstat-Walker formula must
  be below some maximum tolerance.

  Once both of these criteria are satisifed, we compute updates using
  the exact linearization with right-preconditioned GMRES scheme. This
  method utilizes the limited-memory BFGS or SR1 quasi-Newton
  approximation as a preconditioner. The preconditioned operator,
  K*K_{B}^{-1}, takes a special form where only entries associated
  with the design vector need to be stored.
*/
class ParOptInteriorPoint : public ParOptBase {
 public:
  ParOptInteriorPoint(ParOptProblem *_prob, ParOptOptions *_options = NULL);
  ~ParOptInteriorPoint();

  // Set the default arguments and values
  // ------------------------------------
  static void addDefaultOptions(ParOptOptions *options);
  ParOptOptions *getOptions();

  // Retrieve the optimization problem class
  // ---------------------------------------
  ParOptProblem *getOptProblem() { return prob; }

  // Reset the problem instance - problem sizes must remain the same
  // ---------------------------------------------------------------
  void resetProblemInstance(ParOptProblem *_prob);

  // Perform the optimization
  // ------------------------
  int optimize(const char *checkpoint = NULL);

  // Get the problem sizes from the underlying problem class
  // -------------------------------------------------------
  void getProblemSizes(int *_nvars, int *_ncon, int *_inequality, int *_nwcon,
                       int *_nwblock);

  // Retrieve the values of the design variables and multipliers
  // -----------------------------------------------------------
  void getOptimizedPoint(ParOptVec **_x, ParOptScalar **_z, ParOptVec **_zw,
                         ParOptVec **_zl, ParOptVec **_zu);

  // Retrieve the optimized slack variable values
  // --------------------------------------------
  void getOptimizedSlacks(ParOptScalar **_s, ParOptScalar **_t, ParOptVec **_sw,
                          ParOptVec **_tw);

  // Check the objective and constraint gradients
  // --------------------------------------------
  void checkGradients(double dh);

  // Set optimizer parameters
  // ------------------------
  void setPenaltyGamma(double gamma);
  void setPenaltyGamma(const double *gamma);
  void setBFGSUpdateType(ParOptBFGSUpdateType bfgs_update);

  // Get the barrier parameter and complementarity measure
  // -----------------------------------------------------
  double getBarrierParameter();
  ParOptScalar getComplementarity();

  // Set the parameter to set/use a diagonal Hessian
  // -----------------------------------------------
  void setUseDiagHessian(int truth);

  // Quasi-Newton options
  // --------------------
  void setQuasiNewton(ParOptCompactQuasiNewton *_qn);
  void resetQuasiNewtonHessian();

  // Reset the design point and the bounds using the problem instance
  // ----------------------------------------------------------------
  void resetDesignAndBounds();

  // Write out the design variables to a binary format (fast MPI/IO)
  // ---------------------------------------------------------------
  int writeSolutionFile(const char *filename);
  int readSolutionFile(const char *filename);

  // Check the merit function derivative at the given point
  // ------------------------------------------------------
  void checkMeritFuncGradient(ParOptVec *xpt = NULL, double dh = 1e-6);

  // Get the iteration counts
  // ------------------------
  void getIterationCounters(int *_niter = NULL, int *_neval = NULL,
                            int *_ngeval = NULL, int *_nhvec = NULL) {
    if (_niter) {
      *_niter = niter;
    }
    if (_neval) {
      *_neval = neval;
    }
    if (_ngeval) {
      *_ngeval = ngeval;
    }
    if (_nhvec) {
      *_nhvec = nhvec;
    }
  }

 private:
  static const int PAROPT_LINE_SEARCH_SUCCESS = 1;
  static const int PAROPT_LINE_SEARCH_FAILURE = 2;
  static const int PAROPT_LINE_SEARCH_MIN_STEP = 4;
  static const int PAROPT_LINE_SEARCH_MAX_ITERS = 8;
  static const int PAROPT_LINE_SEARCH_NO_IMPROVEMENT = 16;
  static const int PAROPT_LINE_SEARCH_SHORT_STEP = 32;

  class ParOptVars;

  // Set the size of the GMRES subspace
  void setGMRESSubspaceSize(int m);

  // Set the output file name and write the options summary
  void setOutputFile(const char *filename);

  // Print out the optimizer options to a file
  void printOptionSummary(FILE *fp);

  // Add to the info string
  void addToInfo(size_t info_size, char *info, const char *format, ...);

  // Check and initialize the design variables and their bounds
  void initAndCheckDesignAndBounds();

  // Initialize the multipliers
  void initLeastSquaresMultipliers(ParOptVars &vars);
  void initAffineStepMultipliers(ParOptVars &vars, ParOptVars &res,
                                 ParOptVars &step, ParOptNormType norm_type);

  // Compute the negative of the KKT residuals - return
  // the maximum primal, dual residuals and the max infeasibility
  void computeKKTRes(ParOptVars &vars, double barrier, ParOptVars &res,
                     ParOptNormType norm_type, double *max_prime,
                     double *max_dual, double *max_infeas,
                     double *res_norm = NULL);

  // Add the corrector components to the residual to compute the MPC step
  void addMehrotraCorrectorResidual(ParOptVars &step, ParOptVars &res);

  // Compute the norm of the step
  double computeStepNorm(ParOptNormType norm_type, ParOptVars &step);

  // Set up the diagonal KKT system
  void setUpKKTDiagSystem(ParOptVars &vars, ParOptVec *xtmp, ParOptVec *wtmp,
                          int use_qn);

  // Solve the diagonal KKT system
  void solveKKTDiagSystem(ParOptVars &vars, ParOptVars &b, ParOptVars &y,
                          ParOptVec *d1, ParOptVec *d2);

  // Solve the diagonal KKT system with a specific RHS structure
  void solveKKTDiagSystem(ParOptVars &vars, ParOptVec *bx, ParOptVars &y,
                          ParOptVec *d1, ParOptVec *d2);

  // Solve the diagonal KKT system but only return the components
  // corresponding to the design variables
  void solveKKTDiagSystem(ParOptVars &vars, ParOptVec *bx, ParOptVec *yx,
                          ParOptScalar *yz, ParOptVec *d1, ParOptVec *yzw);

  // Solve the diagonal system
  void solveKKTDiagSystem(ParOptVars &vars, ParOptVec *bx, ParOptScalar alpha,
                          ParOptVars &b, ParOptVars &y, ParOptVec *d1,
                          ParOptVec *d2);

  // Set up the full KKT system
  void setUpKKTSystem(ParOptVars &vars, ParOptScalar *ztmp, ParOptVec *xtmp1,
                      ParOptVec *xtmp2, ParOptVec *wtmp, int use_qn);

  // Solve for the KKT step
  void computeKKTStep(ParOptVars &vars, ParOptVars &res, ParOptVars &step,
                      ParOptScalar *ztmp, ParOptVec *xtmp1, ParOptVec *xtmp2,
                      ParOptVec *wtmp, int use_qn);

  // Compute the full KKT step
  int computeKKTGMRESStep(ParOptVars &vars, ParOptVars &res, ParOptVars &step,
                          ParOptScalar *ztmp, ParOptVec *xtmp1,
                          ParOptVec *xtmp2, ParOptVec *wtmp, double rtol,
                          double atol, int use_qn);

  // Check that the KKT step is computed correctly
  void checkKKTStep(ParOptVars &vars, ParOptVars &step, ParOptVars &res,
                    int iteration, int is_newton);

  // Compute the maximum step length to maintain positivity of
  // all components of the design variables
  void computeMaxStep(ParOptVars &vars, double tau, ParOptVars &step,
                      double *_max_x, double *_max_z);

  // Compute the step so that it satisfies the required bounds
  void scaleStep(ParOptScalar alpha, int nvals, ParOptScalar *);
  void computeStepVec(ParOptVec *xvec, ParOptScalar alpha, ParOptVec *pvec,
                      ParOptVec *lower, ParOptScalar *lower_value,
                      ParOptVec *upper, ParOptScalar *upper_value);
  void computeStep(int nvals, ParOptScalar *xvals, ParOptScalar alpha,
                   const ParOptScalar *pvals, const ParOptScalar *lbvals,
                   const ParOptScalar *lower_value, const ParOptScalar *ubvals,
                   const ParOptScalar *upper_value);

  // Perform the line search
  int lineSearch(double alpha_min, double *_alpha, ParOptScalar m0,
                 ParOptScalar dm0);

  // Scale the step by the distance-to-the-boundary rule
  int scaleKKTStep(ParOptVars &vars, ParOptVars &step, double tau,
                   ParOptScalar comp, int inexact_newton_step, double *_alpha_x,
                   double *_alpha_z);

  // Perform a primal/dual update and optionally upate the quasi-Newton Hessian
  int computeStepAndUpdate(ParOptVars &vars, double alpha, ParOptVars &step,
                           int eval_obj_con, int perform_qn_update);

  // Evaluate the merit function
  ParOptScalar evalMeritFunc(ParOptScalar fk, const ParOptScalar *ck,
                             ParOptVec *xk, const ParOptScalar *sk,
                             const ParOptScalar *tk, ParOptVec *swk);

  // Evaluate the directional derivative of the objective + barrier terms
  ParOptScalar evalObjBarrierDeriv(ParOptVars &vars, ParOptVars &step);

  // Evaluate the merit function, its derivative and the new penalty
  // parameter
  void evalMeritInitDeriv(ParOptVars &vars, ParOptVars &step, double max_x,
                          ParOptScalar *_merit, ParOptScalar *_pmerit,
                          ParOptVec *xtmp, ParOptVec *wtmp1, ParOptVec *wtmp2);

  // Compute the average of the complementarity products at the
  // current point: Complementarity at (x + p)
  ParOptScalar computeComp(ParOptVars &vars);
  ParOptScalar computeCompStep(ParOptVars &vars, double alpha_x, double alpha_z,
                               ParOptVars &step);

  // Check the step
  void checkStep();

  // All the variables in a solution vector
  class ParOptVars {
   public:
    ParOptVars();
    ~ParOptVars();
    void initialize(ParOptProblem *prob);

    // The variables in the optimization problem
    ParOptVec *x;               // The design point
    ParOptVec *zl, *zu;         // Multipliers for the upper/lower bounds
    ParOptScalar *z, *zs, *zt;  // Multipliers for the dense constraints
    ParOptVec *zw, *zsw, *ztw;  // Multipliers for the sparse constraints
    ParOptScalar *s, *t;        // Slack variables
    ParOptVec *sw, *tw;         // Slack variables for the sparse constraints
  };

  // The parallel optimizer problem and constraints
  ParOptProblem *prob;

  // All of the optimizer options
  ParOptOptions *options;

  // Communicator info
  MPI_Comm comm;
  int opt_root;

  // The number of variables and constraints in the problem
  int nvars;        // The number of local (on-processor) variables
  int ncon;         // The number of constraints in the problem
  int ninequality;  // The number of inequality constraints
  int nwcon;        // The number of specially constructed weighting constraints
  int nwinequality;  // The number of sparse inequality constraints
  int nvars_total;   // The total number of variables

  // Distributed variable/constriant ranges
  int *var_range, *wcon_range;

  // Variables
  ParOptVars variables;  // The solution variables
  ParOptVars residual;   // The residual variables
  ParOptVars update;     // The step variables

  // Temporary vectors for internal usage
  ParOptVec *xtemp;
  ParOptVec *wtemp;
  ParOptScalar *ztemp;

  // The lower/upper bounds on the variables
  ParOptVec *lb, *ub;

  // The objective, gradient, constraints, and constraint gradients
  ParOptScalar fobj, *c;
  ParOptVec *g, **Ac;

  // The l1-penalty parameters for the dense constraints and sparse constraints
  double *penalty_gamma_s, *penalty_gamma_t;
  ParOptVec *penalty_gamma_sw, *penalty_gamma_tw;

  ParOptQuasiDefMat *mat;

  // Data required for solving the KKT system
  ParOptVec *Dinv, *Cdiag;

  // The Schur complement for the dense constraints
  ParOptScalar *Gmat;
  int *gpiv;

  // The Schur complement for the quasi-Newton Hessian approximation
  ParOptScalar *Ce;
  int *cpiv;

  // Storage for the Quasi-Newton updates
  ParOptCompactQuasiNewton *qn;
  ParOptVec *y_qn, *s_qn;

  // Control of exact diagonal Hessian
  ParOptVec *hdiag;

  // Keep track of the number of objective and gradient evaluations
  int niter, neval, ngeval, nhvec;

  // Sparse equalities or inequalities?
  int sparse_inequality;

  // Flags to indicate whether to use the upper/lower bounds
  int use_lower, use_upper;

  // The barrier parameter
  double barrier_param;

  // Penalty parameter for the line search
  double rho_penalty_search;

  // Internal information about GMRES
  int gmres_subspace_size;
  ParOptScalar *gmres_H, *gmres_alpha, *gmres_res, *gmres_Q;
  ParOptScalar *gmres_y, *gmres_fproj, *gmres_aproj, *gmres_awproj;
  ParOptVec **gmres_W;

  // The file pointer to use for printing things out
  FILE *outfp;
};

#endif  // PAR_OPT_INTERIOR_POINT_H
