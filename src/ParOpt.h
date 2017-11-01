#ifndef PAR_OPT_OPTIMIZER_H
#define PAR_OPT_OPTIMIZER_H

/*
  Copyright (c) 2014-2016 Graeme Kennedy. All rights reserved
*/

#include <stdio.h>
#include "ParOptVec.h"
#include "ParOptQuasiNewton.h"
#include "ParOptProblem.h"

enum ParOptNormType { PAROPT_INFTY_NORM,
                      PAROPT_L1_NORM,
                      PAROPT_L2_NORM };

enum ParOptQuasiNewtonType { PAROPT_BFGS, 
                             PAROPT_SR1, 
                             PAROPT_NO_HESSIAN_APPROX };

enum ParOptBarrierStrategy { PAROPT_MONOTONE,
                             PAROPT_MEHROTRA,
                             PAROPT_COMPLEMENTARITY_FRACTION };

/*
  ParOpt is a parallel optimizer implemented in C++ for large-scale 
  constrained optimization.

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
  gamma - z - zt = 0
  c(x) - s = 0
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

  ||J(q)*p + r(q)|| <= eta*||r(q)||

  where q are all the optimization variables, r(q) are the perturbed
  KKT residuals and J(q) is either an approximate or exact
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
  
  The full KKT system can be written as follows:
  
  [  B    0 -Ac^{T} -Aw^{T}   0   0   0  -I         I        ][ px  ]
  [  0    0      -I       0   0  -I   0   0         0        ][ pt  ]
  [  Ac   I       0       0  -I   0   0   0         0        ][ pz  ]
  [  Aw   0       0       0   0   0  -I   0         0        ][ pzw ]
  [  0    0       S       0   Z   0   0   0         0        ][ ps  ] = -r
  [  0   Zt       0       0   0   T   0   0         0        ][ pzt ]
  [  0    0       0       Sw  0   0  Zw   0         0        ][ psw ]
  [  Zl   0       0       0   0   0  0    (X - Xl)  0        ][ pzl ]
  [ -Zu   0       0       0   0   0  0    0         (Xu - X) ][ pzu ]

  where B is a quasi-Newton Hessian approximation. 

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
  J*J_{B}^{-1}, takes a special form where only entries associated
  with the design vector need to be stored.
*/
class ParOpt : public ParOptBase {
 public:
  ParOpt( ParOptProblem *_prob,
          int _max_qn_subspace=10,
          ParOptQuasiNewtonType qn_type=PAROPT_BFGS,
          double _max_bound_val=1e20 );
  ~ParOpt();

  // Reset the problem instance - problem sizes must remain the same
  // ---------------------------------------------------------------
  void resetProblemInstance( ParOptProblem *_prob );

  // Perform the optimization
  // ------------------------
  int optimize( const char *checkpoint=NULL );

  // Get the problem sizes from the underlying problem class
  // -------------------------------------------------------
  void getProblemSizes( int *_nvars, int *_ncon, 
                        int *_nwcon, int *_nwblock );
  
  // Retrieve the values of the design variables and multipliers
  // -----------------------------------------------------------
  void getOptimizedPoint( ParOptVec **_x, 
                          ParOptScalar **_z, ParOptVec **_zw,
                          ParOptVec **_zl, ParOptVec **_zu );

  // Check the objective and constraint gradients
  // --------------------------------------------
  void checkGradients( double dh );

  // Set the maximum absolute value of the variable bound
  // ----------------------------------------------------
  void setMaxAbsVariableBound( double max_bound );

  // Set optimizer parameters
  // ------------------------
  void setNormType( ParOptNormType _norm_type );
  void setBarrierStrategy( ParOptBarrierStrategy strategy );
  void setInitStartingPoint( int init );
  void setMaxMajorIterations( int iters );
  void setAbsOptimalityTol( double tol );
  void setRelFunctionTol( double tol );
  void setPenaltyGamma( double gamma );
  void setBarrierFraction( double frac );
  void setBarrierPower( double power );
  void setHessianResetFreq( int freq );
  void setQNDiagonalFactor( double sigma );
  void setBFGSUpdateType( ParOptBFGSUpdateType bfgs_update );
  void setSequentialLinearMethod( int truth );

  // Set/get the barrier parameter
  // -----------------------------
  void setInitBarrierParameter( double mu );
  double getBarrierParameter();
  ParOptScalar getComplementarity();

  // Set parameters associated with the line search
  // ----------------------------------------------
  void setUseLineSearch( int truth );
  void setMaxLineSearchIters( int iters );
  void setBacktrackingLineSearch( int truth );
  void setArmijoParam( double c1 );
  void setPenaltyDescentFraction( double frac );

  // Set the parameter to set/use a diagonal Hessian
  // -----------------------------------------------
  void setUseDiagHessian( int truth );

  // Set parameters for the internal GMRES algorithm
  // -----------------------------------------------
  void setUseHvecProduct( int truth );
  void setUseQNGMRESPreCon( int truth );
  void setNKSwitchTolerance( double tol );
  void setEisenstatWalkerParameters( double gamma, double alpha );
  void setGMRESTolerances( double rtol, double atol );
  void setGMRESSubspaceSize( int _gmres_subspace_size );

  // Quasi-Newton options
  // --------------------
  void setQuasiNewton( ParOptCompactQuasiNewton *_qn );
  void setUseQuasiNewtonUpdates( int truth );
  void resetQuasiNewtonHessian();

  // Reset the design point and the bounds using the problem instance
  // ----------------------------------------------------------------
  void resetDesignAndBounds();

  // Set other parameters
  // --------------------
  void setOutputFrequency( int freq );
  void setMajorIterStepCheck( int step );
  void setGradientCheckFrequency( int freq, double step_size );

  // Set the output print level
  // --------------------------
  void setOutputFile( const char *filename );
  void setOutputLevel( int level );

  // Write out the design variables to a binary format (fast MPI/IO)
  // ---------------------------------------------------------------
  int writeSolutionFile( const char *filename );
  int readSolutionFile( const char *filename );

 private:
  // Print out the optimizer options to a file
  void printOptionSummary( FILE *fp );

  // Check and initialize the design variables and their bounds
  void initAndCheckDesignAndBounds( int init_multipliers );

  // Factor/apply the Cw matrix
  int factorCw();
  int applyCwFactor( ParOptVec *vec );

  // Compute the negative of the KKT residuals - return
  // the maximum primal, dual residuals and the max infeasibility
  void computeKKTRes( double barrier,
                      double *max_prime,
                      double *max_dual, 
                      double *max_infeas );

  // Set up the diagonal KKT system
  void setUpKKTDiagSystem( ParOptVec *xt, ParOptVec *wt, int use_qn );

  // Solve the diagonal KKT system
  void solveKKTDiagSystem( ParOptVec *bx, ParOptScalar *bt,
                           ParOptScalar *bc, ParOptVec *bcw, 
                           ParOptScalar *bs, ParOptVec *bsw,
                           ParOptScalar *bzt,
                           ParOptVec *bzl, ParOptVec *bzu,
                           ParOptVec *yx, ParOptScalar *yt,
                           ParOptScalar *yz, ParOptVec *yzw, 
                           ParOptScalar *ys, ParOptVec *ysw,
                           ParOptScalar *yzt,
                           ParOptVec *yzl, ParOptVec *yzu,
                           ParOptVec *xt, ParOptVec *wt );
  
  // Solve the diagonal KKT system with a specific RHS structure
  void solveKKTDiagSystem( ParOptVec *bx, 
                           ParOptVec *yx, ParOptScalar *yt,
                           ParOptScalar *yz, ParOptVec *yzw, 
                           ParOptScalar *ys, ParOptVec *ysw,
                           ParOptScalar *yzt,
                           ParOptVec *yzl, ParOptVec *yzu,
                           ParOptVec *xt, ParOptVec *wt );

  // Solve the diagonal KKT system but only return the components
  // corresponding to the design variables
  void solveKKTDiagSystem( ParOptVec *bx, ParOptVec *yx,
                           ParOptScalar *zt,
                           ParOptVec *xt, ParOptVec *wt );

  // Solve the diagonal system
  void solveKKTDiagSystem( ParOptVec *bx, 
                           ParOptScalar alpha, 
                           ParOptScalar *bt, ParOptScalar *bc, 
                           ParOptVec *bcw, ParOptScalar *bs,
                           ParOptVec *bsw, ParOptScalar *bzt,
                           ParOptVec *bzl, ParOptVec *bzu,
                           ParOptVec *yx, ParOptScalar *yz,
                           ParOptVec *xt, ParOptVec *wt );

  // Set up the full KKT system
  void setUpKKTSystem( ParOptScalar *zt, 
                       ParOptVec *xt1, ParOptVec *xt2,
                       ParOptVec *wt, int use_bfgs );

  // Solve for the KKT step
  void computeKKTStep( ParOptScalar *zt, ParOptVec *xt1, 
                       ParOptVec *xt2, ParOptVec *wt, int use_bfgs );
  
  // Compute the full KKT step
  int computeKKTInexactNewtonStep( ParOptScalar *zt, ParOptVec *xt1, 
                                   ParOptVec *xt2, ParOptVec *wt,
                                   double rtol, double atol, int use_bfgs );

  // Check that the KKT step is computed correctly
  void checkKKTStep( int iteration, int is_newton );

  // Compute the maximum step length to maintain positivity of 
  // all components of the design variables 
  void computeMaxStep( double tau, 
                       double *_max_x, double *_max_z );

  // Perform the line search
  int lineSearch( double *_alpha, 
                  ParOptScalar m0, ParOptScalar dm0 );

  // Evaluate the merit function
  ParOptScalar evalMeritFunc( ParOptScalar fk,
                              const ParOptScalar *ck,
                              ParOptVec *xk, 
                              const ParOptScalar *sk, 
                              const ParOptScalar *tk,
                              ParOptVec *swk );

  // Evaluate the merit function, its derivative and the new penalty
  // parameter
  void evalMeritInitDeriv( double max_x,
                           ParOptScalar *_merit, ParOptScalar *_pmerit,
                           int inexact_step, ParOptVec *xtmp,
                           ParOptVec *wtmp1, ParOptVec *wtmp2 );
  
  // Compute the average of the complementarity products at the
  // current point: Complementarity at (x + p)
  ParOptScalar computeComp();
  ParOptScalar computeCompStep( double alpha_x,
                                double alpha_z );

  // Check the step
  void checkStep();

  // The parallel optimizer problem and constraints
  ParOptProblem *prob;

  // Communicator info
  MPI_Comm comm;
  int opt_root;

  // THe type of barrier strategy to use
  ParOptBarrierStrategy barrier_strategy;

  // Set the norm type to use
  ParOptNormType norm_type;

  // Set the variable bound value
  double max_bound_val;

  // The number of variables and constraints in the problem
  int nvars; // The number of local (on-processor) variables
  int ncon; // The number of inequality constraints in the problem
  int nwcon; // The number of specially constructed weighting constraints
  int nwblock; // The nuber of constraints per block
  int nvars_total; // The total number of variables

  // Distributed variable/constriant ranges
  int *var_range, *wcon_range;

  // Temporary vectors for internal usage
  ParOptVec *wtemp;
  ParOptScalar *ztemp;

  // The variables in the optimization problem
  ParOptVec *x, *zl, *zu, *zw, *sw;
  ParOptScalar *z, *s, *zt, *t;

  // The lower/upper bounds on the variables
  ParOptVec *lb, *ub;

  // The steps in the variables
  ParOptVec *px, *pzl, *pzu, *pzw, *psw;
  ParOptScalar *pz, *ps, *pzt, *pt;

  // The residuals
  ParOptVec *rx, *rzl, *rzu, *rcw, *rsw;
  ParOptScalar *rc, *rs, *rzt, *rt;

  // The objective, gradient, constraints, and constraint gradients
  ParOptScalar fobj, *c;
  ParOptVec *g, **Ac;

  // The data for the block-diagonal matrix
  ParOptScalar *Cw;

  // Data required for solving the KKT system
  ParOptVec *Cvec;
  ParOptVec **Ew;
  ParOptScalar *Dmat, *Ce;
  int *dpiv, *cpiv;

  // Storage for the Quasi-Newton updates
  ParOptCompactQuasiNewton *qn;
  ParOptVec *y_qn, *s_qn;

  // Diagonal factor added to the Hessian to promote descent
  double qn_sigma;

  // Keep track of the number of objective and gradient evaluations
  int neval, ngeval, nhvec;

  // Sparse equalities or inequalities?
  int sparse_inequality;

  // Dense equality of dense inequalities? 
  int dense_inequality;

  // Flags to indicate whether to use the upper/lower bounds
  int use_lower, use_upper;

  // The l1-penalty parameter
  double penalty_gamma;

  // Parameters for optimization
  int max_major_iters;
  int init_starting_point;
  int write_output_frequency;

  // Parameters for the periodic gradient check option
  int gradient_check_frequency;
  double gradient_check_step;

  // The barrier parameter
  double barrier_param;

  // Stopping criteria tolerances
  double abs_res_tol;
  double rel_func_tol;

  // Parameter for controlling the Hessian reset
  int hessian_reset_freq;

  // Parameter that controls whether quasi-Newton updates are applied
  // at all
  int use_quasi_newton_update;

  // Parameters for the line search
  int max_line_iters;
  int use_line_search, use_backtracking_alpha;
  double rho_penalty_search;
  double penalty_descent_fraction, armijo_constant;

  // Parameters for controling the barrier update
  double monotone_barrier_fraction, monotone_barrier_power;
 
  // The minimum step to the boundary;
  double min_fraction_to_boundary;

  // Control of exact diagonal Hessian
  int use_diag_hessian;
  ParOptVec *hdiag;

  // Control of exact Hessian-vector products
  int use_hvec_product;
  int use_qn_gmres_precon;
  double eisenstat_walker_alpha, eisenstat_walker_gamma;
  double nk_switch_tol;
  double max_gmres_rtol, gmres_atol;

  // Internal information about GMRES
  int gmres_subspace_size;
  ParOptScalar *gmres_H, *gmres_alpha, *gmres_res, *gmres_Q;
  ParOptVec **gmres_W;

  // Check the step at this major iteration - for debugging
  int major_iter_step_check;
  
  // The step length for the merit function derivative test
  double merit_func_check_epsilon;

  // Flag to indicate whether to use a sequential linear programming
  // approach, completely discarding the quasi-Newton approximation
  int sequential_linear_method;

  // The file pointer to use for printing things out
  FILE *outfp;
  int output_level;
};

#endif // PAR_OPT_OPTIMIZER_H
