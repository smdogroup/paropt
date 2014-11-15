#ifndef PAR_OPT_OPTIMIZER_H
#define PAR_OPT_OPTIMIZER_H

#include "ParOptVec.h"
#include "ParOptProblem.h"

/*
  A parallel optimizer implemented in C++ for large-scale constrained
  optimization.

  This code uses an interior-point method to perform gradient-based
  design optimization. The KKT system is solved using a bordered
  solution technique that may suffer from numerical precision issues
  under some circumstances, but is well-suited for large-scale
  applications.

  The optimization problem is formulated as follows:

  min f(x)
  s.t.  lb <= x < ub
  s.t.  c(x) >= 0 
  s.t.  Aw*x = b
  
  where Aw is a large, sparse constraint matrix. The perturbed KKT
  conditions for this problem are:

  g(x) - A(x)^{T}*z - Aw^{T}*zw - zl + zu = 0
  Aw*x - b = 0
  c(x) - s = 0
  S*z - mu*e = 0
  (X - Xl)*zl - mu*e = 0
  (Xu - X)*zu - mu*e = 0

  where g = grad f(x) and A(x) = grad c(x). The Lagrange multipliers
  are z, zw, zl, and zu, respectively.  Note that here we assume that
  c(x) has only a few entries, x is very large, and Aw is also very
  large, but has a special structure.

  At each step of the optimization, we compute a solution to the
  linear system above, using:

  Km*p = - r

  where K is the linearization of the above system of equations, p is
  a search direction, and r are the residuals. Instead of using an
  exact linearization, we use an approximation based on a compact
  limited-memory BFGS representation. To compute the update, we use
  the Sherman-Morrison-Woodbury formula. This is possible due to the
  compact L-BFGS representation.

  The KKT system can be written as follows:
  
  [  B   -Aw^{T} -Ac^{T}  0  -I         I        ][ px  ]
  [  Aw   0       0       0   0         0        ][ pzw ]
  [  Ac   0       0      -I   0         0        ][ pz  ] = -r
  [  0    0       S       Z   0         0        ][ ps  ]
  [  Zl   0       0       0   (X - Xl)  0        ][ pzl ]
  [ -Zu   0       0       0   0         (Xu - X) ][ pzu ]

  where B is a quasi-Newton Hessian approximation. This approximation
  takes the form:
  
  B = b0*I - Z*M*Z^{T}
*/

class ParOpt {
 public:
  ParOpt( ParOptProblem *_prob, 
	  int _max_lbfgs_subspace );
  ~ParOpt();

  // Perform the optimization
  // ------------------------
  int optimize();

 private:
  // Compute the negative of the KKT residuals - return
  // the maximum primal, dual residuals and the max infeasibility
  void computeKKTRes( double * max_prime,
		      double * max_dual, 
		      double * max_infeas );

  // Set up the diagonal KKT system
  void setUpKKTDiagSystem();

  // Solve the diagonal KKT system
  void solveKKTDiagSystem( ParOptVec *bx, double *bc, double *bs,
			   ParOptVec *bzl, ParOptVec *bzu,
			   ParOptVec *yx, double *yz, double *ys,
			   ParOptVec *yzl, ParOptVec *yzu );

  // Solve the diagonal KKT system with a specific RHS structure
  void solveKKTDiagSystem( ParOptVec *bx, 
			   ParOptVec *yx, double *yz, double *ys,
			   ParOptVec *yzl, ParOptVec *yzu );

  // Solve the diagonal KKT system but only return the components
  // corresponding to the design variables
  void solveKKTDiagSystem( ParOptVec *bx, ParOptVec *yx );

  // Set up the full KKT system
  void setUpKKTSystem();

  // Solve for the KKT step
  void computeKKTStep();

  // Compute the maximum step length to maintain positivity of 
  // all components of the design variables 
  void computeMaxStep( double tau, 
		       double *_max_x, double *_max_z );

  // Perform the line search
  int lineSearch( double * _alpha, 
		  double m0, double dm0 );

  // Evaluate the merit function
  double evalMeritFunc( ParOptVec *xk, double *sk );

  // Evaluate the merit function, its derivative and the new penalty
  // parameter
  void evalMeritInitDeriv( double max_x,
			   double * _merit, double * _pmerit );
  
  // Compute the complementarity
  double computeComp(); // Complementarity at the current point
  double computeCompStep( double alpha_x,
			  double alpha_z ); // Complementarity at (x + p)

  // The parallel optimizer problem
  ParOptProblem * prob;

  // Communicator info
  MPI_Comm comm;
  int opt_root;

  // The number of variables and constraints in the problem
  int nvars; // The number of local (on-processor) variables
  int nvars_total; // The total number of variables
  int ncon; // The number of inequality constraints in the problem

  // Temporary vectors for internal usage
  ParOptVec *xtemp;
  double *ztemp;

  // The variables in the optimization problem
  ParOptVec *x, *zl, *zu;
  double *z, *s;

  // The lower/upper bounds on the variables
  ParOptVec *lb, *ub;

  // The steps in the variables
  ParOptVec *px, *pzl, *pzu;
  double *pz, *ps;

  // The residuals
  ParOptVec *rx, *rzl, *rzu;
  double *rc, *rs;

  // The objective, gradient, constraints, and constraint gradients
  double fobj, *c;
  ParOptVec *g, **Ac;

  // Data required for solving the KKT system
  ParOptVec *Cvec; // The diagonal entries
  double *Dmat, *Ce;
  int *dpiv, *cpiv;

  // Storage for the Quasi-Newton updates
  LBFGS *qn;
  ParOptVec *y_qn, *s_qn;

  // Parameters for optimization
  int max_major_iters;
  int init_starting_point;
  int write_output_frequency;

  // The barrier parameter
  double barrier_param;

  // Stopping criteria tolerance
  double abs_res_tol;

  // Parameters for the line search
  int max_line_iters;
  int use_line_search;
  double rho_penalty_search;
  double penalty_descent_fraction, armijo_constant;

  // Parameters for controling the barrier update
  double monotone_barrier_fraction, monotone_barrier_power;
 
  // The minimum step to the boundary;
  double min_fraction_to_boundary;
};

#endif
