/*
  A parallel optimizer implemented in C++ for large-scale constrained
  parallel optimization.

  This code uses an interior-point method to perform gradient-based
  design optimization. The KKT system is solved using a bordered
  solution technique that may suffer from numerical precision issues
  under extreme values.

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
  c(x) is small, x is very large, and Aw is also very large, but has a
  very specialized structure.

  At each step of the optimization, we compute a solution to the linear
  system above, using: 

  K*p = - r

  where K is the linearization of the above system of equations, p is
  a search direction, and r are the residuals. Instead of using an
  exact linearization, we use an approximation based on compact L-BFGS
  representation. To compute the update, we use the
  Sherman-Morrison-Woodbury formula. This is possible due to the
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

class ParOptVec {
 public:
  ParOptVec( int numm );

}


class LBFGSUpdate {
 public:
  BFGSMatrix( );

};

class ParOpt {
 public:
  ParOpt();

  void optimize();

 private:
  
  // Compute the maximum step length to maintain positivity of 
  // all components of the design variables 
  void computeMaxStep( double tau, 
		       double *_max_x, double *_max_z );

  // Perform the line search
  int lineSearch( double * _alpha, 
		  double m0, double dm0 );

  // Evaluate the merit function
  double evalMeritFunc();

  // Eval the merit function, its derivative and the new penalty parameter
  void evalMeritInitDeriv( double * _merit, double * _pmerit );
  
  // Compute the complementarity
  double computeComp(); // Complementarity at the current point
  double computeCompStep( double alpha_x,
			  double alpha_z ); // Complementarity at (x + p)

  // Communicator info
  MPI_Comm opt_root;
  int opt_root;

  // The number of variables and constraints in the problem
  int nvars; // The number of local (on-processor) variables
  int nvars_total; // The total number of variables
  int ncon; // The number of inequality constraints in the problem

  // The variables in the optimization problem
  ParOptVec *x, *zl, *zu;
  double *z, *s;

  // The steps in the variables
  ParOptVec *px, *pzl, *pzu;
  double *pz, *ps;

  // The objective, gradient, constraints, and constraint gradients
  double fobj, *c;
  ParOptVec *g, **Ac;

  // The residuals

  // Storage for the Quasi-Newton updates
  ParOptVec *y_qn, *s_qn;

  // Parameters for optimization
  int init_starting_point;
  int write_output_frequency;

  // The barrier parameter
  double mu;

  // Stopping criteria tolerance
  double abs_resl_tol;

  // Parameters for the line search
  double rho_penalty_search;
  double penalty_descent_fraction, armijo_constant;

  // Parameters for controling the barrier update
  double monotone_barrier_fraction, monotone_barrier_power;
 
  // The minimum step to the boundary;
  double min_fraction_to_boundary;
};

