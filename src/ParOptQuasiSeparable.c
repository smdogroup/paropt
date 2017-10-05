#include <string.h>
#include "ComplexStep.h"
#include "ParOptBlasLapack.h"
#include "ParOptQuasiSeparable.h"

// Helper functions
inline ParOptScalar min2( ParOptScalar a, ParOptScalar b ){
  if (RealPart(a) < RealPart(b)){
    return a;
  }
  else {
    return b;
  }
}

inline ParOptScalar max2( ParOptScalar a, ParOptScalar b ){
  if (RealPart(a) > RealPart(b)){
    return a;
  }
  else {
    return b;
  }
}

/*
  Create the ParOptMMA object
*/
ParOptMMA::ParOptMMA( ParOptProblem *_prob, int _use_true_mma ):
ParOptProblem(_prob->getMPIComm()){
  use_true_mma = _use_true_mma;

  // Set the problem instance
  prob = _prob;
  prob->incref();

  // Pull out the communicator
  comm = prob->getMPIComm();

  // Set default parameters
  asymptote_contract = 0.7;
  asymptote_relax = 1.2;
  init_asymptote_offset = 0.25;
  min_asymptote_offset = 1e-8;
  max_asymptote_offset = 100.0;
  bound_relax = 1e-5;

  // Set the file pointer to NULL
  fp = NULL;
  print_level = 1;
  
  // Set the default to stdout
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0){
    fp = stdout;
  }

  // Get the problem sizes
  int _nwcon, _nwblock;
  prob->getProblemSizes(&n, &m, &_nwcon, &_nwblock);
  if (use_true_mma && _nwcon > 0){
    fprintf(stderr, 
            "ParOptMMA warning: Cannot solve probs with weight constraints\n");
  }
  setProblemSizes(n, m, _nwcon, _nwblock);

  // Set the iteration counter
  mma_iter = 0;
  subproblem_iter = 0;

  // Initialize the data
  initialize();
}

/*
  Deallocate all of the internal data
*/
ParOptMMA::~ParOptMMA(){
  if (fp && fp != stdout){
    fclose(fp);
  }
  prob->decref();

  xvec->decref();
  x1vec->decref();
  x2vec->decref();
  lbvec->decref();
  ubvec->decref();
  delete [] cons;

  gvec->decref();
  for ( int i = 0; i < m; i++ ){
    Avecs[i]->decref();
  }
  delete [] Avecs;

  Lvec->decref();
  Uvec->decref();
  alphavec->decref();
  betavec->decref();
  p0vec->decref();
  q0vec->decref();

  if (use_true_mma){
    delete [] c;
    delete [] lambda;
    delete [] zlb;
    delete [] y;

    for ( int i = 0; i < m; i++ ){
      pivecs[i]->decref();
      qivecs[i]->decref();
    }
    delete [] qivecs;
    delete [] pivecs;
    delete [] b;
  }
  else {
    if (cwvec){
      cwvec->decref();
    }
  }
}

/*
  Allocate all of the data
*/
void ParOptMMA::initialize(){
  // Incref the reference counts to the design vectors
  xvec = prob->createDesignVec();  xvec->incref();
  x1vec = prob->createDesignVec();  x1vec->incref();
  x2vec = prob->createDesignVec();  x2vec->incref();

  // Create the design variable bounds
  lbvec = prob->createDesignVec();  lbvec->incref();
  ubvec = prob->createDesignVec();  ubvec->incref();

  // Allocate the constraint array
  fobj = 0.0;
  cons = new ParOptScalar[ m ];
  memset(cons, 0, m*sizeof(ParOptScalar));

  // Allocate space for the problem gradients
  gvec = prob->createDesignVec();  gvec->incref();
  Avecs = new ParOptVec*[ m ];
  for ( int i = 0; i < m; i++ ){
    Avecs[i] = prob->createDesignVec();  Avecs[i]->incref();
  }

  // Create the move limit/asymptote vectors
  Lvec = prob->createDesignVec();  Lvec->incref();
  Uvec = prob->createDesignVec();  Uvec->incref();
  alphavec = prob->createDesignVec();  alphavec->incref();
  betavec = prob->createDesignVec();  betavec->incref();

  // Create the coefficient vectors
  p0vec = prob->createDesignVec();  p0vec->incref();
  q0vec = prob->createDesignVec();  q0vec->incref();

  // Set the sparse constraint vector to NULL 
  cwvec = NULL;

  if (use_true_mma){
    // Allocate initial values for the penalty parameters
    c = new ParOptScalar[ m ];

    // The Lagrange multipliers for the dual problem
    // and their multipliers for the lower/upper bounds
    lambda = new ParOptScalar[ m ];
    zlb = new ParOptScalar[ m ];

    // The constraint infeasibility
    y = new ParOptScalar[ m ];

    for ( int i = 0; i < m; i++ ){
      c[i] = 1000.0;
      lambda[i] = 10.0;
      zlb[i] = 1.0;
      y[i] = 0.0;
    }

    pivecs = new ParOptVec*[ m ];
    qivecs = new ParOptVec*[ m ];
    for ( int i = 0; i < m; i++ ){
      pivecs[i] = prob->createDesignVec();  pivecs[i]->incref();
      qivecs[i] = prob->createDesignVec();  qivecs[i]->incref();
    }

    b = new ParOptScalar[ m ];
    memset(b, 0, m*sizeof(ParOptScalar));
  }
  else {
    c = NULL;
    lambda = NULL;
    zlb = NULL;
    y = NULL;
    pivecs = NULL;
    qivecs = NULL;
    b = NULL;

    cwvec = prob->createConstraintVec();
    if (cwvec){
      cwvec->incref();
    }
  }

  // Get the design variables and bounds
  prob->getVarsAndBounds(xvec, lbvec, ubvec);
}

/*
  Set the output flag
*/
void ParOptMMA::setPrintLevel( int _print_level ){
  print_level = _print_level;
}

/*
  Set the asymptote contraction factor
*/
void ParOptMMA::setAsymptoteContract( double val ){
  if (val < 1.0){
    asymptote_contract = val;
  }
}

/*
  Set the asymptote relaxation factor
*/
void ParOptMMA::setAsymptoteRelax( double val ){
  if (val > 1.0){
    asymptote_relax = val;
  }
}

/*
  Set the initial asymptote factor
*/
void ParOptMMA::setInitAsymptoteOffset( double val ){
  init_asymptote_offset = val;
}

/*
  Set the minimum asymptote offset
*/
void ParOptMMA::setMinAsymptoteOffset( double val ){
  if (val < 1.0){
    min_asymptote_offset = val;
  }
}

/*
  Set the maximum asymptote offset
*/
void ParOptMMA::setMaxAsymptoteOffset( double val ){
  if (val > 1.0){
    max_asymptote_offset = val;
  }
}
 
/*
  Set the bound relaxation factor
*/
void ParOptMMA::setBoundRelax( double val ){
  bound_relax;
}

/*
  Set the output file (only on the root proc)
*/
void ParOptMMA::setOutputFile( const char *filename ){
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0){
    if (fp && fp != stdout){
      fclose(fp);
    }
    fp = fopen(filename, "w");
  }
}

/*
  Update the problem to find the new values for the design variables
*/
int ParOptMMA::update(){
  if (use_true_mma){
    // Set up the sub-problem
    initializeSubProblem(xvec);

    // Update the newest values of the design variables, and make sure
    // to store the previous updates
    x2vec->copyValues(x1vec);
    x1vec->copyValues(xvec);

    // Solve the dual problem
    int dual_fail = solveDual();

    if (fp){
      fflush(fp);
    }

    return dual_fail;
  }

  // This fails, because we're not actually using MM
  return 1;
}

/*
  Compute the KKT error
*/
void ParOptMMA::computeKKTError( double *l1, 
                                 double *linfty,
                                 double *infeas ){
  if (!use_true_mma){
    *l1 = 0.0;
    *linfty = 0.0;
    *infeas = 0.0;
    return;
  }

  // Get the objective gradient array
  ParOptScalar *g;
  gvec->getArray(&g);

  // Allocate a temp array to store the pointers
  // to the constraint vector
  ParOptScalar **A = new ParOptScalar*[ m ];
  for ( int i = 0; i < m; i++ ){
    Avecs[i]->getArray(&A[i]);
  }
  
  // Get the lower/upper bounds for the variables
  ParOptScalar *lb, *ub;
  lbvec->getArray(&lb);
  ubvec->getArray(&ub);

  // Get the current values of the design variables
  ParOptScalar *x;
  xvec->getArray(&x);

  // Set the infinity norms
  double l1_norm = 0.0;
  double infty_norm = 0.0;

  for ( int j = 0; j < n; j++ ){
    // Compute the first KKT condition
    ParOptScalar r = g[j];
    for ( int i = 0; i < m; i++ ){
      r += A[i][j]*lambda[i];
    }
    
    // Check whether the bound multipliers would eliminate this
    // residual or not. If we're at the lower bound and the KKT
    // residual is negative or if we're at the upper bound and the KKT
    // residual is positive.
    if ((x[j] <= lb[j] + bound_relax) && r >= 0.0){
      r = 0.0;
    }
    if ((x[j] + bound_relax >= ub[j]) && r <= 0.0){
      r = 0.0;
    }

    // Add the contribution to the l1/infinity norms
    double t = fabs(RealPart(r));
    l1_norm += t;
    if (t >= infty_norm){
      infty_norm = t;
    }
  }

  // Measure the infeasibility using the l1 norm
  *infeas = 0.0;
  for ( int i = 0; i < m; i++ ){
    *infeas += fabs(RealPart(max2(0.0, cons[i])));
  }

  // All-reduce the norms across all processors
  MPI_Allreduce(&l1_norm, l1, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&infty_norm, linfty, 1, MPI_DOUBLE, MPI_MAX, comm);
  
  delete [] A;
}

/*
  Get the optimized point
*/
void ParOptMMA::getOptimizedPoint( ParOptVec **_x ){
  *_x = xvec;
}

/*
  Get the asymptotes themselves
*/
void ParOptMMA::getAsymptotes( ParOptVec **_L, ParOptVec **_U ){
  if (_L){
    *_L = Lvec;
  }
  if (_U){
    *_U = Uvec;
  }
}

/*
  Evaluate the gradient and Hessian of the dual sub-problem.

  This is collective on all processors. The function overwrites the
  values stored in the grad and H matrices, and updates the values
  stored in the design vector and ys vectors.
*/
void ParOptMMA::evalDualGradient( ParOptScalar *grad,
                                  ParOptScalar *H,
                                  ParOptScalar *x,
                                  ParOptScalar *ys,
                                  const ParOptScalar *lam,
                                  const ParOptScalar *p0,
                                  const ParOptScalar *q0,
                                  ParOptScalar **pi,
                                  ParOptScalar **qi,
                                  const ParOptScalar *L,
                                  const ParOptScalar *U,
                                  const ParOptScalar *alpha,
                                  const ParOptScalar *beta ){
  // Compute the gradient and the Hessian of the subproblem
  // based on the input multipliers.
  memset(grad, 0, m*sizeof(ParOptScalar));
  memset(H, 0, m*m*sizeof(ParOptScalar));

  // Allocate a temporary vector 
  ParOptScalar *tmp = new ParOptScalar[ m ];

  for ( int i = 0; i < m; i++ ){
    ys[i] = 0.0;
    if (lam[i] > c[i]){
      ys[i] = lam[i] - c[i];
    }
  }

  // Compute the new values of x and the gradient
  // of the Lagrangian w.r.t. lam and the Hessian of
  // the Lagrangian based on  based on the
  for ( int j = 0; j < n; j++ ){
    // Compute the values
    ParOptScalar sL = p0[j];
    ParOptScalar sU = q0[j];
    for ( int i = 0; i < m; i++ ){
      sL += pi[i][j]*lam[i];
      sU += qi[i][j]*lam[i];
    }
    sL = sqrt(sL);
    sU = sqrt(sU);
    x[j] = (sL*L[j] + sU*U[j])/(sL + sU);

    int at_bounds = 0;
    if (x[j] <= alpha[j]){
      at_bounds = 1;
      x[j] = alpha[j];
    }
    else if (x[j] >= beta[j]){
      at_bounds = 1;
      x[j] = beta[j];
    }

    ParOptScalar Uinv = 1.0/(U[j] - x[j]);
    ParOptScalar Linv = 1.0/(x[j] - L[j]);
    for ( int i = 0; i < m; i++ ){
      grad[i] += Uinv*pi[i][j] + Linv*qi[i][j];
      tmp[i] = Uinv*Uinv*pi[i][j] - Linv*Linv*qi[i][j];
    }
    
    if (!at_bounds){
      ParOptScalar scale = 
        -0.5/(sL*sL*Uinv*Uinv*Uinv + sU*sU*Linv*Linv*Linv);
      for ( int ii = 0; ii < m; ii++ ){
        for ( int jj = 0; jj < m; jj++ ){
          H[ii + m*jj] += scale*tmp[ii]*tmp[jj];
        }
      }
    }
  }
  
  delete [] tmp;

  // Sum up the gradient contribution from all processors
  MPI_Allreduce(MPI_IN_PLACE, grad, m, PAROPT_MPI_TYPE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, H, m*m, PAROPT_MPI_TYPE, MPI_SUM, comm);

  for ( int i = 0; i < m; i++ ){
    grad[i] -= b[i] + ys[i];
    if (lam[i] > c[i]){
      H[i*(m+1)] -= 1.0;
    }
  }
}

/*
  Solve the dual problem using a barrier method.

  The dual problem is to maximize W(lambda) subject to lambda >= 0

  The objective is modified to account for the constraint such that
  the barrier objective is:
 
  maximize  W(lambda) + barrier*log(lambda)

  This objective is concave and has a unique minimum. Line search
  steps are truncated to ensure lambda >= 0. No line search is
  performed (this could improve robustness).
*/
int ParOptMMA::solveDual(){
  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Allocate pointers for the constraint pointers
  ParOptScalar **pi = new ParOptScalar*[ m ];
  ParOptScalar **qi = new ParOptScalar*[ m ];
  for ( int i = 0; i < m; i++ ){
    pivecs[i]->getArray(&pi[i]);
    qivecs[i]->getArray(&qi[i]);
  }

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the lower/upper move limits
  ParOptScalar *alpha, *beta;
  alphavec->getArray(&alpha);
  betavec->getArray(&beta);

  // Set the dual variables
  for ( int i = 0; i < m; i++ ){
    lambda[i] = 0.5*c[i];
    zlb[i] = 1.0;
  }

  // Set the gradient and Hessian
  ParOptScalar *grad = new ParOptScalar[ m ];
  ParOptScalar *H = new ParOptScalar[ m*m ];

  // Allocate the multiplier step
  ParOptScalar *pzlb = new ParOptScalar[ m ];

  // Get the values of the design variables
  ParOptScalar *x;
  xvec->getArray(&x);

  // Allocate the pivot array
  int *ipiv = new int[ m ];

  // Set the initial barrier parameter
  double abs_step_tol = 1e-12;
  double tol = 1e-8;
  int max_outer_iters = 20;
  int max_newton_iters = 100;
  double barrier = 1.0;
  double tau = 0.95;

  if (fp && print_level > 1){
    fprintf(fp, "\n%5s %8s %5s %9s %9s %9s\n",
            "MMA", "sub-iter", "iter", "barrier", "max step", "norm");
  }

  // Fail flag - indicating whether or not the problem failed
  int fail = 0;

  // Solve the barrier problems...
  for ( int outer = 0; outer < max_outer_iters; outer++ ){
    for ( int k = 0; k < max_newton_iters; k++ ){
      for ( int i = 0; i < m; i++ ){
        // Check for NaNs
        if (lambda[i] != lambda[i]){
          fail = i+1;
          break;
        }
      }
      if (fail){
        fprintf(stderr, "ParOptMMA: Lambda %d is NaN\n", fail);
        return fail;
      }

      evalDualGradient(grad, H, x, y, lambda, p0, q0, pi, qi,
                       L, U, alpha, beta);

      // Add the terms from the barrier, enforcing lambda >= 0
      for ( int i = 0; i < m; i++ ){
        grad[i] = -(grad[i] + barrier/lambda[i]);
        H[i*(m+1)] -= zlb[i]/lambda[i];
      }

      // Check the norm of the gradient, including the barrier
      // term to check for convergence
      double res_norm = 0.0;
      for ( int i = 0; i < m; i++ ){
        double t = fabs(RealPart(grad[i])); 
        if (i == 0){
          res_norm = t;
        }
        else if (t > res_norm){
          res_norm = t;
        }
      }

      if (res_norm < barrier){
        if (fp && print_level > 1){
          fprintf(fp, "%5d %8d %5d %9.3e %9s %9.3e\n",
                  mma_iter, subproblem_iter, k, barrier, " ", res_norm);
        }
        break;
      }

      if (print_level > 3){
        if (fp){
          fprintf(fp, "\n%4s %15s\n", " ", "grad");
          for ( int i = 0; i < m; i++ ){
            fprintf(fp, "%4d %15.8e\n", i, grad[i]);
          }
          fprintf(fp, "\n%4s %4s %15s\n", " ", " ", "H");
          for ( int j = 0; j < m; j++ ){
            for ( int i = 0; i < m; i++ ){
              fprintf(fp, "%4d %4d %15.8e\n", i, j, H[m*i + j]);
            }
          }
          fprintf(fp, "\n%4s %15s\n", " ", "lambda");
          for ( int i = 0; i < m; i++ ){
            fprintf(fp, "%4d %15.8e\n", i, lambda[i]);
          }
          fprintf(fp, "\n%4s %15s\n", " ", "y");
          for ( int i = 0; i < m; i++ ){
            fprintf(fp, "%4d %15.8e\n", i, y[i]);
          }
        }
      }

      // Compute the step length by solving the system of equations
      int one = 1;
      int info = 0;
      LAPACKdgetrf(&m, &m, H, &m, ipiv, &info);
      if (info != 0){
        if (info < 0){
          fprintf(stderr, "ParOptMMA: DGETRF %d illegal argument value\n",
                  -info);
        }
        else if (info > 0){
          fprintf(stderr, "ParOptMMA: DGETRF factorization failed %d\n",
                  info);
        }
        return -1;
      }

      // Use the factorization
      LAPACKdgetrs("N", &m, &one, H, &m, ipiv, grad, &m, &info);

      // Check the norm of the step
      double step_norm = 0.0;
      for ( int i = 0; i < m; i++ ){
        double t = fabs(RealPart(grad[i])); 
        if (i == 0){
          step_norm = t;
        }
        else if (t > res_norm){
          step_norm = t;
        }
      }

      // Increase the number of sub-iterations
      subproblem_iter++;

      // Compute the step in the multipliers
      for ( int i = 0; i < m; i++ ){
        // zlb*plam + lam*pzlb = -(zlb*lam - barrier)
        pzlb[i] = -zlb[i] + (barrier - grad[i]*zlb[i])/lambda[i];
      }

      // Truncate the step length, depending on the
      // step-to-the-boundary rule with tau as the factor
      double max_step = 1.0;
      for ( int i = 0; i < m; i++ ){
        if (grad[i] < 0.0){
          double step = -tau*RealPart(lambda[i]/grad[i]);
          if (step < max_step){
            max_step = step;
          }
        }
        if (pzlb[i] < 0.0){
          double step = -tau*RealPart(zlb[i]/pzlb[i]);
          if (step < max_step){
            max_step = step;
          }
        }
      }

      if (fp && print_level > 1){
        if (print_level > 3){
          fprintf(fp, "\n%5s %8s %5s %9s %9s %9s\n",
                  "MMA", "sub-iter", "iter", 
                  "barrier", "max step", "norm");
        }
        fprintf(fp, "%5d %8d %5d %9.3e %9.3e %9.3e\n",
                mma_iter, subproblem_iter, k, 
                barrier, max_step, res_norm);
      }

      // Update the multipliers
      for ( int i = 0; i < m; i++ ){
        lambda[i] += max_step*grad[i];
        zlb[i] += max_step*pzlb[i];
      }
    }

    if (barrier < tol){
      break;
    }

    // Reduce the barrier parameter
    barrier *= 0.1;
  }

  // Print out the multipliers at the final step
  if (print_level > 2){
    if (fp){
      fprintf(fp, "\n%4s %15s\n", " ", "lambda");
      for ( int i = 0; i < m; i++ ){
        fprintf(fp, "%4d %15.8e\n", i, lambda[i]);
      }
      fprintf(fp, "\n%4s %15s\n", " ", "zlb");
      for ( int i = 0; i < m; i++ ){
        fprintf(fp, "%4d %15.8e\n", i, zlb[i]);
      }
      fprintf(fp, "\n%4s %15s\n", " ", "y");
      for ( int i = 0; i < m; i++ ){
        fprintf(fp, "%4d %15.8e\n", i, y[i]);
      }
      fprintf(fp, "\n%4s %15s\n", " ", "con");
      for ( int i = 0; i < m; i++ ){
        fprintf(fp, "%4d %15.8e\n", i, cons[i]);
      }
      fprintf(fp, "\n%4s %15s\n", " ", "b");
      for ( int i = 0; i < m; i++ ){
        fprintf(fp, "%4d %15.8e\n", i, b[i]);
      }
    }
  }

  delete [] grad;
  delete [] H;
  delete [] ipiv;
  delete [] pi;
  delete [] qi;
  delete [] pzlb;
}

/*
  Update and initialize data for the convex sub-problem that is solved
  at each iteration. This must be called before solving the dual
  problem.

  This code updates the asymptotes, sets the move limits and forms the
  approximations used in the MMA code.
*/
int ParOptMMA::initializeSubProblem( ParOptVec *xv ){
  if (xv && xv != xvec){
    xvec->copyValues(xv);
  }

  // Evaluate the objective/constraint gradients
  int fail_obj = prob->evalObjCon(xvec, &fobj, cons);
  if (fail_obj){
    fprintf(stderr, 
      "ParOptMMA: Objective evaluation failed\n");
    return fail_obj;
  }

  int fail_grad = prob->evalObjConGradient(xvec, gvec, Avecs);
  if (fail_grad){
    fprintf(stderr, 
      "ParOptMMA: Gradient evaluation failed\n");
    return fail_grad;
  }

  if (use_true_mma){
    // Scale the constraints and gradients to match the standard MMA
    // form for the constraint formation i.e. fi(x) <= 0.0. ParOpt uses
    // the convention that c(x) >= 0.0, so we multiply by -1.0
    for ( int i = 0; i < m; i++ ){
      cons[i] *= -1.0;
      Avecs[i]->scale(-1.0);
    }

    // Compute the KKT error, and print it out to a file
    if (print_level > 0){
      double l1, linfty, infeas;
      computeKKTError(&l1, &linfty, &infeas);
      
      if (fp){
        double l1_lambda = 0.0;
        for ( int i = 0; i < m; i++ ){
          l1_lambda += fabs(RealPart(lambda[i]));
        }      

        if ((print_level == 1 && mma_iter % 10 == 0) ||
            (print_level > 1)){
          fprintf(fp, "\n%5s %8s %15s %9s %9s %9s %9s\n",
                  "MMA", "sub-iter", "fobj", "l1 opt", 
                  "linft opt", "l1 lambd", "infeas");
        }
        fprintf(fp, "%5d %8d %15.6e %9.3e %9.3e %9.3e %9.3e\n",
                mma_iter, subproblem_iter, fobj, l1, 
                linfty, l1_lambda, infeas);
      }
    }
  }
  else {
    if (cwvec){
      prob->evalSparseCon(xvec, cwvec);
    }
  }

  // Get the current values of the design variables
  ParOptScalar *x, *x1, *x2;
  xvec->getArray(&x);
  x1vec->getArray(&x1);
  x2vec->getArray(&x2);

  // Get the values of the assymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the lower/upper bounds for the variables
  ParOptScalar *lb, *ub;
  lbvec->getArray(&lb);
  ubvec->getArray(&ub);

  // Set all of the asymptote values
  if (mma_iter < 2){
    for ( int j = 0; j < n; j++ ){
      L[j] = x[j] - init_asymptote_offset*(ub[j] - lb[j]);
      U[j] = x[j] + init_asymptote_offset*(ub[j] - lb[j]);
    }
  }
  else {
    for ( int j = 0; j < n; j++ ){
      // Compute the product of the difference of the two previous
      // updates to determine how to update the move limits. If the
      // signs are different, then indc < 0.0 and we contract the 
      // asymptotes, otherwise we expand the asymptotes.
      ParOptScalar indc = (x[j] - x1[j])*(x1[j] - x2[j]);

      // Store the previous values of the asymptotes
      ParOptScalar Lprev = L[j];
      ParOptScalar Uprev = U[j];

      // Compute the interval length
      ParOptScalar intrvl = max2(ub[j] - lb[j], 0.01);
      intrvl = min2(intrvl, 100.0);

      if (RealPart(indc) < 0.0){
        // oscillation -> contract the asymptotes
        L[j] = x[j] - asymptote_contract*(x1[j] - Lprev);
        U[j] = x[j] + asymptote_contract*(Uprev - x1[j]);
      }
      else {
        // Relax the asymptotes
        L[j] = x[j] - asymptote_relax*(x1[j] - Lprev);
        U[j] = x[j] + asymptote_relax*(Uprev - x1[j]);        
      }

      // Ensure that the asymptotes do not converge entirely on the
      // design value
      L[j] = min2(L[j], x[j] - min_asymptote_offset*intrvl);
      U[j] = max2(U[j], x[j] + min_asymptote_offset*intrvl);

      // Enforce a maximum offset so that the asymptotes do not
      // move too far away from the design variables
      L[j] = max2(L[j], x[j] - max_asymptote_offset*intrvl);
      U[j] = min2(U[j], x[j] + max_asymptote_offset*intrvl);
    }
  }

  // Get the objective gradient array
  ParOptScalar *g;
  gvec->getArray(&g);

  // Allocate a temp array to store the pointers
  // to the constraint vector
  ParOptScalar **A = new ParOptScalar*[ m ];
  for ( int i = 0; i < m; i++ ){
    Avecs[i]->getArray(&A[i]);
  }

  // Get the coefficients for the objective/constraint approximation
  // information
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Get the move limit vectors
  ParOptScalar *alpha, *beta;
  alphavec->getArray(&alpha);
  betavec->getArray(&beta);

  // Compute the values of the lower/upper assymptotes
  for ( int j = 0; j < n; j++ ){
    // Compute the move limits to avoid division by zero
    alpha[j] = max2(lb[j], 0.9*L[j] + 0.1*x[j]);
    beta[j] = min2(ub[j], 0.9*U[j] + 0.1*x[j]);

    // Compute the coefficients for the objective
    p0[j] = max2(0.0, g[j])*(U[j] - x[j])*(U[j] - x[j]);
    q0[j] = -min2(0.0, g[j])*(x[j] - L[j])*(x[j] - L[j]);
  }

  if (use_true_mma){
    memset(b, 0, m*sizeof(ParOptScalar));
    for ( int i = 0; i < m; i++ ){
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      // Compute the coefficients for the constraints
      for ( int j = 0; j < n; j++ ){
        pi[j] = max2(0.0, A[i][j])*(U[j] - x[j])*(U[j] - x[j]);
        qi[j] = -min2(0.0, A[i][j])*(x[j] - L[j])*(x[j] - L[j]);
        b[i] += pi[j]/(U[j] - x[j]) + qi[j]/(x[j] - L[j]);
      }
    }

    // All reduce the coefficient values
    MPI_Allreduce(MPI_IN_PLACE, b, m, PAROPT_MPI_TYPE, MPI_SUM, comm);

    for ( int i = 0; i < m; i++ ){
      b[i] -= cons[i];
    }
  }

  // Check that the asymptotes, limits and variables are well-defined
  for ( int j = 0; j < n; j++ ){
    if (!(L[j] < alpha[j])){
      fprintf(stderr, "ParOptMMA: Inconsistent lower asymptote\n");
    }
    if (!(alpha[j] <= x[j])){
      fprintf(stderr, "ParOptMMA: Inconsistent lower limit\n");
    }
    if (!(x[j] <= beta[j])){
      fprintf(stderr, "ParOptMMA: Inconsistent upper limit\n");
    }
    if (!(beta[j] < U[j])){
      fprintf(stderr, "ParOptMMA: Inconsistent upper assymptote\n");
    }
  }

  // Increment the number of MMA iterations
  mma_iter++;

  // Free the A pointers
  delete [] A;

  return 0;
}

/*
  Create a design vector
*/
ParOptVec *ParOptMMA::createDesignVec(){
  return prob->createDesignVec(); 
}

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptMMA::createConstraintVec(){
  return prob->createConstraintVec();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptMMA::getMPIComm(){
  return prob->getMPIComm();
}

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptMMA::isDenseInequality(){
  return prob->isDenseInequality();
}

int ParOptMMA::isSparseInequality(){
  return prob->isSparseInequality();
}

int ParOptMMA::useLowerBounds(){
  return prob->useLowerBounds();
}

int ParOptMMA::useUpperBounds(){
  return prob->useUpperBounds();
}

// Get the variables and bounds from the problem
void ParOptMMA::getVarsAndBounds( ParOptVec *x, ParOptVec *lb, 
                                  ParOptVec *ub ){
  x->copyValues(xvec);
  lb->copyValues(alphavec);
  ub->copyValues(betavec);
}

/* 
  Evaluate the objective and constraints
*/
int ParOptMMA::evalObjCon( ParOptVec *xv, ParOptScalar *fval, 
                           ParOptScalar *cvals ){
  // Get the array of design variable values
  ParOptScalar *x, *x0;
  xvec->getArray(&x0);
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Compute the objective
  ParOptScalar fv = 0.0;
  for ( int j = 0; j < n; j++ ){
    fv += p0[j]/(U[j] - x[j]) + q0[j]/(x[j] - L[j]);
  }

  // Compute the linearized constraint
  memset(cvals, 0, m*sizeof(ParOptScalar));
  for ( int i = 0; i < m; i++ ){
    ParOptScalar *A;
    Avecs[i]->getArray(&A);
    for ( int j = 0; j < n; j++ ){
      cvals[i] += A[j]*(x[j] - x0[j]);
    }
  }

  // All reduce the data
  MPI_Allreduce(&fv, fval, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, cvals, m, PAROPT_MPI_TYPE, MPI_SUM, comm);

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptMMA::evalObjConGradient( ParOptVec *xv, ParOptVec *gv, 
                                   ParOptVec **Ac ){

  // Evaluate the gradient
  for ( int i = 0; i < m; i++ ){
    Ac[i]->copyValues(Avecs[i]);
  }

  // Get the gradient vector
  ParOptScalar *g;
  gv->getArray(&g);

  // Get the array of design variable values
  ParOptScalar *x;
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Compute the objective
  ParOptScalar fv = 0.0;
  for ( int j = 0; j < n; j++ ){
    ParOptScalar Uinv = 1.0/(U[j] - x[j]);
    ParOptScalar Linv = 1.0/(x[j] - L[j]);
    g[j] = Uinv*Uinv*p0[j] - Linv*Linv*q0[j];
  }

  return 0;
}

/*
  Evaluate the product of the Hessian with a given vector
*/
int ParOptMMA::evalHvecProduct( ParOptVec *xv, 
                                ParOptScalar *z, ParOptVec *zw,
                                ParOptVec *px, ParOptVec *hvec ){
  // Get the gradient vector
  ParOptScalar *h;
  hvec->getArray(&h);

  // Get the array of design variable values
  ParOptScalar *x;
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Get the components of the vector
  ParOptScalar *p;
  px->getArray(&p);

  // Compute the objective
  ParOptScalar fv = 0.0;
  for ( int j = 0; j < n; j++ ){
    ParOptScalar Uinv = 1.0/(U[j] - x[j]);
    ParOptScalar Linv = 1.0/(x[j] - L[j]);
    h[j] = 2.0*(Uinv*Uinv*Uinv*p0[j] + Linv*Linv*Linv*q0[j])*p[j];
  }

  return 0;
}

/* 
  Evaluate the constraints
*/
void ParOptMMA::evalSparseCon( ParOptVec *x, ParOptVec *out ){
  out->copyValues(cwvec);
  prob->addSparseJacobian(1.0, xvec, x, out);
  prob->addSparseJacobian(-1.0, xvec, xvec, out);
}

/* 
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptMMA::addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                                   ParOptVec *px, ParOptVec *out ){
  prob->addSparseJacobian(alpha, xvec, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptMMA::addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                            ParOptVec *pzw, ParOptVec *out ){
  prob->addSparseJacobianTranspose(alpha, xvec, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such 
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptMMA::addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                                       ParOptVec *cvec, ParOptScalar *A ){
  prob->addSparseInnerProduct(alpha, xvec, cvec, A);
}
