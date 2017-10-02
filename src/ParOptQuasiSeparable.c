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
ParOptMMA::ParOptMMA( ParOptProblem *_prob ){
  prob = _prob;
  prob->incref();

  // Pull out the communicator
  comm = prob->getMPIComm();

  // Set default parameters
  asymptote_relax = 0.9;

  // Get the problem sizes
  int nwcon, nwblock;
  prob->getProblemSizes(&n, &m, &nwcon, &nwblock);
  if (nwcon > 0){
    fprintf(stderr, 
            "ParOptMMA warning: Cannot solve probs with weight constraints\n");
  }

  // Set the iteration counter
  mma_iter = 0;

  // Initialize the data
  initialize();
}

/*
  Deallocate all of the internal data
*/
ParOptMMA::~ParOptMMA(){
  prob->decref();

  delete [] c;
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
  for ( int i = 0; i < m; i++ ){
    pivecs[i]->decref();
    qivecs[i]->decref();
  }
  delete [] qivecs;
  delete [] pivecs;
  delete [] b;
}

/*
  Allocate all of the data
*/
void ParOptMMA::initialize(){
  // Allocate initial values for the penalty parameters
  c = new ParOptScalar[ m ];
  for ( int i = 0; i < m; i++ ){
    c[i] = 1000.0;
  }

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

  pivecs = new ParOptVec*[ m ];
  qivecs = new ParOptVec*[ m ];
  for ( int i = 0; i < m; i++ ){
    pivecs[i] = prob->createDesignVec();  pivecs[i]->incref();
    qivecs[i] = prob->createDesignVec();  qivecs[i]->incref();
  }

  b = new ParOptScalar[ m ];
  memset(b, 0, m*sizeof(ParOptScalar));
}

/*
  Update the problem to find the new values for the design variables
*/
int ParOptMMA::update(){
  if (mma_iter == 0){
    prob->getVarsAndBounds(xvec, lbvec, ubvec);
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

  // Scale the constraints and gradients to match the 
  // standard MMA form for the constraint formation 
  // i.e. fi(x) <= 0.0. ParOpt uses the convention that
  // c(x) >= 0.0, so we multiply by -1.0
  for ( int i = 0; i < m; i++ ){
    cons[i] *= -1.0;
    Avecs[i]->scale(-1.0);
  }

  // Set up the sub-problem
  initSubProblem(mma_iter);
  mma_iter++;

  // Update the newest values of the design variables, and
  // make sure to store the previous updates
  x2vec->copyValues(x1vec);
  x1vec->copyValues(xvec);

  // Solve the dual problem
  solveDual();

  return 0;
}

/*
  Evaluate the gradient and Hessian of the dual sub-problem.

  This is collective on all processors. The function overwrites 
  the values stored in the grad and H matrices, and updates the
  values stored in the design vector and ys vectors.
*/
void ParOptMMA::evalDualGradient( ParOptScalar *grad,
                                  ParOptScalar *H,
                                  ParOptScalar *x,
                                  ParOptScalar *ys,
                                  const ParOptScalar *lambda,
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

  // Set the value of the y variables based on the dual variables
  for ( int i = 0; i < m; i++ ){
    ys[i] = 0.0;
    if (lambda[i] > c[i]){
      ys[i] = cons[i];
    }
  }

  // Allocate a temporary vector 
  ParOptScalar *tmp = new ParOptScalar[ m ];

  // Compute the new values of x and the gradient
  // of the Lagrangian w.r.t. lambda and the Hessian of
  // the Lagrangian based on  based on the
  for ( int j = 0; j < n; j++ ){
    // Compute the values
    ParOptScalar sL = p0[j];
    ParOptScalar sU = q0[j];
    for ( int i = 0; i < m; i++ ){
      sL += pi[i][j]*lambda[i];
      sU += qi[i][j]*lambda[i];
    }
    sL = sqrt(sL);
    sU = sqrt(sU);
    x[j] = (sL*L[j] + sU*U[j])/(sL + sU);

    if (x[j] < alpha[j]){
      x[j] = alpha[j];
    }
    else if (x[j] > beta[j]){
      x[j] = beta[j];
    }
    else {
      ParOptScalar Uinv = 1.0/(U[j] - x[j]);
      ParOptScalar Linv = 1.0/(x[j] - L[j]);
      for ( int i = 0; i < m; i++ ){
        grad[i] += Uinv*pi[i][j] + Linv*qi[i][j];
        tmp[i] = Uinv*Uinv*pi[i][j] - Linv*Linv*qi[i][j];
      }

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

  // Complete the contribution to the gradient
  for ( int i = 0; i < m; i++ ){
    grad[i] -= b[i] + ys[i];
  }
}

/*
  Solve the dual problem

  The approximation of the full problem is:

  min p0[j]/(U[j] - x[j]) + q0[j]/(xj - Lj) + c[i]*y[i]
  s.t. pi[j]/(U[j] - x[j]) + qi[j]/(xj - Lj) - y[i] <= b[i]
  y[i] >= 0

  The Lagrangian is:

  L = p0[j]/(U[j] - x[j]) + q0[j]/(xj - Lj) + c[i]*y[i]
  + lam[i]*(pi[j]/(U[j] - x[j]) + qi[j]/(xj - Lj) - y[i] - b[i])
  - mu[i]*y[i]
*/
void ParOptMMA::solveDual(){
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

  // Allocate space for the the slack varaibles
  ParOptScalar *y = new ParOptScalar[ m ];
  memset(y, 0, m*sizeof(ParOptScalar));

  // Allocate the vector for the dual variables
  ParOptScalar *lambda = new ParOptScalar[ m ];
  for ( int i = 0; i < m; i++ ){
    lambda[i] = 10.0;
  }

  // Set the gradient and Hessian
  ParOptScalar *grad = new ParOptScalar[ m ];
  ParOptScalar *H = new ParOptScalar[ m*m ];

  // Get the values of the design variables
  ParOptScalar *x;
  xvec->getArray(&x);

  // Allocate the pivot array
  int *ipiv = new int[ m ];

  // Set the initial barrier parameter
  int max_outer_iters = 20;
  int max_newton_iters = 100;
  double barrier = 1.0;
  double tau = 0.95;

  for ( int outer = 0; outer < max_outer_iters; outer++ ){
    for ( int k = 0; k < max_newton_iters; k++ ){
      evalDualGradient(grad, H, x, y, lambda, p0, q0, pi, qi,
                       L, U, alpha, beta);

      // Check the norm
      ParOptScalar res_norm = 0.0;
      for ( int i = 0; i < m; i++ ){
        ParOptScalar t = fabs(RealPart(grad[i])); 
        if (i == 0){
          res_norm = t;
        }
        else if (t > res_norm){
          res_norm = t;
        }
      }

      if (res_norm < 10.0*barrier){
        break;
      }

      // Add the terms from the barrier, enforcing lambda >= 0
      for ( int i = 0; i < m; i++ ){
        grad[i] = -grad[i] - barrier/lambda[i];
        H[i*(m+1)] -= barrier/(lambda[i]*lambda[i]);
      }

      // Compute the step length by solving the system of equations
      int one = 1;
      int info = 0;
      LAPACKdgetrf(&m, &m, H, &m, ipiv, &info);
      LAPACKdgetrs("N", &m, &one, H, &m, ipiv, grad, &m, &info);

      // Truncate the step length, depending on the step-to-the-boundary
      // rule with tau as the factor
      double max_step = 1.0;
      for ( int i = 0; i < m; i++ ){
        if (grad[i] < 0.0){
          double step = -tau*RealPart(lambda[i]/grad[i]);
          if (step < max_step){
            max_step = step;
          }
        }
      }

      // Update the multipliers
      for ( int i = 0; i < m; i++ ){
        lambda[i] += max_step*grad[i];
      }
    }

    // Reduce the barrier parameter
    barrier *= 0.25;
  }

  delete [] grad;
  delete [] H;
  delete [] ipiv;
  delete [] pi;
  delete [] qi;
  delete [] y;
  delete [] lambda;
}

/*
  Update and initialize data for the convex sub-problem that
  is solved at each iteration. This must be called before
  solving the dual problem.

  This code updates the asymptotes, sets the move limits and
  forms the approximations used in the MMA code.
*/
void ParOptMMA::initSubProblem( int iter ){
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

  // Set the assymptote relaxation factors
  const double s = asymptote_relax;
  const double sinv = 1.0/asymptote_relax;

  // Set all of the asymptote values
  if (iter < 2){
    for ( int j = 0; j < n; j++ ){
      L[j] = x[j] - 0.5*(x[j] - lb[j]);
      U[j] = x[j] + 0.5*(ub[j] - x[j]);
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

      if (RealPart(indc) < 0.0){
        // oscillation -> contract the asymptotes
        L[j] = x[j] - s*(x1[j] - Lprev);
        U[j] = x[j] + s*(Uprev - x1[j]);
      }
      else {
        // Expand the asymptotes
        L[j] = x[j] - sinv*(x1[j] - Lprev);
        U[j] = x[j] + sinv*(Uprev - x1[j]);        
      }
    }
  }

  // Get the constraint gradient array
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

  // Allocate pointers for the constraint pointers
  ParOptScalar **pi = new ParOptScalar*[ m ];
  ParOptScalar **qi = new ParOptScalar*[ m ];
  for ( int i = 0; i < m; i++ ){
    pivecs[i]->getArray(&pi[i]);
    qivecs[i]->getArray(&qi[i]);
  }

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
    q0[j] = min2(0.0, g[j])*(x[j] - L[j])*(x[j] - L[j]);

    // Compute the coefficients for the constraints
    for ( int i = 0; i < m; i++ ){
      pi[i][j] = max2(0.0, A[i][j])*(U[j] - x[j])*(U[j] - x[j]);
      qi[i][j] = min2(0.0, A[i][j])*(x[j] - L[j])*(x[j] - L[j]);
    }
  }

  // Free the A pointers
  delete [] A;
  delete [] qi;
  delete [] pi;

  // Compute the b-coefficients
  for ( int i = 0; i < m; i++ ){
    for ( int j = 0; j < n; j++ ){
      b[i] += pi[i][j]/(U[j] - x[j]) + qi[i][j]/(x[j] - L[j]);
    }
  }

  // All reduce the coefficient values
  MPI_Allreduce(MPI_IN_PLACE, b, m, PAROPT_MPI_TYPE, MPI_SUM, comm);

  for ( int i = 0; i < m; i++ ){
    b[i] = -cons[i];
  }
}