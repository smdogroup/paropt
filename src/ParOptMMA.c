#include <string.h>
#include "ComplexStep.h"
#include "ParOptBlasLapack.h"
#include "ParOptMMA.h"

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

  delete [] z;
  zwvec->decref();
  rvec->decref();
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
  alphavec->set(0.0);
  betavec->set(1.0);

  // Create the coefficient vectors
  p0vec = prob->createDesignVec();  p0vec->incref();
  q0vec = prob->createDesignVec();  q0vec->incref();

  // Set the sparse constraint vector to NULL 
  cwvec = NULL;

  if (use_true_mma){
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
    pivecs = NULL;
    qivecs = NULL;
    b = NULL;
  }

  cwvec = prob->createConstraintVec();
  if (cwvec){
    cwvec->incref();
  }

  // Get the design variables and bounds
  prob->getVarsAndBounds(xvec, lbvec, ubvec);

  // Set artificial bounds if none are provided
  if (!prob->useUpperBounds()){
    ubvec->set(10.0);
  }
  if (!prob->useLowerBounds()){
    lbvec->set(-9.0);
  }

  // Set the initial multipliers/values to zero
  z = new ParOptScalar[ m ];
  memset(z, 0, m*sizeof(ParOptScalar));
  zwvec = prob->createConstraintVec();

  // Create a sparse constraint vector
  rvec = prob->createDesignVec();
  rvec->incref();
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
  bound_relax = val;
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
  Set the new values of the multipliers
*/
void ParOptMMA::setMultipliers( ParOptScalar *_z, ParOptVec *_zw ){
  // Copy over the values of the multipliers
  memcpy(z, _z, m*sizeof(ParOptScalar));

  // Copy over the values of the constraint multipliers
  if (_zw){
    zwvec->copyValues(_zw);
  }
}

/*
  Compute the KKT error based on the current values of the multipliers
  set in ParOptMMA. If you do not update the multipliers, you will not
  get the correct KKT error.
*/
void ParOptMMA::computeKKTError( double *l1, 
                                 double *linfty,
                                 double *infeas ){
  // Get the lower/upper bounds for the variables
  ParOptScalar *lb, *ub;
  lbvec->getArray(&lb);
  ubvec->getArray(&ub);

  // Get the current values of the design variables
  ParOptScalar *x;
  xvec->getArray(&x);

  // Compute the KKT residual r = g - A^{T}*z
  rvec->copyValues(gvec);
  for ( int i = 0; i < m; i++ ){
    rvec->axpy(-z[i], Avecs[i]);
  }

  // If zw exists, compute r = r - Aw^{T}*zw
  if (zwvec){
    prob->addSparseJacobianTranspose(-1.0, xvec, zwvec, rvec);
  }

  // Set the infinity norms
  double l1_norm = 0.0;
  double infty_norm = 0.0;

  // Get the vector of values
  ParOptScalar *r;
  rvec->getArray(&r);

  for ( int j = 0; j < n; j++ ){
    // Check whether the bound multipliers would eliminate this
    // residual or not. If we're at the lower bound and the KKT
    // residual is negative or if we're at the upper bound and the KKT
    // residual is positive.
    if ((x[j] <= lb[j] + bound_relax) && r[j] >= 0.0){
      r[j] = 0.0;
    }
    if ((x[j] + bound_relax >= ub[j]) && r[j] <= 0.0){
      r[j] = 0.0;
    }

    // Add the contribution to the l1/infinity norms
    double t = fabs(RealPart(r[j]));
    l1_norm += t;
    if (t >= infty_norm){
      infty_norm = t;
    }
  }

  // Measure the infeasibility using the l1 norm
  *infeas = 0.0;
  for ( int i = 0; i < m; i++ ){
    *infeas += fabs(RealPart(min2(0.0, cons[i])));
  }

  // All-reduce the norms across all processors
  MPI_Allreduce(&l1_norm, l1, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&infty_norm, linfty, 1, MPI_DOUBLE, MPI_MAX, comm);
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
  Update and initialize data for the convex sub-problem that is solved
  at each iteration. This must be called before solving the dual
  problem.

  This code updates the asymptotes, sets the move limits and forms the
  approximations used in the MMA code.
*/
int ParOptMMA::initializeSubProblem( ParOptVec *xv ){
  x2vec->copyValues(x1vec);
  x1vec->copyValues(xvec);
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

  if (cwvec){
    prob->evalSparseCon(xvec, cwvec);
  }
  
  // Compute the KKT error, and print it out to a file
  if (print_level > 0){
    double l1, linfty, infeas;
    computeKKTError(&l1, &linfty, &infeas);

    if (fp){
      double l1_lambda = 0.0;
      for ( int i = 0; i < m; i++ ){
        l1_lambda += fabs(RealPart(z[i]));
      }      

      if (mma_iter % 10 == 0){
        fprintf(fp, "\n%5s %8s %15s %9s %9s %9s %9s\n",
                "MMA", "sub-iter", "fobj", "l1 opt", 
                "linft opt", "l1 lambd", "infeas");
      }
      fprintf(fp, "%5d %8d %15.6e %9.3e %9.3e %9.3e %9.3e\n",
              mma_iter, subproblem_iter, fobj, l1, 
              linfty, l1_lambda, infeas);
      fflush(fp);
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
    double eps = 0.0;
    p0[j] = max2(0.0, g[j])*(U[j] - x[j])*(U[j] - x[j]) + eps/(U[j] - L[j]);
    q0[j] = max2(0.0, -g[j])*(x[j] - L[j])*(x[j] - L[j]) + eps/(U[j] - L[j]);
  }

  if (use_true_mma){
    memset(b, 0, m*sizeof(ParOptScalar));
    for ( int i = 0; i < m; i++ ){
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      // Compute the coefficients for the constraints. Note that we use
      // a min here since the constraints in paropt are forumlated as
      // c(x) >= 0 -- so we need concave constraints
      for ( int j = 0; j < n; j++ ){
        pi[j] = min2(0.0, A[i][j])*(U[j] - x[j])*(U[j] - x[j]);
        qi[j] = min2(0.0, -A[i][j])*(x[j] - L[j])*(x[j] - L[j]);
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
  return 1;
}

int ParOptMMA::useUpperBounds(){
  return 1;
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

  if (use_true_mma){
    for ( int i = 0; i < m; i++ ){
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      // Compute the coefficients for the constraints
      for ( int j = 0; j < n; j++ ){
        cvals[i] += pi[j]/(U[j] - x[j]) + qi[j]/(x[j] - L[j]);
      }
    }
  }
  else {
    for ( int i = 0; i < m; i++ ){
      ParOptScalar *A;
      Avecs[i]->getArray(&A);
      for ( int j = 0; j < n; j++ ){
        cvals[i] += A[j]*(x[j] - x0[j]);
      }
    }
  }

  // All reduce the data
  MPI_Allreduce(&fv, fval, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, cvals, m, PAROPT_MPI_TYPE, MPI_SUM, comm);

  if (use_true_mma){
    for ( int i = 0; i < m; i++ ){
      cvals[i] -= b[i];
    }
  }
  else {
    for ( int i = 0; i < m; i++ ){
      cvals[i] += cons[i];
    }
  }

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptMMA::evalObjConGradient( ParOptVec *xv, ParOptVec *gv, 
                                   ParOptVec **Ac ){
  // Keep track of the number of subproblem gradient evaluations
  subproblem_iter++;

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

  // Compute the gradient of the objective
  for ( int j = 0; j < n; j++ ){
    ParOptScalar Uinv = 1.0/(U[j] - x[j]);
    ParOptScalar Linv = 1.0/(x[j] - L[j]);
    g[j] = Uinv*Uinv*p0[j] - Linv*Linv*q0[j];
  }

  // Evaluate the gradient
  if (use_true_mma){
    for ( int i = 0; i < m; i++ ){
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      ParOptScalar *A;
      Ac[i]->getArray(&A);

      for ( int j = 0; j < n; j++ ){
        ParOptScalar Uinv = 1.0/(U[j] - x[j]);
        ParOptScalar Linv = 1.0/(x[j] - L[j]);
        A[j] = Uinv*Uinv*pi[j] - Linv*Linv*qi[j];
      }
    }
  }
  else {
    for ( int i = 0; i < m; i++ ){
      Ac[i]->copyValues(Avecs[i]);
    }
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

  // Compute the hessian of the objective
  for ( int j = 0; j < n; j++ ){
    ParOptScalar Uinv = 1.0/(U[j] - x[j]);
    ParOptScalar Linv = 1.0/(x[j] - L[j]);
    h[j] = 2.0*(Uinv*Uinv*Uinv*p0[j] + Linv*Linv*Linv*q0[j])*p[j];
  }

  return 0;
}

/*
  Evaluate the diagonal Hessian matrix
*/
int ParOptMMA::evalHessianDiag( ParOptVec *xv, 
                                ParOptScalar *z, ParOptVec *zw, 
                                ParOptVec *hdiag ){
  // Get the gradient vector
  ParOptScalar *h;
  hdiag->getArray(&h);

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

  // Compute the Hessian diagonal
  for ( int j = 0; j < n; j++ ){
    ParOptScalar Uinv = 1.0/(U[j] - x[j]);
    ParOptScalar Linv = 1.0/(x[j] - L[j]);
    h[j] = 2.0*(Uinv*Uinv*Uinv*p0[j] + Linv*Linv*Linv*q0[j]);
  }

  if (use_true_mma){
    for ( int i = 0; i < m; i++ ){
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      for ( int j = 0; j < n; j++ ){
        ParOptScalar Uinv = 1.0/(U[j] - x[j]);
        ParOptScalar Linv = 1.0/(x[j] - L[j]);
        h[j] -= 2.0*z[i]*(Uinv*Uinv*Uinv*pi[j] + Linv*Linv*Linv*qi[j]);
      }
    }
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
