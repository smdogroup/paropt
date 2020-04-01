#include "ParOptTrustRegion.h"
#include "ParOptComplexStep.h"

/*
  Summary of the different trust region algorithm options
*/
static const int NUM_TRUST_REGION_PARAMS = 10;
static const char *trust_regions_parameter_help[][2] = {
  {"tr_size",
   "Float: Initial trust region radius size"},

  {"tr_min_size",
   "Float: Minimum trust region radius size"},

  {"tr_max_size",
   "Float: Maximum trust region radius size"},

  {"eta",
   "Float: Trust region step acceptance ratio of actual/predicted improvement"},

  {"bound_relax",
   "Float: Relax the bounds by this tolerance when computing KKT errors"},

  {"adaptive_gamma_update",
   "Boolean: Adaptively update the trust region "},

  {"max_tr_iterations",
   "Integer: Maximum number of trust region radius steps"},

  {"l1_tol",
   "Float: Convergence tolerance for the optimality error in the l1 norm"},

  {"linfty_tol",
   "Float: Convergence tolerance for the optimality error in the l-infinity norm"},

  {"infeas_tol",
   "Float: Convergence tolerance for feasibility in the l1 norm"}};

// Helper functions
inline ParOptScalar min2( ParOptScalar a, ParOptScalar b ){
  if (ParOptRealPart(a) < ParOptRealPart(b)){
    return a;
  }
  else {
    return b;
  }
}

inline ParOptScalar max2( ParOptScalar a, ParOptScalar b ){
  if (ParOptRealPart(a) > ParOptRealPart(b)){
    return a;
  }
  else {
    return b;
  }
}

ParOptQuadraticSubproblem::ParOptQuadraticSubproblem( ParOptProblem *_prob,
                                                      ParOptCompactQuasiNewton *_qn ):
  ParOptTrustRegionSubproblem(_prob){

  // Paropt problem instance
  prob = _prob;
  prob->incref();

  // Get the problem sizes
  prob->getProblemSizes(&n, &m, &nwcon, &nwblock);
  setProblemSizes(n, m, nwcon, nwblock);

  // Set the quasi-Newton method
  if (_qn){
    qn = _qn;
    qn->incref();
  }
  else {
    qn = NULL;
  }

  // Create the vectors
  xk = prob->createDesignVec();  xk->incref();
  lk = prob->createDesignVec();  lk->incref();
  uk = prob->createDesignVec();  uk->incref();
  lb = prob->createDesignVec();  lb->incref();
  ub = prob->createDesignVec();  ub->incref();

  // Set default values for now before initialize
  lk->set(0.0);  uk->set(1.0);
  lb->set(0.0);  ub->set(1.0);
  xk->set(0.5);

  // Create the constraint Jacobian vectors
  fk = 0.0;
  gk = prob->createDesignVec();
  gk->incref();
  ck = new ParOptScalar[ m ];
  Ak = new ParOptVec*[ m ];
  for ( int i = 0; i < m; i++ ){
    Ak[i] = prob->createDesignVec();
    Ak[i]->incref();
  }

  // Create a temporary set of vectors
  ft = 0.0;
  gt = prob->createDesignVec();
  gt->incref();
  ct = new ParOptScalar[ m ];
  At = new ParOptVec*[ m ];
  for ( int i = 0; i < m; i++ ){
    At[i] = prob->createDesignVec();
    At[i]->incref();
  }

  // Create the temporary vector
  s = prob->createDesignVec();  s->incref();
  t = prob->createDesignVec();  t->incref();
}

ParOptQuadraticSubproblem::~ParOptQuadraticSubproblem(){
  prob->decref();
  if (qn){
    qn->decref();
  }

  xk->decref();
  gk->decref();
  delete [] ck;
  for ( int i = 0; i < m; i++ ){
    Ak[i]->decref();
  }
  delete [] Ak;

  delete [] ct;
  gt->decref();
  for ( int i = 0; i < m; i++ ){
    At[i]->decref();
  }
  delete [] At;

  s->decref();
  t->decref();
}

/*
  Return the quasi-Newton approximation of the objective
*/
ParOptCompactQuasiNewton* ParOptQuadraticSubproblem::getQuasiNewton(){
  return qn;
}

/*
  Initialize the model at the starting point
*/
void ParOptQuadraticSubproblem::initModelAndBounds( double tr_size ){
  // Get the lower/upper bounds
  prob->getVarsAndBounds(xk, lb, ub);

  // Set the lower/upper bounds for the trust region
  setTrustRegionBounds(tr_size);

  // Evaluate objective constraints and gradients
  prob->evalObjCon(xk, &fk, ck);
  prob->evalObjConGradient(xk, gk, Ak);
}

/*
  Set the trust region bounds
*/
void ParOptQuadraticSubproblem::setTrustRegionBounds( double tr_size ){
  ParOptScalar *xvals;
  ParOptScalar *lvals, *uvals;
  ParOptScalar *ltrvals, *utrvals;

  int size = xk->getArray(&xvals);
  lb->getArray(&lvals);
  ub->getArray(&uvals);
  lk->getArray(&ltrvals);
  uk->getArray(&utrvals);

  for ( int i = 0; i < size; i++ ){
    ltrvals[i] = max2(xvals[i] - tr_size, lvals[i]);
    utrvals[i] = min2(xvals[i] + tr_size, uvals[i]);
  }
}

int ParOptQuadraticSubproblem::evalTrialPointAndUpdate( ParOptVec *x,
                                                        const ParOptScalar *z,
                                                        ParOptVec *zw,
                                                        ParOptScalar *fobj,
                                                        ParOptScalar *cons ){
  int fail = prob->evalObjCon(x, &ft, ct);
  fail = fail || prob->evalObjConGradient(x, gt, At);

  // If we're using a quasi-Newton Hessian approximation
  if (qn){
    // Compute the step s = x - xk
    s->copyValues(x);
    s->axpy(-1.0, xk);

    // Compute the difference between the gradient of the
    // Lagrangian between the current point and the previous point
    t->copyValues(gt);
    for ( int i = 0; i < m; i++ ){
      t->axpy(-z[i], At[i]);
    }
    if (nwcon > 0){
      prob->addSparseJacobianTranspose(-1.0, x, zw, t);
    }

    t->axpy(-1.0, gk);
    for ( int i = 0; i < m; i++ ){
      t->axpy(z[i], Ak[i]);
    }
    if (nwcon > 0){
      prob->addSparseJacobianTranspose(1.0, xk, zw, t);
    }

    // Perform an update of the quasi-Newton approximation
    qn->update(xk, z, zw, s, t);
  }

  return fail;
}

int ParOptQuadraticSubproblem::acceptTrialPoint( ParOptVec *x,
                                                 const ParOptScalar *z,
                                                 ParOptVec *zw ){
  int fail = 0;

  fk = ft;
  xk->copyValues(x);
  gk->copyValues(gt);
  for ( int i = 0; i < m; i++ ){
   ck[i] = ct[i];
   Ak[i]->copyValues(At[i]);
  }

  return fail;
}

void ParOptQuadraticSubproblem::rejectTrialPoint(){
  ft = 0.0;
  for ( int i = 0; i < m; i++ ){
   ct[i] = 0.0;
  }
}

/*
  Create a design vector
*/
ParOptVec *ParOptQuadraticSubproblem::createDesignVec(){
  return prob->createDesignVec();
}

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptQuadraticSubproblem::createConstraintVec(){
  return prob->createConstraintVec();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptQuadraticSubproblem::getMPIComm(){
  return prob->getMPIComm();
}

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptQuadraticSubproblem::isDenseInequality(){
  return prob->isDenseInequality();
}

int ParOptQuadraticSubproblem::isSparseInequality(){
  return prob->isSparseInequality();
}

int ParOptQuadraticSubproblem::useLowerBounds(){
  return 1;
}

int ParOptQuadraticSubproblem::useUpperBounds(){
  return 1;
}

// Get the variables and bounds from the problem
void ParOptQuadraticSubproblem::getVarsAndBounds( ParOptVec *x,
                                                  ParOptVec *l,
                                                  ParOptVec *u ){
  x->copyValues(xk);
  l->copyValues(lk);
  u->copyValues(uk);
}

/*
  Evaluate the objective and constraint functions
*/
int ParOptQuadraticSubproblem::evalObjCon( ParOptVec *x,
                                           ParOptScalar *fobj,
                                           ParOptScalar *cons ){
  if (x){
    s->copyValues(x);
    s->axpy(-1.0, xk);

    // Compute the objective function
    *fobj = fk + gk->dot(s);
    if (qn){
      qn->mult(s, t);
      *fobj += 0.5*s->dot(t);
    }

    // Compute the constraint functions
    for ( int i = 0; i < m; i++ ){
      cons[i] = ck[i] + Ak[i]->dot(s);
    }
  }
  else {
    // If x is NULL, assume x = xk
    *fobj = fk;

    for ( int i = 0; i < m; i++ ){
      cons[i] = ck[i];
    }
  }

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptQuadraticSubproblem::evalObjConGradient( ParOptVec *x,
                                                   ParOptVec *g,
                                                   ParOptVec **Ac ){
  // Copy the values of constraint gradient
  for ( int i = 0; i < m; i++ ){
    Ac[i]->copyValues(Ak[i]);
  }

  s->copyValues(x);
  s->axpy(-1.0, xk);

  // Evaluate the gradient of the quadratic objective
  if (qn){
    qn->mult(s, g);
    g->axpy(1.0, gk);
  }
  else {
    g->copyValues(gk);
  }

  return 0;
}

/*
  Evaluate the constraints
*/
void ParOptQuadraticSubproblem::evalSparseCon( ParOptVec *x,
                                               ParOptVec *out ){
  prob->evalSparseCon(xk, out);
  prob->addSparseJacobian(1.0, xk, x, out);
  prob->addSparseJacobian(-1.0, xk, xk, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptQuadraticSubproblem::addSparseJacobian( ParOptScalar alpha,
                                                   ParOptVec *x,
                                                   ParOptVec *px,
                                                   ParOptVec *out ){
  prob->addSparseJacobian(alpha, xk, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptQuadraticSubproblem::addSparseJacobianTranspose( ParOptScalar alpha,
                                                            ParOptVec *x,
                                                            ParOptVec *pzw,
                                                            ParOptVec *out ){
  prob->addSparseJacobianTranspose(alpha, xk, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptQuadraticSubproblem::addSparseInnerProduct( ParOptScalar alpha,
                                                       ParOptVec *x,
                                                       ParOptVec *cvec,
                                                       ParOptScalar *A ){
  prob->addSparseInnerProduct(alpha, xk, cvec, A);
}

/*
  Get the model at the current point
*/
int ParOptQuadraticSubproblem::getLinearModel( ParOptVec **_xk,
                                               ParOptVec **_gk,
                                               ParOptVec ***_Ak,
                                               ParOptVec **_lb,
                                               ParOptVec **_ub ){
  if (_xk){ *_xk = xk; }
  if (_gk){ *_gk = gk; }
  if (_Ak){ *_Ak = Ak; }
  if (_lb){ *_lb = lb; }
  if (_ub){ *_ub = ub; }

  return m;
}

/**
  A parallel optimizer with trust region globalization

  The following code implements a trust region method. The algorithm
  uses an l1 penalty function for the constraints with an l-infinity (box)
  constraint for the trust region. The trust region sub-problems are
  solved at each step by an instance of the ParOptInteriorPoint optimizer.

  @param _subproblem the ParOptTrustRegionSubproblem class
  @param _tr_size the initial trust-region size
  @param _tr_min_size the minimum trust-region size
  @param _tr_max_size the maximum trust-region size
  @param _eta the trust-region acceptance parameter
  @param _penalty_value the initial l1 penalty paramter value
  @param _bound_relax bound relaxation parameter
*/
ParOptTrustRegion::ParOptTrustRegion( ParOptTrustRegionSubproblem *_subproblem,
                                      double _tr_size,
                                      double _tr_min_size,
                                      double _tr_max_size,
                                      double _eta,
                                      double _penalty_value,
                                      double _bound_relax ){
  // Create the sub-problem instance
  subproblem = _subproblem;
  subproblem->incref();

  // Get the subproblem sizes
  subproblem->getProblemSizes(&n, &m, &nwcon, &nwblock);

  // Set the solution parameters
  tr_size = _tr_size;
  tr_min_size = _tr_min_size;
  tr_max_size = _tr_max_size;
  eta = _eta;
  bound_relax = _bound_relax;

  // Set the default output parameters
  write_output_frequency = 10;

  // Set default values for the convergence parameters
  adaptive_gamma_update = 1;
  max_tr_iterations = 200;
  l1_tol = 1e-6;
  linfty_tol = 1e-6;
  infeas_tol = 1e-5;
  penalty_gamma_max = 1e4;

  // Set the penalty parameters
  penalty_gamma = new double[ m ];
  for ( int i = 0; i < m; i++ ){
    penalty_gamma[i] = _penalty_value;
  }

  // Set the iteration count to zero
  iter_count = 0;

  // Set the file pointer to NULL
  fp = NULL;
  print_level = 0;

  // Create a temporary vector
  t = subproblem->createDesignVec();
  t->incref();
}

/**
  Delete the trust region object
*/
ParOptTrustRegion::~ParOptTrustRegion(){
  subproblem->decref();

  delete [] penalty_gamma;
  t->decref();

  // Close the file when we quit
  if (fp){
    fclose(fp);
  }
}

/**
  Set the output file (only on the root proc)

  @param filename the output file name
*/
void ParOptTrustRegion::setOutputFile( const char *filename ){
  int rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &rank);
  if (rank == 0){
    if (fp && fp != stdout){
      fclose(fp);
    }
    fp = fopen(filename, "w");

    if (fp){
      fprintf(fp, "ParOptTrustRegion: Parameter summary\n");
      for ( int i = 0; i < NUM_TRUST_REGION_PARAMS; i++ ){
        fprintf(fp, "%s\n%s\n\n",
                trust_regions_parameter_help[i][0],
                trust_regions_parameter_help[i][1]);
      }
    }
  }
}

/**
  Set the print level

  @param _print_level the integer print level
*/
void ParOptTrustRegion::setPrintLevel( int _print_level ){
  print_level = _print_level;
}

/**
  Write the parameters to the output file

  @param fp an open file handle
*/
void ParOptTrustRegion::printOptionSummary( FILE *fp ){
  int rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &rank);
  if (rank == 0){
    fprintf(fp, "ParOptTrustRegion options summary:\n");
    fprintf(fp, "%-30s %15g\n", "tr_size", tr_size);
    fprintf(fp, "%-30s %15g\n", "tr_min_size", tr_min_size);
    fprintf(fp, "%-30s %15g\n", "tr_max_size", tr_max_size);
    fprintf(fp, "%-30s %15g\n", "eta", eta);
    double value = 0.0;
    for ( int i = 0; i < m; i++ ){
      value += penalty_gamma[i];
    }
    fprintf(fp, "%-30s %15g\n", "avg(penalty_gamma)", value/m);
    fprintf(fp, "%-30s %15g\n", "bound_relax", bound_relax);
    fprintf(fp, "%-30s %15d\n", "adaptive_gamma_update", adaptive_gamma_update);
    fprintf(fp, "%-30s %15d\n", "max_tr_iterations", max_tr_iterations);
    fprintf(fp, "%-30s %15g\n", "l1_tol", l1_tol);
    fprintf(fp, "%-30s %15g\n", "linfty_tol", linfty_tol);
    fprintf(fp, "%-30s %15g\n", "infeas_tol", infeas_tol);
    fprintf(fp, "%-30s %15g\n", "gamma_max", penalty_gamma_max);
  }
}

/**
  Set whether or not to adaptively update the penalty parameters

  @param truth flag to indicate whether or not to use adaptive penalty
*/
void ParOptTrustRegion::setAdaptiveGammaUpdate( int truth ){
  adaptive_gamma_update = truth;
}

/**
  Set the maximum number of trust region steps

  @param max_iters the maximum number of trust region iterations
*/
void ParOptTrustRegion::setMaxTrustRegionIterations( int max_iters ){
  max_tr_iterations = max_iters;
}

/**
  Set the trust region stopping criterion values.

  The trust region algorithm terminates when the infeasibility tolerance and
  either the l1 or l-infinity tolerances are satisfied.

  @param _infeas_tol the infeasibility tolerance
  @param _l1_tol the l1 norm
*/
void ParOptTrustRegion::setTrustRegionTolerances( double _infeas_tol,
                                                  double _l1_tol,
                                                  double _linfty_tol ){
  infeas_tol = _infeas_tol;
  l1_tol = _l1_tol;
  linfty_tol = _linfty_tol;
}

/**
   Set the penalty parameter for the l1 penalty function.

   @param gamma is the value of the penalty parameter
*/
void ParOptTrustRegion::setPenaltyGamma( double gamma ){
  if (gamma >= 0.0){
    for ( int i = 0; i < m; i++ ){
      penalty_gamma[i] = gamma;
    }
  }
}

/**
   Set the individual penalty parameters for the l1 penalty function.

   @param gamma is the array of penalty parameter values.
*/
void ParOptTrustRegion::setPenaltyGamma( const double *gamma ){
  for ( int i = 0; i < m; i++ ){
    if (gamma[i] >= 0.0){
      penalty_gamma[i] = gamma[i];
    }
  }
}

/**
   Retrieve the penalty parameter values.

   @param _penalty_gamma is the array of penalty parameter values.
*/
int ParOptTrustRegion::getPenaltyGamma( const double **_penalty_gamma ){
  if (_penalty_gamma){ *_penalty_gamma = penalty_gamma;}
  return m;
}

/**
  Set the maximum value of the penalty parameters

  @param _gamma_max the maximum penalty value
*/
void ParOptTrustRegion::setPenaltyGammaMax( double _gamma_max ){
  penalty_gamma_max = _gamma_max;
}

/**
  Set the output frequency

  @param _write_output_frequency frequency with which ouput is written
*/
void ParOptTrustRegion::setOutputFrequency( int _write_output_frequency ){
  write_output_frequency = _write_output_frequency;
}

/**
  Initialize the problem
*/
void ParOptTrustRegion::initialize(){
  subproblem->initModelAndBounds(tr_size);

  // Set the iteration count to zero
  iter_count = 0;

  int mpi_rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &mpi_rank);
  if (mpi_rank == 0){
    if (fp){
      printOptionSummary(fp);
    }
    else {
      printOptionSummary(stdout);
    }
  }
}

/**
  Update the trust region problem
*/
void ParOptTrustRegion::update( ParOptVec *xt,
                                const ParOptScalar *z,
                                ParOptVec *zw,
                                double *infeas,
                                double *l1,
                                double *linfty ){
  // Get the mpi rank for printing
  int mpi_rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &mpi_rank);

  // Compute the value of the objective model and model
  // constraints at x = xk (The current iterate)
  ParOptScalar fk;
  ParOptScalar *ck = new ParOptScalar[ m ];
  subproblem->evalObjCon(NULL, &fk, ck);

  // Compute the value of the objective model and model
  // constraints at x = xt (The trial point)
  ParOptScalar ft;
  ParOptScalar *ct = new ParOptScalar[ m ];
  subproblem->evalObjCon(xt, &ft, ct);

  // Compute the reduction in the objective value
  ParOptScalar obj_reduc = fk - ft;

  // Compute the model infeasibility
  ParOptScalar infeas_k = 0.0;
  ParOptScalar infeas_model = 0.0;
  for ( int i = 0; i < m; i++ ){
    infeas_k += penalty_gamma[i]*max2(0.0, -ck[i]);
    infeas_model += penalty_gamma[i]*max2(0.0, -ct[i]);
  }

  // Evaluate the model at the trial point and update the trust region model
  // Hessian and bounds. Note that here, we're re-using the ft/ct memory.
  subproblem->evalTrialPointAndUpdate(xt, z, zw, &ft, ct);

  // Compute the infeasibilities of the last two iterations
  ParOptScalar infeas_t = 0.0;
  for ( int i = 0; i < m; i++ ){
    infeas_t += penalty_gamma[i]*max2(0.0, -ct[i]);
  }

  // Compute the actual reduction and the predicted reduction
  ParOptScalar actual_reduc =
    (fk - ft + (infeas_k - infeas_t));
  ParOptScalar model_reduc =
    obj_reduc + (infeas_k - infeas_model);

  if (mpi_rank == 0 && print_level > 0){
    FILE *outfp = stdout;
    if (fp){
      outfp = fp;
    }
    fprintf(outfp, "%-12s %2s %9s %9s %9s\n",
            "Constraints", "i", "c(x)", "c(x+p)", "gamma");
    for ( int i = 0; i < m; i++ ){
      fprintf(outfp, "%12s %2d %9.2e %9.2e %9.2e\n",
              " ", i, ParOptRealPart(ck[i]), ParOptRealPart(ct[i]),
              penalty_gamma[i]);
    }
    fprintf(outfp, "\n%-15s %9s %9s %9s %9s\n",
            "Model", "ared(f)", "pred(f)", "ared(c)", "pred(c)");
    fprintf(outfp, "%15s %9.2e %9.2e %9.2e %9.2e\n",
            " ", ParOptRealPart(fk - ft), ParOptRealPart(obj_reduc),
            ParOptRealPart(infeas_k - infeas_t),
            ParOptRealPart(infeas_k - infeas_model));
  }
  delete [] ck;
  delete [] ct;

  // Compute the ratio
  ParOptScalar rho = actual_reduc/model_reduc;

  // Compute the infeasibility
  ParOptScalar infeas_new = 0.0;
  for ( int i = 0; i < m; i++ ){
    infeas_new += max2(0.0, -ct[i]);
  }
  *infeas = ParOptRealPart(infeas_new);

  // Compute the max absolute value
  double smax = 0.0;

  // Check whether to accept the new point or not. If the trust region
  // radius size is at the lower bound, the step is always accepted
  if (ParOptRealPart(rho) >= eta || tr_size <= tr_min_size){
    // Compute the length of the step for log entry purposes
    ParOptVec *xk;
    subproblem->getLinearModel(&xk);
    t->copyValues(xt);
    t->axpy(-1.0, xk);
    smax = ParOptRealPart(t->maxabs());

    subproblem->acceptTrialPoint(xt, z, zw);
  }
  else {
    // Set the step size to zero (rejected step)
    subproblem->rejectTrialPoint();
    smax = 0.0;
  }

  // Compute the KKT error at the current point
  computeKKTError(z, zw, l1, linfty);

  // Set the new trust region radius
  if (ParOptRealPart(rho) < 0.25){
    tr_size = ParOptRealPart(max2(0.25*tr_size, tr_min_size));
  }
  else if (ParOptRealPart(rho) > 0.75){
    tr_size = ParOptRealPart(min2(1.5*tr_size, tr_max_size));
  }

  // Reset the trust region radius bounds
  subproblem->setTrustRegionBounds(tr_size);

  // Keep track of the max z/average z and max gamma/average gamma
  double zmax = 0.0, zav = 0.0, gmax = 0.0, gav = 0.0;
  for ( int i = 0; i < m; i++ ){
    zav += ParOptRealPart(z[i]);
    gav += penalty_gamma[i];
    if (ParOptRealPart(z[i]) > zmax){
      zmax = ParOptRealPart(z[i]);
    }
    if (penalty_gamma[i] > gmax){
      gmax = penalty_gamma[i];
    }
  }

  if (mpi_rank == 0){
    FILE *outfp = stdout;
    if (fp){
      outfp = fp;
    }
    if (iter_count % 10 == 0 || print_level > 0){
      fprintf(outfp,
              "\n%5s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
              "iter", "fobj", "infeas", "l1", "linfty", "|x - xk|", "tr",
              "rho", "mod red.", "avg z", "max z", "avg pen.", "max pen.");
      fflush(outfp);
    }
    fprintf(outfp,
            "%5d %12.5e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e "
            "%9.2e %9.2e %9.2e %9.2e\n",
            iter_count, ParOptRealPart(fk), *infeas, *l1, *linfty, smax, tr_size,
            ParOptRealPart(rho), ParOptRealPart(model_reduc),
            zav/m, zmax, gav/m, gmax);
    fflush(outfp);
  }

  // Update the iteration counter
  iter_count++;
}

/**
  Perform the optimization

  This performs all steps in the optimization: initialization, trust-region
  updates and quasi-Newton updates. This should be called once In some cases, you 

  @param optimizer the instance of the ParOptInteriorPoint optimizer
*/
void ParOptTrustRegion::optimize( ParOptInteriorPoint *optimizer ){
  if (optimizer->getOptProblem() != subproblem){
    fprintf(stderr,
            "ParOptTrustRegion: The optimizer must be associated with "
            "the subproblem object\n");
    return;
  }

  // Set up the optimizer so that it uses the quasi-Newton approximation
  ParOptCompactQuasiNewton *qn = subproblem->getQuasiNewton();
  optimizer->setQuasiNewton(qn);

  // Do not update the Hessian within the interior-point method
  optimizer->setUseQuasiNewtonUpdates(0);

  // Set the initial values for gamma
  optimizer->setPenaltyGamma(penalty_gamma);

  // Allocate arrays to store infeasibility information
  ParOptScalar *con_infeas = NULL;
  ParOptScalar *model_con_infeas = NULL;
  ParOptScalar *best_con_infeas = NULL;
  if (adaptive_gamma_update){
    con_infeas = new ParOptScalar[ m ];
    model_con_infeas = new ParOptScalar[ m ];
    best_con_infeas = new ParOptScalar[ m ];
  }

  // Get the MPI rank
  int mpi_rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &mpi_rank);

  // Initialize the trust region problem for the first iteration
  initialize();

  // Iterate over the trust region subproblem until convergence
  for ( int i = 0; i < max_tr_iterations; i++ ){
    if (adaptive_gamma_update){
      // Set the penalty parameter to a large value
      double gamma = 1e6;
      if (1e2*penalty_gamma_max > gamma){
        gamma = 1e2*penalty_gamma_max;
      }
      optimizer->setPenaltyGamma(gamma);

      // Initialize the barrier parameter
      optimizer->setInitBarrierParameter(10.0);
      optimizer->resetDesignAndBounds();

      // Optimize the subproblem
      optimizer->optimize();

      // Get the design variables
      ParOptVec *x, *zw;
      ParOptScalar *z;
      optimizer->getOptimizedPoint(&x, &z, &zw, NULL, NULL);

      // Evaluate the model at the best point to obtain the infeasibility
      ParOptScalar fbest;
      subproblem->evalObjCon(x, &fbest, best_con_infeas);

      // Compute the best-case infeasibility achieved by setting the
      // penalty parameters to a large value
      for ( int j = 0; j < m; j++ ){
        best_con_infeas[j] = max2(0.0, -best_con_infeas[j]);
      }

      // Set the penalty parameters
      optimizer->setPenaltyGamma(penalty_gamma);
    }

    // Print out the current solution progress using the
    // hook in the problem definition
    if (i % write_output_frequency == 0){
      ParOptVec *xk;
      subproblem->getLinearModel(&xk);
      subproblem->writeOutput(i, xk);
    }

    // Initialize the barrier parameter
    optimizer->setInitBarrierParameter(10.0);
    optimizer->resetDesignAndBounds();

    // Optimize the subproblem
    optimizer->optimize();

    // Get the design variables
    ParOptVec *x, *zw;
    ParOptScalar *z;
    optimizer->getOptimizedPoint(&x, &z, &zw, NULL, NULL);

    if (adaptive_gamma_update){
      // Find the infeasibility at the origin x = xk
      ParOptScalar f0;
      subproblem->evalObjCon(NULL, &f0, con_infeas);

      // Find the actual infeasibility reduction
      ParOptScalar fmodel;
      subproblem->evalObjCon(x, &fmodel, model_con_infeas);
      for ( int j = 0; j < m; j++ ){
        con_infeas[j] = max2(0.0, -con_infeas[j]);
        model_con_infeas[j] = max2(0.0, -model_con_infeas[j]);
      }
    }

    // Update the trust region based on the performance at the new
    // point.
    double infeas, l1, linfty;
    update(x, z, zw, &infeas, &l1, &linfty);

    // Check for convergence of the trust region problem
    if (infeas < infeas_tol){
      if (l1 < l1_tol ||
          linfty < linfty_tol){
        // Success!
        break;
      }
    }

    // Adapat the penalty parameters
    if (adaptive_gamma_update){
      FILE *outfp = stdout;
      if (fp){
        outfp = fp;
      }
      if (mpi_rank == 0 && print_level > 0){
        fprintf(outfp, "%-12s %2s %9s %9s %9s %9s %9s %9s %9s\n",
                "Penalty", "i", "|c(x)|", "|c+Ap|", "min|c+Ap|",
                "pred", "min. pred", "gamma", "update");
      }

      for ( int i = 0; i < m; i++ ){
        // Compute the actual infeasibility reduction and the best
        // possible infeasibility reduction
        double infeas_reduction = ParOptRealPart(con_infeas[i] - model_con_infeas[i]);
        double best_reduction = ParOptRealPart(con_infeas[i] - best_con_infeas[i]);

        char info[8];
        if (print_level > 0){
          sprintf(info, "---");
        }

        // If the ratio of the predicted to actual improvement is good,
        // and the constraints are satisfied, decrease the penalty
        // parameter. Otherwise, if the best case infeasibility is
        // significantly better, increase the penalty parameter.
        if (ParOptRealPart(z[i]) > infeas_tol &&
            ParOptRealPart(con_infeas[i]) < infeas_tol &&
            penalty_gamma[i] >= 2.0*ParOptRealPart(z[i])){
          // Reduce gamma
          penalty_gamma[i] = 0.5*(penalty_gamma[i] + ParOptRealPart(z[i]));
          if (print_level > 0){
            sprintf(info, "decr");
          }
        }
        else if (ParOptRealPart(con_infeas[i]) > infeas_tol &&
                 0.995*best_reduction > infeas_reduction){
          // Increase gamma
          penalty_gamma[i] = ParOptRealPart(min2(1.5*penalty_gamma[i],
                                                 penalty_gamma_max));
          if (print_level > 0){
            sprintf(info, "incr");
          }
        }

        if (mpi_rank == 0 && print_level > 0){
                fprintf(outfp, "%12s %2d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9s\n",
                        " ", i, ParOptRealPart(con_infeas[i]),
                        ParOptRealPart(model_con_infeas[i]),
                        ParOptRealPart(best_con_infeas[i]),
                        infeas_reduction, best_reduction, penalty_gamma[i], info);
        }
      }

      if (mpi_rank == 0 && print_level > 0){
        fprintf(outfp, "\n");
        fflush(outfp);
      }
    }
  }

  // Free the allocated data
  if (adaptive_gamma_update){
    delete [] con_infeas;
    delete [] model_con_infeas;
    delete [] best_con_infeas;
  }
}

/**
  Compute the KKT error based on the current values of the multipliers
  set in ParOptMMA. If you do not update the multipliers, you will not
  get the correct KKT error.
*/
void ParOptTrustRegion::computeKKTError( const ParOptScalar *z,
                                         ParOptVec *zw,
                                         double *l1,
                                         double *linfty ){
  // Extract the point, objective/constraint gradients, and
  // lower and upper bounds
  ParOptVec *xk, *gk, **Ak, *lb, *ub;
  subproblem->getLinearModel(&xk, &gk, &Ak, &lb, &ub);

  // Get the lower/upper bounds for the variables
  ParOptScalar *l, *u;
  lb->getArray(&l);
  ub->getArray(&u);

  // Get the current values of the design variables
  ParOptScalar *x;
  xk->getArray(&x);

  // Compute the KKT residual r = g - A^{T}*z
  t->copyValues(gk);
  for ( int i = 0; i < m; i++ ){
    t->axpy(-z[i], Ak[i]);
  }

  // If zw exists, compute r = r - Aw^{T}*zw
  if (nwcon > 0){
    subproblem->addSparseJacobianTranspose(-1.0, xk, zw, t);
  }

  // Set the infinity norms
  double l1_norm = 0.0;
  double infty_norm = 0.0;

  // Get the vector of values
  ParOptScalar *r;
  t->getArray(&r);

  for ( int j = 0; j < n; j++ ){
    double w = ParOptRealPart(r[j]);

    // Check if we're on the lower bound
    if ((ParOptRealPart(x[j]) <= ParOptRealPart(l[j]) + bound_relax) && w > 0.0){
      w = 0.0;
    }

    // Check if we're on the upper bound
    if ((ParOptRealPart(x[j]) >= ParOptRealPart(u[j]) - bound_relax) && w < 0.0){
      w = 0.0;
    }

    // Add the contribution to the l1/infinity norms
    double tw = fabs(w);
    l1_norm += tw;
    if (tw >= infty_norm){
      infty_norm = tw;
    }
  }

  // All-reduce the norms across all processors
  MPI_Allreduce(&l1_norm, l1, 1, MPI_DOUBLE,
                MPI_SUM, subproblem->getMPIComm());
  MPI_Allreduce(&infty_norm, linfty, 1, MPI_DOUBLE,
                MPI_MAX, subproblem->getMPIComm());
}
