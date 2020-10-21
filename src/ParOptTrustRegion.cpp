#include <string.h>
#include "ParOptTrustRegion.h"
#include "ParOptCompactEigenvalueApprox.h"
#include "ParOptComplexStep.h"

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
  ParOptTrustRegionSubproblem(_prob->getMPIComm()){

  // Paropt problem instance
  prob = _prob;
  prob->incref();

  // Get the problem sizes
  prob->getProblemSizes(&n, &m, &ninequality, &nwcon, &nwblock);
  setProblemSizes(n, m, ninequality, nwcon, nwblock);

  // Set the quasi-Newton method
  if (_qn){
    qn = _qn;
    qn->incref();
  }
  else {
    qn = NULL;
  }
  qn_update_type = 0;

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
  t = prob->createDesignVec();
  t->incref();
  xtemp = prob->createDesignVec();
  xtemp->incref();

  // Initialization for second order correction
  is_soc_step = 0;
  c_soc = new ParOptScalar[ m ];
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

  delete [] c_soc;

  t->decref();
  xtemp->decref();
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
    ltrvals[i] = max2(-tr_size, lvals[i] - xvals[i]);
    utrvals[i] = min2( tr_size, uvals[i] - xvals[i]);
  }
}

int ParOptQuadraticSubproblem::evalTrialStepAndUpdate( ParOptVec *step,
                                                       const ParOptScalar *z,
                                                       ParOptVec *zw,
                                                       ParOptScalar *fobj,
                                                       ParOptScalar *cons ){
  xtemp->copyValues(xk);
  xtemp->axpy(1.0, step);
  int fail = prob->evalObjCon(xtemp, &ft, ct);
  fail = fail || prob->evalObjConGradient(xtemp, gt, At);

  // Copy the values of the objective and constraints
  *fobj = ft;
  for ( int i = 0; i < m; i++ ){
    cons[i] = ct[i];
  }

  // If we're using a quasi-Newton Hessian approximation
  if (qn){
    // Compute the difference between the gradient of the
    // Lagrangian between the current point and the previous point
    t->copyValues(gt);
    for ( int i = 0; i < m; i++ ){
      t->axpy(-z[i], At[i]);
    }
    if (nwcon > 0){
      prob->addSparseJacobianTranspose(-1.0, xtemp, zw, t);
    }

    t->axpy(-1.0, gk);
    for ( int i = 0; i < m; i++ ){
      t->axpy(z[i], Ak[i]);
    }
    if (nwcon > 0){
      prob->addSparseJacobianTranspose(1.0, xk, zw, t);
    }

    // Perform an update of the quasi-Newton approximation
    prob->computeQuasiNewtonUpdateCorrection(step, t);
    qn_update_type = qn->update(xk, z, zw, step, t);
  }

  return fail;
}

int ParOptQuadraticSubproblem::acceptTrialStep( ParOptVec *step,
                                                const ParOptScalar *z,
                                                ParOptVec *zw ){
  int fail = 0;

  fk = ft;
  xk->axpy(1.0, step);
  gk->copyValues(gt);
  for ( int i = 0; i < m; i++ ){
    ck[i] = ct[i];
    Ak[i]->copyValues(At[i]);
  }

  return fail;
}

void ParOptQuadraticSubproblem::rejectTrialStep(){
  ft = 0.0;
  for ( int i = 0; i < m; i++ ){
    ct[i] = 0.0;
  }
}

int ParOptQuadraticSubproblem::getQuasiNewtonUpdateType(){
  return qn_update_type;
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
void ParOptQuadraticSubproblem::getVarsAndBounds( ParOptVec *step,
                                                  ParOptVec *l,
                                                  ParOptVec *u ){
  step->zeroEntries();
  step->axpy(0.5, lk);
  step->axpy(0.5, uk);
  l->copyValues(lk);
  u->copyValues(uk);
}

/*
  Evaluate the objective and constraint functions
*/
int ParOptQuadraticSubproblem::evalObjCon( ParOptVec *step,
                                           ParOptScalar *fobj,
                                           ParOptScalar *cons ){
  if (step){
    // Compute the objective function
    *fobj = fk + gk->dot(step);
    if (qn){
      qn->mult(step, t);
      *fobj += 0.5*step->dot(t);
    }

    // Compute the constraint functions
    if (is_soc_step){
      for ( int i = 0; i < m; i++ ){
        cons[i] = c_soc[i] + Ak[i]->dot(step);
      }
    }
    else{
      for ( int i = 0; i < m; i++ ){
        cons[i] = ck[i] + Ak[i]->dot(step);
      }
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
int ParOptQuadraticSubproblem::evalObjConGradient( ParOptVec *step,
                                                   ParOptVec *g,
                                                   ParOptVec **Ac ){
  // Copy the values of constraint gradient
  for ( int i = 0; i < m; i++ ){
    Ac[i]->copyValues(Ak[i]);
  }

  // Evaluate the gradient of the quadratic objective
  if (qn){
    qn->mult(step, g);
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
void ParOptQuadraticSubproblem::evalSparseCon( ParOptVec *step,
                                               ParOptVec *out ){
  prob->evalSparseCon(xk, out);
  prob->addSparseJacobian(1.0, xk, step, out);
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
                                               ParOptScalar *_fk,
                                               ParOptVec **_gk,
                                               const ParOptScalar **_ck,
                                               ParOptVec ***_Ak,
                                               ParOptVec **_lb,
                                               ParOptVec **_ub ){
  if (_xk){ *_xk = xk; }
  if (_fk){ *_fk = fk; }
  if (_gk){ *_gk = gk; }
  if (_ck){ *_ck = ck; }
  if (_Ak){ *_Ak = Ak; }
  if (_lb){ *_lb = lb; }
  if (_ub){ *_ub = ub; }

  return m;
}
/**
  Evaluate SOC trial point and get data pair (func val, constr val)
*/
int ParOptQuadraticSubproblem::evalSocTrialPoint( ParOptVec *xt,
                                                  int soc_use_quad_model,
                                                  ParOptScalar *f,
                                                  ParOptScalar *h ){
  // Get nineq and m
  int nineq = this->ninequality;
  int m = this->m;
  // Compute f
  xtemp->copyValues(xk);
  xtemp->axpy(1.0, xt);
  int fail = 0;
  if (soc_use_quad_model){
    // Evaluate the function and constraint values
    // of the quadratic model
    fail = this->evalObjCon(xtemp, f, ct);
  }
  else{
    // Evaluate the function and constraint values
    // of the original problem
    fail = prob->evalObjCon(xtemp, f, ct);
  }
  // Store f in this->ft
  this->ft = *f;
  // Compute h
  *h = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      *h += max2(0.0, -ct[i]);
    }
    else {
      *h += fabs(ct[i]);
    }
  }
  return fail;
}

/**
  Evaluate gradient for SOC trial point
*/
int ParOptQuadraticSubproblem::evalSocTrialGrad( ParOptVec *xt,
                                                 int soc_use_quad_model ){
  xtemp->copyValues(xk);
  xtemp->axpy(1.0, xt);
  int fail = 0;
  if (soc_use_quad_model){
    fail = this->evalObjConGradient(xtemp, gt, At);
  }
  else{
    fail = prob->evalObjConGradient(xtemp, gt, At);
  }
  return fail;
}


ParOptInfeasSubproblem::ParOptInfeasSubproblem( ParOptTrustRegionSubproblem *_prob,
                                                int _subproblem_objective,
                                                int _subproblem_constraint ):
  ParOptProblem(_prob->getMPIComm()){

  // Paropt problem instance
  prob = _prob;
  prob->incref();

  // Set the default scaling factor to unity
  obj_scale = 1.0;

  // Set the objective and constraint types
  subproblem_objective = _subproblem_objective;
  subproblem_constraint = _subproblem_constraint;

  // Get the problem sizes
  prob->getProblemSizes(&n, &m, &ninequality, &nwcon, &nwblock);
  setProblemSizes(n, m, ninequality, nwcon, nwblock);
}

ParOptInfeasSubproblem::~ParOptInfeasSubproblem(){
  prob->decref();
}

/*
  Create a design vector
*/
ParOptVec *ParOptInfeasSubproblem::createDesignVec(){
  return prob->createDesignVec();
}

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptInfeasSubproblem::createConstraintVec(){
  return prob->createConstraintVec();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptInfeasSubproblem::getMPIComm(){
  return prob->getMPIComm();
}

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptInfeasSubproblem::isSparseInequality(){
  return prob->isSparseInequality();
}

int ParOptInfeasSubproblem::useLowerBounds(){
  return 1;
}

int ParOptInfeasSubproblem::useUpperBounds(){
  return 1;
}

// Get the variables and bounds from the problem
void ParOptInfeasSubproblem::getVarsAndBounds( ParOptVec *step,
                                               ParOptVec *l,
                                               ParOptVec *u ){
  prob->getVarsAndBounds(step, l, u);
}

/*
  Evaluate the objective and constraint functions
*/
int ParOptInfeasSubproblem::evalObjCon( ParOptVec *step,
                                        ParOptScalar *fobj,
                                        ParOptScalar *cons ){
  // Get the components of the linearization
  ParOptScalar fk;
  const ParOptScalar *ck;
  ParOptVec *gk, **Ak;
  prob->getLinearModel(NULL, &fk, &gk, &ck, &Ak);

  if (step){
    if (subproblem_objective == PAROPT_SUBPROBLEM_OBJECTIVE ||
        subproblem_constraint == PAROPT_SUBPROBLEM_CONSTRAINT){
      prob->evalObjCon(step, fobj, cons);
    }

    if (subproblem_objective == PAROPT_LINEAR_OBJECTIVE){
      *fobj = fk + gk->dot(step);
    }
    else if (subproblem_objective == PAROPT_CONSTANT_OBJECTIVE){
      *fobj = fk;
    }

    if (subproblem_constraint == PAROPT_LINEAR_CONSTRAINT){
      for ( int i = 0; i < m; i++ ){
        cons[i] = ck[i] + Ak[i]->dot(step);
      }
    }
  }
  else {
    // If x is NULL, assume x = xk
    *fobj = fk;

    for ( int i = 0; i < m; i++ ){
      cons[i] = ck[i];
    }
  }

  // Apply the objective scaling
  *fobj *= obj_scale;

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptInfeasSubproblem::evalObjConGradient( ParOptVec *step,
                                                ParOptVec *g,
                                                ParOptVec **Ac ){
  // Get the components needed in the linearization;
  ParOptVec *gk, **Ak;
  prob->getLinearModel(NULL, NULL, &gk, NULL, &Ak);

  if (subproblem_objective == PAROPT_SUBPROBLEM_OBJECTIVE ||
      subproblem_constraint == PAROPT_SUBPROBLEM_CONSTRAINT){
    prob->evalObjConGradient(step, g, Ac);
  }

  if (subproblem_objective == PAROPT_LINEAR_OBJECTIVE){
    g->copyValues(gk);
  }
  else if (subproblem_objective == PAROPT_CONSTANT_OBJECTIVE){
    g->zeroEntries();
  }

  if (subproblem_constraint == PAROPT_LINEAR_CONSTRAINT){
    for ( int i = 0; i < m; i++ ){
      Ac[i]->copyValues(Ak[i]);
    }
  }

  g->scale(obj_scale);

  return 0;
}

/*
  Evaluate the constraints
*/
void ParOptInfeasSubproblem::evalSparseCon( ParOptVec *step,
                                            ParOptVec *out ){
  prob->evalSparseCon(step, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptInfeasSubproblem::addSparseJacobian( ParOptScalar alpha,
                                                ParOptVec *x,
                                                ParOptVec *px,
                                                ParOptVec *out ){
  prob->addSparseJacobian(alpha, x, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptInfeasSubproblem::addSparseJacobianTranspose( ParOptScalar alpha,
                                                         ParOptVec *x,
                                                         ParOptVec *pzw,
                                                         ParOptVec *out ){
  prob->addSparseJacobianTranspose(alpha, x, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptInfeasSubproblem::addSparseInnerProduct( ParOptScalar alpha,
                                                    ParOptVec *x,
                                                    ParOptVec *cvec,
                                                    ParOptScalar *A ){
  prob->addSparseInnerProduct(alpha, x, cvec, A);
}

/**
  A parallel optimizer with trust region globalization

  The following code implements a trust region method. The algorithm
  uses an l1 penalty function for the constraints with an l-infinity (box)
  constraint for the trust region. The trust region sub-problems are
  solved at each step by an instance of the ParOptInteriorPoint optimizer.

  @param _subproblem the ParOptTrustRegionSubproblem class
  @param _options The trust region options
*/
ParOptTrustRegion::ParOptTrustRegion( ParOptTrustRegionSubproblem *_subproblem,
                                      ParOptOptions *_options ){
  // Create the sub-problem instance
  subproblem = _subproblem;
  subproblem->incref();

  if (_options){
    options = _options;
  }
  else {
    options = new ParOptOptions(subproblem->getMPIComm());
    addDefaultOptions(options);
  }
  options->incref();

  // Get the subproblem sizes
  subproblem->getProblemSizes(&n, &m, &nineq, &nwcon, &nwblock);

  // Set the penalty parameters
  const double gamma = options->getFloatOption("penalty_gamma");
  penalty_gamma = new double[ m ];
  for ( int i = 0; i < m; i++ ){
    penalty_gamma[i] = gamma;
  }

  // Set the trust region radius
  tr_size = options->getFloatOption("tr_init_size");

  // Set the iteration count to zero
  iter_count = 0;

  // Set the number of parameters
  subproblem_iters = 0;
  adaptive_subproblem_iters = 0;

  // Create a temporary vector
  t = subproblem->createDesignVec();
  t->incref();

  // Allocate memory and initialize filters
  filter = new std::vector<std::pair<ParOptScalar, ParOptScalar> >;
  filter_size = 0;

  // If second order correction is used, create another temporary vector
  int tr_use_soc = options->getBoolOption("tr_use_soc");
  if (tr_use_soc){
    best_step = subproblem->createDesignVec();
    best_step->incref();
  }

  // By default, set the file pointer to stdout. If a filename is specified,
  // set the new filename.
  outfp = stdout;

  const char *filename = options->getStringOption("tr_output_file");
  if (filename){
    setOutputFile(filename);
  }
}

/**
  Delete the trust region object
*/
ParOptTrustRegion::~ParOptTrustRegion(){
  int tr_use_soc = options->getBoolOption("tr_use_soc");
  subproblem->decref();
  options->decref();

  filter->clear();
  delete filter;

  delete [] penalty_gamma;
  t->decref();

  if (tr_use_soc){
    best_step->decref();
  }

  // Close the file when we quit
  if (outfp && outfp != stdout){
    fclose(outfp);
  }
}

void ParOptTrustRegion::addDefaultOptions( ParOptOptions *options ){
  options->addStringOption("tr_output_file", "paropt.tr",
    "Trust region output file");

  options->addIntOption("output_level", 0, 0, 1000000,
    "Output level indicating how verbose the output should be");

  // Float options
  options->addFloatOption("tr_init_size", 0.1, 0.0, 1e20,
    "The initial trust region radius");

  options->addFloatOption("tr_min_size", 1e-3, 0.0, 1e20,
    "The minimum trust region radius");

  options->addFloatOption("tr_max_size", 1.0, 0.0, 1e20,
    "The maximum trust region radius");

  options->addFloatOption("tr_eta", 0.25, 0.0, 1.0,
    "Trust region trial step acceptance ratio");

  options->addFloatOption("tr_bound_relax", 1e-4, 0.0, 1e20,
    "Upper and lower bound relaxing parameter");

  options->addIntOption("tr_write_output_frequency", 10, 0, 1000000,
    "Write output frequency");

  options->addFloatOption("function_precision", 1e-10, 0.0, 1.0,
    "The absolute precision of the function and constraints");

  options->addFloatOption("design_precision", 1e-14, 0.0, 1.0,
    "The absolute precision of the design variables");

  options->addBoolOption("tr_adaptive_gamma_update", 1,
    "Adaptive penalty parameter update");

  const char *accept_step_options[2] = {"penalty_method", "filter_method"};
  options->addEnumOption("tr_accept_step_strategy",
    "penalty_method", 2, accept_step_options,
    "Which strategy to use to decide if a trial point can be accepted or not");

  options->addBoolOption("tr_use_soc", 0,
    "Use second order correction when trial step is rejeccted");

  options->addBoolOption("tr_soc_use_quad_model", 0,
    "In second order correction phase, use quadratic model to compute function values");

  options->addIntOption("tr_max_soc_iterations", 20, 0, 1000000,
    "Maximum number of trust region iterations");

  options->addIntOption("tr_max_iterations", 200, 0, 1000000,
    "Maximum number of trust region iterations");

  options->addFloatOption("tr_l1_tol", 1e-6, 0.0, 1e20,
    "l1 tolerance for the optimality tolerance");

  options->addFloatOption("tr_linfty_tol", 1e-6, 0.0, 1e20,
    "l-infinity tolerance for the optimality tolerance");

  options->addFloatOption("tr_infeas_tol", 1e-5, 0.0, 1e20,
    "Infeasibility tolerance ");

  options->addFloatOption("tr_penalty_gamma_max", 1e4, 0.0, 1e20,
    "Maximum value for the penalty parameter");

  options->addFloatOption("tr_penalty_gamma_min", 0.0, 0.0, 1e20,
    "Minimum value for the penalty parameter");

  const char *obj_options[3] = {"constant_objective",
                                "linear_objective",
                                "subproblem_objective"};
  options->addEnumOption("tr_adaptive_objective",
    "linear_objective", 3, obj_options,
    "The type of objective to use for the adaptive penalty subproblem");

  const char *con_options[2] = {"linear_constraint",
                                "subproblem_constraint"};
  options->addEnumOption("tr_adaptive_constraint",
    "linear_constraint", 2, con_options,
    "The type of constraint to use for the adaptive penalty subproblem");

  const char *barrier_options[5] = {"monotone",
                                    "mehrotra",
                                    "mehrotra_predictor_corrector",
                                    "complementarity_fraction",
                                    "default"};
  options->addEnumOption("tr_steering_barrier_strategy",
    "mehrotra_predictor_corrector", 5, barrier_options,
    "The barrier update strategy to use for the steering method subproblem");

  const char *start_options[4] = {"least_squares_multipliers",
                                  "affine_step",
                                  "no_start_strategy",
                                  "default"};
  options->addEnumOption("tr_steering_starting_point_strategy",
    "affine_step", 4, start_options,
    "The barrier update strategy to use for the steering method subproblem");
}

ParOptOptions* ParOptTrustRegion::getOptions(){
  return options;
}

/**
  Set the output file (only on the root proc)

  @param filename the output file name
*/
void ParOptTrustRegion::setOutputFile( const char *filename ){
  if (outfp && outfp != stdout){
    fclose(outfp);
  }
  outfp = NULL;

  int rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &rank);

  if (filename && rank == 0){
    outfp = fopen(filename, "w");
  }
}

/**
  Get the optimized point from the subproblem class

  @param x The values of the design variables at the optimized point
*/
void ParOptTrustRegion::getOptimizedPoint( ParOptVec **_x ){
  if (_x && subproblem){
    subproblem->getLinearModel(_x);
  }
}

/**
  Add pair (f, h) to filter set, and then remove dominated pairs

  @param f [in] Candidate function value
  @param h [in] Candidate constraint violation
*/
void ParOptTrustRegion::addToFilter( const ParOptScalar f,
                                     const ParOptScalar h ){
  const double tr_infeas_tol = options->getFloatOption("tr_infeas_tol");
  // Add current pair (f, h) to filter set
  filter->push_back(std::pair<ParOptScalar, ParOptScalar>
    (f, max2(h, (ParOptScalar)tr_infeas_tol)));
  filter_size += 1;
  // Delete dominated pairs
  for ( std::vector<std::pair<ParOptScalar, ParOptScalar> >::iterator i
        = filter->begin(); i != filter->end(); i++ ){
    if ( ParOptRealPart(i->first) > ParOptRealPart(f) &&
         ParOptRealPart(i->second) > ParOptRealPart(h) ){
      filter->erase(i);
      filter_size -= 1;
    }
  }
  return;
}

/**
  Use the filter to check if current design point can be accepted or
  rejected. If the current design is accepted, then add is to filter set.

  @param f [in] Candidate function value
  @param h [in] Candidate constraint violation
*/
int ParOptTrustRegion::isAcceptedByFilter( const ParOptScalar f,
                                           const ParOptScalar h ){
  const double tr_infeas_tol = options->getFloatOption("tr_infeas_tol");
  // Check if candidate pair (f, h) is dominated or not
  for ( std::vector<std::pair<ParOptScalar, ParOptScalar> >::iterator i
        = filter->begin(); i != filter->end(); i++ ){
    if ( ParOptRealPart(f) > ParOptRealPart(i->first) &&
         ParOptRealPart(h) > ParOptRealPart(i->second) ){
      // (f, h) is dominated, rejected
      return 0;
    }
  }
  // (f, h) is not dominated, add the pair to filter set, and delete
  // new dominated points
  this->addToFilter(f, h);
  return 1;
}

/**
  Write the parameters to the output file

  @param fp an open file handle
*/
void ParOptTrustRegion::printOptionSummary( FILE *fp ){
  const int output_level = options->getIntOption("output_level");
  int rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &rank);
  if (fp && rank == 0){
    fprintf(fp, "ParOptTrustRegion Parameter Summary:\n");
    options->printSummary(fp, output_level);
  }
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
  Initialize the problem
*/
void ParOptTrustRegion::initialize(){
  tr_size = options->getFloatOption("tr_init_size");
  subproblem->initModelAndBounds(tr_size);

  // Set the iteration count to zero
  iter_count = 0;

  int mpi_rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &mpi_rank);
  if (mpi_rank == 0){
    if (outfp){
      printOptionSummary(outfp);
    }
  }
}

/**
  Update the subproblem using SL1QP method
*/
void ParOptTrustRegion::sl1qp_update( ParOptVec *step,
                                      const ParOptScalar *z,
                                      ParOptVec *zw,
                                      double *infeas,
                                      double *l1,
                                      double *linfty ){
  // Extract options from the options object
  const double tr_eta        = options->getFloatOption("tr_eta");
  const double tr_min_size   = options->getFloatOption("tr_min_size");
  const double tr_max_size   = options->getFloatOption("tr_max_size");
  const double tr_infeas_tol = options->getFloatOption("tr_infeas_tol");
  const int output_level     = options->getIntOption("output_level");
  const int tr_adaptive_gamma_update =
    options->getBoolOption("tr_adaptive_gamma_update");
  const double function_precision    =
    options->getFloatOption("function_precision");

  // Get the mpi rank for printing
  int mpi_rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &mpi_rank);

  // Compute the value of the objective model and model
  // constraints at the current iterate
  ParOptScalar fk;
  ParOptScalar *ck = new ParOptScalar[ m ];
  subproblem->evalObjCon(NULL, &fk, ck);

  // Compute the model infeasibility at x = xk
  ParOptScalar infeas_k = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_k += penalty_gamma[i]*max2(0.0, -ck[i]);
    }
    else {
      infeas_k += penalty_gamma[i]*fabs(ck[i]);
    }
  }

  // Compute the value of the objective model and model
  // constraints at the trial step location
  ParOptScalar ft;
  ParOptScalar *ct = new ParOptScalar[ m ];
  subproblem->evalObjCon(step, &ft, ct);

  // Compute the reduction in the objective value
  ParOptScalar obj_reduc = fk - ft;

  // Compute the model infeasibility at the new point
  ParOptScalar infeas_model = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_model += penalty_gamma[i]*max2(0.0, -ct[i]);
    }
    else {
      infeas_model += penalty_gamma[i]*fabs(ct[i]);
    }
  }

  // Evaluate the model at the trial point and update the trust region model
  // Hessian and bounds. Note that here, we're re-using the ft/ct memory.
  subproblem->evalTrialStepAndUpdate(step, z, zw, &ft, ct);

  // Compute the infeasibilities of the last two iterations
  ParOptScalar infeas_t = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_t += penalty_gamma[i]*max2(0.0, -ct[i]);
    }
    else {
      infeas_t += penalty_gamma[i]*fabs(ct[i]);
    }
  }

  // Compute the actual reduction and the predicted reduction
  ParOptScalar actual_reduc =
    (fk - ft + (infeas_k - infeas_t));
  ParOptScalar model_reduc =
    obj_reduc + (infeas_k - infeas_model);

  if (mpi_rank == 0 && output_level > 0){
    FILE *fp = stdout;
    if (outfp){
      fp = outfp;
    }
    fprintf(fp, "%-12s %2s %12s %12s %12s\n",
            "Constraints", "i", "c(x)", "c(x+p)", "gamma");
    for ( int i = 0; i < m; i++ ){
      fprintf(fp, "%12s %2d %12.5e %12.5e %12.5e\n",
              " ", i, ParOptRealPart(ck[i]), ParOptRealPart(ct[i]),
              penalty_gamma[i]);
    }
    fprintf(fp, "\n%-15s %12s %12s %12s %12s\n",
            "Model", "ared(f)", "pred(f)", "ared(c)", "pred(c)");
    fprintf(fp, "%15s %12.5e %12.5e %12.5e %12.5e\n",
            " ", ParOptRealPart(fk - ft), ParOptRealPart(obj_reduc),
            ParOptRealPart(infeas_k - infeas_t),
            ParOptRealPart(infeas_k - infeas_model));
  }

  // Compute the ratio of the actual reduction
  ParOptScalar rho = 1.0;
  if (fabs(ParOptRealPart(model_reduc)) <= function_precision &&
      fabs(ParOptRealPart(actual_reduc)) <= function_precision){
    rho = 1.0;
  }
  else {
    rho = actual_reduc/model_reduc;
  }

  // Compute the infeasibility
  ParOptScalar infeas_new = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_new += max2(0.0, -ct[i]);
    }
    else {
      infeas_new += fabs(ct[i]);
    }
  }
  *infeas = ParOptRealPart(infeas_new);

  delete [] ck;
  delete [] ct;

  // Compute the max absolute value
  double smax = 0.0;

  // Check whether to accept the new point or not. If the trust region
  // radius size is at the lower bound, the step is always accepted
  int step_is_accepted = 0;
  if (ParOptRealPart(rho) >= tr_eta || tr_size <= tr_min_size){
    // Compute the length of the step for log entry purposes
    smax = ParOptRealPart(step->maxabs());
    subproblem->acceptTrialStep(step, z, zw);
    step_is_accepted = 1;
  }
  else {
    // Set the step size to zero (rejected step)
    subproblem->rejectTrialStep();
    smax = 0.0;
  }

  // Set the new trust region radius
  if (ParOptRealPart(rho) < 0.25){
    tr_size = ParOptRealPart(max2(0.25*tr_size, tr_min_size));
  }
  else if (ParOptRealPart(rho) > 0.75){
    tr_size = ParOptRealPart(min2(1.5*tr_size, tr_max_size));
  }

  // Reset the trust region radius bounds
  subproblem->setTrustRegionBounds(tr_size);

  // Compute the KKT error at the current point
  computeKKTError(z, zw, l1, linfty);

  // Compute the max z/average z and max gamma/average gamma
  double zmax = 0.0, zav = 0.0, gmax = 0.0, gav = 0.0;
  for ( int i = 0; i < m; i++ ){
    zav += ParOptRealPart(fabs(z[i]));
    gav += penalty_gamma[i];
    if (ParOptRealPart(fabs(z[i])) > zmax){
      zmax = ParOptRealPart(fabs(z[i]));
    }
    if (penalty_gamma[i] > gmax){
      gmax = penalty_gamma[i];
    }
  }
  zav = zav/m;
  gav = gav/m;

  // Create an info string for the update type
  int update_type = subproblem->getQuasiNewtonUpdateType();
  char info[64];
  info[0] = '\0';
  if (update_type == 1){
    // Damped BFGS update
    sprintf(&info[strlen(info)], "%s ", "dampH");
  }
  else if (update_type == 2){
    // Skipped update
    sprintf(&info[strlen(info)], "%s ", "skipH");
  }
  // Write out the number of subproblem iterations
  if (tr_adaptive_gamma_update){
    sprintf(&info[strlen(info)], "%d/%d ", subproblem_iters,
            adaptive_subproblem_iters);
  }
  else {
    sprintf(&info[strlen(info)], "%d ", subproblem_iters);
  }

  // Write information about whether the step is accepted or rejected
  if (!step_is_accepted){
    sprintf(&info[strlen(info)], "%s ", "rej");
  }

  if (mpi_rank == 0){
    FILE *fp = stdout;
    if (outfp){
      fp = outfp;
    }
    if (iter_count % 10 == 0 || output_level > 0){
      fprintf(fp,
              "\n%5s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %-12s\n",
              "iter", "fobj", "infeas", "l1", "linfty", "|x - xk|", "tr",
              "rho", "mod red.", "avg z", "max z", "avg pen.", "max pen.", "info");
      fflush(fp);
    }
    fprintf(fp,
            "%5d %12.5e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e "
            "%9.2e %9.2e %9.2e %9.2e %-12s\n",
            iter_count, ParOptRealPart(fk), *infeas, *l1, *linfty, smax, tr_size,
            ParOptRealPart(rho), ParOptRealPart(model_reduc),
            zav, zmax, gav, gmax, info);
    fflush(fp);
  }

  // Update the iteration counter
  iter_count++;
}

/**
  Update the subproblem using filterSQP method
*/
void ParOptTrustRegion::filtersqp_update( ParOptInteriorPoint *optimizer,
                                ParOptVec *step,
                                const ParOptScalar *z,
                                ParOptVec *zw,
                                double *infeas,
                                double *l1,
                                double *linfty ){
  // Extract options from the options object
  const double tr_eta             = options->getFloatOption("tr_eta");
  const double tr_min_size        = options->getFloatOption("tr_min_size");
  const double tr_max_size        = options->getFloatOption("tr_max_size");
  const double tr_infeas_tol      = options->getFloatOption("tr_infeas_tol");
  const int output_level          = options->getIntOption("output_level");
  const double function_precision = options->getFloatOption("function_precision");
  const int tr_use_soc            = options->getBoolOption("tr_use_soc");
  const int tr_soc_use_quad_model = options->getBoolOption("tr_soc_use_quad_model");
  const int tr_max_soc_iterations = options->getIntOption("tr_max_soc_iterations");

  // Get the mpi rank for printing
  int mpi_rank;
  MPI_Comm_rank(subproblem->getMPIComm(), &mpi_rank);

  // Compute the value of the objective model and model
  // constraints at the current iterate
  ParOptScalar fk;
  ParOptScalar *ck = new ParOptScalar[ m ];
  subproblem->evalObjCon(NULL, &fk, ck);

  // Compute the model infeasibility at x = xk
  ParOptScalar infeas_k = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_k += max2(0.0, -ck[i]);
    }
    else {
      infeas_k += fabs(ck[i]);
    }
  }

  // Store the (f, h) for initial point to filter set
  if (iter_count == 0){
    addToFilter(fk, infeas_k);
    addToFilter(fk, max2(100, 1.25*infeas_k));
  }

  // Compute the value of the objective model and model
  // constraints at the trial step location
  ParOptScalar ft;
  ParOptScalar *ct = new ParOptScalar[ m ];
  subproblem->evalObjCon(step, &ft, ct);

  // Compute the reduction in the objective value
  ParOptScalar obj_reduc = fk - ft;

  // Compute the model infeasibility at the new point
  ParOptScalar infeas_model = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_model += max2(0.0, -ct[i]);
    }
    else {
      infeas_model += fabs(ct[i]);
    }
  }

  // Evaluate the model at the trial point and update the trust region model
  // Hessian and bounds. Note that here, we're re-using the ft/ct memory.
  subproblem->evalTrialStepAndUpdate(step, z, zw, &ft, ct);

  // Compute the infeasibilities of the last two iterations
  ParOptScalar infeas_t = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_t += max2(0.0, -ct[i]);
    }
    else {
      infeas_t += fabs(ct[i]);
    }
  }

  if (mpi_rank == 0 && output_level > 0){
    FILE *fp = stdout;
    if (outfp){
      fp = outfp;
    }
    fprintf(fp, "%-12s %2s %12s %12s %12s\n",
            "Constraints", "i", "c(x)", "c(x+p)", "gamma");
    for ( int i = 0; i < m; i++ ){
      fprintf(fp, "%12s %2d %12.5e %12.5e %12.5e\n",
              " ", i, ParOptRealPart(ck[i]), ParOptRealPart(ct[i]),
              penalty_gamma[i]);
    }
    fprintf(fp, "\n%-15s %12s %12s %12s %12s\n",
            "Model", "ared(f)", "pred(f)", "ared(c)", "pred(c)");
    fprintf(fp, "%15s %12.5e %12.5e %12.5e %12.5e\n",
            " ", ParOptRealPart(fk - ft), ParOptRealPart(obj_reduc),
            ParOptRealPart(infeas_k - infeas_t),
            ParOptRealPart(infeas_k - infeas_model));
  }

  // Compute the infeasibility
  ParOptScalar infeas_new = 0.0;
  for ( int i = 0; i < m; i++ ){
    if (i < nineq){
      infeas_new += max2(0.0, -ct[i]);
    }
    else {
      infeas_new += fabs(ct[i]);
    }
  }

  delete [] ck;
  delete [] ct;

  // Compute the max absolute value
  double smax = 0.0;

  // Check whether to accept the new point or not. This check can be
  // done using filter method or penalty method. If the trust region
  // radius size is at the lower bound, the step is always accepted
  int step_is_accepted                = 0;
  int enter_soc_phase                 = 0;
  int soc_is_accepted                 = 0;
  int filter_increase_tr_size         = 0;
  int filter_reduce_tr_size           = 0;
  double step_length_equals_tr_radius = 1e-10;

  // When filter method is used, then the step will be accepted if either:
  //   - trust region radius <= minimum trust region radius, or
  //   - the candidate pair (f, h) is accepted by filter
  step_is_accepted = (tr_size <= tr_min_size) || isAcceptedByFilter(ft, infeas_new);

  // If step is accepted, update design
  if (step_is_accepted){
    smax = ParOptRealPart(step->maxabs());
    subproblem->acceptTrialStep(step, z, zw);
    // If also the new step is at the trust region bound,
    // We increase the trust region radius for next iteration
    if (abs(smax - tr_size) < step_length_equals_tr_radius){
      filter_increase_tr_size = 1;
    }
  }

  // Otherwise, the step is not yet acceptable, but:
  // if second order correction is switched on, then enter SOC phase
  else if (tr_use_soc){
    enter_soc_phase = 1;
    ParOptScalar r = 0; // rate of convergence of the SOC steps
    for ( int i = 0; i < tr_max_soc_iterations; i++ ){
      // update quadratic model and optimize
      subproblem->updateSocCon(step, ct);
      optimizer->resetDesignAndBounds();

      // Call this right before optimization to switch to SOC problem
      subproblem->startSecondOrderCorrection();

      optimizer->optimize();

      // Call this right after optimization to switch back to original problem
      subproblem->endSecondOrderCorrection();

      // Get the new trial point
      optimizer->getOptimizedPoint(&step, NULL, NULL, NULL, NULL);

      // Compute (f, h) and SOC convergence rate at new trial point
      r = 1.0/infeas_new;
      subproblem->evalSocTrialPoint(step, tr_soc_use_quad_model, &ft, &infeas_new);
      r *= infeas_new;

      // If the new trial is acceptable, accept the step
      if ( isAcceptedByFilter(ft, infeas_new) ){
        soc_is_accepted = 1;
        break;
      }
      // Else if rate of SOC convergence step is slow, or
      // an almost feasible point is generated, or
      // hit the maximum number of iteration,
      // then SOC phase fails, reject the step and reduce trust region step
      else if ( ParOptRealPart(r) > 0.25 ||
                ParOptRealPart(infeas_new) < tr_infeas_tol ||
                i >= tr_max_soc_iterations - 1 ){
        soc_is_accepted = 0;
        filter_reduce_tr_size = 1;
        break;
      }
      // Otherwise, continue for next SOC iteration
      else{
        continue;
      }
    }
    // If second order step is accepted
    if (soc_is_accepted){
      smax = ParOptRealPart(step->maxabs());
      subproblem->evalSocTrialGrad(step, tr_soc_use_quad_model);
      subproblem->acceptTrialStep(step, z, zw);
      step_is_accepted = 1;
      // If the rate of convergence of the SOC step is sufficiently
      // good, we increase the trust region step
      if (r < 0.1){
        filter_increase_tr_size = 1;
      }
    }
    // Otherwise, reject the second order step
    else {
        smax = 0.0;
        subproblem->rejectTrialStep();
        step_is_accepted = 0;
        filter_reduce_tr_size = 1;
      }
    }

  // Otherwise, the step is directly rejected without considering SOC
  else {
    subproblem->rejectTrialStep();
    smax = 0.0;
    step_is_accepted = 0;
    filter_reduce_tr_size = 1;
  }

  // After figuring out the new optimization step,
  // we update the trust region radius. When filter method
  // is used, an alternative metric is needed other than
  // rho because we don't have penalty parameter now
  if (filter_increase_tr_size){
    // Increase trust region radius
    tr_size = ParOptRealPart(min2(1.5*tr_size, tr_max_size));
  }
  else if(filter_reduce_tr_size){
    // Reduce trust region radius
    tr_size = ParOptRealPart(max2(0.25*tr_size, tr_min_size));
  }

  // Update infeasibility for output because it may change by SOC
  *infeas = ParOptRealPart(infeas_new);

  // Reset the trust region radius bounds
  subproblem->setTrustRegionBounds(tr_size);

  // Compute the KKT error at the current point
  computeKKTError(z, zw, l1, linfty);

  // Compute the max z/average z and max gamma/average gamma
  double zmax = 0.0, zav = 0.0, gmax = 0.0, gav = 0.0;
  for ( int i = 0; i < m; i++ ){
    zav += ParOptRealPart(fabs(z[i]));
    gav += penalty_gamma[i];
    if (ParOptRealPart(fabs(z[i])) > zmax){
      zmax = ParOptRealPart(fabs(z[i]));
    }
    if (penalty_gamma[i] > gmax){
      gmax = penalty_gamma[i];
    }
  }
  zav = zav/m;
  gav = gav/m;

  // Create an info string for the update type
  int update_type = subproblem->getQuasiNewtonUpdateType();
  char info[64];
  info[0] = '\0';
  if (update_type == 1){
    // Damped BFGS update
    sprintf(&info[strlen(info)], "%s ", "dampH");
  }
  else if (update_type == 2){
    // Skipped update
    sprintf(&info[strlen(info)], "%s ", "skipH");
  }
  // Write out the number of subproblem iterations
  sprintf(&info[strlen(info)], "%d ", subproblem_iters);

  // Write out the size of filter set
  sprintf(&info[strlen(info)], "f%d ", filter_size);

  // Write information about whether the step is accepted or rejected
  if (!step_is_accepted){
    sprintf(&info[strlen(info)], "%s ", "rej");
  }

  // Write information about whether the SOC step is accepted or rejected
  if (enter_soc_phase){
    if (!soc_is_accepted){
      sprintf(&info[strlen(info)], "%s ", "socFail");
    }
    else{
      sprintf(&info[strlen(info)], "%s ", "socSucc");
    }
  }

  if (mpi_rank == 0){
    FILE *fp = stdout;
    if (outfp){
      fp = outfp;
    }
    if (iter_count % 10 == 0 || output_level > 0){
      fprintf(fp,
              "\n%5s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %-12s\n",
              "iter", "fobj", "infeas", "l1", "linfty", "|x - xk|", "tr",
              "rho", "mod red.", "avg z", "max z", "avg pen.", "max pen.", "info");
      fflush(fp);
    }
    fprintf(fp,
            "%5d %12.5e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e "
            "%9.2e %9.2e %9.2e %9.2e %-12s\n",
            iter_count, ParOptRealPart(fk), *infeas, *l1, *linfty, smax, tr_size,
            0.0, 0.0, zav, zmax, gav, gmax, info);
    fflush(fp);
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

  // Extract options
  const int tr_adaptive_gamma_update =
    options->getBoolOption("tr_adaptive_gamma_update");
  const int tr_max_iterations = options->getIntOption("tr_max_iterations");
  const double tr_penalty_gamma_max =
    options->getFloatOption("tr_penalty_gamma_max");
  const double tr_penalty_gamma_min =
    options->getFloatOption("tr_penalty_gamma_min");

  const double tr_infeas_tol = options->getFloatOption("tr_infeas_tol");
  const double tr_l1_tol = options->getFloatOption("tr_l1_tol");
  const double tr_linfty_tol = options->getFloatOption("tr_linfty_tol");
  const int output_level = options->getIntOption("output_level");
  const int tr_write_output_frequency =
    options->getIntOption("tr_write_output_frequency");

  const char *obj_problem = options->getEnumOption("tr_adaptive_objective");
  int adaptive_objective_flag = ParOptInfeasSubproblem::PAROPT_LINEAR_OBJECTIVE;
  if (strcmp(obj_problem, "constant_objective") == 0){
    adaptive_objective_flag = ParOptInfeasSubproblem::PAROPT_CONSTANT_OBJECTIVE;
  }
  else if (strcmp(obj_problem, "subproblem_objective") == 0){
    adaptive_objective_flag = ParOptInfeasSubproblem::PAROPT_SUBPROBLEM_OBJECTIVE;
  }

  const char *con_problem = options->getEnumOption("tr_adaptive_constraint");
  int adaptive_constraint_flag = ParOptInfeasSubproblem::PAROPT_LINEAR_CONSTRAINT;
  if (strcmp(con_problem, "subproblem_constraint") == 0){
    adaptive_constraint_flag = ParOptInfeasSubproblem::PAROPT_SUBPROBLEM_CONSTRAINT;
  }

  const char *tr_accept_step_strategy =
    options->getEnumOption("tr_accept_step_strategy");

  // Set the starting point strategy for the subproblem
  const char* tr_barrier_strategy =
    options->getEnumOption("tr_steering_barrier_strategy");
  const char* tr_starting_strategy =
    options->getEnumOption("tr_steering_starting_point_strategy");

  // Set up the optimizer so that it uses the quasi-Newton approximation
  ParOptCompactQuasiNewton *qn = subproblem->getQuasiNewton();
  optimizer->setQuasiNewton(qn);

  // Get the interior point options
  ParOptOptions *ip_options = optimizer->getOptions();

  // Do not update the Hessian within the interior-point method
  ip_options->setOption("use_quasi_newton_update", 0);

  // Set the output frequency for the subproblem iterations to zero so we
  // don't generate output files from subiterations.
  ip_options->setOption("write_output_frequency", 0);

  // Extract and store the barrier strategy and starting point strategy.
  // During the linear optimization subproblem, these will be reset to
  // more appropriate values. During the QP part of the optimization, these
  // will be set back to their original values.
  const char *barrier_strategy = ip_options->getEnumOption("barrier_strategy");
  char *barrier_option = new char[ strlen(barrier_strategy)+1 ];
  strcpy(barrier_option, barrier_strategy);

  const char *start_strategy = ip_options->getEnumOption("starting_point_strategy");
  char *start_option = new char[ strlen(start_strategy)+1 ];
  strcpy(start_option, start_strategy);

  // Set the initial values for the penalty parameter
  optimizer->setPenaltyGamma(penalty_gamma);

  // If needed, allocate a subproblem instance
  ParOptInfeasSubproblem *infeas_problem = NULL;
  if (tr_adaptive_gamma_update){
    infeas_problem = new ParOptInfeasSubproblem(subproblem,
                                                adaptive_objective_flag,
                                                adaptive_constraint_flag);
    infeas_problem->incref();
  }

  // Allocate arrays to store infeasibility information
  ParOptScalar *con_infeas = NULL;
  ParOptScalar *model_con_infeas = NULL;
  ParOptScalar *best_con_infeas = NULL;
  if (tr_adaptive_gamma_update){
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
  for ( int i = 0; i < tr_max_iterations; i++ ){
    if (tr_adaptive_gamma_update){
      // Reset the problem instance
      optimizer->resetProblemInstance(infeas_problem);

      // Set the starting point strategy
      if (strcmp(tr_barrier_strategy, "default") != 0){
        ip_options->setOption("barrier_strategy", tr_barrier_strategy);
      }
      if (strcmp(tr_starting_strategy, "default") != 0){
        ip_options->setOption("starting_point_strategy", tr_starting_strategy);
      }

      // Check if this is an compact representation using the eigenvalue Hessian
      ParOptEigenQuasiNewton *eig_qn = dynamic_cast<ParOptEigenQuasiNewton*>(qn);

      // Check what type of infeasible subproblem
      int is_seq = ip_options->getBoolOption("sequential_linear_method");
      if (adaptive_objective_flag == ParOptInfeasSubproblem::PAROPT_LINEAR_OBJECTIVE ||
          adaptive_objective_flag == ParOptInfeasSubproblem::PAROPT_CONSTANT_OBJECTIVE){
        if (eig_qn){
          eig_qn->setUseQuasiNewtonObjective(0);
        }

        // Linear (or constant) objective and linear constraints - sequential linear problem
        if (adaptive_constraint_flag == ParOptInfeasSubproblem::PAROPT_LINEAR_CONSTRAINT){
          ip_options->setOption("sequential_linear_method", 1);
        }
      }

      // Set the penalty parameter to a large value
      double gamma = 1e6;
      if (1e2*tr_penalty_gamma_max > gamma){
        gamma = 1e2*tr_penalty_gamma_max;
      }

      // Set the objective scaling to 1.0/gamma so that the contribution from
      // the objective is small
      infeas_problem->setObjectiveScaling(1.0/gamma);

      // Set the penalty parameters to zero
      optimizer->setPenaltyGamma(1.0);

      // Initialize the barrier parameter
      optimizer->resetDesignAndBounds();

      // Optimize the subproblem
      optimizer->optimize();

      // Get the design variables
      ParOptVec *step;
      optimizer->getOptimizedPoint(&step, NULL, NULL, NULL, NULL);

      // Get the number of subproblem iterations
      optimizer->getIterationCounters(&adaptive_subproblem_iters);

      // Evaluate the model at the best point to obtain the infeasibility
      ParOptScalar fbest;
      subproblem->evalObjCon(step, &fbest, best_con_infeas);

      // Compute the best-case infeasibility achieved by setting the
      // penalty parameters to a large value
      for ( int j = 0; j < m; j++ ){
        if (j < nineq){
          best_con_infeas[j] = max2(0.0, -best_con_infeas[j]);
        }
        else {
          best_con_infeas[j] = fabs(best_con_infeas[j]);
        }
      }

      // Set the penalty parameters
      optimizer->setPenaltyGamma(penalty_gamma);

      // Reset the problem instance and turn off the sequential linear method
      optimizer->resetProblemInstance(subproblem);

      // Set the quasi-Newton method so that it uses the objective again
      if (eig_qn){
        eig_qn->setUseQuasiNewtonObjective(1);
      }

      // Reset the options
      ip_options->setOption("starting_point_strategy", start_option);
      ip_options->setOption("barrier_strategy", barrier_option);
      ip_options->setOption("sequential_linear_method", is_seq);
    }

    // Print out the current solution progress using the
    // hook in the problem definition
    if (tr_write_output_frequency > 0 &&
        i % tr_write_output_frequency == 0){
      ParOptVec *xk;
      subproblem->getLinearModel(&xk);
      subproblem->writeOutput(i, xk);
    }

    // Initialize the barrier parameter
    optimizer->resetDesignAndBounds();

    // Optimize the subproblem
    optimizer->optimize();

    // Get the design variables
    ParOptVec *step, *zw;
    ParOptScalar *z;
    optimizer->getOptimizedPoint(&step, &z, &zw, NULL, NULL);

    // Get the number of subproblem iterations
    optimizer->getIterationCounters(&subproblem_iters);

    if (tr_adaptive_gamma_update){
      // Find the infeasibility at the origin x = xk
      ParOptScalar f0;
      subproblem->evalObjCon(NULL, &f0, con_infeas);

      // Find the actual infeasibility reduction
      ParOptScalar fmodel;
      subproblem->evalObjCon(step, &fmodel, model_con_infeas);
      for ( int j = 0; j < m; j++ ){
        if (j < nineq){
          con_infeas[j] = max2(0.0, -con_infeas[j]);
          model_con_infeas[j] = max2(0.0, -model_con_infeas[j]);
        }
        else {
          con_infeas[j] = fabs(con_infeas[j]);
          model_con_infeas[j] = fabs(model_con_infeas[j]);
        }
      }
    }

    // Update the trust region based on the performance at the new
    // point.
    double infeas, l1, linfty;
    // If filter method is specific, use filterSQP update
    if (strcmp(tr_accept_step_strategy, "filter_method") == 0){
      filtersqp_update(optimizer, step, z, zw, &infeas, &l1, &linfty);
    }
    // Otherwise, use sl1qp update
    else{
      sl1qp_update(step, z, zw, &infeas, &l1, &linfty);
    }

    // Check for convergence of the trust region problem
    if (infeas < tr_infeas_tol){
      if (l1 < tr_l1_tol ||
          linfty < tr_linfty_tol){
        // Success!
        break;
      }
    }

    // Adapat the penalty parameters
    if (tr_adaptive_gamma_update){
      FILE *fp = stdout;
      if (outfp){
        fp = outfp;
      }
      if (mpi_rank == 0 && output_level > 0){
        fprintf(fp, "%-12s %2s %12s %12s %12s %12s %12s %12s %9s\n",
                "Penalty", "i", "|c(x)|", "|c+Ap|", "min|c+Ap|",
                "pred", "min. pred", "gamma", "update");
      }

      for ( int i = 0; i < m; i++ ){
        // Compute the actual infeasibility reduction and the best
        // possible infeasibility reduction
        double infeas_reduction = ParOptRealPart(con_infeas[i] - model_con_infeas[i]);
        double best_reduction = ParOptRealPart(con_infeas[i] - best_con_infeas[i]);

        char info[8];
        if (output_level > 0){
          sprintf(info, "---");
        }

        // If the ratio of the predicted to actual improvement is good,
        // and the constraints are satisfied, decrease the penalty
        // parameter. Otherwise, if the best case infeasibility is
        // significantly better, increase the penalty parameter.
        if (ParOptRealPart(fabs(z[i])) > tr_infeas_tol &&
            ParOptRealPart(con_infeas[i]) < tr_infeas_tol &&
            penalty_gamma[i] >= 2.0*ParOptRealPart(z[i])){
          // Reduce gamma
          penalty_gamma[i] =
            ParOptRealPart(max2(0.5*(penalty_gamma[i] + fabs(z[i])),
                                tr_penalty_gamma_min));
          if (output_level > 0){
            sprintf(info, "decr");
          }
        }
        else if (ParOptRealPart(con_infeas[i]) > tr_infeas_tol &&
                 0.995*best_reduction > infeas_reduction){
          // Increase gamma
          penalty_gamma[i] = ParOptRealPart(min2(1.5*penalty_gamma[i],
                                                 tr_penalty_gamma_max));
          if (output_level > 0){
            sprintf(info, "incr");
          }
        }

        if (mpi_rank == 0 && output_level > 0){
          fprintf(fp, "%12s %2d %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %9s\n",
                  " ", i, ParOptRealPart(con_infeas[i]),
                  ParOptRealPart(model_con_infeas[i]),
                  ParOptRealPart(best_con_infeas[i]),
                  infeas_reduction, best_reduction, penalty_gamma[i], info);
        }
      }

      if (mpi_rank == 0 && output_level > 0){
        fprintf(fp, "\n");
        fflush(fp);
      }
    }
  }

  // Free the allocated data
  if (tr_adaptive_gamma_update){
    delete [] con_infeas;
    delete [] model_con_infeas;
    delete [] best_con_infeas;

    if (infeas_problem){
      infeas_problem->decref();
    }
  }

  delete [] barrier_option;
  delete [] start_option;
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
  const double tr_bound_relax = options->getFloatOption("tr_bound_relax");

  // Extract the point, objective/constraint gradients, and
  // lower and upper bounds
  ParOptVec *xk, *gk, **Ak, *lb, *ub;
  subproblem->getLinearModel(&xk, NULL, &gk, NULL, &Ak, &lb, &ub);

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
    if ((ParOptRealPart(x[j]) <= ParOptRealPart(l[j]) + tr_bound_relax) && w > 0.0){
      w = 0.0;
    }
    else if ((ParOptRealPart(x[j]) >= ParOptRealPart(u[j]) - tr_bound_relax) && w < 0.0){
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

  // Find the maximum absolute multiplier value
  ParOptScalar zmax = 0.0;
  if (nwcon > 0){
    zmax = zw->maxabs();
  }
  for ( int i = 0; i < m; i++ ){
    if (ParOptRealPart(fabs(z[i])) > ParOptRealPart(zmax)){
      zmax = fabs(z[i]);
    }
  }

  // Compute the stopping criterion:
  // *l1 = ||g - A^{T}*z - Aw^{T}*zw||_{1}/max(1, zmax, ||g||_{1})
  // *linfty = ||g - A^{T}*z - Aw^{T}*zw||_{infinity}/max(1, zmax, ||g||_{infinity})
  zmax = max2(1.0, zmax);
  *l1 = *l1/ParOptRealPart(max2(gk->l1norm(), zmax));
  *linfty = *linfty/ParOptRealPart(max2(gk->maxabs(), zmax));
}
