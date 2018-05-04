#include "ParOptTrustRegion.h"
#include "ComplexStep.h"

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

ParOptTrustRegion::ParOptTrustRegion( ParOptProblem *_prob,
                                      ParOptCompactQuasiNewton *_qn,
                                      double _tr_size,
                                      double _tr_min_size,
                                      double _tr_max_size,
                                      double _eta,
                                      double _penalty_value,
                                      double _bound_relax ):
ParOptProblem(_prob->getMPIComm()){
  // Paropt problem instance
  prob = _prob;
  prob->incref();

  // Get the problem sizes
  int _nwcon, _nwblock;
  prob->getProblemSizes(&n, &m, &_nwcon, &_nwblock);
  setProblemSizes(n, m, _nwcon, _nwblock);

  // Set the quasi-Newton method
  qn = _qn;
  qn->incref();

  // Set the solution parameters
  tr_size = _tr_size;
  tr_min_size = _tr_min_size;
  tr_max_size = _tr_max_size;
  eta = _eta;
  penalty_value = _penalty_value;
  bound_relax = _bound_relax;

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

/*
  Delete the trust region object
*/
ParOptTrustRegion::~ParOptTrustRegion(){
  prob->decref();
  qn->decref();

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
  Set the trust region bounds
*/
void ParOptTrustRegion::setTrustRegionBounds( double tr,
                                              ParOptVec *x,
                                              ParOptVec *l,
                                              ParOptVec *u,
                                              ParOptVec *ltr,
                                              ParOptVec *utr ){
  ParOptScalar *xvals;
  ParOptScalar *lvals, *uvals;
  ParOptScalar *ltrvals, *utrvals;

  int size = x->getArray(&xvals);
  l->getArray(&lvals);
  u->getArray(&uvals);
  ltr->getArray(&ltrvals);
  utr->getArray(&utrvals);

  for ( int i = 0; i < size; i++ ){
    ltrvals[i] = max2(xvals[i] - tr, lvals[i]);
    utrvals[i] = min2(xvals[i] + tr, uvals[i]);
  }
}

/*
  Initialize the problem
*/
void ParOptTrustRegion::initialize(){
  // Get the lower/upper bounds
  prob->getVarsAndBounds(xk, lb, ub);

  // Set the lower/upper bounds for the trust region
  setTrustRegionBounds(tr_size, xk, lb, ub, lk, uk);

  // Evaluate objective constraints and gradients
  prob->evalObjCon(xk, &fk, ck);
  prob->evalObjConGradient(xk, gk, Ak);
}

/*
  Update the trust region problem
*/
void ParOptTrustRegion::update( ParOptVec *xt,
                                const ParOptScalar *z,
                                ParOptVec *zw,
                                double *infeas,
                                double *l1,
                                double *linfty ){
  // Compute the step
  s->copyValues(xt);
  s->axpy(-1.0, xk);

  // Compute the decrease in the model objective function
  qn->mult(s, t);
  ParOptScalar obj_reduc = -(gk->dot(s) + 0.5*t->dot(s));

  // Compute the model infeasibility
  ParOptScalar infeas_model = 0.0;
  for ( int i = 0; i < m; i++ ){
    ParOptScalar cval = ck[i] + Ak[i]->dot(s);
    infeas_model += max2(0.0, -cval);
  }

  // Evaluate the objective and constraints and their gradients at
  // the new, optimized point
  prob->evalObjCon(xt, &ft, ct);
  prob->evalObjConGradient(xt, gt, At);

  // Compute the infeasibilities of the last two iterations
  ParOptScalar infeas_k = 0.0;
  ParOptScalar infeas_t = 0.0;
  for ( int i = 0; i < m; i++ ){
    infeas_k += max2(0.0, -ck[i]);
    infeas_t += max2(0.0, -ct[i]);
  }

  // Compute the actual reduction and the predicted reduction
  ParOptScalar actual_reduc =
    (fk - ft + penalty_value*(infeas_k - infeas_t));
  ParOptScalar model_reduc =
    obj_reduc + penalty_value*(infeas_k - infeas_model);
  
  // Compute the ratio
  ParOptScalar rho = actual_reduc/model_reduc;

  // Compute the difference between the gradient of the
  // Lagrangian between the current point and the previous point
  t->copyValues(gt);
  for ( int i = 0; i < m; i++ ){
    t->axpy(-z[i], At[i]);
  }
  prob->addSparseJacobianTranspose(-1.0, xt, zw, t);

  t->axpy(-1.0, gk);
  for ( int i = 0; i < m; i++ ){
    t->axpy(z[i], Ak[i]);
  }
  prob->addSparseJacobianTranspose(1.0, xk, zw, t);

  // Perform an update of the quasi-Newton approximation
  qn->update(s, t);

  // Compute the KKT error at the current point
  computeKKTError(xt, gt, At, z, zw, l1, linfty);
  *infeas = RealPart(infeas_t);

  // Check whether to accept the new point or not
  if (RealPart(rho) >= eta){
    fk = ft;
    xk->copyValues(xt);
    gk->copyValues(gt);
    for ( int i = 0; i < m; i++ ){
      ck[i] = ct[i];
      Ak[i]->copyValues(At[i]);
    }
  }

  // Set the new trust region radius
  if (RealPart(rho) < 0.25){
    tr_size = max2(0.25*tr_size, tr_min_size);
  }
  else if (RealPart(rho) > 0.75){
    tr_size = min2(2.0*tr_size, tr_max_size);
  }

  // Reset the trust region radius bounds
  setTrustRegionBounds(tr_size, xk, lb, ub, lk, uk);

  int mpi_rank;
  MPI_Comm_rank(prob->getMPIComm(), &mpi_rank);
  if (mpi_rank == 0){
    printf("%12s %9s %9s %9s %9s %9s %9s\n",
           "fobj", "infeas", "l1", "linfty", "tr", "rho", "mod red.");
    printf("%12.5e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
           fk, *infeas, *l1, *linfty, tr_size, rho, model_reduc);
  }
}

/*
  Compute the KKT error based on the current values of the multipliers
  set in ParOptMMA. If you do not update the multipliers, you will not
  get the correct KKT error.
*/
void ParOptTrustRegion::computeKKTError( ParOptVec *xv,
                                         ParOptVec *g,
                                         ParOptVec **A,
                                         const ParOptScalar *z,
                                         ParOptVec *zw,
                                         double *l1,
                                         double *linfty ){
  // Get the lower/upper bounds for the variables
  ParOptScalar *l, *u;
  lb->getArray(&l);
  ub->getArray(&u);

  // Get the current values of the design variables
  ParOptScalar *x;
  xv->getArray(&x);

  // Compute the KKT residual r = g - A^{T}*z
  t->copyValues(g);
  for ( int i = 0; i < m; i++ ){
    t->axpy(-z[i], A[i]);
  }

  // If zw exists, compute r = r - Aw^{T}*zw
  if (zw){
    prob->addSparseJacobianTranspose(-1.0, xv, zw, t);
  }

  // Set the infinity norms
  double l1_norm = 0.0;
  double infty_norm = 0.0;

  // Get the vector of values
  ParOptScalar *r;
  t->getArray(&r);

  for ( int j = 0; j < n; j++ ){
    double w = RealPart(r[j]);

    // Check if we're on the lower bound
    if ((x[j] <= l[j] + bound_relax) && w > 0.0){
      w = 0.0;
    }

    // Check if we're on the upper bound
    if ((x[j] >= u[j] - bound_relax) && w < 0.0){
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
                MPI_SUM, prob->getMPIComm());
  MPI_Allreduce(&infty_norm, linfty, 1, MPI_DOUBLE,
                MPI_MAX, prob->getMPIComm());
}

/*
  Create a design vector
*/
ParOptVec *ParOptTrustRegion::createDesignVec(){
  return prob->createDesignVec();
}

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptTrustRegion::createConstraintVec(){
  return prob->createConstraintVec();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptTrustRegion::getMPIComm(){
  return prob->getMPIComm();
}

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptTrustRegion::isDenseInequality(){
  return prob->isDenseInequality();
}

int ParOptTrustRegion::isSparseInequality(){
  return prob->isSparseInequality();
}

int ParOptTrustRegion::useLowerBounds(){
  return 1;
}

int ParOptTrustRegion::useUpperBounds(){
  return 1;
}

// Get the variables and bounds from the problem
void ParOptTrustRegion::getVarsAndBounds( ParOptVec *x, ParOptVec *l,
                                          ParOptVec *u ){
  x->copyValues(xk);
  l->copyValues(lk);
  u->copyValues(uk);
}

/*
  Evaluate the objective and constraint functions
*/
int ParOptTrustRegion::evalObjCon( ParOptVec *x, ParOptScalar *fobj, 
                                   ParOptScalar *cons ){
  s->copyValues(x);
  s->axpy(-1.0, xk);
  qn->mult(s, t);

  // Compute the objective function
  *fobj = fk + gk->dot(s) + 0.5*s->dot(t);

  // Compute the constraint functions
  for ( int i = 0; i < m; i++ ){
    cons[i] = ck[i] + Ak[i]->dot(s);
  }

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptTrustRegion::evalObjConGradient( ParOptVec *x, ParOptVec *g, 
                                           ParOptVec **Ac ){
  // Copy the values of constraint gradient
  for ( int i = 0; i < m; i++ ){
    Ac[i]->copyValues(Ak[i]);
  }

  s->copyValues(x);
  s->axpy(-1.0, xk);
  
  // Evaluate the gradient of the quadratic objective
  qn->mult(s, g);
  g->axpy(1.0, gk);

  return 0;
}

/*
  Evaluate the constraints
*/
void ParOptTrustRegion::evalSparseCon( ParOptVec *x, ParOptVec *out ){
  prob->evalSparseCon(xk, out);
  prob->addSparseJacobian(1.0, xk, x, out);
  prob->addSparseJacobian(-1.0, xk, xk, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptTrustRegion::addSparseJacobian( ParOptScalar alpha, 
                                           ParOptVec *x,
                                           ParOptVec *px, 
                                           ParOptVec *out ){
  prob->addSparseJacobian(alpha, xk, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptTrustRegion::addSparseJacobianTranspose( ParOptScalar alpha, 
                                                    ParOptVec *x,
                                                    ParOptVec *pzw,
                                                    ParOptVec *out ){
  prob->addSparseJacobianTranspose(alpha, xk, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptTrustRegion::addSparseInnerProduct( ParOptScalar alpha,
                                               ParOptVec *x,
                                               ParOptVec *cvec,
                                               ParOptScalar *A ){
  prob->addSparseInnerProduct(alpha, xk, cvec, A);
}
