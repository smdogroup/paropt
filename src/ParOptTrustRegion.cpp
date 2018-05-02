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
                                      double _penalty_value ):
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

  // Create the vectors
  xk = prob->createDesignVec();  xk->incref();
  lk = prob->createDesignVec();  lk->incref();
  uk = prob->createDesignVec();  uk->incref();
  lb = prob->createDesignVec();  lb->incref();
  ub = prob->createDesignVec();  ub->incref();

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

  // Set the initial values of the multipliers
  lamb = new ParOptScalar[ m ];
  for ( int i = 0; i < m; i++ ){
    lamb[i] = 0.0;
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
  delete [] lamb;

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
  
  // Add the contributions of the augmented Lagrangian
  for ( int i = 0; i < m; i++ ){
    fk = fk - lamb[i]*ck[i] + 0.5*penalty_value*ck[i]*ck[i];
    gk->axpy(-lamb[i] + penalty_value*ck[i], Ak[i]);
  }
}

/*
  Update the trust region problem
*/
void ParOptTrustRegion::update( ParOptVec *xt ){
  // Compute the step
  s->copyValues(xt);
  s->axpy(-1.0, xk);

  // Evaluate the objective and constraints and their gradients
  prob->evalObjCon(xt, &ft, ct);
  prob->evalObjConGradient(xt, gt, At);

  // Add the contributions of the augmented Lagrangian
  for ( int i = 0; i < m; i++ ){
    ft = ft - lamb[i]*ct[i] + 0.5*penalty_value*ct[i]*ct[i];
    gt->axpy(-lamb[i] + penalty_value*ct[i], At[i]);
  }

  // Compute the decrease in the model function
  qn->mult(s, t);
  ParOptScalar mk = -(gk->dot(s) + 0.5*t->dot(s));

  // Compute the ratio
  ParOptScalar rho = (fk - ft)/mk;

  // Perform an update of the quasi-Newton approximation
  t->copyValues(gt);
  t->axpy(-1.0, gk);
  qn->update(s, t);

  // Check the new point
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
  if (rho < 0.25){
    tr_size = max2(0.25*tr_size, tr_min_size);
  }
  else if (rho > 0.75){
    tr_size = min2(2.0*tr_size, tr_max_size);
  }

  // Reset the trust region radius bounds
  setTrustRegionBounds(tr_size, xk, lb, ub, lk, uk);
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
