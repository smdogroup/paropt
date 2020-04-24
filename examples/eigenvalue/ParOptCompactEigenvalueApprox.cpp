
#include "ParOptCompactEigenvalueApprox.h"
#include "ParOptComplexStep.h"

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

ParOptCompactEigenApprox::ParOptCompactEigenApprox( ParOptProblem *problem,
                                                    int _N ){
  c0 = 0.0;
  g0 = problem->createDesignVec();
  g0->incref();

  N = _N;
  tmp = new ParOptScalar[ N ];
  M = new ParOptScalar[ N*N ];
  memset(M, 0, N*N*sizeof(ParOptScalar));
  hvecs = new ParOptVec*[ N ];
  for ( int i = 0; i < N; i++ ){
    hvecs[i] = problem->createDesignVec();
    hvecs[i]->incref();
  }
}

ParOptCompactEigenApprox::~ParOptCompactEigenApprox(){
  delete [] tmp;
  delete [] M;
  g0->decref();
  for ( int i = 0; i < N; i++ ){
    hvecs[i]->decref();
  }
}

void ParOptCompactEigenApprox::multAdd( ParOptScalar alpha,
                                        ParOptVec *x,
                                        ParOptVec *y ){
  x->mdot(hvecs, N, tmp);

  for ( int i = 0; i < N; i++ ){
    ParOptScalar scale = 0.0;
    for ( int j = 0; j < N; j++ ){
      scale += M[i*N + j]*tmp[j];
    }

    y->axpy(alpha*scale, hvecs[i]);
  }
}

void ParOptCompactEigenApprox::getApproximation( ParOptScalar **_c0,
                                                 ParOptVec **_g0,
                                                 int *_N,
                                                 ParOptScalar **_M,
                                                 ParOptVec ***_hvecs ){
  if (_c0){
    *_c0 = &c0;
  }
  if (_g0){
    *_g0 = g0;
  }
  if (_N){
    *_N = N;
  }
  if (_M){
    *_M = M;
  }
  if (_hvecs){
    *_hvecs = hvecs;
  }
}

ParOptScalar ParOptCompactEigenApprox::evalApproximation( ParOptVec *s,
                                                          ParOptVec *t ){
  ParOptScalar c = c0;
  if (s && t){
    c += g0->dot(s);

    s->mdot(hvecs, N, tmp);
    for ( int i = 0; i < N; i++ ){
      for ( int j = 0; j < N; j++ ){
        c += 0.5*M[i*N + j]*tmp[i]*tmp[j];
      }
    }
  }

  return c;
}

void ParOptCompactEigenApprox::evalApproximationGradient( ParOptVec *s,
                                                          ParOptVec *grad ){
  grad->copyValues(g0);

  s->mdot(hvecs, N, tmp);
  for ( int i = 0; i < N; i++ ){
    ParOptScalar scale = 0.0;
    for ( int j = 0; j < N; j++ ){
      scale += M[i*N + j]*tmp[j];
    }
    grad->axpy(scale, hvecs[i]);
  }
}

ParOptEigenQuasiNewton::ParOptEigenQuasiNewton( ParOptCompactQuasiNewton *_qn,
                                                ParOptCompactEigenApprox *_eigh ){
  qn = _qn;
  if (qn){
    qn->incref();
  }

  eigh = _eigh;
  eigh->incref();

  // Set the initial multiplier
  z0 = 0.0;

  // Set the max number of vectors used to approximate
  int N;
  eigh->getApproximation(NULL, NULL, &N, NULL, NULL);
  max_vecs = N;
  if (qn){
    max_vecs = qn->getMaxLimitedMemorySize() + N;
  }

  d = new ParOptScalar[ max_vecs ];
  M = new ParOptScalar[ max_vecs*max_vecs ];
  Z = new ParOptVec*[ max_vecs ];
}

ParOptEigenQuasiNewton::~ParOptEigenQuasiNewton(){
  if (qn){
    qn->decref();
  }
  eigh->decref();

  delete [] Z;
  delete [] M;
  delete [] d;
}

// Reset the internal data
void ParOptEigenQuasiNewton::reset(){
  if (qn){
    qn->reset();
  }
}

int ParOptEigenQuasiNewton::update( ParOptVec *x,
                                    const ParOptScalar *z,
                                    ParOptVec *zw,
                                    ParOptVec *s,
                                    ParOptVec *y ){
  return 0;
}

int ParOptEigenQuasiNewton::update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw ){
  // Set the multiplier
  z0 = z[0];

  return 0;
}

void ParOptEigenQuasiNewton::mult( ParOptVec *x, ParOptVec *y ){
  if (qn){
    qn->mult(x, y);
  }
  else {
    y->zeroEntries();
  }
  eigh->multAdd(-z0, x, y);
}

void ParOptEigenQuasiNewton::multAdd( ParOptScalar alpha,
                                      ParOptVec *x,
                                      ParOptVec *y ){
  if (qn){
    qn->multAdd(alpha, x, y);
  }
  eigh->multAdd(-alpha*z0, x, y);
}

// Get the compact representation of the quasi-Newton method
int ParOptEigenQuasiNewton::getCompactMat( ParOptScalar *b0,
                                           const ParOptScalar **_d,
                                           const ParOptScalar **_M,
                                           ParOptVec ***_Z ){
  memset(M, 0, max_vecs*max_vecs*sizeof(ParOptScalar));

  // Get the size of the eigenvalue hessian approximation
  int N;
  eigh->getApproximation(NULL, NULL, &N, NULL, NULL);

  // Set the matrix size, neglecting the quasi-Newton Hessian
  int mat_size = N;
  int qn_size = 0;

  if (qn){
    // Get the compact quasi-Newton approximation
    ParOptVec **Z0;
    const ParOptScalar *d0, *M0;
    qn_size = qn->getCompactMat(b0, &d0, &M0, &Z0);

    // Set the matrix size
    mat_size = N + qn_size;

    // Set the values into the matrix
    for ( int i = 0; i < qn_size; i++ ){
      d[i] = d0[i];
      Z[i] = Z0[i];

      for ( int j = 0; j < qn_size; j++ ){
        M[i*mat_size + j] = M0[i*qn_size + j];
      }
    }
  }

  // Get the Hessian approximation of the approximation
  ParOptVec **Z1;
  ParOptScalar *M1;
  eigh->getApproximation(NULL, NULL, &N, &M1, &Z1);

  for ( int i = 0; i < N; i++ ){
    d[qn_size + i] = 1.0;
    Z[qn_size + i] = Z1[i];

    for ( int j = 0; j < N; j++ ){
      M[(qn_size + i)*mat_size + qn_size + j] = -z0*M1[i*N + j];
    }
  }

  if (_d){ *_d = d; }
  if (_M){ *_M = M; }
  if (_Z){ *_Z = Z; }

  return mat_size;
}

// Get the maximum size of the limited-memory BFGS
int ParOptEigenQuasiNewton::getMaxLimitedMemorySize(){
  return max_vecs;
}

ParOptEigenSubproblem::ParOptEigenSubproblem( ParOptProblem *_prob,
                                              ParOptEigenQuasiNewton *_approx ):
  ParOptTrustRegionSubproblem(_prob->getMPIComm()){
  // Paropt problem instance
  prob = _prob;
  prob->incref();

  // Get and set the problem sizes
  prob->getProblemSizes(&n, &m, &nwcon, &nwblock);
  setProblemSizes(n, m, nwcon, nwblock);

  // Set the quasi-Newton method
  approx = _approx;
  approx->incref();

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

ParOptEigenSubproblem::~ParOptEigenSubproblem(){
  prob->decref();
  approx->decref();

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
ParOptCompactQuasiNewton* ParOptEigenSubproblem::getQuasiNewton(){
  return approx;
}

/*
  Initialize the model at the starting point
*/
void ParOptEigenSubproblem::initModelAndBounds( double tr_size ){
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
void ParOptEigenSubproblem::setTrustRegionBounds( double tr_size ){
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

int ParOptEigenSubproblem::evalTrialPointAndUpdate( ParOptVec *x,
                                                    const ParOptScalar *z,
                                                    ParOptVec *zw,
                                                    ParOptScalar *fobj,
                                                    ParOptScalar *cons ){
  int fail = prob->evalObjCon(x, &ft, ct);
  fail = fail || prob->evalObjConGradient(x, gt, At);

  // If we're using a quasi-Newton Hessian approximation
  ParOptCompactQuasiNewton *qn = approx->getCompactQuasiNewton();
  if (qn){
    // Compute the step s = x - xk
    s->copyValues(x);
    s->axpy(-1.0, xk);

    // Compute the difference between the gradient of the
    // Lagrangian between the current point and the previous point
    t->copyValues(gt);
    t->axpy(-1.0, gk);

    // Perform an update of the quasi-Newton approximation
    qn->update(xk, z, zw, s, t);
  }

  return fail;
}

int ParOptEigenSubproblem::acceptTrialPoint( ParOptVec *x,
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

void ParOptEigenSubproblem::rejectTrialPoint(){
  ft = 0.0;
  for ( int i = 0; i < m; i++ ){
    ct[i] = 0.0;
  }
}

/*
  Create a design vector
*/
ParOptVec *ParOptEigenSubproblem::createDesignVec(){
  return prob->createDesignVec();
}

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptEigenSubproblem::createConstraintVec(){
  return prob->createConstraintVec();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptEigenSubproblem::getMPIComm(){
  return prob->getMPIComm();
}

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptEigenSubproblem::isDenseInequality(){
  return prob->isDenseInequality();
}

int ParOptEigenSubproblem::isSparseInequality(){
  return prob->isSparseInequality();
}

int ParOptEigenSubproblem::useLowerBounds(){
  return 1;
}

int ParOptEigenSubproblem::useUpperBounds(){
  return 1;
}

// Get the variables and bounds from the problem
void ParOptEigenSubproblem::getVarsAndBounds( ParOptVec *x,
                                              ParOptVec *l,
                                              ParOptVec *u ){
  x->copyValues(xk);
  l->copyValues(lk);
  u->copyValues(uk);
}

/*
  Evaluate the objective and constraint functions
*/
int ParOptEigenSubproblem::evalObjCon( ParOptVec *x,
                                       ParOptScalar *fobj,
                                       ParOptScalar *cons ){
  if (x){
    s->copyValues(x);
    s->axpy(-1.0, xk);

    // Compute the objective function
    *fobj = fk + gk->dot(s);
    ParOptCompactQuasiNewton *qn = approx->getCompactQuasiNewton();
    if (qn){
      qn->mult(s, t);
      *fobj += 0.5*s->dot(t);
    }

    // Compute the constraint functions
    ParOptCompactEigenApprox *eigh = approx->getCompactEigenApprox();
    cons[0] = eigh->evalApproximation(s, t);
    for ( int i = 1; i < m; i++ ){
      cons[i] = ck[i] + Ak[i]->dot(s);
    }
  }
  else {
    // If x is NULL, assume x = xk
    *fobj = fk;

    ParOptCompactEigenApprox *eigh = approx->getCompactEigenApprox();
    cons[0] = eigh->evalApproximation(NULL, NULL);
    for ( int i = 1; i < m; i++ ){
      cons[i] = ck[i];
    }
  }

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptEigenSubproblem::evalObjConGradient( ParOptVec *x,
                                               ParOptVec *g,
                                               ParOptVec **Ac ){
  s->copyValues(x);
  s->axpy(-1.0, xk);

  // Copy the values of constraint gradient
  ParOptCompactEigenApprox *eigh = approx->getCompactEigenApprox();
  eigh->evalApproximationGradient(s, Ac[0]);
  for ( int i = 1; i < m; i++ ){
    Ac[i]->copyValues(Ak[i]);
  }

  // Evaluate the gradient of the quadratic objective
  ParOptCompactQuasiNewton *qn = approx->getCompactQuasiNewton();
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
void ParOptEigenSubproblem::evalSparseCon( ParOptVec *x,
                                           ParOptVec *out ){
  prob->evalSparseCon(xk, out);
  prob->addSparseJacobian(1.0, xk, x, out);
  prob->addSparseJacobian(-1.0, xk, xk, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptEigenSubproblem::addSparseJacobian( ParOptScalar alpha,
                                               ParOptVec *x,
                                               ParOptVec *px,
                                               ParOptVec *out ){
  prob->addSparseJacobian(alpha, xk, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptEigenSubproblem::addSparseJacobianTranspose( ParOptScalar alpha,
                                                        ParOptVec *x,
                                                        ParOptVec *pzw,
                                                        ParOptVec *out ){
  prob->addSparseJacobianTranspose(alpha, xk, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptEigenSubproblem::addSparseInnerProduct( ParOptScalar alpha,
                                                   ParOptVec *x,
                                                   ParOptVec *cvec,
                                                   ParOptScalar *A ){
  prob->addSparseInnerProduct(alpha, xk, cvec, A);
}

/*
  Get the model at the current point
*/
int ParOptEigenSubproblem::getLinearModel( ParOptVec **_xk,
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
