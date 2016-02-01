#include "CyParOptProblem.h"

/*
  Copyright (c) 2014-2016 Graeme Kennedy. All rights reserved
*/

/*
  The constructor for the ParOptProblem wrapper
*/ 
CyParOptProblem::CyParOptProblem( MPI_Comm _comm,
				  int _nvars, int _ncon, 
				  int _nwcon, int _nwblock ):
ParOptProblem(_comm, _nvars, _ncon, _nwcon, _nwblock){
  // Set the default options
  isDenseInequal = 1;
  isSparseInequal = 1;
  useLower = 1;
  useUpper = 1;
  
  // Set the initial values for the callbacks
  self = NULL;
  getvarsandbounds = NULL;
  evalobjcon = NULL;
  evalobjcongradient = NULL;
  evalhvecproduct = NULL;
  evalsparsecon = NULL;
  addsparsejacobian = NULL;
  addsparsejacobiantranspose = NULL;
  addsparseinnerproduct = NULL;
}

CyParOptProblem::~CyParOptProblem(){}

/*
  Set options associated with the inequality constraints
*/
void CyParOptProblem::setInequalityOptions( int _isDenseInequal,
					    int _isSparseInequal,
					    int _useLower, 
					    int _useUpper ){
  isDenseInequal = _isDenseInequal;
  isSparseInequal = _isSparseInequal;
  useLower = _useLower;
  useUpper = _useUpper;
}

/*
  Function to indicate the type of sparse constraints
*/
int CyParOptProblem::isDenseInequality(){ 
  return isDenseInequal; 
}
int CyParOptProblem::isSparseInequality(){ 
  return isSparseInequal; 
}
int CyParOptProblem::useLowerBounds(){ 
  return useLower; 
}
int CyParOptProblem::useUpperBounds(){ 
  return useUpper; 
}

/*
  Set the member callback functions that are required
*/
void CyParOptProblem::setSelfPointer( void *_self ){
  self = _self;
}
void CyParOptProblem::setGetVarsAndBounds( void (*func)(void *, int, ParOptScalar*, 
							ParOptScalar*, ParOptScalar*) ){
  getvarsandbounds = func;
}
void CyParOptProblem::setEvalObjCon( int (*func)(void*, int, int, ParOptScalar*, 
						 ParOptScalar*, ParOptScalar*) ){
  evalobjcon = func;
}
void CyParOptProblem::setEvalObjConGradient( int (*func)(void*, int, int, ParOptScalar*, 
							 ParOptScalar*, ParOptScalar*) ){
  evalobjcongradient = func;
}
void CyParOptProblem::setEvalHvecProduct( int (*func)(void *, int, int, int,
						      ParOptScalar*, ParOptScalar*,
						      ParOptScalar*, ParOptScalar*,
						      ParOptScalar*) ){
  evalhvecproduct = func;
}
void CyParOptProblem::setEvalSparseCon( void (*func)(void *, int, int,
						     ParOptScalar*, ParOptScalar*) ){
  evalsparsecon = func;
}
void CyParOptProblem::setAddSparseJacobian( void (*func)(void *, int, int,
							 ParOptScalar, ParOptScalar*, 
							 ParOptScalar*, ParOptScalar*) ){
  addsparsejacobian = func;
}
void CyParOptProblem::setAddSparseJacobianTranspose( void (*func)(void *, int, int,
								  ParOptScalar, 
								  ParOptScalar*, 
								  ParOptScalar*, 
								  ParOptScalar*) ){
  addsparsejacobiantranspose = func;
}
void CyParOptProblem::setAddSparseInnerProduct( void (*func)(void *, int, int, int,
							     ParOptScalar, ParOptScalar*, 
							     ParOptScalar*, ParOptScalar*) ){
  addsparseinnerproduct = func;
}

/*
  Get the variables and bounds from the problem
*/
void CyParOptProblem::getVarsAndBounds( ParOptVec *x, 
					ParOptVec *lb, 
					ParOptVec *ub ){
  if (!getvarsandbounds){
    fprintf(stderr, "getvarsandbounds callback not defined\n");
    return;
  }
  ParOptScalar *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  getvarsandbounds(self, nvars, xvals, lbvals, ubvals);
}

/*
  Evaluate the objective and constraints
*/
int CyParOptProblem::evalObjCon( ParOptVec *x, 
				 ParOptScalar *fobj, 
				 ParOptScalar *cons ){
  if (!evalobjcon){
    fprintf(stderr, "evalobjcon callback not defined\n");
    return 1;
  }
  
  // Retrieve the (local) values of the design variables
  ParOptScalar *xvals;
  x->getArray(&xvals);
  
  // Evaluate the objective and constraints
  int fail = evalobjcon(self, nvars, ncon, 
			xvals, fobj, cons);
  
  return fail;
}

/*
  Evaluate the objective and constraint gradients
*/
int CyParOptProblem::evalObjConGradient( ParOptVec *x,
					 ParOptVec *g, 
					 ParOptVec **Ac ){
  if (!evalobjcongradient){
    fprintf(stderr, "evalobjcongradient callback not defined\n");
    return 1;
  }
  
  // Get the design and gradient arrays
  ParOptScalar *xvals, *gvals;
  x->getArray(&xvals);
  g->getArray(&gvals);

  // Allocate a contiguous arary for the Jacobian entries
  ParOptScalar *A = new ParOptScalar[ ncon*nvars ];
  
  // Evaluate the objective/constraint gradient 
  int fail = evalobjcongradient(self, nvars, ncon, 
				xvals, gvals, A);

  // Copy the values back into the vectors
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar *avals;
    Ac[i]->getArray(&avals);
    memcpy(avals, &A[i*nvars], nvars*sizeof(ParOptScalar));
  }
  
  // Free the contiguous array values
  delete [] A;
  
  return fail;    
}

/*
  Evaluate the product of the Hessian with a given vector
*/
int CyParOptProblem::evalHvecProduct( ParOptVec *x,
				      ParOptScalar *z, 
				      ParOptVec *zw,
				      ParOptVec *px, 
				      ParOptVec *hvec ){
  if (!evalhvecproduct){
    fprintf(stderr, "evalhvecproduct callback not defined\n");
    return 1;
  }

  // Get the arrays from the input vectors
  ParOptScalar *xvals, *zwvals, *pxvals, *hvals;
  x->getArray(&xvals);
  zw->getArray(&zwvals);
  px->getArray(&pxvals);
  hvec->getArray(&hvals);
  
  // Evaluate the Hessian-vector callback
  int fail = evalhvecproduct(self, nvars, ncon, nwcon,
			     xvals, z, zwvals, pxvals, hvals);
  return fail;
}

/*
  Evaluate the constraints
*/
void CyParOptProblem::evalSparseCon( ParOptVec *x, 
				     ParOptVec *out ){
  if (!evalsparsecon){
    fprintf(stderr, "evalsparsecon callback not defined\n");
    return;
  }

  // Get the arrays from the input vectors
  ParOptScalar *xvals, *outvals;
  x->getArray(&xvals);
  out->getArray(&outvals);
  
  // Evaluate the Hessian-vector callback
  evalsparsecon(self, nvars, nwcon,
		xvals, outvals);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void CyParOptProblem::addSparseJacobian( ParOptScalar alpha, 
					 ParOptVec *x,
					 ParOptVec *px, 
					 ParOptVec *out ){
  if (!addsparsejacobian){
    fprintf(stderr, "addsparsejacobian callback not defined\n");
    return;
  }
  
  // Get the arrays from the input vectors
  ParOptScalar *xvals, *pxvals, *outvals;
  x->getArray(&xvals);
  px->getArray(&pxvals);
  out->getArray(&outvals);
  
  // Evaluate the sparse Jacobian output
  addsparsejacobian(self, nvars, nwcon,
		    alpha, xvals, pxvals, outvals);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void CyParOptProblem::addSparseJacobianTranspose( ParOptScalar alpha, 
						  ParOptVec *x,
						  ParOptVec *pzw, 
						  ParOptVec *out ){
  if (!addsparsejacobiantranspose){
    fprintf(stderr, "addsparsejacobiantranspose callback not defined\n");
    return;
  }

  // Get the arrays from the input vectors
  ParOptScalar *xvals, *pzwvals, *outvals;
  x->getArray(&xvals);
  pzw->getArray(&pzwvals);
  out->getArray(&outvals);
  
  // Evaluate the sparse Jacobian output
  addsparsejacobiantranspose(self, nvars, nwcon,
			     alpha, xvals, pzwvals, outvals);
}

/*
  Add the inner product of the constraints to the matrix such that
  A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void CyParOptProblem::addSparseInnerProduct( ParOptScalar alpha, 
					     ParOptVec *x,
					     ParOptVec *cvec, 
					     ParOptScalar *A ){
  if (!addsparseinnerproduct){
    fprintf(stderr, "addsparseinnerproduct callback not defined\n");
    return;
  }
  // Get the arrays from the input vectors
  ParOptScalar *xvals, *cvals;
  x->getArray(&xvals);
  cvec->getArray(&cvals);
  
  // Evaluate the sparse Jacobian output
  addsparseinnerproduct(self, nvars, nwcon, nwblock, 
			alpha, xvals, cvals, A);
}
