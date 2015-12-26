#ifndef PAR_OPT_CYTHON_PROBLEM_H
#define PAR_OPT_CYTHON_PROBLEM_H

/*
  Copyright (c) 2014-2015 Graeme Kennedy. All rights reserved
*/

#include "ParOptProblem.h"

/*
  This code implements a simplifed interface for the ParOptProblem
  that can be wrapped using Cython. The class uses function pointers
  as an intermediate layer between the ParOptProblem class and the
  python layer. These callback functions can then be wrapped and set
  from python.

  Note: The class is currently restricted to utilize only the
  objective and gradient calculations and not the Hessian code. In
  addition, the sparse constraints are not implemented. This class
  then solves the (distributed) optimization problem:

  min    f(x)
  w.r.t. lb <= x <= ub
  s.t.   c(x) >= 0
*/
class CyParOptProblem : public ParOptProblem {
 public:
  CyParOptProblem( MPI_Comm _comm,
		   int _nvars, int _ncon ):
  ParOptProblem(_comm, _nvars, _ncon, 0, 0){
    self = NULL;
    getvarsandbounds = NULL;
    evalobjcon = NULL;
    evalobjcongradient = NULL;
  }
  ~CyParOptProblem(){}

  // Public member function pointers to the callbacks that are
  // required before class can be used
  // ---------------------------------------------------------
  void *self;
  void (*getvarsandbounds)( void *self, int nvars,
			    ParOptScalar *x, ParOptScalar *lb, ParOptScalar *ub );
  int (*evalobjcon)( void *self, int nvars, int ncon,
		     ParOptScalar *x, ParOptScalar *fobj, ParOptScalar *cons );
  int (*evalobjcongradient)( void *self, int nvars, int ncon,
			     ParOptScalar *x, ParOptScalar *gobj,
			     ParOptScalar *A );

  // Set the member callback functions that are required
  // ---------------------------------------------------
  void setSelfPointer( void *_self ){
    self = _self;
  }
  void setGetVarsAndBounds( void (*func)(void *, int, ParOptScalar*, 
					 ParOptScalar*, ParOptScalar*) ){
    getvarsandbounds = func;
  }
  void setEvalObjCon( int (*func)(void*, int, int, ParOptScalar*, 
				  ParOptScalar*, ParOptScalar*) ){
    evalobjcon = func;
  }
  void setEvalObjConGradient( int (*func)(void*, int, int, ParOptScalar*, 
					  ParOptScalar*, ParOptScalar*) ){
    evalobjcongradient = func;
  }

  // Function to indicate the type of sparse constraints
  // ---------------------------------------------------
  int isSparseInequality(){ return 1; }
  int isDenseInequality(){ return 1; }
  int useLowerBounds(){ return 1; }
  int useUpperBounds(){ return 1; }

  // Get the variables and bounds from the problem
  // ---------------------------------------------
  void getVarsAndBounds( ParOptVec *x, ParOptVec *lb, 
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
  
  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *x, ParOptScalar *fobj, 
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

  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *x,
			  ParOptVec *g, ParOptVec **Ac ){
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

  // Evaluate the product of the Hessian with a given vector
  // -------------------------------------------------------
  int evalHvecProduct( ParOptVec *x,
		       ParOptScalar *z, ParOptVec *zw,
		       ParOptVec *px, ParOptVec *hvec ){}

  // Evaluate the constraints
  // ------------------------
  void evalSparseCon( ParOptVec *x, ParOptVec *out ){}
  
  // Compute the Jacobian-vector product out = J(x)*px
  // --------------------------------------------------
  void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
			  ParOptVec *px, ParOptVec *out ){}

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
				   ParOptVec *pzw, ParOptVec *out ){}

  // Add the inner product of the constraints to the matrix such 
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  // ------------------------------------------------------------
  void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
			      ParOptVec *cvec, ParOptScalar *A ){}

};

#endif // PAR_OPT_CYTHON_PROBLEM_H
