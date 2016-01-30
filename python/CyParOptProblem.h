#ifndef PAR_OPT_CYTHON_PROBLEM_H
#define PAR_OPT_CYTHON_PROBLEM_H

/*
  Copyright (c) 2014-2016 Graeme Kennedy. All rights reserved
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
		   int _nvars, int _ncon, 
		   int _nwcon, int _nwblock );
  ~CyParOptProblem();

  // Set options associated with the inequality constraints
  // ------------------------------------------------------
  void setInequalityOptions( int _isDenseInequal,
			     int _isSparseInequal,
			     int _useLower, int _useUpper );

  // Function to indicate the type of sparse constraints
  // ---------------------------------------------------
  int isDenseInequality();
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Set the member callback functions that are required
  // ---------------------------------------------------
  void setSelfPointer( void *_self );
  void setGetVarsAndBounds( void (*func)(void*, int, ParOptScalar*, 
					 ParOptScalar*, ParOptScalar*) );
  void setEvalObjCon( int (*func)(void*, int, int, ParOptScalar*, 
				  ParOptScalar*, ParOptScalar*) );
  void setEvalObjConGradient( int (*func)(void*, int, int, ParOptScalar*, 
					  ParOptScalar*, ParOptScalar*) );
  void setEvalHvecProduct( int (*func)(void*, int, int, int,
				       ParOptScalar *, ParOptScalar *,
				       ParOptScalar *, ParOptScalar *,
				       ParOptScalar *) );
  void setEvalSparseCon( void (*func)(void*, int, int, int,
				      ParOptScalar*, ParOptScalar*) );
  void setAddSparseJacobian( void (*func)(void*, int, int, int,
					  ParOptScalar, ParOptScalar*, 
					  ParOptScalar*, ParOptScalar*) );
  void setAddSparseJacobianTranspose( void (*func)(void*, int, int, int,
						   ParOptScalar, ParOptScalar*,
						   ParOptScalar*, ParOptScalar*) );
  void setAddSparseInnerProduct( void (*func)(void*, int, int, int,
					      ParOptScalar, ParOptScalar*, 
					      ParOptScalar*, ParOptScalar*) );

  // Get the variables and bounds from the problem
  // ---------------------------------------------
  void getVarsAndBounds( ParOptVec *x, ParOptVec *lb, 
			 ParOptVec *ub );
  
  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *x, ParOptScalar *fobj, 
		  ParOptScalar *cons );

  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *x,
			  ParOptVec *g, ParOptVec **Ac );

  // Evaluate the product of the Hessian with a given vector
  // -------------------------------------------------------
  int evalHvecProduct( ParOptVec *x,
		       ParOptScalar *z, ParOptVec *zw,
		       ParOptVec *px, ParOptVec *hvec );

  // Evaluate the constraints
  // ------------------------
  void evalSparseCon( ParOptVec *x, ParOptVec *out );
  
  // Compute the Jacobian-vector product out = J(x)*px
  // --------------------------------------------------
  void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
			  ParOptVec *px, ParOptVec *out );

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
				   ParOptVec *pzw, ParOptVec *out );

  // Add the inner product of the constraints to the matrix such 
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  // ------------------------------------------------------------
  void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
			      ParOptVec *cvec, ParOptScalar *A );

 private:
  // Public member function pointers to the callbacks that are
  // required before class can be used
  // ---------------------------------------------------------
  void *self;
  void (*getvarsandbounds)( void *self, int nvars,
			    ParOptScalar *x, ParOptScalar *lb, 
			    ParOptScalar *ub );
  int (*evalobjcon)( void *self, int nvars, int ncon,
		     ParOptScalar *x, ParOptScalar *fobj, 
		     ParOptScalar *cons );
  int (*evalobjcongradient)( void *self, int nvars, int ncon,
			     ParOptScalar *x, ParOptScalar *gobj,
			     ParOptScalar *A );
  int (*evalhvecproduct)( void *self, int nvars, int ncon, int nwcon,
			  ParOptScalar *x, ParOptScalar *z,
			  ParOptScalar *zw, ParOptScalar *px,
			  ParOptScalar *hvec );
  void (*evalsparsecon)( void *self, int nvars, int nwcon, int nwblock,
			 ParOptScalar *x, ParOptScalar *out );
  void (*addsparsejacobian)( void *self, int nvars, int nwcon, int nwblock,
			     ParOptScalar alpha, ParOptScalar *x, 
			     ParOptScalar *px, ParOptScalar *out );
  void (*addsparsejacobiantranspose)( void *self, int nvars, 
				      int nwcon, int nwblock,
				      ParOptScalar alpha, ParOptScalar *x, 
				      ParOptScalar *px, ParOptScalar *out );
  void (*addsparseinnerproduct)( void *self,
				 int nvars, int nwcon, int nwblock,
				 ParOptScalar alpha, ParOptScalar *x, 
				 ParOptScalar *c, ParOptScalar *A );

  // Store information about the type of problem to solve
  int isDenseInequal;
  int isSparseInequal;
  int useLower;
  int useUpper;
};

#endif // PAR_OPT_CYTHON_PROBLEM_H
