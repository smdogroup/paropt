#ifndef PAR_OPT_PROBLEM_H
#define PAR_OPT_PROBLEM_H

/*
  Copyright (c) 2014 Graeme Kennedy. All rights reserved
*/

#include "ParOptVec.h"

/*
  This code is the virtual base class problem definition for the
  parallel optimizer.

  To get this to work you must override the following virtual
  functions:

  getVarsAndBounds(): This function returns the variables and bounds
  for the problem. This is called once at the initialization. The
  starting point is taken from the

  evalObjCon(): This function takes in the design variables and
  returns an objective and constraint values. The function returns a
  fail flag (e.g. fail = evalObjCon(x, &fobj, con);). Return fail != 0
  if the function cannot be evaluated at the provided values of the
  design variables.

  evalObjConGradient(): This function evaluates the objective and
  constraint gradients at the current point. Again, the fail flag is
  given as above. Note that the constraint gradients are returned as a
  series of dense vectors.

  The class takes as input the communicator for the optimizer, the
  number of local design variables on the given process, and the
  number of constraints in the problem.

  input:
  comm:    the communicator
  nvars:   the number of local variables
  ncon:    the number of dense constraints
  nwcon:   the number of sparse constraints
  nwblock: the block size of the Aw*D*Aw^{T} matrix
*/
class ParOptProblem {
 public:
  ParOptProblem( MPI_Comm _comm,
		 int _nvars, int _ncon,
		 int _nwcon, int _nwblock ){
    comm = _comm;
    nvars = _nvars;
    ncon = _ncon;
    nwcon = _nwcon;
    nwblock = _nwblock;
  }
  virtual ~ParOptProblem(){}

  // Get the communicator for the problem
  // ------------------------------------
  MPI_Comm getMPIComm(){
    return comm;
  }
    
  // Get the problem dimensions
  // --------------------------
  void getProblemSizes( int *_nvars, int *_ncon, 
			int *_nwcon, int *_nwblock ){
    *_nvars = nvars;
    *_ncon = ncon;
    *_nwcon = nwcon;
    *_nwblock = nwblock;
  }

  // Function to indicate whether the sparse constraints
  // ---------------------------------------------------
  virtual int isSparseInequality() = 0;

  // Get the variables and bounds from the problem
  // ---------------------------------------------
  virtual void getVarsAndBounds( ParOptVec *x,
				 ParOptVec *lb, 
				 ParOptVec *ub ) = 0;
  
  // Evaluate the objective and constraints
  // --------------------------------------
  virtual int evalObjCon( ParOptVec *x, 
			  double *fobj, double *cons ) = 0;

  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  virtual int evalObjConGradient( ParOptVec *x,
				  ParOptVec *g, ParOptVec **Ac ) = 0;

  // Evaluate the product of the Hessian with a given vector
  // -------------------------------------------------------
  virtual int evalHvecProduct( ParOptVec *x,
			       double *z, ParOptVec *zw,
			       ParOptVec *px, ParOptVec *hvec ) = 0;

  // Evaluate the constraints
  // ------------------------
  virtual void evalSparseCon( ParOptVec *x, ParOptVec *out ) = 0;
  
  // Compute the Jacobian-vector product out = J(x)*px
  // --------------------------------------------------
  virtual void addSparseJacobian( double alpha, ParOptVec *x,
				  ParOptVec *px, ParOptVec *out ) = 0;

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  virtual void addSparseJacobianTranspose( double alpha, ParOptVec *x,
					   ParOptVec *pzw, ParOptVec *out ) = 0;

  // Add the inner product of the constraints to the matrix such 
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  // ------------------------------------------------------------
  virtual void addSparseInnerProduct( double alpha, ParOptVec *x,
				      ParOptVec *cvec, double *A ) = 0;

  // Over-write this function if you'd like to print out
  // something with the same frequency as the output files
  // -----------------------------------------------------
  virtual void writeOutput( int iter, ParOptVec * x ){}
  
 protected:
  MPI_Comm comm;
  int nvars, ncon, nwcon, nwblock;
};

#endif
