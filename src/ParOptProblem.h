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
  comm:   the communicator
  nvars:  the number of local variables
  ncon:   the number of consstraints
*/
class ParOptProblem {
 public:
  ParOptProblem( MPI_Comm _comm,
		 int _nvars, int _ncon ){
    comm = _comm;
    nvars = _nvars;
    ncon = _ncon;
  }
  virtual ~ParOptProblem(){}

  // Get the communicator for the problem
  // ------------------------------------
  MPI_Comm getMPIComm(){
    return comm;
  }
    
  // Get the problem dimensions
  // --------------------------
  void getProblemSizes( int *_nvars, int *_ncon ){
    *_nvars = nvars;
    *_ncon = ncon;
  }

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

  // Over-write this function if you'd like to print out
  // something with the same frequency as the output files
  // -----------------------------------------------------
  virtual void writeOutput( int iter, ParOptVec * x ){}
  
 protected:
  MPI_Comm comm;
  int nvars, ncon;
};

/*
  The following class defines the sparse constraints that can be used
  during the optimization.  

  The following class implements a series of sparse constraints.

*/
class ParOptConstraint {
 public:
  ParOptConstraint( int _nwcon, int _nwstart, 
		    int _nw, int _nwskip ){
    nwcon = _nwcon;
    nwstart = _nwstart;
    nw = _nw;
    nwskip = _nwskip;
  }

  // Function to indicate whether the constraints are inequalities
  // -------------------------------------------------------------
  virtual int inequality(){
    return 0;
  }

  // Get the number of constriaints on this processor
  // ------------------------------------------------
  virtual int getNumConstraints(){
    return nwcon;
  }

  // Return the block size of the problem
  // ------------------------------------
  virtual int getBlockSize(){
    return 1;
  }

  // Evaluate the constraints
  // ------------------------
  virtual void evalCon( ParOptVec *x, ParOptVec *out ){
    double *xvals, *outvals; 
    x->getArray(&xvals);
    out->getArray(&outvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      outvals[i] = 1.0;
      for ( int k = 0; k < nw; k++, j++ ){
	outvals[i] -= xvals[j];
      }
    }
  }
  
  // Compute the Jacobian-vector product out = J(x)*px
  // --------------------------------------------------
  virtual void addJacobian( double alpha, ParOptVec *x,
			    ParOptVec *px, ParOptVec *out ){
    double *pxvals, *outvals; 
    px->getArray(&pxvals);
    out->getArray(&outvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	outvals[i] += alpha*pxvals[j];
      }
    }
  }

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  virtual void addJacobianTranspose( double alpha, ParOptVec *x,
				     ParOptVec *pzw, ParOptVec *out ){
    double *outvals, *pzwvals;
    out->getArray(&outvals);
    pzw->getArray(&pzwvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	outvals[j] += alpha*pzwvals[i];
      }
    }
  }

  // Add the inner product of the constraints to the matrix such 
  // that A += As*cvec*As where cvec is a diagonal matrix
  // ------------------------------------------------------
  virtual void addInnerProduct( double alpha, ParOptVec *x,
				ParOptVec *cvec, double *A ){
    double *cvals;
    cvec->getArray(&cvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	A[i] += alpha*cvals[j];
      }
    }
  }

 private:
  int nwcon;
  int nwstart, nw, nwskip;
};

#endif
