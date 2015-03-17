#ifndef ROSENBROCK_H
#define ROSENBROCK_H

#include "ParOptProblem.h"
#include "ParOptVec.h"

class Rosenbrock : public ParOptProblem {
 public:
  Rosenbrock( MPI_Comm comm, int _nvars, int _nwcon,
	      int _nwstart, int _nw, int _nwskip);
  //Determine whether there is an inequality constraint
  int isSparseInequality();
  int isDenseInequality();
  int useLowerBounds();
  int useUpperBounds();

  //Get the communicator for the problem
  MPI_Comm getMPIComm(){
    return comm;
    
  }

  //Get the problem dimensions
  void getProblemSizes( int *_nvars, int *_ncon,
			int *_nwcon, int *_nwblock){
    *_nvars = nvars;
    *_ncon = ncon;
    *_nwcon = nwcon;
    *_nwblock = nwblock;

  }
  //Get variables and bounds
  void getVarsAndBounds( ParOptVec *xvec,
			 ParOptVec *lbvec,
			 ParOptVec *ubvec);

  //Evaluate the objective and constraints
  int evalObjCon( ParOptVec *xvec, double *fobj,
		  double *cons);

  //Evaluate the objective and constraints gradients
  int evalObjConGradient( ParOptVec *xvec, ParOptVec *gvec, 
			  ParOptVec **Ac);

  //Evaluate the Hessian vector products
  int evalHvecProduct( ParOptVec *xvec, double *z, ParOptVec *zwvec,
		       ParOptVec *pxvec, ParOptVec *hvec);

  //Evaluate the sparse constraints
  void evalSparseCon( ParOptVec *x, ParOptVec *out);

  //Compute the Jacobian-vector product out = J(x)*px
  void addSparseJacobian( double alpha, ParOptVec *x, 
			  ParOptVec *px, ParOptVec *out);

  //Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  void addSparseJacobianTranspose( double alpha, ParOptVec *x, 
				   ParOptVec *pzw, ParOptVec *out);

  //Add the inner product of the constraints to the matrix such
  //that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  void addSparseInnerProduct( double alpha, ParOptVec *x, 
			      ParOptVec *cvec, double *A);

  int nwcon;
  int nwstart;
  int nw, nwskip;
  double scale;

};
#endif
