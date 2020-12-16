#ifndef PAR_OPT_PROBLEM_H
#define PAR_OPT_PROBLEM_H

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
class ParOptProblem : public ParOptBase {
 public:
  /**
    Create an empty ParOptProblem class without defining the problem layout.

    @param _comm is the MPI communicator
  */
  ParOptProblem( MPI_Comm _comm ){
    comm = _comm;
    nvars = ncon = ninequality = nwcon = nwblock = 0;
  }

  /**
    Create a ParOptProblem class and define the problem layout.

    @param _comm is the MPI communicator
    @param _nvars the number of local design variables
    @param _ncon the global number of dense constraints
    @param _ninequality the number of inequality constraints
    @param _nwcon the local number of sparse separable constraints
    @param _nwblock the block size of the separable constraints
  */
  ParOptProblem( MPI_Comm _comm,
                 int _nvars, int _ncon, int _ninequality,
                 int _nwcon, int _nwblock ){
    comm = _comm;
    nvars = _nvars;
    ncon = _ncon;
    ninequality = _ninequality;
    if (ninequality > ncon){
      ninequality = ncon;
    }
    nwcon = _nwcon;
    nwblock = _nwblock;
  }
  virtual ~ParOptProblem(){}

  /**
    Create a new distributed design vector

    @return a new distributed design vector
  */
  virtual ParOptVec *createDesignVec(){
    return new ParOptBasicVec(comm, nvars);
  }

  /**
    Create a new distributed sparse constraint vector

    @return a new distributed sparse constraint vector
  */
  virtual ParOptVec *createConstraintVec(){
    return new ParOptBasicVec(comm, nwcon);
  }

  /**
    Get the communicator for the problem

    @return the MPI communicator for the problem
  */
  MPI_Comm getMPIComm(){
    return comm;
  }

  /**
    Set the problem size

    @param _nvars the number of local design variables
    @param _ncon the global number of dense constraints
    @param _ninequality the number of inequality constraints
    @param _nwcon the local number of sparse separable constraints
    @param _nwblock the block size of the separable constraints
  */
  void setProblemSizes( int _nvars, int _ncon, int _ninequality,
                        int _nwcon, int _nwblock ){
    nvars = _nvars;
    ncon = _ncon;
    ninequality = _ninequality;
    if (ninequality > ncon){
      ninequality = ncon;
    }
    nwcon = _nwcon;
    nwblock = _nwblock;
  }

  /**
    Get the problem size

    @param _nvars the number of local design variables
    @param _ncon the global number of dense constraints
    @param _ninequality the number of dense inequality constraints
    @param _nwcon the local number of sparse separable constraints
    @param _nwblock the block size of the separable constraints
  */
  void getProblemSizes( int *_nvars, int *_ncon, int *_ninequality,
                        int *_nwcon, int *_nwblock ){
    if (_nvars){ *_nvars = nvars; }
    if (_ncon){ *_ncon = ncon; }
    if (_ninequality){ *_ninequality = ninequality; }
    if (_nwcon){ *_nwcon = nwcon; }
    if (_nwblock){ *_nwblock = nwblock; }
  }

  /**
    Are the dense constraints inequalities? Default is true.

    @return flag indicating if the dense constraints are inequalities
  */
  virtual int isSparseInequality(){ return 1; }

  /**
    Indicate whether to use the lower variable bounds. Default is true.

    @return flag indicating whether to use lower variable bound.
  */
  virtual int useLowerBounds(){ return 1; }

 /**
    Indicate whether to use the upper variable bounds. Default is true.

    @return flag indicating whether to use upper variable bounds.
  */
  virtual int useUpperBounds(){ return 1; }

  /**
    Get the initial variable values and bounds for the problem

    @param x is a vector of the initial design variable values
    @param lb are the lower variable bounds
    @param ub are the upper variable bounds
  */
  virtual void getVarsAndBounds( ParOptVec *x,
                                 ParOptVec *lb,
                                 ParOptVec *ub ) = 0;

  /**
    Evaluate the objective and constraints.

    Given the design variables x, compute the scalar objective value
    and the dense constraints. The constraints and objective must be
    consistent across all processors.

    @param x is the design variable vector
    @param fobj is the objective value at x
    @param cons is the array of constraint vaules at x
    @return zero on success, non-zero fail flag on error
  */
  virtual int evalObjCon( ParOptVec *x,
                          ParOptScalar *fobj,
                          ParOptScalar *cons ) = 0;

  /**
    Evaluate the objective and constraint gradients.

    Given the desgin variables x, compute the gradient of the objective
    and dense constraint functions. This call is made only after a call
    to evaluate the objective and dense constraint functions.

    @param x is the design variable vector
    @param g is the gradient of the objective at x
    @param Ac are the gradients of the dense constraints at x
    @return zero on success, non-zero fail flag on error
  */
  virtual int evalObjConGradient( ParOptVec *x,
                                  ParOptVec *g, ParOptVec **Ac ) = 0;

  /**
    Evaluate the product of the Hessian with a given vector.

    This function is only called if Hessian-vector products are requested
    by the optimizer. By default, no implementation is provided.

    @param x is the design variable vector
    @param z is the array of multipliers for the dense constraints
    @param zw is the vector of multipliers for the sparse constraints
    @param px is the direction vector
    @param hvec is the output vector hvec = H(x,z,zw)*px
    @return zero on success, non-zero flag on error
  */
  virtual int evalHvecProduct( ParOptVec *x,
                               ParOptScalar *z, ParOptVec *zw,
                               ParOptVec *px, ParOptVec *hvec ){
    return 0;
  }

  /**
    Evaluate the diagonal of the Hessian.

    This is only used by MMA.
  */
  virtual int evalHessianDiag( ParOptVec *x,
                               ParOptScalar *z, ParOptVec *zw,
                               ParOptVec *hdiag ){
    return 0;
  }

  /**
    Compute a correction or modification of the quasi-Newton update.

    The vectors s and y represent the step and gradient difference,
    respectively between iterations. By default, no correction or
    modification is performed. However, some problems may benefit by
    modifying the gradient difference term.

    @param x is the design variable vector
    @param z is the array of multipliers for the dense constraints
    @param zw is the vector of multipliers for the sparse constraints
    @param s The step in the quasi-Newton update
    @param y The gradient difference in the quasi-Newton update
  */
  virtual void computeQuasiNewtonUpdateCorrection( ParOptVec *x,
                                                   ParOptScalar *z,
                                                   ParOptVec *zw,
                                                   ParOptVec *s,
                                                   ParOptVec *y ){}

  /**
    Evaluate the sparse constraints.

    Give the design variable vector x, compute the sparse constraints.

    @param x is the design variable vector
    @param out is the sparse constraint vector
  */
  virtual void evalSparseCon( ParOptVec *x, ParOptVec *out ){}

  /**
    Compute the Jacobian-vector product of the sparse constraints.

    This code computes the action of the Jacobian of the sparse constraint
    matrix on the input vector px, to compute out = alpha*J(x)*px.

    @param alpha is a scalar factor
    @param x is the design variable vector
    @param px is the input direction vector
    @param out is the sparse product vector
  */
  virtual void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *px, ParOptVec *out ){}

  /**
    Compute the tranpose Jacobian-vector product of the sparse constraints.

    This code computes the action of the transpose Jacobian of the sparse
    constraint matrix on the input vector pzw, to compute
    out = alpha*J(x)^{T}*pzw.

    @param alpha is a scalar factor
    @param x is the design variable vector
    @param pzw is the input direction vector
    @param out is the sparse product vector
  */
  virtual void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                           ParOptVec *pzw, ParOptVec *out ){}

  /**
    Add the inner product of the constraints to the matrix such
    that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix

    @param alpha is a scalar factor
    @param x is the design variable vector
    @param cvec are input components of the diagonal matrix
    @param A is the output block-diagonal matrix
  */
  virtual void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                                      ParOptVec *cvec, ParOptScalar *A ){}

  /**
    Check the objective and constraint gradients for this problem instance

    @param dh Finite difference step size used for verification
    @param x The design vector at which point to check the gradient (can be NULL)
  */
  void checkGradients( double dh, ParOptVec *x=NULL,
                       int check_hvec_product=0 );

  //! Implement this function if you'd like to print out
  //! something with the same frequency as the output files
  // ------------------------------------------------------
  virtual void writeOutput( int iter, ParOptVec *x ){}

 protected:
  MPI_Comm comm;
  int nvars, ncon, ninequality, nwcon, nwblock;
};

#endif // PAR_OPT_PROBLEM_H
