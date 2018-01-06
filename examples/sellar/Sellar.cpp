#include "ParOpt.h"
#include "ParOptBlasLapack.h"
#include "time.h"

/*
  The following is a simple implementation of a Sellar function with
  constraints that can be used to test the parallel optimizer.
*/

class Sellar : public ParOptProblem {
 public:
  static const int nvars = 5;
  static const int ncon = 1;
  Sellar( MPI_Comm _comm ): ParOptProblem(_comm, nvars, ncon, 0, 0){}

  // Set whether this is an inequality constraint
  int isSparseInequality(){ return 0; }
  int isDenseInequality(){ return 1; }
  int useLowerBounds(){ return 1; }
  int useUpperBounds(){ return 1; }

  // Get the variables/bounds
  void getVarsAndBounds( ParOptVec *xvec,
			 ParOptVec *lbvec, 
			 ParOptVec *ubvec ){
    // declare design variable and bounds vector
    ParOptScalar *x, *lb, *ub;

    // store the memory addresses of the class variables
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set the initial design variableS
    x[0] = 5.0;
    x[1] = 2.0;
    x[2] = 1.0;
    x[3] = 0.0;
    x[4] = 0.0;
    
    // set lower and upper bounds to design variables
    lb[0] = -10.0; lb[1] = 0.0; lb[2]  = 0.0;  lb[3] = -1.0;  lb[4] = -1.0;
    ub[0] =  10.0; ub[1] = 10.0; ub[2] = 10.0; ub[3] = 3.16; ub[4] = 24.0; 
    
  }
  
  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *xvec, 
		  ParOptScalar *fobj, ParOptScalar *cons ){

    // declare local variables
    ParOptScalar *x;
    xvec->getArray(&x);

    // the objective function
    *fobj = x[2]*x[2] + x[1] + x[3] + exp(-x[4]);
    cons[0] = x[1] + x[2] - 1.0;

    return 0;
  }
  
  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *xvec,
			  ParOptVec *gvec, ParOptVec **Ac ){

    // define the local variables
    double *x, *g, *c;

    // get the local variables values
    xvec->getArray(&x);

    // derivative of the objective function wrt to the DV
    gvec->zeroEntries();
    gvec->getArray(&g);
    g[0] = 0.0;
    g[1] = 1.0;
    g[2] = 2.0*x[2];
    g[3] = 1.0;
    g[4] = -exp(-x[4]);

    // Derivative of the constraint
    Ac[0]->zeroEntries();
    Ac[0]->getArray(&g);
    g[1] = 1.0;
    g[2] = 1.0;
    
    return 0;
  }

  // Evaluate the product of the Hessian with the given vector
  // ---------------------------------------------------------
  int evalHvecProduct( ParOptVec *xvec,
		       double *z, ParOptVec *zwvec,
		       ParOptVec *pxvec, ParOptVec *hvec ){
    return 1;
  }

  // Evaluate the sparse constraints
  // -------------------------------
  void evalSparseCon( ParOptVec *x, ParOptVec *out ){}
  
  // Compute the Jacobian-vector product out = J(x)*px
  // --------------------------------------------------
  void addSparseJacobian( double alpha, ParOptVec *x,
			  ParOptVec *px, ParOptVec *out ){
  }

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose( double alpha, ParOptVec *x,
				   ParOptVec *pzw, ParOptVec *out ){}

  // Add the inner product of the constraints to the matrix such 
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  // ------------------------------------------------------------
  void addSparseInnerProduct( double alpha, ParOptVec *x,
			      ParOptVec *cvec, double *A ){}
};

int main( int argc, char* argv[] ){
  MPI_Init(&argc, &argv);

  // Allocate the Sellar function
  Sellar * sellar = new Sellar(MPI_COMM_SELF);
  
  // Allocate the optimizer
  int max_lbfgs = 20;
  ParOpt * opt = new ParOpt(sellar, max_lbfgs);

  opt->setMaxMajorIterations(100);
  opt->checkGradients(1e-6);
  
  double start = MPI_Wtime();
  opt->optimize();
  double diff = MPI_Wtime() - start;
  printf("Time taken: %f seconds \n", diff);

  delete sellar;
  delete opt;

  MPI_Finalize();
  return (0);
}

