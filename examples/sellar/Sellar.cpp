#include "ParOpt.h"

/*
  The following is a simple implementation of a Sellar function with
  constraints that can be used to test the parallel optimizer.
*/
class Sellar : public ParOptProblem {
 public:
  static const int nvars = 4;
  static const int ncon = 1;
  Sellar( MPI_Comm _comm ): ParOptProblem(_comm, nvars, ncon, 0, 0){}

  //! Get the variables/bounds
  void getVarsAndBounds( ParOptVec *xvec,
                         ParOptVec *lbvec,
                         ParOptVec *ubvec ){
    // declare design variable and bounds vector
    ParOptScalar *x, *lb, *ub;

    // store the memory addresses of the class variables
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set the initial design variables
    x[0] = 2.0;
    x[1] = 1.0;
    x[2] = 0.0;
    x[3] = 0.0;
    
    // set lower and upper bounds to design variables
    lb[0] = 0.0;  lb[1]  = 0.0; lb[2] = -1.0; lb[3] = -1.0;
    ub[0] = 10.0; ub[1] = 10.0; ub[2] = 3.16; ub[3] = 24.0; 
  }
  
  //! Evaluate the objective and constraints
  int evalObjCon( ParOptVec *xvec,
                  ParOptScalar *fobj, ParOptScalar *cons ){

    // declare local variables
    ParOptScalar *x;
    xvec->getArray(&x);

    // the objective function
    *fobj = x[1]*x[1] + x[0] + x[2] + exp(-x[3]);
    cons[0] = x[0] + x[1] - 1.0;

    return 0;
  }
  
  //! Evaluate the objective and constraint gradients
  int evalObjConGradient( ParOptVec *xvec, ParOptVec *gvec, ParOptVec **Ac ){

    // define the local variables
    double *x, *g;

    // get the local variables values
    xvec->getArray(&x);

    // derivative of the objective function wrt to the DV
    gvec->zeroEntries();
    gvec->getArray(&g);
    g[0] = 1.0;
    g[1] = 2.0*x[1];
    g[2] = 1.0;
    g[3] = -exp(-x[3]);

    // Derivative of the constraint
    Ac[0]->zeroEntries();
    Ac[0]->getArray(&g);
    g[0] = 1.0;
    g[1] = 1.0;
    
    return 0;
  }
};

int main( int argc, char* argv[] ){
  MPI_Init(&argc, &argv);

  // Allocate the Sellar function
  Sellar *sellar = new Sellar(MPI_COMM_SELF);
  sellar->incref();
  
  // Allocate the optimizer
  int max_lbfgs = 20;
  ParOpt *opt = new ParOpt(sellar, max_lbfgs);

  opt->setMaxMajorIterations(100);
  opt->checkGradients(1e-6);
  
  double start = MPI_Wtime();
  opt->optimize();
  double diff = MPI_Wtime() - start;
  printf("Time taken: %f seconds \n", diff);

  sellar->decref();
  opt->decref();

  MPI_Finalize();
  return (0);
}
