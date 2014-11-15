#include "ParOpt.h"

/*
  The following is a simple implementation of a scalable Rosenbrock
  function with constraints that can be used to test the parallel
  optimizer. 
*/

class Rosenbrock : public ParOptProblem {
 public:
  Rosenbrock( MPI_Comm comm, int n ): 
  ParOptProblem(comm, n+1, 1){}
  
  void getVarsAndBounds( ParOptVec *xvec,
			 ParOptVec *lbvec, 
			 ParOptVec *ubvec ){
    double *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set the design variable bounds
    for ( int i = 0; i < nvars; i++ ){
      x[i] = -1.0;
      lb[i] = -2.0;
      ub[i] = 2.0;
    }
  }
  
  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *xvec, 
		  double *fobj, double *cons ){
    double obj = 0.0, con = 0.0;
    double *x;
    xvec->getArray(&x);

    for ( int i = 0; i < nvars-1; i++ ){
      obj += ((1.0 - x[i])*(1.0 - x[i]) + 
	      100*(x[i+1] - x[i]*x[i])*(x[i+1] - x[i]*x[i]));
    }

    for ( int i = 0; i < nvars; i++ ){
      con += x[i];
    }

    *fobj = obj;
    cons[0] = con;

    return 0;
  }
  
  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *xvec,
			  ParOptVec *gvec, ParOptVec **Ac ){
    double *x, *g, *c;
    xvec->getArray(&x);
    gvec->getArray(&g);
    Ac[0]->getArray(&c);

    gvec->zeroEntries();
    
    for ( int i = 0; i < nvars-1; i++ ){
      g[i] += (-2.0*(1.0 - x[i])*(1.0 - x[i]) + 
	       200*(x[i+1] - x[i]*x[i])*(-2.0*x[i]));
      g[i+1] += 200*(x[i+1] - x[i]*x[i]);
    }

    for ( int i = 0; i < nvars; i++ ){
      c[i] = 1.0;
    }

    return 0;
  }
};

int main( int argc, char* argv[] ){
  MPI_Init(&argc, &argv);

  // Allocate the Rosenbrock function
  int nvars = 100;
  Rosenbrock * rosen = new Rosenbrock(MPI_COMM_WORLD, nvars-1);
  
  // Allocate the optimizer
  int max_lbfgs = 20;
  ParOpt * opt = new ParOpt(rosen, max_lbfgs);
  opt->optimize();

  delete rosen;
  delete opt;

  MPI_Finalize();
  return (0);
}

