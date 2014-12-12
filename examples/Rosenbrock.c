#include "ParOpt.h"

/*
  The following is a simple implementation of a scalable Rosenbrock
  function with constraints that can be used to test the parallel
  optimizer. 
*/

class Rosenbrock : public ParOptProblem {
 public:
  Rosenbrock( MPI_Comm comm, int n ): 
  ParOptProblem(comm, n+1, 2){
    scale = 1.0;
  }
  
  void getVarsAndBounds( ParOptVec *xvec,
			 ParOptVec *lbvec, 
			 ParOptVec *ubvec ){
    double *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set the design variable bounds
    for ( int i = 0; i < nvars; i++ ){
      x[i] = -1.0 + i*0.01;
      lb[i] = -2.0;
      ub[i] = 4.0;
    }
  }
  
  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *xvec, 
		  double *fobj, double *cons ){
    double obj = 0.0;
    double *x;
    xvec->getArray(&x);

    for ( int i = 0; i < nvars-1; i++ ){
      obj += ((1.0 - x[i])*(1.0 - x[i]) + 
	      100*(x[i+1] - x[i]*x[i])*(x[i+1] - x[i]*x[i]));
    }

    double con[2];
    con[0] = con[1] = 0.0;
    for ( int i = 0; i < nvars; i++ ){
      con[0] -= x[i]*x[i];
    }

    for ( int i = 0; i < nvars; i += 2 ){
      con[1] += x[i];
    }

    MPI_Allreduce(&obj, fobj, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(con, cons, 2, MPI_DOUBLE, MPI_SUM, comm);

    int size; 
    MPI_Comm_size(comm, &size);
    cons[0] += 20*size*nvars;

    cons[0] *= scale;
    cons[1] *= scale;

    return 0;
  }
  
  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *xvec,
			  ParOptVec *gvec, ParOptVec **Ac ){
    double *x, *g, *c;
    xvec->getArray(&x);
    gvec->getArray(&g);
    gvec->zeroEntries();

    for ( int i = 0; i < nvars-1; i++ ){
      g[i] += (-2.0*(1.0 - x[i]) + 
	       200*(x[i+1] - x[i]*x[i])*(-2.0*x[i]));
      g[i+1] += 200*(x[i+1] - x[i]*x[i]);
    }

    Ac[0]->getArray(&c);
    for ( int i = 0; i < nvars; i++ ){
      c[i] = -2.0*scale*x[i];
    }

    Ac[1]->getArray(&c);
    for ( int i = 0; i < nvars; i += 2 ){
      c[i] = scale;
    }

    return 0;
  }

  double scale;
};

int main( int argc, char* argv[] ){
  MPI_Init(&argc, &argv);

  // Allocate the Rosenbrock function
  int nvars = 100;
  Rosenbrock * rosen = new Rosenbrock(MPI_COMM_WORLD, nvars-1);
  
  // Allocate the optimizer
  int max_lbfgs = 10;
  int nwcon = 0;
  int nw = 5;
  int nwstart = 1;
  int nwskip = 1;
  ParOpt * opt = new ParOpt(rosen, nwcon, nwstart, nw, nwskip, max_lbfgs);
  
  // opt->checkGradients(1e-6);
  // opt->setMajorIterStepCheck(29);
  opt->optimize();
  opt->setSequentialLinearMethod(1);

  delete rosen;
  delete opt;

  MPI_Finalize();
  return (0);
}

