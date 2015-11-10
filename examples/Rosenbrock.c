#include "ParOpt.h"
#include "time.h"
/*
  The following is a simple implementation of a scalable Rosenbrock
  function with constraints that can be used to test the parallel
  optimizer. 
*/

class Rosenbrock : public ParOptProblem {
 public:
  Rosenbrock( MPI_Comm comm, int _nvars,
	      int _nwcon, int _nwstart, 
	      int _nw, int _nwskip ): 
  ParOptProblem(comm, _nvars, 2,
		_nwcon, 1){
    nwcon = _nwcon;
    nwstart = _nwstart;
    nw = _nw;
    nwskip = _nwskip;
    scale = 1.0;
  }

  // Set whether this is an inequality constraint
  int isSparseInequality(){ return 1; }
  int isDenseInequality(){ return 1; }
  int useLowerBounds(){ return 1; }
  int useUpperBounds(){ return 1; }

  // Get the variables/bounds
  void getVarsAndBounds( ParOptVec *xvec,
			 ParOptVec *lbvec, 
			 ParOptVec *ubvec ){
    ParOptScalar *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set the design variable bounds
    for ( int i = 0; i < nvars; i++ ){
      x[i] = -1.0 + i*0.01;
      lb[i] = -1.0;

      ub[i] = 1e20;
      if (i % 2 == 0){
	ub[i] = 0.5;
      }
    }
  }
  
  // Evaluate the objective and constraints
  // --------------------------------------
  int evalObjCon( ParOptVec *xvec, 
		  ParOptScalar *fobj, ParOptScalar *cons ){
    ParOptScalar obj = 0.0;
    ParOptScalar *x;
    xvec->getArray(&x);

    for ( int i = 0; i < nvars-1; i++ ){
      obj += ((1.0 - x[i])*(1.0 - x[i]) + 
	      100*(x[i+1] - x[i]*x[i])*(x[i+1] - x[i]*x[i]));
    }

    ParOptScalar con[2];
    con[0] = con[1] = 0.0;
    for ( int i = 0; i < nvars; i++ ){
      con[0] -= x[i]*x[i];
    }

    for ( int i = 0; i < nvars; i += 2 ){
      con[1] += x[i];
    }

    MPI_Allreduce(&obj, fobj, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
    MPI_Allreduce(con, cons, 2, PAROPT_MPI_TYPE, MPI_SUM, comm);

    int size; 
    MPI_Comm_size(comm, &size);
    cons[0] += 100*size*nvars;

    cons[0] *= scale;
    cons[1] *= scale;

    return 0;
  }
  
  // Evaluate the objective and constraint gradients
  // -----------------------------------------------
  int evalObjConGradient( ParOptVec *xvec,
			  ParOptVec *gvec, ParOptVec **Ac ){
    ParOptScalar *x, *g, *c;
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

  // Evaluate the product of the Hessian with the given vector
  // ---------------------------------------------------------
  int evalHvecProduct( ParOptVec *xvec,
		       ParOptScalar *z, ParOptVec *zwvec,
		       ParOptVec *pxvec, ParOptVec *hvec ){
    ParOptScalar *hvals;
    hvec->zeroEntries();
    hvec->getArray(&hvals);

    ParOptScalar *px, *x;
    xvec->getArray(&x);
    pxvec->getArray(&px);

    for ( int i = 0; i < nvars-1; i++ ){
      hvals[i] += (2.0*px[i] + 
		   200*(x[i+1] - x[i]*x[i])*(-2.0*px[i]) +
		   200*(px[i+1] - 2.0*x[i]*px[i])*(-2.0*x[i]));

      hvals[i+1] += 200*(px[i+1] - 2.0*x[i]*px[i]);
    }

    for ( int i = 0; i < nvars; i++ ){
      hvals[i] += 2.0*scale*z[0]*px[i];
    }
  }

  // Evaluate the sparse constraints
  // ------------------------
  void evalSparseCon( ParOptVec *x, ParOptVec *out ){
    ParOptScalar *xvals, *outvals; 
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
  void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
			  ParOptVec *px, ParOptVec *out ){
    ParOptScalar *pxvals, *outvals; 
    px->getArray(&pxvals);
    out->getArray(&outvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	outvals[i] -= alpha*pxvals[j];
      }
    }
  }

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
				   ParOptVec *pzw, ParOptVec *out ){
    ParOptScalar *outvals, *pzwvals;
    out->getArray(&outvals);
    pzw->getArray(&pzwvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	outvals[j] -= alpha*pzwvals[i];
      }
    }
  }

  // Add the inner product of the constraints to the matrix such 
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  // ------------------------------------------------------------
  void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
			      ParOptVec *cvec, ParOptScalar *A ){
    ParOptScalar *cvals;
    cvec->getArray(&cvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	A[i] += alpha*cvals[j];
      }
    }
  }

  int nwcon;
  int nwstart;
  int nw, nwskip;
  ParOptScalar scale;
};

int main( int argc, char* argv[] ){
  MPI_Init(&argc, &argv);

  // Allocate the Rosenbrock function
  int nvars = 100;
  int nwcon = 5, nw = 5;
  int nwstart = 1, nwskip = 1;  
  Rosenbrock * rosen = new Rosenbrock(MPI_COMM_WORLD, nvars-1, 
				      nwcon, nwstart, nw, nwskip);
  
  // Allocate the optimizer
  int max_lbfgs = 20;
  ParOpt * opt = new ParOpt(rosen, max_lbfgs);

  opt->setGMRESSusbspaceSize(30);
  opt->setNKSwitchTolerance(1e3);
  opt->setGMRESTolerances(1.0, 1e-30);
  opt->setUseHvecProduct(1);
  opt->setMajorIterStepCheck(20);
  opt->setMaxMajorIterations(1500);
  opt->checkGradients(1e-6);
  opt->setQNDiagonalFactor(1.0);
  
  double start = MPI_Wtime();
  opt->optimize();
  double diff = MPI_Wtime() - start;
  printf("Time taken: %f seconds \n", diff);

  delete rosen;
  delete opt;

  MPI_Finalize();
  return (0);
}

