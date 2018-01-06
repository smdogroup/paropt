#include "ParOpt.h"
#include "ParOptMMA.h"
#include "time.h"

/*
  The following is a simple implementation of a scalable Rosenbrock
  function with constraints that can be used to test the parallel
  optimizer. 
*/

ParOptScalar min2( ParOptScalar a, ParOptScalar b ){
  if (a < b){
    return a;
  }
  return b;
}

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
      x[i] = -1.0;
      lb[i] = -2.0;
      ub[i] = 1.0;
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

    cons[0] += 0.25;
    cons[1] += 10.0;
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

    return 0;
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

  // Set the MPI communicator
  MPI_Comm comm = MPI_COMM_WORLD;

  // Get the rank
  int mpi_rank = 0;
  MPI_Comm_rank(comm, &mpi_rank);

  // Get the prefix from the input arguments
  int nvars = 100;
  const char *prefix = NULL;
  char buff[512];
  for ( int k = 0; k < argc; k++ ){
    if (sscanf(argv[k], "prefix=%s", buff) == 1){
      prefix = buff;
    }
    if (sscanf(argv[k], "nvars=%d", &nvars) == 1){
      if (nvars < 100){
        nvars = 100;
      }
    }
  }

  if (mpi_rank == 0){
    printf("prefix = %s\n", prefix);
    fflush(stdout);
  }

  // Allocate the Rosenbrock function
  int nwcon = 0, nw = 5;
  int nwstart = 1, nwskip = 1;  
  Rosenbrock *rosen = new Rosenbrock(comm, nvars-1,
                                     nwcon, nwstart, nw, nwskip);
  rosen->incref();

  // Create the MMA problem
  int use_mma = 0;
  ParOptMMA *mma = new ParOptMMA(rosen, use_mma);
  mma->incref();

  // Allocate the optimizer
  int max_lbfgs = 20;
  ParOpt *opt = new ParOpt(mma, max_lbfgs);
  opt->incref();

  opt->setGMRESSubspaceSize(30);
  opt->setNKSwitchTolerance(1e3);
  opt->setGMRESTolerances(1.0, 1e-30);
  opt->setUseHvecProduct(1);
  opt->setMaxMajorIterations(1500);
  opt->setOutputFrequency(1);

  mma->initializeSubProblem(NULL);
  opt->checkGradients(1e-6);
  
  for ( int i = 0; i < 5; i++ ){
    char filename[512];
    sprintf(filename, "paropt%03d.out", i);
    opt->setOutputFile(filename);  
    opt->optimize();

    ParOptVec *xvec;
    opt->getOptimizedPoint(&xvec, NULL, NULL, NULL, NULL);
    mma->initializeSubProblem(xvec);
    opt->resetDesignAndBounds();
  }

  opt->decref(); 
  mma->decref();
  rosen->decref();

  MPI_Finalize();
  return (0);
}

