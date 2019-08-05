Rosenbrock function
===================

This example extends the previous Sellar problem by considering a distributed design vector, separable sparse constraints, and Hessian-vector products.

.. code-block:: c++

      int evalHvecProduct( ParOptVec *xvec,
                           ParOptScalar *z, ParOptVec *zwvec,
                           ParOptVec *pxvec, ParOptVec *hvec );

The Hessian-vector products can be used to accelerate the convergence of the optimizer.
They are generally used once the optimization problem has converged to a point that is closer to the optimum.
These products are implemented by the user in the ParOptProblem class, using the template provided above.
The Lagrangian function is defined as

.. math::

    \mathcal{L} \triangleq f(x) - z^{T} c(x) - z_{w}^{T} c_{w}(x)

where :math:`\mathcal{L}` is the Lagrangian, :math:`f(x), c(x), c_{w}(x)` are the objective, dense constraints and sparse separable constraints, respectively, and :math:`z, z_{w}` are multipliers associated with the dense and sparse constraints.
The Hessian-vector product is then computed as

.. math::

    h = \nabla^{2} \mathcal{L}(x, z, z_{w}) p_{x}

The interface to the sparse separable constraint code consists of four functions.
These functions consist of the following:

.. code-block:: c++

      void evalSparseCon( ParOptVec *x, ParOptVec *out );
      void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                              ParOptVec *px, ParOptVec *out );
      void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                       ParOptVec *pzw, ParOptVec *out );
      void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *cvec, ParOptScalar *A );

These member functions provide the following mathematical operations:

.. math::

    \begin{align}
        \mathrm{out} & \leftarrow c_{w} \leftarrow c_{w}(x) \\
        \mathrm{out} & \leftarrow \alpha A_{w}(x) p_{x} \\
        \mathrm{out} & \leftarrow \alpha A_{w}(x)^{T} p_{z_{w}} \\
        \mathrm{out} & \leftarrow \alpha A_{w} C A_{w}(x)^{T}  \\
    \end{align}

Here :math:`A_{w}(x) = \nabla c_{w}(x)$` and :math:`C` is a diagonal matrix.

.. code-block:: c++

    #include "ParOpt.h"

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
      ParOptProblem(comm, _nvars, 2, _nwcon, 1){
        nwcon = _nwcon;
        nwstart = _nwstart;
        nw = _nw;
        nwskip = _nwskip;
        scale = 1.0;
      }

      //! Get the variables/bounds
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
      
      //! Evaluate the objective and constraints
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
      
      //! Evaluate the objective and constraint gradients
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

      //! Evaluate the product of the Hessian with the given vector
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

      //! Evaluate the sparse constraints
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
      
      //! Compute the Jacobian-vector product out = J(x)*px
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

      //! Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
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

      //! Add the inner product of the constraints to the matrix such 
      //! that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
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
      int nwcon = 5, nw = 5;
      int nwstart = 1, nwskip = 1;  
      Rosenbrock *rosen = new Rosenbrock(comm, nvars-1,
                                        nwcon, nwstart, nw, nwskip);
      rosen->incref();

      // Allocate the optimizer
      int max_lbfgs = 20;
      ParOpt *opt = new ParOpt(rosen, max_lbfgs);
      opt->incref();

      opt->setMaxMajorIterations(1500);
      opt->setBarrierStrategy(PAROPT_MEHROTRA);
      opt->setOutputFrequency(1);
      opt->setOutputFile("paropt.out");
      
      // Set the checkpoint file
      double start = MPI_Wtime();
      if (prefix){
        char output[512];
        sprintf(output, "%s/rosenbrock_output.bin", prefix);
        opt->optimize(output);
      }
      else {
        opt->optimize();
      }
      double diff = MPI_Wtime() - start;

      if (mpi_rank == 0){
        printf("ParOpt time: %f seconds \n", diff);
      }

      opt->decref(); 
      rosen->decref();

      MPI_Finalize();
      return (0);
    }
