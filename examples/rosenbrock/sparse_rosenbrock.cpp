#include "ParOptOptimizer.h"

/*
  The following is a simple implementation of a scalable Rosenbrock
  function with constraints that can be used to test the parallel
  optimizer.
*/

class SparseRosenbrock : public ParOptSparseProblem {
 public:
  SparseRosenbrock(MPI_Comm comm, int _nvars) : ParOptSparseProblem(comm) {
    // Set the base class problem sizes
    setProblemSizes(_nvars, 2, _nvars - 1);

    setNumInequalities(2, _nvars - 1);

    // Set the non-zero pattern for the inequality constraints
    int *rowp = new int[nwcon + 1];
    int *cols = new int[2 * nwcon];

    rowp[0] = 0;
    for (int i = 0; i < nwcon; i++) {
      rowp[i + 1] = 2 * (i + 1);
      cols[2 * i] = i;
      cols[2 * i + 1] = i + 1;
    }
    setSparseJacobianData(rowp, cols);
    delete[] rowp;
    delete[] cols;
  }

  int isSparseInequality() { return 1; }

  //! Get the variables/bounds
  void getVarsAndBounds(ParOptVec *xvec, ParOptVec *lbvec, ParOptVec *ubvec) {
    ParOptScalar *x, *lb, *ub;
    xvec->getArray(&x);
    lbvec->getArray(&lb);
    ubvec->getArray(&ub);

    // Set the design variable bounds
    for (int i = 0; i < nvars; i++) {
      x[i] = 0.0;
      lb[i] = -2.0;
      ub[i] = 2.0;
    }
  }

  //! Evaluate the objective and constraints
  int evalSparseObjCon(ParOptVec *xvec, ParOptScalar *fobj, ParOptScalar *cons,
                       ParOptVec *sparse) {
    ParOptScalar obj = 0.0;
    ParOptScalar *x;
    xvec->getArray(&x);

    for (int i = 0; i < nvars - 1; i++) {
      obj += ((1.0 - x[i]) * (1.0 - x[i]) +
              100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]));
    }

    ParOptScalar con[2];
    con[0] = con[1] = 0.0;
    for (int i = 0; i < nvars; i++) {
      con[0] -= x[i] * x[i];
    }

    for (int i = 0; i < nvars; i += 2) {
      con[1] += x[i];
    }

    MPI_Allreduce(&obj, fobj, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
    MPI_Allreduce(con, cons, 2, PAROPT_MPI_TYPE, MPI_SUM, comm);

    cons[0] += 0.25;
    cons[1] += 10.0;

    // Evaluate the sparse constraints
    ParOptScalar *c;
    sparse->getArray(&c);
    for (int i = 0; i < nwcon; i++) {
      c[i] = 1.0 - x[i] * x[i] - x[i + 1] * x[i + 1];
    }

    return 0;
  }

  //! Evaluate the objective and constraint gradients
  int evalObjConSparseGradient(ParOptVec *xvec, ParOptVec *gvec, ParOptVec **Ac,
                               ParOptScalar *data) {
    ParOptScalar *x, *g, *c;
    xvec->getArray(&x);
    gvec->getArray(&g);
    gvec->zeroEntries();

    for (int i = 0; i < nvars - 1; i++) {
      g[i] += (-2.0 * (1.0 - x[i]) +
               200.0 * (x[i + 1] - x[i] * x[i]) * (-2.0 * x[i]));
      g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i]);
    }

    Ac[0]->getArray(&c);
    for (int i = 0; i < nvars; i++) {
      c[i] = -2.0 * x[i];
    }

    Ac[1]->getArray(&c);
    for (int i = 0; i < nvars; i += 2) {
      c[i] = 1.0;
    }

    // Compute the sparse constraint Jacobian
    for (int i = 0; i < nwcon; i++) {
      data[2 * i] = -2.0 * x[i];
      data[2 * i + 1] = -2.0 * x[i + 1];
    }

    return 0;
  }
};

int main(int argc, char *argv[]) {
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
  for (int k = 0; k < argc; k++) {
    if (sscanf(argv[k], "prefix=%s", buff) == 1) {
      prefix = buff;
    }
    if (sscanf(argv[k], "nvars=%d", &nvars) == 1) {
      if (nvars < 100) {
        nvars = 100;
      }
    }
  }

  if (mpi_rank == 0) {
    printf("prefix = %s\n", prefix);
    fflush(stdout);
  }

  // Allocate the Rosenbrock function
  SparseRosenbrock *rosen = new SparseRosenbrock(comm, nvars);
  rosen->incref();

  // Create the options class, and create default values
  ParOptOptions *options = new ParOptOptions();
  ParOptOptimizer::addDefaultOptions(options);

  options->setOption("algorithm", "tr");
  options->setOption("barrier_strategy", "mehrotra");
  options->setOption("output_level", 1);
  options->setOption("qn_type", "bfgs");
  options->setOption("qn_subspace_size", 10);
  options->setOption("abs_res_tol", 1e-6);
  options->setOption("output_file", "paropt.out");
  options->setOption("tr_output_file", "paropt.tr");
  options->setOption("mma_output_file", "paropt.mma");

  // options->setOption("use_line_search", 0);
  // options->setOption("tr_steering_barrier_strategy", "monotone");

  ParOptOptimizer *opt = new ParOptOptimizer(rosen, options);
  opt->incref();

  // Set the checkpoint file
  double start = MPI_Wtime();
  if (prefix) {
    char output[512];
    snprintf(output, sizeof(output), "%s/rosenbrock_output.bin", prefix);
    options->setOption("ip_checkpoint_file", output);
  }
  opt->optimize();
  double diff = MPI_Wtime() - start;

  if (mpi_rank == 0) {
    printf("ParOpt time: %f seconds \n", diff);
  }

  opt->decref();
  rosen->decref();

  MPI_Finalize();
  return (0);
}
