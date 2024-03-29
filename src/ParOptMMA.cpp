#include "ParOptMMA.h"

#include <string.h>

#include "ParOptBlasLapack.h"
#include "ParOptComplexStep.h"

// Helper functions
inline ParOptScalar min2(ParOptScalar a, ParOptScalar b) {
  if (ParOptRealPart(a) < ParOptRealPart(b)) {
    return a;
  } else {
    return b;
  }
}

inline ParOptScalar max2(ParOptScalar a, ParOptScalar b) {
  if (ParOptRealPart(a) > ParOptRealPart(b)) {
    return a;
  } else {
    return b;
  }
}

/*
  Create the ParOptMMA object
*/
ParOptMMA::ParOptMMA(ParOptProblem *_prob, ParOptOptions *_options)
    : ParOptProblem(_prob->getMPIComm()) {
  // Set the problem instance
  prob = _prob;
  prob->incref();

  // Store the input options
  options = _options;
  options->incref();

  // Pull out the communicator
  comm = prob->getMPIComm();

  // Set the file pointer to NULL
  first_print = 1;
  fp = NULL;

  // By default, use the real MMA. But this will be set from
  // the options object during the optimization.
  use_true_mma = 1;

  // Set the default to stdout
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    fp = stdout;
  }

  // Over-ride the settings argument
  const char *mma_output_file = options->getStringOption("mma_output_file");
  setOutputFile(mma_output_file);

  // Set the problem sizes
  int _nwcon;
  prob->getProblemSizes(&n, &m, &_nwcon);
  setProblemSizes(n, m, _nwcon);

  // Set the number of inequalities
  int nineq, nwineq;
  prob->getNumInequalities(&nineq, &nwineq);
  setNumInequalities(nineq, nwineq);

  // Set the iteration counter
  mma_iter = 0;
  subproblem_iter = 0;

  // Initialize the data
  initialize();
}

/*
  Deallocate all of the internal data
*/
ParOptMMA::~ParOptMMA() {
  if (fp && fp != stdout) {
    fclose(fp);
  }
  prob->decref();

  xvec->decref();
  x1vec->decref();
  x2vec->decref();
  lbvec->decref();
  ubvec->decref();
  delete[] cons;

  gvec->decref();
  for (int i = 0; i < m; i++) {
    Avecs[i]->decref();
  }
  delete[] Avecs;

  Lvec->decref();
  Uvec->decref();
  alphavec->decref();
  betavec->decref();
  p0vec->decref();
  q0vec->decref();

  if (use_true_mma) {
    for (int i = 0; i < m; i++) {
      pivecs[i]->decref();
      qivecs[i]->decref();
    }
    delete[] qivecs;
    delete[] pivecs;
    delete[] b;
  } else {
    if (cwvec) {
      cwvec->decref();
    }
  }

  delete[] z;
  zwvec->decref();
  zlvec->decref();
  zuvec->decref();
  rvec->decref();
}

/*
  Allocate all of the data
*/
void ParOptMMA::initialize() {
  // Incref the reference counts to the design vectors
  xvec = prob->createDesignVec();
  xvec->incref();
  x1vec = prob->createDesignVec();
  x1vec->incref();
  x2vec = prob->createDesignVec();
  x2vec->incref();

  // Create the design variable bounds
  lbvec = prob->createDesignVec();
  lbvec->incref();
  ubvec = prob->createDesignVec();
  ubvec->incref();

  // Allocate the constraint array
  fobj = 0.0;
  cons = new ParOptScalar[m];
  memset(cons, 0, m * sizeof(ParOptScalar));

  // Allocate space for the problem gradients
  gvec = prob->createDesignVec();
  gvec->incref();
  Avecs = new ParOptVec *[m];
  for (int i = 0; i < m; i++) {
    Avecs[i] = prob->createDesignVec();
    Avecs[i]->incref();
  }

  // Create the move limit/asymptote vectors
  Lvec = prob->createDesignVec();
  Lvec->incref();
  Uvec = prob->createDesignVec();
  Uvec->incref();
  alphavec = prob->createDesignVec();
  alphavec->incref();
  betavec = prob->createDesignVec();
  betavec->incref();
  alphavec->set(0.0);
  betavec->set(1.0);

  // Create the coefficient vectors
  p0vec = prob->createDesignVec();
  p0vec->incref();
  q0vec = prob->createDesignVec();
  q0vec->incref();

  // Set the sparse constraint vector to NULL
  cwvec = NULL;

  if (use_true_mma) {
    pivecs = new ParOptVec *[m];
    qivecs = new ParOptVec *[m];
    for (int i = 0; i < m; i++) {
      pivecs[i] = prob->createDesignVec();
      pivecs[i]->incref();
      qivecs[i] = prob->createDesignVec();
      qivecs[i]->incref();
    }

    b = new ParOptScalar[m];
    memset(b, 0, m * sizeof(ParOptScalar));
  } else {
    pivecs = NULL;
    qivecs = NULL;
    b = NULL;
  }

  if (nwcon > 0) {
    cwvec = prob->createConstraintVec();
    if (cwvec) {
      cwvec->incref();
    }
  }

  // Get the design variables and bounds
  prob->getVarsAndBounds(xvec, lbvec, ubvec);

  // Set artificial bounds if none are provided
  if (!prob->useUpperBounds()) {
    ubvec->set(10.0);
  }
  if (!prob->useLowerBounds()) {
    lbvec->set(-9.0);
  }

  // Set the initial multipliers/values to zero
  z = new ParOptScalar[m];
  memset(z, 0, m * sizeof(ParOptScalar));
  zwvec = prob->createConstraintVec();
  zwvec->incref();

  // Set the bound multipliers
  zlvec = prob->createDesignVec();
  zuvec = prob->createDesignVec();
  zlvec->incref();
  zuvec->incref();

  // Create a sparse constraint vector
  rvec = prob->createDesignVec();
  rvec->incref();
}

void ParOptMMA::addDefaultOptions(ParOptOptions *options) {
  // Set default parameters
  options->addStringOption("mma_output_file", "paropt.mma",
                           "Ouput file name for MMA");

  options->addIntOption("mma_max_iterations", 200, 0, 1000000,
                        "Maximum number of iterations");

  options->addFloatOption("mma_l1_tol", 1e-6, 0.0, 1e20,
                          "l1 tolerance for the optimality tolerance");

  options->addFloatOption("mma_linfty_tol", 1e-6, 0.0, 1e20,
                          "l-infinity tolerance for the optimality tolerance");

  options->addFloatOption("mma_infeas_tol", 1e-5, 0.0, 1e20,
                          "Infeasibility tolerance ");

  options->addIntOption(
      "output_level", 0, 0, 1000000,
      "Output level indicating how verbose the output should be");

  options->addBoolOption(
      "mma_use_constraint_linearization", 0,
      "Use a linearization of the constraints in the MMA subproblem");

  options->addFloatOption("mma_asymptote_contract", 0.7, 0.0, 1.0,
                          "Contraction factor applied to the asymptotes");

  options->addFloatOption("mma_asymptote_relax", 1.2, 1.0, 1e20,
                          "Expansion factor applied to the asymptotes");

  options->addFloatOption("mma_init_asymptote_offset", 0.5, 0.0, 1.0,
                          "Initial asymptote offset from the variable bounds");

  options->addFloatOption("mma_min_asymptote_offset", 0.01, 0.0, 1e20,
                          "Minimum asymptote offset from the variable bounds");

  options->addFloatOption("mma_max_asymptote_offset", 10.0, 0.0, 1e20,
                          "Maximum asymptote offset from the variable bounds");

  options->addFloatOption(
      "mma_bound_relax", 0.0, 0.0, 1e20,
      "Relaxation bound for computing the error in the KKT conditions");

  options->addFloatOption(
      "mma_eps_regularization", 1e-5, 0.0, 1e20,
      "Regularization term applied in the MMA approximation");

  options->addFloatOption(
      "mma_delta_regularization", 1e-3, 0.0, 1e20,
      "Regularization term applied in the MMA approximation");

  options->addFloatOption(
      "mma_move_limit", 0.2, 0.0, 1e20,
      "Move limit for design variables to prevent oscillation");
}

ParOptOptions *ParOptMMA::getOptions() { return options; }

/*
  Set the output file (only on the root proc)
*/
void ParOptMMA::setOutputFile(const char *filename) {
  if (fp && fp != stdout) {
    fclose(fp);
  }
  fp = NULL;

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (filename && rank == 0) {
    fp = fopen(filename, "w");
  }
}

/*
  Write the parameters to the output file
*/
void ParOptMMA::printOptionsSummary(FILE *file) {
  const int output_level = options->getIntOption("output_level");
  options->printSummary(file, output_level);
}

void ParOptMMA::optimize(ParOptInteriorPoint *optimizer) {
  if (optimizer->getOptProblem() != this) {
    fprintf(stderr,
            "ParOptMMA: The optimizer must be associated with "
            "the MMA object\n");
    return;
  }

  // Get the stopping criteria data
  const int max_iterations = options->getIntOption("mma_max_iterations");
  const double infeas_tol = options->getFloatOption("mma_infeas_tol");
  const double l1_tol = options->getFloatOption("mma_l1_tol");
  const double linfty_tol = options->getFloatOption("mma_linfty_tol");

  // Set what type of sub-problem we're going to use. Check if the flag
  // has been set to use a linearization of the constraints. If so, then
  // we're not using the "use_true_mma" option.
  int use_linearized =
      options->getBoolOption("mma_use_constraint_linearization");

  // Set the member controlling the use of the MMA constraint approximation
  use_true_mma = !use_linearized;

  // Set the interior point optimizer data to be compatible
  ParOptOptions *options = optimizer->getOptions();
  options->setOption("use_diag_hessian", 1);
  options->setOption("use_line_search", 0);

  initializeSubProblem(xvec);
  optimizer->resetDesignAndBounds();

  for (int i = 0; i < max_iterations; i++) {
    // Optimize the sub-problem
    optimizer->optimize();

    // Get the optimized point
    ParOptVec *x, *zw, *zl, *zu;
    ParOptScalar *z;
    optimizer->getOptimizedPoint(&x, &z, &zw, &zl, &zu);

    // Set the multipliers
    setMultipliers(z, zw, zl, zu);

    // Initialize the subproblem about the new point
    initializeSubProblem(x);

    // Reset the variable bounds
    optimizer->resetDesignAndBounds();

    // Compute the KKT error;
    double infeas, l1, linfty;
    computeKKTError(&infeas, &l1, &linfty);

    // Check for convergence of the trust region problem
    if (infeas < infeas_tol) {
      if (l1 < l1_tol || linfty < linfty_tol) {
        // Success!
        break;
      }
    }
  }
}

/*
  Set the new values of the multipliers
*/
void ParOptMMA::setMultipliers(ParOptScalar *_z, ParOptVec *_zw, ParOptVec *_zl,
                               ParOptVec *_zu) {
  // Copy over the values of the multipliers
  memcpy(z, _z, m * sizeof(ParOptScalar));

  // Copy over the values of the constraint multipliers
  if (_zw) {
    zwvec->copyValues(_zw);
  }
  if (_zl) {
    zlvec->copyValues(_zl);
  }
  if (_zu) {
    zuvec->copyValues(_zu);
  }
}

/*
  Compute the KKT error based on the current values of the multipliers
  set in ParOptMMA. If you do not update the multipliers, you will not
  get the correct KKT error.
*/
void ParOptMMA::computeKKTError(double *l1, double *linfty, double *infeas) {
  const double mma_bound_relax = options->getFloatOption("mma_bound_relax");

  // Get the lower/upper bounds for the variables
  ParOptScalar *lb, *ub;
  lbvec->getArray(&lb);
  ubvec->getArray(&ub);

  // Get the current values of the design variables
  ParOptScalar *x;
  xvec->getArray(&x);

  // Compute the KKT residual r = g - A^{T}*z
  rvec->copyValues(gvec);
  for (int i = 0; i < m; i++) {
    rvec->axpy(-z[i], Avecs[i]);
  }

  // If zw exists, compute r = r - Aw^{T}*zw
  if (nwcon > 0) {
    prob->addSparseJacobianTranspose(-1.0, xvec, zwvec, rvec);
  }

  // Set the infinity norms
  double l1_norm = 0.0;
  double infty_norm = 0.0;

  // Get the vector of values
  ParOptScalar *r;
  rvec->getArray(&r);

  if (mma_bound_relax <= 0.0) {
    // Add r = r - zl + zu
    rvec->axpy(-1.0, zlvec);
    rvec->axpy(1.0, zuvec);

    for (int j = 0; j < n; j++) {
      // Find the contribution to the l1/infinity norms
      double t = fabs(ParOptRealPart(r[j]));
      l1_norm += t;
      if (t >= infty_norm) {
        infty_norm = t;
      }
    }
  } else {
    for (int j = 0; j < n; j++) {
      double w = ParOptRealPart(r[j]);

      // Check if we're on the lower bound
      if ((ParOptRealPart(x[j]) <= ParOptRealPart(lb[j]) + mma_bound_relax) &&
          w > 0.0) {
        w = 0.0;
      }

      // Check if we're on the upper bound
      if ((ParOptRealPart(x[j]) >= ParOptRealPart(ub[j]) - mma_bound_relax) &&
          w < 0.0) {
        w = 0.0;
      }

      // Add the contribution to the l1/infinity norms
      double t = fabs(w);
      l1_norm += t;
      if (t >= infty_norm) {
        infty_norm = t;
      }
    }
  }

  // Measure the infeasibility using the l1 norm
  *infeas = 0.0;
  for (int i = 0; i < m; i++) {
    *infeas += fabs(ParOptRealPart(min2(0.0, cons[i])));
  }

  // All-reduce the norms across all processors
  MPI_Allreduce(&l1_norm, l1, 1, MPI_DOUBLE, MPI_SUM, comm);
  MPI_Allreduce(&infty_norm, linfty, 1, MPI_DOUBLE, MPI_MAX, comm);
}

/*
  Get the optimized point
*/
void ParOptMMA::getOptimizedPoint(ParOptVec **_x) { *_x = xvec; }

/*
  Get the asymptotes themselves
*/
void ParOptMMA::getAsymptotes(ParOptVec **_L, ParOptVec **_U) {
  if (_L) {
    *_L = Lvec;
  }
  if (_U) {
    *_U = Uvec;
  }
}

/*
  Get the design history
*/
void ParOptMMA::getDesignHistory(ParOptVec **_x1, ParOptVec **_x2) {
  if (_x1) {
    *_x1 = x1vec;
  }
  if (_x2) {
    *_x2 = x2vec;
  }
}

/*
  Update and initialize data for the convex sub-problem that is solved
  at each iteration. This must be called before solving the dual
  problem.

  This code updates the asymptotes, sets the move limits and forms the
  approximations used in the MMA code.
*/
int ParOptMMA::initializeSubProblem(ParOptVec *xv) {
  // Extract data used to determine the asymptote behavior
  const double init_asymptote_offset =
      options->getFloatOption("mma_init_asymptote_offset");
  const double asymptote_contract =
      options->getFloatOption("mma_asymptote_contract");
  const double asymptote_relax = options->getFloatOption("mma_asymptote_relax");
  const double max_asymptote_offset =
      options->getFloatOption("mma_max_asymptote_offset");
  const double min_asymptote_offset =
      options->getFloatOption("mma_min_asymptote_offset");

  // Extract parameters used in the computation of the objective/constraint
  // approximations
  const double eps = options->getFloatOption("mma_eps_regularization");
  const double delta = options->getFloatOption("mma_delta_regularization");

  // Get the output level
  const int output_level = options->getIntOption("output_level");

  // Move limit for the design variables
  const double movlim = options->getFloatOption("mma_move_limit");

  x2vec->copyValues(x1vec);
  x1vec->copyValues(xvec);
  if (xv && xv != xvec) {
    xvec->copyValues(xv);
  }

  // Evaluate the objective/constraint gradients
  int fail_obj = prob->evalObjCon(xvec, &fobj, cons);
  if (fail_obj) {
    fprintf(stderr, "ParOptMMA: Objective evaluation failed\n");
    return fail_obj;
  }

  int fail_grad = prob->evalObjConGradient(xvec, gvec, Avecs);
  if (fail_grad) {
    fprintf(stderr, "ParOptMMA: Gradient evaluation failed\n");
    return fail_grad;
  }

  // Evaluate the sparse constraints
  if (nwcon > 0) {
    prob->evalSparseCon(xvec, cwvec);
  }

  // Compute the KKT error, and print it out to a file
  if (output_level >= 0) {
    double l1, linfty, infeas;
    computeKKTError(&l1, &linfty, &infeas);

    if (fp) {
      double l1_lambda = 0.0;
      for (int i = 0; i < m; i++) {
        l1_lambda += fabs(ParOptRealPart(z[i]));
      }

      if (first_print) {
        printOptionsSummary(fp);
      }
      if (first_print || mma_iter % 10 == 0) {
        fprintf(fp, "\n%5s %8s %15s %9s %9s %9s %9s\n", "MMA", "sub-iter",
                "fobj", "l1-opt", "linft-opt", "l1-lambd", "infeas");
      }
      fprintf(fp, "%5d %8d %15.6e %9.3e %9.3e %9.3e %9.3e\n", mma_iter,
              subproblem_iter, ParOptRealPart(fobj), l1, linfty, l1_lambda,
              infeas);
      fflush(fp);

      // Set the first print flag to false
      first_print = 0;
    }
  }

  // Get the current values of the design variables
  ParOptScalar *x, *x1, *x2;
  xvec->getArray(&x);
  x1vec->getArray(&x1);
  x2vec->getArray(&x2);

  // Get the values of the assymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the lower/upper bounds for the variables
  ParOptScalar *lb, *ub;
  lbvec->getArray(&lb);
  ubvec->getArray(&ub);

  // Set all of the asymptote values
  if (mma_iter < 2) {
    for (int j = 0; j < n; j++) {
      // Apply move limit for the design variables
      ParOptScalar lower = max2(lb[j], x[j] - movlim);
      ParOptScalar upper = min2(ub[j], x[j] + movlim);

      L[j] = x[j] - init_asymptote_offset * (upper - lower);
      U[j] = x[j] + init_asymptote_offset * (upper - lower);
    }
  } else {
    for (int j = 0; j < n; j++) {
      // Apply move limit for the design variables
      ParOptScalar lower = max2(lb[j], x[j] - movlim);
      ParOptScalar upper = min2(ub[j], x[j] + movlim);

      // Compute the product of the difference of the two previous
      // updates to determine how to update the move limits. If the
      // signs are different, then indc < 0.0 and we contract the
      // asymptotes, otherwise we expand the asymptotes.
      ParOptScalar indc = (x[j] - x1[j]) * (x1[j] - x2[j]);

      // Store the previous values of the asymptotes
      ParOptScalar Lprev = L[j];
      ParOptScalar Uprev = U[j];

      // Compute the interval length
      ParOptScalar intrvl = max2(upper - lower, 0.01);
      intrvl = min2(intrvl, 100.0);

      if (ParOptRealPart(indc) < 0.0) {
        // oscillation -> contract the asymptotes
        L[j] = x[j] - asymptote_contract * (x1[j] - Lprev);
        U[j] = x[j] + asymptote_contract * (Uprev - x1[j]);
      } else {
        // Relax the asymptotes
        L[j] = x[j] - asymptote_relax * (x1[j] - Lprev);
        U[j] = x[j] + asymptote_relax * (Uprev - x1[j]);
      }

      // Ensure that the asymptotes do not converge entirely on the
      // design variable value
      L[j] = min2(L[j], x[j] - min_asymptote_offset * intrvl);
      U[j] = max2(U[j], x[j] + min_asymptote_offset * intrvl);

      // Enforce a maximum offset so that the asymptotes do not
      // move too far away from the design variables
      L[j] = max2(L[j], x[j] - max_asymptote_offset * intrvl);
      U[j] = min2(U[j], x[j] + max_asymptote_offset * intrvl);
    }
  }

  // Get the objective gradient array
  ParOptScalar *g;
  gvec->getArray(&g);

  // Allocate a temp array to store the pointers
  // to the constraint vector
  ParOptScalar **A = new ParOptScalar *[m];
  for (int i = 0; i < m; i++) {
    Avecs[i]->getArray(&A[i]);
  }

  // Get the coefficients for the objective/constraint approximation
  // information
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Get the move limit vectors
  ParOptScalar *alpha, *beta;
  alphavec->getArray(&alpha);
  betavec->getArray(&beta);

  // Compute the values of the lower/upper assymptotes
  for (int j = 0; j < n; j++) {
    // Apply move limit for the design variables
    ParOptScalar lower = max2(lb[j], x[j] - movlim);
    ParOptScalar upper = min2(ub[j], x[j] + movlim);

    // Compute the move limits to avoid division by zero
    alpha[j] = max2(max2(lower, 0.9 * L[j] + 0.1 * x[j]),
                    x[j] - 0.5 * (upper - lower));
    beta[j] = min2(min2(upper, 0.9 * U[j] + 0.1 * x[j]),
                   x[j] + 0.5 * (upper - lower));

    // Compute the coefficients for the objective
    ParOptScalar gpos = max2(0.0, g[j]);
    ParOptScalar gneg = max2(0.0, -g[j]);
    p0[j] = (U[j] - x[j]) * (U[j] - x[j]) *
            ((1.0 + delta) * gpos + delta * gneg + eps / (U[j] - L[j]));
    q0[j] = (x[j] - L[j]) * (x[j] - L[j]) *
            ((1.0 + delta) * gneg + delta * gpos + eps / (U[j] - L[j]));
  }

  if (use_true_mma) {
    memset(b, 0, m * sizeof(ParOptScalar));
    for (int i = 0; i < m; i++) {
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      // Compute the coefficients for the constraints.  Here we form a
      // convex approximation for -c(x) since the constraints are
      // formulated as c(x) >= 0.
      for (int j = 0; j < n; j++) {
        ParOptScalar gpos = max2(0.0, -A[i][j]);
        ParOptScalar gneg = max2(0.0, A[i][j]);
        pi[j] = (U[j] - x[j]) * (U[j] - x[j]) * gpos;
        qi[j] = (x[j] - L[j]) * (x[j] - L[j]) * gneg;
        b[i] += pi[j] / (U[j] - x[j]) + qi[j] / (x[j] - L[j]);
      }
    }

    // All reduce the coefficient values
    MPI_Allreduce(MPI_IN_PLACE, b, m, PAROPT_MPI_TYPE, MPI_SUM, comm);

    for (int i = 0; i < m; i++) {
      b[i] = -(cons[i] + b[i]);
    }
  }

  // Check that the asymptotes, limits and variables are well-defined
  for (int j = 0; j < n; j++) {
    if (!(ParOptRealPart(L[j]) < ParOptRealPart(alpha[j]))) {
      fprintf(stderr, "ParOptMMA: Inconsistent lower asymptote\n");
    }
    if (!(ParOptRealPart(alpha[j]) <= ParOptRealPart(x[j]))) {
      fprintf(stderr, "ParOptMMA: Inconsistent lower limit\n");
    }
    if (!(ParOptRealPart(x[j]) <= ParOptRealPart(beta[j]))) {
      fprintf(stderr, "ParOptMMA: Inconsistent upper limit\n");
    }
    if (!(ParOptRealPart(beta[j]) < ParOptRealPart(U[j]))) {
      fprintf(stderr, "ParOptMMA: Inconsistent upper assymptote\n");
    }
  }

  // Increment the number of MMA iterations
  mma_iter++;

  // Free the A pointers
  delete[] A;

  return 0;
}

/*
  Create a design vector
*/
ParOptVec *ParOptMMA::createDesignVec() { return prob->createDesignVec(); }

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptMMA::createConstraintVec() {
  return prob->createConstraintVec();
}

/*
  Create the subproblem quasi-definite matrix
*/
ParOptQuasiDefMat *ParOptMMA::createQuasiDefMat() {
  return prob->createQuasiDefMat();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptMMA::getMPIComm() { return prob->getMPIComm(); }

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptMMA::isSparseInequality() { return prob->isSparseInequality(); }

int ParOptMMA::useLowerBounds() { return 1; }

int ParOptMMA::useUpperBounds() { return 1; }

// Get the variables and bounds from the problem
void ParOptMMA::getVarsAndBounds(ParOptVec *x, ParOptVec *lb, ParOptVec *ub) {
  x->copyValues(xvec);
  lb->copyValues(alphavec);
  ub->copyValues(betavec);
}

/*
  Evaluate the objective and constraints
*/
int ParOptMMA::evalObjCon(ParOptVec *xv, ParOptScalar *fval,
                          ParOptScalar *cvals) {
  // Get the array of design variable values
  ParOptScalar *x, *x0;
  xvec->getArray(&x0);
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Compute the objective
  ParOptScalar fv = 0.0;
  for (int j = 0; j < n; j++) {
    fv += p0[j] / (U[j] - x[j]) + q0[j] / (x[j] - L[j]);
  }

  // Compute the linearized constraint
  memset(cvals, 0, m * sizeof(ParOptScalar));

  if (use_true_mma) {
    for (int i = 0; i < m; i++) {
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      // Compute the coefficients for the constraints
      for (int j = 0; j < n; j++) {
        cvals[i] += pi[j] / (U[j] - x[j]) + qi[j] / (x[j] - L[j]);
      }
    }
  } else {
    for (int i = 0; i < m; i++) {
      ParOptScalar *A;
      Avecs[i]->getArray(&A);
      for (int j = 0; j < n; j++) {
        cvals[i] += A[j] * (x[j] - x0[j]);
      }
    }
  }

  // All reduce the data
  MPI_Allreduce(&fv, fval, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, cvals, m, PAROPT_MPI_TYPE, MPI_SUM, comm);

  if (use_true_mma) {
    for (int i = 0; i < m; i++) {
      cvals[i] = -(cvals[i] + b[i]);
    }
  } else {
    for (int i = 0; i < m; i++) {
      cvals[i] += cons[i];
    }
  }

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptMMA::evalObjConGradient(ParOptVec *xv, ParOptVec *gv,
                                  ParOptVec **Ac) {
  // Keep track of the number of subproblem gradient evaluations
  subproblem_iter++;

  // Get the gradient vector
  ParOptScalar *g;
  gv->getArray(&g);

  // Get the array of design variable values
  ParOptScalar *x;
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Compute the gradient of the objective
  for (int j = 0; j < n; j++) {
    ParOptScalar Uinv = 1.0 / (U[j] - x[j]);
    ParOptScalar Linv = 1.0 / (x[j] - L[j]);
    g[j] = Uinv * Uinv * p0[j] - Linv * Linv * q0[j];
  }

  // Evaluate the gradient
  if (use_true_mma) {
    for (int i = 0; i < m; i++) {
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      ParOptScalar *A;
      Ac[i]->getArray(&A);

      for (int j = 0; j < n; j++) {
        ParOptScalar Uinv = 1.0 / (U[j] - x[j]);
        ParOptScalar Linv = 1.0 / (x[j] - L[j]);
        A[j] = Linv * Linv * qi[j] - Uinv * Uinv * pi[j];
      }
    }
  } else {
    for (int i = 0; i < m; i++) {
      Ac[i]->copyValues(Avecs[i]);
    }
  }

  return 0;
}

/*
  Evaluate the product of the Hessian with a given vector
*/
int ParOptMMA::evalHvecProduct(ParOptVec *xv, ParOptScalar *z, ParOptVec *zw,
                               ParOptVec *px, ParOptVec *hvec) {
  // Get the gradient vector
  ParOptScalar *h;
  hvec->getArray(&h);

  // Get the array of design variable values
  ParOptScalar *x;
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Get the components of the vector
  ParOptScalar *p;
  px->getArray(&p);

  // Compute the hessian of the objective
  for (int j = 0; j < n; j++) {
    ParOptScalar Uinv = 1.0 / (U[j] - x[j]);
    ParOptScalar Linv = 1.0 / (x[j] - L[j]);
    h[j] =
        2.0 * (Uinv * Uinv * Uinv * p0[j] + Linv * Linv * Linv * q0[j]) * p[j];
  }

  return 0;
}

/*
  Evaluate the diagonal Hessian matrix
*/
int ParOptMMA::evalHessianDiag(ParOptVec *xv, ParOptScalar *z, ParOptVec *zw,
                               ParOptVec *hdiag) {
  // Get the gradient vector
  ParOptScalar *h;
  hdiag->getArray(&h);

  // Get the array of design variable values
  ParOptScalar *x;
  xv->getArray(&x);

  // Get the asymptotes
  ParOptScalar *L, *U;
  Lvec->getArray(&L);
  Uvec->getArray(&U);

  // Get the coefficients for the objective
  ParOptScalar *p0, *q0;
  p0vec->getArray(&p0);
  q0vec->getArray(&q0);

  // Compute the Hessian diagonal
  for (int j = 0; j < n; j++) {
    ParOptScalar Uinv = 1.0 / (U[j] - x[j]);
    ParOptScalar Linv = 1.0 / (x[j] - L[j]);
    h[j] = 2.0 * (Uinv * Uinv * Uinv * p0[j] + Linv * Linv * Linv * q0[j]);
  }

  if (use_true_mma) {
    for (int i = 0; i < m; i++) {
      ParOptScalar *pi, *qi;
      pivecs[i]->getArray(&pi);
      qivecs[i]->getArray(&qi);

      for (int j = 0; j < n; j++) {
        ParOptScalar Uinv = 1.0 / (U[j] - x[j]);
        ParOptScalar Linv = 1.0 / (x[j] - L[j]);
        h[j] += 2.0 * z[i] *
                (Uinv * Uinv * Uinv * pi[j] + Linv * Linv * Linv * qi[j]);
      }
    }
  }

  return 0;
}

/*
  Evaluate the constraints
*/
void ParOptMMA::evalSparseCon(ParOptVec *x, ParOptVec *out) {
  if (nwcon > 0) {
    out->copyValues(cwvec);
    prob->addSparseJacobian(1.0, xvec, x, out);
    prob->addSparseJacobian(-1.0, xvec, xvec, out);
  }
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptMMA::addSparseJacobian(ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *px, ParOptVec *out) {
  if (nwcon > 0) {
    prob->addSparseJacobian(alpha, xvec, px, out);
  }
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptMMA::addSparseJacobianTranspose(ParOptScalar alpha, ParOptVec *x,
                                           ParOptVec *pzw, ParOptVec *out) {
  if (nwcon > 0) {
    prob->addSparseJacobianTranspose(alpha, xvec, pzw, out);
  }
}

/*
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptMMA::addSparseInnerProduct(ParOptScalar alpha, ParOptVec *x,
                                      ParOptVec *cvec, ParOptScalar *A) {
  if (nwcon > 0) {
    prob->addSparseInnerProduct(alpha, xvec, cvec, A);
  }
}
