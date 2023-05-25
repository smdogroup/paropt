#include "ParOptInteriorPoint.h"

#include <math.h>
#include <string.h>

#include <cstdarg>

#include "ParOptBlasLapack.h"
#include "ParOptComplexStep.h"

/*
  Static helper functions
*/
ParOptScalar max2(const ParOptScalar a, const ParOptScalar b) {
  if (ParOptRealPart(a) > ParOptRealPart(b)) {
    return a;
  }
  return b;
}

ParOptScalar min2(const ParOptScalar a, const ParOptScalar b) {
  if (ParOptRealPart(a) < ParOptRealPart(b)) {
    return a;
  }
  return b;
}

ParOptInteriorPoint::ParOptVars::ParOptVars() {
  x = NULL;
  zl = NULL;
  zu = NULL;
  s = NULL;
  t = NULL;
  z = NULL;
  zs = NULL;
  zt = NULL;
  sw = NULL;
  tw = NULL;
  zw = NULL;
  zsw = NULL;
  ztw = NULL;
}

ParOptInteriorPoint::ParOptVars::~ParOptVars() {
  if (x) {
    x->decref();
  }
  if (zl) {
    zl->decref();
  }
  if (zu) {
    zu->decref();
  }
  if (s) {
    delete[] s;
  }
  if (t) {
    delete[] t;
  }
  if (z) {
    delete[] z;
  }
  if (zs) {
    delete[] zs;
  }
  if (zt) {
    delete[] zt;
  }
  if (sw) {
    sw->decref();
  }
  if (tw) {
    tw->decref();
  }
  if (zw) {
    zw->decref();
  }
  if (zsw) {
    zsw->decref();
  }
  if (ztw) {
    ztw->decref();
  }
}

void ParOptInteriorPoint::ParOptVars::initialize(ParOptProblem *prob) {
  int ncon;
  prob->getProblemSizes(NULL, &ncon, NULL);

  x = prob->createDesignVec();
  x->incref();

  // Allocate storage space for the variables etc.
  zl = prob->createDesignVec();
  zl->incref();
  zu = prob->createDesignVec();
  zu->incref();

  // Allocate space for the sparse constraints
  zw = prob->createConstraintVec();
  zw->incref();
  sw = prob->createConstraintVec();
  sw->incref();
  tw = prob->createConstraintVec();
  tw->incref();
  zsw = prob->createConstraintVec();
  zsw->incref();
  ztw = prob->createConstraintVec();
  ztw->incref();

  // Set the initial values of the Lagrange multipliers
  z = new ParOptScalar[ncon];
  s = new ParOptScalar[ncon];
  t = new ParOptScalar[ncon];

  // Set the multipliers for l1-penalty term
  zs = new ParOptScalar[ncon];
  zt = new ParOptScalar[ncon];

  // Zero the initial values
  memset(z, 0, ncon * sizeof(ParOptScalar));
  memset(s, 0, ncon * sizeof(ParOptScalar));
  memset(t, 0, ncon * sizeof(ParOptScalar));
  memset(zs, 0, ncon * sizeof(ParOptScalar));
  memset(zt, 0, ncon * sizeof(ParOptScalar));
}

/**
   ParOpt interior point optimization constructor.

   This function allocates and initializes the data that is required
   for parallel optimization. This includes initialization of the
   variables, allocation of the matrices and the BFGS approximate
   Hessian. This code also sets the default parameters for
   optimization. These parameters can be modified through member
   functions.

   @param prob the optimization problem
   @param options the options object
*/
ParOptInteriorPoint::ParOptInteriorPoint(ParOptProblem *_prob,
                                         ParOptOptions *_options) {
  // Set the problem object
  prob = _prob;
  prob->incref();

  if (_options) {
    options = _options;
  } else {
    options = new ParOptOptions(prob->getMPIComm());
    addDefaultOptions(options);
  }
  options->incref();

  // Record the communicator
  comm = prob->getMPIComm();
  opt_root = 0;

  // Get the number of variables/constraints
  prob->getProblemSizes(&nvars, &ncon, &nwcon);
  prob->getNumInequalities(&ninequality, &nwinequality);

  // Check whether to use upper/lower bounds
  use_lower = prob->useLowerBounds();
  use_upper = prob->useUpperBounds();

  // Calculate the total number of variable across all processors
  int size, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Allocate space to store the variable ranges
  var_range = new int[size + 1];
  wcon_range = new int[size + 1];
  var_range[0] = 0;
  wcon_range[0] = 0;

  // Count up the displacements/variable ranges
  MPI_Allgather(&nvars, 1, MPI_INT, &var_range[1], 1, MPI_INT, comm);
  MPI_Allgather(&nwcon, 1, MPI_INT, &wcon_range[1], 1, MPI_INT, comm);

  for (int k = 0; k < size; k++) {
    var_range[k + 1] += var_range[k];
    wcon_range[k + 1] += wcon_range[k];
  }

  // Set the total number of variables
  nvars_total = var_range[size];

  variables.initialize(prob);
  update.initialize(prob);
  residual.initialize(prob);

  // Create the bounds
  lb = prob->createDesignVec();
  lb->incref();
  ub = prob->createDesignVec();
  ub->incref();

  // Allocate space for the Quasi-Newton updates
  y_qn = prob->createDesignVec();
  y_qn->incref();
  s_qn = prob->createDesignVec();
  s_qn->incref();

  // Allocate vectors for the weighting constraints
  wtemp = prob->createConstraintVec();
  wtemp->incref();
  xtemp = prob->createDesignVec();
  xtemp->incref();

  // Allocate space for the Dmatrix
  Gmat = new ParOptScalar[ncon * ncon];
  gpiv = new int[ncon];

  // Allocate the quasi-Newton approximation
  const char *qn_type = options->getEnumOption("qn_type");
  const int qn_subspace_size = options->getIntOption("qn_subspace_size");

  qn = NULL;
  if (strcmp(qn_type, "bfgs") == 0) {
    ParOptLBFGS *bfgs = new ParOptLBFGS(prob, qn_subspace_size);
    qn = bfgs;
    qn->incref();

    const char *update_type = options->getEnumOption("qn_update_type");
    if (strcmp(update_type, "skip_negative_curvature") == 0) {
      bfgs->setBFGSUpdateType(PAROPT_SKIP_NEGATIVE_CURVATURE);
    } else if (strcmp(update_type, "damped_update") == 0) {
      bfgs->setBFGSUpdateType(PAROPT_DAMPED_UPDATE);
    }
  } else if (strcmp(qn_type, "sr1") == 0) {
    qn = new ParOptLSR1(prob, qn_subspace_size);
    qn->incref();
  }

  if (qn) {
    const char *diag_type = options->getEnumOption("qn_diag_type");
    if (strcmp(diag_type, "yty_over_yts") == 0) {
      qn->setInitDiagonalType(PAROPT_YTY_OVER_YTS);
    } else if (strcmp(diag_type, "yts_over_sts") == 0) {
      qn->setInitDiagonalType(PAROPT_YTS_OVER_STS);
    } else if (strcmp(diag_type, "inner_yty_over_yts") == 0) {
      qn->setInitDiagonalType(PAROPT_INNER_PRODUCT_YTY_OVER_YTS);
    } else {
      qn->setInitDiagonalType(PAROPT_INNER_PRODUCT_YTS_OVER_STS);
    }
  }

  hdiag = NULL;

  // Get the maximum subspace size
  int max_qn_size = 0;
  if (qn) {
    max_qn_size = qn->getMaxLimitedMemorySize();
  }

  if (max_qn_size > 0) {
    // Allocate storage for bfgs/constraint sized things
    int zsize = max_qn_size;
    if (ncon > zsize) {
      zsize = ncon;
    }
    ztemp = new ParOptScalar[zsize];

    // Allocate space for the Ce matrix
    Ce = new ParOptScalar[max_qn_size * max_qn_size];
    cpiv = new int[max_qn_size];
  } else {
    ztemp = NULL;
    Ce = NULL;
    cpiv = NULL;
  }

  // Allocate space for the diagonal matrix components
  Dinv = prob->createDesignVec();
  Dinv->incref();
  Cdiag = prob->createConstraintVec();
  Cdiag->incref();

  mat = prob->createQuasiDefMat();
  mat->incref();

  // Set the value of the objective
  fobj = 0.0;

  // Set the constraints to zero
  c = new ParOptScalar[ncon];
  memset(c, 0, ncon * sizeof(ParOptScalar));

  // Set the objective and constraint gradients
  g = prob->createDesignVec();
  g->incref();

  Ac = new ParOptVec *[ncon];
  for (int i = 0; i < ncon; i++) {
    Ac[i] = prob->createDesignVec();
    Ac[i]->incref();
  }

  // Set the default penalty values
  const double gamma = options->getFloatOption("penalty_gamma");
  penalty_gamma_s = new double[ncon];
  penalty_gamma_t = new double[ncon];
  for (int i = 0; i < ncon; i++) {
    if (i < ninequality) {
      penalty_gamma_s[i] = 0.0;
      penalty_gamma_t[i] = gamma;
    } else {
      penalty_gamma_s[i] = gamma;
      penalty_gamma_t[i] = gamma;
    }
  }

  // Set the penalties for the sparse constraints
  penalty_gamma_sw = prob->createConstraintVec();
  penalty_gamma_sw->incref();
  penalty_gamma_tw = prob->createConstraintVec();
  penalty_gamma_tw->incref();

  ParOptScalar *sw_gamma, *tw_gamma;
  penalty_gamma_sw->getArray(&sw_gamma);
  penalty_gamma_tw->getArray(&tw_gamma);
  for (int i = 0; i < nwcon; i++) {
    if (i < nwinequality) {
      sw_gamma[i] = 0.0;
      tw_gamma[i] = gamma;
    } else {
      sw_gamma[i] = gamma;
      tw_gamma[i] = gamma;
    }
  }

  // Set parameters that will be over-written later
  barrier_param = options->getFloatOption("init_barrier_param");
  rho_penalty_search = options->getFloatOption("init_rho_penalty_search");

  // Zero the number of evals
  neval = ngeval = nhvec = 0;

  // Set the default information about GMRES
  gmres_subspace_size = 0;
  gmres_H = NULL;
  gmres_alpha = NULL;
  gmres_res = NULL;
  gmres_y = NULL;
  gmres_fproj = NULL;
  gmres_aproj = NULL;
  gmres_awproj = NULL;
  gmres_Q = NULL;
  gmres_W = NULL;

  // Check if we're going to use an optimization problem with an inexact
  // optimization method.
  int m = options->getIntOption("gmres_subspace_size");
  if (m > 0) {
    setGMRESSubspaceSize(m);
  }

  // By default, set the file pointer to stdout. If a filename is specified,
  // set the new filename.
  outfp = NULL;
  if (rank == opt_root) {
    outfp = stdout;
  }

  const char *filename = options->getStringOption("output_file");
  if (filename) {
    setOutputFile(filename);
  }

  // Initialize the design variables and bounds
  initAndCheckDesignAndBounds();

  // Set initial values of the multipliers
  variables.zl->set(1.0);
  variables.zu->set(1.0);

  // Set the multipliers and slack variables associated with the
  // sparse constraints all to 1.0
  variables.zw->set(1.0);
  variables.sw->set(1.0);
  variables.tw->set(1.0);
  variables.zsw->set(1.0);
  variables.ztw->set(1.0);

  // Set the Largrange multipliers and slack variables associated
  // with the dense constraints to 1.0
  for (int i = 0; i < ncon; i++) {
    variables.z[i] = 1.0;
    variables.s[i] = 1.0;
    variables.t[i] = 1.0;
    variables.zt[i] = 1.0;
    variables.zs[i] = 1.0;
  }
}

/**
   Free the data allocated during the creation of the object
*/
ParOptInteriorPoint::~ParOptInteriorPoint() {
  prob->decref();
  options->decref();
  if (qn) {
    qn->decref();
  }

  // Delete the bounds
  lb->decref();
  ub->decref();

  // Delete the quasi-Newton updates
  y_qn->decref();
  s_qn->decref();

  // Delete the temp data
  wtemp->decref();
  xtemp->decref();
  if (ztemp) {
    delete[] ztemp;
  }

  // Delete the Schur complement for the dense constraints
  delete[] Gmat;
  delete[] gpiv;

  // Delete the Schur complement for the quasi-Newton matrix
  if (Ce) {
    delete[] Ce;
  }
  if (cpiv) {
    delete[] cpiv;
  }

  // Delete the vector of penalty parameters
  delete[] penalty_gamma_s;
  delete[] penalty_gamma_t;
  penalty_gamma_sw->decref();
  penalty_gamma_tw->decref();

  // Free the quasi-def matrix
  mat->decref();

  // Delete the diagonal matrices
  Dinv->decref();
  Cdiag->decref();

  // Free the variable ranges
  delete[] var_range;
  delete[] wcon_range;

  // Free the diagonal hessian (if allocated)
  if (hdiag) {
    hdiag->decref();
  }

  // Delete the constraint/gradient information
  delete[] c;
  g->decref();
  for (int i = 0; i < ncon; i++) {
    Ac[i]->decref();
  }
  delete[] Ac;

  // Delete the GMRES information if any
  if (gmres_subspace_size > 0) {
    delete[] gmres_H;
    delete[] gmres_alpha;
    delete[] gmres_res;
    delete[] gmres_y;
    delete[] gmres_fproj;
    delete[] gmres_aproj;
    delete[] gmres_awproj;
    delete[] gmres_Q;

    // Delete the subspace
    for (int i = 0; i < gmres_subspace_size; i++) {
      gmres_W[i]->decref();
    }
    delete[] gmres_W;
  }

  // Close the output file if it's not stdout
  if (outfp && outfp != stdout) {
    fclose(outfp);
  }
}

/**
  Add the default options to the input options object

  @param options The input options argument
*/
void ParOptInteriorPoint::addDefaultOptions(ParOptOptions *options) {
  // Set the string options
  options->addStringOption("output_file", "paropt.out", "Output file name");

  options->addStringOption("problem_name", NULL, "The problem name");

  // Set the float options
  options->addFloatOption(
      "max_bound_value", 1e20, 0.0, 1e300,
      "Maximum bound value at which bound constraints are omitted");

  options->addFloatOption("abs_res_tol", 1e-6, 0.0, 1e20,
                          "Absolute stopping criterion");

  options->addFloatOption("rel_func_tol", 0.0, 0.0, 1e20,
                          "Relative function value stopping criterion");

  options->addFloatOption("abs_step_tol", 0.0, 0.0, 1e20,
                          "Absolute stopping norm on the step size");

  options->addFloatOption("init_barrier_param", 0.1, 0.0, 1e20,
                          "The initial value of the barrier parameter");

  options->addFloatOption("penalty_gamma", 1000.0, 0.0, 1e20,
                          "l1 penalty parameter applied to slack variables");

  options->addFloatOption(
      "penalty_descent_fraction", 0.3, 1e-6, 1.0,
      "Fraction of infeasibility used to enforce a descent direction");

  options->addFloatOption("min_rho_penalty_search", 0.0, 0.0, 1e20,
                          "Minimum value of the line search penalty parameter");

  options->addFloatOption("init_rho_penalty_search", 0.0, 0.0, 1e20,
                          "Initial value of the line search penalty parameter");

  options->addFloatOption("armijo_constant", 1e-5, 0.0, 1.0,
                          "The Armijo constant for the line search");

  options->addFloatOption("monotone_barrier_fraction", 0.25, 0.0, 1.0,
                          "Factor applied to the barrier update < 1");

  options->addFloatOption("monotone_barrier_power", 1.1, 1.0, 10.0,
                          "Exponent for barrier parameter update > 1");

  options->addFloatOption(
      "rel_bound_barrier", 1.0, 0.0, 1e20,
      "Relative factor applied to barrier parameter for bound constraints");

  options->addFloatOption("min_fraction_to_boundary", 0.95, 0.0, 1.0,
                          "Minimum fraction to the boundary rule < 1");

  options->addFloatOption(
      "qn_sigma", 0.0, 0.0, 1e20,
      "Scalar added to the diagonal of the quasi-Newton approximation > 0");

  options->addFloatOption(
      "nk_switch_tol", 1e-3, 0.0, 1e20,
      "Switch to the Newton-Krylov method at this residual tolerance");

  options->addFloatOption(
      "eisenstat_walker_alpha", 1.5, 0.0, 2.0,
      "Exponent in the Eisenstat-Walker INK forcing equation");

  options->addFloatOption(
      "eisenstat_walker_gamma", 1.0, 0.0, 1.0,
      "Multiplier in the Eisenstat-Walker INK forcing equation");

  options->addFloatOption(
      "max_gmres_rtol", 0.1, 0.0, 1.0,
      "The maximum relative tolerance used for GMRES, above this "
      "the quasi-Newton approximation is used");

  options->addFloatOption(
      "gmres_atol", 1e-30, 0.0, 1.0,
      "The absolute GMRES tolerance (almost never relevant)");

  options->addFloatOption(
      "function_precision", 1e-10, 0.0, 1.0,
      "The absolute precision of the function and constraints");

  options->addFloatOption("design_precision", 1e-14, 0.0, 1.0,
                          "The absolute precision of the design variables");

  options->addFloatOption(
      "start_affine_multiplier_min", 1.0, 0.0, 1e20,
      "Minimum multiplier for the affine step initialization strategy");

  // Set the boolean options
  options->addBoolOption("use_line_search", 1,
                         "Perform or skip the line search");

  options->addBoolOption("use_backtracking_alpha", 0,
                         "Perform a back-tracking line search");

  options->addBoolOption("sequential_linear_method", 0,
                         "Discard the quasi-Newton approximation (but not "
                         "necessarily the exact Hessian)");

  options->addBoolOption(
      "use_quasi_newton_update", 1,
      "Update the quasi-Newton approximation at each iteration");

  options->addBoolOption("use_hvec_product", 0,
                         "Use or do not use Hessian-vector products");

  options->addBoolOption("use_diag_hessian", 0,
                         "Use or do not use the diagonal Hessian computation");

  options->addBoolOption(
      "use_qn_gmres_precon", 1,
      "Use or do not use the quasi-Newton method as a preconditioner");

  options->addFloatOption("gradient_check_step_length", 1e-6, 0.0, 1.0,
                          "Step length used to check the gradient");

  // Set the integer options
  options->addIntOption(
      "qn_subspace_size", 10, 0, 1000,
      "The maximum dimension of the quasi-Newton approximation");

  options->addIntOption(
      "max_major_iters", 5000, 0, 1000000,
      "The maximum number of major iterations before quiting");

  options->addIntOption("max_line_iters", 10, 1, 100,
                        "Maximum number of line search iterations");

  options->addIntOption("gmres_subspace_size", 0, 0, 1000,
                        "The subspace size for GMRES");

  options->addIntOption("write_output_frequency", 10, 0, 1000000,
                        "Write out the solution file and checkpoint file "
                        "at this frequency");

  options->addIntOption("gradient_verification_frequency", -1, -1000000,
                        1000000,
                        "Print to screen the output of the gradient check "
                        "at this frequency during an optimization");

  options->addIntOption(
      "hessian_reset_freq", 1000000, 1, 1000000,
      "Do a hard reset of the Hessian at this specified major "
      "iteration frequency");

  options->addIntOption(
      "output_level", 0, 0, 1000000,
      "Output level indicating how verbose the output should be");

  // Set the enumerated options
  const char *qn_type[4] = {"bfgs", "scaled_bfgs", "sr1", "none"};
  options->addEnumOption(
      "qn_type", "bfgs", 4, qn_type,
      "The type of quasi-Newton approximation to use, note that "
      "scaled_bfgs should be only used when there's single constraint "
      "and objective is linear");

  const char *bfgs_type[2] = {"skip_negative_curvature", "damped_update"};
  options->addEnumOption(
      "qn_update_type", "skip_negative_curvature", 2, bfgs_type,
      "The type of BFGS update to apply when the curvature condition fails");

  const char *qn_diag_type[4] = {"yty_over_yts", "yts_over_sts",
                                 "inner_yty_over_yts", "inner_yts_over_sts"};
  options->addEnumOption(
      "qn_diag_type", "yty_over_yts", 4, qn_diag_type,
      "The type of initial diagonal to use in the quasi-Newton approximation");

  const char *norm_options[3] = {"infinity", "l1", "l2"};
  options->addEnumOption("norm_type", "infinity", 3, norm_options,
                         "The type of norm to use in all computations");

  const char *barrier_options[4] = {"monotone", "mehrotra",
                                    "mehrotra_predictor_corrector",
                                    "complementarity_fraction"};
  options->addEnumOption("barrier_strategy", "monotone", 4, barrier_options,
                         "The type of barrier update strategy to use");

  const char *start_options[3] = {"least_squares_multipliers", "affine_step",
                                  "no_start_strategy"};
  options->addEnumOption(
      "starting_point_strategy", "affine_step", 3, start_options,
      "Initialize the Lagrange multiplier estimates and slack variables");
}

/**
  Get the options object associated with the interior point method

  @return The options object
*/
ParOptOptions *ParOptInteriorPoint::getOptions() { return options; }

/**
   Reset the problem instance.

   The new problem instance must have the same number of constraints
   design variables, and design vector distribution as the original
   problem.

   @param problem ParOptProblem instance
*/
void ParOptInteriorPoint::resetProblemInstance(ParOptProblem *problem) {
  // Check to see if the new problem instance is congruent with
  // the previous problem instance - it has to be otherwise
  // we can't use it.
  int _nvars, _ncon, _nwcon;
  problem->getProblemSizes(&_nvars, &_ncon, &_nwcon);

  int nineq, nwineq;
  problem->getNumInequalities(&nineq, &nwineq);

  if (_nvars != nvars || _ncon != ncon || _nwcon != nwcon ||
      nineq != ninequality || nwineq != nwinequality) {
    fprintf(stderr, "ParOpt: Incompatible problem instance\n");
    problem = NULL;
  } else {
    problem->incref();
    prob->decref();
    prob = problem;
  }
}

/**
   Retrieve the problem sizes from the underlying problem class

   @param _nvars the local number of variables
   @param _ncon the number of global constraints
   @param _nwcon the number of sparse constraints
*/
void ParOptInteriorPoint::getProblemSizes(int *_nvars, int *_ncon,
                                          int *_nwcon) {
  prob->getProblemSizes(_nvars, _ncon, _nwcon);
}

/**
   Retrieve the values of the design variables and multipliers.

   This call can be made during the course of an optimization, but
   changing the values in x/zw/zl/zu is not recommended and the
   behavior after doing so is not defined. Note that inputs that are
   NULL are not assigned. If no output is available, for instance if
   use_lower == False, then NULL is assigned to the output.

   @param _x the design variable values
   @param _z the dense constraint multipliers
   @param _zw the sparse constraint multipliers
   @param _zl the lower bound multipliers
   @param _zu the upper bound multipliers
*/
void ParOptInteriorPoint::getOptimizedPoint(ParOptVec **_x, ParOptScalar **_z,
                                            ParOptVec **_zw, ParOptVec **_zl,
                                            ParOptVec **_zu) {
  if (_x) {
    *_x = variables.x;
  }
  if (_z) {
    *_z = NULL;
    *_z = variables.z;
  }
  if (_zw) {
    *_zw = NULL;
    *_zw = variables.zw;
  }
  if (_zl) {
    *_zl = NULL;
    if (use_lower) {
      *_zl = variables.zl;
    }
  }
  if (_zu) {
    *_zu = NULL;
    if (use_upper) {
      *_zu = variables.zu;
    }
  }
}

/**
   Retrieve the values of the optimized slack variables.

   Note that the dense inequality constraints are formualted as

   c(x) = s - t

   where s, t > 0. And the sparse inequality constraints are formulated
   as:

   cw(x) = sw - tw

   where sw, tw > 0. When equality rather than inequality constraints are
   present, sw and tw may be NULL.

   @param _s the postive slack for the dense constraints
   @param _t the negative slack for the dense constraints
   @param _sw the positive slack variable vector for the sparse constraints
   @param _tw the negative slack variable vector for the sparse constraints
*/
void ParOptInteriorPoint::getOptimizedSlacks(ParOptScalar **_s,
                                             ParOptScalar **_t, ParOptVec **_sw,
                                             ParOptVec **_tw) {
  if (_s) {
    *_s = variables.s;
  }
  if (_t) {
    *_t = variables.t;
  }
  if (_sw) {
    *_sw = NULL;
    *_sw = variables.sw;
  }
  if (_tw) {
    *_tw = NULL;
    *_tw = variables.tw;
  }
}

/**
   Write out all of the options that have been set to a output
   stream.

   This function is typically invoked when the output summary file
   is written.

   @param fp an open file handle
*/
void ParOptInteriorPoint::printOptionSummary(FILE *fp) {
  if (fp) {
    const int output_level = options->getIntOption("output_level");
    fprintf(fp, "ParOptInteriorPoint Parameter Summary:\n");
    options->printSummary(fp, output_level);
  }
}

/**
   Write out the design variables, Lagrange multipliers and
   slack variables to a binary file in parallel.

   @param filename is the name of the file to write
*/
int ParOptInteriorPoint::writeSolutionFile(const char *filename) {
  char *fname = new char[strlen(filename) + 1];
  strcpy(fname, filename);

  int fail = 1;
  MPI_File fp = NULL;
  MPI_File_open(comm, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL,
                &fp);

  if (fp) {
    // Calculate the total number of variable across all processors
    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Successfull opened the file
    fail = 0;

    // Write the problem sizes on the root processor
    if (rank == opt_root) {
      int var_sizes[3];
      var_sizes[0] = var_range[size];
      var_sizes[1] = wcon_range[size];
      var_sizes[2] = ncon;

      MPI_File_write(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);
      MPI_File_write(fp, &barrier_param, 1, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, variables.s, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, variables.t, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, variables.z, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, variables.zs, ncon, PAROPT_MPI_TYPE,
                     MPI_STATUS_IGNORE);
      MPI_File_write(fp, variables.zt, ncon, PAROPT_MPI_TYPE,
                     MPI_STATUS_IGNORE);
    }

    size_t offset = 3 * sizeof(int) + (5 * ncon + 1) * sizeof(ParOptScalar);

    // Use the native representation for the data
    char datarep[] = "native";

    // Extract the design variables
    ParOptScalar *xvals;
    int xsize = variables.x->getArray(&xvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                      MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], xvals, xsize, PAROPT_MPI_TYPE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size] * sizeof(ParOptScalar);

    // Extract the lower Lagrange multipliers
    ParOptScalar *zlvals, *zuvals;
    variables.zl->getArray(&zlvals);
    variables.zu->getArray(&zuvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                      MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], zlvals, xsize, PAROPT_MPI_TYPE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size] * sizeof(ParOptScalar);

    // Write out the upper Lagrange multipliers
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                      MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], zuvals, xsize, PAROPT_MPI_TYPE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size] * sizeof(ParOptScalar);

    // Write out the extra constraint bounds
    if (wcon_range[size] > 0) {
      ParOptScalar *zwvals, *swvals;
      int nwsize = variables.zw->getArray(&zwvals);
      variables.sw->getArray(&swvals);
      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                        MPI_INFO_NULL);
      MPI_File_write_at_all(fp, wcon_range[rank], zwvals, nwsize,
                            PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      offset += wcon_range[size] * sizeof(ParOptScalar);

      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                        MPI_INFO_NULL);
      MPI_File_write_at_all(fp, wcon_range[rank], swvals, nwsize,
                            PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fp);
  }

  delete[] fname;

  return fail;
}

/**
   Read in the design variables, Lagrange multipliers and slack
   variables from a binary file.

   This function requires that the same problem structure as the
   original problem.

   @param filename is the name of the file input
*/
int ParOptInteriorPoint::readSolutionFile(const char *filename) {
  char *fname = new char[strlen(filename) + 1];
  strcpy(fname, filename);

  int fail = 1;
  MPI_File fp = NULL;
  MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
  delete[] fname;

  if (fp) {
    // Calculate the total number of variable across all processors
    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Successfully opened the file for reading
    fail = 0;

    // Keep track of whether the failure to load is due to a problem
    // size failure
    int size_fail = 0;

    // Read in the sizes
    if (rank == opt_root) {
      int var_sizes[3];
      MPI_File_read(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);

      if (var_sizes[0] != var_range[size] || var_sizes[1] != wcon_range[size] ||
          var_sizes[2] != ncon) {
        size_fail = 1;
      }

      if (!size_fail) {
        MPI_File_read(fp, &barrier_param, 1, PAROPT_MPI_TYPE,
                      MPI_STATUS_IGNORE);
        MPI_File_read(fp, variables.s, ncon, PAROPT_MPI_TYPE,
                      MPI_STATUS_IGNORE);
        MPI_File_read(fp, variables.t, ncon, PAROPT_MPI_TYPE,
                      MPI_STATUS_IGNORE);
        MPI_File_read(fp, variables.z, ncon, PAROPT_MPI_TYPE,
                      MPI_STATUS_IGNORE);
        MPI_File_read(fp, variables.zs, ncon, PAROPT_MPI_TYPE,
                      MPI_STATUS_IGNORE);
        MPI_File_read(fp, variables.zt, ncon, PAROPT_MPI_TYPE,
                      MPI_STATUS_IGNORE);
      }
    }
    MPI_Bcast(&size_fail, 1, MPI_INT, opt_root, comm);

    // The problem sizes are inconsistent, return
    if (size_fail) {
      fail = 1;
      if (rank == opt_root) {
        fprintf(stderr,
                "ParOpt: Problem size incompatible with solution file\n");
      }

      MPI_File_close(&fp);
      return fail;
    }

    // Broadcast the multipliers and slack variables for the dense constraints
    MPI_Bcast(variables.s, ncon, PAROPT_MPI_TYPE, opt_root, comm);
    MPI_Bcast(variables.t, ncon, PAROPT_MPI_TYPE, opt_root, comm);
    MPI_Bcast(variables.z, ncon, PAROPT_MPI_TYPE, opt_root, comm);
    MPI_Bcast(variables.zs, ncon, PAROPT_MPI_TYPE, opt_root, comm);
    MPI_Bcast(variables.zt, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Set the initial offset
    size_t offset = 3 * sizeof(int) + (5 * ncon + 1) * sizeof(ParOptScalar);

    // Use the native representation for the data
    char datarep[] = "native";

    // Extract the design variables
    ParOptScalar *xvals;
    int xsize = variables.x->getArray(&xvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                      MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], xvals, xsize, PAROPT_MPI_TYPE,
                         MPI_STATUS_IGNORE);
    offset += var_range[size] * sizeof(ParOptScalar);

    // Extract the lower Lagrange multipliers
    ParOptScalar *zlvals, *zuvals;
    variables.zl->getArray(&zlvals);
    variables.zu->getArray(&zuvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                      MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], zlvals, xsize, PAROPT_MPI_TYPE,
                         MPI_STATUS_IGNORE);
    offset += var_range[size] * sizeof(ParOptScalar);

    // Read in the upper Lagrange multipliers
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                      MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], zuvals, xsize, PAROPT_MPI_TYPE,
                         MPI_STATUS_IGNORE);
    offset += var_range[size] * sizeof(ParOptScalar);

    // Read in the extra constraint Lagrange multipliers
    if (wcon_range[size] > 0) {
      ParOptScalar *zwvals, *swvals;
      int nwsize = variables.zw->getArray(&zwvals);
      variables.sw->getArray(&swvals);
      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                        MPI_INFO_NULL);
      MPI_File_read_at_all(fp, wcon_range[rank], zwvals, nwsize,
                           PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      offset += wcon_range[size] * sizeof(ParOptScalar);

      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE, datarep,
                        MPI_INFO_NULL);
      MPI_File_read_at_all(fp, wcon_range[rank], swvals, nwsize,
                           PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fp);
  }

  return fail;
}

/**
   Retrieve the barrier parameter

   @return the barrier parameter value.
*/
double ParOptInteriorPoint::getBarrierParameter() { return barrier_param; }

/**
   Get the average of the complementarity products at the current
   point.  This function call is collective on all processors.

   @return the current complementarity.
*/
ParOptScalar ParOptInteriorPoint::getComplementarity() {
  return computeComp(variables);
}

/**
   Set the penalty parameter for the l1 penalty function.

   @param gamma is the value of the penalty parameter
*/
void ParOptInteriorPoint::setPenaltyGamma(double gamma) {
  if (gamma >= 0.0) {
    for (int i = 0; i < ncon; i++) {
      if (i < ninequality) {
        penalty_gamma_s[i] = 0.0;
        penalty_gamma_t[i] = gamma;
      } else {
        penalty_gamma_s[i] = gamma;
        penalty_gamma_t[i] = gamma;
      }
    }

    ParOptScalar *sw_gamma, *tw_gamma;
    penalty_gamma_sw->getArray(&sw_gamma);
    penalty_gamma_tw->getArray(&tw_gamma);
    for (int i = 0; i < nwcon; i++) {
      if (i < nwinequality) {
        sw_gamma[i] = 0.0;
        tw_gamma[i] = gamma;
      } else {
        sw_gamma[i] = gamma;
        tw_gamma[i] = gamma;
      }
    }
  }
}

/**
   Set the individual penalty parameters for the l1 penalty function.

   @param gamma is the array of penalty parameter values.
*/
void ParOptInteriorPoint::setPenaltyGamma(const double *gamma) {
  for (int i = 0; i < ncon; i++) {
    if (gamma[i] >= 0.0) {
      if (i < ninequality) {
        penalty_gamma_s[i] = 0.0;
        penalty_gamma_t[i] = gamma[i];
      } else {
        penalty_gamma_s[i] = gamma[i];
        penalty_gamma_t[i] = gamma[i];
      }
    }
  }
}

/**
   Set the type of BFGS update.

   @param update is the type of BFGS update to use
*/
void ParOptInteriorPoint::setBFGSUpdateType(ParOptBFGSUpdateType update) {
  if (qn) {
    ParOptLBFGS *lbfgs = dynamic_cast<ParOptLBFGS *>(qn);
    if (lbfgs) {
      lbfgs->setBFGSUpdateType(update);
    }
  }
}

/**
   Set the quasi-Newton update object.

   @param _qn the compact quasi-Newton approximation
*/
void ParOptInteriorPoint::setQuasiNewton(ParOptCompactQuasiNewton *_qn) {
  if (_qn) {
    _qn->incref();
  }
  if (qn) {
    qn->decref();
  }
  qn = _qn;

  // Free the old data
  if (ztemp) {
    delete[] ztemp;
  }
  if (Ce) {
    delete[] Ce;
  }
  if (cpiv) {
    delete[] cpiv;
  }

  // Get the maximum subspace size
  int max_qn_subspace = 0;
  if (qn) {
    max_qn_subspace = qn->getMaxLimitedMemorySize();
  }

  if (max_qn_subspace > 0) {
    // Allocate storage for bfgs/constraint sized things
    int zsize = max_qn_subspace;
    if (ncon > zsize) {
      zsize = ncon;
    }
    ztemp = new ParOptScalar[zsize];

    // Allocate space for the Ce matrix
    Ce = new ParOptScalar[max_qn_subspace * max_qn_subspace];
    cpiv = new int[max_qn_subspace];
  } else {
    ztemp = NULL;
    Ce = NULL;
    cpiv = NULL;
  }
}

/**
   Reset the Quasi-Newton Hessian approximation if it is used.
*/
void ParOptInteriorPoint::resetQuasiNewtonHessian() {
  if (qn) {
    qn->reset();
  }
}

/**
   Reset the design variables and bounds.
*/
void ParOptInteriorPoint::resetDesignAndBounds() {
  prob->getVarsAndBounds(variables.x, lb, ub);
}

/**
   Set the size of the GMRES subspace and allocate the vectors
   required. Note that the old subspace information is deleted before
   the new subspace data is allocated.

   @param m the GMRES subspace size.
*/
void ParOptInteriorPoint::setGMRESSubspaceSize(int m) {
  if (gmres_H) {
    delete[] gmres_H;
    delete[] gmres_alpha;
    delete[] gmres_res;
    delete[] gmres_y;
    delete[] gmres_fproj;
    delete[] gmres_aproj;
    delete[] gmres_awproj;
    delete[] gmres_Q;

    for (int i = 0; i < gmres_subspace_size; i++) {
      gmres_W[i]->decref();
    }
    delete[] gmres_W;
  }

  if (m > 0) {
    gmres_subspace_size = m;
    gmres_H = new ParOptScalar[(m + 1) * (m + 2) / 2];
    gmres_alpha = new ParOptScalar[m + 1];
    gmres_res = new ParOptScalar[m + 1];
    gmres_y = new ParOptScalar[m + 1];
    gmres_fproj = new ParOptScalar[m + 1];
    gmres_aproj = new ParOptScalar[m + 1];
    gmres_awproj = new ParOptScalar[m + 1];
    gmres_Q = new ParOptScalar[2 * m];

    gmres_W = new ParOptVec *[m + 1];
    for (int i = 0; i < m + 1; i++) {
      gmres_W[i] = prob->createDesignVec();
      gmres_W[i]->incref();
    }
  } else {
    gmres_subspace_size = 0;
  }
}

/**
   Set the optimization history file name to use.

   Note that the file is only opened on the root processor.

   @param filename the output file name
*/
void ParOptInteriorPoint::setOutputFile(const char *filename) {
  if (outfp && outfp != stdout) {
    fclose(outfp);
  }
  outfp = NULL;

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (filename && rank == opt_root) {
    outfp = fopen(filename, "w");
  }
}

/**
   Compute the residual of the KKT system. This code utilizes the data
   stored internally in the ParOpt optimizer.

   This code computes the following terms:

   Residuals for the design variables and bounds
   rx  = -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu)
   rzu = -((x - xl)*zl - mu*e)
   rzl = -((ub - x)*zu - mu*e)

   // These residuals are repeated for both the dense and sparse constraints
   rs  = -(penalty_gamma_s - zs + z)
   rt  = -(penalty_gamma_t - zt - z)
   rz  = -(c(x) - s + t)
   rzs = -(S*zs - mu*e)
   rzt = -(T*zt - mu*e)
*/
void ParOptInteriorPoint::computeKKTRes(ParOptVars &vars, double barrier,
                                        ParOptVars &res,
                                        ParOptNormType norm_type,
                                        double *max_prime, double *max_dual,
                                        double *max_infeas, double *res_norm) {
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Zero the values of the maximum residuals
  *max_prime = 0.0;
  *max_dual = 0.0;
  *max_infeas = 0.0;

  // Assemble the negative of the residual of the first KKT equation:
  // -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu)
  if (use_lower) {
    res.x->copyValues(vars.zl);
  } else {
    res.x->zeroEntries();
  }
  if (use_upper) {
    res.x->axpy(-1.0, vars.zu);
  }
  res.x->axpy(-1.0, g);

  for (int i = 0; i < ncon; i++) {
    res.x->axpy(vars.z[i], Ac[i]);
  }

  if (nwcon > 0) {
    // Add rx = rx + Aw^{T}*zw
    prob->addSparseJacobianTranspose(1.0, vars.x, vars.zw, res.x);

    // Compute the residuals from the weighting constraints
    // res.zw = -(cw(x) - vars.sw + vars.tw)
    prob->evalSparseCon(vars.x, res.zw);
    res.zw->axpy(-1.0, vars.sw);
    res.zw->axpy(1.0, vars.tw);
    res.zw->scale(-1.0);

    // res.sw = -(penalty_gamma_sw - vars.zsw + vars.zw)
    res.sw->copyValues(vars.zsw);
    res.sw->axpy(-1.0, penalty_gamma_sw);
    res.sw->axpy(-1.0, vars.zw);

    // res.tw = -(penalty_gamma_tw - vars.ztw - vars.zw)
    res.tw->copyValues(vars.ztw);
    res.tw->axpy(-1.0, penalty_gamma_tw);
    res.tw->axpy(1.0, vars.zw);

    // Set the values of the perturbed complementarity
    // constraints for the sparse slack variables
    ParOptScalar *swvals, *twvals, *zswvals, *ztwvals;
    vars.sw->getArray(&swvals);
    vars.tw->getArray(&twvals);
    vars.zsw->getArray(&zswvals);
    vars.ztw->getArray(&ztwvals);

    // Get the residuals
    ParOptScalar *rzswvals, *rztwvals;
    res.zsw->getArray(&rzswvals);
    res.ztw->getArray(&rztwvals);

    // res.zsw = -(vars.sw * vars.zsw - barrier)
    // res.ztw = -(vars.tw * vars.ztw - barrier)
    for (int i = 0; i < nwcon; i++) {
      rzswvals[i] = barrier - swvals[i] * zswvals[i];
      rztwvals[i] = barrier - twvals[i] * ztwvals[i];
    }
  }

  // Compute the residuals in the KKT condition
  // Account for the contributions to the norms
  if (norm_type == PAROPT_INFTY_NORM) {
    *max_prime = res.x->maxabs();
    *max_infeas = res.zw->maxabs();

    double dual_sw = res.sw->maxabs();
    double dual_tw = res.tw->maxabs();
    double dual_zsw = res.zsw->maxabs();
    double dual_ztw = res.ztw->maxabs();
    if (dual_sw > *max_dual) {
      *max_dual = dual_sw;
    }
    if (dual_tw > *max_dual) {
      *max_dual = dual_tw;
    }
    if (dual_zsw > *max_dual) {
      *max_dual = dual_zsw;
    }
    if (dual_ztw > *max_dual) {
      *max_dual = dual_ztw;
    }
  } else if (norm_type == PAROPT_L1_NORM) {
    *max_prime = res.x->l1norm();
    *max_infeas = res.zw->l1norm();

    *max_dual += res.sw->l1norm();
    *max_dual += res.tw->l1norm();
    *max_dual += res.zsw->l1norm();
    *max_dual += res.ztw->l1norm();
  } else {  // norm_type == PAROPT_L2_NORM
    double prime_rx = res.x->norm();
    double prime_rzw = res.zw->norm();
    *max_prime = prime_rx * prime_rx;
    *max_infeas = prime_rzw * prime_rzw;

    double dual_sw = res.sw->l1norm();
    double dual_tw = res.tw->l1norm();
    double dual_zsw = res.zsw->l1norm();
    double dual_ztw = res.ztw->l1norm();
    *max_dual += (dual_sw * dual_sw + dual_tw * dual_tw + dual_zsw * dual_zsw +
                  dual_ztw * dual_ztw);
  }

  // Evaluate the residuals differently depending on whether
  // we're using a dense equality or inequality constraint
  for (int i = 0; i < ncon; i++) {
    res.z[i] = -(c[i] - vars.s[i] + vars.t[i]);
    res.s[i] = -(penalty_gamma_s[i] - vars.zs[i] + vars.z[i]);
    res.t[i] = -(penalty_gamma_t[i] - vars.zt[i] - vars.z[i]);
    res.zs[i] = -(vars.s[i] * vars.zs[i] - barrier);
    res.zt[i] = -(vars.t[i] * vars.zt[i] - barrier);
  }

  if (norm_type == PAROPT_INFTY_NORM) {
    for (int i = 0; i < ncon; i++) {
      if (fabs(ParOptRealPart(res.s[i])) > *max_prime) {
        *max_prime = fabs(ParOptRealPart(res.s[i]));
      }
      if (fabs(ParOptRealPart(res.t[i])) > *max_prime) {
        *max_prime = fabs(ParOptRealPart(res.t[i]));
      }
      if (fabs(ParOptRealPart(res.z[i])) > *max_infeas) {
        *max_infeas = fabs(ParOptRealPart(res.z[i]));
      }
      if (fabs(ParOptRealPart(res.zs[i])) > *max_dual) {
        *max_dual = fabs(ParOptRealPart(res.zs[i]));
      }
      if (fabs(ParOptRealPart(res.zt[i])) > *max_dual) {
        *max_dual = fabs(ParOptRealPart(res.zt[i]));
      }
    }
  } else if (norm_type == PAROPT_L1_NORM) {
    for (int i = 0; i < ncon; i++) {
      *max_prime += fabs(ParOptRealPart(res.s[i]));
      *max_prime += fabs(ParOptRealPart(res.t[i]));
      *max_infeas += fabs(ParOptRealPart(res.z[i]));
      *max_dual += fabs(ParOptRealPart(res.zs[i]));
      *max_dual += fabs(ParOptRealPart(res.zt[i]));
    }
  } else {  // norm_type == PAROPT_L2_NORM
    double prime = 0.0, infeas = 0.0, dual = 0.0;
    for (int i = 0; i < ncon; i++) {
      prime += ParOptRealPart(res.s[i] * res.s[i] + res.t[i] * res.t[i]);
      infeas += ParOptRealPart(res.z[i] * res.z[i]);
      dual += ParOptRealPart(res.zs[i] * res.zs[i] + res.zt[i] * res.zt[i]);
    }
    *max_prime += prime;
    *max_infeas += infeas;
    *max_dual += dual;
  }

  // Extract the values of the variables and lower/upper bounds
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  if (use_lower) {
    // Compute the residuals for the lower bounds
    ParOptScalar *rzlvals;
    res.zl->getArray(&rzlvals);

    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        rzlvals[i] =
            -((xvals[i] - lbvals[i]) * zlvals[i] - rel_bound_barrier * barrier);
      } else {
        rzlvals[i] = 0.0;
      }
    }

    if (norm_type == PAROPT_INFTY_NORM) {
      double dual_zl = res.zl->maxabs();
      if (dual_zl > *max_dual) {
        *max_dual = dual_zl;
      }
    } else if (norm_type == PAROPT_L1_NORM) {
      *max_dual += res.zl->l1norm();
    } else {  // norm_type == PAROPT_L2_NORM
      double dual_zl = res.zl->norm();
      *max_dual += dual_zl * dual_zl;
    }
  }
  if (use_upper) {
    // Compute the residuals for the upper bounds
    ParOptScalar *rzuvals;
    res.zu->getArray(&rzuvals);

    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        rzuvals[i] =
            -((ubvals[i] - xvals[i]) * zuvals[i] - rel_bound_barrier * barrier);
      } else {
        rzuvals[i] = 0.0;
      }
    }

    if (norm_type == PAROPT_INFTY_NORM) {
      double dual_zu = res.zu->maxabs();
      if (ParOptRealPart(dual_zu) > ParOptRealPart(*max_dual)) {
        *max_dual = dual_zu;
      }
    } else if (norm_type == PAROPT_L1_NORM) {
      *max_dual += res.zu->l1norm();
    } else {  // norm_type == PAROPT_L2_NORM
      double dual_zu = res.zu->norm();
      *max_dual += dual_zu * dual_zu;
    }
  }

  // If this is the l2 norm, take the square root
  if (norm_type == PAROPT_L2_NORM) {
    *max_dual = sqrt(*max_dual);
    *max_prime = sqrt(*max_prime);
    *max_infeas = sqrt(*max_infeas);
  }

  // Compute the max norm
  if (res_norm) {
    *res_norm = *max_prime;
    if (*max_dual > *res_norm) {
      *res_norm = *max_dual;
    }
    if (*max_infeas > *res_norm) {
      *res_norm = *max_infeas;
    }
  }
}

/*
  Add the contributions to the residual from the affine predictor
  step due to the Mehrotra predictor-corrector
*/
void ParOptInteriorPoint::addMehrotraCorrectorResidual(ParOptVars &step,
                                                       ParOptVars &res) {
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Add the contribution from the sparse constraints
  if (nwcon > 0) {
    ParOptScalar *swvals, *twvals, *zswvals, *ztwvals;
    step.sw->getArray(&swvals);
    step.tw->getArray(&twvals);
    step.zsw->getArray(&zswvals);
    step.ztw->getArray(&ztwvals);

    // Get the residuals
    ParOptScalar *rzswvals, *rztwvals;
    res.zsw->getArray(&rzswvals);
    res.ztw->getArray(&rztwvals);

    for (int i = 0; i < nwcon; i++) {
      rzswvals[i] -= swvals[i] * zswvals[i];
      rztwvals[i] -= twvals[i] * ztwvals[i];
    }
  }

  // Add the contributions from the dense constraints
  for (int i = 0; i < ncon; i++) {
    res.zs[i] -= step.s[i] * step.zs[i];
    res.zt[i] -= step.t[i] * step.zt[i];
  }

  // Extract the values of the variables and lower/upper bounds
  ParOptScalar *pxvals, *lbvals, *ubvals, *pzlvals, *pzuvals;
  step.x->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  step.zl->getArray(&pzlvals);
  step.zu->getArray(&pzuvals);

  if (use_lower) {
    // Compute the residuals for the lower bounds
    ParOptScalar *rzlvals;
    res.zl->getArray(&rzlvals);

    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        rzlvals[i] -= pxvals[i] * pzlvals[i];
      }
    }
  }

  if (use_upper) {
    // Compute the residuals for the upper bounds
    ParOptScalar *rzuvals;
    res.zu->getArray(&rzuvals);

    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        rzuvals[i] += pxvals[i] * pzuvals[i];
      }
    }
  }
}

/*
  Compute the maximum norm of the step
*/
double ParOptInteriorPoint::computeStepNorm(ParOptNormType norm_type,
                                            ParOptVars &step) {
  double step_norm = 0.0;
  if (norm_type == PAROPT_INFTY_NORM) {
    step_norm = step.x->maxabs();
  } else if (norm_type == PAROPT_L1_NORM) {
    step_norm = step.x->l1norm();
  } else {  // if (norm_type == PAROPT_L2_NORM)
    step_norm = step.x->norm();
  }
  return step_norm;
}

/*
  This function computes the terms required to solve the KKT system
  using a bordering method.  The initialization process computes the
  following matrix:

  Dinv = [b0 + zl/(x - lb) + zu/(ub - x)]^{-1}

  where D is a diagonal matrix. The components of D^{-1} (also a
  diagonal matrix) are stored in Dvec.

  Next, we compute:

  Cdiag = Zw^{-1} * Sw

  This is then used to construct the quasi-definite matrix D0 given by

  D0 = [ Dinv^{-1}   Aw^{T} ]
  .    [ Aw          -Cdiag ]

  Finally, the code computes a factorization of the matrix:

  G = (Zs^{-1}*S + Zt^{-1}*T) + (A, 0) * D0^{-1} * (A, 0)^{T}

  which is required to compute the solution of the KKT step.
*/
void ParOptInteriorPoint::setUpKKTDiagSystem(ParOptVars &vars, ParOptVec *xtmp,
                                             ParOptVec *wtmp, int use_qn) {
  // Diagonal coefficient used for the quasi-Newton Hessian aprpoximation
  const double qn_sigma = options->getFloatOption("qn_sigma");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");

  // Retrive the diagonal entry for the BFGS update
  ParOptScalar b0 = 0.0;
  ParOptScalar *h = NULL;
  if (hdiag && use_diag_hessian) {
    hdiag->getArray(&h);
  } else if (qn && use_qn) {
    const ParOptScalar *d, *M;
    ParOptVec **Z;
    qn->getCompactMat(&b0, &d, &M, &Z);
  }

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Set the components of the diagonal matrix
  ParOptScalar *dvals;
  Dinv->getArray(&dvals);

  // Set the values of the c matrix
  if (use_lower && use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (h) {
        b0 = h[i];
      }
      if (ParOptRealPart(lbvals[i]) > -max_bound_value &&
          ParOptRealPart(ubvals[i]) < max_bound_value) {
        dvals[i] = 1.0 / (b0 + qn_sigma + zlvals[i] / (xvals[i] - lbvals[i]) +
                          zuvals[i] / (ubvals[i] - xvals[i]));
      } else if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        dvals[i] = 1.0 / (b0 + qn_sigma + zlvals[i] / (xvals[i] - lbvals[i]));
      } else if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        dvals[i] = 1.0 / (b0 + qn_sigma + zuvals[i] / (ubvals[i] - xvals[i]));
      } else {
        dvals[i] = 1.0 / (b0 + qn_sigma);
      }
    }
  } else if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (h) {
        b0 = h[i];
      }
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        dvals[i] = 1.0 / (b0 + qn_sigma + zlvals[i] / (xvals[i] - lbvals[i]));
      } else {
        dvals[i] = 1.0 / (b0 + qn_sigma);
      }
    }
  } else if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (h) {
        b0 = h[i];
      }
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        dvals[i] = 1.0 / (b0 + qn_sigma + zuvals[i] / (ubvals[i] - xvals[i]));
      } else {
        dvals[i] = 1.0 / (b0 + qn_sigma);
      }
    }
  } else {
    for (int i = 0; i < nvars; i++) {
      if (h) {
        b0 = h[i];
      }
      dvals[i] = 1.0 / (b0 + qn_sigma);
    }
  }

  Cdiag->zeroEntries();
  if (nwcon > 0) {
    // Compute C = Zsw^{-1} * Sw + Ztw^{-1} * Tw
    ParOptScalar *swvals, *twvals, *zswvals, *ztwvals;
    vars.sw->getArray(&swvals);
    vars.tw->getArray(&twvals);
    vars.zsw->getArray(&zswvals);
    vars.ztw->getArray(&ztwvals);

    ParOptScalar *cvals;
    Cdiag->getArray(&cvals);

    for (int i = 0; i < nwcon; i++) {
      cvals[i] = swvals[i] / zswvals[i] + twvals[i] / ztwvals[i];
    }
  }

  // Factor the quasi-definite matrix
  mat->factor(vars.x, Dinv, Cdiag);

  // Set the value of the G matrix
  memset(Gmat, 0, ncon * ncon * sizeof(ParOptScalar));

  // Now, compute the Schur complement with the Dmatrix
  for (int j = 0; j < ncon; j++) {
    mat->apply(Ac[j], xtmp, wtmp);

    for (int i = j; i < ncon; i++) {
      Gmat[i + ncon * j] += Ac[i]->dot(xtmp);
    }
  }

  // Populate the remainder of the matrix because it is
  // symmetric
  for (int j = 0; j < ncon; j++) {
    for (int i = j + 1; i < ncon; i++) {
      Gmat[j + ncon * i] = Gmat[i + ncon * j];
    }
  }

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Add the diagonal component to the matrix
    if (rank == opt_root) {
      for (int i = 0; i < ncon; i++) {
        Gmat[i * (ncon + 1)] += vars.s[i] / vars.zs[i] + vars.t[i] / vars.zt[i];
      }
    }

    // Broadcast the result to all processors. Note that this ensures
    // that the factorization will be the same on all processors
    MPI_Bcast(Gmat, ncon * ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Factor the matrix for future use
    int info = 0;
    LAPACKdgetrf(&ncon, &ncon, Gmat, &ncon, gpiv, &info);
  }
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms. The system of
  equations consists of the following terms:

  (1) B0*yx - A^{T}*yz - Aw^{T}*yzw - yzl + yzu = bx
  (2) A*yx - ys + yt = bz
  (3) Aw*yx - ysw + ytw = bzw

  With the additional equations

  -(X - Xl) * yzl + Zl * yx = bzl
  -(Xu - X) * yzu - Zu * yx = bzl

  Substitution of these two equations into (1) yields the following system of
  equations:

  ((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))*yx - A^{T}*yz - Aw^{T}*yzw
  = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu

  which we rewrite as the equation:

  (4) D * yx - A^{T} * yz - Aw^{T} * yzw = d1

  where D = ((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))
  and d1 = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu.

  The slack and slack multipliers satisfy the following relationships

  -yzs + yz = bs
  -yzt - yz = bt
  S * yzs + Zs * ys = bzs
  T * yzt + Zt * yt = bzt

  Rearranging the first two equations gives

  yzs = yz - bs
  yzt = -(yz + bt)

  and substituting them into the second two equations gives

  S * (yz - bs) + Zs * ys = bzs
  - T * (yz + bt) + Zt * yt = bzt

  so that

  ys = -Zs^{-1}*S * yz + Zs^{-1} * (bzs + S * bs)
  yt = Zt^{-1}*T * yz + Zt^{-1} * (bzt + T * bt)

  Substitution of these equations into (2) gives the result

  A * yx + (Zs^{-1} * S + Zt^{-1}*T) * yz =
    bz + Zs^{-1} * (bzs + S * bs) - Zt^{-1} * (bzt + T * bt)

  We rewrite this equation as

  A * yx + C0 * yz = d3

  where C0 = (Z^{-1}*S + Zt^{-1}*T)
  and d3 = bz + Zs^{-1}*(bzs + S*bs) - Zt^{-1}*(bzt + T*bt).

  Similarly, the final equation (3) gives

  Aw * yx + C * yzw = d2

  where C = Zsw^{-1} * Sw + Ztw^{-1} * Tw and
  d2 = bzw + Zsw^{-1} * (bzsw + Sw * bsw) - Ztw^{-1} * (bztw + Tw * btw)

  We can solve for yx, yzw and yz by solving the following system of equations:

  [[ D   Aw^{T} ]  A^{T} ][  yx  ] = [ d1 ]
  [[ Aw     -C  ]        ][ -yzw ] = [ d2 ]
  [  A               -C0 ][ -yz  ] = [ d3 ]

  We solve this via a Schur complement on the yz variables. Writing the matrix

  D0 =
  [ D   Aw^{T} ]
  [ Aw     -C  ]

  We compute

  G = C0 + (A, 0) * D0^{-1} * (A^{T}, 0)

  Then solve

  G * yz = d3 - (A, 0) * D0^{-1} (d1, d2))

  We can then compute yx and yzw via

  [ D   Aw^{T} ][  yx  ] = [ d1 + A^{T} * yz ]
  [ Aw     -C  ][ -yzw ] = [ d2              ]

  Note: This code uses the temporary arrays xt and wt which therefore
  cannot be inputs/outputs for this function, otherwise strange
  behavior will occur.
*/
void ParOptInteriorPoint::solveKKTDiagSystem(ParOptVars &vars, ParOptVars &b,
                                             ParOptVars &y, ParOptVec *d1,
                                             ParOptVec *d2) {
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the arrays for the variables and upper/lower bounds
  ParOptScalar *xvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Get the arrays for the right-hand-sides
  ParOptScalar *bzlvals, *bzuvals;
  b.zl->getArray(&bzlvals);
  b.zu->getArray(&bzuvals);

  // Compute d1 = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu
  d1->copyValues(b.x);

  ParOptScalar *d1vals;
  d1->getArray(&d1vals);
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        d1vals[i] += (bzlvals[i] / (xvals[i] - lbvals[i]));
      }
    }
  }
  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        d1vals[i] -= (bzuvals[i] / (ubvals[i] - xvals[i]));
      }
    }
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0) {
    // Compute d2 =
    // = bzw + Zsw^{-1} * (bzsw + Sw * bsw) - Ztw^{-1} * (bztw + Tw * btw)
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    // Right-hand-sides
    ParOptScalar *bzw, *bsw, *btw, *bzsw, *bztw;
    b.zw->getArray(&bzw);
    b.sw->getArray(&bsw);
    b.tw->getArray(&btw);
    b.zsw->getArray(&bzsw);
    b.ztw->getArray(&bztw);

    // Get the right-hand-side that will be modified
    ParOptScalar *d2vals;
    d2->getArray(&d2vals);

    for (int i = 0; i < nwcon; i++) {
      d2vals[i] = bzw[i] + (bzsw[i] + sw[i] * bsw[i]) / zsw[i] -
                  (bztw[i] + tw[i] * btw[i]) / ztw[i];
    }
  }

  // Solve for the update and store in y.x
  mat->apply(d1, d2, y.x, y.zw);

  // Now, compute yz = A^{T} * y.x
  memset(y.z, 0, ncon * sizeof(ParOptScalar));
  y.x->mdot(Ac, ncon, y.z);

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Compute the full right-hand-side on the root proc
    if (rank == opt_root) {
      // Compute the full right-hand-side on the root processor
      // and solve for the Lagrange multipliers
      for (int i = 0; i < ncon; i++) {
        y.z[i] = (b.z[i] + (b.zs[i] + vars.s[i] * b.s[i]) / vars.zs[i] -
                  (b.zt[i] + vars.t[i] * b.t[i]) / vars.zt[i] - y.z[i]);
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one, Gmat, &ncon, gpiv, y.z, &ncon, &info);
    }

    MPI_Bcast(y.z, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Compute the step in the slack variables
    for (int i = 0; i < ncon; i++) {
      y.zs[i] = y.z[i] - b.s[i];
      y.zt[i] = -b.t[i] - y.z[i];
      y.s[i] = (b.zs[i] - vars.s[i] * y.zs[i]) / vars.zs[i];
      y.t[i] = (b.zt[i] - vars.t[i] * y.zt[i]) / vars.zt[i];
    }
  }

  for (int i = 0; i < ncon; i++) {
    d1->axpy(y.z[i], Ac[i]);
  }

  mat->apply(d1, d2, y.x, y.zw);

  // Compute the updates to the sparse constraints
  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    // Right-hand-sides
    ParOptScalar *bsw, *btw, *bzsw, *bztw;
    b.sw->getArray(&bsw);
    b.tw->getArray(&btw);
    b.zsw->getArray(&bzsw);
    b.ztw->getArray(&bztw);

    // Updates
    ParOptScalar *yzw, *ysw, *ytw, *yzsw, *yztw;
    y.zw->getArray(&yzw);
    y.sw->getArray(&ysw);
    y.tw->getArray(&ytw);
    y.zsw->getArray(&yzsw);
    y.ztw->getArray(&yztw);

    for (int i = 0; i < nwcon; i++) {
      yzsw[i] = yzw[i] - bsw[i];
      yztw[i] = -btw[i] - yzw[i];
      ysw[i] = (bzsw[i] - sw[i] * yzsw[i]) / zsw[i];
      ytw[i] = (bztw[i] - tw[i] * yztw[i]) / ztw[i];
    }
  }

  // Retrieve the lagrange multipliers
  ParOptScalar *zlvals, *zuvals;
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Retrieve the lagrange multiplier update vectors
  ParOptScalar *yxvals, *yzlvals, *yzuvals;
  y.x->getArray(&yxvals);
  y.zl->getArray(&yzlvals);
  y.zu->getArray(&yzuvals);

  // Compute the steps in the bound Lagrange multipliers
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        yzlvals[i] =
            (bzlvals[i] - zlvals[i] * yxvals[i]) / (xvals[i] - lbvals[i]);
      } else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        yzuvals[i] =
            (bzuvals[i] + zuvals[i] * yxvals[i]) / (ubvals[i] - xvals[i]);
      } else {
        yzuvals[i] = 0.0;
      }
    }
  }
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate Hessian
  is replaced with only the diagonal terms.

  In this case, we assume that the only non-zero input components correspond to
  the the unknowns in the first equation of the  KKT system. This is the case
  when solving systems used with the limited-memory BFGS approximation.
*/
void ParOptInteriorPoint::solveKKTDiagSystem(ParOptVars &vars, ParOptVec *bx,
                                             ParOptVars &y, ParOptVec *d1,
                                             ParOptVec *d2) {
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the arrays for the variables and upper/lower bounds
  ParOptScalar *xvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Compute d1 = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu
  d1->copyValues(bx);

  // Compute the terms from the weighting constraints
  d2->zeroEntries();

  // Solve for the update and store in y.x
  mat->apply(d1, d2, y.x, y.zw);

  // Now, compute yz = A^{T} * y.x
  memset(y.z, 0, ncon * sizeof(ParOptScalar));
  y.x->mdot(Ac, ncon, y.z);

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Compute the full right-hand-side on the root proc
    if (rank == opt_root) {
      // Compute the full right-hand-side on the root processor
      // and solve for the Lagrange multipliers
      for (int i = 0; i < ncon; i++) {
        y.z[i] = -y.z[i];
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one, Gmat, &ncon, gpiv, y.z, &ncon, &info);
    }

    MPI_Bcast(y.z, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Compute the step in the slack variables
    for (int i = 0; i < ncon; i++) {
      y.zs[i] = y.z[i];
      y.zt[i] = -y.z[i];
      y.s[i] = -(vars.s[i] * y.zs[i]) / vars.zs[i];
      y.t[i] = -(vars.t[i] * y.zt[i]) / vars.zt[i];
    }
  }

  for (int i = 0; i < ncon; i++) {
    d1->axpy(y.z[i], Ac[i]);
  }

  mat->apply(d1, d2, y.x, y.zw);

  // Compute the updates to the sparse constraints
  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    // Updates
    ParOptScalar *yzw, *ysw, *ytw, *yzsw, *yztw;
    y.zw->getArray(&yzw);
    y.sw->getArray(&ysw);
    y.tw->getArray(&ytw);
    y.zsw->getArray(&yzsw);
    y.ztw->getArray(&yztw);

    for (int i = 0; i < nwcon; i++) {
      yzsw[i] = yzw[i];
      yztw[i] = -yzw[i];
      ysw[i] = -(sw[i] * yzsw[i]) / zsw[i];
      ytw[i] = -(tw[i] * yztw[i]) / ztw[i];
    }
  }

  // Retrieve the lagrange multipliers
  ParOptScalar *zlvals, *zuvals;
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Retrieve the lagrange multiplier update vectors
  ParOptScalar *yxvals, *yzlvals, *yzuvals;
  y.x->getArray(&yxvals);
  y.zl->getArray(&yzlvals);
  y.zu->getArray(&yzuvals);

  // Compute the steps in the bound Lagrange multipliers
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        yzlvals[i] = -(zlvals[i] * yxvals[i]) / (xvals[i] - lbvals[i]);
      } else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        yzuvals[i] = (zuvals[i] * yxvals[i]) / (ubvals[i] - xvals[i]);
      } else {
        yzuvals[i] = 0.0;
      }
    }
  }
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate Hessian
  is replaced with only the diagonal terms.

  In this case, we assume that the only non-zero input components correspond to
  the the unknowns in the first equation of the KKT system. And in addition, the
  only relevant output components are the x-components of the output. This is
  required for setting up the Schur complement with the limited-memory BFGS
  approximation.
*/
void ParOptInteriorPoint::solveKKTDiagSystem(ParOptVars &vars, ParOptVec *bx,
                                             ParOptVec *yx, ParOptScalar *yz,
                                             ParOptVec *d1, ParOptVec *yzw) {
  // Get the arrays for the variables and upper/lower bounds
  ParOptScalar *xvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Compute d1 = bx
  d1->copyValues(bx);

  // Solve for the update and store in y.x
  mat->apply(d1, yx, yzw);

  // Now, compute yz = A^{T} * y.x
  memset(yz, 0, ncon * sizeof(ParOptScalar));
  yx->mdot(Ac, ncon, yz);

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Compute the full right-hand-side on the root proc
    if (rank == opt_root) {
      // Compute the full right-hand-side on the root processor
      // and solve for the Lagrange multipliers
      for (int i = 0; i < ncon; i++) {
        yz[i] = -yz[i];
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one, Gmat, &ncon, gpiv, yz, &ncon, &info);
    }

    MPI_Bcast(yz, ncon, PAROPT_MPI_TYPE, opt_root, comm);
  }

  for (int i = 0; i < ncon; i++) {
    d1->axpy(yz[i], Ac[i]);
  }

  mat->apply(d1, yx, yzw);
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms.

  Note that in this variant of the function, the right-hand-side
  includes components that are scaled by a given alpha-parameter.
*/
void ParOptInteriorPoint::solveKKTDiagSystem(ParOptVars &vars, ParOptVec *bx,
                                             ParOptScalar alpha, ParOptVars &b,
                                             ParOptVars &y, ParOptVec *d1,
                                             ParOptVec *d2) {
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the arrays for the variables and upper/lower bounds
  ParOptScalar *xvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Get the arrays for the right-hand-sides
  ParOptScalar *bzlvals, *bzuvals;
  b.zl->getArray(&bzlvals);
  b.zu->getArray(&bzuvals);

  // Get the right-hand-side for the design variables from bx
  // Compute d1 = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu
  d1->copyValues(bx);

  ParOptScalar *d1vals;
  d1->getArray(&d1vals);
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        d1vals[i] += alpha * (bzlvals[i] / (xvals[i] - lbvals[i]));
      }
    }
  }
  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        d1vals[i] -= alpha * (bzuvals[i] / (ubvals[i] - xvals[i]));
      }
    }
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0) {
    // Compute d2 =
    // = bzw + Zsw^{-1} * (bzsw + Sw * bsw) - Ztw^{-1} * (bztw + Tw * btw)
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    // Right-hand-sides
    ParOptScalar *bzw, *bsw, *btw, *bzsw, *bztw;
    b.zw->getArray(&bzw);
    b.sw->getArray(&bsw);
    b.tw->getArray(&btw);
    b.zsw->getArray(&bzsw);
    b.ztw->getArray(&bztw);

    // Get the right-hand-side that will be modified
    ParOptScalar *d2vals;
    d2->getArray(&d2vals);

    for (int i = 0; i < nwcon; i++) {
      d2vals[i] = alpha * (bzw[i] + (bzsw[i] + sw[i] * bsw[i]) / zsw[i] -
                           (bztw[i] + tw[i] * btw[i]) / ztw[i]);
    }
  }

  // Solve for the update and store in y.x
  mat->apply(d1, d2, y.x, y.zw);

  // Now, compute yz = A^{T} * y.x
  memset(y.z, 0, ncon * sizeof(ParOptScalar));
  y.x->mdot(Ac, ncon, y.z);

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Compute the full right-hand-side on the root proc
    if (rank == opt_root) {
      // Compute the full right-hand-side on the root processor
      // and solve for the Lagrange multipliers
      for (int i = 0; i < ncon; i++) {
        y.z[i] =
            (alpha * (b.z[i] + (b.zs[i] + vars.s[i] * b.s[i]) / vars.zs[i] -
                      (b.zt[i] + vars.t[i] * b.t[i]) / vars.zt[i]) -
             y.z[i]);
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one, Gmat, &ncon, gpiv, y.z, &ncon, &info);
    }

    MPI_Bcast(y.z, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Compute the step in the slack variables
    for (int i = 0; i < ncon; i++) {
      y.zs[i] = y.z[i] - alpha * b.s[i];
      y.zt[i] = -alpha * b.t[i] - y.z[i];
      y.s[i] = (alpha * b.zs[i] - vars.s[i] * y.zs[i]) / vars.zs[i];
      y.t[i] = (alpha * b.zt[i] - vars.t[i] * y.zt[i]) / vars.zt[i];
    }
  }

  for (int i = 0; i < ncon; i++) {
    d1->axpy(y.z[i], Ac[i]);
  }

  mat->apply(d1, d2, y.x, y.zw);

  // Compute the updates to the sparse constraints
  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    // Right-hand-sides
    ParOptScalar *bsw, *btw, *bzsw, *bztw;
    b.sw->getArray(&bsw);
    b.tw->getArray(&btw);
    b.zsw->getArray(&bzsw);
    b.ztw->getArray(&bztw);

    // Updates
    ParOptScalar *yzw, *ysw, *ytw, *yzsw, *yztw;
    y.zw->getArray(&yzw);
    y.sw->getArray(&ysw);
    y.tw->getArray(&ytw);
    y.zsw->getArray(&yzsw);
    y.ztw->getArray(&yztw);

    for (int i = 0; i < nwcon; i++) {
      yzsw[i] = yzw[i] - alpha * bsw[i];
      yztw[i] = -alpha * btw[i] - yzw[i];
      ysw[i] = (alpha * bzsw[i] - sw[i] * yzsw[i]) / zsw[i];
      ytw[i] = (alpha * bztw[i] - tw[i] * yztw[i]) / ztw[i];
    }
  }

  // Retrieve the lagrange multipliers
  ParOptScalar *zlvals, *zuvals;
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Retrieve the lagrange multiplier update vectors
  ParOptScalar *yxvals, *yzlvals, *yzuvals;
  y.x->getArray(&yxvals);
  y.zl->getArray(&yzlvals);
  y.zu->getArray(&yzuvals);

  // Compute the steps in the bound Lagrange multipliers
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        yzlvals[i] = (alpha * bzlvals[i] - zlvals[i] * yxvals[i]) /
                     (xvals[i] - lbvals[i]);
      } else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        yzuvals[i] = (alpha * bzuvals[i] + zuvals[i] * yxvals[i]) /
                     (ubvals[i] - xvals[i]);
      } else {
        yzuvals[i] = 0.0;
      }
    }
  }
}

/*
  This code computes terms required for the solution of the KKT system
  of equations. The KKT system takes the form:

  K - Z*diag{d}*M^{-1}*diag{d}*Z^{T}

  where the Z*M*Z^{T} contribution arises from the limited memory BFGS
  approximation. The K matrix are the linear/diagonal terms from the
  linearization of the KKT system.

  This code computes the factorization of the Ce matrix which is given
  by:

  Ce = Z^{T}*K^{-1}*Z - diag{d}^{-1}*M*diag{d}^{-1}

  Note that Z only has contributions in components corresponding to
  the design variables.
*/
void ParOptInteriorPoint::setUpKKTSystem(ParOptVars &vars, ParOptScalar *ztmp,
                                         ParOptVec *xtmp1, ParOptVec *xtmp2,
                                         ParOptVec *wtmp, int use_qn) {
  if (qn && use_qn) {
    // Get the size of the limited-memory BFGS subspace
    ParOptScalar b0;
    const ParOptScalar *d0, *M;
    ParOptVec **Z;
    int size = qn->getCompactMat(&b0, &d0, &M, &Z);

    if (size > 0) {
      memset(Ce, 0, size * size * sizeof(ParOptScalar));

      // Solve the KKT system
      for (int i = 0; i < size; i++) {
        // Compute K^{-1}*Z[i]
        solveKKTDiagSystem(vars, Z[i], xtmp1, ztmp, xtmp2, wtmp);

        // Compute the dot products Z^{T}*K^{-1}*Z[i]
        xtmp1->mdot(Z, size, &Ce[i * size]);
      }

      // Compute the Schur complement
      for (int j = 0; j < size; j++) {
        for (int i = 0; i < size; i++) {
          Ce[i + j * size] -= M[i + j * size] / (d0[i] * d0[j]);
        }
      }

      int info = 0;
      LAPACKdgetrf(&size, &size, Ce, &size, cpiv, &info);
    }
  }
}

/*
  Sovle the KKT system for the next step. This relies on the diagonal
  KKT system solver above and uses the information from the set up
  computation above. The KKT system with the limited memory BFGS update
  is written as follows:

  K + Z*diag{d}*M^{-1}*diag{d}*Z^{T}

  where K is the KKT matrix with the diagonal entries. (With I*b0 +
  Z*diag{d}*M^{-1}*diag{d}*Z0^{T} from the LBFGS Hessian.) This code
  computes:

  y <- [ K + Z*diag{d}*M^{-1}*diag{d}*Z^{T} ]^{-1}*x,

  which can be written in terms of the operations y <- K^{-1}*x and
  r <- Ce^{-1}*S. Where Ce is given by:

  Ce = Z^{T}*K^{-1}*Z - diag{d}^{-1}*M*diag{d}^{-1}

  The code computes the following:

  y <- K^{-1}*x - K^{-1}*Z*Ce^{-1}*Z^{T}*K^{-1}*x

  The code computes the following:

  1. p = K^{-1}*r
  2. ztemp = Z^{T}*p
  3. ztemp <- Ce^{-1}*ztemp
  4. rx = Z^{T}*ztemp
  5. p -= K^{-1}*rx
*/
void ParOptInteriorPoint::computeKKTStep(ParOptVars &vars, ParOptVars &res,
                                         ParOptVars &step, ParOptScalar *ztmp,
                                         ParOptVec *xtmp1, ParOptVec *xtmp2,
                                         ParOptVec *wtmp, int use_qn) {
  // Get the size of the limited-memory BFGS subspace
  ParOptScalar b0;
  const ParOptScalar *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (qn && use_qn) {
    size = qn->getCompactMat(&b0, &d, &M, &Z);
  }

  // After this point the residuals are no longer required.
  solveKKTDiagSystem(vars, res, step, xtmp1, wtmp);

  if (size > 0) {
    // dz = Z^{T}*px
    step.x->mdot(Z, size, ztmp);

    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one, Ce, &size, cpiv, ztmp, &size, &info);

    // Compute rx = Z^{T}*dz
    xtmp1->zeroEntries();
    for (int i = 0; i < size; i++) {
      xtmp1->axpy(ztmp[i], Z[i]);
    }

    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(vars, xtmp1, res, xtmp2, wtmp);

    // Add the final contributions
    step.x->axpy(-1.0, res.x);
    step.zl->axpy(-1.0, res.zl);
    step.zu->axpy(-1.0, res.zu);

    step.sw->axpy(-1.0, res.sw);
    step.tw->axpy(-1.0, res.tw);
    step.zw->axpy(-1.0, res.zw);
    step.zsw->axpy(-1.0, res.zsw);
    step.ztw->axpy(-1.0, res.ztw);

    // Add the terms from the slacks/multipliers
    for (int i = 0; i < ncon; i++) {
      step.z[i] -= res.z[i];
      step.s[i] -= res.s[i];
      step.t[i] -= res.t[i];
      step.zs[i] -= res.zs[i];
      step.zt[i] -= res.zt[i];
    }
  }
}

/*
  Compute the complementarity at the current solution
*/
ParOptScalar ParOptInteriorPoint::computeComp(ParOptVars &vars) {
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Sum up the complementarity from each individual processor
  ParOptScalar product = 0.0, sum = 0.0;

  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        product += zlvals[i] * (xvals[i] - lbvals[i]);
        sum += 1.0;
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        product += zuvals[i] * (ubvals[i] - xvals[i]);
        sum += 1.0;
      }
    }
  }

  // Modify the complementarity by the bound scalar factor
  product = product / rel_bound_barrier;

  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    for (int i = 0; i < nwcon; i++) {
      product += sw[i] * zsw[i] + tw[i] * ztw[i];
      sum += 2.0;
    }
  }

  // Add up the contributions from all processors
  ParOptScalar in[2], out[2];
  in[0] = product;
  in[1] = sum;
  MPI_Reduce(in, out, 2, PAROPT_MPI_TYPE, MPI_SUM, opt_root, comm);
  product = out[0];
  sum = out[1];

  // Compute the complementarity only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  ParOptScalar comp = 0.0;
  if (rank == opt_root) {
    for (int i = 0; i < ncon; i++) {
      product += vars.s[i] * vars.zs[i] + vars.t[i] * vars.zt[i];
      sum += 2.0;
    }

    if (sum != 0.0) {
      comp = product / sum;
    }
  }

  // Broadcast the result to all processors
  MPI_Bcast(&comp, 1, PAROPT_MPI_TYPE, opt_root, comm);

  return comp;
}

/*
  Compute the complementarity at the given step
*/
ParOptScalar ParOptInteriorPoint::computeCompStep(ParOptVars &vars,
                                                  double alpha_x,
                                                  double alpha_z,
                                                  ParOptVars &step) {
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Retrieve the values of the steps
  ParOptScalar *pxvals, *pzlvals, *pzuvals;
  step.x->getArray(&pxvals);
  step.zl->getArray(&pzlvals);
  step.zu->getArray(&pzuvals);

  // Sum up the complementarity from each individual processor
  ParOptScalar product = 0.0, sum = 0.0;
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        ParOptScalar xnew = xvals[i] + alpha_x * pxvals[i];
        product += (zlvals[i] + alpha_z * pzlvals[i]) * (xnew - lbvals[i]);
        sum += 1.0;
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        ParOptScalar xnew = xvals[i] + alpha_x * pxvals[i];
        product += (zuvals[i] + alpha_z * pzuvals[i]) * (ubvals[i] - xnew);
        sum += 1.0;
      }
    }
  }

  // Modify the complementarity by the bound scalar factor
  product = product / rel_bound_barrier;

  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    ParOptScalar *ysw, *ytw, *yzsw, *yztw;
    step.sw->getArray(&ysw);
    step.tw->getArray(&ytw);
    step.zsw->getArray(&yzsw);
    step.ztw->getArray(&yztw);

    for (int i = 0; i < nwcon; i++) {
      product += (sw[i] + alpha_x * ysw[i]) * (zsw[i] + alpha_z * yzsw[i]) +
                 (tw[i] + alpha_x * ytw[i]) * (ztw[i] + alpha_z * yztw[i]);
      sum += 2.0;
    }
  }

  // Add up the contributions from all processors
  ParOptScalar in[2], out[2];
  in[0] = product;
  in[1] = sum;
  MPI_Reduce(in, out, 2, PAROPT_MPI_TYPE, MPI_SUM, opt_root, comm);
  product = out[0];
  sum = out[1];

  // Compute the complementarity only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  ParOptScalar comp = 0.0;
  if (rank == opt_root) {
    for (int i = 0; i < ncon; i++) {
      product += ((vars.s[i] + alpha_x * step.s[i]) *
                      (vars.zs[i] + alpha_z * step.zs[i]) +
                  (vars.t[i] + alpha_x * step.t[i]) *
                      (vars.zt[i] + alpha_z * step.zt[i]));
      sum += 2.0;
    }

    if (sum != 0.0) {
      comp = product / sum;
    }
  }

  // Broadcast the result to all processors
  MPI_Bcast(&comp, 1, PAROPT_MPI_TYPE, opt_root, comm);

  return comp;
}

/*
  Compute the maximum step length along the given direction
  given the specified fraction to the boundary tau. This
  computes:

  The lower/upper bounds on x are enforced as follows:

  alpha =  tau*(ub - x)/px   px > 0
  alpha = -tau*(x - lb)/px   px < 0

  input:
  tau:   the fraction to the boundary

  output:
  max_x: the maximum step length in the design variables
  max_z: the maximum step in the lagrange multipliers
*/
void ParOptInteriorPoint::computeMaxStep(ParOptVars &vars, double tau,
                                         ParOptVars &step, double *_max_x,
                                         double *_max_z) {
  // Set the initial step length along the design and multiplier
  // directions
  double max_x = 1.0, max_z = 1.0;

  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  ParOptScalar *xvals, *pxvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  step.x->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Check the design variable step
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(pxvals[i]) < 0.0) {
        double numer = ParOptRealPart(xvals[i] - lbvals[i]);
        double alpha = -tau * numer / ParOptRealPart(pxvals[i]);
        if (alpha < max_x) {
          max_x = alpha;
        }
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(pxvals[i]) > 0.0) {
        double numer = ParOptRealPart(ubvals[i] - xvals[i]);
        double alpha = tau * numer / ParOptRealPart(pxvals[i]);
        if (alpha < max_x) {
          max_x = alpha;
        }
      }
    }
  }

  // Check the slack variable step
  for (int i = 0; i < ncon; i++) {
    if (ParOptRealPart(step.s[i]) < 0.0) {
      double numer = ParOptRealPart(vars.s[i]);
      double alpha = -tau * numer / ParOptRealPart(step.s[i]);
      if (alpha < max_x) {
        max_x = alpha;
      }
    }
    if (ParOptRealPart(step.t[i]) < 0.0) {
      double numer = ParOptRealPart(vars.t[i]);
      double alpha = -tau * numer / ParOptRealPart(step.t[i]);
      if (alpha < max_x) {
        max_x = alpha;
      }
    }
    // Check the step for the Lagrange multipliers
    if (ParOptRealPart(step.zs[i]) < 0.0) {
      double numer = ParOptRealPart(vars.zs[i]);
      double alpha = -tau * numer / ParOptRealPart(step.zs[i]);
      if (alpha < max_z) {
        max_z = alpha;
      }
    }
    if (ParOptRealPart(step.zt[i]) < 0.0) {
      double numer = ParOptRealPart(vars.zt[i]);
      double alpha = -tau * numer / ParOptRealPart(step.zt[i]);
      if (alpha < max_z) {
        max_z = alpha;
      }
    }
  }

  // Check the Lagrange and slack variable steps for the
  // sparse inequalities if any
  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *zsw, *ztw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    ParOptScalar *psw, *ptw, *pzsw, *pztw;
    step.sw->getArray(&psw);
    step.tw->getArray(&ptw);
    step.zsw->getArray(&pzsw);
    step.ztw->getArray(&pztw);

    for (int i = 0; i < nwcon; i++) {
      if (ParOptRealPart(psw[i]) < 0.0) {
        double numer = ParOptRealPart(sw[i]);
        double alpha = -tau * numer / ParOptRealPart(psw[i]);
        if (alpha < max_x) {
          max_x = alpha;
        }
      }
      if (ParOptRealPart(ptw[i]) < 0.0) {
        double numer = ParOptRealPart(tw[i]);
        double alpha = -tau * numer / ParOptRealPart(ptw[i]);
        if (alpha < max_x) {
          max_x = alpha;
        }
      }
      // Check the step for the Lagrange multipliers
      if (ParOptRealPart(pzsw[i]) < 0.0) {
        double numer = ParOptRealPart(zsw[i]);
        double alpha = -tau * numer / ParOptRealPart(pzsw[i]);
        if (alpha < max_z) {
          max_z = alpha;
        }
      }
      if (ParOptRealPart(pztw[i]) < 0.0) {
        double numer = ParOptRealPart(ztw[i]);
        double alpha = -tau * numer / ParOptRealPart(pztw[i]);
        if (alpha < max_z) {
          max_z = alpha;
        }
      }
    }
  }

  // Retrieve the values of the lower/upper Lagrange multipliers
  ParOptScalar *zlvals, *zuvals, *pzlvals, *pzuvals;
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);
  step.zl->getArray(&pzlvals);
  step.zu->getArray(&pzuvals);

  // Check the step for the lower/upper Lagrange multipliers
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(pzlvals[i]) < 0.0) {
        double numer = ParOptRealPart(zlvals[i]);
        double alpha = -tau * numer / ParOptRealPart(pzlvals[i]);
        if (alpha < max_z) {
          max_z = alpha;
        }
      }
    }
  }
  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(pzuvals[i]) < 0.0) {
        double numer = ParOptRealPart(zuvals[i]);
        double alpha = -tau * numer / ParOptRealPart(pzuvals[i]);
        if (alpha < max_z) {
          max_z = alpha;
        }
      }
    }
  }

  // Compute the minimum step sizes from across all processors
  double input[2], output[2];
  input[0] = max_x;
  input[1] = max_z;
  MPI_Allreduce(input, output, 2, MPI_DOUBLE, MPI_MIN, comm);

  // Return the minimum values
  *_max_x = output[0];
  *_max_z = output[1];
}

/*
  Scale the step by the specified alpha value
*/
void ParOptInteriorPoint::scaleStep(ParOptScalar alpha, int nvals,
                                    ParOptScalar *pvals) {
  for (int i = 0; i < nvals; i++) {
    pvals[i] *= alpha;
  }
}

/*
  Make sure that the step x + alpha*p lies strictly within the
  bounds l + design_precision <= x + alpha*p <= u - design_precision.

  If the bounds are violated, adjust the step so that they will be
  satisfied.
*/
void ParOptInteriorPoint::computeStepVec(ParOptVec *xvec, ParOptScalar alpha,
                                         ParOptVec *pvec, ParOptVec *lower,
                                         ParOptScalar *lower_value,
                                         ParOptVec *upper,
                                         ParOptScalar *upper_value) {
  ParOptScalar *xvals = NULL;
  ParOptScalar *pvals = NULL;
  ParOptScalar *ubvals = NULL;
  ParOptScalar *lbvals = NULL;
  int size = xvec->getArray(&xvals);
  pvec->getArray(&pvals);

  if (lower) {
    lower->getArray(&lbvals);
  }
  if (upper) {
    upper->getArray(&ubvals);
  }

  computeStep(size, xvals, alpha, pvals, lbvals, lower_value, ubvals,
              upper_value);
}

/*
  Make sure that the step is within the prescribed bounds
*/
void ParOptInteriorPoint::computeStep(int nvals, ParOptScalar *xvals,
                                      ParOptScalar alpha,
                                      const ParOptScalar *pvals,
                                      const ParOptScalar *lbvals,
                                      const ParOptScalar *lower_value,
                                      const ParOptScalar *ubvals,
                                      const ParOptScalar *upper_value) {
  const double design_precision = options->getFloatOption("design_precision");
  for (int i = 0; i < nvals; i++) {
    xvals[i] = xvals[i] + alpha * pvals[i];
  }
  if (lbvals) {
    for (int i = 0; i < nvals; i++) {
      if (ParOptRealPart(xvals[i]) <=
          ParOptRealPart(lbvals[i]) + design_precision) {
        xvals[i] = lbvals[i] + design_precision;
      }
    }
  } else if (lower_value) {
    double lbval = ParOptRealPart(*lower_value);
    for (int i = 0; i < nvals; i++) {
      if (ParOptRealPart(xvals[i]) <= lbval + design_precision) {
        xvals[i] = lbval + design_precision;
      }
    }
  }

  if (ubvals) {
    for (int i = 0; i < nvals; i++) {
      if (ParOptRealPart(xvals[i]) + design_precision >=
          ParOptRealPart(ubvals[i])) {
        xvals[i] = ubvals[i] - design_precision;
      }
    }
  } else if (upper_value) {
    double ubval = ParOptRealPart(*upper_value);
    for (int i = 0; i < nvals; i++) {
      if (ParOptRealPart(xvals[i]) + design_precision >= ubval) {
        xvals[i] = ubval - design_precision;
      }
    }
  }
}

/*
  Scale the KKT step by the maximum allowable step length
*/
int ParOptInteriorPoint::scaleKKTStep(ParOptVars &vars, ParOptVars &step,
                                      double tau, ParOptScalar comp,
                                      int inexact_newton_step, double *_alpha_x,
                                      double *_alpha_z) {
  double alpha_x = 1.0, alpha_z = 1.0;
  computeMaxStep(vars, tau, step, &alpha_x, &alpha_z);

  // Keep track of whether we set both the design and Lagrange
  // multiplier steps equal to one another
  int ceq_step = 0;

  // Check if we're using a Newton step or not
  if (!inexact_newton_step) {
    // First, bound the difference between the step lengths. This
    // code cuts off the difference between the step lengths if the
    // difference is greater that 100.
    double max_bnd = 100.0;
    if (alpha_x > alpha_z) {
      if (alpha_x > max_bnd * alpha_z) {
        alpha_x = max_bnd * alpha_z;
      } else if (alpha_x < alpha_z / max_bnd) {
        alpha_x = alpha_z / max_bnd;
      }
    } else {
      if (alpha_z > max_bnd * alpha_x) {
        alpha_z = max_bnd * alpha_x;
      } else if (alpha_z < alpha_x / max_bnd) {
        alpha_z = alpha_x / max_bnd;
      }
    }

    // As a last check, compute the average of the complementarity
    // products at the full step length. If the complementarity
    // increases, use equal step lengths.
    ParOptScalar comp_new = computeCompStep(vars, alpha_x, alpha_z, step);

    if (ParOptRealPart(comp_new) > 10.0 * ParOptRealPart(comp)) {
      ceq_step = 1;
      if (alpha_x > alpha_z) {
        alpha_x = alpha_z;
      } else {
        alpha_z = alpha_x;
      }
    }
  } else {
    // If we're using a Newton method, use the same step
    // size for both the multipliers and variables
    if (alpha_x > alpha_z) {
      alpha_x = alpha_z;
    } else {
      alpha_z = alpha_x;
    }
  }

  // Scale the steps by the maximum permissible step lengths
  step.x->scale(alpha_x);
  if (use_lower) {
    step.zl->scale(alpha_z);
  }
  if (use_upper) {
    step.zu->scale(alpha_z);
  }

  step.sw->scale(alpha_x);
  step.tw->scale(alpha_x);
  step.zw->scale(alpha_z);
  step.zsw->scale(alpha_z);
  step.ztw->scale(alpha_z);
  scaleStep(alpha_x, ncon, step.s);
  scaleStep(alpha_x, ncon, step.t);
  scaleStep(alpha_z, ncon, step.z);
  scaleStep(alpha_z, ncon, step.zs);
  scaleStep(alpha_z, ncon, step.zt);

  *_alpha_x = alpha_x;
  *_alpha_z = alpha_z;

  return ceq_step;
}

/*
  Check the gradient of the merit function using finite-difference
  or complex-step
*/
void ParOptInteriorPoint::checkMeritFuncGradient(ParOptVec *xpt, double dh) {
  if (xpt) {
    variables.x->copyValues(xpt);
  }

  // Evaluate the objective and constraints and their gradients
  int fail_obj = prob->evalObjCon(variables.x, &fobj, c);
  neval++;
  if (fail_obj) {
    fprintf(stderr, "ParOpt: Function and constraint evaluation failed\n");
    return;
  }

  int fail_gobj = prob->evalObjConGradient(variables.x, g, Ac);
  ngeval++;
  if (fail_gobj) {
    fprintf(stderr, "ParOpt: Gradient evaluation failed\n");
    return;
  }

  // Set pointers
  ParOptVec *x = variables.x;
  ParOptVec *px = update.x;
  ParOptVec *rx = residual.x;

  ParOptVec *sw = variables.sw;
  ParOptVec *psw = update.sw;
  ParOptVec *rsw = residual.sw;

  ParOptVec *tw = variables.tw;
  ParOptVec *ptw = update.tw;
  ParOptVec *rtw = residual.tw;

  const ParOptScalar *s = variables.s;
  ParOptScalar *ps = update.s;
  ParOptScalar *rs = residual.s;

  const ParOptScalar *t = variables.t;
  ParOptScalar *pt = update.t;
  ParOptScalar *rt = residual.t;

  ParOptScalar *rc = residual.z;

  // If the point is specified, pick a direction and use it,
  // otherwise use the existing step
  if (xpt) {
    // Set a step in the design variables
    px->copyValues(g);
    px->scale(-1.0 / ParOptRealPart(px->norm()));

    // Zero all other components in the step computation
    for (int i = 0; i < ncon; i++) {
      ps[i] = -0.259 * (1 + (i % 3));
      pt[i] = -0.349 * (4 - (i % 2));
    }

    if (nwcon > 0) {
      psw->zeroEntries();
      ptw->zeroEntries();
      ParOptScalar *pswvals, *ptwvals;
      psw->getArray(&pswvals);
      ptw->getArray(&ptwvals);

      for (int i = 0; i < nwcon; i++) {
        pswvals[i] = -0.419 * (1 + (i % 5));
      }

      for (int i = 0; i < nwcon; i++) {
        ptwvals[i] = -0.7513 * (1 + (i % 19));
      }
    }
  }

  // Evaluate the merit function and its derivative
  ParOptScalar m0 = 0.0, dm0 = 0.0;
  double max_x = 1.0;
  evalMeritInitDeriv(variables, update, max_x, &m0, &dm0, residual.x, wtemp,
                     residual.zw);

#ifdef PAROPT_USE_COMPLEX
  ParOptScalar *xvals, *rxvals, *pxvals;
  int size = x->getArray(&xvals);
  rx->getArray(&rxvals);
  px->getArray(&pxvals);

  for (int i = 0; i < size; i++) {
    rxvals[i] =
        ParOptScalar(ParOptRealPart(xvals[i]), dh * ParOptRealPart(pxvals[i]));
  }

  for (int i = 0; i < ncon; i++) {
    rs[i] = s[i] + ParOptScalar(0.0, dh) * ps[i];
    rt[i] = t[i] + ParOptScalar(0.0, dh) * pt[i];
  }

  rsw->copyValues(sw);
  rsw->axpy(ParOptScalar(0.0, dh), psw);

  rtw->copyValues(tw);
  rtw->axpy(ParOptScalar(0.0, dh), ptw);
#else
  rx->copyValues(x);
  rx->axpy(dh, px);

  for (int i = 0; i < ncon; i++) {
    rs[i] = s[i] + dh * ps[i];
    rt[i] = t[i] + dh * pt[i];
  }

  rsw->copyValues(sw);
  rsw->axpy(dh, psw);

  rtw->copyValues(tw);
  rtw->axpy(dh, ptw);
#endif  // PAROPT_USE_COMPLEX

  // Evaluate the objective
  ParOptScalar ftemp;
  fail_obj = prob->evalObjCon(rx, &ftemp, rc);
  neval++;
  if (fail_obj) {
    fprintf(stderr, "ParOpt: Function and constraint evaluation failed\n");
    return;
  }
  ParOptScalar m1 = evalMeritFunc(ftemp, rc, rx, rs, rt, rsw, rtw);

  ParOptScalar fd = 0.0;
#ifdef PAROPT_USE_COMPLEX
  fd = ParOptImagPart(m1) / dh;
#else
  fd = (m1 - m0) / dh;
#endif  // PAROPT_USE_COMPLEX

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root) {
    fprintf(stdout, "Merit function test\n");
    fprintf(
        stdout, "dm FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
        ParOptRealPart(fd), ParOptRealPart(dm0), fabs(ParOptRealPart(fd - dm0)),
        fabs(ParOptRealPart((fd - dm0) / fd)));
  }

#ifdef PAROPT_USE_COMPLEX
  // Evaluate the objective again back at the original x
  fail_obj = prob->evalObjCon(x, &ftemp, rc);
  neval++;
  if (fail_obj) {
    fprintf(stderr, "ParOpt: Function and constraint evaluation failed\n");
    return;
  }
#endif
}

/*
  Evaluate the infeasibility of the combined dense and sparse constraints
*/
ParOptScalar ParOptInteriorPoint::evalInfeas(
    const ParOptScalar *ck, ParOptVec *xk, const ParOptScalar *sk,
    const ParOptScalar *tk, ParOptVec *swk, ParOptVec *twk, ParOptVec *rw) {
  // Compute the infeasibility
  ParOptScalar dense_infeas = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar cval = (ck[i] - sk[i] + tk[i]);
    dense_infeas += cval * cval;
  }

  // Compute the sparse infeasibility
  if (nwcon > 0) {
    prob->evalSparseCon(xk, rw);
  }
  rw->axpy(-1.0, swk);
  rw->axpy(1.0, twk);
  ParOptScalar sparse_infeas = rw->norm();

  // Compute the l2 norm of the infeasibility
  ParOptScalar infeas = sqrt(dense_infeas + sparse_infeas * sparse_infeas);

  return infeas;
}

/*
  Evaluate the directional derivative of the infeasibility
*/
ParOptScalar ParOptInteriorPoint::evalInfeasDeriv(ParOptVars &vars,
                                                  ParOptVars &step,
                                                  ParOptScalar *pinfeas,
                                                  ParOptVec *rw1,
                                                  ParOptVec *rw2) {
  // Compute the infeasibility and directional derivative
  ParOptScalar dense_infeas = 0.0;
  ParOptScalar pdense_infeas = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar cval = (c[i] - vars.s[i] + vars.t[i]);
    ParOptScalar pcval = (Ac[i]->dot(step.x) - step.s[i] + step.t[i]);

    dense_infeas += cval * cval;
    pdense_infeas += cval * pcval;
  }

  // Compute the contributions from the sparse constraints
  if (nwcon > 0) {
    prob->evalSparseCon(vars.x, rw1);
  }
  rw1->axpy(-1.0, vars.sw);
  rw1->axpy(1.0, vars.tw);
  ParOptScalar sparse_infeas = rw1->norm();

  // Compute the l2 norm of the infeasibility
  ParOptScalar infeas = sqrt(dense_infeas + sparse_infeas * sparse_infeas);

  // Compute (cw(x) - sw + tw)^{T}*(Aw(x)*px - psw + ptw)
  ParOptScalar psparse_infeas = 0.0;
  if (nwcon > 0) {
    rw2->zeroEntries();
    prob->addSparseJacobian(1.0, vars.x, step.x, rw2);
    rw2->axpy(-1.0, step.sw);
    rw2->axpy(1.0, step.tw);
    psparse_infeas = rw1->dot(rw2);
  }

  if (ParOptRealPart(infeas) > 0.0) {
    *pinfeas = (pdense_infeas + psparse_infeas) / infeas;
  } else {
    *pinfeas = 0.0;
  }

  return infeas;
}

/*
  Evaluate the merit function at the current point, assuming that the
  objective and constraint values are up to date.

  The merit function is given as follows:

  varphi(alpha) =

  f(x) - mu*(log(s) + log(t) + log(x - xl) + log(xu - x)) +
  rho*(||c(x) - s + t||_{2} + ||cw(x) - sw + tw||_{2})

  output: The value of the merit function
*/
ParOptScalar ParOptInteriorPoint::evalMeritFunc(
    ParOptScalar fk, const ParOptScalar *ck, ParOptVec *xk,
    const ParOptScalar *sk, const ParOptScalar *tk, ParOptVec *swk,
    ParOptVec *twk) {
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Get the value of the lower/upper bounds and variables
  ParOptScalar *xvals, *lbvals, *ubvals;
  xk->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Add the contribution from the lower/upper bounds. Note
  // that we keep track of the positive and negative contributions
  // separately to try to avoid issues with numerical cancellations.
  // The difference is only taken at the end of the computation.
  ParOptScalar pos_result = 0.0, neg_result = 0.0;

  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        if (ParOptRealPart(xvals[i] - lbvals[i]) > 1.0) {
          pos_result += log(xvals[i] - lbvals[i]);
        } else {
          neg_result += log(xvals[i] - lbvals[i]);
        }
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        if (ParOptRealPart(ubvals[i] - xvals[i]) > 1.0) {
          pos_result += log(ubvals[i] - xvals[i]);
        } else {
          neg_result += log(ubvals[i] - xvals[i]);
        }
      }
    }
  }

  // Scale by the relative barrier contribution
  pos_result *= rel_bound_barrier;
  neg_result *= rel_bound_barrier;

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0) {
    ParOptScalar *sw, *tw;
    swk->getArray(&sw);
    twk->getArray(&tw);

    for (int i = 0; i < nwcon; i++) {
      if (ParOptRealPart(sw[i]) > 1.0) {
        pos_result += log(sw[i]);
      } else {
        neg_result += log(sw[i]);
      }
      if (ParOptRealPart(tw[i]) > 1.0) {
        pos_result += log(tw[i]);
      } else {
        neg_result += log(tw[i]);
      }
    }
  }

  // Sum up the result from all processors
  ParOptScalar input[2];
  ParOptScalar result[2];
  input[0] = pos_result;
  input[1] = neg_result;
  MPI_Reduce(input, result, 2, PAROPT_MPI_TYPE, MPI_SUM, opt_root, comm);

  // Extract the result of the summation over all processors
  pos_result = result[0];
  neg_result = result[1];

  // Add the contribution from the slack variables
  for (int i = 0; i < ncon; i++) {
    if (ParOptRealPart(sk[i]) > 1.0) {
      pos_result += log(sk[i]);
    } else {
      neg_result += log(sk[i]);
    }
    if (ParOptRealPart(tk[i]) > 1.0) {
      pos_result += log(tk[i]);
    } else {
      neg_result += log(tk[i]);
    }
  }

  // Compute the full merit function only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // Compute the norm of the weight constraint infeasibility
  ParOptScalar infeas = evalInfeas(ck, xk, sk, tk, swk, twk, wtemp);

  // The values of the merit function and its derivative
  ParOptScalar merit =
      fk + (penalty_gamma_sw->dot(swk) + penalty_gamma_tw->dot(twk)) -
      barrier_param * (pos_result + neg_result) + rho_penalty_search * infeas;

  for (int i = 0; i < ncon; i++) {
    merit += (penalty_gamma_s[i] * sk[i] + penalty_gamma_t[i] * tk[i]);
  }

  // Broadcast the result to all processors
  MPI_Bcast(&merit, 1, PAROPT_MPI_TYPE, opt_root, comm);

  return merit;
}

/*
  Find the minimum value of the penalty parameter which will guarantee
  that we have a descent direction. Then, using the new value of the
  penalty parameter, compute the value of the merit function and its
  derivative.

  input:
  max_x:         the maximum value of the x-scaling

  output:
  merit:     the value of the merit function
  pmerit:    the value of the derivative of the merit function
*/
void ParOptInteriorPoint::evalMeritInitDeriv(ParOptVars &vars, ParOptVars &step,
                                             double max_x, ParOptScalar *_merit,
                                             ParOptScalar *_pmerit,
                                             ParOptVec *xtmp, ParOptVec *wtmp1,
                                             ParOptVec *wtmp2) {
  const double min_rho_penalty_search =
      options->getFloatOption("min_rho_penalty_search");
  const double penalty_descent_fraction =
      options->getFloatOption("penalty_descent_fraction");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");
  const double abs_res_tol = options->getFloatOption("abs_res_tol");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");
  const int sequential_linear_method =
      options->getBoolOption("sequential_linear_method");
  const double penalty_gamma = options->getFloatOption("penalty_gamma");

  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  ParOptScalar *xvals, *pxvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  step.x->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Add the contribution from the lower/upper bounds. Note
  // that we keep track of the positive and negative contributions
  // separately to try to avoid issues with numerical cancellations.
  // The difference is only taken at the end of the computation.
  ParOptScalar pos_result = 0.0, neg_result = 0.0;
  ParOptScalar pos_presult = 0.0, neg_presult = 0.0;

  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        if (ParOptRealPart(xvals[i] - lbvals[i]) > 1.0) {
          pos_result += log(xvals[i] - lbvals[i]);
        } else {
          neg_result += log(xvals[i] - lbvals[i]);
        }

        if (ParOptRealPart(pxvals[i]) > 0.0) {
          pos_presult += pxvals[i] / (xvals[i] - lbvals[i]);
        } else {
          neg_presult += pxvals[i] / (xvals[i] - lbvals[i]);
        }
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        if (ParOptRealPart(ubvals[i] - xvals[i]) > 1.0) {
          pos_result += log(ubvals[i] - xvals[i]);
        } else {
          neg_result += log(ubvals[i] - xvals[i]);
        }

        if (ParOptRealPart(pxvals[i]) > 0.0) {
          neg_presult -= pxvals[i] / (ubvals[i] - xvals[i]);
        } else {
          pos_presult -= pxvals[i] / (ubvals[i] - xvals[i]);
        }
      }
    }
  }

  // Scale by the relative barrier contribution
  pos_result *= rel_bound_barrier;
  neg_result *= rel_bound_barrier;
  pos_presult *= rel_bound_barrier;
  neg_presult *= rel_bound_barrier;

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *psw, *ptw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    step.sw->getArray(&psw);
    step.tw->getArray(&ptw);

    for (int i = 0; i < nwcon; i++) {
      if (ParOptRealPart(sw[i]) > 1.0) {
        pos_result += log(sw[i]);
      } else {
        neg_result += log(sw[i]);
      }
      if (ParOptRealPart(tw[i]) > 1.0) {
        pos_result += log(tw[i]);
      } else {
        neg_result += log(tw[i]);
      }

      if (ParOptRealPart(psw[i]) > 0.0) {
        pos_presult += psw[i] / sw[i];
      } else {
        neg_presult += psw[i] / sw[i];
      }
      if (ParOptRealPart(ptw[i]) > 0.0) {
        pos_presult += ptw[i] / tw[i];
      } else {
        neg_presult += ptw[i] / tw[i];
      }
    }
  }

  // Sum up the result from all processors
  ParOptScalar input[4];
  ParOptScalar result[4];
  input[0] = pos_result;
  input[1] = neg_result;
  input[2] = pos_presult;
  input[3] = neg_presult;

  MPI_Reduce(input, result, 4, PAROPT_MPI_TYPE, MPI_SUM, opt_root, comm);

  // Extract the result of the summation over all processors
  pos_result = result[0];
  neg_result = result[1];
  pos_presult = result[2];
  neg_presult = result[3];

  // Add the contribution from the slack variables
  for (int i = 0; i < ncon; i++) {
    // Add the terms from the s-slack variables
    if (ParOptRealPart(vars.s[i]) > 1.0) {
      pos_result += log(vars.s[i]);
    } else {
      neg_result += log(vars.s[i]);
    }
    if (ParOptRealPart(step.s[i]) > 0.0) {
      pos_presult += step.s[i] / vars.s[i];
    } else {
      neg_presult += step.s[i] / vars.s[i];
    }

    // Add the terms from the t-slack variables
    if (ParOptRealPart(vars.t[i]) > 1.0) {
      pos_result += log(vars.t[i]);
    } else {
      neg_result += log(vars.t[i]);
    }
    if (ParOptRealPart(step.t[i]) > 0.0) {
      pos_presult += step.t[i] / vars.t[i];
    } else {
      neg_presult += step.t[i] / vars.t[i];
    }
  }

  // Perform the computations only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  ParOptScalar infeas_proj;
  ParOptScalar infeas = evalInfeasDeriv(vars, step, &infeas_proj, wtmp1, wtmp2);

  // Compute the product px^{T}*B*px
  ParOptScalar pTBp = 0.0;
  if (use_diag_hessian) {
    ParOptScalar local = 0.0;
    ParOptScalar *hvals;
    hdiag->getArray(&hvals);
    for (int i = 0; i < nvars; i++) {
      local += pxvals[i] * pxvals[i] * hvals[i];
    }
    MPI_Allreduce(&local, &pTBp, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  } else if (qn && !sequential_linear_method) {
    qn->mult(step.x, xtmp);
    pTBp = 0.5 * xtmp->dot(step.x);
  }

  // The values of the merit function and its derivative
  ParOptScalar merit =
      fobj + (penalty_gamma_sw->dot(vars.sw) + penalty_gamma_tw->dot(vars.tw)) -
      barrier_param * (pos_result + neg_result);
  ParOptScalar pmerit =
      g->dot(step.x) +
      (penalty_gamma_sw->dot(step.sw) + penalty_gamma_tw->dot(step.tw)) -
      barrier_param * (pos_presult + neg_presult);

  for (int i = 0; i < ncon; i++) {
    merit += (penalty_gamma_s[i] * vars.s[i] + penalty_gamma_t[i] * vars.t[i]);
    pmerit += (penalty_gamma_s[i] * step.s[i] + penalty_gamma_t[i] * step.t[i]);
  }

  // Compute the numerator term
  ParOptScalar numer = pmerit;
  if (ParOptRealPart(pTBp) > 0.0) {
    numer += 0.5 * pTBp;
  }

  // Compute the new penalty parameter initial guess:
  // numer + rho*infeas_proj <= - penalty_descent_frac*rho*max_x*infeas
  // numer <= rho*(-infeas_proj - penalty_descent_frac*max_x*infeas)
  // We must have that:
  //     -infeas_proj - penalty_descent_frac*max_x*infeas > 0

  // Therefore rho >= -numer/(infeas_proj +
  //                          penalty_descent_fraction*max_x*infeas)
  // Note that if we have taken an exact step:
  //      infeas_proj = -max_x*infeas

  double rho_hat = 0.0;

  // min_descent is always positive. We want to enforce that
  // the descent direction is at most as negative as min_descent.
  ParOptScalar min_descent =
      rho_penalty_search * penalty_descent_fraction * max_x * infeas;
  if (ParOptRealPart(infeas) < 0.1 * abs_res_tol) {
    // Since the infeasibility is small, estimate the descent direction
    // infeas_proj = - max_x * infeas
    ParOptScalar denom = -(1.0 - penalty_descent_fraction) * max_x * infeas;
    if (ParOptRealPart(numer) >= 0.0 && ParOptRealPart(denom) < 0.0) {
      rho_hat = -ParOptRealPart(numer / denom);
    } else {
      // Assuming here that denom < 0.0
      rho_hat = 0.0;
    }
  } else {
    // We now need to increase rho somehow to meet our criteria.
    ParOptScalar denom =
        infeas_proj + penalty_descent_fraction * max_x * infeas;

    // If numer is positive and denom is sufficiently negative, we
    // can compute an estimate for rho_hat
    if (ParOptRealPart(numer) >= 0.0 && ParOptRealPart(denom) < 0.0) {
      rho_hat = -ParOptRealPart(numer / denom);
    } else if (ParOptRealPart(pmerit) >= 0.0 && ParOptRealPart(denom) < 0.0) {
      // Try the relaxed condition
      // pmerit + rho * infease_proj <= - penalty_descent_frac * rho * max_x *
      // infeas
      rho_hat = -ParOptRealPart(pmerit / denom);
    } else if (ParOptRealPart(infeas_proj) < 0.0 &&
               ParOptRealPart(min_descent + numer) < 0.0) {
      // Try a relaxed criterial
      rho_hat = -ParOptRealPart((min_descent + numer) / infeas_proj);
    } else if (ParOptRealPart(numer) >= 0.0) {
      denom = -(1.0 - penalty_descent_fraction) * max_x * infeas;
      rho_hat = -ParOptRealPart(numer / denom);
    } else {
      // Assuming here that numer < 0.0
      rho_hat = 0.0;
    }
  }

  if (rho_hat >= penalty_gamma) {
    rho_hat = penalty_gamma;
  }

  // Set the penalty parameter to the smallest value
  // if it is greater than the old value
  if (rho_hat > rho_penalty_search) {
    rho_penalty_search = rho_hat;
  } else {
    // Damp the value of the penalty parameter
    rho_penalty_search *= 0.5;
    if (rho_penalty_search < rho_hat) {
      rho_penalty_search = rho_hat;
    }
  }

  // Last check: Make sure that the penalty parameter is at
  // least larger than the minimum allowable value
  if (rho_penalty_search < min_rho_penalty_search) {
    rho_penalty_search = min_rho_penalty_search;
  }

  // Now, evaluate the merit function and its derivative
  // based on the new value of the penalty parameter
  merit += rho_penalty_search * infeas;
  if (ParOptRealPart(infeas) < 0.1 * abs_res_tol) {
    pmerit -= rho_penalty_search * max_x * infeas;
  } else {
    pmerit += rho_penalty_search * infeas_proj;
  }

  input[0] = merit;
  input[1] = pmerit;
  input[2] = rho_penalty_search;

  // Broadcast the penalty parameter to all procs
  MPI_Bcast(input, 3, PAROPT_MPI_TYPE, opt_root, comm);

  *_merit = input[0];
  *_pmerit = input[1];
  rho_penalty_search = ParOptRealPart(input[2]);
}

/**
  Perform a backtracking line search from the current point along the
  specified direction. Note that this is a very simple line search
  without a second-order correction which may be required to alleviate
  the Maratos effect. (This should work regardless for compliance
  problems when the problem should be nearly convex.)

  @param alpha_min Minimum allowable step length
  @param alpha (in/out) Initial line search step length
  @param m0 The merit function value at alpha = 0
  @param dm0 Derivative of the merit function along p at alpha = 0
  @return Failure flag value
*/
int ParOptInteriorPoint::lineSearch(double alpha_min, double *_alpha,
                                    ParOptScalar m0, ParOptScalar dm0) {
  // Get parameters for the line search method
  const int max_line_iters = options->getIntOption("max_line_iters");
  const int use_backtracking_alpha =
      options->getBoolOption("use_backtracking_alpha");
  const double armijo_constant = options->getFloatOption("armijo_constant");
  const double function_precision =
      options->getFloatOption("function_precision");
  const int output_level = options->getIntOption("output_level");

  // Perform a backtracking line search until the sufficient decrease
  // conditions are satisfied
  double alpha = *_alpha;
  int fail = PAROPT_LINE_SEARCH_FAILURE;

  // Keep track of the merit function value
  ParOptScalar merit = 0.0;

  // Keep track of the best alpha value thus far and the best
  // merit function value
  ParOptScalar best_merit = 0.0;
  double best_alpha = -1.0;

  // Set pointers for the variables and search direction and temp variables
  ParOptVec *x = variables.x;
  ParOptVec *px = update.x;
  ParOptVec *rx = residual.x;

  ParOptScalar *s = variables.s;
  ParOptScalar *ps = update.s;
  ParOptScalar *rs = residual.s;

  ParOptScalar *t = variables.t;
  ParOptScalar *pt = update.t;
  ParOptScalar *rt = residual.t;

  ParOptVec *sw = variables.sw;
  ParOptVec *psw = update.sw;
  ParOptVec *rsw = residual.sw;

  ParOptVec *tw = variables.tw;
  ParOptVec *ptw = update.tw;
  ParOptVec *rtw = residual.tw;

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (output_level > 0) {
    double pxnorm = px->maxabs();
    if (outfp && rank == opt_root) {
      fprintf(outfp, "%5s %7s %25s %12s %12s %12s\n", "iter", "alpha", "merit",
              "dmerit", "||px||", "min(alpha)");
      fprintf(outfp, "%5d %7s %25.16e %12.5e %12.5e %12.5e\n", 0, " ",
              ParOptRealPart(m0), ParOptRealPart(dm0), pxnorm, alpha_min);
    }
  }

  int j = 0;
  for (; j < max_line_iters; j++) {
    // Set rx = x + alpha*px
    rx->copyValues(x);
    computeStepVec(rx, alpha, px, lb, NULL, ub, NULL);

    // Set rsw = sw + alpha*psw
    ParOptScalar zero = 0.0;
    rsw->copyValues(sw);
    computeStepVec(rsw, alpha, psw, NULL, &zero, NULL, NULL);
    rtw->copyValues(tw);
    computeStepVec(rtw, alpha, ptw, NULL, &zero, NULL, NULL);

    // Set rs = s + alpha*ps and rt = t + alpha*pt
    memcpy(rs, s, ncon * sizeof(ParOptScalar));
    computeStep(ncon, rs, alpha, ps, NULL, &zero, NULL, NULL);
    memcpy(rt, t, ncon * sizeof(ParOptScalar));
    computeStep(ncon, rt, alpha, pt, NULL, &zero, NULL, NULL);

    // Evaluate the objective and constraints at the new point
    int fail_obj = prob->evalObjCon(rx, &fobj, c);
    neval++;

    if (fail_obj) {
      fprintf(stderr,
              "ParOpt: Evaluation failed during line search, "
              "trying new point\n");

      // Multiply alpha by 1/10 to avoid the undefined region
      alpha *= 0.1;
      continue;
    }

    // Evaluate the merit function
    merit = evalMeritFunc(fobj, c, rx, rs, rt, rsw, rtw);

    // Print out the merit function and step at the current iterate
    if (outfp && rank == opt_root && output_level > 0) {
      fprintf(outfp, "%5d %7.1e %25.16e %12.5e\n", j + 1, alpha,
              ParOptRealPart(merit), ParOptRealPart((merit - m0) / alpha));
    }

    // If the best alpha value is negative, then this must be the
    // first successful evaluation. Otherwise, if this merit value
    // is better than previous merit function values, store the new
    // best merit function value.
    if (best_alpha < 0.0 ||
        ParOptRealPart(merit) < ParOptRealPart(best_merit)) {
      best_alpha = alpha;
      best_merit = merit;
    }

    // Check the sufficient decrease condition. Note that this is
    // relaxed by the specified function precision. This allows
    // acceptance of steps that violate the sufficient decrease
    // condition within the precision limit of the objective/merit
    // function.
    if (ParOptRealPart(merit) - armijo_constant * alpha * ParOptRealPart(dm0) <
        (ParOptRealPart(m0) + function_precision)) {
      // If this is the minimum alpha value, then we're at the minimum
      // line search step and we have had success
      if (fail & PAROPT_LINE_SEARCH_MIN_STEP) {
        fail = PAROPT_LINE_SEARCH_SUCCESS | PAROPT_LINE_SEARCH_MIN_STEP;
      } else {
        // We have successfully found a point
        fail = PAROPT_LINE_SEARCH_SUCCESS;
      }

      // The line search may be successful but the merit function value may
      // not have resulted in no improvement.
      if ((ParOptRealPart(merit) <= ParOptRealPart(m0) + function_precision) &&
          (ParOptRealPart(merit) + function_precision >= ParOptRealPart(m0))) {
        fail |= PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
      }
      break;
    } else if (fail & PAROPT_LINE_SEARCH_MIN_STEP) {
      // If this is the minimum alpha value, then quit the line search loop
      break;
    }

    // Update the new value of alpha
    if (j < max_line_iters - 1) {
      if (use_backtracking_alpha) {
        alpha = 0.5 * alpha;
        if (alpha <= alpha_min) {
          alpha = alpha_min;
          fail |= PAROPT_LINE_SEARCH_MIN_STEP;
        }
      } else {
        double alpha_new = -0.5 * ParOptRealPart(dm0) * (alpha * alpha) /
                           ParOptRealPart(merit - m0 - dm0 * alpha);

        // Bound the new step length from below by 0.01
        if (alpha_new <= alpha_min) {
          alpha = alpha_min;
          fail |= PAROPT_LINE_SEARCH_MIN_STEP;
        } else if (alpha_new < 0.01 * alpha) {
          alpha = 0.01 * alpha;
        } else {
          alpha = alpha_new;
        }
      }
    }
  }

  // The line search existed with the maximum number of line search
  // iterations
  if (j == max_line_iters) {
    fail |= PAROPT_LINE_SEARCH_MAX_ITERS;
  }

  // Check the status and return.
  if (!(fail & PAROPT_LINE_SEARCH_SUCCESS)) {
    // Check for a simple decrease within the function precision,
    // then this is sufficient to accept the step.
    if (ParOptRealPart(best_merit) <= ParOptRealPart(m0) + function_precision) {
      // We're going to say that this is success
      fail |= PAROPT_LINE_SEARCH_SUCCESS;
      // Turn off the fail flag
      fail &= ~PAROPT_LINE_SEARCH_FAILURE;
    } else if ((ParOptRealPart(merit) <=
                ParOptRealPart(m0) + function_precision) &&
               (ParOptRealPart(merit) + function_precision >=
                ParOptRealPart(m0))) {
      // Check if there is no significant change in the function value
      fail |= PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
    }

    // If we're about to accept the best alpha value, then we have
    // to re-evaluate the function at this point since the gradient
    // will be evaluated here next, and we always have to evaluate
    // function then gradient.
    if (alpha != best_alpha) {
      alpha = best_alpha;

      // Set rx = x + alpha*px
      rx->copyValues(x);
      computeStepVec(rx, alpha, px, lb, NULL, ub, NULL);

      // Evaluate the objective and constraints at the new point
      int fail_obj = prob->evalObjCon(rx, &fobj, c);
      neval++;

      // This should not happen, since we've already evaluated
      // the function at this point at a previous line search
      // iteration.
      if (fail_obj) {
        fprintf(stderr, "ParOpt: Evaluation failed during line search\n");
        fail = PAROPT_LINE_SEARCH_FAILURE;
      }
    } else {
      alpha = best_alpha;
    }
  }

  // Set the final value of alpha used in the line search
  // iteration
  *_alpha = alpha;

  return fail;
}

/**
  Compute the step, evaluate the objective and constraints and their gradients
  at the new point and update the quasi-Newton approximation.

  @param vars The variable values
  @param alpha The step length to take
  @param step The step to take
  @param eval_obj_con Flag indicating whether to evaluate the obj/cons
  @param perform_qn_update Flag to update the quasi-Newton method
  @returns The type of quasi-Newton update performed
*/
int ParOptInteriorPoint::computeStepAndUpdate(ParOptVars &vars, double alpha,
                                              ParOptVars &step,
                                              int eval_obj_con,
                                              int perform_qn_update) {
  const int use_quasi_newton_update =
      options->getBoolOption("use_quasi_newton_update");

  // Set the new values of the variables
  ParOptScalar zero = 0.0;
  computeStepVec(vars.sw, alpha, step.sw, NULL, &zero, NULL, NULL);
  computeStepVec(vars.tw, alpha, step.tw, NULL, &zero, NULL, NULL);
  computeStepVec(vars.zw, alpha, step.zw, NULL, NULL, NULL, NULL);
  computeStepVec(vars.zsw, alpha, step.zsw, NULL, &zero, NULL, NULL);
  computeStepVec(vars.ztw, alpha, step.ztw, NULL, &zero, NULL, NULL);

  if (use_lower) {
    computeStepVec(vars.zl, alpha, step.zl, NULL, &zero, NULL, NULL);
  }
  if (use_upper) {
    computeStepVec(vars.zu, alpha, step.zu, NULL, &zero, NULL, NULL);
  }

  computeStep(ncon, vars.s, alpha, step.s, NULL, &zero, NULL, NULL);
  computeStep(ncon, vars.t, alpha, step.t, NULL, &zero, NULL, NULL);
  computeStep(ncon, vars.z, alpha, step.z, NULL, NULL, NULL, NULL);
  computeStep(ncon, vars.zs, alpha, step.zs, NULL, &zero, NULL, NULL);
  computeStep(ncon, vars.zt, alpha, step.zt, NULL, &zero, NULL, NULL);

  // Compute the negative gradient of the Lagrangian using the
  // old gradient information with the new multiplier estimates
  if (qn && perform_qn_update && use_quasi_newton_update) {
    y_qn->copyValues(g);
    y_qn->scale(-1.0);
    for (int i = 0; i < ncon; i++) {
      y_qn->axpy(vars.z[i], Ac[i]);
    }

    // Add the term: Aw^{T}*zw
    if (nwcon > 0) {
      prob->addSparseJacobianTranspose(1.0, vars.x, vars.zw, y_qn);
    }
  }

  // Apply the step to the design variables only
  // after computing the contribution of the constraint
  // Jacobian to the BFGS update
  computeStepVec(vars.x, alpha, step.x, lb, NULL, ub, NULL);

  // Evaluate the objective if needed. This step is not required
  // if a line search has just been performed.
  if (eval_obj_con) {
    int fail_obj = prob->evalObjCon(vars.x, &fobj, c);
    neval++;
    if (fail_obj) {
      fprintf(stderr, "ParOpt: Function and constraint evaluation failed\n");
      return fail_obj;
    }
  }

  // Evaluate the derivative at the new point
  int fail_gobj = prob->evalObjConGradient(vars.x, g, Ac);
  ngeval++;
  if (fail_gobj) {
    fprintf(stderr,
            "ParOpt: Gradient evaluation failed at final line search\n");
  }

  // Compute the Quasi-Newton update
  int update_type = 0;
  if (qn && perform_qn_update) {
    if (use_quasi_newton_update) {
      // Add the new gradient of the Lagrangian with the new
      // multiplier estimates.
      // Compute the step - scale by the step length
      s_qn->copyValues(step.x);
      s_qn->scale(alpha);

      // Finish computing the difference in gradients
      y_qn->axpy(1.0, g);
      for (int i = 0; i < ncon; i++) {
        y_qn->axpy(-vars.z[i], Ac[i]);
      }

      // Add the term: -Aw^{T}*zw
      if (nwcon > 0) {
        prob->addSparseJacobianTranspose(-1.0, vars.x, vars.zw, y_qn);
      }

      prob->computeQuasiNewtonUpdateCorrection(vars.x, vars.z, vars.zw, s_qn,
                                               y_qn);
      update_type = qn->update(vars.x, vars.z, vars.zw, s_qn, y_qn);
    } else {
      update_type = qn->update(vars.x, vars.z, vars.zw);
    }
  }

  return update_type;
}

/*
  Get the initial design variable values, and the lower and upper
  bounds. Perform a check to see that the bounds are consistent and
  modify the design variable to conform to the bounds if neccessary.

  input:
  init_multipliers:  Flag to indicate whether to initialize multipliers
*/
void ParOptInteriorPoint::initAndCheckDesignAndBounds() {
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the design variables and bounds
  prob->getVarsAndBounds(variables.x, lb, ub);

  // Check the design variables and bounds, move things that
  // don't make sense and print some warnings
  ParOptScalar *xvals, *lbvals, *ubvals;
  variables.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Check the variable values to see if they are reasonable
  double rel_bound = 0.001 * barrier_param;
  int check_flag = 0;
  if (use_lower && use_upper) {
    for (int i = 0; i < nvars; i++) {
      // Fixed variables are not allowed
      ParOptScalar delta = 1.0;
      if (ParOptRealPart(lbvals[i]) > -max_bound_value &&
          ParOptRealPart(ubvals[i]) < max_bound_value) {
        if (ParOptRealPart(lbvals[i]) >= ParOptRealPart(ubvals[i])) {
          check_flag = (check_flag | 1);
          // Make up bounds
          lbvals[i] = 0.5 * (lbvals[i] + ubvals[i]) - 0.5 * rel_bound;
          ubvals[i] = lbvals[i] + rel_bound;
        }
        delta = ubvals[i] - lbvals[i];
      }

      // Check if x is too close the boundary
      if (ParOptRealPart(lbvals[i]) > -max_bound_value &&
          ParOptRealPart(xvals[i]) <
              ParOptRealPart(lbvals[i] + rel_bound * delta)) {
        check_flag = (check_flag | 2);
        xvals[i] = lbvals[i] + rel_bound * delta;
      }
      if (ParOptRealPart(ubvals[i]) < max_bound_value &&
          ParOptRealPart(xvals[i]) >
              ParOptRealPart(ubvals[i] - rel_bound * delta)) {
        check_flag = (check_flag | 4);
        xvals[i] = ubvals[i] - rel_bound * delta;
      }
    }
  }

  // Perform a bitwise global OR operation
  int tmp_check_flag = check_flag;
  MPI_Allreduce(&tmp_check_flag, &check_flag, 1, MPI_INT, MPI_BOR, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Print the results of the warnings
  if (rank == 0 && outfp) {
    if (check_flag & 1) {
      fprintf(outfp, "ParOpt Warning: Variable bounds are inconsistent\n");
    }
    if (check_flag & 2) {
      fprintf(outfp,
              "ParOpt Warning: Variables may be too close to lower bound\n");
    }
    if (check_flag & 4) {
      fprintf(outfp,
              "ParOpt Warning: Variables may be too close to upper bound\n");
    }
  }

  // Set the largrange multipliers with bounds outside the limits to
  // zero. This ensures that they have no effect because they will not
  // be updated once the optimization begins.
  ParOptScalar *zlvals, *zuvals;
  variables.zl->getArray(&zlvals);
  variables.zu->getArray(&zuvals);

  for (int i = 0; i < nvars; i++) {
    if (ParOptRealPart(lbvals[i]) <= -max_bound_value) {
      zlvals[i] = 0.0;
    }
    if (ParOptRealPart(ubvals[i]) >= max_bound_value) {
      zuvals[i] = 0.0;
    }
  }
}

/*
  Add to the info string
*/
void ParOptInteriorPoint::addToInfo(size_t info_size, char *info,
                                    const char *format, ...) {
  va_list args;
  size_t offset = strlen(info);
  size_t buff_size = info_size - offset;

  va_start(args, format);
  vsnprintf(&info[offset], buff_size, format, args);
  va_end(args);
}

/**
   Perform the optimization.

   This is the main function that performs the actual optimization.
   The optimization uses an interior-point method. The barrier
   parameter (mu/barrier_param) is controlled using a monotone approach
   where successive barrier problems are solved and the barrier
   parameter is subsequently reduced.

   The method uses a quasi-Newton method where the Hessian is
   approximated using a limited-memory BFGS approximation. The special
   structure of the Hessian approximation is used to compute the
   updates. This computation relies on there being relatively few dense
   global inequality constraints (e.g. < 100).

   The code also has the capability to handle very sparse linear
   constraints with the special structure that the rows of the
   constraints are nearly orthogonal. This capability is still under
   development.

   @param checkpoint the name of the checkpoint file (NULL if not needed)
*/
int ParOptInteriorPoint::optimize(const char *checkpoint) {
  // Retrieve the rank of the processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  // Set the stopping criteria constants
  const double abs_res_tol = options->getFloatOption("abs_res_tol");
  const double rel_func_tol = options->getFloatOption("rel_func_tol");
  const double abs_step_tol = options->getFloatOption("abs_step_tol");

  // Set the default norm type
  const char *norm_name = options->getEnumOption("norm_type");
  ParOptNormType norm_type = PAROPT_L2_NORM;
  if (strcmp(norm_name, "infinity") == 0) {
    norm_type = PAROPT_INFTY_NORM;
  } else if (strcmp(norm_name, "l1") == 0) {
    norm_type = PAROPT_L1_NORM;
  }

  // Set the default starting point strategy
  const char *start_name = options->getEnumOption("starting_point_strategy");
  ParOptStartingPointStrategy starting_point_strategy =
      PAROPT_NO_START_STRATEGY;
  if (strcmp(start_name, "least_squares_multipliers") == 0) {
    starting_point_strategy = PAROPT_LEAST_SQUARES_MULTIPLIERS;
  } else if (strcmp(start_name, "affine_step") == 0) {
    starting_point_strategy = PAROPT_AFFINE_STEP;
  }

  // Set the barrier strategy - always start with a monotone approach then
  // switch to the specified strategy after the first barrier problem is
  // solved
  const char *barrier_name = options->getEnumOption("barrier_strategy");
  ParOptBarrierStrategy barrier_strategy = PAROPT_MONOTONE;

  ParOptBarrierStrategy input_barrier_strategy =
      PAROPT_COMPLEMENTARITY_FRACTION;
  if (strcmp(barrier_name, "monotone") == 0) {
    input_barrier_strategy = PAROPT_MONOTONE;
  } else if (strcmp(barrier_name, "mehrotra") == 0) {
    input_barrier_strategy = PAROPT_MEHROTRA;
  } else if (strcmp(barrier_name, "mehrotra_predictor_corrector") == 0) {
    input_barrier_strategy = PAROPT_MEHROTRA_PREDICTOR_CORRECTOR;
  }

  // Set the initial barrier parameter
  barrier_param = options->getFloatOption("init_barrier_param");

  // Set the initial value of the penalty parameter for the line search
  rho_penalty_search = options->getFloatOption("init_rho_penalty_search");

  // Maximum number of iterations (major since we sometimes use GMRES an
  // the inner loop)
  const int max_major_iters = options->getIntOption("max_major_iters");

  // Get options about the Hessian approximation (if any is defined)
  const int use_quasi_newton_update =
      options->getBoolOption("use_quasi_newton_update");
  const int hessian_reset_freq = options->getIntOption("hessian_reset_freq");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");
  const int sequential_linear_method =
      options->getBoolOption("sequential_linear_method");

  // Adjust whether to use the diagonal contribution to the Hessian
  if (!hdiag && use_diag_hessian) {
    hdiag = prob->createDesignVec();
    hdiag->incref();
  } else if (hdiag && !use_diag_hessian) {
    hdiag->decref();
    hdiag = NULL;
  }

  // Check if the GMRES subspace is large enough
  int m = options->getIntOption("gmres_subspace_size");
  if (m != gmres_subspace_size) {
    setGMRESSubspaceSize(m);
  }

  // Get settings related to the Hessian-vector products
  const int use_hvec_product = options->getBoolOption("use_hvec_product");
  const double nk_switch_tol = options->getFloatOption("nk_switch_tol");
  const double eisenstat_walker_gamma =
      options->getFloatOption("eisenstat_walker_gamma");
  const double eisenstat_walker_alpha =
      options->getFloatOption("eisenstat_walker_alpha");
  const double max_gmres_rtol = options->getFloatOption("max_gmres_rtol");
  const double gmres_atol = options->getFloatOption("gmres_atol");
  const int use_qn_gmres_precon = options->getBoolOption("use_qn_gmres_precon");

  // Fraction to the boundary rule
  const double min_fraction_to_boundary =
      options->getFloatOption("min_fraction_to_boundary");

  // Use a line search or not?
  const int use_line_search = options->getBoolOption("use_line_search");

  // Get the precision parameter values
  const double function_precision =
      options->getFloatOption("function_precision");
  const double design_precision = options->getFloatOption("design_precision");

  // Perform a gradient check at a specified frequency
  const int gradient_verification_frequency =
      options->getIntOption("gradient_verification_frequency");
  const double gradient_check_step_length =
      options->getFloatOption("gradient_check_step_length");

  // Frequency at which the output is written to a file
  const int write_output_frequency =
      options->getIntOption("write_output_frequency");

  // Set the output level
  const int output_level = options->getIntOption("output_level");

  // Perform an initial check of the gradient, if set by the options
  if (gradient_verification_frequency > 0) {
    prob->checkGradients(gradient_check_step_length, variables.x,
                         use_hvec_product);
  }

  // Zero out the number of function/gradient/hessian evaluations
  niter = neval = ngeval = nhvec = 0;

  // If no quasi-Newton method is defined, use a sequential linear method
  // instead
  if (!sequential_linear_method && !qn) {
    if (rank == 0) {
      fprintf(stderr,
              "ParOpt Error: Must use a sequential linear method if no "
              "quasi-Newton approximation is defined\n");
    }
    return 1;
  }

  // Initialize and check the design variables and bounds
  initAndCheckDesignAndBounds();

  // Print what options we're using to the file
  printOptionSummary(outfp);

  // Evaluate the objective, constraint and their gradients at the
  // current values of the design variables
  int fail_obj = prob->evalObjCon(variables.x, &fobj, c);
  neval++;
  if (fail_obj) {
    fprintf(stderr,
            "ParOpt: Initial function and constraint evaluation failed\n");
    return fail_obj;
  }
  int fail_gobj = prob->evalObjConGradient(variables.x, g, Ac);
  ngeval++;
  if (fail_gobj) {
    fprintf(stderr, "ParOpt: Initial gradient evaluation failed\n");
    return fail_obj;
  }

  if (starting_point_strategy == PAROPT_AFFINE_STEP) {
    initAffineStepMultipliers(variables, residual, update, norm_type);
  } else if (starting_point_strategy == PAROPT_LEAST_SQUARES_MULTIPLIERS) {
    initLeastSquaresMultipliers(variables, residual, update.x);
  }

  // Some quasi-Newton methods can be updated with only the design variable
  // values and the multiplier estimates
  if (qn && !use_quasi_newton_update) {
    qn->update(variables.x, variables.z, variables.zw);
  }

  // The previous value of the objective function
  ParOptScalar fobj_prev = 0.0;

  // Store the previous steps in the x/z directions for the purposes
  // of printing them out on the screen and modified convergence check
  double alpha_prev = 0.0;
  double alpha_xprev = 0.0;
  double alpha_zprev = 0.0;

  // Keep track of the projected merit function derivative
  ParOptScalar dm0_prev = 0.0;
  double res_norm_prev = 0.0;
  double step_norm_prev = 0.0;

  // Keep track of whether the line search resulted in no difference
  // to function precision between the previous and current
  // iterates. If the infeasibility and duality measures are satisfied
  // to sufficient accuracy, then the barrier problem will be declared
  // converged, if the MONTONE strategy is used.
  int no_merit_function_improvement = 0;

  // Store whether the line search test exceeds 2: Two consecutive
  // occurences when there is no improvement in the merit function.
  int line_search_test = 0;

  // Keep track of whether the previous line search failed
  int line_search_failed = 0;

  // Information about what happened on the previous major iteration
  char info[64];
  memset(info, '\0', sizeof(info));

  for (int k = 0; k < max_major_iters; k++, niter++) {
    // Keep track if the quasi-Newton Hessian was reset
    int qn_hessian_reset = 0;
    if (qn && !sequential_linear_method) {
      if (k > 0 && k % hessian_reset_freq == 0 && use_quasi_newton_update) {
        // Reset the quasi-Newton Hessian approximation
        qn->reset();
        qn_hessian_reset = 1;
      }
    }

    // Print out the current solution progress using the
    // hook in the problem definition
    if (write_output_frequency > 0 && k % write_output_frequency == 0) {
      if (checkpoint) {
        // Write the checkpoint file, if it fails once, set
        // the file pointer to null so it won't print again
        if (writeSolutionFile(checkpoint)) {
          fprintf(stderr, "ParOpt: Checkpoint file %s creation failed\n",
                  checkpoint);
          checkpoint = NULL;
        }
      }
      prob->writeOutput(k, variables.x);
    }

    // Print to screen the gradient check results at
    // iteration k
    if (k > 0 && (gradient_verification_frequency > 0) &&
        (k % gradient_verification_frequency == 0)) {
      prob->checkGradients(gradient_check_step_length, variables.x,
                           use_hvec_product);
    }

    // Determine if we should switch to a new barrier problem or not
    int rel_function_test = (alpha_xprev == 1.0 && alpha_zprev == 1.0 &&
                             (fabs(ParOptRealPart(fobj - fobj_prev)) <
                              rel_func_tol * fabs(ParOptRealPart(fobj_prev))));

    // Set the line search check. If there is no change in the merit
    // function value, and we're feasible and the complementarity
    // conditions are satisfied, then declare the test passed.
    if (no_merit_function_improvement) {
      line_search_test += 1;
    } else {
      line_search_test = 0;
    }

    // Compute the complementarity
    ParOptScalar comp = computeComp(variables);

    // Keep track of the norm of the different parts of the
    // KKT conditions
    double max_prime = 0.0, max_dual = 0.0, max_infeas = 0.0;

    // Compute the overall norm of the KKT conditions
    double res_norm = 0.0;

    // Keep tract if the monotone barrier problem has converged
    int monotone_barrier_converged = 0;

    if (barrier_strategy == PAROPT_MONOTONE) {
      // Compute the residual of the KKT system
      computeKKTRes(variables, barrier_param, residual, norm_type, &max_prime,
                    &max_dual, &max_infeas, &res_norm);

      // Compute the maximum of the norm of the residuals
      if (k == 0) {
        res_norm_prev = res_norm;
      }

      // Set the flag to indicate whether the barrier problem has
      // converged
      if (k > 0 && ((res_norm < 10.0 * barrier_param) || rel_function_test ||
                    (line_search_test >= 2))) {
        monotone_barrier_converged = 1;
      }

      // Keep track of the new barrier parameter (if any). Only set the
      // new barrier parameter after we've check for convergence of the
      // overall algorithm. This ensures that the previous barrier
      // parameter is saved if we successfully converge.
      double new_barrier_param = 0.0;

      // Broadcast the result of the test from the root processor
      MPI_Bcast(&monotone_barrier_converged, 1, MPI_INT, opt_root, comm);

      if (monotone_barrier_converged) {
        const double monotone_barrier_fraction =
            options->getFloatOption("monotone_barrier_fraction");
        const double monotone_barrier_power =
            options->getFloatOption("monotone_barrier_power");

        // If the barrier problem converged, we need a new convergence
        // test, but if the barrier parameter is  converged
        if (barrier_param > 0.1 * abs_res_tol) {
          line_search_test = 0;
        }

        // Compute the new barrier parameter: It is either:
        // 1. A fixed fraction of the old value
        // 2. A function mu**exp for some exp > 1.0
        // Point 2 ensures superlinear convergence (eventually)
        double mu_frac = monotone_barrier_fraction * barrier_param;
        double mu_pow = pow(barrier_param, monotone_barrier_power);

        new_barrier_param = mu_frac;
        if (mu_pow < mu_frac) {
          new_barrier_param = mu_pow;
        }

        // Truncate the barrier parameter at 0.1*abs_res_tol. If this
        // truncation occurs, set the flag that this is the final
        // barrier problem
        if (new_barrier_param < 0.1 * abs_res_tol) {
          new_barrier_param = 0.09999 * abs_res_tol;
        }

        // Compute the new barrier parameter value
        computeKKTRes(variables, new_barrier_param, residual, norm_type,
                      &max_prime, &max_dual, &max_infeas, &res_norm);

        // Reset the penalty parameter to the min allowable value
        rho_penalty_search = options->getFloatOption("min_rho_penalty_search");

        // Set the new barrier parameter
        barrier_param = new_barrier_param;
      }
    } else if (barrier_strategy == PAROPT_MEHROTRA ||
               barrier_strategy == PAROPT_MEHROTRA_PREDICTOR_CORRECTOR) {
      // Compute the residual of the KKT system
      computeKKTRes(variables, barrier_param, residual, norm_type, &max_prime,
                    &max_dual, &max_infeas, &res_norm);

      if (k == 0) {
        res_norm_prev = res_norm;
      }
    } else if (barrier_strategy == PAROPT_COMPLEMENTARITY_FRACTION) {
      const double monotone_barrier_fraction =
          options->getFloatOption("monotone_barrier_fraction");

      barrier_param = monotone_barrier_fraction * ParOptRealPart(comp);
      if (barrier_param < 0.1 * abs_res_tol) {
        barrier_param = 0.1 * abs_res_tol;
      }

      // Compute the residual of the KKT system
      computeKKTRes(variables, barrier_param, residual, norm_type, &max_prime,
                    &max_dual, &max_infeas, &res_norm);

      if (k == 0) {
        res_norm_prev = res_norm;
      }
    }

    // Print all the information we can to the screen...
    if (outfp && rank == opt_root) {
      if (k % 10 == 0 || output_level > 0) {
        fprintf(outfp,
                "\n%4s %4s %4s %4s %7s %7s %7s %12s %7s %7s %7s "
                "%7s %7s %8s %7s info\n",
                "iter", "nobj", "ngrd", "nhvc", "alpha", "alphx", "alphz",
                "fobj", "|opt|", "|infes|", "|dual|", "mu", "comp", "dmerit",
                "rho");
      }

      if (k == 0) {
        fprintf(outfp,
                "%4d %4d %4d %4d %7s %7s %7s %12.5e %7.1e %7.1e "
                "%7.1e %7.1e %7.1e %8s %7s %s\n",
                k, neval, ngeval, nhvec, "--", "--", "--", ParOptRealPart(fobj),
                max_prime, max_infeas, max_dual, barrier_param,
                ParOptRealPart(comp), "--", "--", info);
      } else {
        fprintf(outfp,
                "%4d %4d %4d %4d %7.1e %7.1e %7.1e %12.5e %7.1e "
                "%7.1e %7.1e %7.1e %7.1e %8.1e %7.1e %s\n",
                k, neval, ngeval, nhvec, alpha_prev, alpha_xprev, alpha_zprev,
                ParOptRealPart(fobj), max_prime, max_infeas, max_dual,
                barrier_param, ParOptRealPart(comp), ParOptRealPart(dm0_prev),
                rho_penalty_search, info);
      }

      // Flush the buffer so that we can see things immediately
      fflush(outfp);
    }

    // Check for convergence. We apply two different convergence
    // criteria at this point: the first based on the norm of
    // the KKT condition residuals, and the second based on the
    // difference between subsequent calls.
    int converged = 0;
    if (k > 0 && (barrier_param <= 0.1 * abs_res_tol) &&
        (res_norm < abs_res_tol || rel_function_test ||
         (line_search_test >= 2))) {
      if (outfp && rank == opt_root) {
        if (rel_function_test) {
          fprintf(
              outfp,
              "\nParOpt: Successfully converged on relative function test\n");
        } else if (line_search_test >= 2) {
          fprintf(
              outfp,
              "\nParOpt Warning: Current design point could not be improved. "
              "No barrier function decrease in previous two iterations\n");
        } else {
          fprintf(outfp,
                  "\nParOpt: Successfully converged to requested tolerance\n");
        }
      }
      converged = 1;
    }

    // Broadcast the convergence result from the root processor. This avoids
    // comparing values that might be different on different procs.
    MPI_Bcast(&converged, 1, MPI_INT, opt_root, comm);

    // Everybody quit altogether if we've converged
    if (converged) {
      break;
    }

    // Check if we should compute a Newton step or a quasi-Newton
    // step. Note that at this stage, we use s_qn and y_qn as
    // temporary arrays to help compute the KKT step. After
    // the KKT step is computed, we use them to store the
    // change in variables/gradient for the BFGS update.
    int gmres_iters = 0;

    // Flag to indicate whether to use the quasi-Newton Hessian
    // approximation to compute the next step
    int inexact_newton_step = 0;

    if (use_hvec_product) {
      // Compute the relative GMRES tolerance given the residuals
      double gmres_rtol =
          eisenstat_walker_gamma *
          pow((res_norm / res_norm_prev), eisenstat_walker_alpha);

      if (max_prime < nk_switch_tol && max_dual < nk_switch_tol &&
          max_infeas < nk_switch_tol && gmres_rtol < max_gmres_rtol) {
        // Set the flag which determines whether or not to use
        // the quasi-Newton method as a preconditioner
        int use_qn = 1;
        if (sequential_linear_method || !use_qn_gmres_precon) {
          use_qn = 0;
        }

        // Set up the KKT diagonal system
        setUpKKTDiagSystem(variables, s_qn, wtemp, use_qn);

        // Set up the full KKT system
        setUpKKTSystem(variables, ztemp, s_qn, y_qn, wtemp, use_qn);

        // Compute the inexact step using GMRES
        gmres_iters =
            computeKKTGMRESStep(variables, residual, update, ztemp, y_qn, s_qn,
                                wtemp, gmres_rtol, gmres_atol, use_qn);

        if (abs_step_tol > 0.0) {
          step_norm_prev = computeStepNorm(norm_type, update);
        }

        if (gmres_iters < 0) {
          // Print out an error code that we've failed
          if (rank == opt_root && output_level > 0) {
            fprintf(outfp, "      %9s\n", "step failed");
          }

          // Recompute the residual of the KKT system - the residual
          // was destroyed during the failed GMRES iteration
          computeKKTRes(variables, barrier_param, residual, norm_type,
                        &max_prime, &max_dual, &max_infeas);
        } else {
          // We've successfully computed a KKT step using
          // exact Hessian-vector products
          inexact_newton_step = 1;
        }
      }
    }

    // Store the objective/res_norm for next time through the loop.
    // The assignment takes place here since the GMRES computation
    // requires the use of the res_norm value.
    fobj_prev = fobj;
    res_norm_prev = res_norm;

    // Is this a sequential linear step that did not use the quasi-Newton
    // approx.
    int seq_linear_step = 0;

    // This is a step that uses only the diagonal contribution from the
    // quasi-Newton approximation
    int diagonal_quasi_newton_step = 0;

    // Compute a step based on the quasi-Newton Hessian approximation
    if (!inexact_newton_step) {
      int use_qn = 1;

      if (sequential_linear_method) {
        // If we're using a sequential linear method, set use_qn = 0.
        use_qn = 0;
      } else if (line_search_failed && !use_quasi_newton_update) {
        // In this case, the line search failed, and we are using a fixed
        // quasi-Newton method which was therefore not reset when the
        // line search failed. As a result, we try either a sequential linear
        // step or discard the vectors in the quasi-Newton Hessian approx.
        // leaving only the diagonal contributions, which will only work if
        // b0 is positive.
        // Check if the coefficient b0 is positive.
        use_qn = 0;
        seq_linear_step = 1;
        if (qn) {
          ParOptScalar b0;
          qn->getCompactMat(&b0, NULL, NULL, NULL);
          if (ParOptRealPart(b0) > 0.0) {
            seq_linear_step = 0;
            diagonal_quasi_newton_step = 1;
          }
        }
      } else if (use_diag_hessian) {
        // If we're using a diagonal Hessian approximation, compute it here
        use_qn = 0;
        int fail = prob->evalHessianDiag(variables.x, variables.z, variables.zw,
                                         hdiag);
        if (fail) {
          fprintf(stderr, "ParOpt: Hessian diagonal evaluation failed\n");
          return fail;
        }
      }

      // Compute the affine residual with barrier = 0.0 if we are using
      // the Mehrotra probing barrier strategy
      if (barrier_strategy == PAROPT_MEHROTRA ||
          barrier_strategy == PAROPT_MEHROTRA_PREDICTOR_CORRECTOR) {
        computeKKTRes(variables, 0.0, residual, norm_type, &max_prime,
                      &max_dual, &max_infeas);
      }

      // Set up the KKT diagonal system. If we're using only the
      // diagonal entries from a quasi-Newton approximation, turn those
      // on here.
      if (diagonal_quasi_newton_step) {
        use_qn = 1;
      }
      setUpKKTDiagSystem(variables, s_qn, wtemp, use_qn);

      // Set up the full KKT system
      setUpKKTSystem(variables, ztemp, s_qn, y_qn, wtemp, use_qn);

      // Solve for the KKT step. If we're using only the diagonal entries,
      // turn off the off-diagonal entries to compute the step.
      if (diagonal_quasi_newton_step) {
        use_qn = 0;
      }
      computeKKTStep(variables, residual, update, ztemp, s_qn, y_qn, wtemp,
                     use_qn);

      // Compute the norm of the step length. This is only used if the
      // abs_step_tol is set. It defaults to zero.
      if (abs_step_tol > 0.0) {
        step_norm_prev = computeStepNorm(norm_type, update);
      }

      if (barrier_strategy == PAROPT_MEHROTRA ||
          barrier_strategy == PAROPT_MEHROTRA_PREDICTOR_CORRECTOR) {
        // Compute the affine step to the boundary, allowing
        // the variables to go right to zero
        double max_x, max_z;
        computeMaxStep(variables, 1.0, update, &max_x, &max_z);

        // Compute the complementarity at the full step
        ParOptScalar comp_affine =
            computeCompStep(variables, max_x, max_z, update);

        // Use the Mehrotra rule
        double s1 = ParOptRealPart(comp_affine / comp);
        double sigma = s1 * s1 * s1;

        // Set a bound on sigma so that it is sigma >= 0.01
        if (sigma < 0.01) {
          sigma = 0.01;
        }

        // Compute the new adaptive barrier parameter
        barrier_param = sigma * ParOptRealPart(comp);
        if (barrier_param < 0.09999 * abs_res_tol) {
          barrier_param = 0.09999 * abs_res_tol;
        }

        // Compute the residual with the new barrier parameter
        computeKKTRes(variables, barrier_param, residual, norm_type, &max_prime,
                      &max_dual, &max_infeas);

        // Add the contributions to the residual from the predictor
        // corrector step
        if (barrier_strategy == PAROPT_MEHROTRA_PREDICTOR_CORRECTOR) {
          addMehrotraCorrectorResidual(update, residual);
        }

        // Compute the KKT Step
        computeKKTStep(variables, residual, update, ztemp, s_qn, y_qn, wtemp,
                       use_qn);
      }
    }

    // Check the KKT step
    if (gradient_verification_frequency > 0 &&
        ((k % gradient_verification_frequency) == 0)) {
      checkKKTStep(variables, update, residual, k, inexact_newton_step);
    }

    // Compute the maximum permitted line search lengths
    double tau = min_fraction_to_boundary;
    double tau_mu = 1.0 - barrier_param;
    if (tau_mu >= tau) {
      tau = tau_mu;
    }

    double alpha_x = 1.0, alpha_z = 1.0;
    int ceq_step = scaleKKTStep(variables, update, tau, comp,
                                inexact_newton_step, &alpha_x, &alpha_z);

    // Keep track of the step length size
    double alpha = 1.0;

    // Flag to indicate whether the line search failed
    int line_fail = PAROPT_LINE_SEARCH_FAILURE;

    // The type of quasi-Newton update performed
    int update_type = 0;

    // Keep track of whether the line search was skipped or not
    int line_search_skipped = 0;

    // By default, we assume that there is an improvement in the merit
    // function
    no_merit_function_improvement = 0;

    if (use_line_search) {
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter
      ParOptScalar m0, dm0;
      evalMeritInitDeriv(variables, update, alpha_x, &m0, &dm0, residual.x,
                         wtemp, residual.zw);

      // Store the merit function derivative
      dm0_prev = dm0;

      // If the derivative of the merit function is positive, but within
      // the function precision of zero, then go ahead and skip the line
      // search and update.
      if (ParOptRealPart(dm0) >= 0.0 &&
          ParOptRealPart(dm0) <= function_precision) {
        line_search_skipped = 1;

        // Perform a step and update the quasi-Newton Hessian approximation
        int eval_obj_con = 1;
        int perform_qn_update = 1;
        update_type = computeStepAndUpdate(variables, alpha, update,
                                           eval_obj_con, perform_qn_update);

        // Check if there was no change in the objective function
        if ((ParOptRealPart(fobj_prev) + function_precision <=
             ParOptRealPart(fobj)) &&
            (ParOptRealPart(fobj) + function_precision <=
             ParOptRealPart(fobj_prev))) {
          line_fail = PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
        }
      } else {
        // The derivative of the merit function is positive. We revert to one
        // of two approaches. We reset the Hessian approximation and try
        // again.
        if (ParOptRealPart(dm0) >= 0.0) {
          // Reset the Hessian approximation
          if (qn) {
            qn_hessian_reset = 1;
            qn->reset();
          }

          // Re-compute the KKT residuals since they may be over-written
          // during the line search step
          computeKKTRes(variables, barrier_param, residual, norm_type,
                        &max_prime, &max_dual, &max_infeas);

          // Set up the KKT diagonal system
          diagonal_quasi_newton_step = 1;
          int use_qn = 1;
          setUpKKTDiagSystem(variables, s_qn, wtemp, use_qn);

          // Compute the step
          computeKKTStep(variables, residual, update, ztemp, s_qn, y_qn, wtemp,
                         use_qn);

          // Scale the step
          int inexact_newton_step = 0;
          ceq_step = scaleKKTStep(variables, update, tau, comp,
                                  inexact_newton_step, &alpha_x, &alpha_z);

          // Re-evaluate the merit function derivative
          evalMeritInitDeriv(variables, update, alpha_x, &m0, &dm0, residual.x,
                             wtemp, residual.zw);

          // Store the merit function derivative
          dm0_prev = dm0;
        }

        // Check that the merit function derivative is correct and print
        // the derivative to the screen on the optimization-root processor
        if (gradient_verification_frequency > 0 &&
            ((k % gradient_verification_frequency) == 0)) {
          checkMeritFuncGradient(NULL, gradient_check_step_length);
        }

        if (ParOptRealPart(dm0) >= 0.0) {
          line_fail = PAROPT_LINE_SEARCH_FAILURE;
        } else {
          // Prepare to perform the line search. First, compute the minimum
          // allowable line search step length
          double px_norm = update.x->maxabs();
          double alpha_min = 1.0;
          if (px_norm != 0.0) {
            alpha_min = function_precision / px_norm;
          }
          if (alpha_min > 0.5) {
            alpha_min = 0.5;
          }
          line_fail = lineSearch(alpha_min, &alpha, m0, dm0);

          // If the step length is less than the design precision
          if (px_norm < design_precision) {
            line_fail |= PAROPT_LINE_SEARCH_SHORT_STEP;
          }

          // If the line search was successful, quit
          if (!(line_fail & PAROPT_LINE_SEARCH_FAILURE)) {
            // Do not evaluate the objective and constraints at the new point
            // since we've just performed a successful line search and the
            // last point was evaluated there. Perform a quasi-Newton update
            // if required.
            int eval_obj_con = 0;
            int perform_qn_update = 1;
            update_type = computeStepAndUpdate(variables, alpha, update,
                                               eval_obj_con, perform_qn_update);
          }
        }
      }
    } else {
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter. This
      // occurs even thought we are not using a line search.
      ParOptScalar m0, dm0;
      evalMeritInitDeriv(variables, update, alpha_x, &m0, &dm0, residual.x,
                         wtemp, residual.zw);

      // Store the merit function derivative to print to the output file
      dm0_prev = dm0;

      // We signal success here
      line_fail = PAROPT_LINE_SEARCH_SUCCESS;

      // Evaluate the objective/constraints at the new point since we skipped
      // the line search step here.
      int eval_obj_con = 1;
      int perform_qn_update = 1;
      update_type = computeStepAndUpdate(variables, alpha, update, eval_obj_con,
                                         perform_qn_update);

      // No line search has been performed, but there may have been no
      // improvement in the last step
      ParOptScalar m1 = evalMeritFunc(fobj, c, variables.x, variables.s,
                                      variables.t, variables.sw, variables.tw);
      if ((ParOptRealPart(m1) <= ParOptRealPart(m0) + function_precision) &&
          (ParOptRealPart(m1) + function_precision >= ParOptRealPart(m0))) {
        line_fail |= PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
      } else if (fabs(ParOptRealPart(dm0)) <= function_precision) {
        line_fail = PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
      }
    }

    // Check whether there was a change in the merit function to
    // machine precision
    no_merit_function_improvement =
        ((line_fail & PAROPT_LINE_SEARCH_NO_IMPROVEMENT) ||
         (line_fail & PAROPT_LINE_SEARCH_MIN_STEP) ||
         (line_fail & PAROPT_LINE_SEARCH_SHORT_STEP) ||
         (line_fail & PAROPT_LINE_SEARCH_FAILURE));

    // Keep track of whether the last step failed
    line_search_failed = (line_fail & PAROPT_LINE_SEARCH_FAILURE);

    // Store the steps in x/z for printing later
    alpha_prev = alpha;
    alpha_xprev = alpha_x;
    alpha_zprev = alpha_z;

    // Reset the quasi-Newton Hessian if there is a line search failure
    if (qn && use_quasi_newton_update &&
        (line_fail & PAROPT_LINE_SEARCH_FAILURE)) {
      qn_hessian_reset = 1;
      qn->reset();
    }

    // Create a string to print to the screen
    if (rank == opt_root) {
      memset(info, '\0', sizeof(info));
      if (gmres_iters != 0) {
        // Print how well GMRES is doing
        addToInfo(sizeof(info), info, "%s%d ", "iNK", gmres_iters);
      }
      if (update_type == 1) {
        // Damped BFGS update
        addToInfo(sizeof(info), info, "%s ", "dampH");
      } else if (update_type == 2) {
        // Skipped update
        addToInfo(sizeof(info), info, "%s ", "skipH");
      }
      if (qn_hessian_reset) {
        // Hessian reset
        addToInfo(sizeof(info), info, "%s ", "resetH");
      }
      if (line_fail & PAROPT_LINE_SEARCH_FAILURE) {
        // Line search failure
        addToInfo(sizeof(info), info, "%s ", "LFail");
      }
      if (line_fail & PAROPT_LINE_SEARCH_MIN_STEP) {
        // Line search reached the minimum step length
        addToInfo(sizeof(info), info, "%s ", "LMnStp");
      }
      if (line_fail & PAROPT_LINE_SEARCH_MAX_ITERS) {
        // Line search reached the max. number of iterations
        addToInfo(sizeof(info), info, "%s ", "LMxItr");
      }
      if (line_fail & PAROPT_LINE_SEARCH_NO_IMPROVEMENT) {
        // Line search did not improve merit function
        addToInfo(sizeof(info), info, "%s ", "LNoImprv");
      }
      if (seq_linear_step) {
        // Sequential linear step (even though we're using a QN approx.)
        addToInfo(sizeof(info), info, "%s ", "SLP");
      }
      if (diagonal_quasi_newton_step) {
        // Step generated using only the diagonal from a quasi-Newton approx.
        addToInfo(sizeof(info), info, "%s ", "DQN");
      }
      if (line_search_skipped) {
        // Line search reached the max. number of iterations
        addToInfo(sizeof(info), info, "%s ", "LSkip");
      }
      if (ceq_step) {
        // The step lengths are equal due to an increase in the
        // the complementarity at the new step
        addToInfo(sizeof(info), info, "%s ", "cmpEq");
      }
    }

    // If the first problem converged, switched to the user specified barrier
    // strategy for the remainder of the optimization
    if (monotone_barrier_converged) {
      barrier_strategy = input_barrier_strategy;
    }
  }

  // Success - we completed the optimization
  return 0;
}

/*
  Compute an initial multiplier estimate using a least-squares method

  The least squares multipliers can be found by solving the following system
  of equations where eps is a small number:

  [ [ I   Aw^{T} ]  A^{T} ][  yx ] = [ -(g - zl + zu) ]
  [ [ Aw   -eps  ]     0  ][ -zw ] = [ 0 ]
  [   A              -eps ][  -z ] = [ 0 ]

  Defining

  D0 =
  [ I   Aw^{T} ]
  [ Aw   -eps  ]

  We can compute the Schur complement

  G = eps + (A, 0) * D0^{-1} * (A, 0)^{T}

  Then solve

  G * z = (A, 0)^{T} * D0^{-1} * (g - zl + zu)

  Then we can solve

  [ I   Aw^{T} ][  yx ] = [ -(g - zl + zu ) + A^{T} * z ]
  [ Aw   -eps  ][ -zw ] = [  0                          ]

  for zw.
*/
void ParOptInteriorPoint::initLeastSquaresMultipliers(ParOptVars &vars,
                                                      ParOptVars &res,
                                                      ParOptVec *yx) {
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const double init_barrier_param =
      options->getFloatOption("init_barrier_param");

  // Set the largrange multipliers with bounds outside the
  // limits to zero
  ParOptScalar *lbvals, *ubvals, *zlvals, *zuvals;
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Set the Largrange multipliers associated with the
  // the lower/upper bounds to the initial barrier parameter
  vars.zl->set(init_barrier_param);
  vars.zu->set(init_barrier_param);

  // Set the Lagrange multipliers and slack variables
  // associated with the sparse constraints initial barrier parameter
  vars.zw->set(init_barrier_param);
  vars.sw->set(init_barrier_param);
  vars.tw->set(init_barrier_param);
  vars.zsw->set(init_barrier_param);
  vars.ztw->set(init_barrier_param);

  // Set the Largrange multipliers and slack variables associated
  // with the dense constraints to 1.0
  for (int i = 0; i < ncon; i++) {
    vars.z[i] = init_barrier_param;
    vars.s[i] = max2(init_barrier_param, c[i] + init_barrier_param);
    vars.t[i] = max2(init_barrier_param, -c[i] + init_barrier_param);
    vars.zs[i] = init_barrier_param;
    vars.zt[i] = init_barrier_param;
  }

  // Zero the multipliers for bounds that are out-of-range
  for (int i = 0; i < nvars; i++) {
    if (ParOptRealPart(lbvals[i]) <= -max_bound_value) {
      zlvals[i] = 0.0;
    }
    if (ParOptRealPart(ubvals[i]) >= max_bound_value) {
      zuvals[i] = 0.0;
    }
  }

  double small = 1e-4;

  // Set the components of the diagonal matrix
  ParOptScalar *dvals, *cvals;
  Dinv->getArray(&dvals);
  Cdiag->getArray(&cvals);

  for (int i = 0; i < nvars; i++) {
    dvals[i] = 1.0;
  }
  for (int i = 0; i < nwcon; i++) {
    cvals[i] = small;
  }

  // Factor the quasi-definite matrix
  mat->factor(vars.x, Dinv, Cdiag);

  // Set the value of the G matrix
  memset(Gmat, 0, ncon * ncon * sizeof(ParOptScalar));

  // Now, compute the Schur complement with the Dmatrix
  for (int j = 0; j < ncon; j++) {
    mat->apply(Ac[j], res.x, res.zw);

    for (int i = j; i < ncon; i++) {
      Gmat[i + ncon * j] += Ac[i]->dot(res.x);
    }
  }

  // Populate the remainder of the matrix because it is
  // symmetric
  for (int j = 0; j < ncon; j++) {
    for (int i = j + 1; i < ncon; i++) {
      Gmat[j + ncon * i] = Gmat[i + ncon * j];
    }
  }

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Add the diagonal component to the matrix
    if (rank == opt_root) {
      for (int i = 0; i < ncon; i++) {
        Gmat[i * (ncon + 1)] += small;
      }
    }

    // Broadcast the result to all processors. Note that this ensures
    // that the factorization will be the same on all processors
    MPI_Bcast(Gmat, ncon * ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Factor the matrix for future use
    int info = 0;
    LAPACKdgetrf(&ncon, &ncon, Gmat, &ncon, gpiv, &info);
  }

  // Compute the right-hand-side
  // Note that we scale the right-hand-side to get rhs = -(g - zl + zu)
  res.x->copyValues(g);
  res.x->axpy(-1.0, vars.zl);
  res.x->axpy(1.0, vars.zu);
  res.x->scale(-1.0);

  // Compute the terms from the weighting constraints
  res.zw->zeroEntries();

  // Solve for the update and store in y.x
  mat->apply(res.x, res.zw, yx, vars.zw);

  // Now, compute yz = A^{T} * y.x
  memset(vars.z, 0, ncon * sizeof(ParOptScalar));
  yx->mdot(Ac, ncon, vars.z);

  if (ncon > 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Compute the full right-hand-side on the root proc
    if (rank == opt_root) {
      for (int i = 0; i < ncon; i++) {
        vars.z[i] = -vars.z[i];
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one, Gmat, &ncon, gpiv, vars.z, &ncon, &info);
    }

    MPI_Bcast(vars.z, ncon, PAROPT_MPI_TYPE, opt_root, comm);
  }

  for (int i = 0; i < ncon; i++) {
    res.x->axpy(vars.z[i], Ac[i]);
  }

  mat->apply(res.x, res.zw, yx, vars.zw);

  // Keep the Lagrange multipliers if they are within a reasonable range.
  for (int i = 0; i < ncon; i++) {
    double gamma =
        10 * ParOptRealPart(max2(penalty_gamma_s[i], penalty_gamma_t[i]));
    if (ParOptRealPart(vars.z[i]) < -gamma ||
        ParOptRealPart(vars.z[i]) > gamma) {
      vars.z[i] = 0.0;
    }
  }

  if (nwcon > 0) {
    ParOptScalar *gamma_sw, *gamma_tw, *zw;
    penalty_gamma_sw->getArray(&gamma_sw);
    penalty_gamma_tw->getArray(&gamma_tw);
    vars.zw->getArray(&zw);

    for (int i = 0; i < nwcon; i++) {
      double gamma = 10 * ParOptRealPart(max2(gamma_sw[i], gamma_tw[i]));
      if (ParOptRealPart(zw[i]) < -gamma || ParOptRealPart(zw[i]) > gamma) {
        zw[i] = 0.0;
      }
    }
  }
}

void ParOptInteriorPoint::initAffineStepMultipliers(ParOptVars &vars,
                                                    ParOptVars &res,
                                                    ParOptVars &step,
                                                    ParOptNormType norm_type) {
  // Set the minimum allowable multiplier
  const double start_affine_multiplier_min =
      options->getFloatOption("start_affine_multiplier_min");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const int sequential_linear_method =
      options->getBoolOption("sequential_linear_method");
  const int use_qn_gmres_precon = options->getBoolOption("use_qn_gmres_precon");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");

  // Perform a preliminary estimate of the multipliers using the least-squares
  // method
  initLeastSquaresMultipliers(vars, res, step.x);

  // Set the largrange multipliers with bounds outside the
  // limits to zero
  ParOptScalar *lbvals, *ubvals, *zlvals, *zuvals;
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Zero the multipliers for bounds that are out-of-range
  for (int i = 0; i < nvars; i++) {
    if (ParOptRealPart(lbvals[i]) <= -max_bound_value) {
      zlvals[i] = 0.0;
    }
    if (ParOptRealPart(ubvals[i]) >= max_bound_value) {
      zuvals[i] = 0.0;
    }
  }

  // Find the affine scaling step
  double max_prime, max_dual, max_infeas;
  computeKKTRes(vars, 0.0, res, norm_type, &max_prime, &max_dual, &max_infeas);

  // Set the flag which determines whether or not to use
  // the quasi-Newton method as a preconditioner
  int use_qn = 1;
  if (sequential_linear_method || !use_qn_gmres_precon || use_diag_hessian) {
    use_qn = 0;
  }

  // Set up the KKT diagonal system
  setUpKKTDiagSystem(vars, s_qn, wtemp, use_qn);

  // Set up the full KKT system
  setUpKKTSystem(vars, ztemp, s_qn, y_qn, wtemp, use_qn);

  // Solve for the KKT step
  computeKKTStep(vars, res, step, ztemp, s_qn, y_qn, wtemp, use_qn);

  // Copy over the values
  for (int i = 0; i < ncon; i++) {
    vars.z[i] = vars.z[i] + step.z[i];
    vars.s[i] = max2(start_affine_multiplier_min,
                     fabs(ParOptRealPart(vars.s[i] + step.s[i])));
    vars.t[i] = max2(start_affine_multiplier_min,
                     fabs(ParOptRealPart(vars.t[i] + step.t[i])));
    vars.zs[i] = max2(start_affine_multiplier_min,
                      fabs(ParOptRealPart(vars.zs[i] + step.zs[i])));
    vars.zt[i] = max2(start_affine_multiplier_min,
                      fabs(ParOptRealPart(vars.zt[i] + step.zt[i])));
  }

  // Copy the values
  if (nwcon > 0) {
    ParOptScalar *zw, *sw, *tw, *zsw, *ztw;
    vars.zw->getArray(&zw);
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    vars.zsw->getArray(&zsw);
    vars.ztw->getArray(&ztw);

    ParOptScalar *yzw, *ysw, *ytw, *yzsw, *yztw;
    step.zw->getArray(&yzw);
    step.sw->getArray(&ysw);
    step.tw->getArray(&ytw);
    step.zsw->getArray(&yzsw);
    step.ztw->getArray(&yztw);

    for (int i = 0; i < nwcon; i++) {
      zw[i] = zw[i] + yzw[i];
      sw[i] = max2(start_affine_multiplier_min,
                   fabs(ParOptRealPart(sw[i] + ysw[i])));
      tw[i] = max2(start_affine_multiplier_min,
                   fabs(ParOptRealPart(tw[i] + ytw[i])));
      zsw[i] = max2(start_affine_multiplier_min,
                    fabs(ParOptRealPart(zsw[i] + yzsw[i])));
      ztw[i] = max2(start_affine_multiplier_min,
                    fabs(ParOptRealPart(ztw[i] + yztw[i])));
    }
  }

  if (use_lower) {
    ParOptScalar *zlvals, *pzlvals;
    vars.zl->getArray(&zlvals);
    step.zl->getArray(&pzlvals);
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        zlvals[i] = max2(start_affine_multiplier_min,
                         fabs(ParOptRealPart(zlvals[i] + pzlvals[i])));
      }
    }
  }
  if (use_upper) {
    ParOptScalar *zuvals, *pzuvals;
    vars.zu->getArray(&zuvals);
    step.zu->getArray(&pzuvals);
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        zuvals[i] = max2(start_affine_multiplier_min,
                         fabs(ParOptRealPart(zuvals[i] + pzuvals[i])));
      }
    }
  }

  // Set the initial barrier parameter
  barrier_param = ParOptRealPart(computeComp(vars));
}

/*
  Evaluate the directional derivative of the objective and barrier
  terms (the merit function without the penalty term)

  This is used by the GMRES preconditioned iteration to determine
  when we have a descent direction.

  Note that this call is collective on all procs in comm and uses
  the values in the primal variables (x, s, t) and the primal
  directions (px, ps, pt).
*/
ParOptScalar ParOptInteriorPoint::evalObjBarrierDeriv(ParOptVars &vars,
                                                      ParOptVars &step) {
  const double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  ParOptScalar *xvals, *pxvals, *lbvals, *ubvals;
  vars.x->getArray(&xvals);
  step.x->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Compute the contribution from the bound variables.
  ParOptScalar pos_presult = 0.0, neg_presult = 0.0;

  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        if (ParOptRealPart(pxvals[i]) > 0.0) {
          pos_presult += rel_bound_barrier * pxvals[i] / (xvals[i] - lbvals[i]);
        } else {
          neg_presult += rel_bound_barrier * pxvals[i] / (xvals[i] - lbvals[i]);
        }
      }
    }
  }

  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        if (ParOptRealPart(pxvals[i]) > 0.0) {
          neg_presult -= rel_bound_barrier * pxvals[i] / (ubvals[i] - xvals[i]);
        } else {
          pos_presult -= rel_bound_barrier * pxvals[i] / (ubvals[i] - xvals[i]);
        }
      }
    }
  }

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0) {
    ParOptScalar *sw, *tw, *psw, *ptw;
    vars.sw->getArray(&sw);
    vars.tw->getArray(&tw);
    step.sw->getArray(&psw);
    step.tw->getArray(&ptw);

    for (int i = 0; i < nwcon; i++) {
      if (ParOptRealPart(psw[i]) > 0.0) {
        pos_presult += psw[i] / sw[i];
      } else {
        neg_presult += psw[i] / sw[i];
      }
      if (ParOptRealPart(ptw[i]) > 0.0) {
        pos_presult += ptw[i] / tw[i];
      } else {
        neg_presult += ptw[i] / tw[i];
      }
    }
  }

  // Sum up the result from all processors
  ParOptScalar input[2];
  ParOptScalar result[2];
  input[0] = pos_presult;
  input[1] = neg_presult;

  MPI_Allreduce(input, result, 2, PAROPT_MPI_TYPE, MPI_SUM, comm);

  // Extract the result of the summation over all processors
  pos_presult = result[0];
  neg_presult = result[1];

  for (int i = 0; i < ncon; i++) {
    // Add the terms from the s-slack variables
    if (ParOptRealPart(step.s[i]) > 0.0) {
      pos_presult += step.s[i] / vars.s[i];
    } else {
      neg_presult += step.s[i] / vars.s[i];
    }

    // Add the terms from the t-slack variables
    if (ParOptRealPart(step.t[i]) > 0.0) {
      pos_presult += step.t[i] / vars.t[i];
    } else {
      neg_presult += step.t[i] / vars.t[i];
    }
  }

  ParOptScalar pmerit =
      g->dot(step.x) - barrier_param * (pos_presult + neg_presult);

  for (int i = 0; i < ncon; i++) {
    pmerit += (penalty_gamma_s[i] * step.s[i] + penalty_gamma_t[i] * step.t[i]);
  }

  pmerit += penalty_gamma_sw->dot(step.sw) + penalty_gamma_tw->dot(step.tw);

  // px now contains the current estimate of the step in the design
  // variables.
  return pmerit;
}

/*
  This function approximately solves the linearized KKT system with
  Hessian-vector products using right-preconditioned GMRES.  This
  procedure uses a preconditioner formed from a portion of the KKT
  system.  Grouping the Lagrange multipliers and slack variables from
  the remaining portion of the matrix, yields the following
  decomposition:

  K = [ B; A ] + [ H - B; 0 ]
  .   [ E; C ]   [     0; 0 ]

  Setting the precontioner as:

  M = [ B; A ]
  .   [ E; C ]

  We use right-preconditioning and solve the following system:

  K*M^{-1}*u = b

  where M*x = u, so we compute x = M^{-1}*u

  {[ I; 0 ] + [ H - B; 0 ]*M^{-1}}[ ux ] = [ bx ]
  {[ 0; I ] + [     0; 0 ]       }[ uy ]   [ by ]
*/
int ParOptInteriorPoint::computeKKTGMRESStep(ParOptVars &vars, ParOptVars &res,
                                             ParOptVars &step,
                                             ParOptScalar *ztmp,
                                             ParOptVec *xtmp1, ParOptVec *xtmp2,
                                             ParOptVec *wtmp, double rtol,
                                             double atol, int use_qn) {
  // Set the output level
  const int output_level = options->getIntOption("output_level");

  // Check that the subspace has been allocated
  if (gmres_subspace_size <= 0) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == opt_root) {
      fprintf(stderr, "ParOpt error: gmres_subspace_size not set\n");
    }
    return 0;
  }

  // Initialize the data from the gmres object
  ParOptScalar *H = gmres_H;
  ParOptScalar *alpha = gmres_alpha;
  ParOptScalar *gres = gmres_res;
  ParOptScalar *y = gmres_y;
  ParOptScalar *fproj = gmres_fproj;
  ParOptScalar *aproj = gmres_aproj;
  ParOptScalar *awproj = gmres_awproj;
  ParOptScalar *Qcos = &gmres_Q[0];
  ParOptScalar *Qsin = &gmres_Q[gmres_subspace_size];
  ParOptVec **W = gmres_W;

  // Compute the beta factor: the product of the diagonal terms
  // after normalization
  ParOptScalar beta = 0.0;
  for (int i = 0; i < ncon; i++) {
    beta += res.z[i] * res.z[i];
    beta += res.s[i] * res.s[i];
    beta += res.t[i] * res.t[i];
    beta += res.zs[i] * res.zs[i];
    beta += res.zt[i] * res.zt[i];
  }
  if (use_lower) {
    beta += res.zl->dot(res.zl);
  }
  if (use_upper) {
    beta += res.zu->dot(res.zu);
  }
  if (nwcon > 0) {
    beta += res.zw->dot(res.zw);
    beta += res.sw->dot(res.sw);
    beta += res.tw->dot(res.tw);
    beta += res.zsw->dot(res.zsw);
    beta += res.ztw->dot(res.ztw);
  }

  // Compute the norm of the initial vector
  ParOptScalar bnorm = sqrt(res.x->dot(res.x) + beta);

  // Broadcast the norm of the residuals and the beta parameter to
  // keep things consistent across processors
  ParOptScalar temp[2];
  temp[0] = bnorm;
  temp[1] = beta;
  MPI_Bcast(temp, 2, PAROPT_MPI_TYPE, opt_root, comm);

  bnorm = temp[0];
  beta = temp[1];

  // Compute the final value of the beta term
  beta *= 1.0 / (bnorm * bnorm);

  // Compute the inverse of the l2 norm of the dense inequality constraint
  // infeasibility and store it for later computations.
  ParOptScalar cinfeas = 0.0, cscale = 0.0;
  for (int i = 0; i < ncon; i++) {
    cinfeas += (c[i] - vars.s[i] + vars.t[i]) * (c[i] - vars.s[i] + vars.t[i]);
  }
  if (ParOptRealPart(cinfeas) != 0.0) {
    cinfeas = sqrt(cinfeas);
    cscale = 1.0 / cinfeas;
  }

  // Compute the inverse of the l2 norm of the sparse constraint
  // infeasibility and store it.
  ParOptScalar cwinfeas = 0.0, cwscale = 0.0;
  if (nwcon > 0) {
    cwinfeas = sqrt(res.zw->dot(res.zw));
    if (ParOptRealPart(cwinfeas) != 0.0) {
      cwscale = 1.0 / cwinfeas;
    }
  }

  // Initialize the residual norm
  gres[0] = bnorm;
  W[0]->copyValues(res.x);
  W[0]->scale(1.0 / gres[0]);
  alpha[0] = 1.0;

  // Keep track of the actual number of iterations
  int niters = 0;

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root && outfp && output_level > 0) {
    fprintf(outfp, "%5s %4s %4s %7s %7s %8s %8s gmres rtol: %7.1e\n", "gmres",
            "nhvc", "iter", "res", "rel", "fproj", "cproj", rtol);
    fprintf(outfp, "      %4d %4d %7.1e %7.1e\n", nhvec, 0,
            fabs(ParOptRealPart(gres[0])), 1.0);
  }

  for (int i = 0; i < gmres_subspace_size; i++) {
    // Compute M^{-1}*[ W[i], alpha[i]*yc, ... ]
    // Get the size of the limited-memory BFGS subspace
    ParOptScalar b0;
    const ParOptScalar *d, *M;
    ParOptVec **Z;
    int size = 0;
    if (qn && use_qn) {
      size = qn->getCompactMat(&b0, &d, &M, &Z);
    }

    // Solve the first part of the equation
    solveKKTDiagSystem(vars, W[i], alpha[i] / bnorm, res, step, xtmp2, wtmp);

    if (size > 0) {
      // dz = Z^{T}*xt1
      step.x->mdot(Z, size, ztmp);

      // Compute dz <- Ce^{-1}*dz
      int one = 1, info = 0;
      LAPACKdgetrs("N", &size, &one, Ce, &size, cpiv, ztmp, &size, &info);

      // Compute rx = Z^{T}*dz
      xtmp2->zeroEntries();
      for (int k = 0; k < size; k++) {
        xtmp2->axpy(ztmp[k], Z[k]);
      }

      // Solve the digaonal system again, this time simplifying the
      // result due to the structure of the right-hand-side.  Note
      // that this call uses W[i+1] as a temporary vector.
      solveKKTDiagSystem(vars, xtmp2, xtmp1, ztmp, W[i + 1], wtmp);

      // Add the final contributions
      step.x->axpy(-1.0, xtmp1);
    }

    // px now contains the current estimate of the step in the design
    // variables.
    fproj[i] = evalObjBarrierDeriv(vars, step);

    // Compute the directional derivative of the l2 constraint infeasibility
    // along the direction px.
    aproj[i] = 0.0;
    for (int j = 0; j < ncon; j++) {
      ParOptScalar cj_deriv = (Ac[j]->dot(step.x) - step.s[j] + step.t[j]);
      aproj[i] -= cscale * res.z[j] * cj_deriv;
    }

    // Add the contributions from the sparse constraints (if any are defined)
    awproj[i] = 0.0;
    if (nwcon > 0) {
      // rzw = -(cw - sw + tw)
      xtmp1->zeroEntries();
      prob->addSparseJacobianTranspose(1.0, vars.x, res.zw, xtmp1);
      awproj[i] = -cwscale * step.x->dot(xtmp1);
      awproj[i] += cwscale * res.zw->dot(step.sw);
      awproj[i] -= cwscale * res.zw->dot(step.tw);
    }

    // Compute the vector product with the exact Hessian
    prob->evalHvecProduct(vars.x, vars.z, vars.zw, step.x, W[i + 1]);
    nhvec++;

    // Add the term -B*W[i]
    if (qn && use_qn) {
      qn->multAdd(-1.0, step.x, W[i + 1]);
    }

    // Add the term from the diagonal
    W[i + 1]->axpy(1.0, W[i]);

    // Set the value of the scalar
    alpha[i + 1] = alpha[i];

    // Build the orthogonal factorization MGS
    int hptr = (i + 1) * (i + 2) / 2 - 1;
    for (int j = i; j >= 0; j--) {
      H[j + hptr] = W[i + 1]->dot(W[j]) + beta * alpha[i + 1] * alpha[j];

      W[i + 1]->axpy(-H[j + hptr], W[j]);
      alpha[i + 1] -= H[j + hptr] * alpha[j];
    }

    // Compute the norm of the combined vector
    H[i + 1 + hptr] =
        sqrt(W[i + 1]->dot(W[i + 1]) + beta * alpha[i + 1] * alpha[i + 1]);

    // Normalize the combined vector
    W[i + 1]->scale(1.0 / H[i + 1 + hptr]);
    alpha[i + 1] *= 1.0 / H[i + 1 + hptr];

    // Apply the existing part of Q to the new components of the
    // Hessenberg matrix
    for (int k = 0; k < i; k++) {
      ParOptScalar h1 = H[k + hptr];
      ParOptScalar h2 = H[k + 1 + hptr];
      H[k + hptr] = h1 * Qcos[k] + h2 * Qsin[k];
      H[k + 1 + hptr] = -h1 * Qsin[k] + h2 * Qcos[k];
    }

    // Now, compute the rotation for the new column that was just added
    ParOptScalar h1 = H[i + hptr];
    ParOptScalar h2 = H[i + 1 + hptr];
    ParOptScalar sq = sqrt(h1 * h1 + h2 * h2);

    Qcos[i] = h1 / sq;
    Qsin[i] = h2 / sq;
    H[i + hptr] = h1 * Qcos[i] + h2 * Qsin[i];
    H[i + 1 + hptr] = -h1 * Qsin[i] + h2 * Qcos[i];

    // Update the residual
    h1 = gres[i];
    gres[i] = h1 * Qcos[i];
    gres[i + 1] = -h1 * Qsin[i];

    niters++;

    // Check the contribution to the projected derivative terms. First
    // evaluate the weights y[] for each
    for (int j = niters - 1; j >= 0; j--) {
      y[j] = gres[j];
      for (int k = j + 1; k < niters; k++) {
        int hptr = (k + 1) * (k + 2) / 2 - 1;
        y[j] = y[j] - H[j + hptr] * y[k];
      }

      int hptr = (j + 1) * (j + 2) / 2 - 1;
      y[j] = y[j] / H[j + hptr];
    }

    // Compute the projection of the solution px on to the gradient
    // direction and the constraint Jacobian directions
    ParOptScalar fpr = 0.0, cpr = 0.0;
    for (int j = 0; j < niters; j++) {
      fpr += y[j] * fproj[j];
      cpr += y[j] * (aproj[j] + awproj[j]);
    }

    if (rank == opt_root && output_level > 0) {
      fprintf(outfp, "      %4d %4d %7.1e %7.1e %8.1e %8.1e\n", nhvec, i + 1,
              fabs(ParOptRealPart(gres[i + 1])),
              fabs(ParOptRealPart(gres[i + 1] / bnorm)), ParOptRealPart(fpr),
              ParOptRealPart(cpr));
      fflush(outfp);
    }

    // Check first that the direction is a candidate descent direction
    int constraint_descent = 0;
    if (ParOptRealPart(cpr) <= -0.01 * ParOptRealPart(cinfeas + cwinfeas)) {
      constraint_descent = 1;
    }
    if (ParOptRealPart(fpr) < 0.0 || constraint_descent) {
      // Check for convergence
      if (fabs(ParOptRealPart(gres[i + 1])) < atol ||
          fabs(ParOptRealPart(gres[i + 1])) < rtol * ParOptRealPart(bnorm)) {
        break;
      }
    }
  }

  // Now, compute the solution - the linear combination of the
  // Arnoldi vectors. H is now an upper triangular matrix.
  for (int i = niters - 1; i >= 0; i--) {
    for (int j = i + 1; j < niters; j++) {
      int hptr = (j + 1) * (j + 2) / 2 - 1;
      gres[i] = gres[i] - H[i + hptr] * gres[j];
    }

    int hptr = (i + 1) * (i + 2) / 2 - 1;
    gres[i] = gres[i] / H[i + hptr];
  }

  // Compute the linear combination of the vectors
  // that will be the output
  W[0]->scale(gres[0]);
  ParOptScalar gamma = gres[0] * alpha[0];

  for (int i = 1; i < niters; i++) {
    W[0]->axpy(gres[i], W[i]);
    gamma += gres[i] * alpha[i];
  }

  // Normalize the gamma parameter
  gamma /= bnorm;

  // Copy the values to res.x
  res.x->copyValues(W[0]);

  // Scale the right-hand-side by gamma
  for (int i = 0; i < ncon; i++) {
    res.z[i] *= gamma;
    res.s[i] *= gamma;
    res.t[i] *= gamma;
    res.zs[i] *= gamma;
    res.zt[i] *= gamma;
  }

  res.zl->scale(gamma);
  res.zu->scale(gamma);
  if (nwcon > 0) {
    res.zw->scale(gamma);
    res.sw->scale(gamma);
    res.tw->scale(gamma);
    res.ztw->scale(gamma);
    res.zsw->scale(gamma);
  }

  // Apply M^{-1} to the result to obtain the final answer
  // After this point the residuals are no longer required.
  solveKKTDiagSystem(vars, res, step, xtmp1, wtmp);

  // Get the size of the limited-memory BFGS subspace
  ParOptScalar b0;
  const ParOptScalar *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (qn && use_qn) {
    size = qn->getCompactMat(&b0, &d, &M, &Z);
  }

  if (size > 0) {
    // dz = Z^{T}*px
    step.x->mdot(Z, size, ztmp);

    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one, Ce, &size, cpiv, ztmp, &size, &info);

    // Compute xtmp1 = Z^{T}*dz
    xtmp1->zeroEntries();
    for (int i = 0; i < size; i++) {
      xtmp1->axpy(ztmp[i], Z[i]);
    }

    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(vars, xtmp1, res, xtmp2, wtmp);

    // Add the final contributions
    step.x->axpy(-1.0, res.x);
    step.zl->axpy(-1.0, res.zl);
    step.zu->axpy(-1.0, res.zu);

    step.sw->axpy(-1.0, res.sw);
    step.tw->axpy(-1.0, res.tw);
    step.zw->axpy(-1.0, res.zw);
    step.zsw->axpy(-1.0, res.zsw);
    step.ztw->axpy(-1.0, res.ztw);

    // Add the terms from the slacks/multipliers
    for (int i = 0; i < ncon; i++) {
      step.z[i] -= res.z[i];
      step.s[i] -= res.s[i];
      step.t[i] -= res.t[i];
      step.zs[i] -= res.zs[i];
      step.zt[i] -= res.zt[i];
    }
  }

  // Add the contributions from the objective and dense constraints
  ParOptScalar fpr = evalObjBarrierDeriv(vars, step);
  ParOptScalar cpr = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar deriv = (Ac[i]->dot(step.x) - step.s[i] + step.t[i]);
    cpr += cscale * (c[i] - vars.s[i] + vars.t[i]) * deriv;
  }

  // Add the contributions from the sparse constraints
  if (nwcon > 0) {
    // Compute the residual rzw = (cw - sw + tw)
    prob->evalSparseCon(vars.x, res.zw);
    res.zw->axpy(-1.0, vars.sw);
    res.zw->axpy(1.0, vars.tw);

    xtmp1->zeroEntries();
    prob->addSparseJacobianTranspose(1.0, vars.x, res.zw, xtmp1);
    cpr += cwscale * step.x->dot(xtmp1);
    cpr -= cwscale * step.sw->dot(res.zw);
    cpr -= cwscale * step.tw->dot(res.zw);
  }

  if (rank == opt_root && output_level > 0) {
    fprintf(outfp, "      %9s %7s %7s %8.1e %8.1e\n", "final", " ", " ",
            ParOptRealPart(fpr), ParOptRealPart(cpr));
    fflush(outfp);
  }

  // Check if this should be considered a failure based on the
  // convergence criteria
  if (ParOptRealPart(fpr) < 0.0 ||
      ParOptRealPart(cpr) < -0.01 * ParOptRealPart(cinfeas + cwinfeas)) {
    return niters;
  }

  // We've failed.
  return -niters;
}

/*
  Check that the gradients match along a projected direction.
*/
void ParOptInteriorPoint::checkGradients(double dh) {
  const int use_hvec_product = options->getBoolOption("use_hvec_product");
  prob->checkGradients(dh, variables.x, use_hvec_product);
}

/*
  Check that the step is correct. This code computes the maximum
  component of the following residual equations and prints out the
  result to the screen:

  H*px - Ac^{T}*pz - pzl + pzu + (g - Ac^{T}*z - zl + zu) = 0
  A*px - ps + (c - s) = 0
  z*ps + s*pz + (z*s - mu) = 0
  zl*px + (x - lb)*pzl + (zl*(x - lb) - mu) = 0
  zu*px + (ub - x)*pzu + (zu*(ub - x) - mu) = 0
*/
void ParOptInteriorPoint::checkKKTStep(ParOptVars &vars, ParOptVars &step,
                                       ParOptVars &res, int iteration,
                                       int is_newton) {
  // Diagonal coefficient used for the quasi-Newton Hessian aprpoximation
  const double qn_sigma = options->getFloatOption("qn_sigma");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const int sequential_linear_method =
      options->getBoolOption("sequential_linear_method");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  vars.x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  vars.zl->getArray(&zlvals);
  vars.zu->getArray(&zuvals);

  // Retrieve the values of the steps
  ParOptScalar *pxvals, *pzlvals, *pzuvals;
  step.x->getArray(&pxvals);
  step.zl->getArray(&pzlvals);
  step.zu->getArray(&pzuvals);

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == opt_root) {
    printf("\nResidual step check for iteration %d:\n", iteration);
  }

  // Check the first residual equation
  if (is_newton) {
    prob->evalHvecProduct(vars.x, vars.z, vars.zw, step.x, res.x);
  } else if (use_diag_hessian) {
    prob->evalHessianDiag(vars.x, vars.z, vars.zw, hdiag);

    // Retrieve the components of px and hdiag
    ParOptScalar *rxvals, *hvals;
    res.x->getArray(&rxvals);
    hdiag->getArray(&hvals);
    for (int i = 0; i < nvars; i++) {
      rxvals[i] = pxvals[i] * hvals[i];
    }
  } else {
    if (qn && !sequential_linear_method) {
      qn->mult(step.x, res.x);
      res.x->axpy(qn_sigma, step.x);
    } else {
      res.x->zeroEntries();
    }
  }
  for (int i = 0; i < ncon; i++) {
    res.x->axpy(-step.z[i] - vars.z[i], Ac[i]);
  }
  if (use_lower) {
    res.x->axpy(-1.0, step.zl);
    res.x->axpy(-1.0, vars.zl);
  }
  if (use_upper) {
    res.x->axpy(1.0, step.zu);
    res.x->axpy(1.0, vars.zu);
  }
  res.x->axpy(1.0, g);

  // Add the contributions from the constraint
  if (nwcon > 0) {
    prob->addSparseJacobianTranspose(-1.0, vars.x, vars.zw, res.x);
    prob->addSparseJacobianTranspose(-1.0, vars.x, step.zw, res.x);
  }
  double max_val = res.x->maxabs();

  if (rank == opt_root) {
    printf(
        "max |(H + sigma*I)*px - Ac^{T}*pz - Aw^{T}*pzw - pzl + pzu + "
        "(g - Ac^{T}*z - Aw^{T}*zw - zl + zu)|: %10.4e\n",
        max_val);
  }

  // Compute the residuals from the weighting constraints
  if (nwcon > 0) {
    prob->evalSparseCon(vars.x, res.zw);
    prob->addSparseJacobian(1.0, vars.x, step.x, res.zw);
    res.zw->axpy(-1.0, vars.sw);
    res.zw->axpy(-1.0, step.sw);
    res.zw->axpy(1.0, vars.tw);
    res.zw->axpy(1.0, step.tw);

    max_val = res.zw->maxabs();
    if (rank == opt_root) {
      printf("max |cw(x) - sw + tw + Aw*pw - psw + ptw|: %10.4e\n", max_val);
    }
  }

  // Find the maximum value of the residual equations
  // for the constraints
  max_val = 0.0;
  step.x->mdot(Ac, ncon, res.z);
  for (int i = 0; i < ncon; i++) {
    ParOptScalar val =
        res.z[i] - step.s[i] + step.t[i] + (c[i] - vars.s[i] + vars.t[i]);
    if (fabs(ParOptRealPart(val)) > max_val) {
      max_val = fabs(ParOptRealPart(val));
    }
  }
  if (rank == opt_root) {
    printf("max |A*px - ps + pt + (c - s + t)|: %10.4e\n", max_val);
  }

  // Find the maximum value of the residual equations for
  // the dual slack variables
  max_val = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar val =
        penalty_gamma_s[i] - vars.zs[i] + vars.z[i] - step.zs[i] + step.z[i];
    if (fabs(ParOptRealPart(val)) > max_val) {
      max_val = fabs(ParOptRealPart(val));
    }
  }
  if (rank == opt_root) {
    printf("max |gamma - zs + z - pzs + pz|: %10.4e\n", max_val);
  }

  max_val = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar val =
        penalty_gamma_t[i] - vars.zt[i] - vars.z[i] - step.zt[i] - step.z[i];
    if (fabs(ParOptRealPart(val)) > max_val) {
      max_val = fabs(ParOptRealPart(val));
    }
  }
  if (rank == opt_root) {
    printf("max |gamma - zt - z - pzt - pz|: %10.4e\n", max_val);
  }

  if (nwcon > 0) {
    res.zsw->copyValues(penalty_gamma_sw);
    res.zsw->axpy(-1.0, vars.zsw);
    res.zsw->axpy(1.0, vars.zw);
    res.zsw->axpy(-1.0, step.zsw);
    res.zsw->axpy(1.0, step.zw);
    max_val = res.zsw->maxabs();
    if (rank == opt_root) {
      printf("max |gamma - zsw + zw - pzsw + pzw|: %10.4e\n", max_val);
    }

    res.ztw->copyValues(penalty_gamma_tw);
    res.ztw->axpy(-1.0, vars.ztw);
    res.ztw->axpy(-1.0, vars.zw);
    res.ztw->axpy(-1.0, step.ztw);
    res.ztw->axpy(-1.0, step.zw);
    max_val = res.ztw->maxabs();
    if (rank == opt_root) {
      printf("max |gamma - ztw - zw - pztw - pzw|: %10.4e\n", max_val);
    }
  }

  max_val = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar val = vars.t[i] * step.zt[i] + vars.zt[i] * step.t[i] +
                       (vars.t[i] * vars.zt[i] - barrier_param);
    if (fabs(ParOptRealPart(val)) > max_val) {
      max_val = fabs(ParOptRealPart(val));
    }
  }
  if (rank == opt_root) {
    printf("max |T*pzt + Zt*pt + (T*zt - mu)|: %10.4e\n", max_val);
  }

  max_val = 0.0;
  for (int i = 0; i < ncon; i++) {
    ParOptScalar val = vars.s[i] * step.zs[i] + vars.zs[i] * step.s[i] +
                       (vars.zs[i] * vars.s[i] - barrier_param);
    if (fabs(ParOptRealPart(val)) > max_val) {
      max_val = fabs(ParOptRealPart(val));
    }
  }
  if (rank == opt_root) {
    printf("max |Zs*ps + S*pz + (S*zs - mu)|: %10.4e\n", max_val);
  }

  if (nwcon > 0) {
    ParOptScalar *rztw, *tw, *ztw, *ptw, *pztw;
    res.ztw->getArray(&rztw);
    vars.tw->getArray(&tw);
    vars.ztw->getArray(&ztw);
    step.tw->getArray(&ptw);
    step.ztw->getArray(&pztw);
    for (int i = 0; i < nwcon; i++) {
      rztw[i] =
          tw[i] * pztw[i] + ztw[i] * ptw[i] + (tw[i] * ztw[i] - barrier_param);
    }
    max_val = res.ztw->maxabs();
    if (rank == opt_root) {
      printf("max |Tw*pztw + Ztw*ptw + (Tw*ztw - mu)|: %10.4e\n", max_val);
    }

    ParOptScalar *rzsw, *sw, *zsw, *psw, *pzsw;
    res.zsw->getArray(&rzsw);
    vars.sw->getArray(&sw);
    vars.zsw->getArray(&zsw);
    step.sw->getArray(&psw);
    step.zsw->getArray(&pzsw);
    for (int i = 0; i < nwcon; i++) {
      rzsw[i] =
          sw[i] * pzsw[i] + zsw[i] * psw[i] + (zsw[i] * sw[i] - barrier_param);
    }
    max_val = res.ztw->maxabs();
    if (rank == opt_root) {
      printf("max |Zsw*psw + Sw*pzw + (Sw*zsw - mu)|: %10.4e\n", max_val);
    }
  }

  // Find the maximum of the residual equations for the
  // lower-bound dual variables
  max_val = 0.0;
  if (use_lower) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(lbvals[i]) > -max_bound_value) {
        ParOptScalar val =
            (zlvals[i] * pxvals[i] + (xvals[i] - lbvals[i]) * pzlvals[i] +
             (zlvals[i] * (xvals[i] - lbvals[i]) - barrier_param));
        if (fabs(ParOptRealPart(val)) > max_val) {
          max_val = fabs(ParOptRealPart(val));
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);

  if (rank == opt_root && use_lower) {
    printf("max |Zl*px + (X - LB)*pzl + (Zl*(x - lb) - mu)|: %10.4e\n",
           max_val);
  }

  // Find the maximum value of the residual equations for the
  // upper-bound dual variables
  max_val = 0.0;
  if (use_upper) {
    for (int i = 0; i < nvars; i++) {
      if (ParOptRealPart(ubvals[i]) < max_bound_value) {
        ParOptScalar val =
            (-zuvals[i] * pxvals[i] + (ubvals[i] - xvals[i]) * pzuvals[i] +
             (zuvals[i] * (ubvals[i] - xvals[i]) - barrier_param));
        if (fabs(ParOptRealPart(val)) > max_val) {
          max_val = fabs(ParOptRealPart(val));
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);

  if (rank == opt_root && use_upper) {
    printf("max |-Zu*px + (UB - X)*pzu + (Zu*(ub - x) - mu)|: %10.4e\n",
           max_val);
  }
}
