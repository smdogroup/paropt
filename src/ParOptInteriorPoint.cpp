#include <math.h>
#include <string.h>
#include "ParOptComplexStep.h"
#include "ParOptBlasLapack.h"
#include "ParOptInteriorPoint.h"

/*
  Static helper functions
*/
ParOptScalar max2( const ParOptScalar a, const ParOptScalar b ){
  if (ParOptRealPart(a) > ParOptRealPart(b)){
    return a;
  }
  return b;
}

ParOptScalar min2( const ParOptScalar a, const ParOptScalar b ){
  if (ParOptRealPart(a) < ParOptRealPart(b)){
    return a;
  }
  return b;
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
ParOptInteriorPoint::ParOptInteriorPoint( ParOptProblem *_prob,
                                          ParOptOptions *_options ){
  // Set the problem object
  prob = _prob;
  prob->incref();

  if (_options){
    options = _options;
  }
  else {
    options = new ParOptOptions();
    addDefaultOptions(options);
  }
  options->incref();

  // Record the communicator
  comm = prob->getMPIComm();
  opt_root = 0;

  // Get the number of variables/constraints
  prob->getProblemSizes(&nvars, &ncon, &nwcon, &nwblock);

  // Are these sparse inequalties or equalities?
  sparse_inequality = prob->isSparseInequality();
  dense_inequality = prob->isDenseInequality();
  use_lower = prob->useLowerBounds();
  use_upper = prob->useUpperBounds();

  // Assign the values from the sparsity constraints
  if (nwcon > 0 && nwcon % nwblock != 0){
    fprintf(stderr, "ParOpt: Weighted block size inconsistent\n");
  }

  // Calculate the total number of variable across all processors
  int size, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Allocate space to store the variable ranges
  var_range = new int[ size+1 ];
  wcon_range = new int[ size+1 ];
  var_range[0] = 0;
  wcon_range[0] = 0;

  // Count up the displacements/variable ranges
  MPI_Allgather(&nvars, 1, MPI_INT, &var_range[1], 1, MPI_INT, comm);
  MPI_Allgather(&nwcon, 1, MPI_INT, &wcon_range[1], 1, MPI_INT, comm);

  for ( int k = 0; k < size; k++ ){
    var_range[k+1] += var_range[k];
    wcon_range[k+1] += wcon_range[k];
  }

  // Set the total number of variables
  nvars_total = var_range[size];

  // Set the values of the variables/bounds
  x = prob->createDesignVec(); x->incref();
  lb = prob->createDesignVec(); lb->incref();
  ub = prob->createDesignVec(); ub->incref();

  // Allocate storage space for the variables etc.
  zl = prob->createDesignVec(); zl->incref();
  zu = prob->createDesignVec(); zu->incref();

  // Allocate space for the sparse constraints
  zw = prob->createConstraintVec(); zw->incref();
  sw = prob->createConstraintVec(); sw->incref();

  // Set the initial values of the Lagrange multipliers
  z = new ParOptScalar[ ncon ];
  s = new ParOptScalar[ ncon ];

  // Set the multipliers for l1-penalty term
  zt = new ParOptScalar[ ncon ];
  t = new ParOptScalar[ ncon ];

  // Zero the initial values
  memset(z, 0, ncon*sizeof(ParOptScalar));
  memset(s, 0, ncon*sizeof(ParOptScalar));
  memset(zt, 0, ncon*sizeof(ParOptScalar));
  memset(t, 0, ncon*sizeof(ParOptScalar));

  // Allocate space for the steps
  px = prob->createDesignVec(); px->incref();
  pzl = prob->createDesignVec(); pzl->incref();
  pzu = prob->createDesignVec(); pzu->incref();
  pz = new ParOptScalar[ ncon ];
  ps = new ParOptScalar[ ncon ];
  pzt = new ParOptScalar[ ncon ];
  pt = new ParOptScalar[ ncon ];
  pzw = prob->createConstraintVec(); pzw->incref();
  psw = prob->createConstraintVec(); psw->incref();

  // Allocate space for the residuals
  rx = prob->createDesignVec(); rx->incref();
  rzl = prob->createDesignVec(); rzl->incref();
  rzu = prob->createDesignVec(); rzu->incref();
  rc = new ParOptScalar[ ncon ];
  rs = new ParOptScalar[ ncon ];
  rt = new ParOptScalar[ ncon ];
  rzt = new ParOptScalar[ ncon ];
  rcw = prob->createConstraintVec(); rcw->incref();
  rsw = prob->createConstraintVec(); rsw->incref();

  // Allocate space for the Quasi-Newton updates
  y_qn = prob->createDesignVec(); y_qn->incref();
  s_qn = prob->createDesignVec(); s_qn->incref();

  // Allocate vectors for the weighting constraints
  wtemp = prob->createConstraintVec(); wtemp->incref();
  xtemp = prob->createDesignVec(); xtemp->incref();

  // Allocate space for the block-diagonal matrix
  Cw = new ParOptScalar[ nwcon*(nwblock+1)/2 ];

  // Allocate space for off-diagonal entries
  Ew = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ew[i] = prob->createConstraintVec();
    Ew[i]->incref();
  }

  // Allocate space for the Dmatrix
  Dmat = new ParOptScalar[ ncon*ncon ];
  dpiv = new int[ ncon ];

  // Allocate the quasi-Newton approximation
  const char *qn_type = options->getEnumOption("qn_type");
  const int qn_subspace_size = options->getIntOption("qn_subspace_size");

  qn = NULL;
  if (strcmp(qn_type, "bfgs") == 0){
    qn = new ParOptLBFGS(prob, qn_subspace_size);
    qn->incref();
  }
  else if (strcmp(qn_type, "sr1") == 0){
    qn = new ParOptLSR1(prob, qn_subspace_size);
    qn->incref();
  }

  hdiag = NULL;

  // Get the maximum subspace size
  int max_qn_size = 0;
  if (qn){
    max_qn_size = qn->getMaxLimitedMemorySize();
  }

  if (max_qn_size > 0){
    // Allocate storage for bfgs/constraint sized things
    int zsize = max_qn_size;
    if (ncon > zsize){
      zsize = ncon;
    }
    ztemp = new ParOptScalar[ zsize ];

    // Allocate space for the Ce matrix
    Ce = new ParOptScalar[ max_qn_size*max_qn_size ];
    cpiv = new int[ max_qn_size ];
  }
  else {
    ztemp = NULL;
    Ce = NULL;
    cpiv = NULL;
  }

  // Allocate space for the diagonal matrix components
  Cvec = prob->createDesignVec();
  Cvec->incref();

  // Set the value of the objective
  fobj = 0.0;

  // Set the constraints to zero
  c = new ParOptScalar[ ncon ];
  memset(c, 0, ncon*sizeof(ParOptScalar));

  // Set the objective and constraint gradients
  g = prob->createDesignVec();
  g->incref();

  Ac = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ac[i] = prob->createDesignVec();
    Ac[i]->incref();
  }

  // Set the default penalty values
  const double gamma = options->getFloatOption("penalty_gamma");
  penalty_gamma = new double[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    penalty_gamma[i] = gamma;
  }

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
  if (m > 0){
    setGMRESSubspaceSize(m);
  }

  // By default, set the file pointer to stdout. If a filename is specified,
  // set the new filename.
  outfp = NULL;
  if (rank == opt_root){
    outfp = stdout;
  }

  const char *filename = options->getStringOption("output_file");
  if (filename){
    setOutputFile(filename);
  }

  // Initialize the design variables and bounds
  initAndCheckDesignAndBounds();

  // Set initial values of the multipliers
  zl->set(1.0);
  zu->set(1.0);
  zw->set(1.0);
  sw->set(1.0);

  // Set the Largrange multipliers and slack variables associated
  // with the dense constraints to 1.0
  for ( int i = 0; i < ncon; i++ ){
    z[i] = 1.0;
    s[i] = 1.0;
    zt[i] = 1.0;
    t[i] = 1.0;
  }
}

/**
   Free the data allocated during the creation of the object
*/
ParOptInteriorPoint::~ParOptInteriorPoint(){
  prob->decref();
  options->decref();
  if (qn){
    qn->decref();
  }

  // Delete the variables and bounds
  x->decref();
  lb->decref();
  ub->decref();
  zl->decref();
  zu->decref();
  delete [] z;
  delete [] s;
  delete [] zt;
  delete [] t;
  zw->decref();
  sw->decref();

  // Delete the steps
  px->decref();
  pzl->decref();
  pzu->decref();
  delete [] pz;
  delete [] ps;
  delete [] pt;
  delete [] pzt;
  pzw->decref();
  psw->decref();

  // Delete the residuals
  rx->decref();
  rzl->decref();
  rzu->decref();
  delete [] rc;
  delete [] rs;
  delete [] rt;
  delete [] rzt;
  rcw->decref();
  rsw->decref();

  // Delete the quasi-Newton updates
  y_qn->decref();
  s_qn->decref();

  // Delete the temp data
  wtemp->decref();
  xtemp->decref();
  if (ztemp){ delete [] ztemp; }

  // Delete the matrix
  delete [] Cw;

  for ( int i = 0; i < ncon; i++ ){
    Ew[i]->decref();
  }
  delete [] Ew;

  // Delete the various matrices
  delete [] Dmat;
  delete [] dpiv;
  if (Ce){ delete [] Ce; }
  if (cpiv){ delete [] cpiv; }

  // Delete the vector of penalty parameters
  delete [] penalty_gamma;

  // Delete the diagonal matrix
  Cvec->decref();

  // Free the variable ranges
  delete [] var_range;
  delete [] wcon_range;

  // Free the diagonal hessian (if allocated)
  if (hdiag){
    hdiag->decref();
  }

  // Delete the constraint/gradient information
  delete [] c;
  g->decref();
  for ( int i = 0; i < ncon; i++ ){
    Ac[i]->decref();
  }
  delete [] Ac;

  // Delete the GMRES information if any
  if (gmres_subspace_size > 0){
    delete [] gmres_H;
    delete [] gmres_alpha;
    delete [] gmres_res;
    delete [] gmres_y;
    delete [] gmres_fproj;
    delete [] gmres_aproj;
    delete [] gmres_awproj;
    delete [] gmres_Q;

    // Delete the subspace
    for ( int i = 0; i < gmres_subspace_size; i++ ){
      gmres_W[i]->decref();
    }
    delete [] gmres_W;
  }

  // Close the output file if it's not stdout
  if (outfp && outfp != stdout){
    fclose(outfp);
  }
}

/**
  Add the default options to the input options object

  @param options The input options argument
*/
void ParOptInteriorPoint::addDefaultOptions( ParOptOptions *options ){
  // Set the string options
  options->addStringOption("output_file", "paropt.out", "Output file name");

  options->addStringOption("problem_name", NULL, "The problem name");

  // Set the float options
  options->addFloatOption("max_bound_value", 1e20, 0.0, 1e300,
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

  options->addFloatOption("penalty_descent_fraction", 0.3, 1e-6, 1.0,
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

  options->addFloatOption("rel_bound_barrier", 1.0, 0.0, 1e20,
    "Relative factor applied to barrier parameter for bound constraints");

  options->addFloatOption("min_fraction_to_boundary", 0.95, 0.0, 1.0,
    "Minimum fraction to the boundary rule < 1");

  options->addFloatOption("qn_sigma", 0.0, 0.0, 1e20,
    "Scalar added to the diagonal of the quasi-Newton approximation > 0");

  options->addFloatOption("nk_switch_tol", 1e-3, 0.0, 1.0,
    "Switch to the Newton-Krylov method at this residual tolerance");

  options->addFloatOption("eisenstat_walker_alpha", 1.5, 0.0, 2.0,
    "Exponent in the Eisenstat-Walker INK forcing equation");

  options->addFloatOption("eisenstat_walker_gamma", 1.0, 0.0, 1.0,
    "Multiplier in the Eisenstat-Walker INK forcing equation");

  options->addFloatOption("max_gmres_rtol", 0.1, 0.0, 1.0,
    "The maximum relative tolerance used for GMRES, above this "
    "the quasi-Newton approximation is used");

  options->addFloatOption("gmres_atol", 1e-30, 0.0, 1.0,
    "The absolute GMRES tolerance (almost never relevant)");

  options->addFloatOption("function_precision", 1e-10, 0.0, 1.0,
    "The absolute precision of the function and constraints");

  options->addFloatOption("design_precision", 1e-14, 0.0, 1.0,
    "The absolute precision of the design variables");

  options->addFloatOption("start_affine_multiplier_min", 1e-3, 0.0, 1e20,
    "Minimum multiplier for the affine step initialization strategy");

  // Set the boolean options
  options->addBoolOption("use_line_search", 1,
    "Perform or skip the line search");

  options->addBoolOption("use_backtracking_alpha", 0,
    "Perform a back-tracking line search");

  options->addBoolOption("sequential_linear_method", 0,
    "Discard the quasi-Newton approximation (but not "
    "necessarily the exact Hessian)");

  options->addBoolOption("use_quasi_newton_update", 1,
    "Update the quasi-Newton approximation at each iteration");

  options->addBoolOption("use_hvec_product", 0,
    "Use or do not use Hessian-vector products");

  options->addBoolOption("use_diag_hessian", 0,
    "Use or do not use the diagonal Hessian computation");

  options->addBoolOption("use_qn_gmres_precon", 1,
    "Use or do not use the quasi-Newton method as a preconditioner");

  options->addFloatOption("gradient_check_step_length", 1e-6, 0.0, 1.0,
    "Step length used to check the gradient");

  // Set the integer options
  options->addIntOption("qn_subspace_size", 10, 0, 1000,
    "The maximum dimension of the quasi-Newton approximation");

  options->addIntOption("max_major_iters", 5000, 0, 1000000,
    "The maximum number of major iterations before quiting");

  options->addIntOption("max_line_iters", 10, 1, 100,
    "Maximum number of line search iterations");

  options->addIntOption("gmres_subspace_size", 0, 0, 1000,
    "The subspace size for GMRES");

  options->addIntOption("write_output_frequency", 10, 0, 1000000,
    "Write out the solution file and checkpoint file "
    "at this frequency");

  options->addIntOption("gradient_verification_frequency",
    -1, -1000000, 1000000,
    "Print to screen the output of the gradient check "
    "at this frequency during an optimization");

  options->addIntOption("hessian_reset_freq", 0, 1000000, 1000000,
    "Do a hard reset of the Hessian at this specified major "
    "iteration frequency");

  options->addIntOption("output_level", 0, 0, 1000000,
    "Output level indicating how verbose the output should be");

  // Set the enumerated options
  const char *qn_type[3] = {"bfgs", "sr1", "none"};
  options->addEnumOption("qn_type", "bfgs", 3, qn_type,
    "The the of quasi-Newton approximation to use");

  const char *norm_options[3] = {"infinity", "l1", "l2"};
  options->addEnumOption("norm_type", "infinity", 3, norm_options,
    "The type of norm to use in all computations");

  const char *barrier_options[3] = {"monotone",
                                    "mehrotra",
                                    "complementarity_fraction"};
  options->addEnumOption("barrier_strategy",
    "monotone", 3, barrier_options,
    "The type of barrier update strategy to use");

  const char *start_options[3] = {"least_squares_multipliers",
                                  "affine_step",
                                  "no_start_strategy"};
  options->addEnumOption("starting_point_strategy",
    "affine_step", 3, start_options,
    "Initialize the Lagrange multiplier estimates and slack variables");
}

/**
  Get the options object associated with the interior point method

  @return The options object
*/
ParOptOptions* ParOptInteriorPoint::getOptions(){
  return options;
}

/**
   Reset the problem instance.

   The new problem instance must have the same number of constraints
   design variables, and design vector distribution as the original
   problem.

   @param problem ParOptProblem instance
*/
void ParOptInteriorPoint::resetProblemInstance( ParOptProblem *problem ){
  // Check to see if the new problem instance is congruent with
  // the previous problem instance - it has to be otherwise
  // we can't use it.
  int _nvars, _ncon, _nwcon, _nwblock;
  problem->getProblemSizes(&_nvars, &_ncon, &_nwcon, &_nwblock);

  if (_nvars != nvars || _ncon != ncon ||
      _nwcon != nwcon || _nwblock != nwblock){
    fprintf(stderr, "ParOpt: Incompatible problem instance\n");
    problem = NULL;
  }
  else {
    prob = problem;
  }
}

/**
   Retrieve the problem sizes from the underlying problem class

   @param _nvars the local number of variables
   @param _ncon the number of global constraints
   @param _nwcon the number of sparse constraints
   @param _nwblock the size of the sparse constraint block
*/
void ParOptInteriorPoint::getProblemSizes( int *_nvars, int *_ncon,
                                           int *_nwcon, int *_nwblock ){
  prob->getProblemSizes(_nvars, _ncon, _nwcon, _nwblock);
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
void ParOptInteriorPoint::getOptimizedPoint( ParOptVec **_x,
                                             ParOptScalar **_z,
                                             ParOptVec **_zw,
                                             ParOptVec **_zl,
                                             ParOptVec **_zu ){
  if (_x){
    *_x = x;
  }
  if (_z){
    *_z = NULL;
    if (ncon > 0){
      *_z = z;
    }
  }
  if (_zw){
    *_zw = NULL;
    if (nwcon > 0){
      *_zw = zw;
    }
  }
  if (_zl){
    *_zl = NULL;
    if (use_lower){
      *_zl = zl;
    }
  }
  if (_zu){
    *_zu = NULL;
    if (use_upper){
      *_zu = zu;
    }
  }
}

/**
   Retrieve the values of the optimized slack variables.

   Note that the dense inequality constraints are formualted as

   c(x) = s - t

   where s, t > 0. And the sparse inequality constraints are formulated
   as:

   cw(x) = sw

   where sw > 0. When equality rather than inequality constraints are
   present, sw may be NULL.

   @param _s the postive slack for the dense constraints
   @param _t the negative slack for the dense constraints
   @param _sw the slack variable vector for the sparse constraints
*/
void ParOptInteriorPoint::getOptimizedSlacks( ParOptScalar **_s,
                                              ParOptScalar **_t,
                                              ParOptVec **_sw ){
  if (_s){
    *_s = s;
  }
  if (_t){
    *_t = t;
  }
  if (_sw){
    *_sw = NULL;
    if (nwcon > 0){
      *_sw = sw;
    }
  }
}

/**
   Write out all of the options that have been set to a output
   stream.

   This function is typically invoked when the output summary file
   is written.

   @param fp an open file handle
*/
void ParOptInteriorPoint::printOptionSummary( FILE *fp ){
  if (fp){
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
int ParOptInteriorPoint::writeSolutionFile( const char *filename ){
  char *fname = new char[ strlen(filename)+1 ];
  strcpy(fname, filename);

  int fail = 1;
  MPI_File fp = NULL;
  MPI_File_open(comm, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE,
                MPI_INFO_NULL, &fp);

  if (fp){
    // Calculate the total number of variable across all processors
    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Successfull opened the file
    fail = 0;

    // Write the problem sizes on the root processor
    if (rank == opt_root){
      int var_sizes[3];
      var_sizes[0] = var_range[size];
      var_sizes[1] = wcon_range[size];
      var_sizes[2] = ncon;

      MPI_File_write(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);
      MPI_File_write(fp, &barrier_param, 1,
                     PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, z, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, s, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    }

    size_t offset = 3*sizeof(int) + (2*ncon+1)*sizeof(ParOptScalar);

    // Use the native representation for the data
    char datarep[] = "native";

    // Extract the design variables
    ParOptScalar *xvals;
    int xsize = x->getArray(&xvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                      datarep, MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], xvals, xsize,
                          PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(ParOptScalar);

    // Extract the lower Lagrange multipliers
    ParOptScalar *zlvals, *zuvals;
    zl->getArray(&zlvals);
    zu->getArray(&zuvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                      datarep, MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], zlvals, xsize,
                          PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(ParOptScalar);

    // Write out the upper Lagrange multipliers
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                      datarep, MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], zuvals, xsize,
                          PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(ParOptScalar);

    // Write out the extra constraint bounds
    if (wcon_range[size] > 0){
      ParOptScalar *zwvals, *swvals;
      int nwsize = zw->getArray(&zwvals);
      sw->getArray(&swvals);
      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                        datarep, MPI_INFO_NULL);
      MPI_File_write_at_all(fp, wcon_range[rank], zwvals, nwsize,
                            PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      offset += wcon_range[size]*sizeof(ParOptScalar);

      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                        datarep, MPI_INFO_NULL);
      MPI_File_write_at_all(fp, wcon_range[rank], swvals, nwsize,
                            PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fp);
  }

  delete [] fname;

  return fail;
}

/**
   Read in the design variables, Lagrange multipliers and slack
   variables from a binary file.

   This function requires that the same problem structure as the
   original problem.

   @param filename is the name of the file input
*/
int ParOptInteriorPoint::readSolutionFile( const char *filename ){
  char *fname = new char[ strlen(filename)+1 ];
  strcpy(fname, filename);

  int fail = 1;
  MPI_File fp = NULL;
  MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
  delete [] fname;

  if (fp){
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
    if (rank == opt_root){
      int var_sizes[3];
      MPI_File_read(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);

      if (var_sizes[0] != var_range[size] ||
          var_sizes[1] != wcon_range[size] ||
          var_sizes[2] != ncon){
        size_fail = 1;
      }

      if (!size_fail){
        MPI_File_read(fp, &barrier_param, 1,
                      PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
        MPI_File_read(fp, z, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
        MPI_File_read(fp, s, ncon, PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      }
    }
    MPI_Bcast(&size_fail, 1, MPI_INT, opt_root, comm);

    // The problem sizes are inconsistent, return
    if (size_fail){
      fail = 1;
      if (rank == opt_root){
        fprintf(stderr,
                "ParOpt: Problem size incompatible with solution file\n");
      }

      MPI_File_close(&fp);
      return fail;
    }

    // Broadcast the multipliers and slack variables for the dense constraints
    MPI_Bcast(z, ncon, PAROPT_MPI_TYPE, opt_root, comm);
    MPI_Bcast(s, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Set the initial offset
    size_t offset = 3*sizeof(int) + (2*ncon+1)*sizeof(ParOptScalar);

    // Use the native representation for the data
    char datarep[] = "native";

    // Extract the design variables
    ParOptScalar *xvals;
    int xsize = x->getArray(&xvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                      datarep, MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], xvals, xsize,
                         PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(ParOptScalar);

    // Extract the lower Lagrange multipliers
    ParOptScalar *zlvals, *zuvals;
    zl->getArray(&zlvals);
    zu->getArray(&zuvals);
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                      datarep, MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], zlvals, xsize,
                         PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(ParOptScalar);

    // Read in the upper Lagrange multipliers
    MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                      datarep, MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], zuvals, xsize,
                         PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(ParOptScalar);

    // Read in the extra constraint Lagrange multipliers
    if (wcon_range[size] > 0){
      ParOptScalar *zwvals, *swvals;
      int nwsize = zw->getArray(&zwvals);
      sw->getArray(&swvals);
      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                        datarep, MPI_INFO_NULL);
      MPI_File_read_at_all(fp, wcon_range[rank], zwvals, nwsize,
                           PAROPT_MPI_TYPE, MPI_STATUS_IGNORE);
      offset += wcon_range[size]*sizeof(ParOptScalar);

      MPI_File_set_view(fp, offset, PAROPT_MPI_TYPE, PAROPT_MPI_TYPE,
                        datarep, MPI_INFO_NULL);
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
double ParOptInteriorPoint::getBarrierParameter(){
  return barrier_param;
}

/**
   Get the average of the complementarity products at the current
   point.  This function call is collective on all processors.

   @return the current complementarity.
*/
ParOptScalar ParOptInteriorPoint::getComplementarity(){
  return computeComp();
}

/**
   Set the penalty parameter for the l1 penalty function.

   @param gamma is the value of the penalty parameter
*/
void ParOptInteriorPoint::setPenaltyGamma( double gamma ){
  if (gamma >= 0.0){
    for ( int i = 0; i < ncon; i++ ){
      penalty_gamma[i] = gamma;
    }
  }
}

/**
   Set the individual penalty parameters for the l1 penalty function.

   @param gamma is the array of penalty parameter values.
*/
void ParOptInteriorPoint::setPenaltyGamma( const double *gamma ){
  for ( int i = 0; i < ncon; i++ ){
    if (gamma[i] >= 0.0){
      penalty_gamma[i] = gamma[i];
    }
  }
}

/**
   Retrieve the penalty parameter values.

   @param _penalty_gamma is the array of penalty parameter values.
*/
int ParOptInteriorPoint::getPenaltyGamma( const double **_penalty_gamma ){
  if (_penalty_gamma){ *_penalty_gamma = penalty_gamma;}
  return ncon;
}

/**
   Set the type of BFGS update.

   @param update is the type of BFGS update to use
*/
void ParOptInteriorPoint::setBFGSUpdateType( ParOptBFGSUpdateType update ){
  if (qn){
    ParOptLBFGS *lbfgs = dynamic_cast<ParOptLBFGS*>(qn);
    if (lbfgs){
      lbfgs->setBFGSUpdateType(update);
    }
  }
}

/**
   Set the quasi-Newton update object.

   @param _qn the compact quasi-Newton approximation
*/
void ParOptInteriorPoint::setQuasiNewton( ParOptCompactQuasiNewton *_qn ){
  if (_qn){
    _qn->incref();
  }
  if (qn){
    qn->decref();
  }
  qn = _qn;

  // Free the old data
  if (ztemp){ delete [] ztemp; }
  if (Ce){ delete [] Ce; }
  if (cpiv){ delete [] cpiv; }

  // Get the maximum subspace size
  int max_qn_subspace = 0;
  if (qn){
    max_qn_subspace = qn->getMaxLimitedMemorySize();
  }

  if (max_qn_subspace > 0){
    // Allocate storage for bfgs/constraint sized things
    int zsize = max_qn_subspace;
    if (ncon > zsize){
      zsize = ncon;
    }
    ztemp = new ParOptScalar[ zsize ];

    // Allocate space for the Ce matrix
    Ce = new ParOptScalar[ max_qn_subspace*max_qn_subspace ];
    cpiv = new int[ max_qn_subspace ];
  }
  else {
    ztemp = NULL;
    Ce = NULL;
    cpiv = NULL;
  }
}

/**
   Reset the Quasi-Newton Hessian approximation if it is used.
*/
void ParOptInteriorPoint::resetQuasiNewtonHessian(){
  if (qn){
    qn->reset();
  }
}

/**
   Reset the design variables and bounds.
*/
void ParOptInteriorPoint::resetDesignAndBounds(){
  prob->getVarsAndBounds(x, lb, ub);
}

/**
   Set the size of the GMRES subspace and allocate the vectors
   required. Note that the old subspace information is deleted before
   the new subspace data is allocated.

   @param m the GMRES subspace size.
*/
void ParOptInteriorPoint::setGMRESSubspaceSize( int m ){
  if (gmres_H){
    delete [] gmres_H;
    delete [] gmres_alpha;
    delete [] gmres_res;
    delete [] gmres_y;
    delete [] gmres_fproj;
    delete [] gmres_aproj;
    delete [] gmres_awproj;
    delete [] gmres_Q;

    for ( int i = 0; i < gmres_subspace_size; i++ ){
      gmres_W[i]->decref();
    }
    delete [] gmres_W;
  }

  if (m > 0){
    gmres_subspace_size = m;
    gmres_H = new ParOptScalar[ (m+1)*(m+2)/2 ];
    gmres_alpha = new ParOptScalar[ m+1 ];
    gmres_res = new ParOptScalar[ m+1 ];
    gmres_y = new ParOptScalar[ m+1 ];
    gmres_fproj = new ParOptScalar[ m+1 ];
    gmres_aproj = new ParOptScalar[ m+1 ];
    gmres_awproj = new ParOptScalar[ m+1 ];
    gmres_Q = new ParOptScalar[ 2*m ];

    gmres_W = new ParOptVec*[ m+1 ];
    for ( int i = 0; i < m+1; i++ ){
      gmres_W[i] = prob->createDesignVec();
      gmres_W[i]->incref();
    }
  }
  else {
    gmres_subspace_size = 0;
  }
}

/**
   Set the optimization history file name to use.

   Note that the file is only opened on the root processor.

   @param filename the output file name
*/
void ParOptInteriorPoint::setOutputFile( const char *filename ){
  if (outfp && outfp != stdout){
    fclose(outfp);
  }
  outfp = NULL;

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (filename && rank == opt_root){
    outfp = fopen(filename, "w");
  }
}

/**
   Compute the residual of the KKT system. This code utilizes the data
   stored internally in the ParOpt optimizer.

   This code computes the following terms:

   rx  = -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu)
   rt  = -(penalty_gamma - zt - z)
   rc  = -(c(x) - s + t)
   rcw = -(cw(x) - sw)
   rz  = -(S*z - mu*e)
   rzt = -(T*zt - mu*e)
   rzu = -((x - xl)*zl - mu*e)
   rzl = -((ub - x)*zu - mu*e)
*/
void ParOptInteriorPoint::computeKKTRes( double barrier,
                                         ParOptNormType norm_type,
                                         double *max_prime,
                                         double *max_dual,
                                         double *max_infeas,
                                         double *res_norm ){
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Zero the values of the maximum residuals
  *max_prime = 0.0;
  *max_dual = 0.0;
  *max_infeas = 0.0;

  // Assemble the negative of the residual of the first KKT equation:
  // -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu)
  if (use_lower){
    rx->copyValues(zl);
  }
  else {
    rx->zeroEntries();
  }
  if (use_upper){
    rx->axpy(-1.0, zu);
  }
  rx->axpy(-1.0, g);

  for ( int i = 0; i < ncon; i++ ){
    rx->axpy(z[i], Ac[i]);
  }

  if (nwcon > 0){
    // Add rx = rx + Aw^{T}*zw
    prob->addSparseJacobianTranspose(1.0, x, zw, rx);

    // Compute the residuals from the weighting constraints
    prob->evalSparseCon(x, rcw);
    if (sparse_inequality){
      rcw->axpy(-1.0, sw);
    }
    rcw->scale(-1.0);
  }

  // Compute the error in the first KKT condition
  if (norm_type == PAROPT_INFTY_NORM){
    *max_prime = rx->maxabs();
    *max_infeas = rcw->maxabs();
  }
  else if (norm_type == PAROPT_L1_NORM){
    *max_prime = rx->l1norm();
    *max_infeas = rcw->l1norm();
  }
  else { // norm_type == PAROPT_L2_NORM
    double prime_rx = rx->norm();
    double prime_rcw = rcw->norm();
    *max_prime = prime_rx*prime_rx;
    *max_infeas = prime_rcw*prime_rcw;
  }

  // Evaluate the residuals differently depending on whether
  // we're using a dense equality or inequality constraint
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      rc[i] = -(c[i] - s[i] + t[i]);
      rs[i] = -(s[i]*z[i] - barrier);
      rt[i] = -(penalty_gamma[i] - zt[i] - z[i]);
      rzt[i] = -(t[i]*zt[i] - barrier);
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      rc[i] = -c[i];
      rs[i] = 0.0;
      rt[i] = 0.0;
      rzt[i] = 0.0;
    }
  }

  if (norm_type == PAROPT_INFTY_NORM){
    for ( int i = 0; i < ncon; i++ ){
      if (fabs(ParOptRealPart(rt[i])) > *max_prime){
        *max_prime = fabs(ParOptRealPart(rt[i]));
      }
      if (fabs(ParOptRealPart(rc[i])) > *max_infeas){
        *max_infeas = fabs(ParOptRealPart(rc[i]));
      }
      if (fabs(ParOptRealPart(rs[i])) > *max_dual){
        *max_dual = fabs(ParOptRealPart(rs[i]));
      }
      if (fabs(ParOptRealPart(rzt[i])) > *max_dual){
        *max_dual = fabs(ParOptRealPart(rzt[i]));
      }
    }
  }
  else if (norm_type == PAROPT_L1_NORM){
    for ( int i = 0; i < ncon; i++ ){
      *max_prime += fabs(ParOptRealPart(rt[i]));
      *max_infeas += fabs(ParOptRealPart(rc[i]));
      *max_dual += fabs(ParOptRealPart(rs[i]));
      *max_dual += fabs(ParOptRealPart(rzt[i]));
    }
  }
  else { // norm_type == PAROPT_L2_NORM
    double prime = 0.0, infeas = 0.0, dual = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      prime += ParOptRealPart(rt[i]*rt[i]);
      infeas += ParOptRealPart(rc[i]*rc[i]);
      dual += ParOptRealPart(rs[i]*rs[i] + rzt[i]*rzt[i]);
    }
    *max_prime += prime;
    *max_infeas += infeas;
    *max_dual += dual;
  }

  // Extract the values of the variables and lower/upper bounds
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  if (use_lower){
    // Compute the residuals for the lower bounds
    ParOptScalar *rzlvals;
    rzl->getArray(&rzlvals);

    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        rzlvals[i] = -((xvals[i] - lbvals[i])*zlvals[i] -
                       rel_bound_barrier*barrier);
      }
      else {
        rzlvals[i] = 0.0;
      }
    }

    if (norm_type == PAROPT_INFTY_NORM){
      double dual_zl = rzl->maxabs();
      if (dual_zl > *max_dual){
        *max_dual = dual_zl;
      }
    }
    else if (norm_type == PAROPT_L1_NORM){
      *max_dual += rzl->l1norm();
    }
    else { // norm_type == PAROPT_L2_NORM
      double dual_zl = rzl->norm();
      *max_dual += dual_zl*dual_zl;
    }
  }
  if (use_upper){
    // Compute the residuals for the upper bounds
    ParOptScalar *rzuvals;
    rzu->getArray(&rzuvals);

    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        rzuvals[i] = -((ubvals[i] - xvals[i])*zuvals[i] -
                       rel_bound_barrier*barrier);
      }
      else {
        rzuvals[i] = 0.0;
      }
    }

    if (norm_type == PAROPT_INFTY_NORM){
      double dual_zu = rzu->maxabs();
      if (ParOptRealPart(dual_zu) > ParOptRealPart(*max_dual)){
        *max_dual = dual_zu;
      }
    }
    else if (norm_type == PAROPT_L1_NORM){
      *max_dual += rzu->l1norm();
    }
    else { // norm_type == PAROPT_L2_NORM
      double dual_zu = rzu->norm();
      *max_dual += dual_zu*dual_zu;
    }
  }

  if (nwcon > 0 && sparse_inequality){
    // Set the values of the perturbed complementarity
    // constraints for the sparse slack variables
    ParOptScalar *zwvals, *swvals, *rswvals;
    zw->getArray(&zwvals);
    sw->getArray(&swvals);
    rsw->getArray(&rswvals);

    for ( int i = 0; i < nwcon; i++ ){
      rswvals[i] = -(swvals[i]*zwvals[i] - barrier);
    }

    if (norm_type == PAROPT_INFTY_NORM){
      double dual_zw = rsw->maxabs();
      if (ParOptRealPart(dual_zw) > ParOptRealPart(*max_dual)){
        *max_dual = dual_zw;
      }
    }
    else if (norm_type == PAROPT_L1_NORM){
      *max_dual += rsw->l1norm();
    }
    else { // norm_type == PAROPT_L2_NORM
      double dual_zw = rsw->norm();
      *max_dual += dual_zw*dual_zw;
    }
  }

  // If this is the l2 norm, take the square root
  if (norm_type == PAROPT_L2_NORM){
    *max_dual = sqrt(*max_dual);
    *max_prime = sqrt(*max_prime);
    *max_infeas = sqrt(*max_infeas);
  }

  // Compute the max norm
  if (res_norm){
    *res_norm = *max_prime;
    if (*max_dual > *res_norm){
      *res_norm = *max_dual;
    }
    if (*max_infeas > *res_norm){
      *res_norm = *max_infeas;
    }
  }
}

/*
  Compute the maximum norm of the step
*/
double ParOptInteriorPoint::computeStepNorm( ParOptNormType norm_type ){
  double step_norm = 0.0;
  if (norm_type == PAROPT_INFTY_NORM){
    step_norm = px->maxabs();
  }
  else if (norm_type == PAROPT_L1_NORM){
    step_norm = px->l1norm();
  }
  else { // if (norm_type == PAROPT_L2_NORM)
    step_norm = px->norm();
  }
  return step_norm;
}

/*
  Factor the matrix after assembly
*/
int ParOptInteriorPoint::factorCw(){
  if (nwblock == 1){
    for ( int i = 0; i < nwcon; i++ ){
      // Compute and store Cw^{-1}
      if (Cw[i] == 0.0){
        return 1;
      }
      else {
        Cw[i] = 1.0/Cw[i];
      }
    }
  }
  else {
    ParOptScalar *cw = Cw;
    const int incr = ((nwblock + 1)*nwblock)/2;
    for ( int i = 0; i < nwcon; i += nwblock ){
      // Factor the matrix using the Cholesky factorization
      // for upper-triangular packed storage
      int info = 0;
      LAPACKdpptrf("U", &nwblock, cw, &info);

      if (info){
        return i + info;
      }
      cw += incr;
    }
  }

  return 0;
}

/*
  Apply the factored Cw-matrix that is stored as a series of
  block-symmetric matrices.
*/
int ParOptInteriorPoint::applyCwFactor( ParOptVec *vec ){
  ParOptScalar *rhs;
  vec->getArray(&rhs);

  if (nwblock == 1){
    for ( int i = 0; i < nwcon; i++ ){
      rhs[i] *= Cw[i];
    }
  }
  else {
    ParOptScalar *cw = Cw;
    const int incr = ((nwblock + 1)*nwblock)/2;
    for ( int i = 0; i < nwcon; i += nwblock ){
      // Factor the matrix using the Cholesky factorization
      // for the upper-triangular packed storage format
      int info = 0, one = 1;
      LAPACKdpptrs("U", &nwblock, &one, cw, rhs, &nwblock, &info);

      if (info){
        return i + info;
      }

      // Increment the pointers to the next block
      rhs += nwblock;
      cw += incr;
    }
  }

  return 0;
}

/*
  This function computes the terms required to solve the KKT system
  using a bordering method.  The initialization process computes the
  following matrix:

  C = b0 + zl/(x - lb) + zu/(ub - x)

  where C is a diagonal matrix. The components of C^{-1} (also a
  diagonal matrix) are stored in Cvec.

  Next, we compute:

  Cw = Zw^{-1}*Sw + Aw*C^{-1}*Aw^{T}

  where Cw is a block-diagonal matrix. We store the factored block
  matrix Cw in the variable Cw!  The code then computes the
  contribution from the weighting constraints as follows:

  Ew = Aw*C^{-1}*A, followed by:

  Dw = Ew^{T}*Cw^{-1}*Ew

  Finally, the code computes a factorization of the matrix:

  D = Z^{-1}*S + Zt^{-1}*T + A*C^{-1}*A^{T} - Dw

  which is required to compute the solution of the KKT step.
*/
void ParOptInteriorPoint::setUpKKTDiagSystem( ParOptVec *xtmp,
                                              ParOptVec *wtmp,
                                              int use_qn ){
  // Diagonal coefficient used for the quasi-Newton Hessian aprpoximation
  const double qn_sigma = options->getFloatOption("qn_sigma");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");

  // Retrive the diagonal entry for the BFGS update
  ParOptScalar b0 = 0.0;
  ParOptScalar *h = NULL;
  if (hdiag && use_diag_hessian){
    hdiag->getArray(&h);
  }
  else if (qn && use_qn){
    const ParOptScalar *d, *M;
    ParOptVec **Z;
    qn->getCompactMat(&b0, &d, &M, &Z);
  }

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Set the components of the diagonal matrix
  ParOptScalar *cvals;
  Cvec->getArray(&cvals);

  // Set the values of the c matrix
  if (use_lower && use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (h){ b0 = h[i]; }
      if (ParOptRealPart(lbvals[i]) > -max_bound_value &&
          ParOptRealPart(ubvals[i]) < max_bound_value){
        cvals[i] = 1.0/(b0 + qn_sigma +
                        zlvals[i]/(xvals[i] - lbvals[i]) +
                        zuvals[i]/(ubvals[i] - xvals[i]));
      }
      else if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        cvals[i] = 1.0/(b0 + qn_sigma + zlvals[i]/(xvals[i] - lbvals[i]));
      }
      else if (ParOptRealPart(ubvals[i]) < max_bound_value){
        cvals[i] = 1.0/(b0 + qn_sigma + zuvals[i]/(ubvals[i] - xvals[i]));
      }
      else {
        cvals[i] = 1.0/(b0 + qn_sigma);
      }
    }
  }
  else if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (h){ b0 = h[i]; }
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        cvals[i] = 1.0/(b0 + qn_sigma + zlvals[i]/(xvals[i] - lbvals[i]));
      }
      else {
        cvals[i] = 1.0/(b0 + qn_sigma);
      }
    }
  }
  else if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (h){ b0 = h[i]; }
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        cvals[i] = 1.0/(b0 + qn_sigma + zuvals[i]/(ubvals[i] - xvals[i]));
      }
      else {
        cvals[i] = 1.0/(b0 + qn_sigma);
      }
    }
  }
  else {
    for ( int i = 0; i < nvars; i++ ){
      if (h){ b0 = h[i]; }
      cvals[i] = 1.0/(b0 + qn_sigma);
    }
  }

  if (nwcon > 0){
    // Set the values in the Cw diagonal matrix
    memset(Cw, 0, nwcon*(nwblock+1)/2*sizeof(ParOptScalar));

    // Compute Cw = Zw^{-1}*Sw + Aw*C^{-1}*Aw
    // First compute Cw = Zw^{-1}*Sw
    if (sparse_inequality){
      ParOptScalar *swvals, *zwvals;
      zw->getArray(&zwvals);
      sw->getArray(&swvals);

      if (nwblock == 1){
        for ( int i = 0; i < nwcon; i++ ){
          Cw[i] = swvals[i]/zwvals[i];
        }
      }
      else {
        // Set the pointer and the increment for the
        // block-diagonal matrix
        ParOptScalar *cw = Cw;
        const int incr = ((nwblock+1)*nwblock)/2;

        // Iterate over each block matrix
        for ( int i = 0; i < nwcon; i += nwblock ){
          // Index into each block
          for ( int j = 0, k = 0; j < nwblock; j++, k += j+1 ){
            cw[k] = swvals[i+j]/zwvals[i+j];
          }

          // Increment the pointer to the next block
          cw += incr;
        }
      }
    }

    // Next, complete the evaluation of Cw by adding the following
    // contribution to the matrix
    // Cw += Aw*C^{-1}*Aw^{T}
    prob->addSparseInnerProduct(1.0, x, Cvec, Cw);

    // Factor the Cw matrix
    factorCw();

    // Compute Ew = Aw*C^{-1}*A
    for ( int k = 0; k < ncon; k++ ){
      ParOptScalar *avals, *xvals;
      Cvec->getArray(&cvals);
      xtmp->getArray(&xvals);
      Ac[k]->getArray(&avals);

      for ( int i = 0; i < nvars; i++ ){
        xvals[i] = cvals[i]*avals[i];
      }

      Ew[k]->zeroEntries();
      prob->addSparseJacobian(1.0, x, xtmp, Ew[k]);
    }
  }

  // Set the value of the D matrix
  memset(Dmat, 0, ncon*ncon*sizeof(ParOptScalar));

  if (nwcon > 0){
    // Add the term Dw = - Ew^{T}*Cw^{-1}*Ew to the Dmat matrix first
    // by computing the product with Cw^{-1}
    for ( int j = 0; j < ncon; j++ ){
      // Apply Cw^{-1}*Ew[j] -> wt
      wtmp->copyValues(Ew[j]);
      applyCwFactor(wtmp);

      for ( int i = j; i < ncon; i++ ){
        // Get the vectors required
        ParOptScalar *wvals, *ewivals;
        Ew[i]->getArray(&ewivals);
        wtmp->getArray(&wvals);

        ParOptScalar dmat = 0.0;
        int k = 0;
        int remainder = nwcon % 4;
        for ( ; k < remainder; k++ ){
          dmat += ewivals[0]*wvals[0];
          ewivals++; wvals++;
        }

        for ( int k = remainder; k < nwcon; k += 4 ){
          dmat += (ewivals[0]*wvals[0] + ewivals[1]*wvals[1] +
                   ewivals[2]*wvals[2] + ewivals[3]*wvals[3]);
          ewivals += 4; wvals += 4;
        }

        Dmat[i + ncon*j] -= dmat;
      }
    }
  }

  // Compute the lower diagonal portion of the matrix. This
  // code unrolls the loop to achieve better performance. Note
  // that this only computes the on-processor components.
  for ( int j = 0; j < ncon; j++ ){
    for ( int i = j; i < ncon; i++ ){
      // Get the vectors required
      ParOptScalar *aivals, *ajvals;
      Cvec->getArray(&cvals);
      Ac[i]->getArray(&aivals);
      Ac[j]->getArray(&ajvals);

      ParOptScalar dmat = 0.0;
      int k = 0, remainder = nvars % 4;
      for ( ; k < remainder; k++ ){
        dmat += aivals[0]*ajvals[0]*cvals[0];
        aivals++; ajvals++; cvals++;
      }

      for ( int k = remainder; k < nvars; k += 4 ){
        dmat += (aivals[0]*ajvals[0]*cvals[0] +
                 aivals[1]*ajvals[1]*cvals[1] +
                 aivals[2]*ajvals[2]*cvals[2] +
                 aivals[3]*ajvals[3]*cvals[3]);
        aivals += 4; ajvals += 4; cvals += 4;
      }

      Dmat[i + ncon*j] += dmat;
    }
  }

  // Populate the remainder of the matrix because it is
  // symmetric
  for ( int j = 0; j < ncon; j++ ){
    for ( int i = j+1; i < ncon; i++ ){
      Dmat[j + ncon*i] = Dmat[i + ncon*j];
    }
  }

  if (ncon > 0){
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Reduce the result to the root processor
    if (rank == opt_root){
      MPI_Reduce(MPI_IN_PLACE, Dmat, ncon*ncon,
                 PAROPT_MPI_TYPE, MPI_SUM, opt_root, comm);
    }
    else {
      MPI_Reduce(Dmat, NULL, ncon*ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }

    // Add the diagonal component to the matrix
    if (rank == opt_root){
      if (dense_inequality){
        for ( int i = 0; i < ncon; i++ ){
          Dmat[i*(ncon + 1)] += s[i]/z[i] + t[i]/zt[i];
        }
      }
    }

    // Broadcast the result to all processors. Note that this ensures
    // that the factorization will be the same on all processors
    MPI_Bcast(Dmat, ncon*ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Factor the matrix for future use
    int info = 0;
    LAPACKdgetrf(&ncon, &ncon, Dmat, &ncon, dpiv, &info);
  }
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms. The system of
  equations consists of the following terms:

  B0*yx - A^{T}*yz - Aw^{T}*yzw - yzl + yzu = bx
  A*yx - ys + yt = bc
  Aw*yx - ysw = bw

  With the additional equations:

  ys = Z^{-1}*bs - Z^{-1}*S*yz
  yzl = (X - Xl)^{-1}*(bzl - Zl*yx)
  yzu = (Xu - X)^{-1}*(bzu + Zu*yx)

  The slack update for t yields the equation:

  yzt = -bt - yz

  T*yzt + Zt*yt = bzt
  -T*(bt + yz) + Zt*yt = bzt

  yt = Zt^{-1}*T*yz + Zt^{-1}*(bzt + T*bt)

  Substitution of these equations yields the following system of
  equations:

  ((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))*yx - A^{T}*yz - Aw^{T}*yzw
  = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu

  which we rewrite as the equation:

  C*yx - A^{T}*yz - Aw^{T}*yzw = d

  and

  A*yx + (Z^{-1}*S + Zt^{-1}*T)*yz = bc + Z^{-1}*bs - Zt^{-1}*(bzt + T*bt),
  Aw*yx = bw.

  Where we define d as the vector:

  d = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu,

  we can solve for yz by solving the following system of equations:

  D0*yz + Ew^{T}*yzw = bc + Z^{-1}*bs - Zt^{-1}*(bzt + T*bt) - A*C^{-1}*d,
  Ew*yz +     Cw*yzw = bw - Aw*C^{-1}*d

  where C, Ew, and D0 are defined as follows:

  C = B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu,
  Ew = Aw*C^{-1}*A^{T},
  D0 = Z^{-1}*S + A*C^{-1}*A^{T}.

  We can then obtain yz by solving the following system of equations:

  Dmat*yz = bc + Z^{-1}*bs - Zt^{-1}*(bzt + T*bt) - A*C^{-1}*d
  .         - Ew^{T}*Cw^{-1}*(bw - Aw*C^{-1}*d)

  Once yz is obtained, we find yzw and yx as follows:

  yzw = Cw^{-1}*(bw - Ew*yz - Aw*C^{-1}*d)
  yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)

  Note: This code uses the temporary arrays xt and wt which therefore
  cannot be inputs/outputs for this function, otherwise strange
  behavior will occur.
*/
void ParOptInteriorPoint::solveKKTDiagSystem( ParOptVec *bx, ParOptScalar *bt,
                                              ParOptScalar *bc, ParOptVec *bcw,
                                              ParOptScalar *bs, ParOptVec *bsw,
                                              ParOptScalar *bzt,
                                              ParOptVec *bzl, ParOptVec *bzu,
                                              ParOptVec *yx, ParOptScalar *yt,
                                              ParOptScalar *yz, ParOptVec *yzw,
                                              ParOptScalar *ys, ParOptVec *ysw,
                                              ParOptScalar *yzt,
                                              ParOptVec *yzl, ParOptVec *yzu,
                                              ParOptVec *xtmp, ParOptVec *wtmp ){
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the arrays for the variables and upper/lower bounds
  ParOptScalar *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Get the arrays for the right-hand-sides
  ParOptScalar *bxvals, *bzlvals, *bzuvals;
  bx->getArray(&bxvals);
  bzl->getArray(&bzlvals);
  bzu->getArray(&bzuvals);

  // Compute xt = C^{-1}*d =
  // C^{-1}*(bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu)
  ParOptScalar *dvals, *cvals;
  xtmp->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }

  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        dvals[i] += cvals[i]*(bzlvals[i]/(xvals[i] - lbvals[i]));
      }
    }
  }
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        dvals[i] -= cvals[i]*(bzuvals[i]/(ubvals[i] - xvals[i]));
      }
    }
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wtmp = Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    wtmp->copyValues(bcw);

    if (sparse_inequality){
      // Add wtmp += Zw^{-1}*bsw
      ParOptScalar *wvals, *bswvals, *zwvals;
      wtmp->getArray(&wvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);

      for ( int i = 0; i < nwcon; i++ ){
        wvals[i] += bswvals[i]/zwvals[i];
      }
    }

    // Add the following term: wtmp -= Aw*C^{-1}*d
    prob->addSparseJacobian(-1.0, x, xtmp, wtmp);

    // Compute wtmp <- Cw^{-1}*wt
    applyCwFactor(wtmp);
  }

  // Now, compute yz = bc + Z^{-1}*bs - A*C^{-1}*d - Ew^{T}*wt
  memset(yz, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wtmp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      ParOptScalar *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar *avals;
    xtmp->getArray(&dvals);
    Ac[i]->getArray(&avals);

    ParOptScalar ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*dvals[0];
      avals++; dvals++;
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*dvals[0] + avals[1]*dvals[1] +
               avals[2]*dvals[2] + avals[3]*dvals[3]);
      avals += 4; dvals += 4;
    }

    yz[i] += ydot;
  }

  // Reduce all the results to the opt-root processor:
  // yz will now store the following term:
  // yz = - A*C^{-1}*d - Ew^{T}*Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
  if (ncon > 0){
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == opt_root){
      // Reduce the result to the root processor
      MPI_Reduce(MPI_IN_PLACE, yz, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }
    else {
      MPI_Reduce(yz, NULL, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }

    // Compute the full right-hand-side
    if (rank == opt_root){
      // Compute the full right-hand-side on the root processor
      // and solve for the Lagrange multipliers
      if (dense_inequality){
        for ( int i = 0; i < ncon; i++ ){
          yz[i] = bc[i] + bs[i]/z[i] - (bzt[i] + t[i]*bt[i])/zt[i] - yz[i];
        }
      }
      else {
        for ( int i = 0; i < ncon; i++ ){
          yz[i] = bc[i] - yz[i];
        }
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one,
                   Dmat, &ncon, dpiv, yz, &ncon, &info);
    }

    MPI_Bcast(yz, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Compute the step in the slack variables
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        ys[i] = (bs[i] - s[i]*yz[i])/z[i];
        yzt[i] = -bt[i] - yz[i];
        yt[i] = (bzt[i] - t[i]*yzt[i])/zt[i];
      }
    }
  }

  if (nwcon > 0){
    // Compute yzw = Cw^{-1}*(bcw + Zw^{-1}*bsw - Ew*yz - Aw*C^{-1}*d)
    // First set yzw <- bcw - Ew*yz
    yzw->copyValues(bcw);
    for ( int i = 0; i < ncon; i++ ){
      yzw->axpy(-yz[i], Ew[i]);
    }

    // Add the term yzw <- yzw + Zw^{-1}*bsw if we are using
    // inequality constraints
    if (sparse_inequality){
      ParOptScalar *yzwvals, *zwvals, *bswvals;
      yzw->getArray(&yzwvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);

      for ( int i = 0; i < nwcon; i++ ){
        yzwvals[i] += bswvals[i]/zwvals[i];
      }
    }

    // Compute yzw <- Cw^{-1}*(yzw - Aw*C^{-1}*d);
    prob->addSparseJacobian(-1.0, x, xtmp, yzw);
    applyCwFactor(yzw);

    // Compute the update to the weighting slack variables: ysw
    if (sparse_inequality){
      ParOptScalar *zwvals, *swvals;
      zw->getArray(&zwvals);
      sw->getArray(&swvals);

      ParOptScalar *yzwvals, *yswvals, *bswvals;
      yzw->getArray(&yzwvals);
      ysw->getArray(&yswvals);
      bsw->getArray(&bswvals);

      // Compute ysw = Zw^{-1}*(bsw - Sw*yzw)
      for ( int i = 0; i < nwcon; i++ ){
        yswvals[i] = (bswvals[i] - swvals[i]*yzwvals[i])/zwvals[i];
      }
    }
  }

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)
  // therefore yx = C^{-1}*(A^{T}*yz + Aw^{T}*yzw) + xt
  yx->zeroEntries();
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  // Add the term yx += Aw^{T}*yzw
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(1.0, x, yzw, yx);
  }

  // Apply the factor C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  ParOptScalar *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xtmp);

  // Retrieve the lagrange multipliers
  ParOptScalar *zlvals, *zuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the lagrange multiplier update vectors
  ParOptScalar *yzlvals, *yzuvals;
  yzl->getArray(&yzlvals);
  yzu->getArray(&yzuvals);

  // Compute the steps in the bound Lagrange multipliers
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        yzlvals[i] = (bzlvals[i] - zlvals[i]*yxvals[i])/(xvals[i] - lbvals[i]);
      }
      else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        yzuvals[i] = (bzuvals[i] + zuvals[i]*yxvals[i])/(ubvals[i] - xvals[i]);
      }
      else {
        yzuvals[i] = 0.0;
      }
    }
  }
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms.

  In this case, we assume that the only non-zero input components
  correspond the the unknowns in the first KKT system. This is the
  case when solving systems used with the limited-memory BFGS
  approximation.
*/
void ParOptInteriorPoint::solveKKTDiagSystem( ParOptVec *bx,
                                              ParOptVec *yx, ParOptScalar *yt,
                                              ParOptScalar *yz, ParOptVec *yzw,
                                              ParOptScalar *ys, ParOptVec *ysw,
                                              ParOptScalar *yzt,
                                              ParOptVec *yzl, ParOptVec *yzu,
                                              ParOptVec *xtmp, ParOptVec *wtmp ){
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Compute the terms from the weighting constraints
  // Compute xt = C^{-1}*bx
  ParOptScalar *bxvals, *dvals, *cvals;
  bx->getArray(&bxvals);
  xtmp->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = -Aw*C^{-1}*bx
    wtmp->zeroEntries();
    prob->addSparseJacobian(-1.0, x, xtmp, wtmp);

    // Compute wt <- Cw^{-1}*Aw*C^{-1}*bx
    applyCwFactor(wtmp);
  }

  // Now, compute yz = - A*C0^{-1}*bx - Ew^{T}*wt
  memset(yz, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wtmp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      ParOptScalar *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] += BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar *avals;
    xtmp->getArray(&dvals);
    Ac[i]->getArray(&avals);

    ParOptScalar ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*dvals[0];
      avals++; dvals++;
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*dvals[0] + avals[1]*dvals[1] +
               avals[2]*dvals[2] + avals[3]*dvals[3]);
      avals += 4; dvals += 4;
    }

    yz[i] += ydot;
  }

  if (ncon > 0){
    // Reduce the result to the root processor
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == opt_root){
      MPI_Reduce(MPI_IN_PLACE, yz, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }
    else {
      MPI_Reduce(yz, NULL, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }

    // Compute the full right-hand-side
    if (rank == opt_root){
      for ( int i = 0; i < ncon; i++ ){
        yz[i] *= -1.0;
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one,
                   Dmat, &ncon, dpiv, yz, &ncon, &info);
    }

    MPI_Bcast(yz, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Compute the step in the slack variables
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        ys[i] = -(s[i]*yz[i])/z[i];
        yzt[i] = -yz[i];
        yt[i] = -t[i]*yzt[i]/zt[i];
      }
    }
  }

  if (nwcon > 0){
    // Compute yw = -Cw^{-1}*(Ew*yz + Aw*C^{-1}*bx)
    // First set yw <- - Ew*yz
    yzw->zeroEntries();
    for ( int i = 0; i < ncon; i++ ){
      yzw->axpy(-yz[i], Ew[i]);
    }

    // Compute yzw <- Cw^{-1}*(yzw - Aw*C^{-1}*d);
    prob->addSparseJacobian(-1.0, x, xtmp, yzw);
    applyCwFactor(yzw);

    // Compute the update to the weighting slack variables: ysw
    if (sparse_inequality){
      ParOptScalar *zwvals, *swvals;
      zw->getArray(&zwvals);
      sw->getArray(&swvals);

      ParOptScalar *yzwvals, *yswvals;
      yzw->getArray(&yzwvals);
      ysw->getArray(&yswvals);

      // Compute yzw = Zw^{-1}*(bsw - Sw*yzw)
      for ( int i = 0; i < nwcon; i++ ){
        yswvals[i] = -(swvals[i]*yzwvals[i])/zwvals[i];
      }
    }
  }

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)
  // therefore yx = C^{-1}*(A^{T}*yz + Aw^{T}*yzw) + xt
  yx->zeroEntries();
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  // Add the term yx += Aw^{T}*yzw
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(1.0, x, yzw, yx);
  }

  // Apply the factor C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  ParOptScalar *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xtmp);

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the right-hand-sides and the solution vectors
  ParOptScalar *yzlvals, *yzuvals;
  yzl->getArray(&yzlvals);
  yzu->getArray(&yzuvals);

  // Compute the steps in the bound Lagrange multipliers
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        yzlvals[i] = -(zlvals[i]*yxvals[i])/(xvals[i] - lbvals[i]);
      }
      else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        yzuvals[i] = (zuvals[i]*yxvals[i])/(ubvals[i] - xvals[i]);
      }
      else {
        yzuvals[i] = 0.0;
      }
    }
  }
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms.

  In this case, we assume that the only non-zero input components
  correspond the the unknowns in the first KKT system. This is the
  case when solving systems used w
*/
void ParOptInteriorPoint::solveKKTDiagSystem( ParOptVec *bx, ParOptVec *yx,
                                              ParOptScalar *ztmp,
                                              ParOptVec *xtmp, ParOptVec *wtmp ){
  // Compute the terms from the weighting constraints
  // Compute xt = C^{-1}*bx
  ParOptScalar *bxvals, *dvals, *cvals;
  bx->getArray(&bxvals);
  xtmp->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }

  // Compute the contribution from the weighting constraints
  if (nwcon > 0){
    // Compute wt = -Aw*C^{-1}*bx
    wtmp->zeroEntries();
    prob->addSparseJacobian(-1.0, x, xtmp, wtmp);

    // Compute wt <- Cw^{-1}*Aw*C^{-1}*bx
    applyCwFactor(wtmp);
  }

  // Compute ztemp = (S*Z^{-1} - A*C0^{-1}*bx)
  memset(ztmp, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wtmp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      ParOptScalar *ewvals;
      Ew[i]->getArray(&ewvals);
      ztmp[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar *avals;
    xtmp->getArray(&dvals);
    Ac[i]->getArray(&avals);

    ParOptScalar ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*dvals[0];
      avals++; dvals++;
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*dvals[0] + avals[1]*dvals[1] +
               avals[2]*dvals[2] + avals[3]*dvals[3]);
      avals += 4; dvals += 4;
    }

    ztmp[i] += ydot;
  }

  if (ncon > 0){
    // Reduce the result to the root processor
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == opt_root){
      MPI_Reduce(MPI_IN_PLACE, ztmp, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }
    else {
      MPI_Reduce(ztmp, NULL, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }

    if (rank == opt_root){
      for ( int i = 0; i < ncon; i++ ){
        ztmp[i] *= -1.0;
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one,
                   Dmat, &ncon, dpiv, ztmp, &ncon, &info);
    }

    MPI_Bcast(ztmp, ncon, PAROPT_MPI_TYPE, opt_root, comm);
  }

  if (nwcon > 0){
    // Compute wt = -Cw^{-1}*(Ew*yz + Aw*C^{-1}*bx)
    // First set wt <- - Ew*yz
    wtmp->zeroEntries();
    for ( int i = 0; i < ncon; i++ ){
      wtmp->axpy(-ztmp[i], Ew[i]);
    }

    // Compute yzw <- - Cw^{-1}*Aw*C^{-1}*d);
    prob->addSparseJacobian(-1.0, x, xtmp, wtmp);
    applyCwFactor(wtmp);
  }

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)
  // therefore yx = C^{-1}*(A^{T}*yz + Aw^{T}*yzw) + xt
  yx->zeroEntries();
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(ztmp[i], Ac[i]);
  }

  // Add the term yx += Aw^{T}*wt
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(1.0, x, wtmp, yx);
  }

  // Apply the factor C^{-1}*(A^{T}*ztmp + Aw^{T}*wt)
  ParOptScalar *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xtmp);
}

/*
  Solve the linear system

  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms.

  Note that in this variant of the function, the right-hand-side
  includes components that are scaled by a given alpha-parameter.
*/
void ParOptInteriorPoint::solveKKTDiagSystem( ParOptVec *bx,
                                              ParOptScalar alpha,
                                              ParOptScalar *bt, ParOptScalar *bc,
                                              ParOptVec *bcw, ParOptScalar *bs,
                                              ParOptVec *bsw, ParOptScalar *bzt,
                                              ParOptVec *bzl, ParOptVec *bzu,
                                              ParOptVec *yx, ParOptScalar *yt,
                                              ParOptScalar *yz,
                                              ParOptScalar *ys, ParOptVec *ysw,
                                              ParOptVec *xtmp, ParOptVec *wtmp ){
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the arrays for the variables and upper/lower bounds
  ParOptScalar *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Get the arrays for the right-hand-sides
  ParOptScalar *bxvals, *bzlvals, *bzuvals;
  bx->getArray(&bxvals);
  bzl->getArray(&bzlvals);
  bzu->getArray(&bzuvals);

  // Compute xt = C^{-1}*d =
  // C^{-1}*(bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu)
  ParOptScalar *dvals, *cvals;
  xtmp->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }

  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        dvals[i] += alpha*cvals[i]*(bzlvals[i]/(xvals[i] - lbvals[i]));
      }
    }
  }
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        dvals[i] -= alpha*cvals[i]*(bzuvals[i]/(ubvals[i] - xvals[i]));
      }
    }
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    wtmp->copyValues(bcw);
    wtmp->scale(alpha);

    if (sparse_inequality){
      // Add wt += Zw^{-1}*bsw
      ParOptScalar *wvals, *bswvals, *zwvals;
      wtmp->getArray(&wvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);

      for ( int i = 0; i < nwcon; i++ ){
        wvals[i] += alpha*bswvals[i]/zwvals[i];
      }
    }

    // Add the following term: wt -= Aw*C^{-1}*d
    prob->addSparseJacobian(-1.0, x, xtmp, wtmp);

    // Compute wt <- Cw^{-1}*wt
    applyCwFactor(wtmp);
  }

  // Now, compute yz = bc + Z^{-1}*bs - A*C^{-1}*d - Ew^{T}*wt
  memset(yz, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wtmp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      ParOptScalar *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar *avals;
    xtmp->getArray(&dvals);
    Ac[i]->getArray(&avals);

    ParOptScalar ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*dvals[0];
      avals++; dvals++;
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*dvals[0] + avals[1]*dvals[1] +
               avals[2]*dvals[2] + avals[3]*dvals[3]);
      avals += 4; dvals += 4;
    }

    yz[i] += ydot;
  }

  if (ncon > 0){
    // Reduce all the results to the opt-root processor:
    // yz will now store the following term:
    // yz = - A*C^{-1}*d - Ew^{T}*Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == opt_root){
      // Reduce the result to the root processor
      MPI_Reduce(MPI_IN_PLACE, yz, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }
    else {
      MPI_Reduce(yz, NULL, ncon, PAROPT_MPI_TYPE, MPI_SUM,
                 opt_root, comm);
    }

    // Compute the full right-hand-side
    if (rank == opt_root){
      // Compute the full right-hand-side on the root processor
      // and solve for the Lagrange multipliers
      if (dense_inequality){
        for ( int i = 0; i < ncon; i++ ){
          yz[i] = alpha*(bc[i] + bs[i]/z[i] -
                         (bzt[i] + t[i]*bt[i])/zt[i]) - yz[i];
        }
      }
      else {
        for ( int i = 0; i < ncon; i++ ){
          yz[i] = alpha*bc[i] - yz[i];
        }
      }

      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one,
                   Dmat, &ncon, dpiv, yz, &ncon, &info);
    }

    MPI_Bcast(yz, ncon, PAROPT_MPI_TYPE, opt_root, comm);

    // Compute the step in the slack variables
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        ys[i] = (alpha*bs[i] - s[i]*yz[i])/z[i];
        ParOptScalar yzt = -alpha*bt[i] - yz[i];
        yt[i] = (alpha*bzt[i] - t[i]*yzt)/zt[i];
      }
    }
  }

  if (nwcon > 0){
    // Compute yzw = Cw^{-1}*(bcw + Zw^{-1}*bsw - Ew*yz - Aw*C^{-1}*d)
    // First set yzw <- bcw - Ew*yz
    wtmp->copyValues(bcw);
    wtmp->scale(alpha);
    for ( int i = 0; i < ncon; i++ ){
      wtmp->axpy(-yz[i], Ew[i]);
    }

    // Add the term yzw <- yzw + Zw^{-1}*bsw if we are using
    // inequality constraints
    if (sparse_inequality){
      ParOptScalar *yzwvals, *zwvals, *bswvals;
      wtmp->getArray(&yzwvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);

      for ( int i = 0; i < nwcon; i++ ){
        yzwvals[i] += alpha*bswvals[i]/zwvals[i];
      }
    }

    // Compute yzw <- Cw^{-1}*(yzw - Aw*C^{-1}*d);
    prob->addSparseJacobian(-1.0, x, xtmp, wtmp);
    applyCwFactor(wtmp);

    // Compute the update to the weighting slack variables: ysw
    if (sparse_inequality){
      ParOptScalar *zwvals, *swvals;
      zw->getArray(&zwvals);
      sw->getArray(&swvals);

      ParOptScalar *yzwvals, *yswvals, *bswvals;
      wtmp->getArray(&yzwvals);
      ysw->getArray(&yswvals);
      bsw->getArray(&bswvals);

      // Compute ysw = Zw^{-1}*(bsw - Sw*yzw)
      for ( int i = 0; i < nwcon; i++ ){
        yswvals[i] = (alpha*bswvals[i] - swvals[i]*yzwvals[i])/zwvals[i];
      }
    }
  }

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)
  // therefore yx = C^{-1}*(A^{T}*yz + Aw^{T}*yzw) + xt
  yx->zeroEntries();
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  // Add the term yx += Aw^{T}*yzw
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(1.0, x, wtmp, yx);
  }

  // Apply the factor C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  ParOptScalar *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xtmp);
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
void ParOptInteriorPoint::setUpKKTSystem( ParOptScalar *ztmp,
                                          ParOptVec *xtmp1,
                                          ParOptVec *xtmp2,
                                          ParOptVec *wtmp,
                                          int use_qn ){
  if (qn && use_qn){
    // Get the size of the limited-memory BFGS subspace
    ParOptScalar b0;
    const ParOptScalar *d0, *M;
    ParOptVec **Z;
    int size = qn->getCompactMat(&b0, &d0, &M, &Z);

    if (size > 0){
      memset(Ce, 0, size*size*sizeof(ParOptScalar));

      // Solve the KKT system
      for ( int i = 0; i < size; i++ ){
        // Compute K^{-1}*Z[i]
        solveKKTDiagSystem(Z[i], xtmp1,
                           ztmp, xtmp2, wtmp);

        // Compute the dot products Z^{T}*K^{-1}*Z[i]
        xtmp1->mdot(Z, size, &Ce[i*size]);
      }

      // Compute the Schur complement
      for ( int j = 0; j < size; j++ ){
        for ( int i = 0; i < size; i++ ){
          Ce[i + j*size] -= M[i + j*size]/(d0[i]*d0[j]);
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
void ParOptInteriorPoint::computeKKTStep( ParOptScalar *ztmp,
                                          ParOptVec *xtmp1, ParOptVec *xtmp2,
                                          ParOptVec *wtmp, int use_qn ){
  // Get the size of the limited-memory BFGS subspace
  ParOptScalar b0;
  const ParOptScalar *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (qn && use_qn){
    size = qn->getCompactMat(&b0, &d, &M, &Z);
  }

  // After this point the residuals are no longer required.
  solveKKTDiagSystem(rx, rt, rc, rcw, rs, rsw, rzt, rzl, rzu,
                     px, pt, pz, pzw, ps, psw, pzt, pzl, pzu,
                     xtmp1, wtmp);

  if (size > 0){
    // dz = Z^{T}*px
    px->mdot(Z, size, ztmp);

    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one,
                 Ce, &size, cpiv, ztmp, &size, &info);

    // Compute rx = Z^{T}*dz
    xtmp1->zeroEntries();
    for ( int i = 0; i < size; i++ ){
      xtmp1->axpy(ztmp[i], Z[i]);
    }

    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(xtmp1,
                       rx, rt, rc, rcw, rs, rsw, rzt, rzl, rzu,
                       xtmp2, wtmp);

    // Add the final contributions
    px->axpy(-1.0, rx);
    pzw->axpy(-1.0, rcw);
    psw->axpy(-1.0, rsw);
    pzl->axpy(-1.0, rzl);
    pzu->axpy(-1.0, rzu);

    // Add the terms from the slacks/multipliers
    for ( int i = 0; i < ncon; i++ ){
      pz[i] -= rc[i];
      ps[i] -= rs[i];
      pt[i] -= rt[i];
      pzt[i] -= rzt[i];
    }
  }
}

/*
  Compute the complementarity at the current solution
*/
ParOptScalar ParOptInteriorPoint::computeComp(){
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Sum up the complementarity from each individual processor
  ParOptScalar product = 0.0, sum = 0.0;

  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        product += zlvals[i]*(xvals[i] - lbvals[i]);
        sum += 1.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        product += zuvals[i]*(ubvals[i] - xvals[i]);
        sum += 1.0;
      }
    }
  }

  // Modify the complementarity by the bound scalar factor
  product = product/rel_bound_barrier;

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
  if (rank == opt_root){
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        product += s[i]*z[i] + t[i]*zt[i];
        sum += 2.0;
      }
    }

    if (sum != 0.0){
      comp = product/sum;
    }
  }

  // Broadcast the result to all processors
  MPI_Bcast(&comp, 1, PAROPT_MPI_TYPE, opt_root, comm);

  return comp;
}

/*
  Compute the complementarity at the given step
*/
ParOptScalar ParOptInteriorPoint::computeCompStep( double alpha_x, double alpha_z ){
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the values of the steps
  ParOptScalar *pxvals, *pzlvals, *pzuvals;
  px->getArray(&pxvals);
  pzl->getArray(&pzlvals);
  pzu->getArray(&pzuvals);

  // Sum up the complementarity from each individual processor
  ParOptScalar product = 0.0, sum = 0.0;
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        ParOptScalar xnew = xvals[i] + alpha_x*pxvals[i];
        product += (zlvals[i] + alpha_z*pzlvals[i])*(xnew - lbvals[i]);
        sum += 1.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        ParOptScalar xnew = xvals[i] + alpha_x*pxvals[i];
        product += (zuvals[i] + alpha_z*pzuvals[i])*(ubvals[i] - xnew);
        sum += 1.0;
      }
    }
  }

  // Modify the complementarity by the bound scalar factor
  product = product/rel_bound_barrier;

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
  if (rank == opt_root){
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        product += ((s[i] + alpha_x*ps[i])*(z[i] + alpha_z*pz[i]) +
                    (t[i] + alpha_x*pt[i])*(zt[i] + alpha_z*pzt[i]));
        sum += 2.0;
      }
    }

    if (sum != 0.0){
      comp = product/sum;
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
void ParOptInteriorPoint::computeMaxStep( double tau,
                                          double *_max_x, double *_max_z ){
  // Set the initial step length along the design and multiplier
  // directions
  double max_x = 1.0, max_z = 1.0;

  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  ParOptScalar *xvals, *pxvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  px->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Check the design variable step
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(pxvals[i]) < 0.0){
        double numer = ParOptRealPart(xvals[i] - lbvals[i]);
        double alpha = -tau*numer/ParOptRealPart(pxvals[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(pxvals[i]) > 0.0){
        double numer = ParOptRealPart(ubvals[i] - xvals[i]);
        double alpha = tau*numer/ParOptRealPart(pxvals[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
    }
  }

  if (dense_inequality){
    // Check the slack variable step
    for ( int i = 0; i < ncon; i++ ){
      if (ParOptRealPart(ps[i]) < 0.0){
        double numer = ParOptRealPart(s[i]);
        double alpha = -tau*numer/ParOptRealPart(ps[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
      if (ParOptRealPart(pt[i]) < 0.0){
        double numer = ParOptRealPart(t[i]);
        double alpha = -tau*numer/ParOptRealPart(pt[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
      // Check the step for the Lagrange multipliers
      if (ParOptRealPart(pz[i]) < 0.0){
        double numer = ParOptRealPart(z[i]);
        double alpha = -tau*numer/ParOptRealPart(pz[i]);
        if (alpha < max_z){
          max_z = alpha;
        }
      }
      if (ParOptRealPart(pzt[i]) < 0.0){
        double numer = ParOptRealPart(zt[i]);
        double alpha = -tau*numer/ParOptRealPart(pzt[i]);
        if (alpha < max_z){
          max_z = alpha;
        }
      }
    }
  }

  // Check the Lagrange and slack variable steps for the
  // sparse inequalities if any
  if (nwcon > 0 && sparse_inequality){
    ParOptScalar *zwvals, *pzwvals;
    zw->getArray(&zwvals);
    pzw->getArray(&pzwvals);
    for ( int i = 0; i < nwcon; i++ ){
      if (ParOptRealPart(pzwvals[i]) < 0.0){
        double numer = ParOptRealPart(zwvals[i]);
        double alpha = -tau*numer/ParOptRealPart(pzwvals[i]);
        if (alpha < max_z){
          max_z = alpha;
        }
      }
    }

    ParOptScalar *swvals, *pswvals;
    sw->getArray(&swvals);
    psw->getArray(&pswvals);
    for ( int i = 0; i < nwcon; i++ ){
      if (ParOptRealPart(pswvals[i]) < 0.0){
        double numer = ParOptRealPart(swvals[i]);
        double alpha = -tau*numer/ParOptRealPart(pswvals[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
    }
  }

  // Retrieve the values of the lower/upper Lagrange multipliers
  ParOptScalar *zlvals, *zuvals, *pzlvals, *pzuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);
  pzl->getArray(&pzlvals);
  pzu->getArray(&pzuvals);

  // Check the step for the lower/upper Lagrange multipliers
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(pzlvals[i]) < 0.0){
        double numer = ParOptRealPart(zlvals[i]);
        double alpha = -tau*numer/ParOptRealPart(pzlvals[i]);
        if (alpha < max_z){
          max_z = alpha;
        }
      }
    }
  }
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(pzuvals[i]) < 0.0){
        double numer = ParOptRealPart(zuvals[i]);
        double alpha = -tau*numer/ParOptRealPart(pzuvals[i]);
        if (alpha < max_z){
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
void ParOptInteriorPoint::scaleStep( ParOptScalar alpha,
                                     int nvals,
                                     ParOptScalar *pvals ){
  for ( int i = 0; i < nvals; i++ ){
    pvals[i] *= alpha;
  }
}

/*
  Make sure that the step x + alpha*p lies strictly within the
  bounds l + design_precision <= x + alpha*p <= u - design_precision.

  If the bounds are violated, adjust the step so that they will be
  satisfied.
*/
void ParOptInteriorPoint::computeStepVec( ParOptVec *xvec, ParOptScalar alpha,
                                          ParOptVec *pvec,
                                          ParOptVec *lower,
                                          ParOptScalar *lower_value,
                                          ParOptVec *upper,
                                          ParOptScalar *upper_value ){
  ParOptScalar *xvals = NULL;
  ParOptScalar *pvals = NULL;
  ParOptScalar *ubvals = NULL;
  ParOptScalar *lbvals = NULL;
  int size = xvec->getArray(&xvals);
  pvec->getArray(&pvals);

  if (lower){
    lower->getArray(&lbvals);
  }
  if (upper){
    upper->getArray(&ubvals);
  }

  computeStep(size, xvals, alpha, pvals, lbvals, lower_value, ubvals, upper_value);
}

/*
  Make sure that the step is within the prescribed bounds
*/
void ParOptInteriorPoint::computeStep( int nvals,
                                       ParOptScalar *xvals,
                                       ParOptScalar alpha,
                                       const ParOptScalar *pvals,
                                       const ParOptScalar *lbvals,
                                       const ParOptScalar *lower_value,
                                       const ParOptScalar *ubvals,
                                       const ParOptScalar *upper_value ){
  const double design_precision = options->getFloatOption("design_precision");
  for ( int i = 0; i < nvals; i++ ){
    xvals[i] = xvals[i] + alpha*pvals[i];
  }
  if (lbvals){
    for ( int i = 0; i < nvals; i++ ){
      if (ParOptRealPart(xvals[i]) <=
          ParOptRealPart(lbvals[i]) + design_precision){
        xvals[i] = lbvals[i] + design_precision;
      }
    }
  }
  else if (lower_value){
    double lbval = ParOptRealPart(*lower_value);
    for ( int i = 0; i < nvals; i++ ){
      if (ParOptRealPart(xvals[i]) <= lbval + design_precision){
        xvals[i] = lbval + design_precision;
      }
    }
  }

  if (ubvals){
    for ( int i = 0; i < nvals; i++ ){
      if (ParOptRealPart(xvals[i]) + design_precision >=
          ParOptRealPart(ubvals[i])){
        xvals[i] = ubvals[i] - design_precision;
      }
    }
  }
  else if (upper_value){
    double ubval = ParOptRealPart(*upper_value);
    for ( int i = 0; i < nvals; i++ ){
      if (ParOptRealPart(xvals[i]) + design_precision >= ubval){
        xvals[i] = ubval - design_precision;
      }
    }
  }
}

/*
  Scale the KKT step by the maximum allowable step length
*/
int ParOptInteriorPoint::scaleKKTStep( double tau, ParOptScalar comp,
                                       int inexact_newton_step,
                                       double *_alpha_x, double *_alpha_z ){
  double alpha_x = 1.0, alpha_z = 1.0;
  computeMaxStep(tau, &alpha_x, &alpha_z);

  // Keep track of whether we set both the design and Lagrange
  // multiplier steps equal to one another
  int ceq_step = 0;

  // Check if we're using a Newton step or not
  if (!inexact_newton_step){
    // First, bound the difference between the step lengths. This
    // code cuts off the difference between the step lengths if the
    // difference is greater that 100.
    double max_bnd = 100.0;
    if (alpha_x > alpha_z){
      if (alpha_x > max_bnd*alpha_z){
        alpha_x = max_bnd*alpha_z;
      }
      else if (alpha_x < alpha_z/max_bnd){
        alpha_x = alpha_z/max_bnd;
      }
    }
    else {
      if (alpha_z > max_bnd*alpha_x){
        alpha_z = max_bnd*alpha_x;
      }
      else if (alpha_z < alpha_x/max_bnd){
        alpha_z = alpha_x/max_bnd;
      }
    }

    // As a last check, compute the average of the complementarity
    // products at the full step length. If the complementarity
    // increases, use equal step lengths.
    ParOptScalar comp_new = computeCompStep(alpha_x, alpha_z);

    if (ParOptRealPart(comp_new) > 10.0*ParOptRealPart(comp)){
      ceq_step = 1;
      if (alpha_x > alpha_z){
        alpha_x = alpha_z;
      }
      else {
        alpha_z = alpha_x;
      }
    }
  }
  else {
    // If we're using a Newton method, use the same step
    // size for both the multipliers and variables
    if (alpha_x > alpha_z){
      alpha_x = alpha_z;
    }
    else {
      alpha_z = alpha_x;
    }
  }

  // Scale the steps by the maximum permissible step lengths
  px->scale(alpha_x);
  if (nwcon > 0){
    pzw->scale(alpha_z);
    if (sparse_inequality){
      psw->scale(alpha_x);
    }
  }
  if (use_lower){
    pzl->scale(alpha_z);
  }
  if (use_upper){
    pzu->scale(alpha_z);
  }

  scaleStep(alpha_z, ncon, pz);
  if (dense_inequality){
    scaleStep(alpha_x, ncon, ps);
    scaleStep(alpha_x, ncon, pt);
    scaleStep(alpha_z, ncon, pzt);
  }

  *_alpha_x = alpha_x;
  *_alpha_z = alpha_z;

  return ceq_step;
}

/*
  Check the gradient of the merit function using finite-difference or complex-step
*/
void ParOptInteriorPoint::checkMeritFuncGradient( ParOptVec *xpt, double dh ){
  if (xpt){
    x->copyValues(xpt);
  }

  // Evaluate the objective and constraints and their gradients
  int fail_obj = prob->evalObjCon(x, &fobj, c);
  neval++;
  if (fail_obj){
    fprintf(stderr,
            "ParOpt: Function and constraint evaluation failed\n");
    return;
  }

  int fail_gobj = prob->evalObjConGradient(x, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr,
            "ParOpt: Gradient evaluation failed\n");
    return;
  }

  // If the point is specified, pick a direction and use it,
  // otherwise use the existing step
  if (xpt){
    // Set a step in the design variables
    px->copyValues(g);
    px->scale(-1.0/ParOptRealPart(px->norm()));

    // Zero all other components in the step computation
    for ( int i = 0; i < ncon; i++ ){
      ps[i] = -0.259*(1 + (i % 3));
      pt[i] = -0.349*(4 - (i % 2));
    }

    if (nwcon > 0 && sparse_inequality){
      psw->zeroEntries();
      ParOptScalar *pswvals;
      psw->getArray(&pswvals);

      for ( int i = 0; i < nwcon; i++ ){
        pswvals[i] = -0.419*(1 + (i % 5));
      }
    }
  }

  // Evaluate the merit function and its derivative
  ParOptScalar m0 = 0.0, dm0 = 0.0;
  double max_x = 1.0;
  evalMeritInitDeriv(max_x, &m0, &dm0, rx, wtemp, rcw);

#ifdef PAROPT_USE_COMPLEX
  rx->copyValues(x);
  rx->axpy(ParOptScalar(0.0, dh), px);

  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      rs[i] = s[i] + ParOptScalar(0.0, dh)*ps[i];
      rt[i] = t[i] + ParOptScalar(0.0, dh)*pt[i];
    }
  }

  if (nwcon > 0 && sparse_inequality){
    rsw->copyValues(sw);
    rsw->axpy(ParOptScalar(0.0, dh), psw);
  }
#else
  rx->copyValues(x);
  rx->axpy(dh, px);

  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      rs[i] = s[i] + dh*ps[i];
      rt[i] = t[i] + dh*pt[i];
    }
  }

  if (nwcon > 0 && sparse_inequality){
    rsw->copyValues(sw);
    rsw->axpy(dh, psw);
  }
#endif // PAROPT_USE_COMPLEX

  // Evaluate the objective
  ParOptScalar ftemp;
  fail_obj = prob->evalObjCon(rx, &ftemp, rc);
  neval++;
  if (fail_obj){
    fprintf(stderr,
            "ParOpt: Function and constraint evaluation failed\n");
    return;
  }
  ParOptScalar m1 = evalMeritFunc(ftemp, rc, rx, rs, rt, rsw);

  ParOptScalar fd = 0.0;
#ifdef PAROPT_USE_COMPLEX
  fd = ParOptImagPart(m1)/dh;
#else
  fd = (m1 - m0)/dh;
#endif // PAROPT_USE_COMPLEX

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    fprintf(stdout, "Merit function test\n");
    fprintf(stdout, "dm FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
            ParOptRealPart(fd), ParOptRealPart(dm0),
            fabs(ParOptRealPart(fd - dm0)), fabs(ParOptRealPart((fd - dm0)/fd)));
  }
}

/*
  Evaluate the merit function at the current point, assuming that the
  objective and constraint values are up to date.

  The merit function is given as follows:

  varphi(alpha) =

  f(x) +
  mu*(log(s) + log(x - xl) + log(xu - x)) +
  rho*(||c(x) - s + t||_{2} + ||cw(x) - sw||_{2})

  output: The value of the merit function
*/
ParOptScalar ParOptInteriorPoint::evalMeritFunc( ParOptScalar fk,
                                                 const ParOptScalar *ck,
                                                 ParOptVec *xk,
                                                 const ParOptScalar *sk,
                                                 const ParOptScalar *tk,
                                                 ParOptVec *swk ){
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

  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        if (ParOptRealPart(xvals[i] - lbvals[i]) > 1.0){
          pos_result += log(xvals[i] - lbvals[i]);
        }
        else {
          neg_result += log(xvals[i] - lbvals[i]);
        }
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        if (ParOptRealPart(ubvals[i] - xvals[i]) > 1.0){
          pos_result += log(ubvals[i] - xvals[i]);
        }
        else {
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
  if (nwcon > 0 && sparse_inequality){
    ParOptScalar *swvals;
    swk->getArray(&swvals);

    for ( int i = 0; i < nwcon; i++ ){
      if (ParOptRealPart(swvals[i]) > 1.0){
        pos_result += log(swvals[i]);
      }
      else {
        neg_result += log(swvals[i]);
      }
    }
  }

  // Compute the norm of the weight constraint infeasibility
  ParOptScalar weight_infeas = 0.0;
  if (nwcon > 0){
    prob->evalSparseCon(xk, wtemp);
    if (sparse_inequality){
      wtemp->axpy(-1.0, swk);
    }
#ifdef PAROPT_USE_COMPLEX
    ParOptScalar *wvals;
    int wsize = wtemp->getArray(&wvals);
    for ( int i = 0; i < wsize; i++ ){
      weight_infeas += wvals[i]*wvals[i];
    }
    ParOptScalar weight_infeas_temp = weight_infeas;
    MPI_Allreduce(&weight_infeas_temp, &weight_infeas, 1,
                  PAROPT_MPI_TYPE, MPI_SUM, comm);
    weight_infeas = sqrt(weight_infeas);
#else
    weight_infeas = wtemp->norm();
#endif // PAROPT_USE_COMPLEX
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
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      if (ParOptRealPart(sk[i]) > 1.0){
        pos_result += log(sk[i]);
      }
      else {
        neg_result += log(sk[i]);
      }
      if (ParOptRealPart(tk[i]) > 1.0){
        pos_result += log(tk[i]);
      }
      else {
        neg_result += log(tk[i]);
      }
    }
  }

  // Compute the full merit function only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  ParOptScalar merit = 0.0;
  if (rank == opt_root){
    // Compute the infeasibility
    ParOptScalar dense_infeas = 0.0;
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        dense_infeas += (ck[i] - sk[i] + tk[i])*(ck[i] - sk[i] + tk[i]);
      }
    }
    else {
      for ( int i = 0; i < ncon; i++ ){
        dense_infeas += ck[i]*ck[i];
      }
    }
    ParOptScalar infeas = sqrt(dense_infeas) + weight_infeas;

    // Add the contribution from the constraints
    merit = (fk - barrier_param*(pos_result + neg_result) +
             rho_penalty_search*infeas);

    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        merit += penalty_gamma[i]*tk[i];
      }
    }
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
void ParOptInteriorPoint::evalMeritInitDeriv( double max_x,
                                              ParOptScalar *_merit,
                                              ParOptScalar *_pmerit,
                                              ParOptVec *xtmp,
                                              ParOptVec *wtmp1,
                                              ParOptVec *wtmp2 ){
  const double min_rho_penalty_search = options->getFloatOption("min_rho_penalty_search");
  const double penalty_descent_fraction = options->getFloatOption("penalty_descent_fraction");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");
  const double abs_res_tol = options->getFloatOption("abs_res_tol");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");

  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  ParOptScalar *xvals, *pxvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  px->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Add the contribution from the lower/upper bounds. Note
  // that we keep track of the positive and negative contributions
  // separately to try to avoid issues with numerical cancellations.
  // The difference is only taken at the end of the computation.
  ParOptScalar pos_result = 0.0, neg_result = 0.0;
  ParOptScalar pos_presult = 0.0, neg_presult = 0.0;

  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        if (ParOptRealPart(xvals[i] - lbvals[i]) > 1.0){
          pos_result += log(xvals[i] - lbvals[i]);
        }
        else {
          neg_result += log(xvals[i] - lbvals[i]);
        }

        if (ParOptRealPart(pxvals[i]) > 0.0){
          pos_presult += pxvals[i]/(xvals[i] - lbvals[i]);
        }
        else {
          neg_presult += pxvals[i]/(xvals[i] - lbvals[i]);
        }
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        if (ParOptRealPart(ubvals[i] - xvals[i]) > 1.0){
          pos_result += log(ubvals[i] - xvals[i]);
        }
        else {
          neg_result += log(ubvals[i] - xvals[i]);
        }

        if (ParOptRealPart(pxvals[i]) > 0.0){
          neg_presult -= pxvals[i]/(ubvals[i] - xvals[i]);
        }
        else {
          pos_presult -= pxvals[i]/(ubvals[i] - xvals[i]);
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
  if (nwcon > 0 && sparse_inequality){
    ParOptScalar *swvals, *pswvals;
    sw->getArray(&swvals);
    psw->getArray(&pswvals);

    for ( int i = 0; i < nwcon; i++ ){
      if (ParOptRealPart(swvals[i]) > 1.0){
        pos_result += log(swvals[i]);
      }
      else {
        neg_result += log(swvals[i]);
      }

      if (ParOptRealPart(pswvals[i]) > 0.0){
        pos_presult += pswvals[i]/swvals[i];
      }
      else {
        neg_presult += pswvals[i]/swvals[i];
      }
    }
  }

  // Compute the norm of the weight constraint infeasibility
  ParOptScalar weight_infeas = 0.0, weight_proj = 0.0;
  if (nwcon > 0){
    prob->evalSparseCon(x, wtmp1);
    if (sparse_inequality){
      wtmp1->axpy(-1.0, sw);
    }
    weight_infeas = wtmp1->norm();

    // Compute the projection of the weight constraints
    // onto the descent direction

    // Compute (cw(x) - sw)^{T}*(Aw(x)*px - psw)
    wtmp2->zeroEntries();
    prob->addSparseJacobian(1.0, x, px, wtmp2);

    if (sparse_inequality){
      weight_proj = wtmp1->dot(wtmp2) - wtmp1->dot(psw);
    }
    else {
      weight_proj = wtmp1->dot(wtmp2);
    }

    // Complete the weight projection computation
    if (ParOptRealPart(weight_infeas) > 0.0){
      weight_proj = weight_proj/weight_infeas;
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
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      // Add the terms from the s-slack variables
      if (ParOptRealPart(s[i]) > 1.0){
        pos_result += log(s[i]);
      }
      else {
        neg_result += log(s[i]);
      }
      if (ParOptRealPart(ps[i]) > 0.0){
        pos_presult += ps[i]/s[i];
      }
      else {
        neg_presult += ps[i]/s[i];
      }

      // Add the terms from the t-slack variables
      if (ParOptRealPart(t[i]) > 1.0){
        pos_result += log(t[i]);
      }
      else {
        neg_result += log(t[i]);
      }

      if (ParOptRealPart(pt[i]) > 0.0){
        pos_presult += pt[i]/t[i];
      }
      else {
        neg_presult += pt[i]/t[i];
      }
    }
  }

  // Compute the projected derivative
  ParOptScalar proj = g->dot(px);
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      proj += penalty_gamma[i]*pt[i];
    }
  }

  // Perform the computations only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // The values of the merit function and its derivative
  ParOptScalar merit = 0.0;
  ParOptScalar pmerit = 0.0;

  // Compute the infeasibility
  ParOptScalar dense_infeas = 0.0, dense_proj = 0.0;
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      dense_infeas += (c[i] - s[i] + t[i])*(c[i] - s[i] + t[i]);
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      dense_infeas += c[i]*c[i];
    }
  }
  dense_infeas = sqrt(dense_infeas);

  // Compute the projection depending on whether this is
  // for an exact or inexact step
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      dense_proj += (c[i] - s[i] + t[i])*(Ac[i]->dot(px) - ps[i] + pt[i]);
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      dense_proj += c[i]*Ac[i]->dot(px);
    }
  }

  // Complete the projected derivative computation for the dense
  // constraints
  if (ParOptRealPart(dense_infeas) > 0.0){
    dense_proj = dense_proj/dense_infeas;
  }

  // Compute the product px^{T}*B*px
  ParOptScalar pTBp = 0.0;
  if (use_diag_hessian){
    ParOptScalar local = 0.0;
    ParOptScalar *hvals;
    hdiag->getArray(&hvals);
    for ( int i = 0; i < nvars; i++ ){
      local += pxvals[i]*pxvals[i]*hvals[i];
    }
    MPI_Allreduce(&local, &pTBp, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  }
  else if (qn){
    qn->mult(px, xtmp);
    pTBp = 0.5*xtmp->dot(px);
  }

  if (rank == opt_root){
    // Now, set up the full problem infeasibility
    ParOptScalar infeas = dense_infeas + weight_infeas;
    ParOptScalar infeas_proj = dense_proj + weight_proj;

    // Compute the numerator term
    ParOptScalar numer = proj - barrier_param*(pos_presult + neg_presult);
    if (ParOptRealPart(pTBp) > 0.0){
      numer += 0.5*pTBp;
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
    if (ParOptRealPart(infeas) > 0.01*abs_res_tol){
      rho_hat = -ParOptRealPart(numer)/
        ParOptRealPart(infeas_proj + penalty_descent_fraction*max_x*infeas);
    }

    // Set the penalty parameter to the smallest value
    // if it is greater than the old value
    if (rho_hat > rho_penalty_search){
      rho_penalty_search = rho_hat;
    }
    else {
      // Damp the value of the penalty parameter
      rho_penalty_search *= 0.5;
      if (rho_penalty_search < rho_hat){
        rho_penalty_search = rho_hat;
      }
    }

    // Last check: Make sure that the penalty parameter is at
    // least larger than the minimum allowable value
    if (rho_penalty_search < min_rho_penalty_search){
      rho_penalty_search = min_rho_penalty_search;
    }

    // Now, evaluate the merit function and its derivative
    // based on the new value of the penalty parameter
    merit = (fobj - barrier_param*(pos_result + neg_result) +
             rho_penalty_search*infeas);
    pmerit = (proj - barrier_param*(pos_presult + neg_presult) +
              rho_penalty_search*infeas_proj);

    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        merit += penalty_gamma[i]*t[i];
      }
    }
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
int ParOptInteriorPoint::lineSearch( double alpha_min, double *_alpha,
                                     ParOptScalar m0, ParOptScalar dm0 ){
  // Get parameters for the line search method
  const int max_line_iters = options->getIntOption("max_line_iters");
  const int use_backtracking_alpha = options->getBoolOption("use_backtracking_alpha");
  const double armijo_constant = options->getFloatOption("armijo_constant");
  const double function_precision = options->getFloatOption("function_precision");
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

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (output_level > 0){
    double pxnorm = px->maxabs();
    if (outfp && rank == opt_root){
      fprintf(outfp, "%5s %7s %25s %12s %12s %12s\n",
              "iter", "alpha", "merit", "dmerit", "||px||", "min(alpha)");
      fprintf(outfp, "%5d %7s %25.16e %12.5e %12.5e %12.5e\n",
              0, " ", ParOptRealPart(m0), ParOptRealPart(dm0),
              pxnorm, alpha_min);
    }
  }

  int j = 0;
  for ( ; j < max_line_iters; j++ ){
    // Set rx = x + alpha*px
    rx->copyValues(x);
    computeStepVec(rx, alpha, px, lb, NULL, ub, NULL);

    // Set rcw = sw + alpha*psw
    ParOptScalar zero = 0.0;
    if (nwcon > 0 && sparse_inequality){
      rsw->copyValues(sw);
      computeStepVec(rsw, alpha, psw, NULL, &zero, NULL, NULL);
    }

    // Set rs = s + alpha*ps and rt = t + alpha*pt
    if (dense_inequality){
      memcpy(rs, s, ncon*sizeof(ParOptScalar));
      computeStep(ncon, rs, alpha, ps, NULL, &zero, NULL, NULL);
      memcpy(rt, t, ncon*sizeof(ParOptScalar));
      computeStep(ncon, rt, alpha, pt, NULL, &zero, NULL, NULL);
    }

    // Evaluate the objective and constraints at the new point
    int fail_obj = prob->evalObjCon(rx, &fobj, c);
    neval++;

    if (fail_obj){
      fprintf(stderr, "ParOpt: Evaluation failed during line search, "
              "trying new point\n");

      // Multiply alpha by 1/10 to avoid the undefined region
      alpha *= 0.1;
      continue;
    }

    // Evaluate the merit function
    merit = evalMeritFunc(fobj, c, rx, rs, rt, rsw);

    // Print out the merit function and step at the current iterate
    if (outfp && rank == opt_root && output_level > 0){
      fprintf(outfp, "%5d %7.1e %25.16e %12.5e\n", j+1, alpha,
              ParOptRealPart(merit), ParOptRealPart((merit - m0)/alpha));
    }

    // If the best alpha value is negative, then this must be the
    // first successful evaluation. Otherwise, if this merit value
    // is better than previous merit function values, store the new
    // best merit function value.
    if (best_alpha < 0.0 ||
        ParOptRealPart(merit) < ParOptRealPart(best_merit)){
      best_alpha = alpha;
      best_merit = merit;
    }

    // Check the sufficient decrease condition. Note that this is
    // relaxed by the specified function precision. This allows
    // acceptance of steps that violate the sufficient decrease
    // condition within the precision limit of the objective/merit
    // function.
    if (ParOptRealPart(merit) - armijo_constant*alpha*ParOptRealPart(dm0) <
        (ParOptRealPart(m0) + function_precision)){
      // If this is the minimum alpha value, then we're at the minimum
      // line search step and we have had success
      if (fail & PAROPT_LINE_SEARCH_MIN_STEP){
        fail = PAROPT_LINE_SEARCH_SUCCESS | PAROPT_LINE_SEARCH_MIN_STEP;
      }
      else {
        // We have successfully found a point
        fail = PAROPT_LINE_SEARCH_SUCCESS;
      }

      // The line search may be successful but the merit function value may
      // not have resulted in no improvement.
      if ((ParOptRealPart(merit) <= ParOptRealPart(m0) + function_precision) &&
          (ParOptRealPart(merit) + function_precision >= ParOptRealPart(m0))){
        fail |= PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
      }
      break;
    }
    else if (fail & PAROPT_LINE_SEARCH_MIN_STEP){
      // If this is the minimum alpha value, then quit the line search loop
      break;
    }

    // Update the new value of alpha
    if (j < max_line_iters-1){
      if (use_backtracking_alpha){
        alpha = 0.5*alpha;
        if (alpha <= alpha_min){
          alpha = alpha_min;
          fail |= PAROPT_LINE_SEARCH_MIN_STEP;
        }
      }
      else {
        double alpha_new = -0.5*ParOptRealPart(dm0)*(alpha*alpha)/
          ParOptRealPart(merit - m0 - dm0*alpha);

        // Bound the new step length from below by 0.01
        if (alpha_new <= alpha_min){
          alpha = alpha_min;
          fail |= PAROPT_LINE_SEARCH_MIN_STEP;
        }
        else if (alpha_new < 0.01*alpha){
          alpha = 0.01*alpha;
        }
        else {
          alpha = alpha_new;
        }
      }
    }
  }

  // The line search existed with the maximum number of line search
  // iterations
  if (j == max_line_iters){
    fail |= PAROPT_LINE_SEARCH_MAX_ITERS;
  }

  // Check the status and return.
  if (!(fail & PAROPT_LINE_SEARCH_SUCCESS)){
    // Check for a simple decrease within the function precision,
    // then this is sufficient to accept the step.
    if (ParOptRealPart(best_merit) <=
        ParOptRealPart(m0) + function_precision){
      // We're going to say that this is success
      fail |= PAROPT_LINE_SEARCH_SUCCESS;
      // Turn off the fail flag
      fail &= ~PAROPT_LINE_SEARCH_FAILURE;
    }
    else if ((ParOptRealPart(merit) <= ParOptRealPart(m0) + function_precision) &&
             (ParOptRealPart(merit) + function_precision >= ParOptRealPart(m0))){
      // Check if there is no significant change in the function value
      fail |= PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
    }

    // If we're about to accept the best alpha value, then we have
    // to re-evaluate the function at this point since the gradient
    // will be evaluated here next, and we always have to evaluate
    // function then gradient.
    if (alpha != best_alpha){
      alpha = best_alpha;

      // Set rx = x + alpha*px
      rx->copyValues(x);
      computeStepVec(rx, alpha, px, lb, NULL, ub, NULL);

      // Set rcw = sw + alpha*psw
      ParOptScalar zero = 0.0;
      if (nwcon > 0 && sparse_inequality){
        rsw->copyValues(sw);
        computeStepVec(rsw, alpha, psw, NULL, &zero, NULL, NULL);
      }

      // Set rs = s + alpha*ps and rt = t + alpha*pt
      if (dense_inequality){
        memcpy(rs, s, ncon*sizeof(ParOptScalar));
        computeStep(ncon, rs, alpha, ps, NULL, &zero, NULL, NULL);
        memcpy(rt, t, ncon*sizeof(ParOptScalar));
        computeStep(ncon, rt, alpha, pt, NULL, &zero, NULL, NULL);
      }

      // Evaluate the objective and constraints at the new point
      int fail_obj = prob->evalObjCon(rx, &fobj, c);
      neval++;

      // This should not happen, since we've already evaluated
      // the function at this point at a previous line search
      // iteration.
      if (fail_obj){
        fprintf(stderr, "ParOpt: Evaluation failed during line search\n");
        fail = PAROPT_LINE_SEARCH_FAILURE;
      }
    }
    else {
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

  @param alpha The step length to take
  @param eval_obj_con Flag indicating whether to evaluate the obj/cons
  @param perform_qn_update Flag indicating whether to update the quasi-Newton method
  @returns The type of quasi-Newton update performed
*/
int ParOptInteriorPoint::computeStepAndUpdate( double alpha,
                                               int eval_obj_con,
                                               int perform_qn_update ){
  const int use_quasi_newton_update = options->getBoolOption("use_quasi_newton_update");

  // Set the new values of the variables
  ParOptScalar zero = 0.0;
  if (nwcon > 0){
    computeStepVec(zw, alpha, pzw, NULL, &zero, NULL, NULL);
    if (sparse_inequality){
      computeStepVec(sw, alpha, psw, NULL, &zero, NULL, NULL);
    }
  }
  if (use_lower){
    computeStepVec(zl, alpha, pzl, NULL, &zero, NULL, NULL);
  }
  if (use_upper){
    computeStepVec(zu, alpha, pzu, NULL, &zero, NULL, NULL);
  }

  computeStep(ncon, z, alpha, pz, NULL, &zero, NULL, NULL);
  if (dense_inequality){
    computeStep(ncon, s, alpha, ps, NULL, &zero, NULL, NULL);
    computeStep(ncon, t, alpha, pt, NULL, &zero, NULL, NULL);
    computeStep(ncon, zt, alpha, pzt, NULL, &zero, NULL, NULL);
  }

  // Compute the negative gradient of the Lagrangian using the
  // old gradient information with the new multiplier estimates
  if (qn && perform_qn_update && use_quasi_newton_update){
    y_qn->copyValues(g);
    y_qn->scale(-1.0);
    for ( int i = 0; i < ncon; i++ ){
      y_qn->axpy(z[i], Ac[i]);
    }

    // Add the term: Aw^{T}*zw
    if (nwcon > 0){
      prob->addSparseJacobianTranspose(1.0, x, zw, y_qn);
    }
  }

  // Apply the step to the design variables only
  // after computing the contribution of the constraint
  // Jacobian to the BFGS update
  computeStepVec(x, alpha, px, lb, NULL, ub, NULL);

  // Evaluate the objective if needed. This step is not required
  // if a line search has just been performed.
  if (eval_obj_con){
    int fail_obj = prob->evalObjCon(x, &fobj, c);
    neval++;
    if (fail_obj){
      fprintf(stderr,
              "ParOpt: Function and constraint evaluation failed\n");
      return fail_obj;
    }
  }

  // Evaluate the derivative at the new point
  int fail_gobj = prob->evalObjConGradient(x, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr,
            "ParOpt: Gradient evaluation failed at final line search\n");
  }

  // Compute the Quasi-Newton update
  int update_type = 0;
  if (qn && perform_qn_update){
    if (use_quasi_newton_update){
      // Add the new gradient of the Lagrangian with the new
      // multiplier estimates.
      // Compute the step - scale by the step length
      s_qn->copyValues(px);
      s_qn->scale(alpha);

      // Finish computing the difference in gradients
      y_qn->axpy(1.0, g);
      for ( int i = 0; i < ncon; i++ ){
        y_qn->axpy(-z[i], Ac[i]);
      }

      // Add the term: -Aw^{T}*zw
      if (nwcon > 0){
        prob->addSparseJacobianTranspose(-1.0, x, zw, y_qn);
      }

      prob->computeQuasiNewtonUpdateCorrection(s_qn, y_qn);
      update_type = qn->update(x, z, zw, s_qn, y_qn);
    }
    else {
      update_type = qn->update(x, z, zw);
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
void ParOptInteriorPoint::initAndCheckDesignAndBounds(){
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Get the design variables and bounds
  prob->getVarsAndBounds(x, lb, ub);

  // Check the design variables and bounds, move things that
  // don't make sense and print some warnings
  ParOptScalar *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Check the variable values to see if they are reasonable
  double rel_bound = 0.001*barrier_param;
  int check_flag = 0;
  if (use_lower && use_upper){
    for ( int i = 0; i < nvars; i++ ){
      // Fixed variables are not allowed
      ParOptScalar delta = 1.0;
      if (ParOptRealPart(lbvals[i]) > -max_bound_value &&
          ParOptRealPart(ubvals[i]) < max_bound_value){
        if (ParOptRealPart(lbvals[i]) >= ParOptRealPart(ubvals[i])){
          check_flag = (check_flag | 1);
          // Make up bounds
          lbvals[i] = 0.5*(lbvals[i] + ubvals[i]) - 0.5*rel_bound;
          ubvals[i] = lbvals[i] + rel_bound;
        }
        delta = ubvals[i] - lbvals[i];
      }

      // Check if x is too close the boundary
      if (ParOptRealPart(lbvals[i]) > -max_bound_value &&
          ParOptRealPart(xvals[i]) < ParOptRealPart(lbvals[i] + rel_bound*delta)){
        check_flag = (check_flag | 2);
        xvals[i] = lbvals[i] + rel_bound*delta;
      }
      if (ParOptRealPart(ubvals[i]) < max_bound_value &&
          ParOptRealPart(xvals[i]) > ParOptRealPart(ubvals[i] - rel_bound*delta)){
        check_flag = (check_flag | 4);
        xvals[i] = ubvals[i] - rel_bound*delta;
      }
    }
  }

  // Perform a bitwise global OR operation
  int tmp_check_flag = check_flag;
  MPI_Allreduce(&tmp_check_flag, &check_flag, 1, MPI_INT, MPI_BOR, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Print the results of the warnings
  if (rank == 0 && outfp){
    if (check_flag & 1){
      fprintf(outfp, "ParOpt Warning: Variable bounds are inconsistent\n");
    }
    if (check_flag & 2){
      fprintf(outfp,
              "ParOpt Warning: Variables may be too close to lower bound\n");
    }
    if (check_flag & 4){
      fprintf(outfp,
              "ParOpt Warning: Variables may be too close to upper bound\n");
    }
  }

  // Set the largrange multipliers with bounds outside the limits to
  // zero. This ensures that they have no effect because they will not
  // be updated once the optimization begins.
  ParOptScalar *zlvals, *zuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  for ( int i = 0; i < nvars; i++ ){
    if (ParOptRealPart(lbvals[i]) <= -max_bound_value){
      zlvals[i] = 0.0;
    }
    if (ParOptRealPart(ubvals[i]) >= max_bound_value){
      zuvals[i] = 0.0;
    }
  }
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
int ParOptInteriorPoint::optimize( const char *checkpoint ){
  // Set the stopping criteria constants
  const double abs_res_tol = options->getFloatOption("abs_res_tol");
  const double rel_func_tol = options->getFloatOption("rel_func_tol");
  const double abs_step_tol = options->getFloatOption("abs_step_tol");

  // Set the default norm type
  const char *norm_name = options->getEnumOption("norm_type");
  ParOptNormType norm_type = PAROPT_L2_NORM;
  if (strcmp(norm_name, "infinity") == 0){
    norm_type = PAROPT_INFTY_NORM;
  }
  else if (strcmp(norm_name, "l1") == 0){
    norm_type = PAROPT_L1_NORM;
  }

  // Set the default starting point strategy
  const char *start_name = options->getEnumOption("starting_point_strategy");
  ParOptStartingPointStrategy starting_point_strategy =
    PAROPT_NO_START_STRATEGY;
  if (strcmp(start_name, "least_squares_multipliers") == 0){
    starting_point_strategy = PAROPT_LEAST_SQUARES_MULTIPLIERS;
  }
  else if (strcmp(start_name, "affine_step") == 0){
    starting_point_strategy = PAROPT_AFFINE_STEP;
  }

  // Set the barrier strategy
  const char *barrier_name = options->getEnumOption("barrier_strategy");
  ParOptBarrierStrategy barrier_strategy = PAROPT_COMPLEMENTARITY_FRACTION;
  if (strcmp(barrier_name, "monotone") == 0){
    barrier_strategy = PAROPT_MONOTONE;
  }
  else if (strcmp(barrier_name, "mehrotra") == 0){
    barrier_strategy = PAROPT_MEHROTRA;
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
  const int hessian_reset_freq =
    options->getIntOption("hessian_reset_freq");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");
  const int sequential_linear_method =
    options->getBoolOption("sequential_linear_method");

  // Adjust whether to use the diagonal contribution to the Hessian
  if (!hdiag && use_diag_hessian){
    hdiag = prob->createDesignVec();
    hdiag->incref();
  }
  else if (hdiag && !use_diag_hessian){
    hdiag->decref();
    hdiag = NULL;
  }

  // Check if the GMRES subspace is large enough
  int m = options->getIntOption("gmres_subspace_size");
  if (m != gmres_subspace_size){
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
  const double design_precision =
    options->getFloatOption("design_precision");

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
  if (gradient_verification_frequency > 0){
    prob->checkGradients(gradient_check_step_length,
                         x, use_hvec_product);
  }

  // Zero out the number of function/gradient/hessian evaluations
  niter = neval = ngeval = nhvec = 0;

  // If no quasi-Newton method is defined, use a sequential linear method instead
  if (!sequential_linear_method && !qn){
    fprintf(stderr, "ParOpt Error: Must use a sequential linear method if no "
            "quasi-Newton approximation is defined\n");
    return 1;
  }

  // Initialize and check the design variables and bounds
  initAndCheckDesignAndBounds();

  // Print what options we're using to the file
  printOptionSummary(outfp);

  // Evaluate the objective, constraint and their gradients at the
  // current values of the design variables
  int fail_obj = prob->evalObjCon(x, &fobj, c);
  neval++;
  if (fail_obj){
    fprintf(stderr,
            "ParOpt: Initial function and constraint evaluation failed\n");
    return fail_obj;
  }
  int fail_gobj = prob->evalObjConGradient(x, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr, "ParOpt: Initial gradient evaluation failed\n");
    return fail_obj;
  }

  if (starting_point_strategy == PAROPT_AFFINE_STEP){
    initAffineStepMultipliers(norm_type);
  }
  else if (starting_point_strategy == PAROPT_LEAST_SQUARES_MULTIPLIERS){
    initLeastSquaresMultipliers();
  }

  // Some quasi-Newton methods can be updated with only the design variable
  // values and the multiplier estimates
  if (qn && !use_quasi_newton_update){
    qn->update(x, z, zw);
  }

  // Retrieve the rank of the processor
  int rank;
  MPI_Comm_rank(comm, &rank);

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
  info[0] = '\0';

  for ( int k = 0; k < max_major_iters; k++, niter++ ){
    if (qn && !sequential_linear_method){
      if (k > 0 && k % hessian_reset_freq == 0 &&
          use_quasi_newton_update){
        // Reset the quasi-Newton Hessian approximation
        qn->reset();

        // Add a reset flag to the output
        if (rank == opt_root){
          sprintf(&info[strlen(info)], "%s ", "resetH");
        }
      }
    }

    // Print out the current solution progress using the
    // hook in the problem definition
    if (write_output_frequency > 0 && k % write_output_frequency == 0){
      if (checkpoint){
        // Write the checkpoint file, if it fails once, set
        // the file pointer to null so it won't print again
        if (writeSolutionFile(checkpoint)){
          fprintf(stderr, "ParOpt: Checkpoint file %s creation failed\n",
                  checkpoint);
          checkpoint = NULL;
        }
      }
      prob->writeOutput(k, x);
    }

    // Print to screen the gradient check results at
    // iteration k
    if (k > 0 && (gradient_verification_frequency > 0) &&
        (k % gradient_verification_frequency == 0)){
      prob->checkGradients(gradient_check_step_length, x,
                           use_hvec_product);
    }

    // Determine if we should switch to a new barrier problem or not
    int rel_function_test =
      (alpha_xprev == 1.0 && alpha_zprev == 1.0 &&
       (fabs(ParOptRealPart(fobj - fobj_prev)) <
        rel_func_tol*fabs(ParOptRealPart(fobj_prev))));

    // Set the line search check. If there is no change in the merit
    // function value, and we're feasible and the complementarity
    // conditions are satisfied, then declare the test passed.
    if (no_merit_function_improvement){
      line_search_test += 1;
    }
    else {
      line_search_test = 0;
    }

    // Compute the complementarity
    ParOptScalar comp = computeComp();

    // Keep track of the norm of the different parts of the
    // KKT conditions
    double max_prime = 0.0, max_dual = 0.0, max_infeas = 0.0;

    // Compute the overall norm of the KKT conditions
    double res_norm = 0.0;

    if (barrier_strategy == PAROPT_MONOTONE){
      // Compute the residual of the KKT system
      computeKKTRes(barrier_param, norm_type,
                    &max_prime, &max_dual, &max_infeas, &res_norm);

      // Compute the maximum of the norm of the residuals
      if (k == 0){
        res_norm_prev = res_norm;
      }

      // Set the flag to indicate whether the barrier problem has
      // converged
      int barrier_converged = 0;
      if (k > 0 && ((res_norm < 10.0*barrier_param) ||
                    rel_function_test ||
                    (line_search_test >= 2))){
        barrier_converged = 1;
      }

      // Keep track of the new barrier parameter (if any). Only set the
      // new barrier parameter after we've check for convergence of the
      // overall algorithm. This ensures that the previous barrier
      // parameter is saved if we successfully converge.
      double new_barrier_param = 0.0;

      // Broadcast the result of the test from the root processor
      MPI_Bcast(&barrier_converged, 1, MPI_INT, opt_root, comm);

      if (barrier_converged){
        const double monotone_barrier_fraction =
          options->getFloatOption("monotone_barrier_fraction");
        const double monotone_barrier_power =
          options->getFloatOption("monotone_barrier_power");

        // If the barrier problem converged, we need a new convergence
        // test, but if the barrier parameter is  converged
        if (barrier_param > 0.1*abs_res_tol){
          line_search_test = 0;
        }

        // Compute the new barrier parameter: It is either:
        // 1. A fixed fraction of the old value
        // 2. A function mu**exp for some exp > 1.0
        // Point 2 ensures superlinear convergence (eventually)
        double mu_frac = monotone_barrier_fraction*barrier_param;
        double mu_pow = pow(barrier_param, monotone_barrier_power);

        new_barrier_param = mu_frac;
        if (mu_pow < mu_frac){
          new_barrier_param = mu_pow;
        }

        // Truncate the barrier parameter at 0.1*abs_res_tol. If this
        // truncation occurs, set the flag that this is the final
        // barrier problem
        if (new_barrier_param < 0.1*abs_res_tol){
          new_barrier_param = 0.09999*abs_res_tol;
        }

        // Compute the new barrier parameter value
        computeKKTRes(new_barrier_param, norm_type,
                      &max_prime, &max_dual, &max_infeas, &res_norm);

        // Reset the penalty parameter to the min allowable value
        rho_penalty_search =
          options->getFloatOption("min_rho_penalty_search");

        // Set the new barrier parameter
        barrier_param = new_barrier_param;
      }
    }
    else if (barrier_strategy == PAROPT_MEHROTRA){
      // Compute the residual of the KKT system
      computeKKTRes(barrier_param, norm_type,
                    &max_prime, &max_dual, &max_infeas, &res_norm);

      if (k == 0){
        res_norm_prev = res_norm;
      }
    }
    else if (barrier_strategy == PAROPT_COMPLEMENTARITY_FRACTION){
      const double monotone_barrier_fraction =
        options->getFloatOption("monotone_barrier_fraction");

      barrier_param = monotone_barrier_fraction*ParOptRealPart(comp);
      if (barrier_param < 0.1*abs_res_tol){
        barrier_param = 0.1*abs_res_tol;
      }

      // Compute the residual of the KKT system
      computeKKTRes(barrier_param, norm_type,
                    &max_prime, &max_dual, &max_infeas, &res_norm);

      if (k == 0){
        res_norm_prev = res_norm;
      }
    }

    // Print all the information we can to the screen...
    if (outfp && rank == opt_root){
      if (k % 10 == 0 || output_level > 0){
        fprintf(outfp, "\n%4s %4s %4s %4s %7s %7s %7s %12s %7s %7s %7s "
                "%7s %7s %8s %7s info\n",
                "iter", "nobj", "ngrd", "nhvc", "alpha", "alphx", "alphz",
                "fobj", "|opt|", "|infes|", "|dual|", "mu",
                "comp", "dmerit", "rho");
      }

      if (k == 0){
        fprintf(outfp, "%4d %4d %4d %4d %7s %7s %7s %12.5e %7.1e %7.1e "
                "%7.1e %7.1e %7.1e %8s %7s %s\n",
                k, neval, ngeval, nhvec, "--", "--", "--",
                ParOptRealPart(fobj), max_prime, max_infeas, max_dual,
                barrier_param, ParOptRealPart(comp), "--", "--", info);
      }
      else {
        fprintf(outfp, "%4d %4d %4d %4d %7.1e %7.1e %7.1e %12.5e %7.1e "
                "%7.1e %7.1e %7.1e %7.1e %8.1e %7.1e %s\n",
                k, neval, ngeval, nhvec,
                alpha_prev, alpha_xprev, alpha_zprev,
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
    if (k > 0 && (barrier_param <= 0.1*abs_res_tol) &&
        (res_norm < abs_res_tol ||
         rel_function_test ||
         (line_search_test >= 2))){
      if (outfp && rank == opt_root){
        if (rel_function_test){
          fprintf(outfp, "\nParOpt: Successfully converged on relative function test\n");
        }
        else if (line_search_test >= 2){
          fprintf(outfp, "\nParOpt Warning: Current design point could not be improved. "
                  "No barrier function decrease in previous two iterations\n");
        }
        else {
          fprintf(outfp, "\nParOpt: Successfully converged to requested tolerance\n");
        }
      }
      converged = 1;
    }

    // Broadcast the convergence result from the root processor. This avoids
    // comparing values that might be different on different procs.
    MPI_Bcast(&converged, 1, MPI_INT, opt_root, comm);

    // Everybody quit altogether if we've converged
    if (converged){
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

    if (use_hvec_product){
      // Compute the relative GMRES tolerance given the residuals
      double gmres_rtol =
        eisenstat_walker_gamma*pow((res_norm/res_norm_prev),
                                   eisenstat_walker_alpha);

      if (max_prime < nk_switch_tol &&
          max_dual < nk_switch_tol &&
          max_infeas < nk_switch_tol &&
          gmres_rtol < max_gmres_rtol){
        // Set the flag which determines whether or not to use
        // the quasi-Newton method as a preconditioner
        int use_qn = 1;
        if (sequential_linear_method ||
            !use_qn_gmres_precon){
          use_qn = 0;
        }

        // Set up the KKT diagonal system
        setUpKKTDiagSystem(s_qn, wtemp, use_qn);

        // Set up the full KKT system
        setUpKKTSystem(ztemp, s_qn, y_qn, wtemp, use_qn);

        // Compute the inexact step using GMRES
        gmres_iters =
          computeKKTGMRESStep(ztemp, y_qn, s_qn, wtemp,
                              gmres_rtol, gmres_atol, use_qn);

        if (abs_step_tol > 0.0){
          step_norm_prev = computeStepNorm(norm_type);
        }

        if (gmres_iters < 0){
          // Print out an error code that we've failed
          if (rank == opt_root && output_level > 0){
            fprintf(outfp, "      %9s\n", "step failed");
          }

          // Recompute the residual of the KKT system - the residual
          // was destroyed during the failed GMRES iteration
          computeKKTRes(barrier_param, norm_type,
                        &max_prime, &max_dual, &max_infeas);
        }
        else {
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

    // Is this a sequential linear step that did not use the quasi-Newton approx.
    int seq_linear_step = 0;

    // This is a step that uses only the diagonal contribution from the quasi-Newton
    // approximation
    int diagonal_quasi_newton_step = 0;

    // Compute a step based on the quasi-Newton Hessian approximation
    if (!inexact_newton_step){
      int use_qn = 1;

      if (sequential_linear_method){
        // If we're using a sequential linear method, set use_qn = 0.
        use_qn = 0;
      }
      else if (line_search_failed && !use_quasi_newton_update){
        // In this case, the line search failed, and we are using a fixed
        // quasi-Newton method which was therefore not reset when the
        // line search failed. As a result, we try either a sequential linear
        // step or discard the vectors in the quasi-Newton Hessian approx.
        // leaving only the diagonal contributions, which will only work if
        // b0 is positive.
        // Check if the coefficient b0 is positive.
        use_qn = 0;
        seq_linear_step = 1;
        if (qn){
          ParOptScalar b0;
          qn->getCompactMat(&b0, NULL, NULL, NULL);
          if (ParOptRealPart(b0) > 0.0){
            seq_linear_step = 0;
            diagonal_quasi_newton_step = 1;
          }
        }
      }
      else if (use_diag_hessian){
        // If we're using a diagonal Hessian approximation, compute it here
        use_qn = 0;
        int fail = prob->evalHessianDiag(x, z, zw, hdiag);
        if (fail){
          fprintf(stderr,
                  "ParOpt: Hessian diagonal evaluation failed\n");
          return fail;
        }
      }

      // Compute the affine residual with barrier = 0.0 if we are using
      // the Mehrotra probing barrier strategy
      if (barrier_strategy == PAROPT_MEHROTRA){
        computeKKTRes(0.0, norm_type, &max_prime, &max_dual, &max_infeas);
      }

      // Set up the KKT diagonal system. If we're using only the
      // diagonal entries from a quasi-Newton approximation, turn those
      // on here.
      if (diagonal_quasi_newton_step){
        use_qn = 1;
      }
      setUpKKTDiagSystem(s_qn, wtemp, use_qn);

      // Set up the full KKT system
      setUpKKTSystem(ztemp, s_qn, y_qn, wtemp, use_qn);

      // Solve for the KKT step. If we're using only the diagonal entries,
      // turn off the off-diagonal entries to compute the step.
      if (diagonal_quasi_newton_step){
        use_qn = 0;
      }
      computeKKTStep(ztemp, s_qn, y_qn, wtemp, use_qn);

      // Compute the norm of the step length. This is only used if the
      // abs_step_tol is set. It defaults to zero.
      if (abs_step_tol > 0.0){
        step_norm_prev = computeStepNorm(norm_type);
      }

      if (barrier_strategy == PAROPT_MEHROTRA){
        // Compute the affine step to the boundary, allowing
        // the variables to go right to zero
        double max_x, max_z;
        computeMaxStep(1.0, &max_x, &max_z);

        // Compute the complementarity at the full step
        ParOptScalar comp_affine = computeCompStep(max_x, max_z);

        // Use the Mehrotra rule
        double s1 = ParOptRealPart(comp_affine/comp);
        double sigma = s1*s1*s1;

        // Set the bounds on sigma between [0.01, 1.0]
        if (sigma < 0.01){
          sigma = 0.01;
        }
        else if (sigma > 1.0){
          sigma = 1.0;
        }

        // Compute the new adaptive barrier parameter
        barrier_param = sigma*ParOptRealPart(comp);
        if (barrier_param < 0.09999*abs_res_tol){
          barrier_param = 0.09999*abs_res_tol;
        }

        // Compute the residual with the new barrier parameter
        computeKKTRes(barrier_param, norm_type,
                      &max_prime, &max_dual, &max_infeas);

        // Compute the KKT Step
        computeKKTStep(ztemp, s_qn, y_qn, wtemp, use_qn);
      }
    }

    // Check the KKT step
    if (gradient_verification_frequency > 0 &&
        ((k % gradient_verification_frequency) == 0)){
      checkKKTStep(k, inexact_newton_step);
    }

    // Compute the maximum permitted line search lengths
    double tau = min_fraction_to_boundary;
    double tau_mu = 1.0 - barrier_param;
    if (tau_mu >= tau){
      tau = tau_mu;
    }

    double alpha_x = 1.0, alpha_z = 1.0;
    int ceq_step = scaleKKTStep(tau, comp, inexact_newton_step,
                                &alpha_x, &alpha_z);

    // Keep track of the step length size
    double alpha = 1.0;

    // Flag to indicate whether the line search failed
    int line_fail = PAROPT_LINE_SEARCH_FAILURE;

    // The type of quasi-Newton update performed
    int update_type = 0;

    // Keep track of whether the line search was skipped or not
    int line_search_skipped = 0;

    // By default, we assume that there is an improvement in the merit function
    no_merit_function_improvement = 0;

    if (use_line_search){
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter
      ParOptScalar m0, dm0;
      evalMeritInitDeriv(alpha_x, &m0, &dm0, rx, wtemp, rcw);

      // Store the merit function derivative
      dm0_prev = dm0;

      // If the derivative of the merit function is positive, but within
      // the function precision of zero, then go ahead and skip the line
      // search and update.
      if (ParOptRealPart(dm0) >= 0.0 &&
          ParOptRealPart(dm0) <= function_precision){
        line_search_skipped = 1;

        // Perform a step and update the quasi-Newton Hessian approximation
        int eval_obj_con = 1;
        int perform_qn_update = 1;
        update_type = computeStepAndUpdate(alpha, eval_obj_con,
                                           perform_qn_update);

        // Check if there was no change in the objective function
        if ((ParOptRealPart(fobj_prev) + function_precision <= ParOptRealPart(fobj)) &&
            (ParOptRealPart(fobj) + function_precision <= ParOptRealPart(fobj_prev))){
          line_fail = PAROPT_LINE_SEARCH_NO_IMPROVEMENT;
        }
      }
      else {
        // The derivative of the merit function is positive. We revert to one of two
        // approaches. If the quasi-Newton method is defined and the diagonal contribution
        // b0 is positive, then we use only this diagonal contribution to compute the
        // KKT step, otherwise, we use an SLP step.
        if (ParOptRealPart(dm0) >= 0.0){
          // Try again by disbcarding the quasi-Newton approximation. This should
          // always generate a descent direction..
          seq_linear_step = 1;
          diagonal_quasi_newton_step = 0;

          // Re-compute the KKT residuals since they may be over-written
          // during the line search step
          computeKKTRes(barrier_param, norm_type,
                        &max_prime, &max_dual, &max_infeas);

          // Set up the KKT diagonal system
          int use_qn = 0;
          if (diagonal_quasi_newton_step){
            use_qn = 1;
          }
          setUpKKTDiagSystem(s_qn, wtemp, use_qn);

          // Solve for the KKT step. Both modifications do not use the
          // quasi-Newton method at this point
          use_qn = 0;
          computeKKTStep(ztemp, s_qn, y_qn, wtemp, use_qn);

          // Scale the step
          int inexact_newton_step = 0;
          ceq_step = scaleKKTStep(tau, comp, inexact_newton_step,
                                  &alpha_x, &alpha_z);

          // Re-evaluate the merit function derivative
          evalMeritInitDeriv(alpha_x, &m0, &dm0, rx, wtemp, rcw);

          // Store the merit function derivative
          dm0_prev = dm0;
        }

        // Check that the merit function derivative is correct and print
        // the derivative to the screen on the optimization-root processor
        if (gradient_verification_frequency > 0 &&
            ((k % gradient_verification_frequency) == 0)){
          checkMeritFuncGradient(NULL, gradient_check_step_length);
        }

        if (ParOptRealPart(dm0) >= 0.0){
          line_fail = PAROPT_LINE_SEARCH_FAILURE;
        }
        else {
          // Prepare to perform the line search. First, compute the minimum
          // allowable line search step length
          double px_norm = px->maxabs();
          double alpha_min = 1.0;
          if (px_norm != 0.0){
            alpha_min = function_precision/px_norm;
          }
          if (alpha_min > 0.5){
            alpha_min = 0.5;
          }
          line_fail = lineSearch(alpha_min, &alpha, m0, dm0);

          // If the step length is less than the design precision
          if (px_norm < design_precision){
            line_fail |= PAROPT_LINE_SEARCH_SHORT_STEP;
          }

          // If the line search was successful, quit
          if (!(line_fail & PAROPT_LINE_SEARCH_FAILURE)){
            // Do not evaluate the objective and constraints at the new point
            // since we've just performed a successful line search and the
            // last point was evaluated there. Perform a quasi-Newton update
            // if required.
            int eval_obj_con = 0;
            int perform_qn_update = 1;
            update_type = computeStepAndUpdate(alpha, eval_obj_con,
                                               perform_qn_update);
          }
        }
      }
    }
    else {
      // Evaluate the objective/constraints at the new point since we skipped
      // the line search step here.
      int eval_obj_con = 1;
      int perform_qn_update = 1;
      update_type = computeStepAndUpdate(alpha, eval_obj_con,
                                         perform_qn_update);
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
        (line_fail & PAROPT_LINE_SEARCH_FAILURE)){
      qn->reset();
    }

    // Create a string to print to the screen
    if (rank == opt_root){
      // The string of unforseen events
      info[0] = '\0';
      if (gmres_iters != 0){
        // Print how well GMRES is doing
        sprintf(&info[strlen(info)], "%s%d ", "iNK", gmres_iters);
      }
      if (update_type == 1){
        // Damped BFGS update
        sprintf(&info[strlen(info)], "%s ", "dampH");
      }
      else if (update_type == 2){
        // Skipped update
        sprintf(&info[strlen(info)], "%s ", "skipH");
      }
      if (line_fail & PAROPT_LINE_SEARCH_FAILURE){
        // Line search failure
        sprintf(&info[strlen(info)], "%s ", "LFail");
      }
      if (line_fail & PAROPT_LINE_SEARCH_MIN_STEP){
        // Line search reached the minimum step length
        sprintf(&info[strlen(info)], "%s ", "LMnStp");
      }
      if (line_fail & PAROPT_LINE_SEARCH_MAX_ITERS){
        // Line search reached the max. number of iterations
        sprintf(&info[strlen(info)], "%s ", "LMxItr");
      }
      if (line_fail & PAROPT_LINE_SEARCH_NO_IMPROVEMENT){
        // Line search did not improve merit function
        sprintf(&info[strlen(info)], "%s ", "LNoImprv");
      }
      if (seq_linear_step){
        // Sequential linear step (even though we're using a QN approx.)
        sprintf(&info[strlen(info)], "%s ", "SLP");
      }
      if (diagonal_quasi_newton_step){
        // Step generated using only the diagonal from a quasi-Newton approx.
        sprintf(&info[strlen(info)], "%s ", "DQN");
      }
      if (line_search_skipped){
        // Line search reached the max. number of iterations
        sprintf(&info[strlen(info)], "%s ", "LSkip");
      }
      if (ceq_step){
        // The step lengths are equal due to an increase in the
        // the complementarity at the new step
        sprintf(&info[strlen(info)], "%s ", "cmpEq");
      }
    }
  }

  // Success - we completed the optimization
  return 0;
}

/*
  Compute an initial multiplier estimate using a least-squares method
*/
void ParOptInteriorPoint::initLeastSquaresMultipliers(){
  const double max_bound_value =
    options->getFloatOption("max_bound_value");

  // Set the largrange multipliers with bounds outside the
  // limits to zero
  ParOptScalar *lbvals, *ubvals, *zlvals, *zuvals;
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Set the Largrange multipliers associated with the
  // the lower/upper bounds to 1.0
  zl->set(1.0);
  zu->set(1.0);

  // Set the Lagrange multipliers and slack variables
  // associated with the sparse constraints to 1.0
  zw->set(1.0);
  sw->set(1.0);

  // Set the Largrange multipliers and slack variables associated
  // with the dense constraints to 1.0
  for ( int i = 0; i < ncon; i++ ){
    z[i] = 1.0;
    s[i] = 1.0;
    zt[i] = 1.0;
    t[i] = 1.0;
  }

  // Zero the multipliers for bounds that are out-of-range
  for ( int i = 0; i < nvars; i++ ){
    if (ParOptRealPart(lbvals[i]) <= -max_bound_value){
      zlvals[i] = 0.0;
    }
    if (ParOptRealPart(ubvals[i]) >= max_bound_value){
      zuvals[i] = 0.0;
    }
  }

  // Form the right-hand-side of the least squares problem for the
  // dense constraint multipliers
  ParOptVec *xt = y_qn;
  xt->copyValues(g);
  xt->axpy(-1.0, zl);
  xt->axpy(1.0, zu);

  for ( int i = 0; i < ncon; i++ ){
    z[i] = Ac[i]->dot(xt);
  }

  // Compute Dmat = A*A^{T}
  for ( int i = 0; i < ncon; i++ ){
    Ac[i]->mdot(Ac, ncon, &Dmat[i*ncon]);
  }

  if (ncon > 0){
    // Compute the factorization of Dmat
    int info;
    LAPACKdgetrf(&ncon, &ncon, Dmat, &ncon, dpiv, &info);

    // Solve the linear system
    if (!info){
      int one = 1;
      LAPACKdgetrs("N", &ncon, &one, Dmat, &ncon, dpiv,
                    z, &ncon, &info);

      // Keep the Lagrange multipliers if they are within a
      // reasonable range and they are positive.
      if (dense_inequality){
        for ( int i = 0; i < ncon; i++ ){
          if (ParOptRealPart(z[i]) < 0.01 ||
              ParOptRealPart(z[i]) > penalty_gamma[i]){
            z[i] = 1.0;
          }
        }
      }
      else {
        for ( int i = 0; i < ncon; i++ ){
          if (ParOptRealPart(z[i]) < -penalty_gamma[i] ||
              ParOptRealPart(z[i]) > penalty_gamma[i]){
            z[i] = 1.0;
          }
        }
      }
    }
    else {
      for ( int i = 0; i < ncon; i++ ){
        z[i] = 1.0;
      }
    }
  }
}

void ParOptInteriorPoint::initAffineStepMultipliers( ParOptNormType norm_type ){
  // Set the minimum allowable multiplier
  const double start_affine_multiplier_min =
    options->getFloatOption("start_affine_multiplier_min");
  const double max_bound_value =
    options->getFloatOption("max_bound_value");
  const int sequential_linear_method =
    options->getBoolOption("sequential_linear_method");
  const int use_qn_gmres_precon =
     options->getBoolOption("use_qn_gmres_precon");

  // Perform a preliminary estimate of the multipliers using the
  // least-squares method
  initLeastSquaresMultipliers();

  // Set the largrange multipliers with bounds outside the
  // limits to zero
  ParOptScalar *lbvals, *ubvals, *zlvals, *zuvals;
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Zero the multipliers for bounds that are out-of-range
  for ( int i = 0; i < nvars; i++ ){
    if (ParOptRealPart(lbvals[i]) <= -max_bound_value){
      zlvals[i] = 0.0;
    }
    if (ParOptRealPart(ubvals[i]) >= max_bound_value){
      zuvals[i] = 0.0;
    }
  }

  // Find the affine scaling step
  double max_prime, max_dual, max_infeas;
  computeKKTRes(0.0, norm_type, &max_prime, &max_dual, &max_infeas);

  // Set the flag which determines whether or not to use
  // the quasi-Newton method as a preconditioner
  int use_qn = 1;
  if (sequential_linear_method || !use_qn_gmres_precon){
    use_qn = 0;
  }

  // Set up the KKT diagonal system
  setUpKKTDiagSystem(s_qn, wtemp, use_qn);

  // Set up the full KKT system
  setUpKKTSystem(ztemp, s_qn, y_qn, wtemp, use_qn);

  // Solve for the KKT step
  computeKKTStep(ztemp, s_qn, y_qn, wtemp, use_qn);

  // Copy over the values
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      z[i] = max2(start_affine_multiplier_min,
                  fabs(ParOptRealPart(z[i] + pz[i])));
      s[i] = max2(start_affine_multiplier_min,
                  fabs(ParOptRealPart(s[i] + ps[i])));
      t[i] = max2(start_affine_multiplier_min,
                  fabs(ParOptRealPart(t[i] + pt[i])));
      zt[i] = max2(start_affine_multiplier_min,
                    fabs(ParOptRealPart(zt[i] + pzt[i])));
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      z[i] = max2(start_affine_multiplier_min,
                  fabs(ParOptRealPart(z[i] + pz[i])));
    }
  }

  // Copy the values
  if (nwcon > 0){
    ParOptScalar *zwvals, *pzwvals;
    zw->getArray(&zwvals);
    pzw->getArray(&pzwvals);
    for ( int i = 0; i < nwcon; i++ ){
      zwvals[i] = max2(start_affine_multiplier_min,
                        fabs(ParOptRealPart(zwvals[i] + pzwvals[i])));
    }

    if (sparse_inequality){
      ParOptScalar *swvals, *pswvals;
      sw->getArray(&swvals);
      psw->getArray(&pswvals);
      for ( int i = 0; i < nwcon; i++ ){
        swvals[i] = max2(start_affine_multiplier_min,
                          fabs(ParOptRealPart(swvals[i] + pswvals[i])));
      }
    }
  }

  if (use_lower){
    ParOptScalar *zlvals, *pzlvals;
    zl->getArray(&zlvals);
    pzl->getArray(&pzlvals);
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        zlvals[i] = max2(start_affine_multiplier_min,
                          fabs(ParOptRealPart(zlvals[i] + pzlvals[i])));
      }
    }
  }
  if (use_upper){
    ParOptScalar *zuvals, *pzuvals;
    zu->getArray(&zuvals);
    pzu->getArray(&pzuvals);
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        zuvals[i] = max2(start_affine_multiplier_min,
                          fabs(ParOptRealPart(zuvals[i] + pzuvals[i])));
      }
    }
  }

  // Set the initial barrier parameter
  barrier_param = ParOptRealPart(computeComp());
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
ParOptScalar ParOptInteriorPoint::evalObjBarrierDeriv(){
  const double rel_bound_barrier = options->getFloatOption("rel_bound_barrier");
  const double max_bound_value = options->getFloatOption("max_bound_value");

  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  ParOptScalar *xvals, *pxvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  px->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Compute the contribution from the bound variables.
  ParOptScalar pos_presult = 0.0, neg_presult = 0.0;

  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        if (ParOptRealPart(pxvals[i]) > 0.0){
          pos_presult += rel_bound_barrier*pxvals[i]/(xvals[i] - lbvals[i]);
        }
        else {
          neg_presult += rel_bound_barrier*pxvals[i]/(xvals[i] - lbvals[i]);
        }
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        if (ParOptRealPart(pxvals[i]) > 0.0){
          neg_presult -= rel_bound_barrier*pxvals[i]/(ubvals[i] - xvals[i]);
        }
        else {
          pos_presult -= rel_bound_barrier*pxvals[i]/(ubvals[i] - xvals[i]);
        }
      }
    }
  }

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0 && sparse_inequality){
    ParOptScalar *swvals, *pswvals;
    sw->getArray(&swvals);
    psw->getArray(&pswvals);

    for ( int i = 0; i < nwcon; i++ ){
      if (ParOptRealPart(pswvals[i]) > 0.0){
        pos_presult += pswvals[i]/swvals[i];
      }
      else {
        neg_presult += pswvals[i]/swvals[i];
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

  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      // Add the terms from the s-slack variables
      if (ParOptRealPart(ps[i]) > 0.0){
        pos_presult += ps[i]/s[i];
      }
      else {
        neg_presult += ps[i]/s[i];
      }

      // Add the terms from the t-slack variables
      if (ParOptRealPart(pt[i]) > 0.0){
        pos_presult += pt[i]/t[i];
      }
      else {
        neg_presult += pt[i]/t[i];
      }
    }
  }

  ParOptScalar pmerit = g->dot(px) - barrier_param*(pos_presult + neg_presult);

  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      pmerit += penalty_gamma[i]*pt[i];
    }
  }

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
int ParOptInteriorPoint::computeKKTGMRESStep( ParOptScalar *ztmp,
                                              ParOptVec *xtmp1, ParOptVec *xtmp2,
                                              ParOptVec *wtmp,
                                              double rtol, double atol,
                                              int use_qn ){
  // Set the output level
  const int output_level = options->getIntOption("output_level");

  // Check that the subspace has been allocated
  if (gmres_subspace_size <= 0){
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == opt_root){
      fprintf(stderr, "ParOpt error: gmres_subspace_size not set\n");
    }
    return 0;
  }

  // Initialize the data from the gmres object
  ParOptScalar *H = gmres_H;
  ParOptScalar *alpha = gmres_alpha;
  ParOptScalar *res = gmres_res;
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
  for ( int i = 0; i < ncon; i++ ){
    beta += rc[i]*rc[i];
  }
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      beta += rs[i]*rs[i];
      beta += rt[i]*rt[i];
      beta += rzt[i]*rzt[i];
    }
  }
  if (use_lower){
    beta += rzl->dot(rzl);
  }
  if (use_upper){
    beta += rzu->dot(rzu);
  }
  if (nwcon > 0){
    beta += rcw->dot(rcw);
    if (sparse_inequality){
      beta += rsw->dot(rsw);
    }
  }

  // Compute the norm of the initial vector
  ParOptScalar bnorm = sqrt(rx->dot(rx) + beta);

  // Broadcast the norm of the residuals and the beta parameter to
  // keep things consistent across processors
  ParOptScalar temp[2];
  temp[0] = bnorm;
  temp[1] = beta;
  MPI_Bcast(temp, 2, PAROPT_MPI_TYPE, opt_root, comm);

  bnorm = temp[0];
  beta = temp[1];

  // Compute the final value of the beta term
  beta *= 1.0/(bnorm*bnorm);

  // Compute the inverse of the l2 norm of the dense inequality constraint
  // infeasibility and store it for later computations.
  ParOptScalar cinfeas = 0.0, cscale = 0.0;
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      cinfeas += (c[i] - s[i] + t[i])*(c[i] - s[i] + t[i]);
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      cinfeas += c[i]*c[i];
    }
  }
  if (ParOptRealPart(cinfeas) != 0.0){
    cinfeas = sqrt(cinfeas);
    cscale = 1.0/cinfeas;
  }

  // Compute the inverse of the l2 norm of the sparse constraint
  // infeasibility and store it.
  ParOptScalar cwinfeas = 0.0, cwscale = 0.0;
  if (nwcon > 0){
    cwinfeas = sqrt(rcw->dot(rcw));
    if (ParOptRealPart(cwinfeas) != 0.0){
      cwscale = 1.0/cwinfeas;
    }
  }

  // Initialize the residual norm
  res[0] = bnorm;
  W[0]->copyValues(rx);
  W[0]->scale(1.0/res[0]);
  alpha[0] = 1.0;

  // Keep track of the actual number of iterations
  int niters = 0;

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root && outfp && output_level > 0){
    fprintf(outfp, "%5s %4s %4s %7s %7s %8s %8s gmres rtol: %7.1e\n",
            "gmres", "nhvc", "iter", "res", "rel", "fproj", "cproj", rtol);
    fprintf(outfp, "      %4d %4d %7.1e %7.1e\n",
            nhvec, 0, fabs(ParOptRealPart(res[0])), 1.0);
  }

  for ( int i = 0; i < gmres_subspace_size; i++ ){
    // Compute M^{-1}*[ W[i], alpha[i]*yc, ... ]
    // Get the size of the limited-memory BFGS subspace
    ParOptScalar b0;
    const ParOptScalar *d, *M;
    ParOptVec **Z;
    int size = 0;
    if (qn && use_qn){
      size = qn->getCompactMat(&b0, &d, &M, &Z);
    }

    // Solve the first part of the equation
    solveKKTDiagSystem(W[i], alpha[i]/bnorm,
                       rt, rc, rcw, rs, rsw, rzt, rzl, rzu,
                       px, pt, pz, ps, psw, xtmp2, wtmp);

    if (size > 0){
      // dz = Z^{T}*xt1
      px->mdot(Z, size, ztmp);

      // Compute dz <- Ce^{-1}*dz
      int one = 1, info = 0;
      LAPACKdgetrs("N", &size, &one,
                   Ce, &size, cpiv, ztmp, &size, &info);

      // Compute rx = Z^{T}*dz
      xtmp2->zeroEntries();
      for ( int k = 0; k < size; k++ ){
        xtmp2->axpy(ztmp[k], Z[k]);
      }

      // Solve the digaonal system again, this time simplifying the
      // result due to the structure of the right-hand-side.  Note
      // that this call uses W[i+1] as a temporary vector.
      solveKKTDiagSystem(xtmp2, xtmp1, ztmp, W[i+1], wtmp);

      // Add the final contributions
      px->axpy(-1.0, xtmp1);
    }

    // px now contains the current estimate of the step in the design
    // variables.
    fproj[i] = evalObjBarrierDeriv();

    // Compute the directional derivative of the l2 constraint infeasibility
    // along the direction px.
    aproj[i] = 0.0;
    if (dense_inequality){
      for ( int j = 0; j < ncon; j++ ){
        ParOptScalar cj_deriv = (Ac[j]->dot(px) - ps[j] + pt[j]);
        aproj[i] -= cscale*rc[j]*cj_deriv;
      }
    }
    else {
      for ( int j = 0; j < ncon; j++ ){
        aproj[i] -= cscale*rc[j]*Ac[j]->dot(px);
      }
    }

    // Add the contributions from the sparse constraints (if any are defined)
    awproj[i] = 0.0;
    if (nwcon > 0){
      // rcw = -(cw - sw)
      xtmp1->zeroEntries();
      prob->addSparseJacobianTranspose(1.0, x, rcw, xtmp1);
      awproj[i] = -cwscale*px->dot(xtmp1);

      if (sparse_inequality){
        awproj[i] += cwscale*rcw->dot(psw);
      }
    }

    // Compute the vector product with the exact Hessian
    prob->evalHvecProduct(x, z, zw, px, W[i+1]);
    nhvec++;

    // Add the term -B*W[i]
    if (qn && use_qn){
      qn->multAdd(-1.0, px, W[i+1]);
    }

    // Add the term from the diagonal
    W[i+1]->axpy(1.0, W[i]);

    // Set the value of the scalar
    alpha[i+1] = alpha[i];

    // Build the orthogonal factorization MGS
    int hptr = (i+1)*(i+2)/2 - 1;
    for ( int j = i; j >= 0; j-- ){
      H[j + hptr] = W[i+1]->dot(W[j]) + beta*alpha[i+1]*alpha[j];

      W[i+1]->axpy(-H[j + hptr], W[j]);
      alpha[i+1] -= H[j + hptr]*alpha[j];
    }

    // Compute the norm of the combined vector
    H[i+1 + hptr] = sqrt(W[i+1]->dot(W[i+1]) +
                         beta*alpha[i+1]*alpha[i+1]);

    // Normalize the combined vector
    W[i+1]->scale(1.0/H[i+1 + hptr]);
    alpha[i+1] *= 1.0/H[i+1 + hptr];

    // Apply the existing part of Q to the new components of the
    // Hessenberg matrix
    for ( int k = 0; k < i; k++ ){
      ParOptScalar h1 = H[k + hptr];
      ParOptScalar h2 = H[k+1 + hptr];
      H[k + hptr] = h1*Qcos[k] + h2*Qsin[k];
      H[k+1 + hptr] = -h1*Qsin[k] + h2*Qcos[k];
    }

    // Now, compute the rotation for the new column that was just added
    ParOptScalar h1 = H[i + hptr];
    ParOptScalar h2 = H[i+1 + hptr];
    ParOptScalar sq = sqrt(h1*h1 + h2*h2);

    Qcos[i] = h1/sq;
    Qsin[i] = h2/sq;
    H[i + hptr] = h1*Qcos[i] + h2*Qsin[i];
    H[i+1 + hptr] = -h1*Qsin[i] + h2*Qcos[i];

    // Update the residual
    h1 = res[i];
    res[i] = h1*Qcos[i];
    res[i+1] = -h1*Qsin[i];

    niters++;

    // Check the contribution to the projected derivative terms. First
    // evaluate the weights y[] for each
    for ( int j = niters-1; j >= 0; j-- ){
      y[j] = res[j];
      for ( int k = j+1; k < niters; k++ ){
        int hptr = (k+1)*(k+2)/2 - 1;
        y[j] = y[j] - H[j + hptr]*y[k];
      }

      int hptr = (j+1)*(j+2)/2 - 1;
      y[j] = y[j]/H[j + hptr];
    }

    // Compute the projection of the solution px on to the gradient
    // direction and the constraint Jacobian directions
    ParOptScalar fpr = 0.0, cpr = 0.0;
    for ( int j = 0; j < niters; j++ ){
      fpr += y[j]*fproj[j];
      cpr += y[j]*(aproj[j] + awproj[j]);
    }

    if (rank == opt_root && output_level > 0){
      fprintf(outfp, "      %4d %4d %7.1e %7.1e %8.1e %8.1e\n",
              nhvec, i+1, fabs(ParOptRealPart(res[i+1])),
              fabs(ParOptRealPart(res[i+1]/bnorm)),
              ParOptRealPart(fpr), ParOptRealPart(cpr));
      fflush(outfp);
    }

    // Check first that the direction is a candidate descent direction
    int constraint_descent = 0;
    if (ParOptRealPart(cpr) <= -0.01*ParOptRealPart(cinfeas + cwinfeas)){
      constraint_descent = 1;
    }
    if (ParOptRealPart(fpr) < 0.0 || constraint_descent){
      // Check for convergence
      if (fabs(ParOptRealPart(res[i+1])) < atol ||
          fabs(ParOptRealPart(res[i+1])) < rtol*ParOptRealPart(bnorm)){
        break;
      }
    }
  }

  // Now, compute the solution - the linear combination of the
  // Arnoldi vectors. H is now an upper triangular matrix.
  for ( int i = niters-1; i >= 0; i-- ){
    for ( int j = i+1; j < niters; j++ ){
      int hptr = (j+1)*(j+2)/2 - 1;
      res[i] = res[i] - H[i + hptr]*res[j];
    }

    int hptr = (i+1)*(i+2)/2 - 1;
    res[i] = res[i]/H[i + hptr];
  }

  // Compute the linear combination of the vectors
  // that will be the output
  W[0]->scale(res[0]);
  ParOptScalar gamma = res[0]*alpha[0];

  for ( int i = 1; i < niters; i++ ){
    W[0]->axpy(res[i], W[i]);
    gamma += res[i]*alpha[i];
  }

  // Normalize the gamma parameter
  gamma /= bnorm;

  // Scale the right-hand-side by gamma
  for ( int i = 0; i < ncon; i++ ){
    rc[i] *= gamma;
    rs[i] *= gamma;
    rt[i] *= gamma;
    rzt[i] *= gamma;
  }

  rzl->scale(gamma);
  rzu->scale(gamma);
  if (nwcon > 0){
    rcw->scale(gamma);
    rsw->scale(gamma);
  }

  // Apply M^{-1} to the result to obtain the final answer
  solveKKTDiagSystem(W[0], rt, rc, rcw, rs, rsw, rzt, rzl, rzu,
                     px, pt, pz, pzw, ps, psw, pzt, pzl, pzu,
                     xtmp1, wtmp);

  // Get the size of the limited-memory BFGS subspace
  ParOptScalar b0;
  const ParOptScalar *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (qn && use_qn){
    size = qn->getCompactMat(&b0, &d, &M, &Z);
  }

  if (size > 0){
    // dz = Z^{T}*px
    px->mdot(Z, size, ztmp);

    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one,
                 Ce, &size, cpiv, ztmp, &size, &info);

    // Compute rx = Z^{T}*dz
    xtmp1->zeroEntries();
    for ( int i = 0; i < size; i++ ){
      xtmp1->axpy(ztmp[i], Z[i]);
    }

    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(xtmp1, rx, rt, rc, rcw, rs, rsw, rzt, rzl, rzu,
                       xtmp2, wtmp);

    // Add the final contributions
    px->axpy(-1.0, rx);
    pzw->axpy(-1.0, rcw);
    psw->axpy(-1.0, rsw);
    pzl->axpy(-1.0, rzl);
    pzu->axpy(-1.0, rzu);

    // Add the terms from the dense constraints
    for ( int i = 0; i < ncon; i++ ){
      pz[i] -= rc[i];
      ps[i] -= rs[i];
      pt[i] -= rt[i];
      pzt[i] -= rzt[i];
    }
  }

  // Add the contributions from the objective and dense constraints
  ParOptScalar fpr = evalObjBarrierDeriv();
  ParOptScalar cpr = 0.0;
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      ParOptScalar deriv = (Ac[i]->dot(px) - ps[i] + pt[i]);
      cpr += cscale*(c[i] - s[i] + t[i])*deriv;
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      cpr += cscale*c[i]*Ac[i]->dot(px);
    }
  }

  // Add the contributions from the sparse constraints
  if (nwcon > 0){
    // Compute the residual rcw = (cw - sw)
    prob->evalSparseCon(x, rcw);
    if (sparse_inequality){
      rcw->axpy(-1.0, sw);
    }
    xtmp1->zeroEntries();
    prob->addSparseJacobianTranspose(1.0, x, rcw, xtmp1);
    cpr += cwscale*px->dot(xtmp1);

    // Finish depending on whether this is a sparse inequality or not
    if (sparse_inequality){
      cpr += cwscale*psw->dot(rcw);
    }
  }

  if (rank == opt_root && output_level > 0){
    fprintf(outfp, "      %9s %7s %7s %8.1e %8.1e\n",
            "final", " ", " ", ParOptRealPart(fpr),
            ParOptRealPart(cpr));
    fflush(outfp);
  }

  // Check if this should be considered a failure based on the
  // convergence criteria
  if (ParOptRealPart(fpr) < 0.0 ||
      ParOptRealPart(cpr) < -0.01*ParOptRealPart(cinfeas + cwinfeas)){
    return niters;
  }

  // We've failed.
  return -niters;
}

/*
  Check that the gradients match along a projected direction.
*/
void ParOptInteriorPoint::checkGradients( double dh ){
  const int use_hvec_product = options->getBoolOption("use_hvec_product");
  prob->checkGradients(dh, x, use_hvec_product);
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
void ParOptInteriorPoint::checkKKTStep( int iteration, int is_newton ){
  // Diagonal coefficient used for the quasi-Newton Hessian aprpoximation
  const double qn_sigma = options->getFloatOption("qn_sigma");
  const double max_bound_value = options->getFloatOption("max_bound_value");
  const int sequential_linear_method =
    options->getBoolOption("sequential_linear_method");
  const int use_diag_hessian = options->getBoolOption("use_diag_hessian");

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  ParOptScalar *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the values of the steps
  ParOptScalar *pxvals, *pzlvals, *pzuvals;
  px->getArray(&pxvals);
  pzl->getArray(&pzlvals);
  pzu->getArray(&pzuvals);

  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == opt_root){
    printf("\nResidual step check for iteration %d:\n", iteration);
  }

  // Check the first residual equation
  if (is_newton){
    prob->evalHvecProduct(x, z, zw, px, rx);
  }
  else if (use_diag_hessian){
    prob->evalHessianDiag(x, z, zw, hdiag);

    // Retrieve the components of px and hdiag
    ParOptScalar *rxvals, *hvals;
    rx->getArray(&rxvals);
    hdiag->getArray(&hvals);
    for ( int i = 0; i < nvars; i++ ){
      rxvals[i] = pxvals[i]*hvals[i];
    }
  }
  else {
    if (qn && !sequential_linear_method){
      qn->mult(px, rx);
      rx->axpy(qn_sigma, px);
    }
    else {
      rx->zeroEntries();
    }
  }
  for ( int i = 0; i < ncon; i++ ){
    rx->axpy(-pz[i] - z[i], Ac[i]);
  }
  if (use_lower){
    rx->axpy(-1.0, pzl);
    rx->axpy(-1.0, zl);
  }
  if (use_upper){
    rx->axpy(1.0, pzu);
    rx->axpy(1.0, zu);
  }
  rx->axpy(1.0, g);

  // Add the contributions from the constraint
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(-1.0, x, zw, rx);
    prob->addSparseJacobianTranspose(-1.0, x, pzw, rx);
  }
  double max_val = rx->maxabs();

  if (rank == opt_root){
    printf("max |(H + sigma*I)*px - Ac^{T}*pz - Aw^{T}*pzw - pzl + pzu + "
           "(g - Ac^{T}*z - Aw^{T}*zw - zl + zu)|: %10.4e\n", max_val);
  }

  // Compute the residuals from the weighting constraints
  if (nwcon > 0){
    prob->evalSparseCon(x, rcw);
    prob->addSparseJacobian(1.0, x, px, rcw);
    if (sparse_inequality){
      rcw->axpy(-1.0, sw);
      rcw->axpy(-1.0, psw);
    }

    max_val = rcw->maxabs();
    if (rank == opt_root){
      printf("max |cw(x) - sw + Aw*pw - psw|: %10.4e\n", max_val);
    }
  }

  // Find the maximum value of the residual equations
  // for the constraints
  max_val = 0.0;
  px->mdot(Ac, ncon, rc);
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar val = rc[i] + c[i];
    if (dense_inequality){
      val = rc[i] - ps[i] + pt[i] + (c[i] - s[i] + t[i]);
    }
    if (fabs(ParOptRealPart(val)) > max_val){
      max_val = fabs(ParOptRealPart(val));
    }
  }
  if (rank == opt_root){
    printf("max |A*px - ps + pt + (c - s + t)|: %10.4e\n", max_val);
  }

  // Find the maximum value of the residual equations for
  // the dual slack variables
  if (dense_inequality){
    max_val = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      ParOptScalar val = penalty_gamma[i] - z[i] - zt[i] - pz[i] - pzt[i];
      if (fabs(ParOptRealPart(val)) > max_val){
        max_val = fabs(ParOptRealPart(val));
      }
    }
    if (rank == opt_root){
      printf("max |gamma - z - zt - pz - pzt|: %10.4e\n", max_val);
    }

    max_val = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      ParOptScalar val = t[i]*pzt[i] + zt[i]*pt[i] +
        (t[i]*zt[i] - barrier_param);
      if (fabs(ParOptRealPart(val)) > max_val){
        max_val = fabs(ParOptRealPart(val));
      }
    }
    if (rank == opt_root){
      printf("max |T*pzt + Zt*pt + (T*zt - mu)|: %10.4e\n", max_val);
    }

    max_val = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      ParOptScalar val = z[i]*ps[i] + s[i]*pz[i] + (z[i]*s[i] - barrier_param);
      if (fabs(ParOptRealPart(val)) > max_val){
        max_val = fabs(ParOptRealPart(val));
      }
    }
    if (rank == opt_root){
      printf("max |Z*ps + S*pz + (z*s - mu)|: %10.4e\n", max_val);
    }
  }

  // Find the maximum of the residual equations for the
  // lower-bound dual variables
  max_val = 0.0;
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(lbvals[i]) > -max_bound_value){
        ParOptScalar val =
          (zlvals[i]*pxvals[i] + (xvals[i] - lbvals[i])*pzlvals[i] +
           (zlvals[i]*(xvals[i] - lbvals[i]) - barrier_param));
        if (fabs(ParOptRealPart(val)) > max_val){
          max_val = fabs(ParOptRealPart(val));
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);

  if (rank == opt_root && use_lower){
    printf("max |Zl*px + (X - LB)*pzl + (Zl*(x - lb) - mu)|: %10.4e\n",
           max_val);
  }

  // Find the maximum value of the residual equations for the
  // upper-bound dual variables
  max_val = 0.0;
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (ParOptRealPart(ubvals[i]) < max_bound_value){
        ParOptScalar val =
          (-zuvals[i]*pxvals[i] + (ubvals[i] - xvals[i])*pzuvals[i] +
           (zuvals[i]*(ubvals[i] - xvals[i]) - barrier_param));
        if (fabs(ParOptRealPart(val)) > max_val){
          max_val = fabs(ParOptRealPart(val));
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);

  if (rank == opt_root && use_upper){
    printf("max |-Zu*px + (UB - X)*pzu + (Zu*(ub - x) - mu)|: %10.4e\n",
           max_val);
  }
}
