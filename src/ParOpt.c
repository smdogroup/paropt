#include <math.h>
#include <string.h>
#include "ComplexStep.h"
#include "ParOpt.h"
#include "ParOptBlasLapack.h"

/*
  Copyright (c) 2014-2015 Graeme Kennedy. All rights reserved
*/

/*
  The following are the help-strings for each of the parameters
  in the file. These provide some description of the purpose of
  each parameter and how you should set it. 
*/

static const int NUM_PAROPT_PARAMETERS = 27;
static const char *paropt_parameter_help[][2] = {
  {"max_qn_size", 
   "Integer: The maximum dimension of the quasi-Newton approximation"},
  
  {"max_major_iters",
   "Integer: The maximum number of major iterations before quiting"},
  
  {"init_starting_point",
   "Boolean: Initialize the Lagrange multiplier estimates"},
  
  {"barrier_param",
   "Float: The initial value of the barrier parameter"},
  
  {"abs_res_tol",
   "Float: Absolute stopping criterion"},

  {"rel_func_tol",
   "Float: Relative function value stopping criterion"},
  
  {"use_line_search",
   "Boolean: Perform or skip the line search"},
  
  {"use_backtracking_alpha",
   "Boolean: Perform a back-tracking line search"},
  
  {"max_line_iters",
   "Integer: Maximum number of line search iterations"},
  
  {"armijio_constant",
   "Float: The Armijio constant for the line search"},
  
  {"monotone_barrier_fraction",
   "Float: Factor applied to the barrier update < 1"},
  
  {"monotone_barrier_power",
   "Float: Exponent for barrier parameter update > 1"},
  
  {"min_fraction_to_boundary",
   "Float: Minimum fraction to the boundary rule < 1"},
  
  {"major_iter_step_check",
   "Integer: Perform a check of the computed KKT step at this major iteration"},
  
  {"write_output_frequency", 
   "Integer: Write out the solution file and checkpoint file at this frequency"},

  {"gradient_check_frequency",
   "Integer: Print to screen the output of the gradient check at this frequency"},

  {"sequential_linear_method", 
   "Boolean: Discard the quasi-Newton approximation (but not \
necessarily the exact Hessian)"},
  
  {"hessian_reset_freq", 
   "Integer: Do a hard reset of the Hessian at this specified major \
iteration frequency"},

  {"qn_sigma",
   "Float: Scalar added to the diagonal of the quasi-Newton approximation > 0"},

  {"use_hvec_product", 
   "Boolean: Use or do not use Hessian-vector products"},
  
  {"use_qn_gmres_precon", 
   "Boolean: Use or do not use the quasi-Newton method as a preconditioner"},
  
  {"nk_switch_tol", 
   "Float: Switch to the Newton-Krylov method at this residual tolerance"},
  
  {"eisenstat_walker_alpha", 
   "Float: Exponent in the Eisenstat-Walker INK forcing equation"},
  
  {"eisenstat_walker_gamma", 
   "Float: Multiplier in the Eisenstat-Walker INK forcing equation"},
  
  {"gmres_subspace_size", 
   "Integer: The subspace size for GMRES"},
  
  {"max_gmres_rtol", 
   "Float: The maximum relative tolerance used for GMRES, above this \
the quasi-Newton approximation is used"},
  
  {"gmres_atol", 
   "Float: The absolute GMRES tolerance (almost never relevant)"}};

/*
  The Parallel Optimizer constructor

  This function allocates and initializes the data that is required
  for parallel optimization. This includes initialization of the
  variables, allocation of the matrices and the BFGS approximate
  Hessian. This code also sets the default parameters for
  optimization. These parameters can be modified through member
  functions.

  input:
  prob:        the optimization problem
  max_qn_size: the number of steps to store in memory
  qn_type:     the type of quasi-Newton method to use
*/
ParOpt::ParOpt( ParOptProblem *_prob, int max_qn_subspace,
                enum QuasiNewtonType qn_type,
                double _max_bound_val ){
  prob = _prob;

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

  // Allocate the quasi-Newton approximation
  if (qn_type == BFGS){
    qn = new LBFGS(prob, max_qn_subspace);
  }
  else {
    qn = new LSR1(prob, max_qn_subspace);
  }

  // Set the default maximum variable bound
  max_bound_val = _max_bound_val;

  // Set the values of the variables/bounds
  x = prob->createDesignVec();
  lb = prob->createDesignVec();
  ub = prob->createDesignVec();
  
  // Allocate storage space for the variables etc.
  zl = prob->createDesignVec();
  zu = prob->createDesignVec();

  // Allocate space for the sparse constraints
  zw = prob->createConstraintVec();
  sw = prob->createConstraintVec();

  // Set the initial values of the Lagrange multipliers
  z = new ParOptScalar[ ncon ];
  s = new ParOptScalar[ ncon ];

  // Allocate space for the steps
  px = prob->createDesignVec();
  pzl = prob->createDesignVec();
  pzu = prob->createDesignVec();
  pz = new ParOptScalar[ ncon ];
  ps = new ParOptScalar[ ncon ];
  pzw = prob->createConstraintVec();
  psw = prob->createConstraintVec();

  // Allocate space for the residuals
  rx = prob->createDesignVec();
  rzl = prob->createDesignVec();
  rzu = prob->createDesignVec();
  rc = new ParOptScalar[ ncon ];
  rs = new ParOptScalar[ ncon ];
  rcw = prob->createConstraintVec();
  rsw = prob->createConstraintVec();

  // Allocate space for the Quasi-Newton updates
  y_qn = prob->createDesignVec();
  s_qn = prob->createDesignVec();

  // Allocate vectors for the weighting constraints
  wtemp = prob->createConstraintVec();

  // Allocate space for the block-diagonal matrix
  Cw = new ParOptScalar[ nwcon*(nwblock+1)/2 ];

  // Allocate space for off-diagonal entries
  Ew = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ew[i] = prob->createConstraintVec();
  }

  // Allocate storage for bfgs/constraint sized things
  int zsize = 2*max_qn_subspace;
  if (ncon > zsize){
    ncon = zsize;
  }
  ztemp = new ParOptScalar[ zsize ];

  // Allocate space for the Dmatrix
  Dmat = new ParOptScalar[ ncon*ncon ];
  dpiv = new int[ ncon ];

  // Allocate space for the Ce matrix
  Ce = new ParOptScalar[ 4*max_qn_subspace*max_qn_subspace ];
  cpiv = new int[ 2*max_qn_subspace ];

  // Allocate space for the diagonal matrix components
  Cvec = prob->createDesignVec();

  // Set the value of the objective
  fobj = 0.0;
  
  // Set the constraints to zero
  c = new ParOptScalar[ ncon ];
  memset(c, 0, ncon*sizeof(ParOptScalar));
  
  // Set the objective and constraint gradients 
  g = prob->createDesignVec();
  Ac = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ac[i] = prob->createDesignVec();
  }

  // Initialize the design variables and bounds
  int init_multipliers = 1;
  initAndCheckDesignAndBounds(init_multipliers);

  // Zero the number of evals
  neval = ngeval = nhvec = 0;

  // Set the flag to indicate that this is not the final barrier
  // problem
  final_barrier_problem = 0;

  // Initialize the parameters with default values
  max_major_iters = 1000;
  init_starting_point = 1;
  barrier_param = 0.1;
  abs_res_tol = 1e-5;
  rel_func_tol = 0.0;
  use_line_search = 1;
  use_backtracking_alpha = 0;
  max_line_iters = 10;
  rho_penalty_search = 0.0;
  penalty_descent_fraction = 0.3;
  armijio_constant = 1e-3;
  monotone_barrier_fraction = 0.25;
  monotone_barrier_power = 1.1;
  min_fraction_to_boundary = 0.95;
  write_output_frequency = 10;
  gradient_check_frequency = -1;
  gradient_check_step = 1e-6;
  major_iter_step_check = -1;
  sequential_linear_method = 0;
  hessian_reset_freq = 100000000;
  qn_sigma = 0.0;
  merit_func_check_epsilon = 1e-6;

  // Initialize the Hessian-vector product information
  use_hvec_product = 0;
  use_qn_gmres_precon = 1;
  nk_switch_tol = 1e-3;
  eisenstat_walker_alpha = 1.5;
  eisenstat_walker_gamma = 1.0;
  max_gmres_rtol = 0.1;
  gmres_atol = 1e-30;

  // By default, set the file pointer to stdout
  outfp = stdout;

  // Set the default information about GMRES
  gmres_subspace_size = 0;
  gmres_H = NULL;
  gmres_alpha = NULL;
  gmres_res = NULL;
  gmres_Q = NULL;
  gmres_W = NULL;
}

/*
  Free the data allocated during the creation of the object
*/
ParOpt::~ParOpt(){
  delete qn;

  // Delete the variables and bounds
  delete x;
  delete lb;
  delete ub;
  delete zl;
  delete zu;
  delete [] z;
  delete [] s;
  delete zw;
  delete sw;

  // Delete the steps
  delete px;
  delete pzl;
  delete pzu;
  delete [] pz;
  delete [] ps;
  delete pzw;
  delete psw;

  // Delete the residuals
  delete rx;
  delete rzl;
  delete rzu;
  delete [] rc;
  delete [] rs;
  delete rcw;
  delete rsw;

  // Delete the quasi-Newton updates
  delete y_qn;
  delete s_qn;

  // Delete the temp data
  delete wtemp;
  delete [] ztemp;
 
  // Delete the matrix
  delete [] Cw;
  
  for ( int i = 0; i < ncon; i++ ){
    delete Ew[i];
  }
  delete [] Ew;

  // Delete the various matrices
  delete [] Dmat;
  delete [] dpiv;
  delete [] Ce;
  delete [] cpiv;

  // Delete the diagonal matrix
  delete Cvec;

  // Free the variable ranges
  delete [] var_range;
  delete [] wcon_range;

  // Delete the constraint/gradient information
  delete [] c;
  delete g;
  for ( int i = 0; i < ncon; i++ ){
    delete Ac[i];
  }
  delete [] Ac;

  // Delete the GMRES information if any
  if (gmres_subspace_size > 0){
    delete [] gmres_H;
    delete [] gmres_alpha;
    delete [] gmres_res;
    delete [] gmres_Q;

    // Delete the subspace
    for ( int i = 0; i < gmres_subspace_size; i++ ){
      delete gmres_W[i];
    }
    delete [] gmres_W;
  }

  // Close the output file if it's not stdout
  if (outfp && outfp != stdout){
    fclose(outfp);
  }
}

/*
  Reset the problem instance
*/
void ParOpt::resetProblemInstance( ParOptProblem *problem ){
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

/*
  Retrieve the problem sizes from the underlying problem class
*/
void ParOpt::getProblemSizes( int *_nvars, int *_ncon, 
                              int *_nwcon, int *_nwblock ){
  prob->getProblemSizes(_nvars, _ncon, _nwcon, _nwblock);
}

/*
  Get the multiplier variables used internally in ParOpt so that the
  user can set the initial guess.

  Note that this call automatically call setInitStartingPoint(0) so
  that ParOpt does not attempt to guess good initial values for the
  multipliers (since the user probably knows something about the
  problem if they are calling this function...)
*/
void ParOpt::getInitMultipliers( ParOptScalar **_z,
                                 ParOptVec **_zw,
                                 ParOptVec **_zl,
                                 ParOptVec **_zu ){
  init_starting_point = 0;
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

/*
  Retrieve the optimal values of the design variables and multipliers.
 
  This call can be made during the course of an optimization, but
  changing the values in x/zw/zl/zu is not recommended and the
  behavior after doing so is not defined. Note that inputs that are
  NULL are not assigned. If no output is available, for instance if
  use_lower == False, then NULL is assigned to the output.
*/
void ParOpt::getOptimizedPoint( ParOptVec **_x, 
                                const ParOptScalar **_z, 
                                ParOptVec **_zw,
                                ParOptVec **_zl, 
                                ParOptVec **_zu ){
  if (_x){ *_x = x; }
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

/*
  Write out all of the options that have been set to a output
  stream. This is usually the output summary file.
*/
void ParOpt::printOptionSummary( FILE *fp ){
  // Print out all the parameter values to the screen
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (fp && rank == opt_root){
    int qn_size = 0;
    if (!sequential_linear_method){
      qn_size = qn->getMaxLimitedMemorySize();
    }
    
    fprintf(fp, "ParOpt: Parameter values\n");
    fprintf(fp, "%-30s %15d\n", "total variables", nvars_total);
    fprintf(fp, "%-30s %15d\n", "constraints", ncon);
    fprintf(fp, "%-30s %15d\n", "max_qn_size", qn_size);
    fprintf(fp, "%-30s %15d\n", "max_major_iters", max_major_iters);
    fprintf(fp, "%-30s %15d\n", "init_starting_point", 
            init_starting_point);
    fprintf(fp, "%-30s %15g\n", "barrier_param", barrier_param);
    fprintf(fp, "%-30s %15g\n", "abs_res_tol", abs_res_tol);
    fprintf(fp, "%-30s %15g\n", "rel_func_tol", rel_func_tol);
    fprintf(fp, "%-30s %15d\n", "use_line_search", use_line_search);
    fprintf(fp, "%-30s %15d\n", "use_backtracking_alpha", 
            use_backtracking_alpha);
    fprintf(fp, "%-30s %15d\n", "max_line_iters", max_line_iters);
    fprintf(fp, "%-30s %15g\n", "penalty_descent_fraction", 
            penalty_descent_fraction);
    fprintf(fp, "%-30s %15g\n", "armijio_constant", armijio_constant);
    fprintf(fp, "%-30s %15g\n", "monotone_barrier_fraction", 
            monotone_barrier_fraction);
    fprintf(fp, "%-30s %15g\n", "monotone_barrier_power", 
            monotone_barrier_power);
    fprintf(fp, "%-30s %15g\n", "min_fraction_to_boundary", 
            min_fraction_to_boundary);
    fprintf(fp, "%-30s %15d\n", "major_iter_step_check", 
            major_iter_step_check);
    fprintf(fp, "%-30s %15d\n", "write_output_frequency", 
            write_output_frequency);
    fprintf(fp, "%-30s %15d\n", "gradient_check_frequency", 
            gradient_check_frequency);
    fprintf(fp, "%-30s %15g\n", "gradient_check_step", 
            gradient_check_step);
    fprintf(fp, "%-30s %15d\n", "sequential_linear_method",
            sequential_linear_method);
    fprintf(fp, "%-30s %15d\n", "hessian_reset_freq",
            hessian_reset_freq);
    fprintf(fp, "%-30s %15g\n", "qn_sigma", qn_sigma);
    fprintf(fp, "%-30s %15d\n", "use_hvec_product",
            use_hvec_product);
    fprintf(fp, "%-30s %15d\n", "use_qn_gmres_precon",
            use_qn_gmres_precon);
    fprintf(fp, "%-30s %15g\n", "nk_switch_tol", nk_switch_tol);
    fprintf(fp, "%-30s %15g\n", "eisenstat_walker_alpha", 
            eisenstat_walker_alpha);
    fprintf(fp, "%-30s %15g\n", "eisenstat_walker_gamma", 
            eisenstat_walker_gamma);
    fprintf(fp, "%-30s %15d\n", "gmres_subspace_size",
            gmres_subspace_size);
    fprintf(fp, "%-30s %15g\n", "max_gmres_rtol", max_gmres_rtol);
    fprintf(fp, "%-30s %15g\n", "gmres_atol", gmres_atol);
  }
}

/*
  Write out all of the design variables, Lagrange multipliers and
  slack variables to a binary file.
*/
int ParOpt::writeSolutionFile( const char *filename ){
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

/*
  Read in the design variables, lagrange multipliers and slack
  variables from a binary file
*/
int ParOpt::readSolutionFile( const char *filename ){
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

/*
  Set the maximum variable bound. Bounds that exceed this value will
  be ignored within the optimization problem.
*/
void ParOpt::setMaxAbsVariableBound( double max_bound ){
  max_bound_val = max_bound;

  // Set the largrange multipliers with bounds outside the
  // limits to zero
  ParOptScalar *lbvals, *ubvals, *zlvals, *zuvals;
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);
  
  for ( int i = 0; i < nvars; i++ ){
    if (RealPart(lbvals[i]) <= -max_bound_val){
      zlvals[i] = 0.0;
    }
    if (RealPart(ubvals[i]) >= max_bound_val){
      zuvals[i] = 0.0;
    }
  }
}

/*
  Set optimizer parameters
*/
void ParOpt::setInitStartingPoint( int init ){
  init_starting_point = init;
}

void ParOpt::setMaxMajorIterations( int iters ){
  if (iters >= 1){ 
    max_major_iters = iters; 
  }
}

/*
  Set the absolute KKT tolerance
*/
void ParOpt::setAbsOptimalityTol( double tol ){
  if (tol < 1e-2 && tol >= 0.0){
    abs_res_tol = tol;
  }
}

/*
  Set the relative function tolerance
*/
void ParOpt::setRelFunctionTol( double tol ){
  if (tol < 1e-2 && tol >= 0.0){
    rel_func_tol = tol;
  }
}

/*
  Set the initial barrier parameter
*/
void ParOpt::setInitBarrierParameter( double mu ){
  if (mu > 0.0){ 
    barrier_param = mu;
  }
}

/*
  Retrieve the barrier parameter
*/
double ParOpt::getBarrierParameter(){
  return barrier_param;
}

/*
  Get the average of the complementarity products at the current
  point.  Note that this call is collective on all processors.
*/
ParOptScalar ParOpt::getComplementarity(){
  return computeComp();
}

/*
  Set fraction for the barrier update
*/
void ParOpt::setBarrierFraction( double frac ){
  if (frac > 0.0 && frac < 1.0){
    monotone_barrier_fraction = frac;
  }
}

/*
  Set the power for the barrier update
*/
void ParOpt::setBarrierPower( double power ){
  if (power >= 1.0 && power < 10.0){
    monotone_barrier_power = power;
  }
}

/*
  Set the frequency with which the Hessian is updated
*/
void ParOpt::setHessianResetFreq( int freq ){
  if (freq > 0){
    hessian_reset_freq = freq;
  }
}

/*
  Set the diagonal entry to add to the quasi-Newton Hessian approximation
*/
void ParOpt::setQNDiagonalFactor( double sigma ){
  if (sigma >= 0.0){
    qn_sigma = sigma;
  }
}

/*
  Set parameters associated with the line search
*/
void ParOpt::setUseLineSearch( int truth ){
  use_line_search = truth;
}

void ParOpt::setMaxLineSearchIters( int iters ){
  if (iters > 0){ 
    max_line_iters = iters;
  }
}

void ParOpt::setBacktrackingLineSearch( int truth ){
  use_backtracking_alpha = truth;
}

void ParOpt::setArmijioParam( double c1 ){
  if (c1 >= 0){
    armijio_constant = c1;
  }
}

void ParOpt::setPenaltyDescentFraction( double frac ){
  if (frac > 0.0){
    penalty_descent_fraction = frac;
  }
}

void ParOpt::setBFGSUpdateType( LBFGS::BFGSUpdateType update ){
  LBFGS *lbfgs = dynamic_cast<LBFGS*>(qn);
  if (lbfgs){
    lbfgs->setBFGSUpdateType(update);
  }  
}

void ParOpt::setSequentialLinearMethod( int truth ){
  sequential_linear_method = truth;
}

/*
  Set other parameters
*/
void ParOpt::setOutputFrequency( int freq ){
  if (freq >= 1){
    write_output_frequency = freq;
  }
}

void ParOpt::setMajorIterStepCheck( int step ){
  major_iter_step_check = step;
}

void ParOpt::setGradientCheckFrequency( int freq, double step_size ){
  gradient_check_frequency = freq;
  gradient_check_step = step_size;
}

/*
  Set the flag for whether to use the Hessian-vector products or not
*/
void ParOpt::setUseHvecProduct( int truth ){
  use_hvec_product = truth;
}

/*
  Use the limited-memory BFGS update as a preconditioner
*/
void ParOpt::setUseQNGMRESPreCon( int truth ){
  use_qn_gmres_precon = truth;
}

/*
  Set information about when to use the Newton-Krylov method
*/
void ParOpt::setNKSwitchTolerance( double tol ){
  nk_switch_tol = tol;
}

void ParOpt::setGMRESTolerances( double rtol, double atol ){
  max_gmres_rtol = rtol;
  gmres_atol = atol;
}

/*
  Set the parameters for choosing the forcing term in an inexact
  Newton method. These parameters are used to compute the forcing term
  as follows:

  eta = gamma*(||r_{k}||/||r_{k-1}||)^{alpha}
*/
void ParOpt::setEisenstatWalkerParameters( double gamma, double alpha ){
  if (gamma > 0.0 && gamma <= 1.0){
    eisenstat_walker_gamma = gamma;
  }
  if (alpha >= 0.0 && gamma <= 2.0){
    eisenstat_walker_alpha = alpha;
  }
}

/*
  Reset the Quasi-Newton Hessian approximation if it is used
*/
void ParOpt::resetQuasiNewtonHessian(){
  if (qn){
    qn->reset();
  }
}

/*
  Reset the design variables and bounds
*/
void ParOpt::resetDesignAndBounds(){
  prob->getVarsAndBounds(x, lb, ub);
}

/*
  Set the size of the GMRES subspace and allocate the vectors
  required. Note that the old subspace information is deleted before
  the new subspace data is allocated.
*/
void ParOpt::setGMRESSubspaceSize( int m ){
  if (gmres_H){
    delete [] gmres_H;
    delete [] gmres_alpha;
    delete [] gmres_res;
    delete [] gmres_Q;

    for ( int i = 0; i < m; i++ ){
      delete gmres_W[i];
    }
    delete [] gmres_W;
  }

  if (m > 0){
    gmres_subspace_size = m;
    
    gmres_H = new ParOptScalar[ (m+1)*(m+2)/2 ];
    gmres_alpha = new ParOptScalar[ m+1 ];
    gmres_res = new ParOptScalar[ m+1 ];
    gmres_Q = new ParOptScalar[ 2*m ];
    
    gmres_W = new ParOptVec*[ m+1 ];
    for ( int i = 0; i < m+1; i++ ){
      gmres_W[i] = prob->createDesignVec();
    }
  }
  else {
    gmres_subspace_size = 0;
  }
}

/*
  Set the optimization history file name to use.

  The file is only opened on the root processor
*/
void ParOpt::setOutputFile( const char *filename ){
  if (outfp && outfp != stdout){
    fclose(outfp);
  }
  outfp = NULL;

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (filename && rank == opt_root){
    outfp = fopen(filename, "w");
  
    if (outfp){
      fprintf(outfp, "ParOpt: Parameter summary\n");
      for ( int i = 0; i < NUM_PAROPT_PARAMETERS; i++ ){
        fprintf(outfp, "%s\n%s\n\n",
                paropt_parameter_help[i][0],
                paropt_parameter_help[i][1]);
      }
    }
  }
}

/*
  Compute the residual of the KKT system. This code utilizes the data
  stored internally in the ParOpt optimizer.

  This code computes the following terms:

  rx  = -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu) 
  rc  = -(c(x) - s)
  rcw = -(cw(x) - sw)
  rz  = -(S*z - mu*e) 
  rzu = -((x - xl)*zl - mu*e)
  rzl = -((ub - x)*zu - mu*e)
*/
void ParOpt::computeKKTRes( double *max_prime,
                            double *max_dual, 
                            double *max_infeas ){
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
  *max_prime = rx->maxabs();

  // Compute the residuals from the second KKT system:
  *max_infeas = rcw->maxabs();

  // Evaluate the residuals differently depending on whether
  // we're using a dense equality or inequality constraint
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      rc[i] = -(c[i] - s[i]);
      rs[i] = -(s[i]*z[i] - barrier_param);

      if (fabs(RealPart(rc[i])) > *max_infeas){
        *max_infeas = fabs(RealPart(rc[i]));
      }
      if (fabs(RealPart(rs[i])) > *max_dual){
        *max_dual = fabs(RealPart(rs[i]));
      }
    }
  }
  else {
    for ( int i = 0; i < ncon; i++ ){
      rc[i] = -c[i];

      if (fabs(RealPart(rc[i])) > *max_infeas){
        *max_infeas = fabs(RealPart(rc[i]));
      }
    }
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
      if (RealPart(lbvals[i]) > -max_bound_val){
        rzlvals[i] = -((xvals[i] - lbvals[i])*zlvals[i] - barrier_param);
      }
      else {
        rzlvals[i] = 0.0;
      }
    }
  
    double dual_zl = rzl->maxabs();
    if (dual_zl > *max_dual){
      *max_dual = dual_zl;
    }
  }
  if (use_upper){
    // Compute the residuals for the upper bounds
    ParOptScalar *rzuvals;
    rzu->getArray(&rzuvals);

    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
        rzuvals[i] = -((ubvals[i] - xvals[i])*zuvals[i] - barrier_param);
      }
      else {
        rzuvals[i] = 0.0;
      }
    }

    double dual_zu = rzu->maxabs();
    if (RealPart(dual_zu) > RealPart(*max_dual)){
      *max_dual = dual_zu;
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
      rswvals[i] = -(swvals[i]*zwvals[i] - barrier_param);
    }
    
    double dual_zw = rsw->maxabs();
    if (RealPart(dual_zw) > RealPart(*max_dual)){
      *max_dual = dual_zw;
    }
  }
}

/*
  Factor the matrix after assembly
*/
int ParOpt::factorCw(){
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
int ParOpt::applyCwFactor( ParOptVec *vec ){
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

  D = Z^{-1}*S + A*C^{-1}*A^{T} - Dw

  which is required to compute the solution of the KKT step.
*/
void ParOpt::setUpKKTDiagSystem( ParOptVec *xt,
                                 ParOptVec *wt, 
                                 int use_qn ){
  // Retrive the diagonal entry for the BFGS update
  ParOptScalar b0 = 0.0;
  if (use_qn){
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
    ParOptScalar diag_no_bound = 1.0/(b0 + qn_sigma);
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(lbvals[i]) > -max_bound_val && 
          RealPart(ubvals[i]) < max_bound_val){
        cvals[i] = 1.0/(b0 + qn_sigma +
                        zlvals[i]/(xvals[i] - lbvals[i]) + 
                        zuvals[i]/(ubvals[i] - xvals[i]));
      }
      else if (RealPart(lbvals[i]) > -max_bound_val){
        cvals[i] = 1.0/(b0 + qn_sigma + zlvals[i]/(xvals[i] - lbvals[i]));
      }
      else if (RealPart(ubvals[i]) < max_bound_val){
        cvals[i] = 1.0/(b0 + qn_sigma + zuvals[i]/(ubvals[i] - xvals[i]));
      }
      else {
        cvals[i] = diag_no_bound;
      }
    }
  }
  else if (use_lower){
    ParOptScalar diag_no_bound = 1.0/(b0 + qn_sigma);
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(lbvals[i]) > -max_bound_val){
        cvals[i] = 1.0/(b0 + qn_sigma + zlvals[i]/(xvals[i] - lbvals[i]));
      }
      else {
        cvals[i] = diag_no_bound;
      }
    }
  }
  else if (use_upper){
    ParOptScalar diag_no_bound = 1.0/(b0 + qn_sigma);
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
        cvals[i] = 1.0/(b0 + qn_sigma + zuvals[i]/(ubvals[i] - xvals[i]));
      }
      else {
        cvals[i] = diag_no_bound;
      }
    }
  }
  else {
    ParOptScalar diag_no_bound = 1.0/(b0 + qn_sigma);
    for ( int i = 0; i < nvars; i++ ){
      cvals[i] = diag_no_bound;
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
      xt->getArray(&xvals);
      Ac[k]->getArray(&avals);

      for ( int i = 0; i < nvars; i++ ){
        xvals[i] = cvals[i]*avals[i];
      }

      Ew[k]->zeroEntries();
      prob->addSparseJacobian(1.0, x, xt, Ew[k]);
    }
  }

  // Set the value of the D matrix
  memset(Dmat, 0, ncon*ncon*sizeof(ParOptScalar));

  if (nwcon > 0){
    // Add the term Dw = - Ew^{T}*Cw^{-1}*Ew to the Dmat matrix first
    // by computing the product with Cw^{-1}
    for ( int j = 0; j < ncon; j++ ){
      // Apply Cw^{-1}*Ew[j] -> wt
      wt->copyValues(Ew[j]);
      applyCwFactor(wt);

      for ( int i = j; i < ncon; i++ ){
        // Get the vectors required
        ParOptScalar *wvals, *ewivals;
        Ew[i]->getArray(&ewivals);
        wt->getArray(&wvals);

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
          Dmat[i*(ncon + 1)] += s[i]/z[i];
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
  Hessian is replaced with only the diagonal terms.  The system of
  equations consists of the following terms:
  
  B0*yx - A^{T}*yz - Aw^{T}*yzw - yzl + yzu = bx
  A*yx - ys = bc
  Aw*yx - ysw = bw

  With the additional equations:

  ys = Z^{-1}*bs - Z^{-1}*S*yz
  yzl = (X - Xl)^{-1}*(bzl - Zl*yx)
  yzu = (Xu - X)^{-1}*(bzu + Zu*yx)

  Substitution of these three equations yields the following system of
  equations:

  ((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))*yx - A^{T}*yz - Aw^{T}*yzw
  = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu

  which we rewrite as the equation:

  C*yx - A^{T}*yz - Aw^{T}*yzw = d

  and
  
  A*yx + Z^{-1}*S*yz = bc + Z^{-1}*bs,
  Aw*yx = bw.

  Where we define d as the vector:
  
  d = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu,

  we can solve for yz by solving the following system of equations:

  D0*yz + Ew^{T}*yzw = bc + Z^{-1}*bs - A*C^{-1}*d,
  Ew*yz +     Cw*yzw = bw - Aw*C^{-1}*d

  where C, Ew, and D0 are defined as follows:

  C = B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu,
  Ew = Aw*C^{-1}*A^{T},
  D0 = Z^{-1}*S + A*C^{-1}*A^{T}.

  We can then obtain yz by solving the following system of equations:
  
  Dmat*yz = bc + Z^{-1}*bs - A*C^{-1}*d 
  .         - Ew^{T}*Cw^{-1}*(bw - Aw*C^{-1}*d)

  Once yz is obtained, we find yzw and yx as follows:

  yzw = Cw^{-1}*(bw - Ew*yz - Aw*C^{-1}*d) 
  yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)

  Note: This code uses the temporary arrays xt and wt which therefore
  cannot be inputs/outputs for this function, otherwise strange
  behavior will occur.
*/
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, ParOptScalar *bc, 
                                 ParOptVec *bcw, ParOptScalar *bs,
                                 ParOptVec *bsw,
                                 ParOptVec *bzl, ParOptVec *bzu,
                                 ParOptVec *yx, ParOptScalar *yz, 
                                 ParOptVec *yzw, ParOptScalar *ys,
                                 ParOptVec *ysw,
                                 ParOptVec *yzl, ParOptVec *yzu,
                                 ParOptVec *xt, ParOptVec *wt ){
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
  xt->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(lbvals[i]) > -max_bound_val){
        dvals[i] += cvals[i]*(bzlvals[i]/(xvals[i] - lbvals[i]));
      }
    }
  }
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
        dvals[i] -= cvals[i]*(bzuvals[i]/(ubvals[i] - xvals[i]));
      }
    }
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    wt->copyValues(bcw);

    if (sparse_inequality){
      // Add wt += Zw^{-1}*bsw
      ParOptScalar *wvals, *bswvals, *zwvals;
      wt->getArray(&wvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);
      
      for ( int i = 0; i < nwcon; i++ ){
        wvals[i] += bswvals[i]/zwvals[i];
      }
    }

    // Add the following term: wt -= Aw*C^{-1}*d
    prob->addSparseJacobian(-1.0, x, xt, wt);

    // Compute wt <- Cw^{-1}*wt
    applyCwFactor(wt);
  }

  // Now, compute yz = bc + Z^{-1}*bs - A*C^{-1}*d - Ew^{T}*wt
  memset(yz, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wt->getArray(&wvals);
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
    xt->getArray(&dvals);
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
          yz[i] = bc[i] + bs[i]/z[i] - yz[i];
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
    prob->addSparseJacobian(-1.0, x, xt, yzw);
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
  yx->axpy(1.0, xt);

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
      if (RealPart(lbvals[i]) > -max_bound_val){
        yzlvals[i] = (bzlvals[i] - zlvals[i]*yxvals[i])/(xvals[i] - lbvals[i]);
      }
      else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
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
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, 
                                 ParOptVec *yx, ParOptScalar *yz, 
                                 ParOptVec *yzw, ParOptScalar *ys,
                                 ParOptVec *ysw,
                                 ParOptVec *yzl, ParOptVec *yzu,
                                 ParOptVec *xt, ParOptVec *wt ){
  // Compute the terms from the weighting constraints
  // Compute xt = C^{-1}*bx
  ParOptScalar *bxvals, *dvals, *cvals;
  bx->getArray(&bxvals);
  xt->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = -Aw*C^{-1}*bx
    wt->zeroEntries();
    prob->addSparseJacobian(-1.0, x, xt, wt);

    // Compute wt <- Cw^{-1}*Aw*C^{-1}*bx
    applyCwFactor(wt);
  }

  // Now, compute yz = - A*C0^{-1}*bx - Ew^{T}*wt
  memset(yz, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wt->getArray(&wvals);
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
    xt->getArray(&dvals);
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
    prob->addSparseJacobian(-1.0, x, xt, yzw);
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
  yx->axpy(1.0, xt);

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
      if (RealPart(lbvals[i]) > -max_bound_val){
        yzlvals[i] = -(zlvals[i]*yxvals[i])/(xvals[i] - lbvals[i]);
      }
      else {
        yzlvals[i] = 0.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
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
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, ParOptVec *yx,
                                 ParOptScalar *zt, 
                                 ParOptVec *xt, ParOptVec *wt ){
  // Compute the terms from the weighting constraints
  // Compute xt = C^{-1}*bx
  ParOptScalar *bxvals, *dvals, *cvals;
  bx->getArray(&bxvals);
  xt->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }
  
  // Compute the contribution from the weighting constraints
  if (nwcon > 0){
    // Compute wt = -Aw*C^{-1}*bx
    wt->zeroEntries();
    prob->addSparseJacobian(-1.0, x, xt, wt);

    // Compute wt <- Cw^{-1}*Aw*C^{-1}*bx
    applyCwFactor(wt);
  }

  // Compute ztemp = (S*Z^{-1} - A*C0^{-1}*bx)
  memset(zt, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wt->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      ParOptScalar *ewvals;
      Ew[i]->getArray(&ewvals);
      zt[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    ParOptScalar *avals;
    xt->getArray(&dvals);
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

    zt[i] += ydot;
  }

  if (ncon > 0){
    // Reduce the result to the root processor
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    if (rank == opt_root){
      MPI_Reduce(MPI_IN_PLACE, zt, ncon, PAROPT_MPI_TYPE, MPI_SUM, 
                 opt_root, comm);
    }
    else {
      MPI_Reduce(zt, NULL, ncon, PAROPT_MPI_TYPE, MPI_SUM, 
                 opt_root, comm);
    }
    
    if (rank == opt_root){
      for ( int i = 0; i < ncon; i++ ){
        zt[i] *= -1.0;
      }
      
      int one = 1, info = 0;
      LAPACKdgetrs("N", &ncon, &one, 
                   Dmat, &ncon, dpiv, zt, &ncon, &info);
    }
    
    MPI_Bcast(zt, ncon, PAROPT_MPI_TYPE, opt_root, comm);
  }

  if (nwcon > 0){
    // Compute wt = -Cw^{-1}*(Ew*yz + Aw*C^{-1}*bx)
    // First set wt <- - Ew*yz
    wt->zeroEntries();
    for ( int i = 0; i < ncon; i++ ){
      wt->axpy(-zt[i], Ew[i]);
    }

    // Compute yzw <- - Cw^{-1}*Aw*C^{-1}*d);
    prob->addSparseJacobian(-1.0, x, xt, wt);
    applyCwFactor(wt);
  }

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)
  // therefore yx = C^{-1}*(A^{T}*yz + Aw^{T}*yzw) + xt
  yx->zeroEntries();
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(zt[i], Ac[i]);
  }
  
  // Add the term yx += Aw^{T}*wt
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(1.0, x, wt, yx);
  }

  // Apply the factor C^{-1}*(A^{T}*zt + Aw^{T}*wt)
  ParOptScalar *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xt);
}

/*
  Solve the linear system 
  
  y <- K^{-1}*b

  where K consists of the approximate KKT system where the approximate
  Hessian is replaced with only the diagonal terms.

  Note that in this variant of the function, the right-hand-side
  includes components that are scaled by a given alpha-parameter.
*/
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, 
                                 ParOptScalar alpha, ParOptScalar *bc, 
                                 ParOptVec *bcw, ParOptScalar *bs,
                                 ParOptVec *bsw,
                                 ParOptVec *bzl, ParOptVec *bzu,
                                 ParOptVec *yx, ParOptScalar *yz,
                                 ParOptVec *xt, ParOptVec *wt ){
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
  xt->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*bxvals[i];
  }
  
  if (use_lower){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(lbvals[i]) > -max_bound_val){
        dvals[i] += alpha*cvals[i]*(bzlvals[i]/(xvals[i] - lbvals[i]));
      }
    }
  }
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
        dvals[i] -= alpha*cvals[i]*(bzuvals[i]/(ubvals[i] - xvals[i]));
      }
    }
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    wt->copyValues(bcw);
    wt->scale(alpha);
    
    if (sparse_inequality){
      // Add wt += Zw^{-1}*bsw
      ParOptScalar *wvals, *bswvals, *zwvals;
      wt->getArray(&wvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);
      
      for ( int i = 0; i < nwcon; i++ ){
        wvals[i] += alpha*bswvals[i]/zwvals[i];
      }
    }

    // Add the following term: wt -= Aw*C^{-1}*d
    prob->addSparseJacobian(-1.0, x, xt, wt);

    // Compute wt <- Cw^{-1}*wt
    applyCwFactor(wt);
  }

  // Now, compute yz = bc + Z^{-1}*bs - A*C^{-1}*d - Ew^{T}*wt
  memset(yz, 0, ncon*sizeof(ParOptScalar));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    ParOptScalar *wvals;
    int size = wt->getArray(&wvals);
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
    xt->getArray(&dvals);
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
          yz[i] = alpha*(bc[i] + bs[i]/z[i]) - yz[i];
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
  }

  if (nwcon > 0){
    // Compute yzw = Cw^{-1}*(bcw + Zw^{-1}*bsw - Ew*yz - Aw*C^{-1}*d)
    // First set yzw <- bcw - Ew*yz
    wt->copyValues(bcw);
    wt->scale(alpha);
    for ( int i = 0; i < ncon; i++ ){
      wt->axpy(-yz[i], Ew[i]);
    }

    // Add the term yzw <- yzw + Zw^{-1}*bsw if we are using
    // inequality constraints
    if (sparse_inequality){
      ParOptScalar *yzwvals, *zwvals, *bswvals;
      wt->getArray(&yzwvals);
      zw->getArray(&zwvals);
      bsw->getArray(&bswvals);

      for ( int i = 0; i < nwcon; i++ ){
        yzwvals[i] += alpha*bswvals[i]/zwvals[i];
      }
    }

    // Compute yzw <- Cw^{-1}*(yzw - Aw*C^{-1}*d);
    prob->addSparseJacobian(-1.0, x, xt, wt);
    applyCwFactor(wt);
  }

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yzw)
  // therefore yx = C^{-1}*(A^{T}*yz + Aw^{T}*yzw) + xt
  yx->zeroEntries();
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }
  
  // Add the term yx += Aw^{T}*yzw
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(1.0, x, wt, yx);
  }

  // Apply the factor C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  ParOptScalar *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xt);
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
void ParOpt::setUpKKTSystem( ParOptScalar *zt,
                             ParOptVec *xt1,
                             ParOptVec *xt2, 
                             ParOptVec *wt,
                             int use_qn ){
  if (use_qn){
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
        solveKKTDiagSystem(Z[i], xt1, 
                           zt, xt2, wt);
        
        // Compute the dot products Z^{T}*K^{-1}*Z[i]
        xt1->mdot(Z, size, &Ce[i*size]);
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
void ParOpt::computeKKTStep( ParOptScalar *zt,
                             ParOptVec *xt1, ParOptVec *xt2, 
                             ParOptVec *wt, int use_qn ){
  // Get the size of the limited-memory BFGS subspace
  ParOptScalar b0;
  const ParOptScalar *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (use_qn){
    size = qn->getCompactMat(&b0, &d, &M, &Z);
  }

  // After this point the residuals are no longer required.
  solveKKTDiagSystem(rx, rc, rcw, rs, rsw, rzl, rzu,
                     px, pz, pzw, ps, psw, pzl, pzu,
                     xt1, wt);

  if (size > 0){
    // dz = Z^{T}*px
    px->mdot(Z, size, zt);
    
    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one, 
                 Ce, &size, cpiv, zt, &size, &info);
    
    // Compute rx = Z^{T}*dz
    xt1->zeroEntries();
    for ( int i = 0; i < size; i++ ){
      xt1->axpy(zt[i], Z[i]);
    }
    
    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(xt1, rx, rc, rcw, rs, rsw, rzl, rzu,
                       xt2, wt);

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
    }
  }
}

/*
  Compute the complementarity at the current solution
*/
ParOptScalar ParOpt::computeComp(){
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
      if (RealPart(lbvals[i]) > -max_bound_val){
        product += zlvals[i]*(xvals[i] - lbvals[i]);
        sum += 1.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
        product += zuvals[i]*(ubvals[i] - xvals[i]);
        sum += 1.0;
      }
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
  if (rank == opt_root){
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        product += s[i]*z[i];
        sum += 1.0;
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
ParOptScalar ParOpt::computeCompStep( double alpha_x, double alpha_z ){
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
      if (RealPart(lbvals[i]) > -max_bound_val){
        ParOptScalar xnew = xvals[i] + alpha_x*pxvals[i];
        product += (zlvals[i] + alpha_z*pzlvals[i])*(xnew - lbvals[i]);
        sum += 1.0;
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(ubvals[i]) < max_bound_val){
        ParOptScalar xnew = xvals[i] + alpha_x*pxvals[i];
        product += (zuvals[i] + alpha_z*pzuvals[i])*(ubvals[i] - xnew);
        sum += 1.0;
      }
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
  if (rank == opt_root){
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        product += (s[i] + alpha_x*ps[i])*(z[i] + alpha_z*pz[i]);
        sum += 1.0;
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
void ParOpt::computeMaxStep( double tau, 
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
      if (RealPart(pxvals[i]) < 0.0){
        double alpha = 
          -tau*RealPart(xvals[i] - lbvals[i])/RealPart(pxvals[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
    }
  }

  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(pxvals[i]) > 0.0){
        double alpha = 
          tau*RealPart(ubvals[i] - xvals[i])/RealPart(pxvals[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
    }
  }

  if (dense_inequality){
    // Check the slack variable step
    for ( int i = 0; i < ncon; i++ ){
      if (RealPart(ps[i]) < 0.0){
        double alpha = -tau*RealPart(s[i])/RealPart(ps[i]);
        if (alpha < max_x){
          max_x = alpha;
        }
      }
    }
    
    // Check the step for the Lagrange multipliers
    for ( int i = 0; i < ncon; i++ ){
      if (RealPart(pz[i]) < 0.0){
        double alpha = -tau*RealPart(z[i])/RealPart(pz[i]);
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
      if (RealPart(pzwvals[i]) < 0.0){
        double alpha = 
          -tau*RealPart(zwvals[i])/RealPart(pzwvals[i]);
        if (alpha < max_z){
          max_z = alpha;
        }
      }
    }

    ParOptScalar *swvals, *pswvals;
    sw->getArray(&swvals);
    psw->getArray(&pswvals);
    for ( int i = 0; i < nwcon; i++ ){
      if (RealPart(pswvals[i]) < 0.0){
        double alpha = 
          -tau*RealPart(swvals[i])/RealPart(pswvals[i]);
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
      if (RealPart(pzlvals[i]) < 0.0){
        double alpha = 
          -tau*RealPart(zlvals[i])/RealPart(pzlvals[i]);
        if (alpha < max_z){
          max_z = alpha;
        }
      }
    }
  }
  if (use_upper){
    for ( int i = 0; i < nvars; i++ ){
      if (RealPart(pzuvals[i]) < 0.0){
        double alpha = 
          -tau*RealPart(zuvals[i])/RealPart(pzuvals[i]);
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
  Evaluate the merit function at the current point, assuming that the
  objective and constraint values are up to date.

  The merit function is given as follows:

  varphi(alpha) = 
  
  f(x) + 
  mu*(log(s) + log(x - xl) + log(xu - x)) +
  rho*||c(x) - s||_{2}

  output: The value of the merit function
*/
ParOptScalar ParOpt::evalMeritFunc( ParOptVec *xk, ParOptScalar *sk,
                                    ParOptVec *swk ){
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
      if (RealPart(lbvals[i]) > -max_bound_val){
        if (RealPart(xvals[i] - lbvals[i]) > 1.0){
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
      if (RealPart(ubvals[i]) < max_bound_val){
        if (RealPart(ubvals[i] - xvals[i]) > 1.0){
          pos_result += log(ubvals[i] - xvals[i]);
        }
        else {
          neg_result += log(ubvals[i] - xvals[i]);
        }
      }
    }
  }

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0 && sparse_inequality){
    ParOptScalar *swvals;
    swk->getArray(&swvals);

    for ( int i = 0; i < nwcon; i++ ){
      if (RealPart(swvals[i]) > 1.0){
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
    weight_infeas = wtemp->norm();
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
  
  // Compute the full merit function only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  
  ParOptScalar merit = 0.0;
  if (rank == opt_root){
    // Add the contribution from the slack variables
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        if (RealPart(s[i]) > 1.0){
          pos_result += log(sk[i]);
        }
        else {
          neg_result += log(sk[i]);
        }
      }
    }
    
    // Compute the infeasibility
    ParOptScalar infeas = 0.0;
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        infeas += (c[i] - sk[i])*(c[i] - sk[i]);
      }
    }
    else {
      for ( int i = 0; i < ncon; i++ ){
        infeas += c[i]*c[i];
      }
    }
    infeas = sqrt(infeas) + weight_infeas;
    
    // Add the contribution from the constraints
    merit = (fobj - barrier_param*(pos_result + neg_result) +
             rho_penalty_search*infeas);
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
  inexact_step:  is this an inexact Newton step?

  output:
  merit:     the value of the merit function
  pmerit:    the value of the derivative of the merit function
*/
void ParOpt::evalMeritInitDeriv( double max_x, 
                                 ParOptScalar *_merit, 
                                 ParOptScalar *_pmerit,
                                 int inexact_step,
                                 ParOptVec *wt1, ParOptVec *wt2 ){
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
      if (RealPart(lbvals[i]) > -max_bound_val){
        if (RealPart(xvals[i] - lbvals[i]) > 1.0){ 
          pos_result += log(xvals[i] - lbvals[i]);
        }
        else {
          neg_result += log(xvals[i] - lbvals[i]);
        }
        
        if (RealPart(pxvals[i]) > 0.0){
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
      if (RealPart(ubvals[i]) < max_bound_val){
        if (RealPart(ubvals[i] - xvals[i]) > 1.0){
          pos_result += log(ubvals[i] - xvals[i]);
        }
        else {
          neg_result += log(ubvals[i] - xvals[i]);
        }
        
        if (RealPart(pxvals[i]) > 0.0){
          neg_presult -= pxvals[i]/(ubvals[i] - xvals[i]);
        }
        else {
          pos_presult -= pxvals[i]/(ubvals[i] - xvals[i]);
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
      if (RealPart(swvals[i]) > 1.0){
        pos_result += log(swvals[i]);
      }
      else {
        neg_result += log(swvals[i]);
      }

      if (RealPart(pswvals[i]) > 0.0){
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
    prob->evalSparseCon(x, wt1);
    if (sparse_inequality){
      wt1->axpy(-1.0, sw);
    }
    weight_infeas = wt1->norm();
    
    // Compute the projection of the weight constraints
    // onto the descent direction
    if (inexact_step){
      // Compute (cw(x) - sw)^{T}*(Aw(x)*px - psw)
      wt2->zeroEntries();
      prob->addSparseJacobian(1.0, x, px, wt2);

      if (sparse_inequality){
        weight_proj = wt1->dot(wt2) - wt1->dot(psw);
      }
      else {
        weight_proj = wt1->dot(wt2);
      }

      // Complete the weight projection computation
      if (RealPart(weight_infeas) > 0.0){
        weight_proj = weight_proj/weight_infeas;
      }
      else {
        weight_proj = 0.0;
      }
    }
    else {
      weight_proj = -max_x*weight_infeas;
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

  // Compute the projected derivative
  ParOptScalar proj = g->dot(px);
  
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
      dense_infeas += (c[i] - s[i])*(c[i] - s[i]);
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
  if (inexact_step){
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        dense_proj += (c[i] - s[i])*(Ac[i]->dot(px) - ps[i]);
      }
    }
    else {
      for ( int i = 0; i < ncon; i++ ){
        dense_proj += c[i]*Ac[i]->dot(px);
      }
    }

    // Complete the projected derivative computation for the dense
    // constraints
    if (RealPart(dense_infeas) > 0.0){
      dense_proj = dense_proj/dense_infeas;
    }
  }
  else {
    dense_proj = -max_x*dense_infeas;
  }

  // Add the contribution from the slack variables
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      if (RealPart(s[i]) > 1.0){
        pos_result += log(s[i]);
      }
      else {
        neg_result += log(s[i]);
      }
      
      if (RealPart(ps[i]) > 0.0){
        pos_presult += ps[i]/s[i];
      }
      else {
        neg_presult += ps[i]/s[i];
      }
    }
  }
  
  if (rank == opt_root){
    // Now, set up the full problem infeasibility
    ParOptScalar infeas = dense_infeas + weight_infeas;
    ParOptScalar infeas_proj = dense_proj + weight_proj;
          
    // Compute the numerator term
    ParOptScalar numer = proj - barrier_param*(pos_presult + neg_presult);
      
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
    if (RealPart(infeas) > 0.0){
      rho_hat = -RealPart(numer)/
        RealPart(infeas_proj + penalty_descent_fraction*max_x*infeas);
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
    
    // Now, evaluate the merit function and its derivative
    // based on the new value of the penalty parameter
    merit = (fobj - barrier_param*(pos_result + neg_result) + 
             rho_penalty_search*infeas);
    pmerit = numer + rho_penalty_search*infeas_proj;
  }

  input[0] = merit;
  input[1] = pmerit;
  input[2] = rho_penalty_search;

  // Broadcast the penalty parameter to all procs
  MPI_Bcast(input, 3, PAROPT_MPI_TYPE, opt_root, comm);

  *_merit = input[0];
  *_pmerit = input[1];
  rho_penalty_search = RealPart(input[2]);
}

/*
  Perform a backtracking line search from the current point along the
  specified direction. Note that this is a very simple line search
  without a second-order correction which may be required to alleviate
  the Maratos effect. (This should work regardless for compliance
  problems when the problem should be nearly convex.)

  input:
  alpha:  (in/out) the initial line search step length
  m0:     the merit function 
  dm0:    the projected derivative of the merit function along p

  returns: 
  fail:   did the line search find an acceptable point
*/
int ParOpt::lineSearch( double *_alpha, 
                        ParOptScalar m0, ParOptScalar dm0 ){
  // Perform a backtracking line search until the sufficient decrease
  // conditions are satisfied 
  double alpha = *_alpha;
  int fail = 1;

  for ( int j = 0; j < max_line_iters; j++ ){
    // Set rx = x + alpha*px
    rx->copyValues(x);
    rx->axpy(alpha, px);

    // Set rcw = sw + alpha*psw
    if (nwcon > 0 && sparse_inequality){
      rsw->copyValues(sw);
      rsw->axpy(alpha, psw);
    }

    // Set rs = s + alpha*ps
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        rs[i] = s[i] + alpha*ps[i];
      }
    }

    // Evaluate the objective and constraints at the new point
    int fail_obj = prob->evalObjCon(rx, &fobj, c);
    neval++;

    if (fail_obj){
      fprintf(stderr, 
              "ParOpt: Evaluation failed during line search, trying new point\n");

      // Multiply alpha by 1/10 like SNOPT
      alpha *= 0.1;
      continue;
    }

    // Evaluate the merit function
    ParOptScalar merit = evalMeritFunc(rx, rs, rsw);

    // Check the sufficient decrease condition
    if (RealPart(merit) < 
        RealPart(m0 + armijio_constant*alpha*dm0)){
      // We have successfully found a point
      fail = 0;
      break;
    }

    // Update the new value of alpha
    if (j < max_line_iters-1){
      if (use_backtracking_alpha){
        alpha = 0.5*alpha;
      }
      else {
        double alpha_new = -0.5*RealPart(dm0)*(alpha*alpha)/
          RealPart(merit - m0 - dm0*alpha);

        // Bound the new step length from below by 0.01
        if (alpha_new < 0.01*alpha){
          alpha = 0.01*alpha;
        }
        else {
          alpha = alpha_new;
        }
      }
    }
  }

  // Set the new values of the variables
  if (nwcon > 0){
    zw->axpy(alpha, pzw);
    if (sparse_inequality){
      sw->axpy(alpha, psw);
    }
  }
  if (use_lower){
    zl->axpy(alpha, pzl);
  }
  if (use_upper){
    zu->axpy(alpha, pzu);
  }

  for ( int i = 0; i < ncon; i++ ){
    z[i] += alpha*pz[i];
  }
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      s[i] += alpha*ps[i];
    }
  }
  
  // Compute the negative gradient of the Lagrangian using the
  // old gradient information with the new multiplier estimates
  if (!sequential_linear_method){
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
  x->axpy(alpha, px);

  // Evaluate the derivative
  int fail_gobj = prob->evalObjConGradient(x, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr, 
            "ParOpt: Gradient evaluation failed at final line search\n");
  }

  // Add the new gradient of the Lagrangian with the new
  // multiplier estimates
  if (!sequential_linear_method){
    y_qn->axpy(1.0, g);
    for ( int i = 0; i < ncon; i++ ){
      y_qn->axpy(-z[i], Ac[i]);
    }
         
    // Add the term: -Aw^{T}*zw
    if (nwcon > 0){
      prob->addSparseJacobianTranspose(-1.0, x, zw, y_qn);
    }
  }
  
  // Set the final value of alpha used in the line search
  // iteration
  *_alpha = alpha;

  return fail;
}

/*
  Get the initial design variable values, and the lower and upper
  bounds. Perform a check to see that the bounds are consistent and
  modify the design variable to conform to the bounds if neccessary.

  input:
  init_multipliers:  Flag to indicate whether to initialize multipliers
*/
void ParOpt::initAndCheckDesignAndBounds( int init_multipliers ){
  // Get the design variables and bounds
  prob->getVarsAndBounds(x, lb, ub);
  
  if (init_multipliers){
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
    }
  }

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
      if (RealPart(lbvals[i]) > -max_bound_val && 
          RealPart(ubvals[i]) < max_bound_val){
        if (RealPart(lbvals[i]) >= RealPart(ubvals[i])){
          check_flag = (check_flag | 1);
          // Make up bounds
          lbvals[i] = 0.5*(lbvals[i] + ubvals[i]) - 0.5*rel_bound;
          ubvals[i] = lbvals[i] + rel_bound;
        }

        delta = ubvals[i] - lbvals[i];
      }

      // Check if x is too close the boundary
      if (RealPart(lbvals[i]) > -max_bound_val &&
          RealPart(xvals[i]) < RealPart(lbvals[i] + rel_bound*delta)){
        check_flag = (check_flag | 2);
        xvals[i] = lbvals[i] + rel_bound*delta;
      }
      if (RealPart(ubvals[i]) < max_bound_val &&
          RealPart(xvals[i]) > RealPart(ubvals[i] - rel_bound*delta)){
        check_flag = (check_flag | 4);
        xvals[i] = ubvals[i] - rel_bound*delta;
      }
    }
  }

  // Print the results of the warnings
  if (check_flag & 1){
    fprintf(stderr, "ParOpt Warning: Variable bounds are inconsistent\n");
  }
  if (check_flag & 2){
    fprintf(stderr, 
            "ParOpt Warning: Variables may be too close to lower bound\n");
  }
  if (check_flag & 4){
    fprintf(stderr, 
            "ParOpt Warning: Variables may be too close to upper bound\n");
  }

  // Set the largrange multipliers with bounds outside the limits to
  // zero. This ensures that they have no effect because they will not
  // be updated once the optimization begins.
  ParOptScalar *zlvals, *zuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);
  
  for ( int i = 0; i < nvars; i++ ){
    if (RealPart(lbvals[i]) <= -max_bound_val){
      zlvals[i] = 0.0;
    }
    if (RealPart(ubvals[i]) >= max_bound_val){
      zuvals[i] = 0.0;
    }
  }
}

/*
  Perform the actual optimization. 

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
*/
int ParOpt::optimize( const char *checkpoint ){
  if (gradient_check_frequency > 0){
    checkGradients(gradient_check_step);
  }
  
  // This is not the final barrier problem
  final_barrier_problem = 0;

  // Zero out the number of function/gradient evaluations
  neval = ngeval = nhvec = 0;

  // Initialize and check the design variables and bounds
  initAndCheckDesignAndBounds(init_starting_point); 

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

  // Set the largrange multipliers with bounds outside the
  // limits to zero
  ParOptScalar *lbvals, *ubvals, *zlvals, *zuvals;
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);
  
  for ( int i = 0; i < nvars; i++ ){
    if (RealPart(lbvals[i]) <= -max_bound_val){
      zlvals[i] = 0.0;
    }
    if (RealPart(ubvals[i]) >= max_bound_val){
      zuvals[i] = 0.0;
    }
  }

  // Find an initial estimate of the Lagrange multipliers for the
  // inequality constraints
  if (init_starting_point){
    // Form the right-hand-side of the least squares eigenvalue
    // problem
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

        // Keep the Lagrange multipliers if they are within 
        // a reasonable range and they are positive.
        for ( int i = 0; i < ncon; i++ ){
          if (RealPart(z[i]) < 0.01 || 
              RealPart(z[i]) > 1000.0){
            z[i] = 1.0;
          }
        }
      }
    }
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

  // Keep track of how many GMRES iterations there were
  int gmres_iters = 0;

  // Information about what happened on the previous major iteration
  char info[64];
  info[0] = '\0';

  for ( int k = 0; k < max_major_iters; k++ ){
    if (!sequential_linear_method){
      if (k > 0 && k % hessian_reset_freq == 0){
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
    if (k % write_output_frequency == 0){
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
    if (k > 0 &&
        (gradient_check_frequency > 0) && 
        (k % gradient_check_frequency == 0)){
      checkGradients(gradient_check_step);
    }

    // Compute the complementarity
    ParOptScalar comp = computeComp();
    
    // Compute the residual of the KKT system 
    double max_prime, max_dual, max_infeas;
    computeKKTRes(&max_prime, &max_dual, &max_infeas);

    // Compute the norm of the residuals
    double res_norm = max_prime;
    if (max_dual > res_norm){ res_norm = max_dual; }
    if (max_infeas > res_norm){ res_norm = max_infeas; }
    if (k == 0){
      res_norm_prev = res_norm;
    }

    // Determine if we should switch to a new barrier problem or not
    int rel_function_test = 
      (alpha_xprev == 1.0 && alpha_zprev == 1.0 &&
       (fabs(RealPart(fobj - fobj_prev)) < rel_func_tol*fabs(RealPart(fobj_prev))));

    // Set the flag to indicate whether the barrier problem has
    // converged
    int barrier_converged = 0;
    if (k > 0 && ((res_norm < 10.0*barrier_param) || 
                  rel_function_test)){
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
      // Record the value of the old barrier function
      double mu_old = barrier_param;

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
        final_barrier_problem = 1;
        new_barrier_param = 0.1*abs_res_tol;
      }

      // Now, that we have adjusted the barrier parameter, we have
      // to modify the residuals to match
      max_dual = 0.0;
      if (dense_inequality){
        for ( int i = 0; i < ncon; i++ ){
          rs[i] -= (mu_old - new_barrier_param);
        
          if (fabs(RealPart(rs[i])) > max_dual){
            max_dual = fabs(RealPart(rs[i]));
          }
        }
      }

      if (nwcon > 0 && sparse_inequality){
        // Set the values of the perturbed complementarity
        // constraints for the sparse slack variables
        ParOptScalar  *rswvals;
        rsw->getArray(&rswvals);
    
        for ( int i = 0; i < nwcon; i++ ){
          rswvals[i] -= (mu_old - new_barrier_param);
        }
    
        double dual_zw = rsw->maxabs();
        if (dual_zw > max_dual){
          max_dual = dual_zw;
        }
      }

      // Adjust the lower-bound residuals if required
      if (use_lower){
        ParOptScalar *lbvals, *rzlvals;
        lb->getArray(&lbvals);
        rzl->getArray(&rzlvals);

        for ( int i = 0; i < nvars; i++ ){
          if (RealPart(lbvals[i]) > -max_bound_val){
            rzlvals[i] -= (mu_old - new_barrier_param);
          }
          else {
            rzlvals[i] = 0.0;
          }
        }

        double dual_zl = rzl->maxabs();
        if (RealPart(dual_zl) > RealPart(max_dual)){
          max_dual = dual_zl;
        }
      }

      // Adjust the upper-bound residuals if required
      if (use_upper){
        ParOptScalar *ubvals, *rzuvals;
        lb->getArray(&ubvals);
        rzu->getArray(&rzuvals);

        for ( int i = 0; i < nvars; i++ ){
          if (RealPart(ubvals[i]) < max_bound_val){
            rzuvals[i] -= (mu_old - new_barrier_param);
          }
          else {
            rzuvals[i] = 0.0;
          }
        }

        double dual_zu = rzu->maxabs();
        if (dual_zu > max_dual){
          max_dual = dual_zu;
        }
      }

      // Reset the penalty parameter to zero
      rho_penalty_search = 0.0;

      // Recompute the maximum residual norm after the update
      res_norm = max_prime;
      if (max_dual > res_norm){ res_norm = max_dual; }
      if (max_infeas > res_norm){ res_norm = max_infeas; }
    }

    // Print all the information we can to the screen...
    if (outfp && rank == opt_root){
      if (k % 10 == 0 || gmres_iters > 0){
        fprintf(outfp, "\n%4s %4s %4s %4s %7s %7s %7s %12s \
%7s %7s %7s %7s %7s %8s %7s info\n",
                "iter", "nobj", "ngrd", "nhvc", "alpha", "alphx", "alphz", 
                "fobj", "|opt|", "|infes|", "|dual|", "mu", 
                "comp", "dmerit", "rho");
      }

      if (k == 0){
        fprintf(outfp, "%4d %4d %4d %4d %7s %7s %7s %12.5e \
%7.1e %7.1e %7.1e %7.1e %7.1e %8s %7s %s\n",
                k, neval, ngeval, nhvec, " ", " ", " ",
                RealPart(fobj), max_prime, max_infeas, max_dual, 
                barrier_param, RealPart(comp), " ", " ", info);
      }
      else {
        fprintf(outfp, "%4d %4d %4d %4d %7.1e %7.1e %7.1e %12.5e \
%7.1e %7.1e %7.1e %7.1e %7.1e %8.1e %7.1e %s\n",
                k, neval, ngeval, nhvec, alpha_prev, alpha_xprev, alpha_zprev,
                RealPart(fobj), max_prime, max_infeas, max_dual, 
                barrier_param, RealPart(comp), RealPart(dm0_prev), 
                rho_penalty_search, info);
      }
      
      // Flush the buffer so that we can see things immediately
      fflush(outfp);
    }

    // Check for convergence. We apply two different convergence
    // criteria at this point: the first based on the infinity norm of
    // the KKT condition residuals, and the second based on the
    // difference between subsequent calls.

    // Check if the barrier term has converged. This is required for
    // both convergence checks
    int barrier_term = (final_barrier_problem || 
                        (barrier_param <= 0.1*abs_res_tol));
    if (barrier_converged){
      barrier_term = (final_barrier_problem ||
                      (new_barrier_param <= 0.1*abs_res_tol));
    }

    // Check either of the two convergence criteria
    int converged = 0;
    if (k > 0 && barrier_term && 
        (res_norm < abs_res_tol || rel_function_test)){
      converged = 1;
    }

    // Broadcast the convergence result from the root processor. This avoids
    // comparing values that might be different on different procs.
    MPI_Bcast(&converged, 1, MPI_INT, opt_root, comm);

    // Everybody quit altogether if we've converged
    if (converged){
      break;
    }

    // Set/store the new barrier parameter now.
    if (barrier_converged){
      barrier_param = new_barrier_param;
    }

    // Compute the relative GMRES tolerance given the residuals
    double gmres_rtol = 
      eisenstat_walker_gamma*pow((res_norm/res_norm_prev),
                                 eisenstat_walker_alpha);
    
    // Assign the previous objective/norm for next time through
    fobj_prev = fobj;
    res_norm_prev = res_norm;

    // Check if we should compute a Newton step or a quasi-Newton
    // step. Note that at this stage, we use s_qn and y_qn as
    // temporary arrays to help compute the KKT step. After
    // the KKT step is computed, we use them to store the
    // change in variables/gradient for the BFGS update.
    gmres_iters = 0;

    if (use_hvec_product && 
        (max_prime < nk_switch_tol &&
         max_dual < nk_switch_tol && 
         max_infeas < nk_switch_tol) && 
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

      // Compute the inexact step using GMRES - note that this
      // uses a fixed tolerance -- this may lead to over-solving
      // if rtol is too tight
      gmres_iters = 
        computeKKTInexactNewtonStep(ztemp, y_qn, s_qn, wtemp,
                                    gmres_rtol, gmres_atol,
                                    use_qn);
    }
    else {
      int use_qn = 1;
      if (sequential_linear_method){
        use_qn = 0;
      }
      
      // Set up the KKT diagonal system
      setUpKKTDiagSystem(s_qn, wtemp, use_qn);
      
      // Set up the full KKT system
      setUpKKTSystem(ztemp, s_qn, y_qn, wtemp, use_qn);
      
      // Solve for the KKT step
      computeKKTStep(ztemp, s_qn, y_qn, wtemp, use_qn);
    }

    // Check the KKT step
    if (k == major_iter_step_check){
      checkKKTStep(gmres_iters > 0);
    }

    // Compute the maximum permitted line search lengths
    double tau = min_fraction_to_boundary;
    double tau_mu = 1.0 - barrier_param;
    if (tau_mu >= tau){
      tau = tau_mu;
    } 

    double alpha_x = 1.0, alpha_z = 1.0;
    computeMaxStep(tau, &alpha_x, &alpha_z);

    // Keep track of whether we set both the design and Lagrange
    // multiplier steps equal to one another
    int ceq_step = 0;

    // Check if we're using a Newton step or not
    if (gmres_iters == 0){
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

      if (RealPart(comp_new) > 10.0*RealPart(comp)){
        ceq_step = 1;
        if (alpha_x > alpha_z){
          alpha_x = alpha_z;
        }
        else {
          alpha_z = alpha_x;
        }
      }
    }
    else if (gmres_iters > 0){
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

    for ( int i = 0; i < ncon; i++ ){
      pz[i] *= alpha_z;
    }
    if (dense_inequality){
      for ( int i = 0; i < ncon; i++ ){
        ps[i] *= alpha_x;
      }
    }

    // Store the design variable locations for the Hessian update. The
    // gradient difference update is done after the step has been
    // selected, but before the new gradient is evaluated (so we have
    // the new multipliers)
    if (!sequential_linear_method){
      s_qn->copyValues(x);
      s_qn->scale(-1.0);
    }

    // Keep track of the step length size
    double alpha = 1.0;
    int line_fail = 0;

    if (use_line_search){
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter
      ParOptScalar m0, dm0;
      evalMeritInitDeriv(alpha_x, &m0, &dm0, (gmres_iters > 0),
                         wtemp, rcw);

      // Check that the merit function derivative is correct and print
      // the derivative to the screen on the optimization-root
      // processor
      if (k == major_iter_step_check){
        double dh = merit_func_check_epsilon;
        rx->copyValues(x);
        rx->axpy(dh, px);
        
        if (dense_inequality){
          for ( int i = 0; i < ncon; i++ ){
            rs[i] = s[i] + dh*ps[i];
          }
        }

        if (nwcon > 0 && sparse_inequality){
          rsw->copyValues(sw);
          rsw->axpy(dh, psw);
        }

        // Evaluate the objective
        int fail_obj = prob->evalObjCon(rx, &fobj, c);
        neval++;
        if (fail_obj){
          fprintf(stderr, 
                  "ParOpt: Function and constraint evaluation failed\n");
          return fail_obj;
        }
        ParOptScalar m1 = evalMeritFunc(rx, rs, rsw);

        if (rank == opt_root){
          ParOptScalar fd = (m1 - m0)/dh;
          printf("Merit function test\n");
          printf("dm FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
                 RealPart(fd), RealPart(dm0), 
                 fabs(RealPart(fd - dm0)), fabs(RealPart((fd - dm0)/fd)));
        }
      }
      
      // If the directional derivative is negative, take the full
      // step, regardless. This can happen when an inexact Newton step
      // is used. Also, if the directional derivative is too small we
      // also apply the full step.
      if (RealPart(dm0) > 0.0 ||
          RealPart(dm0) > -abs_res_tol*abs_res_tol){
        // Apply the full step to the Lagrange multipliers and
        // slack variables
        alpha = 1.0;
        if (nwcon > 0){
          zw->axpy(alpha, pzw);
          if (sparse_inequality){
            sw->axpy(alpha, psw);
          }
        }
        if (use_lower){
          zl->axpy(alpha, pzl);
        }
        if (use_upper){
          zu->axpy(alpha, pzu);
        }

        for ( int i = 0; i < ncon; i++ ){
          z[i] += alpha*pz[i];
        }
        if (dense_inequality){
          for ( int i = 0; i < ncon; i++ ){
            s[i] += alpha*ps[i];
          }
        }
        
        // Compute the negative gradient of the Lagrangian using the
        // old gradient information with the new multiplier estimates
        if (!sequential_linear_method){
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

        // Update x here so that we don't impact the design variables
        // when computing Aw(x)^{T}*zw
        x->axpy(alpha, px);

        // Evaluate the objective, constraint and their gradients at
        // the current values of the design variables
        int fail_obj = prob->evalObjCon(x, &fobj, c);
        neval++;
        if (fail_obj){
          fprintf(stderr, 
                  "ParOpt: Function and constraint evaluation failed\n");
          return fail_obj;
        }
        int fail_gobj = prob->evalObjConGradient(x, g, Ac);
        ngeval++;
        if (fail_gobj){
          fprintf(stderr, 
                  "ParOpt: Gradient evaluation failed\n");
          return fail_obj;
        }

        // Add the new gradient of the Lagrangian with the new
        // multiplier estimates
        if (!sequential_linear_method){
          y_qn->axpy(1.0, g);
          for ( int i = 0; i < ncon; i++ ){
            y_qn->axpy(-z[i], Ac[i]);
          }

          // Add the term: -Aw^{T}*zw
          if (nwcon > 0){
            prob->addSparseJacobianTranspose(-1.0, x, zw, y_qn);
          }
        }
      }
      else {
        // Perform the line search
        line_fail = lineSearch(&alpha, m0, dm0);
      }

      // Store the previous merit function derivative
      dm0_prev = dm0;
    }
    else {
      // Apply the full step to the Lagrange multipliers and
      // slack varaibles
      if (nwcon > 0){
        zw->axpy(alpha, pzw);
        if (sparse_inequality){
          sw->axpy(alpha, psw);
        }
      }
      if (use_lower){
        zl->axpy(alpha, pzl);
      }
      if (use_upper){
        zu->axpy(alpha, pzu);
      }

      for ( int i = 0; i < ncon; i++ ){
        z[i] += alpha*pz[i];
      }
      if (dense_inequality){
        for ( int i = 0; i < ncon; i++ ){
          s[i] += alpha*ps[i];
        }
      }
 
      // Compute the negative gradient of the Lagrangian using the
      // old gradient information with the new multiplier estimates
      if (!sequential_linear_method){
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
      x->axpy(alpha, px);

      // Evaluate the objective, constraint and their gradients at the
      // current values of the design variables
      int fail_obj = prob->evalObjCon(x, &fobj, c);
      neval++;
      if (fail_obj){
        fprintf(stderr, 
                "ParOpt: Function and constraint evaluation failed\n");
        return fail_obj;
      }
      int fail_gobj = prob->evalObjConGradient(x, g, Ac);
      ngeval++;
      if (fail_gobj){
        fprintf(stderr, "ParOpt: Gradient evaluation failed\n");
        return fail_obj;
      }

      // Add the new gradient of the Lagrangian with the new
      // multiplier estimates to complete the y-update step  
      if (!sequential_linear_method){
        y_qn->axpy(1.0, g);
        for ( int i = 0; i < ncon; i++ ){
          y_qn->axpy(-z[i], Ac[i]);
        }
         
        // Add the term: -Aw^{T}*zw
        if (nwcon > 0){
          prob->addSparseJacobianTranspose(-1.0, x, zw, y_qn);
        }
      }
    }

    // Complete the updated step
    if (!sequential_linear_method){
      s_qn->axpy(1.0, x);
    }

    // Store the steps in x/z for printing later
    alpha_prev = alpha;
    alpha_xprev = alpha_x;
    alpha_zprev = alpha_z;

    // Compute the Quasi-Newton update
    int up_type = 0;
    if (!sequential_linear_method && !line_fail){
      up_type = qn->update(s_qn, y_qn);
    }

    // Reset the quasi-Newton Hessian if there is a line search failure
    if (line_fail){
      qn->reset();
    }

    // Create a string to print to the screen
    if (rank == opt_root){
      // The string of unforseen events
      info[0] = '\0';     
      if (gmres_iters > 0){
        // Print how well GMRES is doing
        sprintf(&info[strlen(info)], "%s%d ", "iNK", gmres_iters);
      }
      if (up_type == 1){ 
        // Damped BFGS update
        sprintf(&info[strlen(info)], "%s ", "dampH");
      }
      else if (up_type == 2){
        // Skipped update
        sprintf(&info[strlen(info)], "%s ", "skipH");
      }
      if (line_fail){
        // Line search failure
        sprintf(&info[strlen(info)], "%s ", "LF");
      }
      if (RealPart(dm0_prev) > -abs_res_tol*abs_res_tol){
        // Skip the line search b/c descent direction is not
        // sufficiently descent-y
        sprintf(&info[strlen(info)], "%s ", "Lskp");
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
  The following functions create 
*/
/*
void ParOpt::leftSymmTransform( ParOptVec *bx,
                                ParOptScalar *bc,
                                ParOptVec *bcw, 
                                ParOptScalar *bs,
                                ParOptVec *bsw,
                                ParOptVec *bzl,
                                ParOptVec *bzu ){
  // bs <--  -S^{-1/2}*bs
  if (dense_inequality && bs){
    for ( int k = 0; k < ncon; k++ ){
      if (s[k] != 0.0){
        bs[k] *= -1.0/sqrt(s[k]);
      }
    }
  }

  // bsw <--  -Sw^{-1/2}*bsw
  if (sparse_inequality && rsw){
    ParOptScalar *swvals, *bswvals;
    zw->getArray(&zwvals);
    bsw->getArray(&rswvals);
    
    for ( int i = 0; i < nwcon; i++ ){
      bswvals[i] *= -1.0/sqrt(swvals[i]);
    }
  }

  // bzl <--  -Zl^{-1/2}*bzl
  if (use_lower && bzl){
    ParOptScalar *zlvals, *bzlvals;
    zl->getArray(&zlvals);
    bzl->getArray(&bzlvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zlvals[i] != 0.0){
        bzlvals[i] *= -1.0/sqrt(zlvals[i]);
      }
    }
  }

  // bzu <--  -Zu^{-1/2}*bzu
  if (use_upper && bzu){
    ParOptScalar *zuvals, *bzuvals;
    zu->getArray(&zuvals);
    bzu->getArray(&bzuvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zuvals[i] != 0.0){
        bzuvals[i] *= -1.0/sqrt(zuvals[i]);
      }
    }
  }
}

void ParOpt::leftInvSymmTransform( ParOptVec *bx,
                                   ParOptScalar *bc,
                                   ParOptVec *bcw, 
                                   ParOptScalar *bs,
                                   ParOptVec *bsw,
                                   ParOptVec *bzl,
                                   ParOptVec *bzu ){
  // bs <--  -S^{1/2}*bs
  if (dense_inequality && bs){
    for ( int k = 0; k < ncon; k++ ){
      if (s[k] != 0.0){
        bs[k] *= -sqrt(s[k]);
      }
    }
  }

  // bsw <--  -Sw^{1/2}*bsw
  if (sparse_inequality && rsw){
    ParOptScalar *swvals, *bswvals;
    zw->getArray(&zwvals);
    bsw->getArray(&rswvals);
    
    for ( int i = 0; i < nwcon; i++ ){
      bswvals[i] *= -sqrt(swvals[i]);
    }
  }

  // bzl <--  -Zl^{1/2}*bzl
  if (use_lower && bzl){
    ParOptScalar *zlvals, *bzlvals;
    zl->getArray(&zlvals);
    bzl->getArray(&bzlvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zlvals[i] != 0.0){
        bzlvals[i] *= -sqrt(zlvals[i]);
      }
    }
  }

  // bzu <--  -Zu^{1/2}*bzu
  if (use_upper && bzu){
    ParOptScalar *zuvals, *bzuvals;
    zu->getArray(&zuvals);
    bzu->getArray(&bzuvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zuvals[i] != 0.0){
        bzuvals[i] *= -sqrt(zuvals[i]);
      }
    }
  }
}

void ParOpt::rightSymmTransform( ParOptVec *dx,
                                 ParOptScalar *dz,
                                 ParOptVec *dzw, 
                                 ParOptScalar *ds,
                                 ParOptVec *dsw,
                                 ParOptVec *dzl,
                                 ParOptVec *dzu ){
  for ( int k = 0; k < ncon; k++ ){
    dz[k] *= -1.0;
  }

  // ds <--  -S^{1/2}*ds
  if (dense_inequality && ds){
    for ( int k = 0; k < ncon; k++ ){
      if (s[k] != 0.0){
        ds[k] *= sqrt(s[k]);
      }
    }
  }

  // dsw <--  Sw^{1/2}*dsw
  if (sparse_inequality && rsw){
    ParOptScalar *swvals, *dswvals;
    zw->getArray(&zwvals);
    dsw->getArray(&rswvals);
    
    for ( int i = 0; i < nwcon; i++ ){
      dswvals[i] *= sqrt(swvals[i]);
    }
  }

  // dzl <--  Zl^{1/2}*dzl
  if (use_lower && dzl){
    ParOptScalar *zlvals, *dzlvals;
    zl->getArray(&zlvals);
    dzl->getArray(&dzlvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zlvals[i] != 0.0){
        dzlvals[i] *= sqrt(zlvals[i]);
      }
    }
  }

  // dzu <--  Zu^{1/2}*dzu
  if (use_upper && dzu){
    ParOptScalar *zuvals, *dzuvals;
    zu->getArray(&zuvals);
    dzu->getArray(&dzuvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zuvals[i] != 0.0){
        dzuvals[i] *= sqrt(zuvals[i]);
      }
    }
  }
}

void ParOpt::rightInvSymmTransform( ParOptVec *dx,
                                    ParOptScalar *dz,
                                    ParOptVec *dzw, 
                                    ParOptScalar *ds,
                                    ParOptVec *dsw,
                                    ParOptVec *dzl,
                                    ParOptVec *dzu ){
  for ( int k = 0; k < ncon; k++ ){
    dz[k] *= -1.0;
  }

  // ds <--  -S^{1/2}*ds
  if (dense_inequality && ds){
    for ( int k = 0; k < ncon; k++ ){
      if (s[k] != 0.0){
        ds[k] *= 1.0/sqrt(s[k]);
      }
    }
  }

  // dsw <--  Sw^{1/2}*dsw
  if (sparse_inequality && rsw){
    ParOptScalar *swvals, *dswvals;
    zw->getArray(&zwvals);
    dsw->getArray(&rswvals);
    
    for ( int i = 0; i < nwcon; i++ ){
      dswvals[i] *= 1.0/sqrt(swvals[i]);
    }
  }

  // dzl <--  Zl^{1/2}*dzl
  if (use_lower && dzl){
    ParOptScalar *zlvals, *dzlvals;
    zl->getArray(&zlvals);
    dzl->getArray(&dzlvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zlvals[i] != 0.0){
        dzlvals[i] *= 1.0/sqrt(zlvals[i]);
      }
    }
  }

  // dzu <--  Zu^{1/2}*dzu
  if (use_upper && dzu){
    ParOptScalar *zuvals, *dzuvals;
    zu->getArray(&zuvals);
    dzu->getArray(&dzuvals);

    for ( int i = 0; i < nvars; i++ ){
      if (zuvals[i] != 0.0){
        dzuvals[i] *= 1.0/sqrt(zuvals[i]);
      }
    }
  }
}
*/
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
int ParOpt::computeKKTInexactNewtonStep( ParOptScalar *zt, 
                                         ParOptVec *xt1, ParOptVec *xt2,
                                         ParOptVec *wt,
                                         double rtol, double atol,
                                         int use_qn ){
  // Initialize the data from the gmres object
  ParOptScalar *H = gmres_H;
  ParOptScalar *alpha = gmres_alpha;
  ParOptScalar *res = gmres_res;
  ParOptScalar *Qcos = &gmres_Q[0];
  ParOptScalar *Qsin = &gmres_Q[gmres_subspace_size];
  ParOptVec **W = gmres_W;

  int use_symmetrized_newton = 0;

  // Compute the beta factor: the product of the diagonal terms
  // after normalization
  ParOptScalar beta = 0.0;
  for ( int i = 0; i < ncon; i++ ){
    beta += rc[i]*rc[i];
    }
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      beta += rs[i]*rs[i];
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
  
  // Broadcast the norm of the residuals and the
  // beta parameter to keep things consistent across processors
  ParOptScalar temp[2];
  temp[0] = bnorm;
  temp[1] = beta;
  MPI_Bcast(temp, 2, PAROPT_MPI_TYPE, opt_root, comm);

  bnorm = temp[0];
  beta = temp[1];

  // Compute the final value of the beta term
  beta *= 1.0/(bnorm*bnorm);

  // Initialize the residual norm
  res[0] = bnorm;
  W[0]->copyValues(rx);
  W[0]->scale(1.0/res[0]);
  alpha[0] = 1.0;

  // Keep track of the actual number of iterations
  int solve_flag = 0;
  int niters = 0;

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root && outfp){
    fprintf(outfp, "%5s %4s %4s %7s %7s %7.1e\n", 
            "gmres", "nhvc", "iter", "res", "rel", rtol);
    fprintf(outfp, "      %4d %4d %7.1e %7.1e\n", 
            nhvec, 0, fabs(RealPart(res[0])), 1.0);
  }

  for ( int i = 0; i < gmres_subspace_size; i++ ){
    // Compute M^{-1}*[ W[i], alpha[i]*yc, ... ] 
    // Get the size of the limited-memory BFGS subspace
    ParOptScalar b0;
    const ParOptScalar *d, *M;
    ParOptVec **Z;
    int size = 0;
    if (use_qn){
      size = qn->getCompactMat(&b0, &d, &M, &Z);
    }

    // At this point the residuals are no longer required.
    solveKKTDiagSystem(W[i], alpha[i]/bnorm, 
                       rc, rcw, rs, rsw, rzl, rzu,
                       xt1, zt, xt2, wt);

    if (size > 0){
      // dz = Z^{T}*xt1
      xt1->mdot(Z, size, zt);
    
      // Compute dz <- Ce^{-1}*dz
      int one = 1, info = 0;
      LAPACKdgetrs("N", &size, &one, 
                   Ce, &size, cpiv, zt, &size, &info);
    
      // Compute rx = Z^{T}*dz
      xt2->zeroEntries();
      for ( int k = 0; k < size; k++ ){
        xt2->axpy(zt[k], Z[k]);
      }
    
      // Solve the digaonal system again, this time simplifying the
      // result due to the structure of the right-hand-side.  Note
      // that this call uses W[i+1] as a temporary vector.
      solveKKTDiagSystem(xt2, px,
                         zt, W[i+1], wt);

      // Add the final contributions 
      xt1->axpy(-1.0, px);
    }

    // Compute the vector product with the exact Hessian
    prob->evalHvecProduct(x, z, zw, xt1, W[i+1]);
    nhvec++;

    // Add the term -B*W[i]
    if (!sequential_linear_method){
      qn->multAdd(-1.0, xt1, W[i+1]);
    }

    // Add the term from the diagonal
    W[i+1]->axpy(1.0, W[i]);

    // Set the initial value of the scalar
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
    
    if (rank == opt_root){
      fprintf(outfp, "      %4d %4d %7.1e %7.1e\n", 
              nhvec, i+1, fabs(RealPart(res[i+1])), 
              fabs(RealPart(res[i+1]/bnorm)));
      fflush(outfp);
    }
   
    // Check for convergence
    if (fabs(RealPart(res[i+1])) < atol ||
        fabs(RealPart(res[i+1])) < rtol*RealPart(bnorm)){
      solve_flag = 1;
      break;
    }
  }
  
  // Now, compute the solution - the linear combination of the
  // Arnoldi vectors. H is now an upper triangular matrix.

  // Compute the weights
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
  }

  rzl->scale(gamma);
  rzu->scale(gamma);
  if (nwcon > 0){
    rcw->scale(gamma);
    rsw->scale(gamma);
  }

  // Apply M^{-1} to the result to obtain the final answer
  solveKKTDiagSystem(W[0], rc, rcw, rs, rsw, rzl, rzu, 
                     px, pz, pzw, ps, psw, pzl, pzu,
                     xt1, wt);

  // Get the size of the limited-memory BFGS subspace
  ParOptScalar b0;
  const ParOptScalar *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (use_qn){
    size = qn->getCompactMat(&b0, &d, &M, &Z);
  }

  if (size > 0){
    // dz = Z^{T}*px
    px->mdot(Z, size, zt);
    
    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one, 
                 Ce, &size, cpiv, zt, &size, &info);
    
    // Compute rx = Z^{T}*dz
    xt1->zeroEntries();
    for ( int i = 0; i < size; i++ ){
      xt1->axpy(zt[i], Z[i]);
    }
    
    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(xt1, rx, rc, rcw, rs, rsw, rzl, rzu,
                       xt2, wt);

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
    }
  }

  return niters;
}

/*
  Check that the gradients match along a projected direction.
*/
void ParOpt::checkGradients( double dh ){
  // Evaluate the objective/constraint and gradients
  prob->evalObjCon(x, &fobj, c);
  prob->evalObjConGradient(x, g, Ac);
  
  ParOptScalar *pxvals, *gvals;
  px->getArray(&pxvals);
  g->getArray(&gvals);
  for ( int i = 0; i < nvars; i++ ){
    if (RealPart(gvals[i]) >= 0.0){
      pxvals[i] = 1.0;
    }
    else {
      pxvals[i] = -1.0;
    }
  }
  
  // Compute the projected derivative
  ParOptScalar pobj = g->dot(px);
  px->mdot(Ac, ncon, rs);

  // Set the step direction in the sparse Lagrange multipliers
  // to an initial vector
  if (nwcon > 0){
    ParOptScalar *pzwvals;
    pzw->getArray(&pzwvals);

    // Set a value for the pzw array
    for ( int i = 0; i < nwcon; i++ ){
      pzwvals[i] = 1.05 + 0.25*(i % 21);
    }
  }

  // Evaluate the Hessian-vector product
  ParOptVec *hvec = NULL;
  if (use_hvec_product){
    for ( int i = 0; i < ncon; i++ ){
      ztemp[i] = 2.3 - 0.15*(i % 5);
    }

    // Add the contribution to gradient of the Lagrangian 
    // from the sparse constraints
    prob->addSparseJacobianTranspose(-1.0, x, pzw, g);

    for ( int i = 0; i < ncon; i++ ){
      g->axpy(-ztemp[i], Ac[i]);
    }

    // Evaluate the Hessian-vector product
    hvec = prob->createDesignVec();
    prob->evalHvecProduct(x, ztemp, pzw, px, hvec);
  
    // Check that multiple calls to the Hvec code
    // produce the same result
    prob->evalHvecProduct(x, ztemp, pzw, px, rx);

    int rank;
    MPI_Comm_rank(comm, &rank);
    rx->axpy(-1.0, hvec);
    double diff_nrm = rx->norm();

    if (rank == opt_root){
      printf("Hvec code reproducibility test\n");
      printf("Difference between multiple calls: %15.8e\n\n", diff_nrm);
    }
  }

  // Compute the point xt = x + dh*px
  ParOptVec *xt = y_qn;
  xt->copyValues(x);

#ifdef PAROPT_USE_COMPLEX
  xt->axpy(ParOptScalar(0.0, dh), px);
#else
  xt->axpy(dh, px);
#endif // PAROPT_USE_COMPLEX

  // Compute the finite-difference product
  ParOptScalar fobj2;
  prob->evalObjCon(xt, &fobj2, rc);

#ifdef PAROPT_USE_COMPLEX
  ParOptScalar pfd = ImagPart(fobj2)/dh;
#else
  ParOptScalar pfd = (fobj2 - fobj)/dh;
#endif // PAROPT_USE_COMPLEX

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    printf("Objective gradient test\n");
    printf("Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
           RealPart(pfd), RealPart(pobj), 
           fabs(RealPart(pobj - pfd)), fabs(RealPart((pobj - pfd)/pfd)));

    printf("\nConstraint gradient test\n");
    for ( int i = 0; i < ncon; i++ ){
#ifdef PAROPT_USE_COMPLEX
      ParOptScalar fd = ImagPart(rc[i])/dh;
#else
      ParOptScalar fd = (rc[i] - c[i])/dh;
#endif // PAROPT_USE_COMPLEX

      printf("Con[%3d]  FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
             i, RealPart(fd), RealPart(rs[i]), 
             fabs(RealPart(fd - rs[i])), fabs(RealPart((fd - rs[i])/fd)));
    }
  }

  if (use_hvec_product){
    // Evaluate the objective/constraints
    ParOptVec *g2 = prob->createDesignVec();
    ParOptVec **Ac2 = new ParOptVec*[ ncon ];
    for ( int i = 0; i < ncon; i++ ){
      Ac2[i] = prob->createDesignVec();
    }
    
    // Evaluate the gradient at the perturbed point and add the
    // contribution from the sparse constraints to the Hessian
    prob->evalObjConGradient(xt, g2, Ac2);
    prob->addSparseJacobianTranspose(-1.0, xt, pzw, g2);

    // Add the contribution from the dense constraints
    for ( int i = 0; i < ncon; i++ ){
      g2->axpy(-ztemp[i], Ac2[i]);
    }

#ifdef PAROPT_USE_COMPLEX
    // Evaluate the real part
    ParOptScalar *gvals;
    int gsize = g2->getArray(&gvals);

    for ( int i = 0; i < gsize; i++ ){
      gvals[i] = ImagPart(gvals[i])/dh;
    }
#else
    // Compute the difference
    g2->axpy(-1.0, g);
    g2->scale(1.0/dh);
#endif // TACS_USE_COMPLEX

    // Compute the norm of the finite-difference approximation and
    // actual Hessian-vector products
    double fdnorm = g2->norm();
    double hnorm = hvec->norm();

    // Compute the max error between the two
    g2->axpy(-1.0, hvec);
    double herr = g2->norm();

    if (rank == opt_root){
      printf("\nHessian product test\n");
      printf("Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
             fdnorm, hnorm, herr, herr/hnorm);
    }

    // Clean up the allocated data
    delete hvec;
    delete g2;
    for ( int i = 0; i < ncon; i++ ){
      delete Ac2[i];
    }
    delete [] Ac2;

    hvec = NULL;
  }

  // Now, perform a check of the sparse constraints (if any)
  if (nwcon > 0){
    // Check that the Jacobian is the derivative of the constraints
    prob->evalSparseCon(x, rsw);
    x->axpy(dh, px);
    prob->evalSparseCon(x, rcw);
    x->axpy(-dh, px);

    // Compute rcw = (cw(x + dh*px) - cw(x))/dh
    rcw->axpy(-1.0, rsw);
    rcw->scale(1.0/dh);

    // Compute the Jacobian-vector product
    rsw->zeroEntries();
    prob->addSparseJacobian(1.0, x, px, rsw);

    // Compute the difference between the vectors
    rsw->axpy(-1.0, rcw);

    // Compute the relative difference
    double cw_error = rsw->maxabs();

    if (rank == opt_root){
      printf("\nSparse constraint checks\n");
      printf("||(cw(x + h*px) - cw(x))/h - J(x)*px||: %8.2e\n", cw_error);
    }

    // Check the that the matrix-multiplication and its transpose are
    // equivalent by computing the inner product with two vectors
    // from either side
    rsw->zeroEntries();
    prob->addSparseJacobian(1.0, x, px, rsw);

    rx->zeroEntries();
    prob->addSparseJacobianTranspose(1.0, x, pzw, rx);

    ParOptScalar d1 = rsw->dot(pzw);
    ParOptScalar d2 = rx->dot(px);

    if (rank == opt_root){
      printf("\nTranspose-equivalence\n");
      printf("x^{T}*(J(x)*p): %8.2e  p*(J(x)^{T}*x): %8.2e  Err: %8.2e  Rel Err: %8.2e\n",
             RealPart(d1), RealPart(d2), 
             fabs(RealPart(d1 - d2)), fabs(RealPart((d1 - d2)/d2)));
    }

    // Set Cvec to something more-or-less random
    ParOptScalar *cvals, *rxvals;
    Cvec->getArray(&cvals);
    rx->getArray(&rxvals);
    for ( int i = 0; i < nvars; i++ ){
      cvals[i] = 0.05 + 0.25*(i % 37);
    }

    // Check the inner product pzw^{T}*J(x)*cvec*J(x)^{T}*pzw against the 
    // matrix Cw
    memset(Cw, 0, nwcon*(nwblock+1)/2*sizeof(ParOptScalar));
    prob->addSparseInnerProduct(1.0, x, Cvec, Cw);

    // Compute the vector product using the Jacobians
    rx->zeroEntries();
    prob->addSparseJacobianTranspose(1.0, x, pzw, rx);

    // Multiply component-wise
    for ( int i = 0; i < nvars; i++ ){
      rxvals[i] *= cvals[i];
    }
    rcw->zeroEntries();
    prob->addSparseJacobian(1.0, x, rx, rcw);
    d1 = rcw->dot(pzw);

    d2 = 0.0;
    ParOptScalar *cw = Cw;
    const int incr = ((nwblock+1)*nwblock)/2;

    ParOptScalar *pzwvals;
    pzw->getArray(&pzwvals);

    // Iterate over each block matrix
    for ( int i = 0; i < nwcon; i += nwblock ){
      // Index into each block
      for ( int j = 0; j < nwblock; j++ ){
        for ( int k = 0; k < j; k++ ){
          d2 += 2.0*cw[0]*pzwvals[i+j]*pzwvals[i+k];
          cw++;
        }

        d2 += cw[0]*pzwvals[i+j]*pzwvals[i+j];
        cw++;
      }
    }

    // Add the result across all processors
    ParOptScalar temp = d2;
    MPI_Reduce(&temp, &d2, 1, PAROPT_MPI_TYPE, MPI_SUM, opt_root, comm);

    if (rank == opt_root){
      printf("\nJ(x)*C^{-1}*J(x)^{T} test: \n");
      printf("Product: %8.2e  Matrix: %8.2e  Err: %8.2e  Rel Err: %8.2e\n",
             RealPart(d1), RealPart(d2), 
             fabs(RealPart(d1 - d2)), fabs(RealPart((d1 - d2)/d2)));
    }
  }
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
void ParOpt::checkKKTStep( int is_newton ){
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
    printf("Residual step check:\n");
  }

  // Check the first residual equation
  if (is_newton){
    prob->evalHvecProduct(x, z, zw, px, rx);
  }
  else {
    if (!sequential_linear_method){
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
    printf("max |(H + sigma*I)*px - Ac^{T}*pz - Aw^{T}*pzw - pzl + pzu + \
(g - Ac^{T}*z - Aw^{T}*zw - zl + zu)|: %10.4e\n", max_val);
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
      val = rc[i] - ps[i] + (c[i] - s[i]);
    }
    if (fabs(RealPart(val)) > max_val){
      max_val = fabs(RealPart(val));
    }
  }
  if (rank == opt_root){
    printf("max |A*px - ps + (c - s)|: %10.4e\n", max_val);
  }

  // Find the maximum value of the residual equations for
  // the dual slack variables
  max_val = 0.0;
  if (dense_inequality){
    for ( int i = 0; i < ncon; i++ ){
      ParOptScalar val = z[i]*ps[i] + s[i]*pz[i] + (z[i]*s[i] - barrier_param);
      if (fabs(RealPart(val)) > max_val){
        max_val = fabs(RealPart(val));
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
      if (RealPart(lbvals[i]) > -max_bound_val){
        ParOptScalar val = 
          (zlvals[i]*pxvals[i] + (xvals[i] - lbvals[i])*pzlvals[i] +
           (zlvals[i]*(xvals[i] - lbvals[i]) - barrier_param));
        if (fabs(RealPart(val)) > max_val){
          max_val = fabs(RealPart(val));
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
      if (RealPart(ubvals[i]) < max_bound_val){
        ParOptScalar val = 
          (-zuvals[i]*pxvals[i] + (ubvals[i] - xvals[i])*pzuvals[i] +
           (zuvals[i]*(ubvals[i] - xvals[i]) - barrier_param));
        if (fabs(RealPart(val)) > max_val){
          max_val = fabs(RealPart(val));
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
