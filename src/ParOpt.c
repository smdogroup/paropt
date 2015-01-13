#include <math.h>
#include <string.h>
#include "ParOpt.h"
#include "ParOptBlasLapack.h"

/*
  Copyright (c) 2014 Graeme Kennedy. All rights reserved
*/

/*
  The Parallel Optimizer constructor

  This function allocates and initializes the data that is required
  for parallel optimization. This includes initialization of the
  variables, allocation of the matrices and the BFGS approximate
  Hessian. This code also sets the default parameters for
  optimization. These parameters can be modified through member
  functions.

  The parameters are as follows:

  max_major_iters:        maximum major iterations
  init_starting_point:    (boolean) guess the initial multipliers
  write_output_freq:      the major iter frequency for output
  barrier_param:          the initial barrier parameter
  abs_res_tol:            the absolute residual stopping criterion 
  use_line_search:        (boolean) use/don't use the line search
  penalty_descent_frac:   parameter to ensure sufficient descent
  armijio_constant:       line search sufficient decrease parameter
  monotone_barrier_frac:  decrease the barrier by this fraction
  monotone_barrier_power: decrease the barrier by mu**power
  min_frac_to_boundary:   minimum fraction-to-boundary constant
  major_iter_step_check:  check the step at this major iteration
  hessian_reset_freq:     reset the Hessian at this frequency

  input:
  prob:      the optimization problem
  pcon:      the constraints (if any)
  max_lbfgs: the number of steps to store in the the l-BFGS
*/
ParOpt::ParOpt( ParOptProblem *_prob, 
		int max_lbfgs_subspace ){
  prob = _prob;

  // Record the communicator
  comm = prob->getMPIComm();
  opt_root = 0;

  // Get the number of variables/constraints
  prob->getProblemSizes(&nvars, &ncon, &nwcon, &nwblock);

  // Are these sparse inequalties or equalities?
  sparse_inequality = prob->isSparseInequality();

  // Assign the values from the sparsity constraints
  if (nwcon > 0 && nwcon % nwblock != 0){
    fprintf(stderr, "Weighted block size inconsistent\n");
  }

  // Calculate the total number of variable across all processors
  MPI_Allreduce(&nvars, &nvars_total, 1, MPI_INT, MPI_SUM, comm);

  // Allocate the quasi-Newton LBFGS approximation
  qn = new LBFGS(comm, nvars, max_lbfgs_subspace);

  // Set the values of the variables/bounds
  x = new ParOptVec(comm, nvars);
  lb = new ParOptVec(comm, nvars);
  ub = new ParOptVec(comm, nvars);
  prob->getVarsAndBounds(x, lb, ub);

  // Check the design variables and bounds, move things that 
  // don't make sense and print some warnings
  double *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Check the variable values to see if they are reasonable
  double rel_bound = 1e-3;
  int check_flag = 0;
  for ( int i = 0; i < nvars; i++ ){
    // Fixed variables are not allowed
    if (lbvals[i] >= ubvals[i]){
      check_flag = (check_flag | 1);

      // Make up bounds
      lbvals[i] = 0.5*(lbvals[i] + ubvals[i]) - 0.5*rel_bound;
      ubvals[i] = lbvals[i] + rel_bound;
    }

    // Check if x is too close the boundary
    if (xvals[i] < lbvals[i] + rel_bound*(ubvals[i] - lbvals[i])){
      check_flag = (check_flag | 2);
      xvals[i] = lbvals[i] + rel_bound*(ubvals[i] - lbvals[i]);
    }
    if (xvals[i] > ubvals[i] - rel_bound*(ubvals[i] - lbvals[i])){
      check_flag = (check_flag | 4);
      xvals[i] = ubvals[i] - rel_bound*(ubvals[i] - lbvals[i]);
    }
  }

  // Print the results of the warnings
  if (check_flag & 1){
    fprintf(stderr, "Warning: Variable bounds are inconsistent\n");
  }
  if (check_flag & 2){
    fprintf(stderr, 
	    "Warning: Modification of variables; too close to lower bound\n");
  }
  if (check_flag & 4){
    fprintf(stderr, 
	    "Warning: Modification of variables; too close to upper bound\n");
  }
  
  // Allocate storage space for the variables etc.
  zl = new ParOptVec(comm, nvars);
  zu = new ParOptVec(comm, nvars);
  zl->set(1.0);
  zu->set(1.0);

  zw = new ParOptVec(comm, nwcon);
  sw = new ParOptVec(comm, nwcon);
  zw->set(1.0);
  sw->set(1.0);

  // Set the initial values of the Lagrange multipliers
  z = new double[ ncon ];
  s = new double[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    z[i] = 1.0;
    s[i] = 1.0;
  }

  // Allocate space for the steps
  px = new ParOptVec(comm, nvars);
  pzl = new ParOptVec(comm, nvars);
  pzu = new ParOptVec(comm, nvars);
  pz = new double[ ncon ];
  ps = new double[ ncon ];
  pzw = new ParOptVec(comm, nwcon);
  psw = new ParOptVec(comm, nwcon);

  // Allocate space for the residuals
  rx = new ParOptVec(comm, nvars);
  rzl = new ParOptVec(comm, nvars);
  rzu = new ParOptVec(comm, nvars);
  rc = new double[ ncon ];
  rs = new double[ ncon ];
  rcw = new ParOptVec(comm, nwcon);
  rsw = new ParOptVec(comm, nwcon);

  // Allocate space for the Quasi-Newton updates
  y_qn = new ParOptVec(comm, nvars);
  s_qn = new ParOptVec(comm, nvars);

  // Allocate vectors for the weighting constraints
  wtemp = new ParOptVec(comm, nwcon);

  // Allocate space for the block-diagonal matrix
  Cw = new double[ nwcon*(nwblock+1)/2 ];

  // Allocate space for off-diagonal entries
  Ew = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ew[i] = new ParOptVec(comm, nwcon);
  }

  // Allocate storage for bfgs/constraint sized things
  int zsize = 2*max_lbfgs_subspace;
  if (ncon > zsize){
    ncon = zsize;
  }
  ztemp = new double[ zsize ];

  // Allocate space for the Dmatrix
  Dmat = new double[ ncon*ncon ];
  dpiv = new int[ ncon ];

  // Allocate space for the Ce matrix
  Ce = new double[ 4*max_lbfgs_subspace*max_lbfgs_subspace ];
  cpiv = new int[ 2*max_lbfgs_subspace ];

  // Allocate space for the diagonal matrix components
  Cvec = new ParOptVec(comm, nvars);

  // Set the value of the objective
  fobj = 0.0;
  
  // Set the constraints to zero
  c = new double[ ncon ];
  memset(c, 0, ncon*sizeof(double));
  
  // Set the objective and constraint gradients 
  g = new ParOptVec(comm, nvars);
  Ac = new ParOptVec*[ ncon ];
  for ( int i = 0; i < ncon; i++ ){
    Ac[i] = new ParOptVec(comm, nvars);
  }

  // Initialize the parameters with default values
  max_major_iters = 1000;
  init_starting_point = 1;
  barrier_param = 0.1;
  abs_res_tol = 1e-5;
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
  major_iter_step_check = -1;
  sequential_linear_method = 0;
  hessian_reset_freq = 100000000;
  merit_func_check_epsilon = 1e-6;

  // Initialize the Hessian-vector product information
  use_hvec_product = 0;
  gmres_switch_tol = 1e-3;
  gmres_rtol = 0.1;
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
  Write out all of the design variables, Lagrange multipliers and
  slack variables to a binary file.
*/
int ParOpt::writeSolutionFile( const char * filename ){
  char * fname = new char[ strlen(filename)+1 ];
  strcpy(fname, filename);

  int fail = 1;
  MPI_File fp = NULL;
  MPI_File_open(comm, fname, MPI_MODE_WRONLY | MPI_MODE_CREATE, 
                MPI_INFO_NULL, &fp);

  if (fp){
    // Successfull opened the file
    fail = 0;

    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Allocate space to store the variable ranges
    int *var_range = new int[ size+1 ];
    var_range[0] = 0;

    int *nwcon_range = new int[ size+1 ];
    nwcon_range[0] = 0;
    
    // Count up the displacements/variable ranges
    MPI_Allgather(&nvars, 1, MPI_INT, &var_range[1], 1, MPI_INT, comm);
    MPI_Allgather(&nwcon, 1, MPI_INT, &nwcon_range[1], 1, MPI_INT, comm);

    for ( int k = 0; k < size; k++ ){
      var_range[k+1] += var_range[k];
      nwcon_range[k+1] += nwcon_range[k];
    }

    // Print out the problem sizes on the root processor
    if (rank == opt_root){
      int var_sizes[3];
      var_sizes[0] = var_range[size];
      var_sizes[1] = nwcon_range[size];
      var_sizes[2] = ncon;

      MPI_File_write(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);
      MPI_File_write(fp, &barrier_param, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, z, ncon, MPI_DOUBLE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, s, ncon, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    size_t offset = 3*sizeof(int) + (2*ncon+1)*sizeof(double);

    // Use the native representation for the data
    char datarep[] = "native";

    // Extract the design variables 
    double *xvals;
    int xsize = x->getArray(&xvals);
    MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
                      datarep, MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], xvals, xsize, MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(double);

    // Extract the lower Lagrange multipliers
    double *zlvals, *zuvals;
    zl->getArray(&zlvals);
    zu->getArray(&zuvals);
    MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
                      datarep, MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], zlvals, xsize, MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(double);

    // Write out the upper Lagrange multipliers
    MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
                      datarep, MPI_INFO_NULL);
    MPI_File_write_at_all(fp, var_range[rank], zuvals, xsize, MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(double);
    
    // Write out the extra constraint bounds
    if (nwcon_range[size] > 0){
      double *zwvals, *swvals;
      int nwsize = zw->getArray(&zwvals);
      sw->getArray(&swvals);
      MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
			datarep, MPI_INFO_NULL);
      MPI_File_write_at_all(fp, nwcon_range[rank], zwvals, nwsize, MPI_DOUBLE,
			    MPI_STATUS_IGNORE);
      offset += nwcon_range[size]*sizeof(double);

      MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
			datarep, MPI_INFO_NULL);
      MPI_File_write_at_all(fp, nwcon_range[rank], swvals, nwsize, MPI_DOUBLE,
			    MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fp);

    delete [] var_range;
    delete [] nwcon_range;
  }

  delete [] fname;

  return fail;
}

/*
  Read in the design variables, lagrange multipliers and slack
  variables from a binary file
*/
int ParOpt::readSolutionFile( const char * filename ){
  char * fname = new char[ strlen(filename)+1 ];
  strcpy(fname, filename);

  int fail = 1;
  MPI_File fp = NULL;
  MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
  delete [] fname;

  if (fp){
    // Successfully opened the file for reading
    fail = 0;

    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Allocate space to store the variable ranges
    int *var_range = new int[ size+1 ];
    var_range[0] = 0;

    int *nwcon_range = new int[ size+1 ];
    nwcon_range[0] = 0;
    
    // Count up the displacements/variable ranges
    MPI_Allgather(&nvars, 1, MPI_INT, &var_range[1], 1, MPI_INT, comm);
    MPI_Allgather(&nwcon, 1, MPI_INT, &nwcon_range[1], 1, MPI_INT, comm);

    for ( int k = 0; k < size; k++ ){
      var_range[k+1] += var_range[k];
      nwcon_range[k+1] += nwcon_range[k];
    }

    int size_fail = 0;

    // Read in the sizes
    if (rank == opt_root){
      int var_sizes[3];
      MPI_File_read(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);

      if (var_sizes[0] != var_range[size] ||
	  var_sizes[1] != nwcon_range[size] ||
	  var_sizes[2] != ncon){
	size_fail = 1;
      }

      if (!size_fail){
	MPI_File_read(fp, &barrier_param, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_File_read(fp, z, ncon, MPI_DOUBLE, MPI_STATUS_IGNORE);
	MPI_File_read(fp, s, ncon, MPI_DOUBLE, MPI_STATUS_IGNORE);
      }
    }
    MPI_Bcast(&size_fail, 1, MPI_INT, opt_root, comm);

    // The problem sizes are inconsistent, return
    if (size_fail){
      fail = 1;
      if (rank == opt_root){
	fprintf(stderr, "ParOpt: Problem size incompatible with solution file\n");
      }

      delete [] var_range;
      delete [] nwcon_range;

      MPI_File_close(&fp);
      return fail;
    }

    // Set the initial offset
    size_t offset = 3*sizeof(int) + (2*ncon+1)*sizeof(double);

    // Use the native representation for the data
    char datarep[] = "native";

    // Extract the design variables 
    double *xvals;
    int xsize = x->getArray(&xvals);
    MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
                      datarep, MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], xvals, xsize, MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(double);

    // Extract the lower Lagrange multipliers
    double *zlvals, *zuvals;
    zl->getArray(&zlvals);
    zu->getArray(&zuvals);
    MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
                      datarep, MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], zlvals, xsize, MPI_DOUBLE,
			 MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(double);

    // Read in the upper Lagrange multipliers
    MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
                      datarep, MPI_INFO_NULL);
    MPI_File_read_at_all(fp, var_range[rank], zuvals, xsize, MPI_DOUBLE,
			 MPI_STATUS_IGNORE);
    offset += var_range[size]*sizeof(double);
    
    // Read in the extra constraint Lagrange multipliers
    if (nwcon_range[size] > 0){
      double *zwvals, *swvals;
      int nwsize = zw->getArray(&zwvals);
      sw->getArray(&swvals);
      MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
			datarep, MPI_INFO_NULL);
      MPI_File_read_at_all(fp, nwcon_range[rank], zwvals, nwsize, MPI_DOUBLE,
			   MPI_STATUS_IGNORE);
      offset += nwcon_range[size]*sizeof(double);

      MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
			datarep, MPI_INFO_NULL);
      MPI_File_read_at_all(fp, nwcon_range[rank], swvals, nwsize, MPI_DOUBLE,
			   MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fp);

    delete [] var_range;
    delete [] nwcon_range;
  }

  return fail;
}

/*
  Set optimizer parameters
*/
void ParOpt::setInitStartingPoint( int init ){
  init_starting_point = init;
}

void ParOpt::setMaxMajorIterations( int iters ){
  if (iters > 1){ max_major_iters = iters; }
}

void ParOpt::setAbsOptimalityTol( double tol ){
  if (tol < 1e-2 && tol > 0.0){
    abs_res_tol = tol;
  }
}

void ParOpt::setInitBarrierParameter( double mu ){
  if (mu > 0.0){ barrier_param = mu; }
}

void ParOpt::setBarrierFraction( double frac ){
  if (frac > 0.0 && frac < 1.0){
    monotone_barrier_fraction = frac;
  }
}

void ParOpt::setBarrierPower( double power ){
  if (power > 1.0 && power < 2.0){
    monotone_barrier_power = power;
  }
}

void ParOpt::setHessianResetFreq( int freq ){
  if (freq > 0){
    hessian_reset_freq = freq;
  }
}

/*
  Set parameters associated with the line search
*/
void ParOpt::setUseLineSearch( int truth ){
  use_line_search = truth;
}

void ParOpt::setMaxLineSearchIters( int iters ){
  if (iters > 0){ max_line_iters = iters; }
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

/*
  Set the flag for whether to use the Hessian-vector products or not
*/
void ParOpt::setUseHvecProduct( int truth ){
  use_hvec_product = truth;
}

/*
  Set information about GMRES
*/
void ParOpt::setGMRESSwitchTolerance( double tol ){
  gmres_switch_tol = tol;
}

void ParOpt::setGMRESTolerances( double rtol, double atol ){
  gmres_rtol = rtol;
  gmres_atol = atol;
}

void ParOpt::setGMRESSusbspaceSize( int m ){
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
    
    gmres_H = new double[ (m+1)*(m+2)/2 ];
    gmres_alpha = new double[ m+1 ];
    gmres_res = new double[ m+1 ];
    gmres_Q = new double[ 2*m ];
    
    gmres_W = new ParOptVec*[ m+1 ];
    for ( int i = 0; i < m+1; i++ ){
      gmres_W[i] = new ParOptVec(comm, nvars);
    }
  }
  else {
    gmres_subspace_size = 0;
  }
}

/*
  Set the file to use
*/
void ParOpt::setOutputFile( const char * filename ){
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

/*
  Compute the residual of the KKT system. This code utilizes the data
  stored internally in the ParOpt optimizer.

  This code computes the following terms:

  rx  = -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu) 
  rc  = -(c(x) - s)
  rcw  = -(cw(x) - sw)
  rz  = -(S*z - mu*e) 
  rzu = -((x - xl)*zl - mu*e)
  rzl = -((ub - x)*zu - mu*e)
*/
void ParOpt::computeKKTRes( double * max_prime,
			    double * max_dual, 
			    double * max_infeas ){
  // Zero the values of the maximum residuals 
  *max_prime = 0.0;
  *max_dual = 0.0;
  *max_infeas = 0.0;

  // Assemble the negative of the residual of the first KKT equation:
  // -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu)
  rx->copyValues(zl);
  rx->axpy(-1.0, zu);
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

  for ( int i = 0; i < ncon; i++ ){
    rc[i] = -(c[i] - s[i]);
    rs[i] = -(s[i]*z[i] - barrier_param);

    if (fabs(rc[i]) > *max_infeas){
      *max_infeas = fabs(rc[i]);
    }
    if (fabs(rs[i]) > *max_dual){
      *max_dual = fabs(rs[i]);
    }
  }

  // Extract the values of the variables and lower/upper bounds
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Extract the values of the residuals
  double *rzlvals, *rzuvals;
  rzl->getArray(&rzlvals);
  rzu->getArray(&rzuvals);

  for ( int i = 0; i < nvars; i++ ){
    rzlvals[i] = -((xvals[i] - lbvals[i])*zlvals[i] - barrier_param);
    rzuvals[i] = -((ubvals[i] - xvals[i])*zuvals[i] - barrier_param);
  }

  // Compute the duality errors from the upper/lower bounds
  double dual_zl = rzl->maxabs();
  double dual_zu = rzu->maxabs();
  if (dual_zl > *max_dual){
    *max_dual = dual_zl;
  }
  if (dual_zu > *max_dual){
    *max_dual = dual_zu;
  }

  if (nwcon > 0 && sparse_inequality){
    // Set the values of the perturbed complementarity
    // constraints for the sparse slack variables
    double *zwvals, *swvals, *rswvals;
    zw->getArray(&zwvals);
    sw->getArray(&swvals);
    rsw->getArray(&rswvals);
    
    for ( int i = 0; i < nwcon; i++ ){
      rswvals[i] = -(swvals[i]*zwvals[i] - barrier_param);
    }
    
    double dual_zw = rsw->maxabs();
    if (dual_zw > *max_dual){
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
    double *cw = Cw;
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
  double *rhs;
  vec->getArray(&rhs);
  
  if (nwblock == 1){
    for ( int i = 0; i < nwcon; i++ ){
      rhs[i] *= Cw[i];
    }
  }
  else {
    double *cw = Cw;
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
void ParOpt::setUpKKTDiagSystem( ParOptVec * xt,
				 ParOptVec * wt ){
  // Retrive the diagonal entry for the BFGS update
  double b0 = 0.0;
  if (!sequential_linear_method){
    const double *d, *M;
    ParOptVec **Z;
    qn->getLBFGSMat(&b0, &d, &M, &Z);
  }

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Set the components of the diagonal matrix 
  double *cvals;
  Cvec->getArray(&cvals);

  // Set the values of the c matrix
  for ( int i = 0; i < nvars; i++ ){
    cvals[i] = 1.0/(b0 + 
		    zlvals[i]/(xvals[i] - lbvals[i]) + 
		    zuvals[i]/(ubvals[i] - xvals[i]));
  }

  if (nwcon > 0){
    // Set the values in the Cw diagonal matrix
    memset(Cw, 0, nwcon*(nwblock+1)/2*sizeof(double));
    
    // Compute Cw = Zw^{-1}*Sw + Aw*C^{-1}*Aw
    // First compute Cw = Zw^{-1}*Sw
    if (sparse_inequality){
      double *swvals, *zwvals;
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
	double *cw = Cw;
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

    // Next, complete the evaluation of Cw by 
    // add the following contribution to the matrix
    // Cw += Aw*C^{-1}*Aw^{T}
    prob->addSparseInnerProduct(1.0, x, Cvec, Cw);

    // Factor the Cw matrix
    factorCw();
    
    // Compute Ew = Aw*C^{-1}*A
    for ( int k = 0; k < ncon; k++ ){
      double *avals, *xvals;
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
  memset(Dmat, 0, ncon*ncon*sizeof(double));

  if (nwcon > 0){
    // Add the term Dw = - Ew^{T}*Cw^{-1}*Ew to the Dmat matrix first
    // by computing the inner product with Cw^{-1}
    for ( int j = 0; j < ncon; j++ ){
      // Apply Cw^{-1}*Ew[j] -> wt
      wt->copyValues(Ew[j]);
      applyCwFactor(wt);

      for ( int i = j; i < ncon; i++ ){
	// Get the vectors required
	double *wvals, *ewivals;
	Ew[i]->getArray(&ewivals);
	wt->getArray(&wvals);

	double dmat = 0.0;
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
      double *aivals, *ajvals;
      Cvec->getArray(&cvals);
      Ac[i]->getArray(&aivals);
      Ac[j]->getArray(&ajvals);

      double dmat = 0.0;
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

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Reduce the result to the root processor
  if (rank == opt_root){
    MPI_Reduce(MPI_IN_PLACE, Dmat, ncon*ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  else {
    MPI_Reduce(Dmat, NULL, ncon*ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  
  // Add the diagonal component to the matrix
  if (rank == opt_root){
    for ( int i = 0; i < ncon; i++ ){
      Dmat[i*(ncon + 1)] += s[i]/z[i];
    }
  }

  // Broadcast the result to all processors. Note that this ensures
  // that the factorization will be the same on all processors
  MPI_Bcast(Dmat, ncon*ncon, MPI_DOUBLE, opt_root, comm);

  // Factor the matrix for future use
  int info = 0;
  LAPACKdgetrf(&ncon, &ncon, Dmat, &ncon, dpiv, &info);
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
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, double *bc, 
				 ParOptVec *bcw, double *bs,
				 ParOptVec *bsw,
				 ParOptVec *bzl, ParOptVec *bzu,
				 ParOptVec *yx, double *yz, 
				 ParOptVec *yzw, double *ys,
				 ParOptVec *ysw,
				 ParOptVec *yzl, ParOptVec *yzu,
				 ParOptVec *xt, ParOptVec *wt ){
  // Get the arrays for the variables and upper/lower bounds
  double *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Get the arrays for the right-hand-sides
  double *bxvals, *bzlvals, *bzuvals;
  bx->getArray(&bxvals);
  bzl->getArray(&bzlvals);
  bzu->getArray(&bzuvals);

  // Compute xt = C^{-1}*d = 
  // C^{-1}*(bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu)
  double *dvals, *cvals;
  xt->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*(bxvals[i] +
			 bzlvals[i]/(xvals[i] - lbvals[i]) - 
			 bzuvals[i]/(ubvals[i] - xvals[i]));
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    wt->copyValues(bcw);

    if (sparse_inequality){
      // Add wt += Zw^{-1}*bsw
      double *wvals, *bswvals, *zwvals;
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
  memset(yz, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wt->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    double *avals;
    xt->getArray(&dvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
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
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == opt_root){
    // Reduce the result to the root processor
    MPI_Reduce(MPI_IN_PLACE, yz, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  else {
    MPI_Reduce(yz, NULL, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }

  // Compute the full right-hand-side
  if (rank == opt_root){
    // Compute the full right-hand-side on the root processor
    // and solve for the Lagrange multipliers
    for ( int i = 0; i < ncon; i++ ){
      yz[i] = bc[i] + bs[i]/z[i] - yz[i];
    }

    int one = 1, info = 0;
    LAPACKdgetrs("N", &ncon, &one, 
		 Dmat, &ncon, dpiv, yz, &ncon, &info);
  }

  MPI_Bcast(yz, ncon, MPI_DOUBLE, opt_root, comm);

  // Compute the step in the slack variables 
  for ( int i = 0; i < ncon; i++ ){
    ys[i] = (bs[i] - s[i]*yz[i])/z[i];
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
      double *yzwvals, *zwvals, *bswvals;
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
      double *zwvals, *swvals;
      zw->getArray(&zwvals);
      sw->getArray(&swvals);

      double *yzwvals, *yswvals, *bswvals;
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
  double *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xt);

  // Retrieve the lagrange multipliers
  double *zlvals, *zuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the lagrange multiplier update vectors
  double *yzlvals, *yzuvals;
  yzl->getArray(&yzlvals);
  yzu->getArray(&yzuvals);
   
  // Compute the steps in the bound Lagrange multipliers
  for ( int i = 0; i < nvars; i++ ){
    yzlvals[i] = (bzlvals[i] - zlvals[i]*yxvals[i])/(xvals[i] - lbvals[i]);
    yzuvals[i] = (bzuvals[i] + zuvals[i]*yxvals[i])/(ubvals[i] - xvals[i]);
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
				 ParOptVec *yx, double *yz, 
				 ParOptVec *yzw, double *ys,
				 ParOptVec *ysw,
				 ParOptVec *yzl, ParOptVec *yzu,
				 ParOptVec *xt, ParOptVec *wt ){
  // Compute the terms from the weighting constraints
  // Compute xt = C^{-1}*bx
  double *bxvals, *dvals, *cvals;
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
  memset(yz, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wt->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] += BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    double *avals;
    xt->getArray(&dvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
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

  // Reduce the result to the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    MPI_Reduce(MPI_IN_PLACE, yz, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  else {
    MPI_Reduce(yz, NULL, ncon, MPI_DOUBLE, MPI_SUM, 
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

  MPI_Bcast(yz, ncon, MPI_DOUBLE, opt_root, comm);

  // Compute the step in the slack variables 
  for ( int i = 0; i < ncon; i++ ){
    ys[i] = -(s[i]*yz[i])/z[i];
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
      double *zwvals, *swvals;
      zw->getArray(&zwvals);
      sw->getArray(&swvals);

      double *yzwvals, *yswvals;
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
  double *yxvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Complete the result yx = C^{-1}*d + C^{-1}*(A^{T}*yz + Aw^{T}*yzw)
  yx->axpy(1.0, xt);

  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the right-hand-sides and the solution vectors
  double *yzlvals, *yzuvals;
  yzl->getArray(&yzlvals);
  yzu->getArray(&yzuvals);
   
  // Compute the steps in the bound Lagrange multipliers
  for ( int i = 0; i < nvars; i++ ){
    yzlvals[i] = -(zlvals[i]*yxvals[i])/(xvals[i] - lbvals[i]);
    yzuvals[i] =  (zuvals[i]*yxvals[i])/(ubvals[i] - xvals[i]);
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
				 double *zt, 
				 ParOptVec *xt, ParOptVec *wt ){
  // Compute the terms from the weighting constraints
  // Compute xt = C^{-1}*bx
  double *bxvals, *dvals, *cvals;
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
  memset(zt, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wt->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      zt[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    double *avals;
    xt->getArray(&dvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
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

  // Reduce the result to the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    MPI_Reduce(MPI_IN_PLACE, zt, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  else {
    MPI_Reduce(zt, NULL, ncon, MPI_DOUBLE, MPI_SUM, 
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

  MPI_Bcast(zt, ncon, MPI_DOUBLE, opt_root, comm);

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
  double *yxvals;
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
  includes components that are scaled by a given parameter: alpha.
*/
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, 
				 double alpha, double *bc, 
				 ParOptVec *bcw, double *bs,
				 ParOptVec *bsw,
				 ParOptVec *bzl, ParOptVec *bzu,
				 ParOptVec *yx, double *yz,
				 ParOptVec *xt, ParOptVec *wt ){
  // Get the arrays for the variables and upper/lower bounds
  double *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Get the arrays for the right-hand-sides
  double *bxvals, *bzlvals, *bzuvals;
  bx->getArray(&bxvals);
  bzl->getArray(&bzlvals);
  bzu->getArray(&bzuvals);

  // Compute xt = C^{-1}*d = 
  // C^{-1}*(bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu)
  double *dvals, *cvals;
  xt->getArray(&dvals);
  Cvec->getArray(&cvals);
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = cvals[i]*(bxvals[i] +
			 alpha*bzlvals[i]/(xvals[i] - lbvals[i]) - 
			 alpha*bzuvals[i]/(ubvals[i] - xvals[i]));
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    // Compute wt = Cw^{-1}*(bcw + Zw^{-1}*bsw - Aw*C^{-1}*d)
    wt->copyValues(bcw);
    wt->scale(alpha);
    
    if (sparse_inequality){
      // Add wt += Zw^{-1}*bsw
      double *wvals, *bswvals, *zwvals;
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
  memset(yz, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wt->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the contribution from each processor
  // to the term yz <- yz - A*C^{-1}*d
  for ( int i = 0; i < ncon; i++ ){
    double *avals;
    xt->getArray(&dvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
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
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == opt_root){
    // Reduce the result to the root processor
    MPI_Reduce(MPI_IN_PLACE, yz, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  else {
    MPI_Reduce(yz, NULL, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }

  // Compute the full right-hand-side
  if (rank == opt_root){
    // Compute the full right-hand-side on the root processor
    // and solve for the Lagrange multipliers
    for ( int i = 0; i < ncon; i++ ){
      yz[i] = alpha*(bc[i] + bs[i]/z[i]) - yz[i];
    }

    int one = 1, info = 0;
    LAPACKdgetrs("N", &ncon, &one, 
		 Dmat, &ncon, dpiv, yz, &ncon, &info);
  }

  MPI_Bcast(yz, ncon, MPI_DOUBLE, opt_root, comm);

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
      double *yzwvals, *zwvals, *bswvals;
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
  double *yxvals;
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
void ParOpt::setUpKKTSystem( double *zt,
			     ParOptVec *xt1,
			     ParOptVec *xt2, 
			     ParOptVec *wt ){
  if (!sequential_linear_method){
    // Get the size of the limited-memory BFGS subspace
    double b0;
    const double *d0, *M;
    ParOptVec **Z;
    int size = qn->getLBFGSMat(&b0, &d0, &M, &Z);
    
    if (size > 0){
      memset(Ce, 0, size*size*sizeof(double));
      
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
void ParOpt::computeKKTStep( double *zt,
			     ParOptVec *xt1, ParOptVec *xt2, 
			     ParOptVec *wt ){
  // Get the size of the limited-memory BFGS subspace
  double b0;
  const double *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (!sequential_linear_method){
    size = qn->getLBFGSMat(&b0, &d, &M, &Z);
  }

  // At this point the residuals are no longer required.
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
    
    // Add the terms from the 
    for ( int i = 0; i < ncon; i++ ){
      pz[i] -= rc[i];
      ps[i] -= rs[i];
    }
  }
}

/*
  Compute the complementarity at the current solution
*/
double ParOpt::computeComp(){
  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);
  
  // Sum up the complementarity from this processor
  double comp = 0.0;
  
  for ( int i = 0; i < nvars; i++ ){
    comp += (zlvals[i]*(xvals[i] - lbvals[i]) + 
	     zuvals[i]*(ubvals[i] - xvals[i]));
  }

  double product = 0.0;
  MPI_Reduce(&comp, &product, 1, MPI_DOUBLE, MPI_SUM, opt_root, comm);
  
  // Compute the complementarity only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  
  if (rank == opt_root){
    for ( int i = 0; i < ncon; i++ ){
      product += s[i]*z[i];
    }

    comp = product/(ncon + 2*nvars_total);
  }

  // Broadcast the result to all processors
  MPI_Bcast(&comp, 1, MPI_DOUBLE, opt_root, comm);

  return comp;
}

/*
  Compute the complementarity at the given step
*/
double ParOpt::computeCompStep( double alpha_x, double alpha_z ){
  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the values of the steps
  double *pxvals, *pzlvals, *pzuvals;
  px->getArray(&pxvals);
  pzl->getArray(&pzlvals);
  pzu->getArray(&pzuvals);
  
  // Sum up the complementarity from this processor
  double comp = 0.0;
  
  for ( int i = 0; i < nvars; i++ ){
    double xnew = xvals[i] + alpha_x*pxvals[i];
    comp += ((zlvals[i] + alpha_z*pzlvals[i])*(xnew - lbvals[i]) + 
	     (zuvals[i] + alpha_z*pzuvals[i])*(ubvals[i] - xnew));
  }

  double product = 0.0;
  MPI_Reduce(&comp, &product, 1, MPI_DOUBLE, MPI_SUM, opt_root, comm);
  
  // Compute the complementarity only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  
  if (rank == opt_root){
    for ( int i = 0; i < ncon; i++ ){
      product += (s[i] + alpha_x*ps[i])*(z[i] + alpha_z*pz[i]);
    }

    comp = product/(ncon + 2*nvars_total);
  }

  // Broadcast the result to all processors
  MPI_Bcast(&comp, 1, MPI_DOUBLE, opt_root, comm);

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
  double *xvals, *pxvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  px->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Check the design variable step
  for ( int i = 0; i < nvars; i++ ){
    if (pxvals[i] < 0.0){
      double alpha = -tau*(xvals[i] - lbvals[i])/pxvals[i];
      if (alpha < max_x){
	max_x = alpha;
      }
    }
    else if (pxvals[i] > 0.0){
      double alpha = tau*(ubvals[i] - xvals[i])/pxvals[i];
      if (alpha < max_x){
	max_x = alpha;
      }
    }
  }

  // Check the slack variable step
  for ( int i = 0; i < ncon; i++ ){
    if (ps[i] < 0.0){
      double alpha = -tau*s[i]/ps[i];
      if (alpha < max_x){
	max_x = alpha;
      }
    }
  }

  // Check the step for the Lagrange multipliers
  for ( int i = 0; i < ncon; i++ ){
    if (pz[i] < 0.0){
      double alpha = -tau*z[i]/pz[i];
      if (alpha < max_z){
	max_z = alpha;
      }
    }
  }

  // Check the Lagrange and slack variable steps for the
  // sparse inequalities if any
  if (nwcon > 0 && sparse_inequality){
    double *zwvals, *pzwvals;
    zw->getArray(&zwvals);
    pzw->getArray(&pzwvals);
    for ( int i = 0; i < nwcon; i++ ){
      if (pzwvals[i] < 0.0){
	double alpha = -tau*zwvals[i]/pzwvals[i];
	if (alpha < max_z){
	  max_z = alpha;
	}
      }
    }

    double *swvals, *pswvals;
    sw->getArray(&swvals);
    psw->getArray(&pswvals);
    for ( int i = 0; i < nwcon; i++ ){
      if (pswvals[i] < 0.0){
	double alpha = -tau*swvals[i]/pswvals[i];
	if (alpha < max_x){
	  max_x = alpha;
	}
      }
    }
  }

  // Retrieve the values of the lower/upper Lagrange multipliers
  double *zlvals, *zuvals, *pzlvals, *pzuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);
  pzl->getArray(&pzlvals);
  pzu->getArray(&pzuvals);

  // Check the step for the lower/upper Lagrange multipliers
  for ( int i = 0; i < nvars; i++ ){
    if (pzlvals[i] < 0.0){
      double alpha = -tau*zlvals[i]/pzlvals[i];
      if (alpha < max_z){
	max_z = alpha;
      }
    }
    if (pzuvals[i] < 0.0){
      double alpha = -tau*zuvals[i]/pzuvals[i];
      if (alpha < max_z){
	max_z = alpha;
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
  
  f(x + alpha*px) + 
  mu*(log(s) + log(x - xl) + log(xu - x)) +
  rho*||c(x) - s||_{2}

  output: The value of the merit function
*/
double ParOpt::evalMeritFunc( ParOptVec *xk, double *sk,
			      ParOptVec *swk ){
  // Get the value of the lower/upper bounds and variables
  double *xvals, *lbvals, *ubvals;
  xk->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  
  // Add the contribution from the lower/upper bounds. Note
  // that we keep track of the positive and negative contributions
  // separately to try to avoid issues with numerical cancellations. 
  // The difference is only taken at the end of the computation.
  double pos_result = 0.0, neg_result = 0.0;
  
  for ( int i = 0; i < nvars; i++ ){
    if (xvals[i] - lbvals[i] > 1.0){ 
      pos_result += log(xvals[i] - lbvals[i]);
    }
    else {
      neg_result += log(xvals[i] - lbvals[i]);
    }

    if (ubvals[i] - xvals[i] > 1.0){
      pos_result += log(ubvals[i] - xvals[i]);
    }
    else {
      neg_result += log(ubvals[i] - xvals[i]);
    }
  }

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0 && sparse_inequality){
    double *swvals;
    swk->getArray(&swvals);

    for ( int i = 0; i < nwcon; i++ ){
      if (swvals[i] > 1.0){
	pos_result += log(swvals[i]);
      }
      else {
	neg_result += log(swvals[i]);
      }
    }
  }

  // Compute the norm of the weight constraint infeasibility
  double weight_infeas = 0.0;
  if (nwcon > 0){
    prob->evalSparseCon(xk, wtemp);
    if (sparse_inequality){
      wtemp->axpy(-1.0, swk);
    }
    weight_infeas = wtemp->norm();
  }

  // Sum up the result from all processors
  double input[2];
  double result[2];
  input[0] = pos_result;
  input[1] = neg_result;
  MPI_Reduce(input, result, 2, MPI_DOUBLE, MPI_SUM, opt_root, comm);

  // Extract the result of the summation over all processors
  pos_result = result[0];
  neg_result = result[1];
  
  // Compute the full merit function only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  
  double merit = 0.0;
  if (rank == opt_root){
    // Add the contribution from the slack variables
    for ( int i = 0; i < ncon; i++ ){
      if (s[i] > 1.0){
	pos_result += log(sk[i]);
      }
      else {
	neg_result += log(sk[i]);
      }
    }
    
    // Compute the infeasibility
    double infeas = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      infeas += (c[i] - sk[i])*(c[i] - sk[i]);
    }
    infeas = sqrt(infeas) + weight_infeas;
    
    // Add the contribution from the constraints
    merit = (fobj - barrier_param*(pos_result + neg_result) +
	     rho_penalty_search*infeas);
  }

  // Broadcast the result to all processors
  MPI_Bcast(&merit, 1, MPI_DOUBLE, opt_root, comm);

  return merit;
}

/*
  Find the minimum value of the penalty parameter which will guarantee
  that we have a descent direction. Then, using the new value of the
  penalty parameter, compute the value of the merit function and its
  derivative.

  output:
  merit:   the value of the merit function
  pmerit: the value of the derivative of the merit function
*/
void ParOpt::evalMeritInitDeriv( double max_x, 
				 double * _merit, double * _pmerit ){
  // Retrieve the values of the design variables, the design
  // variable step, and the lower/upper bounds
  double *xvals, *pxvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  px->getArray(&pxvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);

  // Add the contribution from the lower/upper bounds. Note
  // that we keep track of the positive and negative contributions
  // separately to try to avoid issues with numerical cancellations. 
  // The difference is only taken at the end of the computation.
  double pos_result = 0.0, neg_result = 0.0;
  double pos_presult = 0.0, neg_presult = 0.0;
  
  for ( int i = 0; i < nvars; i++ ){
    if (xvals[i] - lbvals[i] > 1.0){ 
      pos_result += log(xvals[i] - lbvals[i]);
    }
    else {
      neg_result += log(xvals[i] - lbvals[i]);
    }

    if (ubvals[i] - xvals[i] > 1.0){
      pos_result += log(ubvals[i] - xvals[i]);
    }
    else {
      neg_result += log(ubvals[i] - xvals[i]);
    }

    if (pxvals[i] > 0.0){
      pos_presult += pxvals[i]/(xvals[i] - lbvals[i]);
      neg_presult -= pxvals[i]/(ubvals[i] - xvals[i]);
    }
    else {
      neg_presult += pxvals[i]/(xvals[i] - lbvals[i]);
      pos_presult -= pxvals[i]/(ubvals[i] - xvals[i]);
    }
  }

  // Add the contributions to the log-barrier terms from
  // weighted-sum sparse constraints
  if (nwcon > 0 && sparse_inequality){
    double *swvals, *pswvals;
    sw->getArray(&swvals);
    psw->getArray(&pswvals);

    for ( int i = 0; i < nwcon; i++ ){
      if (swvals[i] > 1.0){
	pos_result += log(swvals[i]);
      }
      else {
	neg_result += log(swvals[i]);
      }

      if (pswvals[i] > 0.0){
	pos_presult += pswvals[i]/swvals[i]; 
      }
      else {
	neg_presult += pswvals[i]/swvals[i];
      }
    }
  }

  // Compute the norm of the weight constraint infeasibility
  double weight_infeas = 0.0;
  if (nwcon > 0){
    prob->evalSparseCon(x, wtemp);
    if (sparse_inequality){
      wtemp->axpy(-1.0, sw);
    }
    weight_infeas = wtemp->norm();
  }

  // Sum up the result from all processors
  double input[4];
  double result[4];
  input[0] = pos_result;
  input[1] = neg_result;
  input[2] = pos_presult;
  input[3] = neg_presult;

  MPI_Reduce(input, result, 4, MPI_DOUBLE, MPI_SUM, opt_root, comm);

  // Extract the result of the summation over all processors
  pos_result = result[0];
  neg_result = result[1];
  pos_presult = result[2];
  neg_presult = result[3];

  // Compute the projected derivative
  double proj = g->dot(px);
  
  // Perform the computations only on the root processor
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // The values of the merit function and its derivative
  double merit = 0.0;
  double pmerit = 0.0;

  if (rank == opt_root){
    // Add the contribution from the slack variables
    for ( int i = 0; i < ncon; i++ ){
      if (s[i] > 1.0){
	pos_result += log(s[i]);
      }
      else {
	neg_result += log(s[i]);
      }
      
      if (ps[i] > 0.0){
	neg_presult += ps[i]/s[i];
      }
      else {
	neg_presult += ps[i]/s[i];
      }
    }
    
    // Compute the infeasibility
    double infeas = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      infeas += (c[i] - s[i])*(c[i] - s[i]);
    }
    infeas = sqrt(infeas) + weight_infeas;
    
    // Compute the numerator term
    double numer = proj - barrier_param*(pos_presult + neg_presult);
    
    // Compute the first guess for the new
    double rho_hat = 0.0;
    if (infeas > 0.0){
      rho_hat = numer/((1.0 - penalty_descent_fraction)*max_x*infeas);
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
    pmerit = numer - rho_penalty_search*max_x*infeas;
  }

  input[0] = merit;
  input[1] = pmerit;
  input[2] = rho_penalty_search;

  // Broadcast the penalty parameter to all procs
  MPI_Bcast(input, 3, MPI_DOUBLE, opt_root, comm);

  *_merit = input[0];
  *_pmerit = input[1];
  rho_penalty_search = input[2];
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
int ParOpt::lineSearch( double * _alpha, 
			double m0, double dm0 ){
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
    for ( int i = 0; i < ncon; i++ ){
      rs[i] = s[i] + alpha*ps[i];
    }

    // Evaluate the objective and constraints at the new point
    int fail_obj = prob->evalObjCon(rx, &fobj, c);
    neval++;

    if (fail_obj){
      fprintf(stderr, 
	      "Evaluation failed during line search, trying new point\n");

      // Multiply alpha by 1/10 like SNOPT
      alpha *= 0.1;
      continue;
    }

    // Evaluate the merit function
    double merit = evalMeritFunc(rx, rs, rsw);

    // Check the sufficient decrease condition
    if (merit < m0 + armijio_constant*alpha*dm0){
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
	double alpha_new = -0.5*dm0*(alpha*alpha)/(merit - m0 - dm0*alpha);

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
  if (nwcon > 0 && sparse_inequality){
    sw->axpy(alpha, psw);
  }
  zw->axpy(alpha, pzw);
  zl->axpy(alpha, pzl);
  zu->axpy(alpha, pzu);
  
  for ( int i = 0; i < ncon; i++ ){
    s[i] += alpha*ps[i];
    z[i] += alpha*pz[i];
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
  int fail_gobj = prob->evalObjConGradient(rx, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr, 
	    "Gradient evaluation failed at final line search\n");
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
int ParOpt::optimize( const char * checkpoint ){
  // Zero out the number of function/gradient evaluations
  neval = ngeval = 0;
  
  // Print out all the parameter values to the screen
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == opt_root){
    fprintf(outfp, "ParOpt: Parameter summary\n");
    fprintf(outfp, "%-30s %15d\n", "total variables", nvars_total);
    fprintf(outfp, "%-30s %15d\n", "constraints", ncon);
    fprintf(outfp, "%-30s %15d\n", "max_major_iters", max_major_iters);
    fprintf(outfp, "%-30s %15d\n", "init_starting_point", 
	    init_starting_point);
    fprintf(outfp, "%-30s %15g\n", "barrier_param", barrier_param);
    fprintf(outfp, "%-30s %15g\n", "abs_res_tol", abs_res_tol);
    fprintf(outfp, "%-30s %15d\n", "use_line_search", use_line_search);
    fprintf(outfp, "%-30s %15d\n", "use_backtracking_alpha", 
	    use_backtracking_alpha);
    fprintf(outfp, "%-30s %15d\n", "max_line_iters", max_line_iters);
    fprintf(outfp, "%-30s %15g\n", "penalty_descent_fraction", 
	    penalty_descent_fraction);
    fprintf(outfp, "%-30s %15g\n", "armijio_constant", armijio_constant);
    fprintf(outfp, "%-30s %15g\n", "monotone_barrier_fraction", 
	    monotone_barrier_fraction);
    fprintf(outfp, "%-30s %15g\n", "monotone_barrier_power", 
	    monotone_barrier_power);
    fprintf(outfp, "%-30s %15g\n", "min_fraction_to_boundary", 
	    min_fraction_to_boundary);
    fprintf(outfp, "%-30s %15d\n", "major_iter_step_check", 
	    major_iter_step_check);
    fprintf(outfp, "%-30s %15d\n", "write_output_frequency", 
	    write_output_frequency);
    fprintf(outfp, "%-30s %15d\n", "sequential_linear_method",
	    sequential_linear_method);
    fprintf(outfp, "%-30s %15d\n", "hessian_reset_freq",
	    hessian_reset_freq);
    fprintf(outfp, "%-30s %15d\n", "use_hvec_product",
	    use_hvec_product);
    fprintf(outfp, "%-30s %15g\n", "gmres_rtol",
	    gmres_rtol);
    fprintf(outfp, "%-30s %15g\n", "gmres_atol",
	    gmres_atol);
  }

  // Evaluate the objective, constraint and their gradients at the
  // current values of the design variables
  int fail_obj = prob->evalObjCon(x, &fobj, c);
  neval++;
  if (fail_obj){
    fprintf(stderr, 
	    "Initial function and constraint evaluation failed\n");
    return fail_obj;
  }
  int fail_gobj = prob->evalObjConGradient(x, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr, "Initial gradient evaluation failed\n");
    return fail_obj;
  }

  // Assign initial multipliers
  for ( int i = 0; i < ncon; i++ ){
    z[i] = 1.0;
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
	if (z[i] < 0.01 || z[i] > 100.0){
	  z[i] = 1.0;
	}
      }
    }
  }

  // Keep track of whether the algorithm has converged
  int converged = 0;  

  // Store the previous steps in the x/z directions for
  // the purposes of printing them out on the screen
  double alpha_xprev = 0.0;
  double alpha_zprev = 0.0;

  // Keep track of the projected merit function derivative
  double dm0_prev = 0.0;

  // Information about what happened on the previous major iteration
  char info[64];
  info[0] = '\0';

  for ( int k = 0; k < max_major_iters; k++ ){
    if (!sequential_linear_method){
      if (k > 0 && k % hessian_reset_freq == 0){
	qn->reset();
      }
    }

    // Print out the current solution progress using the 
    // hook in the problem definition
    if (k % write_output_frequency == 0){
      if (checkpoint){
	// Write the checkpoint file, if it fails once, set
	// the file pointer to null so it won't print again
	if (writeSolutionFile(checkpoint)){
	  checkpoint = NULL;
	}
      }

      prob->writeOutput(k, x);
    }

    // Compute the complementarity
    double comp = computeComp();
    
    // Compute the residual of the KKT system 
    double max_prime, max_dual, max_infeas;
    computeKKTRes(&max_prime, &max_dual, &max_infeas);

    // Print all the information we can to the screen...
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == opt_root){
      if (k % 10 == 0){
	fprintf(outfp, "\n%4s %4s %4s %7s %7s %12s \
%7s %7s %7s %7s %7s %8s %7s info\n",
		"iter", "nobj", "ngrd", "alphx", "alphz", 
		"fobj", "|opt|", "|infes|", "|dual|", "mu", 
		"comp", "dmerit", "rho");
      }

      if (k == 0){
	fprintf(outfp, "%4d %4d %4d %7s %7s %12.5e \
%7.1e %7.1e %7.1e %7.1e %7.1e %8s %7s %s\n",
		k, neval, ngeval, " ", " ",
		fobj, max_prime, max_infeas, max_dual, 
		barrier_param, comp, " ", " ", info);
      }
      else {
	fprintf(outfp, "%4d %4d %4d %7.1e %7.1e %12.5e \
%7.1e %7.1e %7.1e %7.1e %7.1e %8.1e %7.1e %s\n",
	       k, neval, ngeval, alpha_xprev, alpha_zprev,
	       fobj, max_prime, max_infeas, max_dual, 
		barrier_param, comp, dm0_prev, rho_penalty_search, info);
      }
      
      if (k % write_output_frequency == 0){
	fflush(outfp);
      }
    }

    // Compute the norm of the residuals
    double res_norm = max_prime;
    if (max_dual > res_norm){ res_norm = max_dual; }
    if (max_infeas > res_norm){ res_norm = max_infeas; }

    // Check for convergence
    if (res_norm < abs_res_tol && 
	barrier_param < 0.1*abs_res_tol){
      converged = 1;
      break;
    }

    // Determine if the residual norm has been reduced
    // sufficiently in order to switch to a new barrier
    // problem
    if (res_norm < 10.0*barrier_param){
      // Record the value of the old barrier function
      double mu_old = barrier_param;

      // Compute the new barrier parameter: It is either:
      // 1. A fixed fraction of the old value
      // 2. A function mu**exp for some exp > 1.0
      // Point 2 ensures superlinear convergence (eventually)
      double mu_frac = monotone_barrier_fraction*barrier_param;
      double mu_pow = pow(barrier_param, monotone_barrier_power);

      barrier_param = mu_frac;
      if (mu_pow < mu_frac){
	barrier_param = mu_pow;
      }

      // Now, that we have adjusted the barrier parameter, we have
      // to modify the residuals to match
      for ( int i = 0; i < ncon; i++ ){
	rs[i] -= (mu_old - barrier_param);
      }

      double *rzlvals, *rzuvals;
      rzl->getArray(&rzlvals);
      rzu->getArray(&rzuvals);
      
      for ( int i = 0; i < nvars; i++ ){
	rzlvals[i] -= (mu_old - barrier_param);
	rzuvals[i] -= (mu_old - barrier_param);
      }

      // Reset the penalty parameter to zero
      rho_penalty_search = 0.0;
    }

    // Note that at this stage, we use s_qn and y_qn as 
    // temporary arrays to help compute the KKT step. After
    // the KKT step is computed, we use them to store the
    // change in variables/gradient for the BFGS update.

    // Set up the KKT diagonal system
    setUpKKTDiagSystem(s_qn, wtemp);

    // Set up the full KKT system
    setUpKKTSystem(ztemp, s_qn, y_qn, wtemp);

    int gmres_iters = 0;
    if (use_hvec_product && 
	(max_prime < gmres_switch_tol &&
	 max_dual < gmres_switch_tol && 
	 max_infeas < gmres_switch_tol)){
      // Compute the inexact step using GMRES - note that this
      // uses a fixed tolerance -- this may lead to over-solving
      // if rtol is too tight
      gmres_iters = computeKKTInexactNewtonStep(ztemp, y_qn, s_qn, wtemp,
						gmres_rtol, gmres_atol);
    }
    else {
      // Solve for the KKT step
      computeKKTStep(ztemp, s_qn, y_qn, wtemp);
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

    double max_x = 1.0, max_z = 1.0;
    computeMaxStep(tau, &max_x, &max_z);

    // Bound the difference between the step lengths. This code
    // cuts off the difference between the step lengths by a bound.
    double max_bnd = 1e2;
    if (max_x > max_z){
      if (max_x > max_bnd*max_z){
	max_x = max_bnd*max_z;
      }
      else if (max_x < max_z/max_bnd){
	max_x = max_z/max_bnd;
      }
    }
    else {
      if (max_z > max_bnd*max_x){
	max_z = max_bnd*max_x;
      }
      else if (max_z < max_x/max_bnd){
	max_z = max_x/max_bnd;
      }
    }
    
    // As a last check, compute the complementarity at
    // the full step length. If the complementarity increases,
    // use equal step lengths.
    double comp_new = computeCompStep(max_x, max_z);
    if (comp_new > comp){
      if (max_x > max_z){
	max_x = max_z;
      }
      else {
	max_z = max_x;
      }
    }

    // Scale the steps by the maximum permissible step lengths
    px->scale(max_x);
    if (nwcon > 0 && sparse_inequality){
      psw->scale(max_x);
    }
    pzw->scale(max_z);
    pzl->scale(max_z);
    pzu->scale(max_z);

    for ( int i = 0; i < ncon; i++ ){
      ps[i] *= max_x;
      pz[i] *= max_z;
    }

    // Store the design variable locations for the 
    // Hessian update. The gradient difference update
    // is done after the step has been selected, but
    // before the new gradient is evaluated (so we 
    // have the new multipliers)
    if (!sequential_linear_method){
      s_qn->copyValues(x);
      s_qn->scale(-1.0);
    }

    // Keep track of the step length size
    double alpha = 1.0;
    int line_fail = 0;

    if (gmres_iters == 0 && use_line_search){
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter
      double m0, dm0;
      evalMeritInitDeriv(max_x, &m0, &dm0);

      // Check that the merit function derivative is
      // correct and print the derivative to the screen on the
      // optimization-root processor
      if (k == major_iter_step_check){
	double dh = merit_func_check_epsilon;
	rx->copyValues(x);
	rx->axpy(dh, px);
	
	for ( int i = 0; i < ncon; i++ ){
	  rs[i] = s[i] + dh*ps[i];
	}

	if (nwcon > 0 && sparse_inequality){
	  rsw->copyValues(sw);
	  rsw->axpy(dh, psw);
	}

	prob->evalObjCon(rx, &fobj, c);
	double m1 = evalMeritFunc(rx, rs, rsw);

	if (rank == opt_root){
	  double fd = (m1 - m0)/dh;
	  printf("Merit function test\n");
	  printf("dm FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
		 fd, dm0, fabs(fd - dm0), fabs((fd - dm0)/fd));
	}
      }
      
      // The directional derivative is so small that we apply the
      // full step, regardless
      if (dm0 > -abs_res_tol*abs_res_tol){
	// Apply the full step to the Lagrange multipliers and
	// slack variables
	alpha = 1.0;
	zw->axpy(alpha, pzw);
	if (nwcon > 0 && sparse_inequality){
	  sw->axpy(alpha, psw);
	}
	zl->axpy(alpha, pzl);
	zu->axpy(alpha, pzu);

	for ( int i = 0; i < ncon; i++ ){
	  s[i] += alpha*ps[i];
	  z[i] += alpha*pz[i];
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

	// Update x here so that we don't impact
	// the design variables when computing Aw(x)^{T}*zw
	x->axpy(alpha, px);

	// Evaluate the objective, constraint and their gradients at the
	// current values of the design variables
	int fail_obj = prob->evalObjCon(x, &fobj, c);
	neval++;
	if (fail_obj){
	  fprintf(stderr, "Function and constraint evaluation failed\n");
	  return fail_obj;
	}
	int fail_gobj = prob->evalObjConGradient(x, g, Ac);
	ngeval++;
	if (fail_gobj){
	  fprintf(stderr, "Gradient evaluation failed\n");
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
      if (nwcon > 0 && sparse_inequality){
	sw->axpy(alpha, psw);
      }
      zw->axpy(alpha, pzw);
      zl->axpy(alpha, pzl);
      zu->axpy(alpha, pzu);

      for ( int i = 0; i < ncon; i++ ){
	s[i] += alpha*ps[i];
	z[i] += alpha*pz[i];
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
	fprintf(stderr, "Function and constraint evaluation failed\n");
	return fail_obj;
      }
      int fail_gobj = prob->evalObjConGradient(x, g, Ac);
      ngeval++;
      if (fail_gobj){
	fprintf(stderr, "Gradient evaluation failed\n");
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
    alpha_xprev = alpha*max_x;
    alpha_zprev = alpha*max_z;

    // Compute the Quasi-Newton update
    int up_type = 0;
    if (!sequential_linear_method && !line_fail){
      up_type = qn->update(s_qn, y_qn);
    }

    // Create a string to print to the screen
    if (rank == opt_root){
      // The string of unforseen events
      info[0] = '\0';
      if (gmres_iters > 0){
	sprintf(info, "%s%s%d ", info, "iNK", gmres_iters);
      }
      if (up_type == 1){ 
	// Damped BFGS update
	sprintf(info, "%s ", "dH");
      }
      if (line_fail){
	// Line search failure
	sprintf(info, "%s%s ", info, "LF");
      }
      if (dm0_prev > -abs_res_tol*abs_res_tol){
	// Skip the line search b/c descent direction is not 
	// sufficiently descent-y
	sprintf(info, "%s%s ", info, "sk");
      }
      if (comp_new > comp){
	// The step lengths are equal due to an increase in the
	// the complementarity at the new step
	sprintf(info, "%s%s ", info, "ceq");
      }
    }
  }

  // Success - we completed the optimization
  return 0; 
}

/*
  This function approximately solves the linearized KKT system with
  Hessian-vector products using GMRES.  This procedure uses a
  preconditioner formed from a portion of the KKT system.  Grouping
  the lagrange multipliers and slack variables from the remaining
  portion of the matrix, yields the following decomposition:

  K = [ 0; A ] + [ H; 0 ]
  .   [ E; C ]   [ 0; 0 ]

  Setting the precontioner as:

  M = [ 0; A ]
  .   [ E; C ]

  We use right-preconditioning and solve the following system:

  K*M^{-1}*u = b

  where M*x = u, so we compute x = M^{-1}*u

  {[ I; 0 ] + [ H; 0 ]*M^{-1}}[ ux ] = [ bx ] 
  {[ 0; I ] + [ 0; 0 ]       }[ uy ]   [ by ]
*/
int ParOpt::computeKKTInexactNewtonStep( double *zt, 
					 ParOptVec *xt1, ParOptVec *xt2,
					 ParOptVec *wt,
					 double rtol, double atol ){
  // Initialize the data from the gmres object
  double *H = gmres_H;
  double *alpha = gmres_alpha;
  double *res = gmres_res;
  double *Qcos = &gmres_Q[0];
  double *Qsin = &gmres_Q[gmres_subspace_size];
  ParOptVec **W = gmres_W;

  // Compute the beta factor: the inner product of the
  // diagonal terms after normalization
  double beta = 0.0;
  for ( int i = 0; i < ncon; i++ ){
    beta += rc[i]*rc[i] + rs[i]*rs[i];
  }
  beta += rzl->dot(rzl) + rzu->dot(rzu);

  if (nwcon > 0){
    beta += rcw->dot(rcw) + rsw->dot(rsw);
  }

  // Compute the norm of the initial vector
  double bnorm = sqrt(rx->dot(rx) + beta);

  // Broadcast the norm of the residuals and the
  // beta parameter to keep things consistent across processors
  double temp[2];
  temp[0] = bnorm;
  temp[1] = beta;

  MPI_Bcast(temp, 2, MPI_DOUBLE, opt_root, comm);

  bnorm = temp[0];
  beta = temp[1];

  // Compute the final value of the beta term
  beta *= 1.0/(bnorm*bnorm);

  // Initialize the terms for
  res[0] = bnorm;
  W[0]->copyValues(rx);
  W[0]->scale(1.0/res[0]);
  alpha[0] = 1.0;

  // Keep track of the actual number of iterations
  int solve_flag = 0;
  int niters = 0;

  for ( int i = 0; i < gmres_subspace_size; i++ ){
    // Compute M^{-1}*[ W[i], alpha[i]*yc, ... ] 

    // Get the size of the limited-memory BFGS subspace
    double b0;
    const double *d, *M;
    ParOptVec **Z;
    int size = 0;
    if (!sequential_linear_method){
      size = qn->getLBFGSMat(&b0, &d, &M, &Z);
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
    
      // Solve the digaonal system again, this time simplifying
      // the result due to the structure of the right-hand-side.
      // Note that this call uses W[i+1] as a temporary vector. 
      solveKKTDiagSystem(xt2, px,
			 zt, W[i+1], wt);

      // Add the final contributions 
      xt1->axpy(-1.0, px);
    }

    // Compute the inner product with the exact Hessian
    prob->evalHvecProduct(x, z, zw, xt1, W[i+1]);
    
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
    for ( int j = 0; j < i+1; j++ ){
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
      
    // Apply the existing part of Q to the new components of 
    // the Hessenberg matrix
    for ( int k = 0; k < i; k++ ){
      double h1 = H[k + hptr];
      double h2 = H[k+1 + hptr];
      H[k + hptr] = h1*Qcos[k] + h2*Qsin[k];
      H[k+1 + hptr] = -h1*Qsin[k] + h2*Qcos[k];
    }
      
    // Now, compute the rotation for the new column that was just added
    double h1 = H[i + hptr];
    double h2 = H[i+1 + hptr];
    double sq = sqrt(h1*h1 + h2*h2);
    
    Qcos[i] = h1/sq;
    Qsin[i] = h2/sq;
    H[i + hptr] = h1*Qcos[i] + h2*Qsin[i];
    H[i+1 + hptr] = -h1*Qsin[i] + h2*Qcos[i];
    
    // Update the residual
    h1 = res[i];
    res[i] = h1*Qcos[i];
    res[i+1] = -h1*Qsin[i];
          
    niters++;
      
    // Check for convergence
    if (fabs(res[i+1]) < atol ||
	fabs(res[i+1]) < rtol*bnorm){
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
  double gamma = res[0]*alpha[0];
 
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

  // Apply M^{-1} to the result to obtain the final
  // answer
  solveKKTDiagSystem(W[0], rc, rcw, rs, rsw, rzl, rzu, 
		     px, pz, pzw, ps, psw, pzl, pzu,
		     xt1, wt);

  // Get the size of the limited-memory BFGS subspace
  double b0;
  const double *d, *M;
  ParOptVec **Z;
  int size = 0;
  if (!sequential_linear_method){
    size = qn->getLBFGSMat(&b0, &d, &M, &Z);
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
    
    // Add the terms from the 
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
  
  double *pxvals, *gvals;
  px->getArray(&pxvals);
  g->getArray(&gvals);
  
  for ( int i = 0; i < nvars; i++ ){
    if (gvals[i] >= 0.0){
      pxvals[i] = 1.0;
    }
    else {
      pxvals[i] = -1.0;
    }
  }

  // Compute the projected derivative
  double pobj = g->dot(px);
  px->mdot(Ac, ncon, rs);

  // Set the step direction in the sparse Lagrange multipliers
  // to an initial vector
  if (nwcon > 0){
    double *pzwvals;
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
    hvec = new ParOptVec(comm, nvars);
    prob->evalHvecProduct(x, ztemp, pzw, px, hvec);
  }

  // Compute the point xt = x + dh*px
  ParOptVec *xt = y_qn;
  xt->copyValues(x);
  xt->axpy(dh, px);

  // Compute the finite-difference product
  double fobj2;
  prob->evalObjCon(xt, &fobj2, rc);

  double pfd = (fobj2 - fobj)/dh;

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    printf("Objective gradient test\n");
    printf("Objective FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
	   pfd, pobj, fabs(pobj - pfd), fabs((pobj - pfd)/pfd));

    printf("\nConstraint gradient test\n");
    for ( int i = 0; i < ncon; i++ ){
      double fd = (rc[i] - c[i])/dh;
      printf("Con[%3d]  FD: %15.8e  Actual: %15.8e  Err: %8.2e  Rel err: %8.2e\n",
	     i, fd, rs[i], fabs(fd - rs[i]), fabs((fd - rs[i])/fd));
    }
  }

  if (use_hvec_product){
    // Evaluate the objective/constraints
    ParOptVec *g2 = new ParOptVec(comm, nvars);
    ParOptVec **Ac2 = new ParOptVec*[ ncon ];
    for ( int i = 0; i < ncon; i++ ){
      Ac2[i] = new ParOptVec(comm, nvars);
    }
    
    // Evaluate the gradient at the perturbed point
    // and add the contribution from the sparse constraints
    // to the Hessian
    prob->evalObjConGradient(xt, g2, Ac2);
    prob->addSparseJacobianTranspose(-1.0, xt, pzw, g2);

    // Add the contribution from the dense constraints
    for ( int i = 0; i < ncon; i++ ){
      g2->axpy(-ztemp[i], Ac2[i]);
    }

    // Compute the difference
    g2->axpy(-1.0, g);
    g2->scale(1.0/dh);

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

    double d1 = rsw->dot(pzw);
    double d2 = rx->dot(px);

    if (rank == opt_root){
      printf("\nTranspose-equivalence\n");
      printf("x^{T}*(J(x)*p): %8.2e  p*(J(x)^{T}*x): %8.2e  Err: %8.2e  Rel Err: %8.2e\n",
	     d1, d2, fabs(d1 - d2), fabs((d1 - d2)/d2));
    }

    // Set Cvec to something more-or-less random
    double *cvals, *rxvals;
    Cvec->getArray(&cvals);
    rx->getArray(&rxvals);
    for ( int i = 0; i < nvars; i++ ){
      cvals[i] = 0.05 + 0.25*(i % 37);
    }

    // Check the inner product pzw^{T}*J(x)*cvec*J(x)^{T}*pzw against the 
    // matrix Cw
    memset(Cw, 0, nwcon*(nwblock+1)/2*sizeof(double));
    prob->addSparseInnerProduct(1.0, x, Cvec, Cw);

    // Compute the inner product using the Jacobians
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
    double *cw = Cw;
    const int incr = ((nwblock+1)*nwblock)/2;

    double *pzwvals;
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
    double temp = d2;
    MPI_Reduce(&temp, &d2, 1, MPI_DOUBLE, MPI_SUM, opt_root, comm);

    if (rank == opt_root){
      printf("\nJ(x)*C^{-1}*J(x)^{T} test: \n");
      printf("Product: %8.2e  Matrix: %8.2e  Err: %8.2e  Rel Err: %8.2e\n",
	     d1, d2, fabs(d1 - d2), fabs((d1 - d2)/d2));
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
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the values of the steps
  double *pxvals, *pzlvals, *pzuvals;
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
    }
    else {
      rx->zeroEntries();
    }
  }
  for ( int i = 0; i < ncon; i++ ){
    rx->axpy(-pz[i], Ac[i]);
  }
  rx->axpy(-1.0, pzl);
  rx->axpy(1.0, pzu);
  rx->axpy(1.0, g);
  for ( int i = 0; i < ncon; i++ ){
    rx->axpy(-z[i], Ac[i]);
  }
  rx->axpy(-1.0, zl);
  rx->axpy(1.0, zu);

  // Add the contributions from the constraint
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(-1.0, x, zw, rx);
    prob->addSparseJacobianTranspose(-1.0, x, pzw, rx);
  }
  double max_val = rx->maxabs();
  
  if (rank == opt_root){
    printf("max |H*px - Ac^{T}*pz - Aw^{T}*pzw - pzl + pzu + \
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
    double val = rc[i] - ps[i] + (c[i] - s[i]);
    if (fabs(val) > max_val){
      max_val = fabs(val);
    }
  }
  if (rank == opt_root){
    printf("max |A*px - ps + (c - s)|: %10.4e\n", max_val);
  }

  // Find the maximum value of the residual equations for
  // the dual slack variables
  max_val = 0.0;
  for ( int i = 0; i < ncon; i++ ){
    double val = z[i]*ps[i] + s[i]*pz[i] + (z[i]*s[i] - barrier_param);
    if (fabs(val) > max_val){
      max_val = fabs(val);
    }
  }
  if (rank == opt_root){
    printf("max |Z*ps + S*pz + (z*s - mu)|: %10.4e\n", max_val);
  }

  // Find the maximum of the residual equations for the
  // lower-bound dual variables
  max_val = 0.0;
  for ( int i = 0; i < nvars; i++ ){
    double val = (zlvals[i]*pxvals[i] + (xvals[i] - lbvals[i])*pzlvals[i] +
		  (zlvals[i]*(xvals[i] - lbvals[i]) - barrier_param));
    if (fabs(val) > max_val){
      max_val = fabs(val);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);
  if (rank == opt_root){
    printf("max |Zl*px + (X - LB)*pzl + (Zl*(x - lb) - mu)|: %10.4e\n", 
	   max_val);
  }

  // Find the maximum value of the residual equations for the
  // upper-bound dual variables
  max_val = 0.0;
  for ( int i = 0; i < nvars; i++ ){
    double val = (-zuvals[i]*pxvals[i] + (ubvals[i] - xvals[i])*pzuvals[i] +
		  (zuvals[i]*(ubvals[i] - xvals[i]) - barrier_param));
    if (fabs(val) > max_val){
      max_val = fabs(val);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);
  if (rank == opt_root){
    printf("max |-Zu*px + (UB - X)*pzu + (Zu*(ub - x) - mu)|: %10.4e\n", 
	   max_val);
  }
}
