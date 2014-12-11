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

  input:
  prob:      the optimization problem
  nwcon:     the number of weighting constraints
  nw:        the number of variables in each constraint
  max_lbfgs: the number of steps to store in the the l-BFGS
*/
ParOpt::ParOpt( ParOptProblem * _prob, int _nwcon, 
		int _nwstart, int _nw, int _nwskip,
		int max_lbfgs_subspace ){
  prob = _prob;

  // Record the communicator
  comm = prob->getMPIComm();
  opt_root = 0;

  // Get the number of variables/constraints
  prob->getProblemSizes(&nvars, &ncon);

  // Assign the values from the sparsity constraints
  nwcon = _nwcon;
  nwstart = _nwstart;
  nw = _nw;
  nwskip = _nwskip;
  if (nwskip < 0 || 
      (nwstart + nw*nwcon + nwskip*(nwcon-1) > nvars)){
    fprintf(stderr, "Weighted constraints are inconsistent\n");
    nwcon = 0;
    nw = 0;
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

  // Allocate space for the residuals
  rx = new ParOptVec(comm, nvars);
  rzl = new ParOptVec(comm, nvars);
  rzu = new ParOptVec(comm, nvars);
  rc = new double[ ncon ];
  rs = new double[ ncon ];

  // Allocate space for the Quasi-Newton updates
  y_qn = new ParOptVec(comm, nvars);
  s_qn = new ParOptVec(comm, nvars);

  // Allocate temporary storage for nvars-sized things
  xtemp = new ParOptVec(comm, nvars);

  // Allocate vectors for the weighting constraints
  wtemp = new ParOptVec(comm, nwcon);
  zw = new ParOptVec(comm, nwcon);
  pzw = new ParOptVec(comm, nwcon);
  rw = new ParOptVec(comm, nwcon);
  
  Cwvec = new ParOptVec(comm, nwcon);
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

  // By default, set the file pointer to stdout
  outfp = stdout;
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

  // Delete the steps
  delete px;
  delete pzl;
  delete pzu;
  delete [] pz;
  delete [] ps;

  // Delete the residuals
  delete rx;
  delete rzl;
  delete rzu;
  delete [] rc;
  delete [] rs;

  // Delete the quasi-Newton updates
  delete y_qn;
  delete s_qn;

  // Delete the temp data
  delete xtemp;
  delete [] ztemp;

  // Delete the weighting constraints
  delete wtemp;
  delete zw;
  delete pzw;
  delete rw;
  delete Cwvec;
  
  for ( int i = 0; i < ncon; i++ ){
    delete Ew[i];
  }
  delete [] Ew;

  // Delete the various matrices
  delete [] Dmat;
  delete [] dpiv;
  delete [] Ce;
  delete [] cpiv;
  delete Cvec;

  // Delete the
  delete [] c;
  delete g;
  for ( int i = 0; i < ncon; i++ ){
    delete Ac[i];
  }
  delete [] Ac;

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
  delete [] fname;

  if (fp){
    // Successfull opened the file
    fail = 0;

    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_rank(comm, &size);

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
      MPI_File_write(fp, z, ncon, MPI_DOUBLE, MPI_STATUS_IGNORE);
      MPI_File_write(fp, s, ncon, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    size_t offset = 3*sizeof(int) + 2*ncon*sizeof(double);

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
      double *zwvals;
      int nwsize = zw->getArray(&zwvals);
      MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
			datarep, MPI_INFO_NULL);
      MPI_File_write_at_all(fp, nwcon_range[rank], zwvals, nwsize, MPI_DOUBLE,
			    MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fp);

    delete [] var_range;
    delete [] nwcon_range;
  }

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
    MPI_Comm_rank(comm, &size);

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
	MPI_File_read(fp, var_sizes, 3, MPI_INT, MPI_STATUS_IGNORE);
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
    size_t offset = 3*sizeof(int) + 2*ncon*sizeof(double);

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
      double *zwvals;
      int nwsize = zw->getArray(&zwvals);
      MPI_File_set_view(fp, offset, MPI_DOUBLE, MPI_DOUBLE,
			datarep, MPI_INFO_NULL);
      MPI_File_read_at_all(fp, nwcon_range[rank], zwvals, nwsize, MPI_DOUBLE,
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
  stored internally in the ParOpt optimizer. The only input required
  is the given the governing equations.

  This code computes the following terms:

  rx  = -(g(x) - Ac^{T}*z - Aw^{T}*zw - zl + zu) 
  rc  = -(c(x) - s)
  rw  = -(Aw*x - e)
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

  // Assemble the residual of the first KKT equation:
  // g(x) - Ac^{T}*z - zl + zu
  rx->copyValues(zl);
  rx->axpy(-1.0, zu);
  rx->axpy(-1.0, g);

  for ( int i = 0; i < ncon; i++ ){
    rx->axpy(z[i], Ac[i]);
  }

  // Add rx = rx + Aw^{T}*zw
  if (nwcon > 0){
    double *rxvals, *zwvals;
    rx->getArray(&rxvals);
    zw->getArray(&zwvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	rxvals[j] += zwvals[i];
      }
    }
  }

  // Compute the residuals from the weighting constraints
  if (nwcon > 0){
    double *xvals, *rwvals; 
    x->getArray(&xvals);
    rw->getArray(&rwvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      rwvals[i] = 1.0;
      for ( int k = 0; k < nw; k++, j++ ){
	rwvals[i] -= xvals[j];
      }
    }
  }

  // Compute the error in the first KKT condition
  *max_prime = rx->maxabs();

  // Compute the residuals from the second KKT system:
  *max_infeas = rw->maxabs();

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
}

/*
  This function computes the terms required to solve the KKT system
  using a bordering method.  The initialization process computes the
  following matrix:
  
  C = b0 + zl/(x - lb) + zu/(ub - x)

  where C is a diagonal matrix. The components of C^{-1} (also a
  diagonal matrix) are stored in Cvec.

  Next, we compute:
  
  Cw = Aw*C^{-1}*Aw^{T}

  where Cw is another diagonal matrix. The components of Cw^{-1} are
  stored in Cwvec.  The code then computes the contribution from the
  weighting constraints as follows:

  Ew = Aw*C^{-1}*A, followed by:

  Dw = Ew^{T}*Cw^{-1}*Ew

  Finally, the code computes a factorization of the matrix:

  D = Z^{-1}*S + A*C^{-1}*A^{T} - Dw

  which is required to compute the solution of the KKT step.
*/
void ParOpt::setUpKKTDiagSystem(){ 
  // Retrieve the values of the design variables, lower/upper bounds
  // and the corresponding lagrange multipliers
  double *xvals, *lbvals, *ubvals, *zlvals, *zuvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrive the diagonal entry for the BFGS update
  double b0;
  const double *d, *M;
  ParOptVec **Z;
  qn->getLBFGSMat(&b0, &d, &M, &Z);
   
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
    double *cwvals;
    Cwvec->getArray(&cwvals);
    
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      // Compute Cw = Aw*C^{-1}*Aw
      cwvals[i] = 0.0;
      for ( int k = 0; k < nw; k++, j++ ){
	cwvals[i] += cvals[j];
      }
      
      // Store Cw^{-1}
      cwvals[i] = 1.0/cwvals[i];
    }
    
    // Compute Ew = Aw*C^{-1}*A
    for ( int kk = 0; kk < ncon; kk++ ){
      double *ewvals, *avals;
      Cvec->getArray(&cvals);
      Ew[kk]->getArray(&ewvals);
      Ac[kk]->getArray(&avals);
      
      for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
	ewvals[i] = 0.0;
	for ( int k = 0; k < nw; k++, j++ ){
	  ewvals[i] += cvals[j]*avals[j];
	}
      }
    }
  }

  // Set the value of the D matrix
  memset(Dmat, 0, ncon*ncon*sizeof(double));

  if (nwcon > 0){
    // Add the term Dw = - Ew^{T}*Cw^{-1}*Ew to the Dmat matrix first
    // by computing the inner product with Cwvec = Cw^{-1}
    for ( int j = 0; j < ncon; j++ ){
      for ( int i = j; i < ncon; i++ ){
	// Get the vectors required
	double *cwvals, *ewivals, *ewjvals;
	Cwvec->getArray(&cwvals);
	Ew[i]->getArray(&ewivals);
	Ew[j]->getArray(&ewjvals);
	
	double dmat = 0.0;
	int k = 0;
	int remainder = nwcon % 4;
	for ( ; k < remainder; k++ ){
	  dmat += ewivals[0]*ewjvals[0]*cwvals[0];
	  ewivals++; ewjvals++; cwvals++;
	}
	
	for ( int k = remainder; k < nwcon; k += 4 ){
	  dmat += (ewivals[0]*ewjvals[0]*cwvals[0] +
		   ewivals[1]*ewjvals[1]*cwvals[1] +
		   ewivals[2]*ewjvals[2]*cwvals[2] +
		   ewivals[3]*ewjvals[3]*cwvals[3]);
	  ewivals += 4; ewjvals += 4; cwvals += 4;
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
  
  B0*yx - A^{T}*yz - Aw^{T}*yw - yzl + yzu = bx
  A*yx - ys = bc
  Aw*yx = bw

  With the additional equations:

  ys = Z^{-1}*bs - Z^{-1}*S*yz
  yzl = (X - Xl)^{-1}*(bzl - Zl*yx)
  yzu = (Xu - X)^{-1}*(bzu + Zu*yx)

  Substitution of these three equations yields the following system of
  equations:

  ((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))*yx - A^{T}*yz - Aw^{T}*yw
  = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu

  which we rewrite as the equation:

  C*yx - A^{T}*yz - Aw^{T}*yw = d

  and
  
  A*yx + Z^{-1}*S*yz = bc + Z^{-1}*bs,
  Aw*yx = bw.

  Where we define d as the vector:
  
  d = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu,

  we can solve for yz by solving the following system of equations:

  D0*yz + Ew^{T}*yw = bc + Z^{-1}*bs - A*C^{-1}*d,
  Ew*yz +     Cw*yw = bw - Aw*C^{-1}*d

  where C, Ew, and D0 are defined as follows:

  C = B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu,
  Ew = Aw*C^{-1}*A^{T},
  D0 = Z^{-1}*S + A*C^{-1}*A^{T}.

  We can then obtain yz by solving the following system of equations:
  
  Dmat*yz = bc + Z^{-1}*bs - A*C^{-1}*d 
  .         - Ew^{T}*Cw^{-1}*(bw - Aw*C^{-1}*d)

  Once yz is obtained, we find yw and yx as follows:

  yw = Cw^{-1}*(bw - Ew*yz - Aw*C^{-1}*d) 
  yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yw)

  Note: This code uses the temporary arrays xtemp and wtemp which
  therefore cannot be inputs/outputs for this function, otherwise
  strange behavior will occur.
*/
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, double *bc, 
				 ParOptVec *bw, double *bs,
				 ParOptVec *bzl, ParOptVec *bzu,
				 ParOptVec *yx, double *yz, 
				 ParOptVec *yw, double *ys,
				 ParOptVec *yzl, ParOptVec *yzu ){
  // Set values in the temporary array
  double *dvals;
  xtemp->getArray(&dvals);

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

  // Get the right-hand-side of the first equation
  for ( int i = 0; i < nvars; i++ ){
    dvals[i] = (bxvals[i] +
		bzlvals[i]/(xvals[i] - lbvals[i]) - 
		bzuvals[i]/(ubvals[i] - xvals[i]));
  }

  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    double *wvals, *bwvals, *cwvals, *cvals;
    bw->getArray(&bwvals);
    Cvec->getArray(&cvals);
    Cwvec->getArray(&cwvals);
    wtemp->getArray(&wvals);
 
    // Compute wtemp = Cw^{-1}*(bw - Aw*C^{-1}*d)
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      wvals[i] = bwvals[i];
      for ( int k = 0; k < nw; k++, j++ ){
	wvals[i] -= cvals[j]*dvals[j];
      }
      wvals[i] *= cwvals[i];
    }
  }

  // Now, compute yz = bc + S*Z^{-1} - A*C0^{-1}*d - Ew^{T}*wtemp
  memset(yz, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wtemp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  for ( int i = 0; i < ncon; i++ ){
    double *cvals, *avals;
    xtemp->getArray(&dvals);
    Cvec->getArray(&cvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*dvals[0]*cvals[0];
      avals++; dvals++; cvals++; 
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*dvals[0]*cvals[0] + 
	       avals[1]*dvals[1]*cvals[1] +
	       avals[2]*dvals[2]*cvals[2] + 
	       avals[3]*dvals[3]*cvals[3]);
      avals += 4; dvals += 4; cvals += 4;
    }

    yz[i] += ydot;
  }

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
    // Compute yw = Cw^{-1}*(bw - Ew*yz - Aw*C^{-1}*d)
    // First set yw <- bw - Ew*yz
    yw->copyValues(bw);
    for ( int i = 0; i < ncon; i++ ){
      yw->axpy(-yz[i], Ew[i]);
    }

    // Compute yw <- Cw^{-1}*(yw - Aw*C^{-1}*d);
    double *cvals, *cwvals, *ywvals;
    xtemp->getArray(&dvals);
    Cvec->getArray(&cvals);
    Cwvec->getArray(&cwvals);
    yw->getArray(&ywvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	ywvals[i] -= cvals[j]*dvals[j];
      }
      ywvals[i] *= cwvals[i];
    }
  }

  // Compute the step in the design variables
  double *yxvals, *cvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);

  // Compute yx = C^{-1}*(d + A^{T}*yz + Aw^{T}*yw)
  yx->copyValues(xtemp);
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  if (nwcon > 0){
    double *ywvals;
    yw->getArray(&ywvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	yxvals[j] += ywvals[i];
      }
    }
  }

  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

  // Retrieve the lagrange multipliers
  double *zlvals, *zuvals;
  zl->getArray(&zlvals);
  zu->getArray(&zuvals);

  // Retrieve the lagrange multiplier vectors
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
				 ParOptVec *yw, double *ys,
				 ParOptVec *yzl, ParOptVec *yzu ){
  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    double *wvals, *cwvals, *cvals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Cwvec->getArray(&cwvals);
    wtemp->getArray(&wvals);
 
    // Compute wtemp = Cw^{-1}*(bw - Aw*C^{-1}*d)
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      wvals[i] = 0.0;
      for ( int k = 0; k < nw; k++, j++ ){
	wvals[i] -= cvals[j]*bxvals[j];
      }
      wvals[i] *= cwvals[i];
    }
  }

  // Now, compute yz = - A*C0^{-1}*bx - Ew^{T}*wtemp
  memset(yz, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wtemp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      yz[i] += BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  // Compute the
  for ( int i = 0; i < ncon; i++ ){
    double *cvals, *avals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*bxvals[0]*cvals[0];
      avals++; bxvals++; cvals++; 
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*bxvals[0]*cvals[0] + 
	       avals[1]*bxvals[1]*cvals[1] +
	       avals[2]*bxvals[2]*cvals[2] + 
	       avals[3]*bxvals[3]*cvals[3]);
      avals += 4; bxvals += 4; cvals += 4;
    }

    yz[i] += ydot;
  }

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Reduce the result to the root processor
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
    yw->zeroEntries();
    for ( int i = 0; i < ncon; i++ ){
      yw->axpy(-yz[i], Ew[i]);
    }

    // Compute yw <- Cw^{-1}*(yw - Aw*C^{-1}*bx);
    double *cvals, *cwvals, *ywvals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Cwvec->getArray(&cwvals);
    yw->getArray(&ywvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	ywvals[i] -= cvals[j]*bxvals[j];
      }
      ywvals[i] *= cwvals[i];
    }
  }

  // Compute the step in the design variables
  double *yxvals, *cvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);

  // Compute yx = C^{-1}*(bx + A^{T}*yz)
  yx->copyValues(bx);
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  if (nwcon > 0){
    double *ywvals;
    yw->getArray(&ywvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	yxvals[j] += ywvals[i];
      }
    }
  }

  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
  }

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
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, ParOptVec *yx ){
  // Compute the terms from the weighting constraints
  if (nwcon > 0){
    double *wvals, *cwvals, *cvals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Cwvec->getArray(&cwvals);
    wtemp->getArray(&wvals);
 
    // Compute wtemp = Cw^{-1}*(bw - Aw*C^{-1}*d)
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      wvals[i] = 0.0;
      for ( int k = 0; k < nw; k++, j++ ){
	wvals[i] -= cvals[j]*bxvals[j];
      }
      wvals[i] *= cwvals[i];
    }
  }

  // Compute ztemp = (S*Z^{-1} - A*C0^{-1}*bx)
  memset(ztemp, 0, ncon*sizeof(double));

  // Compute the contribution from the weighing constraints
  if (nwcon > 0){
    double *wvals;
    int size = wtemp->getArray(&wvals);
    for ( int i = 0; i < ncon; i++ ){
      int one = 1;
      double *ewvals;
      Ew[i]->getArray(&ewvals);
      ztemp[i] = BLASddot(&size, wvals, &one, ewvals, &one);
    }
  }

  for ( int i = 0; i < ncon; i++ ){
    double *cvals, *avals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Ac[i]->getArray(&avals);

    double ydot = 0.0;
    int k = 0, remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ydot += avals[0]*bxvals[0]*cvals[0];
      avals++; bxvals++; cvals++; 
    }

    for ( int k = remainder; k < nvars; k += 4 ){
      ydot += (avals[0]*bxvals[0]*cvals[0] + 
	       avals[1]*bxvals[1]*cvals[1] +
	       avals[2]*bxvals[2]*cvals[2] + 
	       avals[3]*bxvals[3]*cvals[3]);
      avals += 4; bxvals += 4; cvals += 4;
    }

    ztemp[i] += ydot;
  }

  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    // Reduce the result to the root processor
    MPI_Reduce(MPI_IN_PLACE, ztemp, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }
  else {
    MPI_Reduce(ztemp, NULL, ncon, MPI_DOUBLE, MPI_SUM, 
	       opt_root, comm);
  }

  if (rank == opt_root){
    for ( int i = 0; i < ncon; i++ ){
      ztemp[i] *= -1.0;
    }

    int one = 1, info = 0;
    LAPACKdgetrs("N", &ncon, &one, 
		 Dmat, &ncon, dpiv, ztemp, &ncon, &info);
  }

  MPI_Bcast(ztemp, ncon, MPI_DOUBLE, opt_root, comm);

  if (nwcon > 0){
    // Compute yw = -Cw^{-1}*(Ew*yz + Aw*C^{-1}*bx)
    // First set yw <- - Ew*yz
    wtemp->zeroEntries();
    for ( int i = 0; i < ncon; i++ ){
      wtemp->axpy(-ztemp[i], Ew[i]);
    }

    // Compute yw <- Cw^{-1}*(yw - Aw*C^{-1}*bx);
    double *cvals, *cwvals, *ywvals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Cwvec->getArray(&cwvals);
    wtemp->getArray(&ywvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	ywvals[i] -= cvals[j]*bxvals[j];
      }
      ywvals[i] *= cwvals[i];
    }
  }

  // Compute the step in the design variables
  double *yxvals, *cvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);

  // Compute yx = C0^{-1}*(bx + A^{T}*yz)
  yx->copyValues(bx);
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(ztemp[i], Ac[i]);
  }

  if (nwcon > 0){
    double *ywvals;
    wtemp->getArray(&ywvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	yxvals[j] += ywvals[i];
      }
    }
  }

  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] *= cvals[i];
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
void ParOpt::setUpKKTSystem(){
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
      solveKKTDiagSystem(Z[i], xtemp);
      
      // Compute the dot products Z^{T}*K^{-1}*Z[i]
      xtemp->mdot(Z, size, &Ce[i*size]);
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

  1. p = K^{-1}*x
  2. ztemp = Z^{T}*p
  3. ztemp <- Ce^{-1}*ztemp
  4. rx = Z^{T}*ztemp
  5. p -= K^{-1}*rx
*/
void ParOpt::computeKKTStep(){
  // Get the size of the limited-memory BFGS subspace
  double b0;
  const double *d, *M;
  ParOptVec **Z;
  int size = qn->getLBFGSMat(&b0, &d, &M, &Z);

  // At this point the residuals are no longer required.
  solveKKTDiagSystem(rx, rc, rw, rs, rzl, rzu,
		     px, pz, pzw, ps, pzl, pzu);

  if (size > 0){
    // dz = Z^{T}*px
    px->mdot(Z, size, ztemp);
    
    // Compute dz <- Ce^{-1}*dz
    int one = 1, info = 0;
    LAPACKdgetrs("N", &size, &one, 
		 Ce, &size, cpiv, ztemp, &size, &info);
    
    // Compute rx = Z^{T}*dz
    xtemp->zeroEntries();
    for ( int i = 0; i < size; i++ ){
      xtemp->axpy(ztemp[i], Z[i]);
    }
    
    // Solve the digaonal system again, this time simplifying
    // the result due to the structure of the right-hand-side
    solveKKTDiagSystem(xtemp,
		       rx, rc, rw, rs, rzl, rzu);

    // Add the final contributions 
    px->axpy(-1.0, rx);
    pzw->axpy(-1.0, rw);
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
double ParOpt::evalMeritFunc( ParOptVec * xk, double *sk ){
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

  // Compute the sum of the squares of the weighting infeasibility
  double weight_infeas = 0.0;
  for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
    double val = 1.0;
    for ( int k = 0; k < nw; k++, j++ ){
      val -= xvals[j];
    }
    weight_infeas += val*val;
  }

  // Sum up the result from all processors
  double input[3];
  double result[3];
  input[0] = pos_result;
  input[1] = neg_result;
  input[2] = weight_infeas;
  MPI_Reduce(input, result, 3, MPI_DOUBLE, MPI_SUM, opt_root, comm);

  // Extract the result of the summation over all processors
  pos_result = result[0];
  neg_result = result[1];
  weight_infeas = result[2];
  
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
    infeas = sqrt(infeas) + sqrt(weight_infeas);

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

  // Compute the sum of the squares of the weighting infeasibility
  double weight_infeas = 0.0;
  for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
    double val = 1.0;
    for ( int k = 0; k < nw; k++, j++ ){
      val -= xvals[j];
    }
    weight_infeas += val*val;
  }

  // Sum up the result from all processors
  double input[5];
  double result[5];
  input[0] = pos_result;
  input[1] = neg_result;
  input[2] = pos_presult;
  input[3] = neg_presult;
  input[4] = weight_infeas;

  MPI_Reduce(input, result, 5, MPI_DOUBLE, MPI_SUM, opt_root, comm);

  // Extract the result of the summation over all processors
  pos_result = result[0];
  neg_result = result[1];
  pos_presult = result[2];
  neg_presult = result[3];
  weight_infeas = result[4];

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
    infeas = sqrt(infeas) + sqrt(weight_infeas);

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
    double merit = evalMeritFunc(rx, rs);

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
    else {
      // This is the final failure, evaluate the gradient
      // so that it is consistent on the final iteration
      int fail_gobj = prob->evalObjConGradient(x, g, Ac);
      ngeval++;
    }
  }

  // Set the new values of the variables
  x->axpy(alpha, px);
  zw->axpy(alpha, pzw);
  zl->axpy(alpha, pzl);
  zu->axpy(alpha, pzu);
  
  for ( int i = 0; i < ncon; i++ ){
    s[i] += alpha*ps[i];
    z[i] += alpha*pz[i];
  }
  
  // Compute the negative gradient of the Lagrangian using the
  // old gradient information with the new multiplier estimates
  y_qn->copyValues(g);
  y_qn->scale(-1.0);
  for ( int i = 0; i < ncon; i++ ){
    y_qn->axpy(z[i], Ac[i]);
  }

  // Evaluate the derivative
  int fail_gobj = prob->evalObjConGradient(rx, g, Ac);
  ngeval++;
  if (fail_gobj){
    fprintf(stderr, 
	    "Gradient evaluation failed at final line search\n");
  }

  // Add the new gradient of the Lagrangian with the new
  // multiplier estimates
  y_qn->axpy(1.0, g);
  for ( int i = 0; i < ncon; i++ ){
    y_qn->axpy(-z[i], Ac[i]);
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

  // Find an initial estimate of the Lagrange multipliers for the
  // inequality constraints
  if (init_starting_point){
    // Form the right-hand-side of the least squares eigenvalue
    // problem
    xtemp->copyValues(g);
    xtemp->axpy(-1.0, zl);
    xtemp->axpy(1.0, zu);

    for ( int i = 0; i < ncon; i++ ){
      z[i] = Ac[i]->dot(xtemp);
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
    }
    else {
      // The system cannot be solved, just assign positive
      // initial multipliers
      for ( int i = 0; i < ncon; i++ ){
	z[i] = 1.0;
      }
    }

    // Keep the Lagrange multipliers if they are within 
    // a reasonable range and they are positive.
    for ( int i = 0; i < ncon; i++ ){
      if (z[i] < 0.01 || z[i] > 100.0){
	z[i] = 1.0;
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

    // Set up the KKT diagonal system
    setUpKKTDiagSystem();

    // Set up the full KKT system
    setUpKKTSystem();

    // Solve for the KKT step
    computeKKTStep();

    // Check the KKT step
    if (k == major_iter_step_check){
      checkKKTStep();
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
    s_qn->copyValues(x);
    s_qn->scale(-1.0);

    // Keep track of the step length size
    double alpha = 1.0;
    int line_fail = 0;

    if (use_line_search){
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter
      double m0, dm0;
      evalMeritInitDeriv(max_x, &m0, &dm0);
      
      // The directional derivative is so small that we apply the
      // full step, regardless
      if (dm0 > -abs_res_tol*abs_res_tol){
	// Apply the full step
	alpha = 1.0;
	x->axpy(alpha, px);
	zw->axpy(alpha, pzw);
	zl->axpy(alpha, pzl);
	zu->axpy(alpha, pzu);
	
	for ( int i = 0; i < ncon; i++ ){
	  s[i] += alpha*ps[i];
	  z[i] += alpha*pz[i];
	}
	
	// Compute the negative gradient of the Lagrangian using the
	// old gradient information with the new multiplier estimates
	y_qn->copyValues(g);
	y_qn->scale(-1.0);
	for ( int i = 0; i < ncon; i++ ){
	  y_qn->axpy(z[i], Ac[i]);
	}

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
	y_qn->axpy(1.0, g);
	for ( int i = 0; i < ncon; i++ ){
	  y_qn->axpy(-z[i], Ac[i]);
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
      // Apply the full step
      x->axpy(alpha, px);
      zw->axpy(alpha, pzw);
      zl->axpy(alpha, pzl);
      zu->axpy(alpha, pzu);

      for ( int i = 0; i < ncon; i++ ){
	s[i] += alpha*ps[i];
	z[i] += alpha*pz[i];
      }
 
      // Compute the negative gradient of the Lagrangian using the
      // old gradient information with the new multiplier estimates
      y_qn->copyValues(g);
      y_qn->scale(-1.0);
      for ( int i = 0; i < ncon; i++ ){
	y_qn->axpy(z[i], Ac[i]);
      }

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
      y_qn->axpy(1.0, g);
      for ( int i = 0; i < ncon; i++ ){
	y_qn->axpy(-z[i], Ac[i]);
      }
    }

    // Complete the updated step
    s_qn->axpy(1.0, x);

    // Store the steps in x/z for printing later
    alpha_xprev = alpha*max_x;
    alpha_zprev = alpha*max_z;

    // Compute the Quasi-Newton update
    int up_type = 0;
    if (!line_fail){
      up_type = qn->update(s_qn, y_qn);
    }

    // Create a string to print to the screen
    if (rank == opt_root){
      // The string of unforseen events
      info[0] = '\0';
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
  
  xtemp->copyValues(x);
  xtemp->axpy(dh, px);

  // Evaluate the objective/constraints
  double fobj2;
  prob->evalObjCon(xtemp, &fobj2, rc);

  // Compute the projected derivative
  double pobj = g->dot(px);
  double pfd = (fobj2 - fobj)/dh;

  // rs = Ac*px
  px->mdot(Ac, ncon, rs);

  // Print out the results on the root processor
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    printf("Objective gradient test\n");
    printf("FD: %15.8e  Actual: %15.8e  Err: %15.4e\n",
	   pfd, pobj, pobj - pfd);

    printf("\nConstraint gradient test\n");
    for ( int i = 0; i < ncon; i++ ){
      double fd = (rc[i] - c[i])/dh;
      printf("Con[%3d] FD: %15.8e  Actual: %15.8e  Err: %15.4e\n",
	     i, fd, rs[i], fd - rs[i]);
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
void ParOpt::checkKKTStep(){
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
  qn->mult(px, rx);
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

  if (nwcon > 0){
    double *rxvals, *pzwvals, *zwvals;
    rx->getArray(&rxvals);
    zw->getArray(&zwvals);
    pzw->getArray(&pzwvals);
    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      for ( int k = 0; k < nw; k++, j++ ){
	rxvals[j] -= (pzwvals[i] + zwvals[i]);
      }
    }
  }

  double max_val = rx->maxabs();
  
  if (rank == opt_root){
    printf("max |H*px - Ac^{T}*pz - Aw^{T}*pzw - pzl + pzu + \
(g - Ac^{T}*z - Aw^{T}*zw - zl + zu)|: %10.4e\n", max_val);
  }
  
  max_val = 0.0;
  if (nwcon > 0){
    double *xvals, *pxvals; 
    x->getArray(&xvals);
    px->getArray(&pxvals);

    for ( int i = 0, j = nwstart; i < nwcon; i++, j += nwskip ){
      double val = 1.0;
      for ( int k = 0; k < nw; k++, j++ ){
	val -= xvals[j] + pxvals[j];
      }
      if (fabs(val) > max_val){
	max_val = val;
      }      
    }
  } 
  MPI_Allreduce(MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, comm);
  if (rank == opt_root){
    printf("max |Aw*(x + px) + e|: %10.4e\n", max_val);
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
    printf("max |z*ps + s*pz + (z*s - mu)|: %10.4e\n", max_val);
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
