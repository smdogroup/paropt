#include <math.h>
#include "ParOpt.h"
#include "ParOptBlasLapack.h"

/*
  The Parallel Optimizer 



*/
ParOpt::ParOpt( MPI_Comm _comm, int _nvars, int _ncon,
		double *_x, double *_lb, double *_ub,
		int max_lbfgs_subspace ){
  // Record the communicator
  comm = _comm;
  opt_root = 0;

  // The number of local variables
  nvars = _nvars;

  // The number of dense global constraints
  ncon = _ncon;

  // Calculate the total number of variable across all processors
  MPI_Allreduce(&nvars, &nvars_total, 1, MPI_INT, MPI_SUM, comm);

  // Allocate the quasi-Newton LBFGS approximation
  qn = new LBFGS(comm, nvars, max_lbfgs_subspace);

  // Set the values of the variables
  x = new ParOptVec(comm, nvars);
  lb = new ParOptVec(comm, nvars);
  ub = new ParOptVec(comm, nvars);

  // Set the values of the variables
  double *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
  lb->getArray(&lbvals);
  ub->getArray(&ubvals);
  memcpy(xvals, _x, nvars*sizeof(double));
  memcpy(lbvals, _lb, nvars*sizeof(double));
  memcpy(ubvals, _ub, nvars*sizeof(double));

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

  // Allocate storage for bfgs/constraint sized things
  int zsize = 2*max_lbfgs_subspace;
  if (ncon > zsize){
    ncon = zsize;
  }
  ztemp = new double[ zsize ];

  // Allocate space for the
  Dmat = new double[ ncon*ncon ];
  dpiv = new int[ ncon ];

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
  write_output_frequency = 10;
  barrier_param = 0.1;
  abs_res_tol = 1e-5;
  use_line_search = 1;
  max_line_iters = 9;
  rho_penalty_search = 0.0;
  penalty_descent_fraction = 0.3;
  armijo_constant = 1e-3;
  monotone_barrier_fraction = 0.25;
  monotone_barrier_power = 1.25;
  min_fraction_to_boundary = 0.95;
}

/*
  Free the data allocated during the creation of the object
*/
ParOpt::~ParOpt(){
  delete qn;
}

/*
  Compute the residual of the KKT system. This code utilizes the data
  stored internally in the ParOpt optimizer. The only input required
  is the given the governing equations.

  This code computes the following terms:

  rx  = -(g(x) - Ac^{T}*z - zl + zu) 
  rc  = -(c(x) - s)
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

  // Compute the residuals from the second KKT system:
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
    rzlvals[i] = (xvals[i] - lbvals[i])*zlvals[i] - barrier_param;
    rzuvals[i] = (ubvals[i] - xvals[i])*zuvals[i] - barrier_param;
  }

  *max_prime = rx->maxabs();
  double dual_zl = rzl->maxabs();
  double dual_zu = rzl->maxabs();
}

/*
  This function computes the terms required to solve the KKT system
  using a bordering method.  The initialization process computes the
  following matrix:
  
  C = b0 + zl/(x - lb) + zu/(ub - x)

  where C is diagonal whose components. The components of C are stored
  in Cvec. The code also computes a factorization of the matrix:

  D = Z^{-1}*S + A*C^{-1}*A^{T}

  which is required to compute the Schur complement.
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
   
  // Set the components of the diagonal matrix 
  double * cvals;
  Cvec->getArray(&cvals);

  // Retrive the diagonal entry for the BFGS update
  double b0;
  const double *d, *M;
  ParOptVec **Z;
  qn->getLBFGSMat(&b0, &d, &M, &Z);

  // Set the values of the c matrix
  for ( int i = 0; i < nvars; i++ ){
    cvals[i] = 1.0/(b0 + 
		    zlvals[i]/(xvals[i] - lbvals[i]) + 
		    zuvals[i]/(ubvals[i] - xvals[i]));
  }

  // Set the value of the D matrix
  memset(Dmat, 0, ncon*ncon*sizeof(double));

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

      int k = 0;
      int remainder = nvars % 4;
      for ( ; k < remainder; k++ ){
	Dmat[i + ncon*j] += aivals[0]*ajvals[0]/cvals[0];
	aivals++; ajvals++; cvals++;
      }

      for ( int k = nvars; k < nvars; k += 4 ){
	Dmat[i + ncon*j] += (aivals[0]*ajvals[0]/cvals[0] +
			     aivals[1]*ajvals[1]/cvals[1] +
			     aivals[2]*ajvals[2]/cvals[2] +
			     aivals[3]*ajvals[3]/cvals[3]);
	aivals += 4; ajvals += 4; cvals += 4;
      }
    }
  }

  // Populate the remainder of the matrix because it is 
  // symmetric
  for ( int j = 0; j < ncon; j++ ){
    for ( int i = j+1; i < ncon; i++ ){
      Dmat[i + ncon*j] = Dmat[j + ncon*i];
    }
  }

  // Reduce the result to the root processor
  MPI_Reduce(MPI_IN_PLACE, Dmat, ncon*ncon, MPI_DOUBLE, MPI_SUM, 
	     opt_root, comm);

  // Add the diagonal component to the matrix
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == opt_root){
    for ( int i = 0; i < ncon; i++ ){
      Dmat[i*(ncon + 1)] = s[i]/z[i];
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
  
  B0*yx - A^{T}*yz - yzl + yzu = bx
  A*yx - ys = bc

  With the additional equations:

  ys = Z^{-1}*bs - Z^{-1}*S*yz
  yzl = (X - Xl)^{-1}*(bzl - Zl*yx)
  yzu = (Xu - X)^{-1}*(bzu + Zu*yx)

  Substitution of these three equations yields the following system of
  equations:

  ((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))*yx + A^{T}*yz
  = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu

  and
  
  A*yx + Z^{-1}*S*yz = bc + Z^{-1}*bs.

  Setting the temporary vector: 
  
  d = bx + (X - Xl)^{-1}*bzl - (Xu - X)^{-1}*bzu,

  we can solve for yz by solving the following system of equations:

  [Z^{-1}*S + A*((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))^{-1}*A^{T} ]*yz
  = bc + Z^{-1}*bs - A*((B0 + (X - Xl)^{-1}*Zl + (Xu - X)^{-1}*Zu))^{-1}*d

  This is:

  Dmat*yz = bc + Z^{-1}*bs - A*C0^{-1}*d 

  Note: This code uses the temporary array xtemp, therefore, xtemp
  cannot be an input/output for this function, otherwise strange
  behavior will occur.
*/
void ParOpt::solveKKTDiagSystem( ParOptVec *bx, double *bc, double *bs,
				 ParOptVec *bzl, ParOptVec *bzu,
				 ParOptVec *yx, double *yz, double *ys,
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

  // Now, compute yz = (bc + S*Z^{-1} - A*C0^{-1}*d)
  memset(yz, 0, ncon*sizeof(double));
  for ( int i = 0; i < ncon; i++ ){
    double *cvals, *avals;
    xtemp->getArray(&dvals);
    Cvec->getArray(&cvals);
    Ac[i]->getArray(&avals);

    int k = 0;
    int remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      yz[i] -= avals[0]*dvals[0]/cvals[0];
      avals++; dvals++; cvals++; 
    }

    for ( int k = nvars; k < nvars; k += 4 ){
      yz[i] -= (avals[0]*dvals[0]/cvals[0] + 
		avals[1]*dvals[1]/cvals[1] +
		avals[2]*dvals[2]/cvals[2] + 
		avals[3]*dvals[3]/cvals[3]);
    }
  }

  // Reduce the result to the root processor
  MPI_Reduce(MPI_IN_PLACE, yz, ncon, MPI_DOUBLE, MPI_SUM, 
	     opt_root, comm);

  // Compute the full right-hand-
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    // Compute the full right-hand-side on the root processor
    // and solve for the Lagrange multipliers
    for ( int i = 0; i < ncon; i++ ){
      yz[i] += bc[i] + bs[i]/z[i];
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

  // Compute the step in the design variables
  double *yxvals, *cvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);

  // Compute yx = C0^{-1}*(d + A^{T}*yz)
  yx->copyValues(xtemp);
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] /= cvals[i];
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
				 ParOptVec *yx, double *yz, double *ys,
				 ParOptVec *yzl, ParOptVec *yzu ){
  // Now, compute yz = (S*Z^{-1} - A*C0^{-1}*bx)
  memset(yz, 0, ncon*sizeof(double));

  for ( int i = 0; i < ncon; i++ ){
    double *cvals, *avals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Ac[i]->getArray(&avals);

    int k = 0;
    int remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      yz[i] -= avals[0]*bxvals[0]/cvals[0];
      avals++; bxvals++; cvals++; 
    }

    for ( int k = nvars; k < nvars; k += 4 ){
      yz[i] -= (avals[0]*bxvals[0]/cvals[0] + 
		avals[1]*bxvals[1]/cvals[1] +
		avals[2]*bxvals[2]/cvals[2] + 
		avals[3]*bxvals[3]/cvals[3]);
    }
  }

  // Reduce the result to the root processor
  MPI_Reduce(MPI_IN_PLACE, yz, ncon, MPI_DOUBLE, MPI_SUM, 
	     opt_root, comm);

  // Compute the full right-hand-
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    int one = 1, info = 0;
    LAPACKdgetrs("N", &ncon, &one, 
		 Dmat, &ncon, dpiv, yz, &ncon, &info);
  }

  MPI_Bcast(yz, ncon, MPI_DOUBLE, opt_root, comm);

  // Compute the step in the slack variables 
  for ( int i = 0; i < ncon; i++ ){
    ys[i] = -(s[i]*yz[i])/z[i];
  }

  // Compute the step in the design variables
  double *yxvals, *cvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);

  // Compute yx = C0^{-1}*(bx + A^{T}*yz)
  yx->copyValues(bx);
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(yz[i], Ac[i]);
  }

  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] /= cvals[i];
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
  // Compute ztemp = (S*Z^{-1} - A*C0^{-1}*bx)
  memset(ztemp, 0, ncon*sizeof(double));

  for ( int i = 0; i < ncon; i++ ){
    double *cvals, *avals, *bxvals;
    bx->getArray(&bxvals);
    Cvec->getArray(&cvals);
    Ac[i]->getArray(&avals);

    int k = 0;
    int remainder = nvars % 4;
    for ( ; k < remainder; k++ ){
      ztemp[i] -= avals[0]*bxvals[0]/cvals[0];
      avals++; bxvals++; cvals++; 
    }

    for ( int k = nvars; k < nvars; k += 4 ){
      ztemp[i] -= (avals[0]*bxvals[0]/cvals[0] + 
		   avals[1]*bxvals[1]/cvals[1] +
		   avals[2]*bxvals[2]/cvals[2] + 
		   avals[3]*bxvals[3]/cvals[3]);
    }
  }

  // Reduce the result to the root processor
  MPI_Reduce(MPI_IN_PLACE, ztemp, ncon, MPI_DOUBLE, MPI_SUM, 
	     opt_root, comm);

  // Compute the full right-hand-
  int rank;
  MPI_Comm_rank(comm, &rank);

  if (rank == opt_root){
    int one = 1, info = 0;
    LAPACKdgetrs("N", &ncon, &one, 
		 Dmat, &ncon, dpiv, ztemp, &ncon, &info);
  }

  MPI_Bcast(ztemp, ncon, MPI_DOUBLE, opt_root, comm);

  // Compute the step in the design variables
  double *yxvals, *cvals;
  yx->getArray(&yxvals);
  Cvec->getArray(&cvals);

  // Compute yx = C0^{-1}*(d + A^{T}*yz)
  yx->copyValues(bx);
  for ( int i = 0; i < ncon; i++ ){
    yx->axpy(ztemp[i], Ac[i]);
  }

  for ( int i = 0; i < nvars; i++ ){
    yxvals[i] /= cvals[i];
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
  const double *d, *M;
  ParOptVec **Z;
  int size = qn->getLBFGSMat(&b0, &d, &M, &Z);

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
      Ce[i + j*size] -= M[i + j*size]/(d[i]*d[j]);
    }
  }

  int info = 0;
  LAPACKdgetrf(&size, &size, Ce, &size, cpiv, &info);
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
  solveKKTDiagSystem(rx, rc, rs, rzl, rzu,
		     px, pz, ps, pzl, pzu);

  // dz = Z^{T}*px
  px->mdot(Z, size, ztemp);
  
  // Compute dz <- Ce^{-1}*dz
  int one = 1, info = 0;
  LAPACKdgetrs("N", &size, &one, 
	       Ce, &size, cpiv, ztemp, &ncon, &info);

  // Compute rx = Z^{T}*dz
  xtemp->zeroEntries();
  for ( int i = 0; i < size; i++ ){
    xtemp->axpy(ztemp[i], Z[i]);
  }

  // Solve the digaonal system again, this time simplifying
  // the result due to the prescence of 
  solveKKTDiagSystem(xtemp,
		     rx, rc, rs, rzl, rzu);

  // Add the final contributions 
  px->axpy(-1.0, rx);
  pzl->axpy(-1.0, rzl);
  pzu->axpy(-1.0, rzu);
  
  // Add the terms from the 
  for ( int i = 0; i < ncon; i++ ){
    pz[i] -= rc[i];
    ps[i] -= rs[i];
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
double ParOpt::evalMeritFunc(){
  // Get the value of the lower/upper bounds and variables
  double *xvals, *lbvals, *ubvals;
  x->getArray(&xvals);
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
	pos_result += log(s[i]);
      }
      else {
	neg_result += log(s[i]);
      }
    }
    
    // Compute the infeasibility
    double infeas = 0.0;
    for ( int i = 0; i < ncon; i++ ){
      infeas += (c[i] - s[i])*(c[i] - s[i]);
    }
    infeas = sqrt(infeas);

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
    infeas = sqrt(infeas);

    // Compute the numerator term
    double numer = proj - barrier_param*(pos_presult + neg_presult);
    
    // Compute the first guess for the new
    double rho_hat = 0.0;
    if (infeas > 0.0){
      rho_hat = numer/((1 - penalty_descent_fraction)*max_x*infeas);
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
    merit = fobj - barrier_param*(pos_result + neg_result) + rho_penalty_search*infeas;
    pmerit = numer - rho_penalty_search*infeas;
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
  double alpha_old = 0.0;
  int fail = 1;

  for ( int j = 0; j < max_line_iters; j++ ){
    x->axpy((alpha - alpha_old), px);
    zl->axpy((alpha - alpha_old), pzl);
    zu->axpy((alpha - alpha_old), pzu);
    
    for ( int i = 0; i < ncon; i++ ){
      s[i] += (alpha - alpha_old)*ps[i];
      z[i] += (alpha - alpha_old)*pz[i];
    }

    // Evaluate the objective and constraints
    eval_objcon();

    // Evaluate the merit function
    double merit = evalMeritFunc();
    
    // Check the sufficient decrease condition
    if (merit < m0 + armijo_constant*alpha*dm0){
      // Evaluate the derivative
      eval_gobjcon();

      // We have successfully found a point satisfying the line 
      // search criteria
      fail = 0;
      break;
    }

    // Update the new value of alpha
    alpha_old = alpha;
    alpha = 0.5*alpha;
  }

  // Set the final value of alpha used in the line search iteration
  *_alpha = alpha;

  return fail;
}

/*
  Perform the optimization
*/
void ParOpt::optimize(){


  // Evaluate the objective, constraint and their gradients at the
  // current values of the design variables
  eval_objcon();
  eval_gobjcon();

  // If this is the starting point, find an initial estimate
  // of the Lagrange multipliers for the inequality constraints
  if (init_starting_point){
    double * C = new double[ ncon*ncon ];
    int * cpiv = new int[ ncon ];

    // Form the right-hand-side of the least squares eigenvalue
    // problem
    xtemp->copyValues(g);
    xtemp->axpy(-1.0, zl);
    xtemp->axpy(1.0, zu);

    for ( int i = 0; i < ncon; i++ ){
      z[i] = Ac[i]->dot(xtemp);
    }

    // This is not the most efficient code for this step,
    // but it will work
    for ( int i = 0; i < ncon; i++ ){
      Ac[i]->mdot(Ac, ncon, &C[i*ncon]);
    }

    // Compute the factorization
    int info;
    LAPACKdgetrf(&ncon, &ncon, C, &ncon, cpiv, &info);
    
    // Solve the linear system
    if (!info){
      int one = 1;
      LAPACKdgetrs("N", &ncon, &one, &ncon, C, &ncon, cpiv,
		   z, &ncon, &info);
    }
    else {
      // The system cannot be solved, just assign
      for ( int i = 0; i < ncon; i++ ){
	z[i] = 1.0;
      }
    }

    // Keep the Lagrange multipliers if they are within 
    // a reasonable range
    for ( int i = 0; i < ncon; i++ ){
      if (z[i] < 0.01 || z[i] > 100.0){
	z[i] = 1.0;
      }
    } 
 
    delete [] C;
    delete [] cpiv;
  }

  int converged = 0;
  
  for ( int k = 0; k < max_major_iters; k++ ){
    // Print out the current solution progress
    if (k % write_output_frequency){
      write_output();
    }

    // Compute the complementarity
    double comp = computeComp();
    
    // Compute the residual of the KKT system 
    double max_prime, max_dual, max_infeas;
    computeKKTRes(&max_prime, &max_dual, &max_infeas);

    // Compute the norm of the residuals
    double res_norm = max_prime;
    if (max_dual > res_norm){ res_norm = max_dual; }
    if (max_infeas > res_norm){ res_norm = max_infeas; }

    // Check for convergence
    if (res_norm < abs_res_tol && barrier_param < 0.1*abs_res_tol){
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
    pzl->scale(max_z);
    pzu->scale(max_z);

    for ( int i = 0; i < ncon; i++ ){
      ps[i] *= max_x;
      pz[i] *= max_z;
    }

    // Store the negative of the nonlinear components of the KKT
    // residual at the initial line search point. This will be used
    // in the quasi-Newton update scheme.
    y_qn->copyValues(g);
    for ( int i = 0; i < ncon; i++ ){
      y_qn->axpy(-z[i], Ac[i]);
    }
    y_qn->scale(-1.0);

    // Store the design variable locations
    s_qn->copyValues(x);
    s_qn->scale(-1.0);

    // Keep track of the step length size
    double alpha = 1.0;

    if (use_line_search){
      // Compute the initial value of the merit function and its
      // derivative and a new value for the penalty parameter
      double m0, dm0;
      evalMeritInitDeriv(max_x, &m0, &dm0);
      
      // Perform the line search
      alpha = lineSearch(&alpha, m0, dm0);
    }
    else {
      // Apply the full step
      x->axpy(alpha, px);
      zl->axpy(alpha, pzl);
      zu->axpy(alpha, pzu);

      for ( int i = 0; i < ncon; i++ ){
	s[i] += alpha*ps[i];
	z[i] += alpha*pz[i];
      }

      // Evaluate the objective, constraint and their gradients at the
      // current values of the design variables
      eval_objcon();
      eval_gobjcon();
    }
    
    // Set up the data for the quasi-Newton update
    y_qn->axpy(1.0, g);
    for ( int i = 0; i < ncon; i++ ){
      y_qn->axpy(-z[i], Ac[i]);
    }
   
    s_qn->axpy(1.0, x);
   
    // Compute the Quasi-Newton update
    qn->update(s_qn, y_qn);
  }
}
