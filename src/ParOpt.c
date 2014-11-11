#include "ParOpt.h"

/*
  Create a parallel vector for optimization

  input:
  

*/
ParOptVec::ParOptVec( MPI_Comm _comm, int n ){
  comm = _comm;
  size = n;
  x = new double[ n ];
  memset(x, 0, size*sizeof(double));
}

/*
  Copy the values from the given vector
*/
void ParOptVec::copyValues( ParOptVec * vec ){
  int one = 1;
  BLAScopy(&size, vec->x, &one, x, &one);
}

/*
  Compute the l2 norm of the vector
*/
double ParOptVec::norm(){
  int one = 1;
  res = BLASnrm2(&size, x, &one);
  res *= res;

  MPI_Allreduce(&res, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return sqrt(sum);
}

/*
  Compute the l-infinity norm of the vector
*/
double ParOptVec::max(){
  int one = 1;
  int max_index = BLASidamax(&size, x, &one);
  double res = fabs(x[max_index]);

  double infty_norm = 0.0;
  MPI_Allreduce(&res, &infty_norm, 1, MPI_DOUBLE, MPI_MAX, comm);

  return infty_norm;
}

/*
  Compute the dot-product of two vectors and return the result.
*/
double ParOptVec::dot( ParOptVec * vec ){
  int one = 1;
  res = BLASdot(&size, x, &one, vec->x, &one);

  double sum = 0.0;
  MPI_Allreduce(&res, &sum, 1, TACS_MPI_TYPE, MPI_SUM, comm);

  return sum;
}

/*
  Compute the dot product of the
*/
void ParOptVec::scale( double alpha ){
  int one = 1;
  BLASscal(&size, &alpha, x, &one);
}

/*
  Compute: self <- self + alpha*x
*/
void ParOptVec::axpy( double alpha, ParOptVec * x ){
  int one = 1;
  BLASaxpy(&size, &alpha, vec->x, &one, x, &one);
}

/*
  Retrieve the locally stored values from the array
*/
int BVec::getArray( TacsScalar ** array ){
  *array = x;
  return size;
}


/*
  The following class implements the limited-memory BFGS update.
*/
LBFGSUpdate::LBFGSUpdate( MPI_Comm _comm, int _n, 
			  ){

  comm = _comm;
  n = _n;

}
 

/*
  Compute the BFGS update
*/
void LBFGSUpdate::update( ParOptVec * s, ParOptVec * y ){

  // Set the diagonal entries of the matrix
  double gamma = y->dot(y);
  double alpha = y->dot(s);
 
  // Compute the multiplication
  mult(s, r);
  
  // Compute dot(r, s)
  double beta = r->dot(s);

  if (alpha <= 0.2*beta){
      

  }

  // Set up the new values
  if (msub < msub_max){
    svecs[msub]->copyValues(s);
    yvecs[msub]->copyValues(r);

    // Compute L_{ij} = s_{i}^{T}*y_{j} for the row/column
    // corresponding to i = msub and j = msub
    for ( int i = 0; i < msub; i++ ){
      L[i + msub*msub_max] = svecs[i]->dot(yvecs[msub]);
    }

    for ( int j = 0; j <= msub; j++ ){
      L[m + j*msub_max] = svecs[msub]->dot(yvecs[j]);
    }
    
    // Update the size of the subspace
    msub++;
  }
  else {
    // Update the vector entires
    svecs[0]->copyValues(s);
    yvecs[0]->copyValues(y);
    
    ParOptVec * stemp = svecs[0];
    ParOptVec * ytemp = yvecs[0];

    for ( int ii = 0; i < msub-1; i++ ){
      svecs[i] = svecs[i+1];
      yvecs[i] = yvecs[i+1];
    }
  }

  // Compute the factorization of M
  /*
            # Compute the M matrix
            for i in xrange(m):
                for j in xrange(m):
                    self.M[i, j] = np.dot(self.S[:, i], 
                                          self.B0*self.S[:, j])
                
            # Fill in the off-diagonal matrices
            self.M[:m, m:] = L
            self.M[m:, :m] = L.T
s
            # Set the elements in the diagonal matrix
            for i in xrange(m):
                self.M[m+i, m+i] = -np.dot(self.S[:, i], self.Y[:, i])

            # Compute the LU-factorization of M
            self.M_inv = scipy.linalg.lu_factor(self.M)
  */
}


ParOpt::ParOpt( MPI_Comm _comm, int _nvars, int _ncon ){


}


/*
  Compute the residual of the KKT residuals at the current
  optimization step. This utilizes the data stored internally in the
  ParOpt optimizer. The only input required is the given the governing
  equations.  

  This code computes the following terms:

  rx  = g(x) - A^{T}*z - Aw^{T}*zw - zl + zu 
  rc  = c(x) - s
  rz  = S*z - mu*e 
  rzu = (x - xl)*zl - mu*e
  rzl = (ub - x)*zu - mu*e
*/
void ParOpt::computeKKTRes(){
  // Assemble the residual of the first KKT equation:
  // g(x) - A^{T}*z - Aw^{T}*zw - zl + zu
  rx->copyValues(gx);
  for ( int i = 0; i < ncon; i++ ){
    rx->axpy(z[i], Acon[i]);
  }

  // Insert code to handle the special residuals

  // Add the contribution from the lagrange multipliers
  rx->axpy(-1.0, zl);
  rx->axpy(1.0, zu);

  // Compute the residuals from the second KKT system:
  for ( int i = 0; i < ncon; i++ ){
    rc[i] = cx[i] - s[i];
  }
}

/*
  This function computes the terms required to solve the KKT system
  using a bordering method.


*/
void ParOpt::setUpKKTSystem( ParOptVec * ctemp ){
  
  double * cvals;
  ctemp->getArray(&cvals);

  for ( int i = 0; i < nvars; i++ ){
    cvals[i] = 1.0/(b0 + 
		    zlvals[i]/(xvals[i] - lbvals[i]) + 
		    zuvals[i]/(ubvals[i] - xvals[i]));
  }

  memset(Dkkt, 0, ncon*ncon*sizeof(double));

  for ( int i = 0; i < ncon; i++ ){
    for ( int j = i; j < ncon; j++ ){
      double dval = 0.0;
      cvals[];


	    
    }
  }


  /*
        # Set number of design variables/constraints
        self.n = x.shape[0]
        self.m = s.shape[0]

        # Keep pointers to the original data
        self.x = x
        self.lb = lb
        self.ub = ub
        self.s = s
        self.z = z
        self.zl = zl
        self.zu = zu
        self.A = A
        
        # Compute the diagonal matrix c 
        self.c = b0 + zl/(x - lb) + zu/(ub - x)

        # Compute and factor the matrix D = Z^{-1}S + A*C^{-1}*A^{T} 
        self.D = np.zeros((self.m, self.m))

        for i in xrange(self.m):
            self.D[i, i] = s[i]/z[i]

        for i in xrange(self.m):
            for j in xrange(self.m):
                self.D[i, j] += np.dot(self.A[i, :], self.A[j, :]/self.c)

        # Factor the matrix D
        self.D_factor = scipy.linalg.lu_factor(self.D)

        return
  */
}

void ParOpt::solveKKTSystem(){
  /*
        '''
        Solve the KKT system
        '''

        # Slice up the array for easier access
        bx = b[0:3*self.n:3]
        bl = b[1:3*self.n:3]
        bu = b[2:3*self.n:3]  
        bc = b[3*self.n:3*self.n+self.m]
        bs = b[3*self.n+self.m:]

        # Get the right-hand-side of the first equation
        d = (bx + bl/(self.x - self.lb) - bu/(self.ub - self.x))
        
        # Compute the right-hand-side for the Lagrange multipliers
        rz = (bc + bs/self.z - np.dot(self.A, d/self.c))

        # Compute the step in the Lagrange multipliers
        pz = scipy.linalg.lu_solve(self.D_factor, rz)

        # Compute the step in the slack variables
        ps = (bs - self.s*pz)/self.z

        # Compute the step in the design variables
        px = (d + np.dot(self.A.T, pz))/self.c

        # Compute the step in the bound Lagrange multipliers
        pzl = (bl - self.zl*px)/(self.x - self.lb)
        pzu = (bu + self.zu*px)/(self.ub - self.x)

        # Now create the output array and assign the values
        x = np.zeros(b.shape)

        x[0:3*self.n:3] = px
        x[1:3*self.n:3] = pzl
        x[2:3*self.n:3] = pzu
        x[3*self.n:3*self.n+self.m] = pz
        x[3*self.n+self.m:] = ps

        return x
  */
}

/*
  Compute the complementarity at the solution
*/
double ParOpt::computeComp(){
  
  // Get the total number of design variables
  double nvars_total = 0.0;

  MPI_Allreduce(nvars, );

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
  double input[2];
  input[0] = max_x;
  input[1] = max_z;

  MPI_Allreduce(input, output, 2, MPI_DOUBLE, MPI_MIN, comm);

  // Return the minimum values
  *_max_x = output[0];
  *_max_z = output[1];
}

void ParOpt::optimize(){


  // Evaluate the objective, constraint and their gradients at the
  // current values of the design variables
  eval_objcon();
  eval_gobjcon();

  // If this is the starting point, find an initial estimate
  // of the Lagrange multipliers for the inequality constraints
  if (init_starting_point){
    /*      
            # Estimate the Lagrange multipliers by finding the
            # minimum norm solution to the problem:
            # A^{T}*z = g - zl + zu
            rhs = np.dot(self.A, self.gobj - self.zl + self.zu)

            # Solve the normal equations
            C = np.dot(self.A, self.A.T)
            z = np.linalg.solve(C, rhs)

            # If the least squares multipliers lie on the interval [0,
            # 1e3] use them, otherwise keep the pre-assigned values
            for i in xrange(self.m):
                if z[i] >= 0.0 and z[i] < 1e3:
                    self.z[i] = z[i]
    */

  }

  int converged = 0;
  
  for ( int k = 0; k < max_major_iters; k++ ){
    // Print out the current solution progress
    if (k % write_output_frequency){

    }

    // Compute the residual of the KKT system 
    computeKKTRes();

    // Compute the complementarity
    double comp = ;
    

    // Compute the norm of the residuals
    double res_norm = ;

    // Check for convergence
    if (res_norm < abs_res_tol && mu < 0.1*abs_res_tol){
      converged = 1;
      break;
    }
    
    // Determine if the residual norm has been reduced
    // sufficiently in order to switch to a new barrier
    // problem
    if (res_norm < 10.0*mu){
      // Record the value of the old barrier function
      double mu_old = mu;

      // Compute the new barrier parameter: It is either:
      // 1. A fixed fraction of the old value
      // 2. A function mu**exp for some exp > 1.0
      // Point 2 ensures superlinear convergence (eventually)
      double mu_frac = monotone_barrier_fraction*mu;
      double mu_pow = pow(mu, monotone_barrier_power);

      mu = mu_frac;
      if (mu_pow < mu_frac){
	mu = mu_pow;
      }

      // Adjust the residual for the new barrier parameter
      // -------------------
      // res[1:3*self.n:3] -= (mu_old - self.mu)
      // res[2:3*self.n:3] -= (mu_old - self.mu)
      // res[3*self.n+self.m:] -= (mu_old - self.mu)
      // -------------------
      
      // Adjust the residual norm - exclude the only 
      // part that has changed
      res_norm = max(opt_norm, con_norm);

      // Reset the penalty parameter to zero
      rho_penalty_search = 0.0;
    }

    // Set up the KKT system of equations
    setUpKKTSystem();

    // Solve the KKT system
    solveKKTSystem();

    // Compute the maximum permitted line search lengths
    double tau = min_fraction_to_boundary;
    double tau_max = 1.0 - self.mu;
    if (tau_max >= tau){
      tau = tau_max;
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
    yupdate->copyValues(g);
    yupdate->axpy(-z[i], Acon[i]);
    yupdate->scale(-1.0);

    // Store the design variable locations
    supdate->copyValues(x);
    supdate->scale(-1.0);

    // Keep track of the step length size
    double alpha = 1.0;

    if (use_line_search){
      // Set the penalty parameter to the smallest 
      double rho_hat = getMinPenaltyParam();
      
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

      // Perform the line search
      alpha = lineSearch();
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
    yupdate->axpy(g);
    for ( int i = 0; i < ncon; i++ ){
      yupdate->axpy(-z[i], Acon[i]);
    }
   
    supdate->axpy(1.0, x);
   
    // Compute the Quasi-Newton update
    qn->update(supdate, yupdate);
  }
}

/*
  Evaluate the merit function at the current point, assuming that the
  objective and constraint values are up to date.

  The merit function is given as follows:

  varphi(alpha) = 
  
  f(x + alpha*px) + 
  mu*(log(s) + log(x - xl) + log(xu - x)) +
  rho*||c(x) - s||

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
    if (xvals[i] - lb_vals[i] > 1.0){ 
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

  // Compute the infeasibility
  double infeas;
  for ( int i = 0; i < ncon; i++ ){
    infeas += (c[i] - s[i])*(c[i] - s[i]);
  }
  infeas = sqrt(infeas);

  // Add the contribution from the constraints
  merit = fobj - mu*(result[0] + result[1]) + rho_penalty_search*infeas;
  for ( int i = 0; i < ncon; i++ ){
    result -= mu*log(s[i]);
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, opt_root, comm);

  return result;
}

/*
  Find the minimum value of the penalty parameter which will guarantee
  that we have a descent direction. Then, compute value of the merit
  function and its derivative, with the new corresponding penalty
  parameter.

  To evaluate the Here, we use a quadratic model of the objectivve
  function which uses the quasi-Newton Hessian approximation.

  Evaluate the penalty parameter required to make the given direction
  a descent direction.

*/
void ParOpt::getMinPenaltyParam( double m0, double dm0 ){
  // Compute the dot-product of the numerator
  double num = g->dot(px);

  if (res <infeas <= rel_opt_tol){
    
  }

  // Compute the infeasibility
  double infeas = 0.0;
  for ( int i = 0; i < ncon; i++ ){
    infeas += (c[i] - s[i])*(c[i] - s[i]);
  }
  
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
    if (xvals[i] - lb_vals[i] > 1.0){ 
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

  // Compute the infeasibility
  double infeas;
  for ( int i = 0; i < ncon; i++ ){
    infeas += (c[i] - s[i])*(c[i] - s[i]);
  }
  infeas = sqrt(infeas);

  double proj = g->dot(px);
  double num = 


  // Evaluate the contribution from the derivative of the lower/upper
  // bounds
  double pos_merit = 0.0, neg_merit = 0.0;
  double pos_pmerit = 0.0, neg_pmerit = 0.0;

  // Add the contribution from the lower/upper bounds
  for ( int i = 0; i < nvars; i++ ){
    merit -= mu*log(xvals[i] - lbvals[i]);
    merit -= mu*log(ubvals[i] - xvals[i]);

  }

  double pmerit = 0.0;
  double result = pos_pmerit + neg_pmerit;
  MPI_Allreduce(&result, pmerit, 

  num += g->dot(px);



  /*
  // Compute the numerator
  num = (np.dot(g, px) 
	 - self.mu*(np.sum(ps/s) + 
		    np.sum(px/(x - self.lb)) + 
		    np.sum(-px/(self.ub - x))))

    pf = self.penalty_descent_fraction

    // # Compute rho_hat using just the first-order condition
        rho_hat = d/((1.0 - pf)*a_x*infeas)

        if self.use_curvature_penalty:
            # Compute gamma = 0.5*px^{T}*H*px 
            r = np.zeros(px.shape)
            self.qn.mult(px, r)
            gamma = 0.5*(np.dot(r, px) + 
                         self.mu*(np.sum(ps**2/s**2) + 
                                  np.sum(px**2/(x - self.lb)**2) +
                                  np.sum(px**2/(self.ub - x)**2)))
            
            pf = self.penalty_descent_fraction

            # If the curvature contribution is positive, add it
            if gamma > 0.0:
                rho_hat = (d + gamma)/((1.0 - pf)*a_x*infeas)

                # If the penalty is too high, use the smaller value
                # from the first-order model only
                if rho_hat > 1e2:
                    rho_hat = d/((1.0 - pf)*a_x*infeas)
        
        return rho_hat
  */
}

/*
  Perform a backtracking line search from the current point along the
  specified direction. Note that this is a very simple line search
  without a second-order correction which may be required to alleviate
  the Maratos effect. (This should work regardless for compliance
  problems when the function should approximate a convex function.)

  input:

*/
int ParOpt::lineSearch( double * _alpha ){
  // Perform a backtracking line search until the sufficient decrease
  // conditions are satisfied 
  double alpha = *_alpha;
  double alpha_old = 0.0;
  int fail = 1;

  for ( int j = 0; j < max_line_iters; j++ ){
    x->axpy((alpha - alpha_old), px);
    pzl->axpy((alpha - alpha_old), pzl);
    pzu->axpy((alpha - alpha_old), pzu);
    
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
