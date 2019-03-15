#include "ParOptTrustRegion.h"
#include "ComplexStep.h"

/*
  Summary of the different trust region algorithm options
*/
static const int NUM_TRUST_REGION_PARAMS = 10;
static const char *trust_regions_parameter_help[][2] = {
  {"tr_size",
   "Float: Initial trust region radius size"},

  {"tr_min_size",
   "Float: Minimum trust region radius size"},

  {"tr_max_size",
   "Float: Maximum trust region radius size"},

  {"eta",
   "Float: Trust region step acceptance ratio of actual/predicted improvement"},

  {"bound_relax",
   "Float: Relax the bounds by this tolerance when computing KKT errors"},

  {"adaptive_gamma_update",
   "Boolean: Adaptively update the trust region "},

  {"max_tr_iterations",
   "Integer: Maximum number of trust region radius steps"},

  {"l1_tol",
   "Float: Convergence tolerance for the optimality error in the l1 norm"},

  {"linfty_tol",
   "Float: Convergence tolerance for the optimality error in the l-infinity norm"},

  {"infeas_tol",
   "Float: Convergence tolerance for feasibility in the l1 norm"}};

// Helper functions
inline ParOptScalar min2( ParOptScalar a, ParOptScalar b ){
  if (RealPart(a) < RealPart(b)){
    return a;
  }
  else {
    return b;
  }
}

inline ParOptScalar max2( ParOptScalar a, ParOptScalar b ){
  if (RealPart(a) > RealPart(b)){
    return a;
  }
  else {
    return b;
  }
}

ParOptTrustRegion::ParOptTrustRegion( ParOptProblem *_prob,
                                      ParOptCompactQuasiNewton *_qn,
                                      double _tr_size,
                                      double _tr_min_size,
                                      double _tr_max_size,
                                      double _eta,
                                      double _penalty_value,
                                      double _bound_relax ):
ParOptProblem(_prob->getMPIComm()){
  // Paropt problem instance
  prob = _prob;
  prob->incref();

  // Get the problem sizes
  prob->getProblemSizes(&n, &m, &nwcon, &nwblock);
  setProblemSizes(n, m, nwcon, nwblock);

  // Set the quasi-Newton method
  if (_qn){
    qn = _qn;
    qn->incref();
  }
  else {
    qn = NULL;
  }

  // Set the solution parameters
  tr_size = _tr_size;
  tr_min_size = _tr_min_size;
  tr_max_size = _tr_max_size;
  eta = _eta;
  bound_relax = _bound_relax;

  // Set the default output parameters
  write_output_frequency = 10;

  // Set default values for the convergence parameters
  adaptive_gamma_update = 1;
  max_tr_iterations = 200;
  l1_tol = 1e-6;
  linfty_tol = 1e-6;
  infeas_tol = 1e-5;
  gamma_max = 1e4;

  // Set the penalty parameters
  penalty_gamma = new double[ m ];
  for ( int i = 0; i < m; i++ ){
    penalty_gamma[i] = _penalty_value;
  }

  // Set the iteration count to zero
  iter_count = 0;

  // Create the vectors
  xk = prob->createDesignVec();  xk->incref();
  lk = prob->createDesignVec();  lk->incref();
  uk = prob->createDesignVec();  uk->incref();
  lb = prob->createDesignVec();  lb->incref();
  ub = prob->createDesignVec();  ub->incref();

  // Set default values for now before initialize
  lk->set(0.0);  uk->set(1.0);
  lb->set(0.0);  ub->set(1.0);
  xk->set(0.5);

  // Create the constraint Jacobian vectors
  fk = 0.0;
  gk = prob->createDesignVec();
  gk->incref();
  ck = new ParOptScalar[ m ];
  Ak = new ParOptVec*[ m ];
  for ( int i = 0; i < m; i++ ){
    Ak[i] = prob->createDesignVec();
    Ak[i]->incref();
  }

  // Create a temporary set of vectors
  ft = 0.0;
  gt = prob->createDesignVec();
  gt->incref();
  ct = new ParOptScalar[ m ];
  At = new ParOptVec*[ m ];
  for ( int i = 0; i < m; i++ ){
    At[i] = prob->createDesignVec();
    At[i]->incref();
  }

  // Create the temporary vector
  s = prob->createDesignVec();  s->incref();
  t = prob->createDesignVec();  t->incref();

  // Set the file pointer to NULL
  fp = NULL;
  print_level = 0;
  fp_log = NULL;
}

/*
  Delete the trust region object
*/
ParOptTrustRegion::~ParOptTrustRegion(){
  prob->decref();
  if (qn){
    qn->decref();
  }
  delete [] penalty_gamma;

  xk->decref();
  gk->decref();
  delete [] ck;
  for ( int i = 0; i < m; i++ ){
    Ak[i]->decref();
  }
  delete [] Ak;

  delete [] ct;
  gt->decref();
  for ( int i = 0; i < m; i++ ){
    At[i]->decref();
  }
  delete [] At;

  s->decref();
  t->decref();

  // Close the file when we quit
  if (fp){
    fclose(fp);
  }
  if (fp_log){
    fclose(fp_log);
  }
}

/*
  Set the trust region bounds
*/
void ParOptTrustRegion::setTrustRegionBounds( double tr,
                                              ParOptVec *x,
                                              ParOptVec *l,
                                              ParOptVec *u,
                                              ParOptVec *ltr,
                                              ParOptVec *utr ){
  ParOptScalar *xvals;
  ParOptScalar *lvals, *uvals;
  ParOptScalar *ltrvals, *utrvals;

  int size = x->getArray(&xvals);
  l->getArray(&lvals);
  u->getArray(&uvals);
  ltr->getArray(&ltrvals);
  utr->getArray(&utrvals);

  for ( int i = 0; i < size; i++ ){
    ltrvals[i] = max2(xvals[i] - tr, lvals[i]);
    utrvals[i] = min2(xvals[i] + tr, uvals[i]);
  }
}

/*
  Set the output file (only on the root proc)
*/
void ParOptTrustRegion::setOutputFile( const char *filename,
				       int _print_level,
				       const char *logname ){
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0){
    if (fp && fp != stdout){
      fclose(fp);
    }
    fp = fopen(filename, "w");

    if (fp){
      fprintf(fp, "ParOptTrustRegion: Parameter summary\n");
      for ( int i = 0; i < NUM_TRUST_REGION_PARAMS; i++ ){
        fprintf(fp, "%s\n%s\n\n",
                trust_regions_parameter_help[i][0],
                trust_regions_parameter_help[i][1]);
      }
    }
  }
  print_level = _print_level;
  // Create a log file for debugging
  if (print_level > 0){
    if (rank == 0){
      if (fp_log && fp_log != stdout){
	fclose(fp_log);
      }
      fp_log = fopen(logname, "w");
    }
  }
}
/*
  Write the parameters to the output file
*/
void ParOptTrustRegion::printOptionSummary( FILE *fp ){
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0){
    fprintf(fp, "ParOptTrustRegion options summary:\n");
    fprintf(fp, "%-30s %15g\n", "tr_size", tr_size);
    fprintf(fp, "%-30s %15g\n", "tr_min_size", tr_min_size);
    fprintf(fp, "%-30s %15g\n", "tr_max_size", tr_max_size);
    fprintf(fp, "%-30s %15g\n", "eta", eta);
    double value = 0.0;
    for ( int i = 0; i < m; i++ ){
      value += penalty_gamma[i];
    }
    fprintf(fp, "%-30s %15g\n", "avg(penalty_gamma)", value/m);
    fprintf(fp, "%-30s %15g\n", "bound_relax", bound_relax);
    fprintf(fp, "%-30s %15d\n", "adaptive_gamma_update", adaptive_gamma_update);
    fprintf(fp, "%-30s %15d\n", "max_tr_iterations", max_tr_iterations);
    fprintf(fp, "%-30s %15g\n", "l1_tol", l1_tol);
    fprintf(fp, "%-30s %15g\n", "linfty_tol", linfty_tol);
    fprintf(fp, "%-30s %15g\n", "infeas_tol", infeas_tol);
    fprintf(fp, "%-30s %15g\n", "gamma_max", gamma_max);
  }
}

/*
  Initialize the problem
*/
void ParOptTrustRegion::initialize(){
  // Get the lower/upper bounds
  prob->getVarsAndBounds(xk, lb, ub);

  // Set the lower/upper bounds for the trust region
  setTrustRegionBounds(tr_size, xk, lb, ub, lk, uk);

  // Evaluate objective constraints and gradients
  prob->evalObjCon(xk, &fk, ck);
  prob->evalObjConGradient(xk, gk, Ak);

  // Set the iteration count to zero
  iter_count = 0;

  int mpi_rank;
  MPI_Comm_rank(prob->getMPIComm(), &mpi_rank);
  if (mpi_rank == 0){
    if (fp){
      printOptionSummary(fp);
    }
    else {
      printOptionSummary(stdout);
    }
  }
}

/*
  Update the trust region problem
*/
void ParOptTrustRegion::update( ParOptVec *xt,
                                const ParOptScalar *z,
                                ParOptVec *zw,
                                double *infeas,
                                double *l1,
                                double *linfty ){
  // Compute the step
  s->copyValues(xt);
  s->axpy(-1.0, xk);

  // Compute the decrease in the model objective function
  ParOptScalar obj_reduc = -(gk->dot(s));

  if (qn){
    qn->mult(s, t);
    obj_reduc -= 0.5*t->dot(s);
  }

  int mpi_rank;
  MPI_Comm_rank(prob->getMPIComm(), &mpi_rank);

  if (mpi_rank == 0 && print_level > 0){
    FILE *outfp = stdout;
    if (fp_log){
      outfp = fp_log;
    }
    fprintf(outfp, "Iteration[%d]\n", iter_count);
    fflush(outfp);
  }

  // Compute the model infeasibility
  ParOptScalar infeas_model = 0.0;
  for ( int i = 0; i < m; i++ ){
    ParOptScalar cval = ck[i] + Ak[i]->dot(s);
    infeas_model += penalty_gamma[i]*max2(0.0, -cval);
    if (mpi_rank == 0 && print_level > 0){
      FILE *outfp = stdout;
      if (fp_log){
	outfp = fp_log;
      }
      fprintf(outfp,
	      "cval[%d]: %e\n", i, cval);
      fflush(outfp);
    }
  }

  // Evaluate the objective and constraints and their gradients at
  // the new, optimized point
  prob->evalObjCon(xt, &ft, ct);
  prob->evalObjConGradient(xt, gt, At);

  // Compute the infeasibilities of the last two iterations
  ParOptScalar infeas_k = 0.0;
  ParOptScalar infeas_t = 0.0;
  for ( int i = 0; i < m; i++ ){
    infeas_k += penalty_gamma[i]*max2(0.0, -ck[i]);
    infeas_t += penalty_gamma[i]*max2(0.0, -ct[i]);
    if (mpi_rank == 0 && print_level > 0){
      FILE *outfp = stdout;
      if (fp_log){
	outfp = fp_log;
      }
      fprintf(outfp, "ck[%d]: %e\n", i, ck[i]);
      fprintf(outfp, "ct[%d]: %e\n", i, ct[i]);
      fflush(outfp);
    }
  }

  // Compute the actual reduction and the predicted reduction
  ParOptScalar actual_reduc =
    (fk - ft + (infeas_k - infeas_t));
  ParOptScalar model_reduc =
    obj_reduc + (infeas_k - infeas_model);

  if (mpi_rank == 0 && print_level > 0){
    FILE *outfp = stdout;
    if (fp_log){
      outfp = fp_log;
    }
    fprintf(outfp,"Actual reduction for obj: %e\n", fk-ft);
    fprintf(outfp,"Model reduction for obj: %e\n", obj_reduc);
    fprintf(outfp,"Actual reduction for infeas: %e\n", infeas_k-infeas_t);
    fprintf(outfp,"Model reduction for infeas: %e\n", infeas_k-infeas_model);
    fflush(outfp);
  }

  // Compute the ratio
  ParOptScalar rho = actual_reduc/model_reduc;

  // Set whether to estimate the Lagrangian using the quasi-Newton method
  int use_quasi_newton_lagrangian_estimate = 0;
  if (qn){
    // Compute the difference between the gradient of the
    // Lagrangian between the current point and the previous point
    t->copyValues(gt);
    if (use_quasi_newton_lagrangian_estimate){
      for ( int i = 0; i < m; i++ ){
	t->axpy(-z[i], At[i]);
      }
      if (nwcon > 0){
	prob->addSparseJacobianTranspose(-1.0, xt, zw, t);
      }
    }

    t->axpy(-1.0, gk);
    if (use_quasi_newton_lagrangian_estimate){
      for ( int i = 0; i < m; i++ ){
	t->axpy(z[i], Ak[i]);
      }
      if (nwcon > 0){
	prob->addSparseJacobianTranspose(1.0, xk, zw, t);
      }
    }

    // Perform an update of the quasi-Newton approximation
    qn->update(s, t);
  }
  // Compute the KKT error at the current point
  computeKKTError(xt, gt, At, z, zw, l1, linfty);

  // Compute the infeasibility
  ParOptScalar infeas_new = 0.0;
  for ( int i = 0; i < m; i++ ){
    infeas_new += max2(0.0, -ct[i]);
  }
  *infeas = RealPart(infeas_new);

  // Compute the max absolute value
  double smax = RealPart(s->maxabs());

  // Check whether to accept the new point or not. If the trust region
  // radius size is at the lower bound, the step is always accepted
  if (RealPart(rho) >= eta || tr_size <= tr_min_size){
    fk = ft;
    xk->copyValues(xt);
    gk->copyValues(gt);
    for ( int i = 0; i < m; i++ ){
      ck[i] = ct[i];
      Ak[i]->copyValues(At[i]);
    }
  }
  else {
    // Set the step size to zero (rejected step)
    smax = 0.0;
  }

  // Set the new trust region radius
  if (RealPart(rho) < 0.25){
    tr_size = max2(0.25*tr_size, tr_min_size);
  }
  else if (RealPart(rho) > 0.75 &&
           smax >= 0.95*tr_size){
    tr_size = min2(2.0*tr_size, tr_max_size);
  }

  // Reset the trust region radius bounds
  setTrustRegionBounds(tr_size, xk, lb, ub, lk, uk);

  // Keep track of the max z/average z and max gamma/average gamma
  double zmax = 0.0, zav = 0.0, gmax = 0.0, gav = 0.0;
  for ( int i = 0; i < m; i++ ){
    zav += RealPart(z[i]);
    gav += penalty_gamma[i];
    if (RealPart(z[i]) > zmax){
      zmax = RealPart(z[i]);
    }
    if (penalty_gamma[i] > gmax){
      gmax = penalty_gamma[i];
    }
  }

  //int mpi_rank;
  MPI_Comm_rank(prob->getMPIComm(), &mpi_rank);
  if (mpi_rank == 0){
    FILE *outfp = stdout;
    if (fp){
      outfp = fp;
    }
    if (iter_count % 10 == 0){
      fprintf(outfp,
              "\n%5s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
              "iter", "fobj", "infeas", "l1", "linfty", "|x - xk|", "tr", "rho",
              "mod red.", "avg z", "max z", "avg pen.", "max pen.");
      fflush(outfp);
    }
    fprintf(outfp,
            "%5d %12.5e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            iter_count, fk, *infeas, *l1, *linfty, smax, tr_size, rho,
            model_reduc, zav/m, zmax, gav/m, gmax);
    fflush(outfp);
  }

  // Update the iteration counter
  iter_count++;
}

/*
  Set whether or not to adaptively update the penalty parameters
*/
void ParOptTrustRegion::setAdaptiveGammaUpdate( int truth ){
  adaptive_gamma_update = truth;
}

/*
  Set the maximum number of trust region steps
*/
void ParOptTrustRegion::setMaxTrustRegionIterations( int max_iters ){
  max_tr_iterations = max_iters;
}

/*
  Set the trust region stopping criterion values
*/
void ParOptTrustRegion::setTrustRegionTolerances( double _infeas_tol,
                                                  double _l1_tol,
                                                  double _linfty_tol ){
  infeas_tol = _infeas_tol;
  l1_tol = _l1_tol;
  linfty_tol = _linfty_tol;
}

/*
  Set the maximum value of the penalty parameters
*/
void ParOptTrustRegion::setPenaltyGammaMax( double _gamma_max ){
  gamma_max = _gamma_max;
}

/*
  Set the output frequency
*/
void ParOptTrustRegion::setOutputFrequency( int _write_output_frequency ){
  write_output_frequency = _write_output_frequency;
}

/*
  Perform the optimization problem
*/
void ParOptTrustRegion::optimize( ParOpt *optimizer ){
  if (optimizer->getOptProblem() != this){
    fprintf(stderr,
            "ParOptTrustRegion: The optimizer must be associated with this object\n");
    return;
  }

  // Set up the optimizer so that it uses the quasi-Newton approximation
  optimizer->setQuasiNewton(qn);
  optimizer->setUseQuasiNewtonUpdates(0); // Don't update the Hessian here

  // Set the initial values for gamma from the internal
  const double *temp_gamma = new double[m];
  optimizer->getPenaltyGamma(&temp_gamma);
  for (int i = 0; i < m; i++){
    penalty_gamma[i] = temp_gamma[i];
  }
  optimizer->setPenaltyGamma(penalty_gamma);

  // Allocate arrays to store infeasibility information
  ParOptScalar *con_infeas = NULL;
  ParOptScalar *model_con_infeas = NULL;
  ParOptScalar *best_con_infeas = NULL;
  if (adaptive_gamma_update){
    con_infeas = new ParOptScalar[ m ];
    model_con_infeas = new ParOptScalar[ m ];
    best_con_infeas = new ParOptScalar[ m ];
  }

  // Initialize the trust region problem for the first iteration
  initialize();

  // Iterate over the trust region subproblem until convergence
  for ( int i = 0; i < max_tr_iterations; i++ ){
    int mpi_rank;
    MPI_Comm_rank(prob->getMPIComm(), &mpi_rank);
    if (mpi_rank == 0){
      printf("Iteration[%d]\n", iter_count);
    }
    optimizer->checkGradients(1e-6);
    if (adaptive_gamma_update){
      // Set the penalty parameter to a large value
      optimizer->setPenaltyGamma(1e6);

      // Initialize the barrier parameter
      optimizer->setInitBarrierParameter(10.0);
      optimizer->resetDesignAndBounds();

      // Optimize the subproblem
      optimizer->optimize();

      // Get the design variables
      ParOptVec *x, *zw;
      ParOptScalar *z;
      optimizer->getOptimizedPoint(&x, &z, &zw, NULL, NULL);

      // Compute the best-case infeasibility achieved by setting the
      // penalty parameters to a large value
      for ( int j = 0; j < m; j++ ){
        ParOptScalar cj = ck[j] + Ak[j]->dot(x) - Ak[j]->dot(xk);
        best_con_infeas[j] = max2(0.0, -cj);
      }

      // Set the penalty parameters
      optimizer->setPenaltyGamma(penalty_gamma);
    }

    // Print out the current solution progress using the
    // hook in the problem definition
    if (i % write_output_frequency == 0){
      prob->writeOutput(i, xk);
    }

    // Initialize the barrier parameter
    optimizer->setInitBarrierParameter(10.0);
    optimizer->resetDesignAndBounds();

    // Optimize the subproblem
    optimizer->optimize();

    // Get the design variables
    ParOptVec *x, *zw;
    ParOptScalar *z;
    optimizer->getOptimizedPoint(&x, &z, &zw, NULL, NULL);

    if (adaptive_gamma_update){
      // Find the actual infeasibility reduction
      for ( int j = 0; j < m; j++ ){
        con_infeas[j] = max2(0.0, -ck[j]);

        ParOptScalar cj = ck[j] + Ak[j]->dot(x) - Ak[j]->dot(xk);
        model_con_infeas[j] = -min2(cj, 0.0);
      }
    }

    // Update the trust region based on the performance at the new
    // point.
    double infeas, l1, linfty;
    update(x, z, zw, &infeas, &l1, &linfty);

    // Check for convergence of the trust region problem
    if (infeas < infeas_tol){
      if (l1 < l1_tol ||
          linfty < linfty_tol){
        // Success!
        break;
      }
    }

    // Adapat the penalty parameters
    if (adaptive_gamma_update){
      for ( int i = 0; i < m; i++ ){
        // Compute the actual infeasibility reduction and the best
        // possible infeasibility reduction
        double infeas_reduction = con_infeas[i] - model_con_infeas[i];
        double best_reduction = con_infeas[i] - best_con_infeas[i];

	if (mpi_rank == 0 && print_level > 0){
	  FILE *outfp = stdout;
	  if (fp_log){
	    outfp = fp_log;
	  }
	  fprintf(outfp, "Infeas: con %e best %e model %e reduction %e %e\n",
		  con_infeas[i], best_con_infeas[i], model_con_infeas[i],
		  best_reduction, infeas_reduction);
	  fflush(outfp);
	}

        // If the ratio of the predicted to actual improvement is good,
        // and the constraints are satisfied, decrease the penalty
        // parameter. Otherwise, if the best case infeasibility is
        // significantly better, increase the penalty parameter.
        if (z[i] > infeas_tol &&
            con_infeas[i] < infeas_tol &&
            penalty_gamma[i] >= 2.0*z[i]){
          // Reduce gamma
          penalty_gamma[i] = 0.5*(penalty_gamma[i] + z[i]);
        }
        else if (con_infeas[i] > infeas_tol &&
                 0.995*best_reduction > infeas_reduction){
          // Increase gamma
          penalty_gamma[i] = min2(1.5*penalty_gamma[i], gamma_max);
        }
      }
    }
  }

  // Free the allocated data
  if (adaptive_gamma_update){
    delete [] con_infeas;
    delete [] model_con_infeas;
    delete [] best_con_infeas;
  }
}

/*
  Compute the KKT error based on the current values of the multipliers
  set in ParOptMMA. If you do not update the multipliers, you will not
  get the correct KKT error.
*/
void ParOptTrustRegion::computeKKTError( ParOptVec *xv,
                                         ParOptVec *g,
                                         ParOptVec **A,
                                         const ParOptScalar *z,
                                         ParOptVec *zw,
                                         double *l1,
                                         double *linfty ){
  // Get the lower/upper bounds for the variables
  ParOptScalar *l, *u;
  lb->getArray(&l);
  ub->getArray(&u);

  // Get the current values of the design variables
  ParOptScalar *x;
  xv->getArray(&x);

  // Compute the KKT residual r = g - A^{T}*z
  t->copyValues(g);
  for ( int i = 0; i < m; i++ ){
    t->axpy(-z[i], A[i]);
  }

  // If zw exists, compute r = r - Aw^{T}*zw
  if (nwcon > 0){
    prob->addSparseJacobianTranspose(-1.0, xv, zw, t);
  }

  // Set the infinity norms
  double l1_norm = 0.0;
  double infty_norm = 0.0;

  // Get the vector of values
  ParOptScalar *r;
  t->getArray(&r);

  for ( int j = 0; j < n; j++ ){
    double w = RealPart(r[j]);

    // Check if we're on the lower bound
    if ((x[j] <= l[j] + bound_relax) && w > 0.0){
      w = 0.0;
    }

    // Check if we're on the upper bound
    if ((x[j] >= u[j] - bound_relax) && w < 0.0){
      w = 0.0;
    }

    // Add the contribution to the l1/infinity norms
    double tw = fabs(w);
    l1_norm += tw;
    if (tw >= infty_norm){
      infty_norm = tw;
    }
  }

  // All-reduce the norms across all processors
  MPI_Allreduce(&l1_norm, l1, 1, MPI_DOUBLE,
                MPI_SUM, prob->getMPIComm());
  MPI_Allreduce(&infty_norm, linfty, 1, MPI_DOUBLE,
                MPI_MAX, prob->getMPIComm());
}

/*
  Create a design vector
*/
ParOptVec *ParOptTrustRegion::createDesignVec(){
  return prob->createDesignVec();
}

/*
  Create the sparse constraint vector
*/
ParOptVec *ParOptTrustRegion::createConstraintVec(){
  return prob->createConstraintVec();
}

/*
  Get the communicator for the problem
*/
MPI_Comm ParOptTrustRegion::getMPIComm(){
  return prob->getMPIComm();
}

/*
  Functions to indicate the type of sparse constraints
*/
int ParOptTrustRegion::isDenseInequality(){
  return prob->isDenseInequality();
}

int ParOptTrustRegion::isSparseInequality(){
  return prob->isSparseInequality();
}

int ParOptTrustRegion::useLowerBounds(){
  return 1;
}

int ParOptTrustRegion::useUpperBounds(){
  return 1;
}

// Get the variables and bounds from the problem
void ParOptTrustRegion::getVarsAndBounds( ParOptVec *x, ParOptVec *l,
                                          ParOptVec *u ){
  x->copyValues(xk);
  l->copyValues(lk);
  u->copyValues(uk);
}

/*
  Evaluate the objective and constraint functions
*/
int ParOptTrustRegion::evalObjCon( ParOptVec *x, ParOptScalar *fobj,
                                   ParOptScalar *cons ){
  s->copyValues(x);
  s->axpy(-1.0, xk);

  // Compute the objective function
  *fobj = fk + gk->dot(s);
  if (qn){
    qn->mult(s, t);
    *fobj += 0.5*s->dot(t);
  }

  // Compute the constraint functions
  for ( int i = 0; i < m; i++ ){
    cons[i] = ck[i] + Ak[i]->dot(s);
  }

  return 0;
}

/*
  Evaluate the objective and constraint gradients
*/
int ParOptTrustRegion::evalObjConGradient( ParOptVec *x, ParOptVec *g,
                                           ParOptVec **Ac ){
  // Copy the values of constraint gradient
  for ( int i = 0; i < m; i++ ){
    Ac[i]->copyValues(Ak[i]);
  }

  s->copyValues(x);
  s->axpy(-1.0, xk);

  // Evaluate the gradient of the quadratic objective
  if (qn){
    qn->mult(s, g);
    g->axpy(1.0, gk);
  }
  else {
    g->copyValues(gk);
  }

  return 0;
}

/*
  Evaluate the constraints
*/
void ParOptTrustRegion::evalSparseCon( ParOptVec *x, ParOptVec *out ){
  prob->evalSparseCon(xk, out);
  prob->addSparseJacobian(1.0, xk, x, out);
  prob->addSparseJacobian(-1.0, xk, xk, out);
}

/*
  Compute the Jacobian-vector product out = J(x)*px
*/
void ParOptTrustRegion::addSparseJacobian( ParOptScalar alpha,
                                           ParOptVec *x,
                                           ParOptVec *px,
                                           ParOptVec *out ){
  prob->addSparseJacobian(alpha, xk, px, out);
}

/*
  Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
*/
void ParOptTrustRegion::addSparseJacobianTranspose( ParOptScalar alpha,
                                                    ParOptVec *x,
                                                    ParOptVec *pzw,
                                                    ParOptVec *out ){
  prob->addSparseJacobianTranspose(alpha, xk, pzw, out);
}

/*
  Add the inner product of the constraints to the matrix such
  that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
*/
void ParOptTrustRegion::addSparseInnerProduct( ParOptScalar alpha,
                                               ParOptVec *x,
                                               ParOptVec *cvec,
                                               ParOptScalar *A ){
  prob->addSparseInnerProduct(alpha, xk, cvec, A);
}

/*
  Get the gradients evaluated at the current point
*/
int ParOptTrustRegion::getGradients( ParOptVec **_gk, ParOptVec ***_Ak ){
  *_gk = gk;
  *_Ak = Ak;

  return m;
}
