#ifndef PAR_OPT_QUASI_SEPARABLE_H
#define PAR_OPT_QUASI_SEPARABLE_H

/*
  Copyright (c) 2014-2017 Graeme Kennedy. All rights reserved
*/

#include "ParOptProblem.h"
#include <stdio.h>

/*
  The following code is designed to be used to implement 
  MMA-type methods that also include sparse constraints.
  There are two modes of operation:

  The first is to run the method of moving asymptotes (MMA), 
  a sequential, separable convex approximation technique, 
  developed by Svanberg, that is commonly used in topology
  optimization. This method cannot incorporate sparse constraints
  directly and so they are ignored.

  The second purpose is to set up and run a convex sub-problem
  where the objective is governed by the same approximation
  technique used in MMA, but the constraints and sparse constraints
  defined by the original problem class are linearized about
  the original point.
*/

class ParOptMMA : public ParOptProblem {
 public:
  ParOptMMA( ParOptProblem *_prob, int _use_true_mma=1 );
  ~ParOptMMA();

  // Update the problem
  int update();

  // Initialize data for the subproblem
  int initializeSubProblem( ParOptVec *x );

  // Get the optimized point
  void getOptimizedPoint( ParOptVec **x );

  // Get the asymptotes
  void getAsymptotes( ParOptVec **_L, ParOptVec **_U );

  // Compute the KKT error
  void computeKKTError( double *l1, double *linfty, double *infeas );

  // Set the print level
  void setPrintLevel( int _print_level );

  // Set parameters in the optimizer
  void setAsymptoteContract( double val );
  void setAsymptoteRelax( double val );
  void setInitAsymptoteOffset( double val );
  void setMinAsymptoteOffset( double val );
  void setMaxAsymptoteOffset( double val );
  void setBoundRelax( double val );

  // Set the output file (only on the root proc)
  void setOutputFile( const char *filename );

  // Create the design vectors
  ParOptVec *createDesignVec();
  ParOptVec *createConstraintVec();
  
  // Get the communicator for the problem
  MPI_Comm getMPIComm();

  // Function to indicate the type of sparse constraints
  int isDenseInequality();
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Get the variables and bounds from the problem
  void getVarsAndBounds( ParOptVec *x, ParOptVec *lb, ParOptVec *ub );

  // Evaluate the objective and constraints
  int evalObjCon( ParOptVec *x, ParOptScalar *fobj, ParOptScalar *cons );

  // Evaluate the objective and constraint gradients
  int evalObjConGradient( ParOptVec *x, ParOptVec *g, ParOptVec **Ac );

  // Evaluate the product of the Hessian with a given vector
  int evalHvecProduct( ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                       ParOptVec *px, ParOptVec *hvec );

  // Evaluate the constraints
  void evalSparseCon( ParOptVec *x, ParOptVec *out );
  
  // Compute the Jacobian-vector product out = J(x)*px
  void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                          ParOptVec *px, ParOptVec *out );

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                   ParOptVec *pzw, ParOptVec *out );

  // Add the inner product of the constraints to the matrix such 
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                              ParOptVec *cvec, ParOptScalar *A );

  // Over-write this function if you'd like to print out
  // something with the same frequency as the output files
  // -----------------------------------------------------
  void writeOutput( int iter, ParOptVec *x ){}

 private:
  // Initialize the data
  void initialize();

  // Solve the dual problem
  int solveDual();

  // Evaluate the dual gradient/hessian
  void evalDualGradient( ParOptScalar *grad, ParOptScalar *H,
                         ParOptScalar *x, ParOptScalar *ys,
                         const ParOptScalar *lambda,
                         const ParOptScalar *p0,
                         const ParOptScalar *q0,
                         ParOptScalar **pi,
                         ParOptScalar **qi,
                         const ParOptScalar *L,
                         const ParOptScalar *U,
                         const ParOptScalar *alpha,
                         const ParOptScalar *beta );

  // File pointer for the summary file - depending on the settings
  FILE *fp;

  // Settings for what to write out to a file or not...
  int print_level; // == 0 => no print, 1 MMA iters, 2 MMA+subproblem

  // Communicator for this problem
  MPI_Comm comm;

  // What is your intention here?? - to use MMA or not...
  int use_true_mma;

  // Parameters used in the problem
  double asymptote_contract; // Contract the asymptotes
  double asymptote_relax; // Relax the coefficient
  double init_asymptote_offset; // The initial asymptote offset
  double min_asymptote_offset; // Enforce a minimum fraction offset
  double max_asymptote_offset; // Enforce a minimum fraction offset
  double bound_relax; // Relax the bound when computing the KKT error

  // Keep track of the number of iterations
  int mma_iter;
  int subproblem_iter;

  int m; // The number of constraints (global)
  int n; // The number of design variables (local)

  // Pointer to the optimization problem
  ParOptProblem *prob;

  // The design variables, and the previous two vectors
  ParOptVec *xvec, *x1vec, *x2vec;

  // The values of the multipliers
  ParOptVec *lbvec, *ubvec;

  // The objective, constraint and gradient information
  ParOptScalar fobj, *cons;
  ParOptVec *gvec, **Avecs;

  // The assymptotes
  ParOptVec *Lvec, *Uvec;

  // The move limits
  ParOptVec *alphavec, *betavec;

  // The coefficients for the approximation
  ParOptVec *p0vec, *q0vec; // The objective coefs
  ParOptVec **pivecs, **qivecs; // The constraint coefs

  // The multiplier variables
  ParOptScalar *lambda;
  ParOptScalar *zlb;
  
  // The slack variables
  ParOptScalar *y;

  // The right-hand side for the constraints in the subproblem
  ParOptScalar *b;

  // Penalty parameters within the subproblem 
  ParOptScalar *c;

  // The sparse constraint vector
  ParOptVec *cwvec;
};

#endif // PAR_OPT_QUASI_SEPARABLE_H
