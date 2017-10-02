#ifndef PAR_OPT_QUASI_SEPARABLE_H
#define PAR_OPT_QUASI_SEPARABLE_H

/*
  Copyright (c) 2014-2017 Graeme Kennedy. All rights reserved
*/

#include "ParOptProblem.h"

/*
  The following code can be used to set up and solve a quasi-separable
  approximation of the given ParOptProblem. This code is designed to
  be used to implement MMA-type methods that also include sparse constraints.
  Each separable approximation is convex, enabling the efficient use
  of Hessian-vector products.
*/

class ParOptMMA : public ParOptBase {
 public:
  ParOptMMA( ParOptProblem *_prob );
  ~ParOptMMA();

  // Update the problem
  int update();

 private:
  // Initialize the data
  void initialize();

  // Initialize data for the subproblem
  void initSubProblem( int iter );

  // Solve the dual problem
  void solveDual();

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

  // Communicator for this problem
  MPI_Comm comm;

  // Parameters used in the problem
  double asymptote_relax; // Relaxation coefficient default = 0.9

  // Keep track of the number of iterations
  int mma_iter;

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

  // The right-hand side for the constraints in the subproblem
  ParOptScalar *b;

  // Penalty parameters within the subproblem 
  ParOptScalar *c;
};

#endif // PAR_OPT_QUASI_SEPARABLE_H
