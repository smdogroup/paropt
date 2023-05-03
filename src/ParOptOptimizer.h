#ifndef PAROPT_OPTIMIZER_H
#define PAROPT_OPTIMIZER_H

#include "ParOptInteriorPoint.h"
#include "ParOptMMA.h"
#include "ParOptTrustRegion.h"

/*
  ParOptOptimizer is a generic interface to the optimizers that are
  avaialbe in the ParOpt library.

  These optimizers consist of the following:
  1. ParOptInteriorPoint
  2. ParOptTrustRegion
  3. ParOptMMA

  ParOptTrustRegion and ParOptMMA use the interior point code to solve
  the optimization subproblems that are formed at each iteration.
*/
class ParOptOptimizer : public ParOptBase {
 public:
  ParOptOptimizer(ParOptProblem *_problem, ParOptOptions *_options);
  ~ParOptOptimizer();

  // Get default optimization options
  static void addDefaultOptions(ParOptOptions *options);
  ParOptOptions *getOptions();

  // Get the optimization problem class
  ParOptProblem *getProblem();

  // Perform the optimization
  void optimize();

  // Get the optimized point
  void getOptimizedPoint(ParOptVec **x, ParOptScalar **z, ParOptVec **zw,
                         ParOptVec **zl, ParOptVec **zu);

  // Set the trust-region subproblem. This is required when non-standard
  // subproblems are used. This is an advanced feature this it not required
  // in most applications.
  void setTrustRegionSubproblem(ParOptTrustRegionSubproblem *_subproblem);

 private:
  // The problem instance
  ParOptProblem *problem;

  // The options
  ParOptOptions *options;

  // Store the optimizers
  ParOptInteriorPoint *ip;
  ParOptTrustRegion *tr;
  ParOptMMA *mma;

  // Specific object for the trust-region subproblem
  ParOptTrustRegionSubproblem *subproblem;
};

#endif  // PAROPT_OPTIMIZER_H
