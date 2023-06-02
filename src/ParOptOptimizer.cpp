#include "ParOptOptimizer.h"

#include <string.h>

ParOptOptimizer::ParOptOptimizer(ParOptProblem *_problem,
                                 ParOptOptions *_options) {
  problem = _problem;
  problem->incref();

  options = _options;
  options->incref();

  ip = NULL;
  tr = NULL;
  mma = NULL;
  subproblem = NULL;
}

ParOptOptimizer::~ParOptOptimizer() {
  problem->decref();
  options->decref();
  if (ip) {
    ip->decref();
  }
  if (tr) {
    tr->decref();
  }
  if (mma) {
    mma->decref();
  }
  if (subproblem) {
    subproblem->decref();
  }
}

/*
  Get default optimization options
*/
void ParOptOptimizer::addDefaultOptions(ParOptOptions *options) {
  const char *optimizers[3] = {"ip", "tr", "mma"};
  options->addEnumOption("algorithm", "tr", 3, optimizers,
                         "The type of optimization algorithm");

  options->addStringOption("ip_checkpoint_file", NULL,
                           "Checkpoint file for the interior point method");

  ParOptInteriorPoint::addDefaultOptions(options);
  ParOptTrustRegion::addDefaultOptions(options);
  ParOptMMA::addDefaultOptions(options);
}

/*
  Get the options set into the optimizer
*/
ParOptOptions *ParOptOptimizer::getOptions() { return options; }

/*
  Get the problem class
*/
ParOptProblem *ParOptOptimizer::getProblem() { return problem; }

/*
  Perform the optimization
*/
void ParOptOptimizer::optimize() {
  int rank;
  MPI_Comm_rank(problem->getMPIComm(), &rank);

  // Check what type of optimization algorithm has been requirested
  int algo_type = 0;
  const char *algorithm = options->getEnumOption("algorithm");

  if (strcmp(algorithm, "ip") == 0) {
    algo_type = 1;
  } else if (strcmp(algorithm, "tr") == 0) {
    algo_type = 2;
  } else if (strcmp(algorithm, "mma") == 0) {
    algo_type = 3;
  } else {
    if (rank == 0) {
      fprintf(stderr,
              "ParOptOptimizer Error: Unrecognized algorithm option %s\n",
              algorithm);
    }
    return;
  }

  if (algo_type == 1) {
    if (tr && ip) {
      tr->decref();
      tr = NULL;
      ip->decref();
      ip = NULL;
    } else if (mma && ip) {
      mma->decref();
      mma = NULL;
      ip->decref();
      ip = NULL;
    }

    if (!ip) {
      ip = new ParOptInteriorPoint(problem, options);
      ip->incref();
    }

    const char *checkpoint = options->getStringOption("ip_checkpoint_file");
    ip->optimize(checkpoint);
  } else if (algo_type == 2) {
    if (mma && ip) {
      mma->decref();
      mma = NULL;
      ip->decref();
      ip = NULL;
    }

    // Create the trust region subproblem
    const char *qn_type = options->getEnumOption("qn_type");
    const int qn_subspace_size = options->getIntOption("qn_subspace_size");

    if (!subproblem) {
      ParOptCompactQuasiNewton *qn = NULL;
      if (strcmp(qn_type, "bfgs") == 0 || strcmp(qn_type, "scaled_bfgs") == 0) {
        ParOptLBFGS *bfgs = new ParOptLBFGS(problem, qn_subspace_size);
        if (strcmp(qn_type, "scaled_bfgs") == 0) {
          // This very specific type of bfgs is only used when ncon = 1
          int nvars, ncon, nwcon;
          problem->getProblemSizes(&nvars, &ncon, &nwcon);
          if (ncon != 1) {
            if (rank == 0) {
              fprintf(stderr,
                      "Can't use scaled_bfgs with more than one constraint!, "
                      "ncon = %d\n",
                      ncon);
            }
          } else {
            ParOptScaledQuasiNewton *scaled_bfgs =
                new ParOptScaledQuasiNewton(problem, bfgs);
            qn = scaled_bfgs;
          }
        } else {
          qn = bfgs;
        }
        qn->incref();

        const char *update_type = options->getEnumOption("qn_update_type");
        if (strcmp(update_type, "skip_negative_curvature") == 0) {
          bfgs->setBFGSUpdateType(PAROPT_SKIP_NEGATIVE_CURVATURE);
        } else if (strcmp(update_type, "damped_update") == 0) {
          bfgs->setBFGSUpdateType(PAROPT_DAMPED_UPDATE);
        }
      } else if (strcmp(qn_type, "sr1") == 0) {
        qn = new ParOptLSR1(problem, qn_subspace_size);
        qn->incref();
      }

      if (qn) {
        const char *diag_type = options->getEnumOption("qn_diag_type");
        if (strcmp(diag_type, "yty_over_yts") == 0) {
          qn->setInitDiagonalType(PAROPT_YTY_OVER_YTS);
        } else if (strcmp(diag_type, "yts_over_sts") == 0) {
          qn->setInitDiagonalType(PAROPT_YTS_OVER_STS);
        } else if (strcmp(diag_type, "inner_yty_over_yts") == 0) {
          qn->setInitDiagonalType(PAROPT_INNER_PRODUCT_YTY_OVER_YTS);
        } else {
          qn->setInitDiagonalType(PAROPT_INNER_PRODUCT_YTS_OVER_STS);
        }
      }

      subproblem = new ParOptQuadraticSubproblem(problem, qn);
      subproblem->incref();
    }

    if (!ip) {
      ip = new ParOptInteriorPoint(subproblem, options);
      ip->incref();
    }

    if (!tr) {
      tr = new ParOptTrustRegion(subproblem, options);
      tr->incref();
    }

    tr->optimize(ip);
  } else {  // algo_type == 3
    // Set up the MMA optimizer
    if (tr && ip) {
      tr->decref();
      tr = NULL;
      ip->decref();
      ip = NULL;
    }

    // Create the the mma object
    if (!mma) {
      mma = new ParOptMMA(problem, options);
      mma->incref();
    }

    if (!ip) {
      ip = new ParOptInteriorPoint(mma, options);
      ip->incref();
    }

    mma->optimize(ip);
  }
}

// Get the optimized point
void ParOptOptimizer::getOptimizedPoint(ParOptVec **x, ParOptScalar **z,
                                        ParOptVec **zw, ParOptVec **zl,
                                        ParOptVec **zu) {
  if (tr && ip) {
    tr->getOptimizedPoint(x);
    ip->getOptimizedPoint(NULL, z, zw, zl, zu);
  } else if (mma && ip) {
    mma->getOptimizedPoint(x);
    ip->getOptimizedPoint(NULL, z, zw, zl, zu);
  } else if (ip) {
    ip->getOptimizedPoint(x, z, zw, zl, zu);
  }
}

/*
  Set the trust-region subproblem
*/
void ParOptOptimizer::setTrustRegionSubproblem(
    ParOptTrustRegionSubproblem *_subproblem) {
  // Should check here if the subproblem's problem
  // is our problem, for consistency??
  if (_subproblem) {
    _subproblem->incref();
  }
  if (subproblem) {
    subproblem->decref();
  }
  subproblem = _subproblem;
}
