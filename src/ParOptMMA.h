#ifndef PAR_OPT_QUASI_SEPARABLE_H
#define PAR_OPT_QUASI_SEPARABLE_H

#include <stdio.h>

#include "ParOptInteriorPoint.h"
#include "ParOptOptions.h"
#include "ParOptProblem.h"

/*
  The following code is designed to be used to implement
  MMA-type methods that also include sparse constraints.
  There are two modes of operation:

  The first mode is to run the method of moving asymptotes (MMA),
  a sequential, separable convex approximation technique,
  developed by Svanberg, that is commonly used in topology
  optimization. This method cannot incorporate sparse constraints
  directly and so they are ignored.

  The second mode is can be used to set up and run a convex sub-problem
  where the objective is governed by the same approximation
  technique used in MMA, but the constraints and sparse constraints
  defined by the original problem class are linearized about
  the original point.
*/

class ParOptMMA : public ParOptProblem {
 public:
  ParOptMMA(ParOptProblem *_prob, ParOptOptions *_options);
  ~ParOptMMA();

  // Get the default option values
  static void addDefaultOptions(ParOptOptions *options);
  ParOptOptions *getOptions();

  // Optimize using MMA
  void optimize(ParOptInteriorPoint *optimizer);

  // Set the MMA iteration
  void setIteration(int _mma_iter);

  // Set the new values of the multipliers
  void setMultipliers(ParOptScalar *_z, ParOptVec *_zw = NULL,
                      ParOptVec *_zlvec = NULL, ParOptVec *_zuvec = NULL);

  // Initialize data for the subproblem
  int initializeSubProblem(ParOptVec *x);

  // Compute the KKT error based on the current multiplier estimates
  void computeKKTError(double *l1, double *linfty, double *infeas);

  // Get the optimized point
  void getOptimizedPoint(ParOptVec **x);

  // Get the asymptotes
  void getAsymptotes(ParOptVec **_L, ParOptVec **_U);

  // Get the previous design iterations
  void getDesignHistory(ParOptVec **_x1, ParOptVec **_x2);

  // Set the print level
  void setPrintLevel(int _print_level);

  // Set parameters in the optimizer
  void setAsymptoteContract(double val);
  void setAsymptoteRelax(double val);
  void setInitAsymptoteOffset(double val);
  void setMinAsymptoteOffset(double val);
  void setMaxAsymptoteOffset(double val);
  void setBoundRelax(double val);
  void setRegularization(double eps, double delta);

  // Create the design vectors
  ParOptVec *createDesignVec();
  ParOptVec *createConstraintVec();
  ParOptQuasiDefMat *createQuasiDefMat();

  // Get the communicator for the problem
  MPI_Comm getMPIComm();

  // Function to indicate the type of sparse constraints
  int isSparseInequality();
  int useLowerBounds();
  int useUpperBounds();

  // Get the variables and bounds from the problem
  void getVarsAndBounds(ParOptVec *x, ParOptVec *lb, ParOptVec *ub);

  // Evaluate the objective and constraints
  int evalObjCon(ParOptVec *x, ParOptScalar *fobj, ParOptScalar *cons);

  // Evaluate the objective and constraint gradients
  int evalObjConGradient(ParOptVec *x, ParOptVec *g, ParOptVec **Ac);

  // Evaluate the product of the Hessian with a given vector
  int evalHvecProduct(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *px, ParOptVec *hvec);

  // Evaluate the diagonal Hessian
  int evalHessianDiag(ParOptVec *x, ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *hdiag);

  // Evaluate the constraints
  void evalSparseCon(ParOptVec *x, ParOptVec *out);

  // Compute the Jacobian-vector product out = J(x)*px
  void addSparseJacobian(ParOptScalar alpha, ParOptVec *x, ParOptVec *px,
                         ParOptVec *out);

  // Compute the transpose Jacobian-vector product out = J(x)^{T}*pzw
  // -----------------------------------------------------------------
  void addSparseJacobianTranspose(ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *pzw, ParOptVec *out);

  // Add the inner product of the constraints to the matrix such
  // that A += J(x)*cvec*J(x)^{T} where cvec is a diagonal matrix
  void addSparseInnerProduct(ParOptScalar alpha, ParOptVec *x, ParOptVec *cvec,
                             ParOptScalar *A);

  // Over-write this function if you'd like to print out
  // something with the same frequency as the output files
  // -----------------------------------------------------
  void writeOutput(int iter, ParOptVec *x) {}

 private:
  // Initialize the data
  void initialize();

  // Set the output file (only on the root proc)
  void setOutputFile(const char *filename);

  // Print the options summary
  void printOptionsSummary(FILE *fp);

  // File pointer for the summary file - depending on the settings
  FILE *fp;
  int first_print;

  // Pointer to the optimization problem
  ParOptProblem *prob;

  // Options
  ParOptOptions *options;

  // Flag which controls the constraint approximation
  int use_true_mma;

  // Communicator for this problem
  MPI_Comm comm;

  // Keep track of the number of iterations
  int mma_iter;
  int subproblem_iter;

  int m;  // The number of constraints (global)
  int n;  // The number of design variables (local)

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
  ParOptVec *p0vec, *q0vec;      // The objective coefs
  ParOptVec **pivecs, **qivecs;  // The constraint coefs

  // The right-hand side for the constraints in the subproblem
  ParOptScalar *b;

  // The sparse constraint vector
  ParOptVec *cwvec;

  // Additional data required for computing the KKT conditions
  ParOptVec *rvec;

  // The multipliers/constraints
  ParOptScalar *z;
  ParOptVec *zwvec;
  ParOptVec *zlvec, *zuvec;
};

#endif  // PAR_OPT_QUASI_SEPARABLE_H
