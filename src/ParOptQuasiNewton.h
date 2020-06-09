#ifndef PAROPT_QUASI_NEWTON_H
#define PAROPT_QUASI_NEWTON_H

#include "ParOptVec.h"
#include "ParOptProblem.h"

enum ParOptBFGSUpdateType { PAROPT_SKIP_NEGATIVE_CURVATURE,
                            PAROPT_DAMPED_UPDATE };

/**
  This is the abstract base class for compact limited-memory
  quasi-Newton update schemes.

  This class can be used to implement both limited-memory BFGS and SR1
  update schemes for quasi-Newton optimization methods.
*/
class ParOptCompactQuasiNewton : public ParOptBase {
 public:
  ParOptCompactQuasiNewton(){}
  virtual ~ParOptCompactQuasiNewton(){}

  // Reset the internal data
  virtual void reset() = 0;

  // Perform the quasi-Newton update with the specified multipliers
  virtual int update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw,
                      ParOptVec *s, ParOptVec *y ) = 0;

  // Update the approximation with only multiplier values - this is used
  // only for certain classes of compact Hessian approximations and does
  // not need to be implemented in general.
  virtual int update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw ){
    return 0;
  }

  // Perform a matrix-vector multiplication
  virtual void mult( ParOptVec *x, ParOptVec *y ) = 0;

  // Perform a matrix-vector multiplication and add the result to y
  virtual void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y ) = 0;

  // Get the compact representation for the limited-memory quasi-Newton method
  virtual int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                             const ParOptScalar **_M, ParOptVec ***Z ) = 0;

  // Get the maximum size of the compact representation
  virtual int getMaxLimitedMemorySize() = 0;
};

/**
  This class implements a limited-memory BFGS updating scheme based on
  computed differences in the step and Lagrange graidents during a
  line search.

  This is based on the paper by Byrd, Nocedal and Schnabel,
  "Representations of quasi-Newton matrices and their use in
  limited-memory methods".

  The limited-memory BFGS formula takes the following form:

  b0*I - Z*diag{d)*M^{-1}*diag{d}*Z^{T}

  Here b0 is a scalar, d is a vector whose entries are either 1.0 or
  b0, M is a matrix and Z is a rectagular matrix stored as a series of
  vectors.

  Note that this class implements a damped update when the curvature
  condition is violated.
*/
class ParOptLBFGS : public ParOptCompactQuasiNewton {
 public:
  ParOptLBFGS( ParOptProblem *prob, int _subspace_size );
  ~ParOptLBFGS();

  // Set the curvature update type
  void setBFGSUpdateType( ParOptBFGSUpdateType _hessian_update_type );

  // Reset the internal data
  void reset();

  // Perform the BFGS update
  int update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw,
              ParOptVec *s, ParOptVec *y );

  // Perform a matrix-vector multiplication
  void mult( ParOptVec *x, ParOptVec *y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y);

  // Get the information for the limited-memory BFGS update
  int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                     const ParOptScalar **_M, ParOptVec ***Z );

  // Get the maximum size of the limited-memory BFGS
  int getMaxLimitedMemorySize();

 protected:
  // Store the type of curvature handling update
  ParOptBFGSUpdateType hessian_update_type;

  // Set the finite-precision tolerance
  double epsilon_precision;

  // The size of the BFGS subspace
  int msub, msub_max;

  // The full list of vectors
  ParOptVec **Z;

  // Temporary data for internal usage
  ParOptVec *r;
  ParOptScalar *rz; // rz = Z^{T}*x

  // The update S/Y vectors
  ParOptVec **S, **Y;
  ParOptScalar b0; // The diagonal scalar

  // The M-matrix
  ParOptScalar *M, *M_factor;
  int *mfpiv; // The pivot array for the M-factorization

  // Data for the internal storage of M/M_factor
  ParOptScalar *B, *L, *D;
  ParOptScalar *d0; // The diagonal matrix
};

/**
  This class implements a limited-memory SR1 updating scheme based on
  computed differences in the step and Lagrange graidents during a
  line search.

  This is based on the paper by Byrd, Nocedal and Schnabel,
  "Representations of quasi-Newton matrices and their use in
  limited-memory methods".

  The limited-memory SR1 formula takes the following form:

  b0*I - Z*M^{-1}*Z^{T}

  Here b0 is a scalar, M is a matrix and Z is a rectagular matrix
  stored as a series of vectors.
*/
class ParOptLSR1 : public ParOptCompactQuasiNewton {
 public:
  ParOptLSR1( ParOptProblem *prob, int _subspace_size );
  ~ParOptLSR1();

  // Reset the internal data
  void reset();

  // Perform the BFGS update
  int update( ParOptVec *x, const ParOptScalar *z, ParOptVec *zw,
              ParOptVec *s, ParOptVec *y );

  // Perform a matrix-vector multiplication
  void mult( ParOptVec *x, ParOptVec *y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y);

  // Get the information for the limited-memory BFGS update
  int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                     const ParOptScalar **_M, ParOptVec ***Z );

  // Get the maximum size of the limited-memory BFGS
  int getMaxLimitedMemorySize();

 protected:
  // The size of the BFGS subspace
  int msub, msub_max;

  // Set the finite-precision tolerance
  double epsilon_precision;

  // The full list of vectors
  ParOptVec **Z;

  // Temporary data for internal usage
  ParOptVec *r;
  ParOptScalar *rz; // rz = Z^{T}*x

  // The update S/Y vectors
  ParOptVec **S, **Y;
  ParOptScalar b0; // The diagonal scalar

  // The M-matrix
  ParOptScalar *M, *M_factor;
  int *mfpiv; // The pivot array for the M-factorization

  // Data for the internal storage of M/M_factor
  ParOptScalar *B, *L, *D;
  ParOptScalar *d0; // The diagonal matrix
};

#endif // PAROPT_QUASI_NEWTON_H
