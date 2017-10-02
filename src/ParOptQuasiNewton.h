#ifndef PAROPT_QUASI_NEWTON_H
#define PAROPT_QUASI_NEWTON_H

/*
  Copyright (c) 2014-2016 Graeme Kennedy. All rights reserved
*/

#include "ParOptVec.h"
#include "ParOptProblem.h"

/*
  This is the abstract base class for compact limited-memory
  quasi-Newton update schemes.

  This class can be used to implement both limited-memory BFGS and SR1
  update schemes for quasi-Newton optimization methods.
*/
class CompactQuasiNewton : public ParOptBase {
 public:
  CompactQuasiNewton(){}
  virtual ~CompactQuasiNewton(){}

  // Reset the internal data
  // -----------------------
  virtual void reset() = 0;
  
  // Perform the BFGS update
  // -----------------------
  virtual int update( ParOptVec *s, ParOptVec *y ) = 0;
  
  // Perform a matrix-vector multiplication
  // --------------------------------------
  virtual void mult( ParOptVec *x, ParOptVec *y ) = 0;
  virtual void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y) = 0;

  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  virtual int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                             const ParOptScalar **_M, ParOptVec ***Z ) = 0;

  // Get the maximum size of the limited-memory BFGS
  // -----------------------------------------------
  virtual int getMaxLimitedMemorySize() = 0;
};

/*
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
class LBFGS : public CompactQuasiNewton {
 public:
  enum BFGSUpdateType { SKIP_NEGATIVE_CURVATURE, DAMPED_UPDATE };

  LBFGS( ParOptProblem *prob, int _subspace_size );
  ~LBFGS();

  // Set the curvature update type
  // -----------------------------
  void setBFGSUpdateType( BFGSUpdateType _hessian_update_type );

  // Reset the internal data
  // -----------------------
  void reset();
  
  // Perform the BFGS update
  // -----------------------
  int update( ParOptVec *s, ParOptVec *y );
  
  // Perform a matrix-vector multiplication
  // --------------------------------------
  void mult( ParOptVec *x, ParOptVec *y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y);

  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                     const ParOptScalar **_M, ParOptVec ***Z );

  // Get the maximum size of the limited-memory BFGS
  // -----------------------------------------------
  int getMaxLimitedMemorySize();

 private:
  // Store the type of curvature handling update
  BFGSUpdateType hessian_update_type;

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

/*
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
class LSR1 : public CompactQuasiNewton {
 public:
  LSR1( ParOptProblem *prob, int _subspace_size );
  ~LSR1();

  // Reset the internal data
  // -----------------------
  void reset();
  
  // Perform the BFGS update
  // -----------------------
  int update( ParOptVec *s, ParOptVec *y );
  
  // Perform a matrix-vector multiplication
  // --------------------------------------
  void mult( ParOptVec *x, ParOptVec *y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec *y);

  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  int getCompactMat( ParOptScalar *_b0, const ParOptScalar **_d,
                     const ParOptScalar **_M, ParOptVec ***Z );

  // Get the maximum size of the limited-memory BFGS
  // -----------------------------------------------
  int getMaxLimitedMemorySize();

 private:
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

#endif // PAROPT_QUASI_NEWTON_H
