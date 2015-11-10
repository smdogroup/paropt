#ifndef PAR_OPT_VEC_H
#define PAR_OPT_VEC_H

/*
  Copyright (c) 2014-2015 Graeme Kennedy. All rights reserved
*/

/*
  The following classes define the vector and limited-memory BFGS
  classes used by the parallel optimizer.
*/

#include "mpi.h"
#include "complexify.h"

#ifdef PAROPT_USE_COMPLEX
#define PAROPT_MPI_TYPE MPI_DOUBLE_COMPLEX
typedef cplx ParOptScalar;
#else
#define PAROPT_MPI_TYPE MPI_DOUBLE
typedef double ParOptScalar;
#endif // PAROPT_USE_COMPLEX

/*
  This vector class implements basic linear algebra operations
  required for design optimization.
*/
class ParOptVec {
 public:
  ParOptVec( MPI_Comm _comm, int n );
  ~ParOptVec();

  // Perform standard operations required for linear algebra
  // -------------------------------------------------------
  void set( ParOptScalar alpha );
  void zeroEntries();
  void copyValues( ParOptVec * vec );
  double norm();
  double maxabs();
  ParOptScalar dot( ParOptVec * vec );
  void mdot( ParOptVec ** vecs, int nvecs, ParOptScalar * output );
  void scale( ParOptScalar alpha );
  void axpy( ParOptScalar alpha, ParOptVec * x );
  int getArray( ParOptScalar ** array );
  int writeToFile( const char * filename );

 private:
  MPI_Comm comm;
  int size;
  ParOptScalar * x;
};

/*
  This is the abstract base class for compact limited-memory
  quasi-Newton update schemes.

  This class can be used to implement both limited-memory BFGS and SR1
  update schemes for quasi-Newton optimization methods.
*/
class CompactQuasiNewton {
 public:
  CompactQuasiNewton(){}

  // Reset the internal data
  // -----------------------
  virtual void reset() = 0;
  
  // Perform the BFGS update
  // -----------------------
  virtual int update( ParOptVec * s, ParOptVec * y ) = 0;
  
  // Perform a matrix-vector multiplication
  // --------------------------------------
  virtual void mult( ParOptVec * x, ParOptVec * y ) = 0;
  virtual void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec * y) = 0;

  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  virtual int getCompactMat( ParOptScalar * _b0, const ParOptScalar ** _d,
			     const ParOptScalar ** _M, ParOptVec *** Z ) = 0;

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
  LBFGS( MPI_Comm _comm, int _nvars, int _subspace_size );
  ~LBFGS();

  // Reset the internal data
  // -----------------------
  void reset();
  
  // Perform the BFGS update
  // -----------------------
  int update( ParOptVec * s, ParOptVec * y );
  
  // Perform a matrix-vector multiplication
  // --------------------------------------
  void mult( ParOptVec * x, ParOptVec * y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec * y);

  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  int getCompactMat( ParOptScalar * _b0, const ParOptScalar ** _d,
		     const ParOptScalar ** _M, ParOptVec *** Z );

  // Get the maximum size of the limited-memory BFGS
  // -----------------------------------------------
  int getMaxLimitedMemorySize();

 private:
  // Information about the parallel decomposition
  MPI_Comm comm;
  int nvars;

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
  LSR1( MPI_Comm _comm, int _nvars, int _subspace_size );
  ~LSR1();

  // Reset the internal data
  // -----------------------
  void reset();
  
  // Perform the BFGS update
  // -----------------------
  int update( ParOptVec * s, ParOptVec * y );
  
  // Perform a matrix-vector multiplication
  // --------------------------------------
  void mult( ParOptVec * x, ParOptVec * y );
  void multAdd( ParOptScalar alpha, ParOptVec *x, ParOptVec * y);

  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  int getCompactMat( ParOptScalar * _b0, const ParOptScalar ** _d,
		     const ParOptScalar ** _M, ParOptVec *** Z );

  // Get the maximum size of the limited-memory BFGS
  // -----------------------------------------------
  int getMaxLimitedMemorySize();

 private:
  // Information about the parallel decomposition
  MPI_Comm comm;
  int nvars;

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

#endif
