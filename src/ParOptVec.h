#ifndef PAR_OPT_VEC_H
#define PAR_OPT_VEC_H

/*
  The following classes define the vector and limited-memory BFGS
  classes used by the parallel optimizer.
*/

#include "mpi.h"

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
  void set( double alpha );
  void zeroEntries();
  void copyValues( ParOptVec * vec );
  double norm();
  double maxabs();
  double dot( ParOptVec * vec );
  void mdot( ParOptVec ** vecs, int nvecs, double * output );
  void scale( double alpha );
  void axpy( double alpha, ParOptVec * x );
  int getArray( double ** array );

 private:
  MPI_Comm comm;
  int size;
  double * x;
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
class LBFGS {
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
  
  // Get the information for the limited-memory BFGS update
  // ------------------------------------------------------
  int getLBFGSMat( double * _b0, const double ** _d,
		   const double ** _M, ParOptVec *** Z );

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
  double *rz; // rz = Z^{T}*x

  // The update S/Y vectors
  ParOptVec **S, **Y;
  double b0; // The diagonal scalar

  // The M-matrix
  double *M, *M_factor;
  int *mfpiv; // The pivot array for the M-factorization

  // Data for the internal storage of M/M_factor
  double *B, *L, *D; 
  double *d0; // The diagonal matrix
};

#endif
