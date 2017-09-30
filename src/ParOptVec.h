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
#include <complex>

// Define the complex ParOpt type
typedef std::complex<double> ParOptComplex;

// Set the type of the ParOptScalar
#ifdef PAROPT_USE_COMPLEX
#define PAROPT_MPI_TYPE MPI_DOUBLE_COMPLEX
typedef std::complex<double> ParOptScalar;
#else
#define PAROPT_MPI_TYPE MPI_DOUBLE
typedef double ParOptScalar;
#endif // PAROPT_USE_COMPLEX

/*
  ParOpt base class for reference counting
*/
class ParOptBase {
 public:
  ParOptBase(){
    ref_count = 0;
  }
  virtual ~ParOptBase(){}

  // Incref/decref the reference count
  // ---------------------------------
  void incref(){
    ref_count++;
  }
  void decref(){
    ref_count--;
    if (ref_count == 0){
      delete this;
    }
  }

 private:
  int ref_count;
};

/*
  This vector class defines the basic linear algebra operations and
  member functions required for design optimization.
*/
class ParOptVec : public ParOptBase {
 public:
  virtual ~ParOptVec(){}

  // Perform standard operations required for linear algebra
  // -------------------------------------------------------
  virtual void set( ParOptScalar alpha ) = 0;
  virtual void zeroEntries() = 0;
  virtual void copyValues( ParOptVec *vec ) = 0;
  virtual double norm() = 0;
  virtual double maxabs() = 0;
  virtual double l1norm() = 0;
  virtual ParOptScalar dot( ParOptVec *vec ) = 0;
  virtual void mdot( ParOptVec **vecs, int nvecs, ParOptScalar *output ) = 0;
  virtual void scale( ParOptScalar alpha ) = 0;
  virtual void axpy( ParOptScalar alpha, ParOptVec *x ) = 0;
  virtual int getArray( ParOptScalar **array ) = 0;
};

/*
  A basic ParOptVec implementation
*/
class ParOptBasicVec : public ParOptVec {
 public:
  ParOptBasicVec( MPI_Comm _comm, int n );
  ~ParOptBasicVec();

  // Perform standard operations required for linear algebra
  // -------------------------------------------------------
  void set( ParOptScalar alpha );
  void zeroEntries();
  void copyValues( ParOptVec *vec );
  double norm();
  double maxabs();
  double l1norm();
  ParOptScalar dot( ParOptVec *vec );
  void mdot( ParOptVec **vecs, int nvecs, ParOptScalar *output );
  void scale( ParOptScalar alpha );
  void axpy( ParOptScalar alpha, ParOptVec *x );
  int getArray( ParOptScalar **array );

 private:
  MPI_Comm comm;
  int size;
  ParOptScalar *x;
};

#endif
