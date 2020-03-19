#include <math.h>
#include <string.h>
#include "ParOptComplexStep.h"
#include "ParOptBlasLapack.h"
#include "ParOptVec.h"

/**
  Create a parallel vector for optimization

  @param comm the communicator for this vector
  @param n the number of vector components on this processor
*/
ParOptBasicVec::ParOptBasicVec( MPI_Comm _comm, int n ){
  comm = _comm;
  size = n;
  x = new ParOptScalar[ size ];
  memset(x, 0, size*sizeof(ParOptScalar));
}

/**
  Free the internally stored data
*/
ParOptBasicVec::~ParOptBasicVec(){
  delete [] x;
}

/**
  Set the vector value

  @param alpha the scalar value to set in all components
*/
void ParOptBasicVec::set( ParOptScalar alpha ){
  for ( int i = 0; i < size; i++ ){
    x[i] = alpha;
  }
}

/**
  Zero the entries of the vector
*/
void ParOptBasicVec::zeroEntries(){
  memset(x, 0, size*sizeof(ParOptScalar));
}

/**
  Copy the values from the given vector

  @param pvec copy the values from pvec to this vector
*/
void ParOptBasicVec::copyValues( ParOptVec *pvec ){
  ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvec);

  if (vec){
    memcpy(x, vec->x, size*sizeof(ParOptScalar));
  }
}

/**
  Compute the l2 norm of the vector

  @return the l2 norm of the vector
*/
double ParOptBasicVec::norm(){
  double res = 0.0;
#ifdef PAROPT_USE_COMPLEX
  for ( int i = 0; i < size; i++ ){
    res += (ParOptRealPart(x[i])*ParOptRealPart(x[i]) +
            ParOptImagPart(x[i])*ParOptImagPart(x[i]));
  }
#else
  int one = 1;
  res = BLASdnrm2(&size, x, &one);
  res *= res;
#endif

  double sum = 0.0;
  MPI_Allreduce(&res, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return sqrt(sum);
}

/**
  Compute the l-infinity norm of the vector

  @return the l-infinity norm of the vector
*/
double ParOptBasicVec::maxabs(){
  double res = 0.0;
  for ( int i = 0; i < size; i++ ){
    if (fabs(ParOptRealPart(x[i])) > res){
      res = fabs(ParOptRealPart(x[i]));
    }
  }

  double infty_norm = 0.0;
  MPI_Allreduce(&res, &infty_norm, 1, MPI_DOUBLE, MPI_MAX, comm);

  return infty_norm;
}

/**
  Compute the l1 norm of the vector

  @return the l1 norm of the vector
*/
double ParOptBasicVec::l1norm(){
  double res = 0.0;
  for ( int i = 0; i < size; i++ ){
    res += fabs(ParOptRealPart(x[i]));
  }

  double l1_norm = 0.0;
  MPI_Allreduce(&res, &l1_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

  return l1_norm;
}

/**
  Compute the dot-product of two vectors and return the result.

  @param pvec the other vector in the dot product
  @return the dot product of the two vectors
*/
ParOptScalar ParOptBasicVec::dot( ParOptVec *pvec ){
  ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvec);

  ParOptScalar sum = 0.0;
  if (vec){
    ParOptScalar res = 0.0;
#ifdef PAROPT_USE_COMPLEX
    for ( int i = 0; i < size; i++ ){
      res += x[i]*vec->x[i];
    }
#else
    int one = 1;
    res = BLASddot(&size, x, &one, vec->x, &one);
#endif

    MPI_Allreduce(&res, &sum, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  }

  return sum;
}

/**
  Compute multiple dot-products simultaneously. This reduces the
  parallel communication overhead.

  @param pvecs an array of vectors
  @param output an array of the dot product results
*/
void ParOptBasicVec::mdot( ParOptVec **pvecs, int nvecs, ParOptScalar *output ){
  for ( int i = 0; i < nvecs; i++ ){
    output[i] = 0.0;
    ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvecs[i]);

    if (vec){
#ifdef PAROPT_USE_COMPLEX
      for ( int j = 0; j < size; j++ ){
        output[i] += x[j]*vec->x[j];
      }
#else
      int one = 1;
      output[i] = BLASddot(&size, x, &one, vec->x, &one);
#endif
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, output, nvecs, PAROPT_MPI_TYPE, MPI_SUM, comm);
}

/**
  Scale the components of the vector
  
  @param alpha the scalar factor
*/
void ParOptBasicVec::scale( ParOptScalar alpha ){
#ifdef PAROPT_USE_COMPLEX
  for ( int i = 0; i < size; i++ ){
    x[i] *= alpha;
  }
#else
  int one = 1;
  BLASdscal(&size, &alpha, x, &one);
#endif
}

/**
  Compute: self <- self + alpha*x
*/
void ParOptBasicVec::axpy( ParOptScalar alpha, ParOptVec *pvec ){
  ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvec);

  if (vec){
#ifdef PAROPT_USE_COMPLEX
    for ( int i = 0; i < size; i++ ){
      x[i] = x[i] + alpha*vec->x[i];
    }
#else
    int one = 1;
    BLASdaxpy(&size, &alpha, vec->x, &one, x, &one);
#endif
  }
}

/**
  Retrieve the locally stored values from the array

  @param array pointer assigned to the memory location of the 
  local vector components
*/
int ParOptBasicVec::getArray( ParOptScalar **array ){
  if (array){
    *array = x;
  }
  return size;
}
