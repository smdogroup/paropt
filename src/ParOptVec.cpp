#include <math.h>
#include <string.h>
#include "ComplexStep.h"
#include "ParOptVec.h"
#include "ParOptBlasLapack.h"

/*
  Create a parallel vector for optimization

  input:
  comm: the communicator for this vector
  n:    the number of vector components on this processor
*/
ParOptBasicVec::ParOptBasicVec( MPI_Comm _comm, int n ){
  comm = _comm;
  size = n;
  x = new ParOptScalar[ size ];
  memset(x, 0, size*sizeof(ParOptScalar));
}

/*
  Free the internally stored data
*/
ParOptBasicVec::~ParOptBasicVec(){
  delete [] x;
}

/*
  Set the vector value
*/
void ParOptBasicVec::set( ParOptScalar alpha ){
  for ( int i = 0; i < size; i++ ){
    x[i] = alpha;
  }
}

/*
  Zero the entries of the vector
*/
void ParOptBasicVec::zeroEntries(){
  memset(x, 0, size*sizeof(ParOptScalar));
}

/*
  Copy the values from the given vector
*/
void ParOptBasicVec::copyValues( ParOptVec *pvec ){
  ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvec);
  
  if (vec){
    memcpy(x, vec->x, size*sizeof(ParOptScalar));
  }
}

/*
  Compute the l2 norm of the vector
*/
double ParOptBasicVec::norm(){
  int one = 1;
  double res = BLASdnrm2(&size, x, &one);
  res *= res;

  double sum = 0.0;
  MPI_Allreduce(&res, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return sqrt(sum);
}

/*
  Compute the l-infinity norm of the vector
*/
double ParOptBasicVec::maxabs(){
  double res = 0.0;
  for ( int i = 0; i < size; i++ ){
    if (fabs(RealPart(x[i])) > res){
      res = fabs(RealPart(x[i]));
    }
  }

  double infty_norm = 0.0;
  MPI_Allreduce(&res, &infty_norm, 1, MPI_DOUBLE, MPI_MAX, comm);

  return infty_norm;
}

/*
  Compute the l1 norm of the vector
*/
double ParOptBasicVec::l1norm(){
  double res = 0.0;
  for ( int i = 0; i < size; i++ ){
    res += fabs(RealPart(x[i]));
  }

  double l1_norm = 0.0;
  MPI_Allreduce(&res, &l1_norm, 1, MPI_DOUBLE, MPI_SUM, comm);

  return l1_norm;
}

/*
  Compute the dot-product of two vectors and return the result.
*/
ParOptScalar ParOptBasicVec::dot( ParOptVec *pvec ){
  ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvec);

  ParOptScalar sum = 0.0;
  if (vec){
    int one = 1;
    ParOptScalar res = BLASddot(&size, x, &one, vec->x, &one);
    MPI_Allreduce(&res, &sum, 1, PAROPT_MPI_TYPE, MPI_SUM, comm);
  }

  return sum;
}

/*
  Compute multiple dot-products simultaneously. This reduces the
  parallel communication overhead.
*/
void ParOptBasicVec::mdot( ParOptVec **pvecs, int nvecs, ParOptScalar *output ){
  int one = 1;
  for ( int i = 0; i < nvecs; i++ ){
    output[i] = 0.0;
    ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvecs[i]);

    if (vec){
      output[i] = BLASddot(&size, x, &one, vec->x, &one);
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, output, nvecs, PAROPT_MPI_TYPE, MPI_SUM, comm);
}

/*
  Compute the dot product of the
*/
void ParOptBasicVec::scale( ParOptScalar alpha ){
  int one = 1;
  BLASdscal(&size, &alpha, x, &one);
}

/*
  Compute: self <- self + alpha*x
*/
void ParOptBasicVec::axpy( ParOptScalar alpha, ParOptVec *pvec ){
  ParOptBasicVec *vec = dynamic_cast<ParOptBasicVec*>(pvec);

  if (pvec){
    int one = 1;
    BLASdaxpy(&size, &alpha, vec->x, &one, x, &one);
  }
}

/*
  Retrieve the locally stored values from the array
*/
int ParOptBasicVec::getArray( ParOptScalar **array ){
  if (array){
    *array = x;
  }
  return size;
}
