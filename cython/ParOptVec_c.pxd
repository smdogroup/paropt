#For the use of MPI
cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport *

import numpy as np
cimport numpy as np

cdef extern from "mpi-compat.h":
   pass

cdef extern from "ParOptVec.h":
   cdef cppclass ParOptVec:
      MPI_Comm _comm
      ParOptVec(MPI_Comm _comm, int n) 
      #Perform standard operations required for linear algebra
      void set(double alpha)
      void zeroEntries()
      
      void copyValues(ParOptVec * vec)
      double norm()
      double maxabs()
      double dot(ParOptVec *vec)
      void mdot(ParOptVec ** vecs, int nvecs, double *output)
      void scale(double alpha)
      void axpy(double alpha, ParOptVec *x)
      int getArray(double **array)
      int writeToFile(const char * filename)
   
   cdef cppclass CompactQuasiNewton:
      CompactQuasiNewton()
      #Reset internal data
      void reset()
      #Perform BFGS update
      int update(ParOptVec *s, ParOptVec *y)
      #Perform matrix-vector multiplication
      void mult(ParOptVec *x, ParOptVec *y)
      void multAdd(double alpha, ParOptVec *x, ParOptVec *y)
      #Get information for the limited-memory BFGS update
      void getCompactMat(double *_b0, const double **_d,
                         const double **_M, ParOptVec ***Z)
      #Get the maximum size of the limited-memory BFGS
      int getMaxLimitedMemorySize()
      
   cdef cppclass LBFGS(CompactQuasiNewton):
      MPI_Comm _comm
      LBFGS(MPI_Comm _comm, int _nvars, int _subspace_size)
      #Reset internal data
      void reset()
      #Perform BFGS update
      int update(ParOptVec *s, ParOptVec *y)
      #Perform matrix-vector multiplication
      void mult(ParOptVec *x, ParOptVec *y)
      void multAdd(double alpha, ParOptVec *x, ParOptVec *y)
      #Get information for the limited memory BFGS update
      void getCompactMat(double *_b0, const double **_d, 
                         const double **_M, ParOptVec ***Z)
      #Get the maximum size of the limited memory BFGS
      int getMaxLimitedMemorySize()

   cdef cppclass LSR1(CompactQuasiNewton):
      MPI_Comm _comm
      LSR1(MPI_Comm _comm, int _nvars, int _subspace_size)
      #Reset internal data
      void reset()
      #Perform BFGS update
      int update(ParOptVec *s, ParOptVec *y)
      #Perform matrix-vector multiplication
      void mult(ParOptVec *x, ParOptVec *y)
      void multAdd(double alpha, ParOptVec *x, ParOptVec *y)
      #Get information for the limited memory BFGS update
      void getCompactMat(double *_b0, const double **_d, 
                         const double **_M, ParOptVec ***Z)
      #Get the maximum size of the limited memory BFGS
      int getMaxLimitedMemorySize()

