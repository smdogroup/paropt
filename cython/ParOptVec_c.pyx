#distuils: language = c++
#distuils: sources = ParOptVec.c

#For the use of MPI
cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport *

#Import the declarations needed
cimport ParOptVec_c

import numpy as np
cimport numpy as np

cdef extern from "mpi-compat.h":
   pass

#Corresponds to the C++ class defined in the pxd file
cdef class pyParOptVec:
   
   cdef ParOptVec_c.ParOptVec *paroptvec_ptr

   def __cinit__(self, MPI.Comm _comm, int n):
      cdef MPI_Comm c_comm = _comm.ob_mpi
      self.paroptvec_ptr = new ParOptVec_c.ParOptVec(c_comm, n)

   def __dealloc__(self):
      del self.paroptvec_ptr

   cdef setThis(self, ParOptVec_c.ParOptVec *other):
      del self.paroptvec_ptr
      self.paroptvec_ptr = other
      return self
       
   def set(self,double alpha):
      self.paroptvec_ptr.set(alpha)
       
   def zeroEntries(self):
      self.paroptvec_ptr.zeroEntries()
       
   def copyValues(self, pyParOptVec vec not None):
      
      self.paroptvec_ptr.copyValues(vec.paroptvec_ptr)
   
   def norm(self):
      return self.paroptvec_ptr.norm()

   def maxabs(self):
      return self.paroptvec_ptr.maxabs()
   
   def dot(self, pyParOptVec vec not None):
      
      return self.paroptvec_ptr.dot(vec.paroptvec_ptr)
   
   def mdot(self, pyParOptVec vecs not None, int nvecs, double[::1] output):
      
      cdef pyParOptVec _vec
      _vec = pyParOptVec().setThis(vecs.paroptvec_ptr)
      self.paroptvec_ptr.mdot(&_vec.paroptvec_ptr, nvecs, &output[0])
   
   def scale(self, double alpha):
      self.paroptvec_ptr.scale(alpha)
      
   def axpy(self, double alpha, pyParOptVec x not None):
      self.paroptvec_ptr.axpy(alpha, x.paroptvec_ptr)
   
   def getArray(self, double[::1]array):
      cdef double *_array
      _array = &array[0]
      return self.paroptvec_ptr.getArray(&_array)
      
   def writeToFile(self, const char[:] filename):
      if filename is None:
         return self.paroptvec_ptr.writeToFile(NULL)
      else:
         return self.paroptvec_ptr.writeToFile(&filename[0])
   
cdef class pyCompactQuasiNewton:
   cdef ParOptVec_c.CompactQuasiNewton *cqnptr

   def __cinit__(self, *args, **kwargs):
      #if type(self) is pyCompactQuasiNewton:
      #   self.cqnptr = new ParOptVec_c.CompactQuasiNewton()
      pass      
   def __dealloc__(self):
      #if type(self) is pyCompactQuasiNewton:
      #   del self.cqnptr
      pass
   #Reset internal data
   def reset(self):
      pass
            
   #Perform BFGS update
   def update(self, pyParOptVec s not None, pyParOptVec y not None):
      pass
            
   #Perform matrix-vector multiplication
   def mult(self, pyParOptVec x not None, pyParOptVec y not None):
      pass
            
   def multAdd(self, double alpha, pyParOptVec x not None, pyParOptVec y not None):
      pass
           
   #Get information for the limited memory BFGS update
   def getCompactMat(self, double[::1] _b0, double[::1]_d, double[:,::1] M,
                     pyParOptVec Z not None):
      
      pass
   
   #Get max size of limited memory BFGS
   def getMaxLimitedMemorySize(self):
      pass

#Wrap LBFGS which inherits from CompactQuasiNewton. Since all the methods are already declared
#in the pyCompactQuasiNewton, they need not be declared in this wrapper     
cdef class pyLBFGS(pyCompactQuasiNewton):
   cdef ParOptVec_c.LBFGS *lbfgsptr

   def __cinit__(self, MPI.Comm _comm, int _nvars, int _subspace_size):

      if type(self) != pyLBFGS:
         return
        
      cdef MPI_Comm c_comm = _comm.ob_mpi
      self.lbfgsptr = self.cqnptr = new ParOptVec_c.LBFGS(c_comm, _nvars, _subspace_size)

   def __dealloc__(self):
      cdef ParOptVec_c.LBFGS *temp
      if self.cqnptr is not NULL:
         temp = <ParOptVec_c.LBFGS *>self.cqnptr
         del temp
         self.cqnptr = NULL
         
#Wrap LSR1 which inherits from CompactQuasiNewton. Since all the methods are already declared
#in the pyCompactQuasiNewton, they need not be declared in this wrapper      
cdef class pyLSR1(pyCompactQuasiNewton):
   cdef ParOptVec_c.LSR1 *lsr1ptr

   def __cinit__(self, MPI.Comm _comm, int _nvars, int _subspace_size):

      if type(self) != pyLSR1:
         return
        
      cdef MPI_Comm c_comm = _comm.ob_mpi
      self.lsr1ptr = self.cqnptr = new ParOptVec_c.LSR1(c_comm, _nvars, _subspace_size)

   def __dealloc__(self):
      cdef ParOptVec_c.LSR1 *temp
      if self.cqnptr is not NULL:
         temp = <ParOptVec_c.LSR1 *>self.cqnptr
         del temp
         self.cqnptr = NULL 
  
