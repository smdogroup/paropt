#distuils: language = c++
#distuils: sources = ParOpt.c

# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import the declarations required
from ParOpt_c cimport ParOptProblem, ParOptVec, ParOpt

import numpy as np
cimport numpy as np

from libc.string cimport const_char

cdef extern from "mpi-compat.h":
   pass

# "Wrap" the abtract base class ParOptProblem 
cdef class pyParOptProblem:
   cdef MPI.Comm _comm
   cdef ParOptProblem *thisptr
   
   def __cinit__(self, MPI.Comm _comm, 
                 int _nvars, int _ncon, int _nwcon, int _nwblock,
                 *args, **kwargs):
      pass

   def __dealloc__(self):      
      pass

   def getMPIComm(self):
      pass
               
   def getProblemSizes(self):      
      pass

   def isSparseInequality(self):
      pass
  
   def isDenseInequality(self):
      pass
  
   def useLowerBounds(self):
      pass
  
   def useUpperBounds(self):
      pass
  
   def getVarsAndBounds(self, pyParOptVec x not None, 
                        pyParOptVec lb not None,
                        pyParOptVec ub not None):
      pass
  
   def evalObjCon(self, pyParOptVec x not None, double[::1] fobj, 
                  double[::1] cons):
      pass
  
   def evalObjconGradient(self, pyParOptVec x not None, 
                          pyParOptVec g not None,
                          pyParOptVec AC not None):
      pass

   def evalHvecProduct(self, pyParOptVec x not None, 
                       double[::1] z, 
                       pyParOptVec zw not None,
                       pyParOptVec px not None, 
                       pyParOptVec hvec not None):
      pass
  
   def evalSparseCon(self, pyParOptVec x not None, 
                     pyParOptVec out not None):
      pass
  
   def addSparseJacobian(self, double alpha, 
                         pyParOptVec x not None, 
                         pyParOptVec px not None,
                         pyParOptVec out not None):
      pass

   def addSparseJacobianTranspose(self, double alpha, 
                                  pyParOptVec x not None,
                                  pyParOptVec pzw not None, 
                                  pyParOptVec out not None):
      pass

   def addSparseInnerProduct(self, double alpha, 
                             pyParOptVec x not None, 
                             pyParOptVec cvec not None,
                             double [:,::1]A):
      pass
  
   def writeOutput(self, int iter, pyParOptVec x not None):
      pass

# Corresponds to the C++ class defined in the pxd file
cdef class pyParOptVec:
   cdef ParOptVec *paroptvec_ptr

   def __cinit__(self, MPI.Comm _comm, int n):
      cdef MPI_Comm c_comm = _comm.ob_mpi
      self.paroptvec_ptr = new ParOptVec(c_comm, n)

   def __dealloc__(self):
      del self.paroptvec_ptr

   cdef setThis(self, ParOptVec *other):
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
   
   def mdot(self, pyParOptVec vecs not None, int nvecs, 
            double[::1] output):
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
   
# Python class for corresponding instance ParOpt
cdef class pyParOpt:
   cdef ParOpt *paropt_ptr
   
   cdef pyParOptVec x
   cdef pyParOptVec zw
   cdef pyParOptVec zl
   cdef pyParOptVec zu
   cdef pyParOptProblem _prob
      
   def __cinit__(self, pyParOptProblem _prob, 
                 int _max_lbfgs_subspace,
                 *args, **kwargs):
      
      self.paropt_ptr =\
          new ParOpt(<ParOptProblem*>_prob.thisptr,
                      _max_lbfgs_subspace)
      
   def __dealloc__(self):
      # print "Cython: running pyParOpt.__dealloc__ on", self 
      del self.paropt_ptr
      
   # Perform the optimization
   def optimize(self, const char[:] checkpoint=None):
      if checkpoint is None: 
         return self.paropt_ptr.optimize(NULL)
      else:
         return self.paropt_ptr.optimize(&checkpoint[0])  
     
   # Retrieve values of design variables and Lagrange multipliers
   def getOptimizedPoint(self, pyParOptVec x not None, double[::1] z,
                         pyParOptVec zw not None, 
                         pyParOptVec zl not None,
                         pyParOptVec zu not None):
      
      cdef const double *_z
      cdef pyParOptVec _x
      cdef pyParOptVec _zw
      cdef pyParOptVec _zl
      cdef pyParOptVec _zu
      
      _z = &z[0]
      _x = pyParOptVec().setThis(x.paroptvec_ptr)
      _zw = pyParOptVec().setThis(zw.paroptvec_ptr)
      _zl = pyParOptVec().setThis(zl.paroptvec_ptr)
      _zu = pyParOptVec().setThis(zu.paroptvec_ptr)
      
      self.paropt_ptr.getOptimizedPoint(&_x.paroptvec_ptr, &_z, 
                                         &_zw.paroptvec_ptr,
                                         &_zl.paroptvec_ptr, 
                                         &_zu.paroptvec_ptr)
   
   # Check objective and constraint gradients
   def checkGradients(self, double dh):
    
      self.paropt_ptr.checkGradients(dh)
      
   # Set optimizer parameters
   def setInitStartingPoint(self, int init):
      self.paropt_ptr.setInitStartingPoint(init)
      
   def setMaxMajorIterations(self, int iters):
      self.paropt_ptr.setMaxMajorIterations(iters)
      
   def setAbsOptimalityTol(self, double tol):
      self.paropt_ptr.setAbsOptimalityTol(tol)
      
   def setBarrierFraction(self, double frac):
      self.paropt_ptr.setBarrierFraction(frac)
      
   def setBarrierPower(self, double power):
      self.paropt_ptr.setBarrierPower(power)
      
   def setHessianResetFreq(self, int freq):
      self.paropt_ptr.setHessianResetFreq(freq)
      
   def setSequentialLinearMethod(self, int truth):
      self.paropt_ptr.setSequentialLinearMethod(truth)
      
   # Set/obtain the barrier parameter
   def setInitBarrierParameter(self, double mu):
      self.paropt_ptr.setInitBarrierParameter(mu)
      
   def getBarrierParameter(self):
      return self.paropt_ptr.getBarrierParameter()
  
   # Set parameters associated with the linesearch
   def setUseLineSearch(self, int truth):
      self.paropt_ptr.setUseLineSearch(truth)
      
   def setMaxLineSearchIters(self, int iters):
      self.paropt_ptr.setMaxLineSearchIters(iters)
      
   def setBacktrackingLineSearch(self, int truth):
      self.paropt_ptr.setBacktrackingLineSearch(truth)
      
   def setArmijioParam(self, double c1):
      self.paropt_ptr.setArmijioParam(c1)
      
   def setPenaltyDescentFraction(self, double frac):
      self.paropt_ptr.setPenaltyDescentFraction(frac)
      
   # Set parameters for the interal GMRES algorithm
   def setUseHvecProduct(self, int truth):
      self.paropt_ptr.setUseHvecProduct(truth)
      
   def setUseQNGMRESPreCon(self, int truth):
      self.paropt_ptr.setUseQNGMRESPreCon(truth)
      
   def setNKSwitchTolerance(self, double tol):
      self.paropt_ptr.setNKSwitchTolerance(tol)
      
   def setEisenstatWalkerParameters(self, double gamma, double alpha):
      self.paropt_ptr.setEisenstatWalkerParameters(gamma, alpha)
      
   def setGMRESTolerances(self, double rtol, double atol):
      self.paropt_ptr.setGMRESTolerances(rtol, atol)
      
   def setGMRESSusbspaceSize(self, int _gmres_subspace_size):
      self.paropt_ptr.setGMRESSusbspaceSize(_gmres_subspace_size)
      
   # Set other parameters
   def setOutputFrequency(self, int freq):
      self.paropt_ptr.setOutputFrequency(freq)
      
   def setMajorIterStepCheck(self, int step):
      self.paropt_ptr.setMajorIterStepCheck(step)
      
   def setOutputFile(self, const char [:] filename):
      if filename is None:
         self.paropt_ptr.setOutputFile(NULL)
      else:     
         self.paropt_ptr.setOutputFile(&filename[0])
      
   # Write out the design variables to binary format (fast MPI/IO)
   def writeSolutionFile(self, const char[:] filename):
      if filename is None:
         return self.paropt_ptr.writeSolutionFile(NULL)
      else: 
         return self.paropt_ptr.writeSolutionFile(&filename[0])
  
   def readSolutionFile(self, const char[:] filename):
      if filename is None:
         return self.paropt_ptr.readSolutionFile(NULL)
      else: 
         return self.paropt_ptr.readSolutionFile(&filename[0])
      
