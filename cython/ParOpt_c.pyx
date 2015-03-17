#distuils: language = c++
#distuils: sources = ParOpt.c, Rosenbrock.c

#For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI
#Import the declarations required
cimport ParOptVec_c
cimport ParOptProblem_c 
from ParOpt_c cimport ParOpt

import numpy as np
cimport numpy as np

include "ParOptVec_c.pyx"

from libc.string cimport const_char

cdef extern from "mpi-compat.h":
   pass

#Declare the public methods that are in Rosenbrock.h
cdef extern from "Rosenbrock.h":
   cdef cppclass Rosenbrock(ParOptProblem_c.ParOptProblem):
   
      Rosenbrock(MPI_Comm, int _nvars, int _nwcon, int _nwstart, int _nw, int _nwskip) except +
   
      #Determine whether there is an inequality constraint
      int isSparseInequality()
      int isDenseInequality()
      int useLowerBounds()
      int useUpperBounds()
      #Get the communicator for the problem
      MPI_Comm getMPIComm()
      #Get the problem dimensions
      void getProblemSizes( int *_nvars, int *_ncon, int *_nwcon, int *_nwblock)
      #Get variables and bounds
      void getVarsAndBounds(ParOptVec_c.ParOptVec *xvec, ParOptVec_c.ParOptVec *lbvec,
                            ParOptVec_c.ParOptVec *ubvec)
      #Evaluate the objective and constraints
      int evalObjCon(ParOptVec_c.ParOptVec *xvec, double *fobj, double *cons)
      #Evaluate the objective and constraints gradients
      int evalObjConGradient(ParOptVec_c.ParOptVec *xvec, ParOptVec_c.ParOptVec *gvec,
                             ParOptVec_c.ParOptVec ** Ac)
      #Evaluate the product of the Hessian with the given vector
      int evalHvecProduct(ParOptVec_c.ParOptVec *xvec, double *z, ParOptVec_c.ParOptVec *zwvec,
                          ParOptVec_c.ParOptVec *pxvec, ParOptVec_c.ParOptVec *hvec)
      #Evaluate the sparse constraints
      void evalSparseCon(ParOptVec_c.ParOptVec *xvec, ParOptVec_c.ParOptVec *out)
      #Compute the Jacobian-vector product out = J(x)*px
      void addSparseJacobian(double alpha, ParOptVec_c.ParOptVec *x, ParOptVec_c.ParOptVec *px,
                             ParOptVec_c.ParOptVec *out)
      #Compute the transpose Jacobian-vector product out=J(x)^{T}*pzw
      void addSparseJacobianTranspose(double alpha, ParOptVec_c.ParOptVec *x,
                                      ParOptVec_c.ParOptVec *pzw, ParOptVec_c.ParOptVec *out)
      #Add inner product of the constraints to the matrix such that A+=J(x)*cvec*J(x)^{T} where
      #cvec is a diagonal matrix
      void addSparseInnerProduct(double alpha, ParOptVec_c.ParOptVec *x,
                                 ParOptVec_c.ParOptVec *cvec, double *A)

#"Wrap" the abtract base class ParOptProblem 
cdef class pyParOptProblem:
   cdef MPI.Comm _comm
   cdef ParOptProblem_c.ParOptProblem *thisptr
   
   def __cinit__(self, MPI.Comm _comm, int _nvars, int _ncon, int _nwcon, int _nwblock,
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
  
   def getVarsAndBounds(self, pyParOptVec x not None, pyParOptVec lb not None,
                        pyParOptVec ub not None):
      pass
  
   def evalObjCon(self, pyParOptVec x not None, double[::1] fobj, double[::1] cons):
      pass
  
   def evalObjconGradient(self, pyParOptVec x not None, pyParOptVec g not None,
                          pyParOptVec AC not None):
      pass
   def evalHvecProduct(self, pyParOptVec x not None, double[::1] z, pyParOptVec zw not None,
                       pyParOptVec px not None, pyParOptVec hvec not None):
      pass
  
   def evalSparseCon(self, pyParOptVec x not None, pyParOptVec out not None):
      pass
  
   def addSparseJacobian(self, double alpha, pyParOptVec x not None, pyParOptVec px not None,
                         pyParOptVec out not None):
      pass

   def addSparseJacobianTranspose(self, double alpha, pyParOptVec x not None,
                                  pyParOptVec pzw not None, pyParOptVec out not None):
      pass

   def addSparseInnerProduct(self, double alpha, pyParOptVec x not None, pyParOptVec cvec not None,
                             double [:,::1]A):
      pass
  
   def writeOutput(self, int iter, pyParOptVec x not None):
      pass

#Wrap the subclass Rosenbrock which is derived from ParOptProblem  
cdef class pyRosenbrock(pyParOptProblem):
   
   cdef Rosenbrock *rosenptr
   def __cinit__(self, MPI.Comm _comm, int _nvars, int _ncon, int _nwcon, int _nwblock,
                 int _nwstart, int _nw, int _nwskip):

      if type(self) != pyRosenbrock:
         return
      cdef ParOptProblem_c.ParOptProblem *base = NULL
      
      cdef MPI_Comm c_comm = _comm.ob_mpi
      
      self.rosenptr = self.thisptr = new Rosenbrock(c_comm, _nvars, _nwcon, _nwstart,_nw, _nwskip)

   def __dealloc__(self):
      #print "Cython: running pyRosenbrock.__dealloc__ on", self 
      cdef Rosenbrock *temp
      if self.thisptr is not NULL:
         temp = <Rosenbrock *>self.thisptr
         del temp
         self.thisptr = NULL  
      
   
#Python class for corresponding instance ParOpt
cdef class pyParOpt:
   cdef ParOpt *paropt_ptr
   
   cdef pyParOptVec x
   cdef pyParOptVec zw
   cdef pyParOptVec zl
   cdef pyParOptVec zu
   cdef pyParOptProblem _prob
      
   def __cinit__(self, pyParOptProblem _prob, int _max_lbfgs_subspace, int qn_type,
                 *args, **kwargs):
      
      self.paropt_ptr = new ParOpt(<ParOptProblem_c.ParOptProblem *>_prob.thisptr,
                                   _max_lbfgs_subspace, qn_type)
      
   def __dealloc__(self):
      #print "Cython: running pyParOpt.__dealloc__ on", self 
      del self.paropt_ptr
      
   #Perform the optimization
   def optimize(self, const char[:] checkpoint=None):
      
      if checkpoint is None: 
         return self.paropt_ptr.optimize(NULL)
      else:
         return self.paropt_ptr.optimize(&checkpoint[0])  
     
   #Retrieve values of design variables and Lagrange multipliers
   def getOptimizedPoint(self, pyParOptVec x not None, double[::1] z,
                         pyParOptVec zw not None, pyParOptVec zl not None,
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
      
      self.paropt_ptr.getOptimizedPoint(&_x.paroptvec_ptr, &_z, &_zw.paroptvec_ptr,
                                        &_zl.paroptvec_ptr, &_zu.paroptvec_ptr)
   
   #Check objective and constraint gradients
   def checkGradients(self, double dh):
    
      self.paropt_ptr.checkGradients(dh)
      
   #Set optimizer parameters
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
      
   #Set/obtain the barrier parameter
   def setInitBarrierParameter(self, double mu):
      self.paropt_ptr.setInitBarrierParameter(mu)
      
   def getBarrierParameter(self):
      return self.paropt_ptr.getBarrierParameter()
  
   #Set parameters associated with the linesearch
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
      
   #Set parameters for the interal GMRES algorithm
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
      
   #Set other parameters
   def setOutputFrequency(self, int freq):
      self.paropt_ptr.setOutputFrequency(freq)
      
   def setMajorIterStepCheck(self, int step):
      self.paropt_ptr.setMajorIterStepCheck(step)
      
   def setOutputFile(self, const char [:] filename):
      if filename is None:
         self.paropt_ptr.setOutputFile(NULL)
      else:     
         self.paropt_ptr.setOutputFile(&filename[0])
      
   #Write out the design variables to binary format (fast MPI/IO)
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
      
