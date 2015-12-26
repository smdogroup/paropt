#distuils: language = c++
#distuils: sources = ParOpt.c

# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import the declarations required from the pxd file
from ParOpt_c cimport CyParOptProblem, ParOpt

# Import numpy 
import numpy as np
cimport numpy as np

# Ensure that numpy is initialized
np.import_array()

# Import the definition required for const strings
from libc.string cimport const_char

# Import C methods for python
from cpython cimport PyObject, Py_INCREF

cdef extern from "mpi-compat.h":
   pass

# This class wraps a C++ array with a numpy array for later useage
cdef class NpArrayWrap:
   cdef int nptype
   cdef int dim1, dim2
   cdef void *data_ptr

   cdef set_data1d(self, int nptype, int dim1, void *data_ptr):
      '''Set data in the array'''
      self.nptype = nptype
      self.dim1 = dim1
      self.dim2 = -1
      self.data_ptr = data_ptr
      return

   cdef set_data2d(self, int nptype, int dim1, int dim2, void *data_ptr):
      '''Set data in the array'''
      self.nptype = nptype
      self.dim1 = dim1
      self.dim2 = dim2
      self.data_ptr = data_ptr
      return

   cdef as_ndarray(self):
      '''Return a numpy version of the array'''
      # Set the shape of the array
      cdef int size = 1
      cdef np.npy_intp shape[2]
      cdef np.ndarray ndarray

      shape[0] = <np.npy_intp> self.dim1
      if (self.dim2 > 0):
         size = 2
         shape[1] = <np.npy_intp> self.dim2
      
      # Create the array itself
      ndarray = np.PyArray_SimpleNewFromData(size, shape,
                                             self.nptype, self.data_ptr)
      
      # Set the base class who owns the memory
      ndarray.base = <PyObject*>self
      Py_INCREF(self)
      
      return ndarray

cdef void _getvarsandbounds(void *_self, int nvars,
                            double *x, double *lb, double *ub):
   # The numpy arrays that will be used to wrap x/lb/ub
   cdef np.ndarray xnp, lbnp, ubnp

   # Create the array wrappers 
   xwrap = NpArrayWrap()
   lbwrap = NpArrayWrap()
   ubwrap = NpArrayWrap()
   xwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>x)
   lbwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>lb)
   ubwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>ub)

   # Get the numpy version of the array
   xnp = xwrap.as_ndarray()
   lbnp = lbwrap.as_ndarray()
   ubnp = ubwrap.as_ndarray()

   # Retrieve the initial variables and their bounds 
   (<object>_self).getVarsAndBounds(xnp, lbnp, ubnp)

   return

cdef int _evalobjcon(void *_self, int nvars, int ncon,
                     double *x, double *fobj, double *cons):
   # The numpy arrays that will be used for x
   cdef np.ndarray xnp
   cdef int i
   
   # Create the array wrapper
   xwrap = NpArrayWrap()
   xwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>x)

   # Get the resulting numpy array
   xnp = xwrap.as_ndarray()

   # Call the objective function
   fail, _fobj, _cons = (<object>_self).evalObjCon(xnp)

   # Copy over the objective value
   fobj[0] = _fobj

   # Copy the values from the numpy arrays
   for i in range(ncon):
      cons[i] = _cons[i]

   return fail

cdef int _evalobjcongradient(void *_self, int nvars, int ncon,
                             double *x, double *g, double *A):
   # The numpy arrays that will be used for x
   cdef np.ndarray xnp, gnp, Anp
   
   # Create the array wrapper
   xwrap = NpArrayWrap()
   gwrap = NpArrayWrap()
   Awrap = NpArrayWrap()
   xwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>x)
   gwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>g)
   Awrap.set_data2d(np.NPY_DOUBLE, ncon, nvars, <void*>A)

   # Ge tthe resulting numpy array
   xnp = xwrap.as_ndarray()
   gnp = gwrap.as_ndarray()
   Anp = Awrap.as_ndarray()

   # Call the objective function
   fail = (<object>_self).evalObjConGradient(xnp, gnp, Anp)

   return fail

# "Wrap" the abtract base class ParOptProblem 
cdef class pyParOptProblem:
   cdef CyParOptProblem *this_ptr
   
   def __init__(self, MPI.Comm comm, int nvars, int ncon):
      # Convert the communicator
      cdef MPI_Comm c_comm = comm.ob_mpi

      # Create the pointer to the underlying C++ object
      self.this_ptr = new CyParOptProblem(c_comm, nvars, ncon)
      self.this_ptr.setSelfPointer(<void*>self)
      self.this_ptr.setGetVarsAndBounds(_getvarsandbounds)
      self.this_ptr.setEvalObjCon(_evalobjcon)
      self.this_ptr.setEvalObjConGradient(_evalobjcongradient)
      return

   def __dealloc__(self):
      del self.this_ptr
      return
   
# Python class for corresponding instance ParOpt
cdef class pyParOpt:
   cdef ParOpt *this_ptr
      
   def __cinit__(self, pyParOptProblem _prob, int _max_lbfgs_subspace):
      self.this_ptr = new ParOpt(_prob.this_ptr, _max_lbfgs_subspace)
      
   def __dealloc__(self):
      del self.this_ptr
      
   # Perform the optimization
   def optimize(self, const char[:] checkpoint=None):
      if checkpoint is None: 
         return self.this_ptr.optimize(NULL)
      else:
         return self.this_ptr.optimize(&checkpoint[0])
   
   # Check objective and constraint gradients
   def checkGradients(self, double dh):    
      self.this_ptr.checkGradients(dh)
      
   # Set optimizer parameters
   def setInitStartingPoint(self, int init):
      self.this_ptr.setInitStartingPoint(init)
      
   def setMaxMajorIterations(self, int iters):
      self.this_ptr.setMaxMajorIterations(iters)
      
   def setAbsOptimalityTol(self, double tol):
      self.this_ptr.setAbsOptimalityTol(tol)
      
   def setBarrierFraction(self, double frac):
      self.this_ptr.setBarrierFraction(frac)
      
   def setBarrierPower(self, double power):
      self.this_ptr.setBarrierPower(power)
      
   def setHessianResetFreq(self, int freq):
      self.this_ptr.setHessianResetFreq(freq)
      
   def setSequentialLinearMethod(self, int truth):
      self.this_ptr.setSequentialLinearMethod(truth)
      
   # Set/obtain the barrier parameter
   def setInitBarrierParameter(self, double mu):
      self.this_ptr.setInitBarrierParameter(mu)
      
   def getBarrierParameter(self):
      return self.this_ptr.getBarrierParameter()
  
   # Set parameters associated with the linesearch
   def setUseLineSearch(self, int truth):
      self.this_ptr.setUseLineSearch(truth)
      
   def setMaxLineSearchIters(self, int iters):
      self.this_ptr.setMaxLineSearchIters(iters)
      
   def setBacktrackingLineSearch(self, int truth):
      self.this_ptr.setBacktrackingLineSearch(truth)
      
   def setArmijioParam(self, double c1):
      self.this_ptr.setArmijioParam(c1)
      
   def setPenaltyDescentFraction(self, double frac):
      self.this_ptr.setPenaltyDescentFraction(frac)
      
   # Set parameters for the interal GMRES algorithm
   def setUseHvecProduct(self, int truth):
      self.this_ptr.setUseHvecProduct(truth)
      
   def setUseQNGMRESPreCon(self, int truth):
      self.this_ptr.setUseQNGMRESPreCon(truth)
      
   def setNKSwitchTolerance(self, double tol):
      self.this_ptr.setNKSwitchTolerance(tol)
      
   def setEisenstatWalkerParameters(self, double gamma, double alpha):
      self.this_ptr.setEisenstatWalkerParameters(gamma, alpha)
      
   def setGMRESTolerances(self, double rtol, double atol):
      self.this_ptr.setGMRESTolerances(rtol, atol)
      
   def setGMRESSusbspaceSize(self, int _gmres_subspace_size):
      self.this_ptr.setGMRESSusbspaceSize(_gmres_subspace_size)
      
   # Set other parameters
   def setOutputFrequency(self, int freq):
      self.this_ptr.setOutputFrequency(freq)
      
   def setMajorIterStepCheck(self, int step):
      self.this_ptr.setMajorIterStepCheck(step)
      
   def setOutputFile(self, const char [:] filename):
      if filename is None:
         self.this_ptr.setOutputFile(NULL)
      else:     
         self.this_ptr.setOutputFile(&filename[0])
      
   # Write out the design variables to binary format (fast MPI/IO)
   def writeSolutionFile(self, const char[:] filename):
      if filename is None:
         return self.this_ptr.writeSolutionFile(NULL)
      else: 
         return self.this_ptr.writeSolutionFile(&filename[0])
  
   def readSolutionFile(self, const char[:] filename):
      if filename is None:
         return self.this_ptr.readSolutionFile(NULL)
      else: 
         return self.this_ptr.readSolutionFile(&filename[0])
      
