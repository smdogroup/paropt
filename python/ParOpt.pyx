#distuils: language = c++
#distuils: sources = ParOpt.c

# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import the declarations required from the pxd file
from ParOpt cimport *
# CyParOptProblem, ParOpt, ParOptVec

# Import numpy 
import numpy as np
cimport numpy as np

# Ensure that numpy is initialized
np.import_array()

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

cdef int _evalhvecproduct(void *_self, int nvars, int ncon, int nwcon,
                          double *x, double *z, double *zw,
                          double *px, double *hvec):
   # The numpy arrays that will be used for x
   cdef np.ndarray xnp, znp, pxnp, hnp
   cdef np.ndarray zwnp = None
   
   # Create the wrapper objects
   xwrap = NpArrayWrap()
   zwrap = NpArrayWrap()
   pxwrap = NpArrayWrap()
   hwrap = NpArrayWrap()

   # Set the arrays
   xwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>x)
   zwrap.set_data1d(np.NPY_DOUBLE, ncon, <void*>z)
   pxwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>px)
   hwrap.set_data1d(np.NPY_DOUBLE, nvars, <void*>hvec)

   # Get the resulting numpy arrays
   xnp = xwrap.as_ndarray()
   znp = zwrap.as_ndarray()
   pxnp = pxwrap.as_ndarray()
   hnp = hwrap.as_ndarray()

   if nwcon > 0:
      zwwrap = NpArrayWrap()
      zwwrap.set_data1d(np.NPY_DOUBLE, nwcon, <void*>zw)
      zwnp = zwwrap.as_ndarray()

   # Call the objective function
   fail = (<object>_self).evalHvecProduct(xnp, znp, zwnp,
                                          pxnp, hnp)

   return fail

# "Wrap" the abtract base class ParOptProblem 
cdef class pyParOptProblem:
   cdef CyParOptProblem *this_ptr
   
   def __init__(self, MPI.Comm comm, int nvars, int ncon,
                int nwcon=0, int nwblock=0):
      # Convert the communicator
      cdef MPI_Comm c_comm = comm.ob_mpi

      # Create the pointer to the underlying C++ object
      self.this_ptr = new CyParOptProblem(c_comm, nvars, ncon,
                                          nwcon, nwblock)
      self.this_ptr.setSelfPointer(<void*>self)
      self.this_ptr.setGetVarsAndBounds(_getvarsandbounds)
      self.this_ptr.setEvalObjCon(_evalobjcon)
      self.this_ptr.setEvalObjConGradient(_evalobjcongradient)
      self.this_ptr.setEvalHvecProduct(_evalhvecproduct)
      return

   def __dealloc__(self):
      del self.this_ptr
      return

   def setInequalityOptions(self, dense_ineq=True, sparse_ineq=True,
                            use_lower=True, use_upper=True):
      # Assume that everything is false
      cdef int dense = 0
      cdef int sparse = 0
      cdef int lower = 0
      cdef int upper = 0

      # Swap the integer values if the flags are set
      if dense_ineq: dense = 1
      if sparse_ineq: sparse = 1
      if use_lower: lower = 1
      if use_upper: upper = 1

      # Set the options
      self.this_ptr.setInequalityOptions(dense, sparse, lower, upper)
   
      return

# Constants that define what Quasi-Newton method to use
BFGS = PAROPT_BFGS
SR1 = PAROPT_SR1

# Python class for corresponding instance ParOpt
cdef class pyParOpt:
   cdef ParOpt *this_ptr
      
   def __cinit__(self, pyParOptProblem _prob, 
                 int max_qn_subspace, 
                 QuasiNewtonType qn_type):
      self.this_ptr = new ParOpt(_prob.this_ptr, 
                                 max_qn_subspace, qn_type)
      
   def __dealloc__(self):
      del self.this_ptr
      
   # Perform the optimization
   def optimize(self, char *checkpoint=''):
      if checkpoint is None: 
         return self.this_ptr.optimize(NULL)
      else:
         return self.this_ptr.optimize(&checkpoint[0])
   
   def getOptimizedPoint(self):
      '''Get the optimized solution from ParOpt'''
      cdef int n = 0
      cdef double *values = NULL
      cdef ParOptVec *vec = NULL
      
      # Retrieve the optimized vector
      self.this_ptr.getOptimizedPoint(&vec, NULL, NULL, NULL, NULL)
      
      # Get the variables from the vector
      n = vec.getArray(&values)

      # Allocate a new numpy array
      x = np.zeros(n, np.double)

      # Assign the new entries
      for i in xrange(n):
         x[i] = values[i]

      return x

   def getOptimizedMultipliers(self):
      '''Get the optimized multipliers'''
      cdef int n = 0, nc = 0, nw = 0
      cdef const double *zvals = NULL
      cdef double *zwvals = NULL
      cdef double *zlvals = NULL
      cdef double *zuvals = NULL
      cdef ParOptVec *zwvec = NULL
      cdef ParOptVec *zlvec = NULL
      cdef ParOptVec *zuvec = NULL
      
      # Set the initial values for the multipliers etc.
      z = None
      zw = None
      zl = None
      zu = None

      # Retrieve the optimized vector
      self.this_ptr.getOptimizedPoint(NULL, &zvals, &zwvec, &zlvec, &zuvec)

      # Get the number of constraints
      self.this_ptr.getProblemSizes(NULL, &nc, NULL, NULL)
      
      # Copy over the Lagrange multipliers
      z = np.zeros(nc, np.double)
      for i in xrange(nc):
         z[i] = zvals[i]

      # Convert the weighting multipliers
      if zwvec:
         nw = zwvec.getArray(&zwvals)
         zw = np.zeros(nw, np.double)
         for i in xrange(nw):
            zw[i] = zwvals[i]

      # Convert the lower bound multipliers
      if zlvec:
         n = zlvec.getArray(&zlvals)
         zl = np.zeros(n, np.double)
         for i in xrange(n):
            zl[i] = zlvals[i]

      # Convert the upper bound multipliers
      if zuvec:
         n = zuvec.getArray(&zuvals)
         zu = np.zeros(n, np.double)
         for i in xrange(n):
            zu[i] = zuvals[i]

      return z, zw, zl, zu

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
   
   def setQNDiagonalFactor(self, double sigma):
      self.this_ptr.setQNDiagonalFactor(sigma)
      
   def setSequentialLinearMethod(self, int truth):
      self.this_ptr.setSequentialLinearMethod(truth)
      
   # Set/obtain the barrier parameter
   def setInitBarrierParameter(self, double mu):
      self.this_ptr.setInitBarrierParameter(mu)
      
   def getBarrierParameter(self):
      return self.this_ptr.getBarrierParameter()
  
   # Reset the quasi-Newton Hessian
   def resetQuasiNewtonHessian(self):
      self.this_ptr.resetQuasiNewtonHessian()

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
      
   def setGMRESSubspaceSize(self, int _gmres_subspace_size):
      self.this_ptr.setGMRESSubspaceSize(_gmres_subspace_size)
      
   # Set other parameters
   def setOutputFrequency(self, int freq):
      self.this_ptr.setOutputFrequency(freq)
      
   def setMajorIterStepCheck(self, int step):
      self.this_ptr.setMajorIterStepCheck(step)
      
   def setOutputFile(self, char *filename):
      if filename is not None:
         self.this_ptr.setOutputFile(filename)
      
   # Write out the design variables to binary format (fast MPI/IO)
   def writeSolutionFile(self, char *filename):
      if filename is not None:
         return self.this_ptr.writeSolutionFile(filename)
  
   def readSolutionFile(self, char *filename):
      if filename is not None:
         return self.this_ptr.readSolutionFile(filename)
      
