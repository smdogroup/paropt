#distuils: language = c++
#distuils: sources = ParOpt.c

# For the use of MPI
from mpi4py.MPI cimport *
cimport mpi4py.MPI as MPI

# Import the declarations required from the pxd file
from ParOpt cimport *

# Import numpy 
import numpy as np
cimport numpy as np

# Ensure that numpy is initialized
np.import_array()

# Import C methods for python
from cpython cimport PyObject, Py_INCREF

# Include the definitions
include "ParOptDefs.pxi"

# Include the mpi4py header
cdef extern from "mpi-compat.h":
   pass

# The quasi-Newton hessian approximation
BFGS = PAROPT_BFGS
SR1 = PAROPT_SR1
NO_HESSIAN_APPROX = PAROPT_NO_HESSIAN_APPROX

# The ParOpt norm type
INFTY_NORM = PAROPT_INFTY_NORM
L1_NORM = PAROPT_L1_NORM
L2_NORM = PAROPT_L2_NORM

# The ParOpt barrier strategies
MONOTONE = PAROPT_MONOTONE
MEHROTRA = PAROPT_MEHROTRA
COMPLEMENTARITY_FRACTION = PAROPT_COMPLEMENTARITY_FRACTION

# Set the update type
SKIP_NEGATIVE_CURVATURE = PAROPT_SKIP_NEGATIVE_CURVATURE
DAMPED_UPDATE = PAROPT_DAMPED_UPDATE

def unpack_output(str filename):
   '''
   Unpack the parameters from the paropt output file and return them
   in a list of numpy arrays. This also returns a small string
   description from the file itself. This code relies ont he
   fixed-width nature of the file, which is guaranteed.
   '''

   # The arguments that we're looking for
   args = ['iter', 'nobj', 'ngrd', 'nhvc', 'alpha', 'alphx', 'alphz',
           'fobj', '|opt|', '|infes|', '|dual|', 'mu', 'comp', 'dmerit',
           'rho']
   fmt = '4d 4d 4d 4d 7e 7e 7e 12e 7e 7e 7e 7e 7e 8e 7e'.split()
   
   # Loop over the file until the end
   content = []
   for f in fmt:
      content.append([])

   # Read the entire
   with open(filename, 'r') as fp:
      lines = fp.readlines()

      index = 0
      while index < len(lines):
         fargs = lines[index].split()
         if (len(fargs) > 2 and
             (fargs[0] == args[0] and fargs[1] == args[1])):
            index += 1
            
            # Read at most 10 lines before searching for the next
            # header
            counter = 0
            while counter < 10 and index < len(lines):
               line = lines[index]
               index += 1
               counter += 1
               if len(line.split()) < len(args):
                  continue

               # Scan through the format list and determine how to
               # convert the object based on the format string
               off = 0
               idx = 0
               for f in fmt:
                  next = int(f[:-1])
                  s = line[off:off+next]
                  off += next+1

                  if f[-1] == 'd':
                     try:
                        content[idx].append(int(s))
                     except:
                        content[idx].append(0)
                  elif f[-1] == 'e':
                     try:
                        content[idx].append(float(s))
                     except:
                        content[idx].append(0.0)
                  idx += 1

         # Increase the index by one
         index += 1

   # Convert the lists to numpy arrays
   objs = []
   for idx in range(len(args)):
      if fmt[idx][1] == 'd':
         objs.append(np.array(content[idx], dtype=np.int))
      else:
         objs.append(np.array(content[idx]))
                  
   return args, objs

def unpack_mma_output(str filename):
   '''
   Unpack the parameters from a file output from MMA
   '''

   args = ['MMA', 'sub-iter', 'fobj', 'l1-opt', 'linft-opt', 'l1-lambd', 'infeas']
   fmt = ['5d', '8d', '15e', '9e', '9e', '9e', '9e']

   # Loop over the file until the end
   content = []
   for f in fmt:
      content.append([])

   # Read the entire
   with open(filename, 'r') as fp:
      lines = fp.readlines()

      index = 0
      while index < len(lines):
         fargs = lines[index].split()
         if (len(fargs) > 2 and
             (fargs[0] == args[0] and fargs[1] == args[1])):
            index += 1
            
            # Read at most 10 lines before searching for the next
            # header
            counter = 0
            while counter < 10 and index < len(lines):
               line = lines[index]
               index += 1
               counter += 1
               if len(line.split()) < len(args):
                  continue

               # Scan through the format list and determine how to
               # convert the object based on the format string
               off = 0
               idx = 0
               for f in fmt:
                  next = int(f[:-1])
                  s = line[off:off+next]
                  off += next+1

                  if f[-1] == 'd':
                     try:
                        content[idx].append(int(s))
                     except:
                        content[idx].append(0)
                  elif f[-1] == 'e':
                     try:
                        content[idx].append(float(s))
                     except:
                        content[idx].append(0.0)
                  idx += 1

         # Increase the index by one
         index += 1

   # Convert the lists to numpy arrays
   objs = []
   for idx in range(len(args)):
      if fmt[idx][1] == 'd':
         objs.append(np.array(content[idx], dtype=np.int))
      else:
         objs.append(np.array(content[idx]))
                  
   return args, objs

# Read in a ParOpt checkpoint file and produce python variables
def unpack_checkpoint(str filename):
   '''Convert the checkpoint file to usable python objects'''

   # Open the file in read-only binary mode
   fp = open(filename, 'rb')
   sfp = fp.read()
   
   # Get the sizes of c integers and doubles
   ib = np.dtype(np.intc).itemsize
   fb = np.dtype(np.double).itemsize

   # Convert the sizes stored in the checkpoint file
   sizes = np.fromstring(sfp[:3*ib], dtype=np.intc)
   nvars = sizes[0]
   nwcon = sizes[1]
   ncon = sizes[2]

   # Skip first three integers and the barrier parameter value
   offset = 3*ib
   barrier = np.fromstring(sfp[offset:offset+fb], dtype=np.double)[0]   
   offset += fb

   # Convert the slack variables and multipliers
   s = np.fromstring(sfp[offset:offset+fb*ncon], dtype=np.double)
   offset += fb*ncon
   z = np.fromstring(sfp[offset:offset+fb*ncon], dtype=np.double)
   offset += fb*ncon

   # Convert the variables and multipliers
   x = np.fromstring(sfp[offset:offset+fb*nvars], dtype=np.double)
   offset += fb*nvars
   zl = np.fromstring(sfp[offset:offset+fb*nvars], dtype=np.double)
   offset += fb*nvars
   zu = np.fromstring(sfp[offset:offset+fb*nvars], dtype=np.double)
   offset += fb*nvars

   return barrier, s, z, x, zl, zu

# This wraps a C++ array with a numpy array for later useage
cdef inplace_array_1d(int nptype, int dim1, void *data_ptr, 
                      object base=None):
   '''Return a numpy version of the array'''
   # Set the shape of the array
   cdef int size = 1
   cdef np.npy_intp shape[1]
   cdef np.ndarray ndarray

   # Set the first entry of the shape array
   shape[0] = <np.npy_intp>dim1
      
   # Create the array itself - Note that this function will not
   # delete the data once the ndarray goes out of scope
   ndarray = np.PyArray_SimpleNewFromData(size, shape,
                                          nptype, data_ptr)

   if base is not None:
      Py_INCREF(base)
      ndarray.base = <PyObject*>base

   return ndarray

cdef void _getvarsandbounds(void *_self, int nvars,
                            ParOptVec *_x, ParOptVec *_lb, 
                            ParOptVec *_ub):
   x = _init_PVec(_x)
   lb = _init_PVec(_lb)
   ub = _init_PVec(_ub) 
   (<object>_self).getVarsAndBounds(x, lb, ub)
   return

cdef int _evalobjcon(void *_self, int nvars, int ncon,
                     ParOptVec *_x, ParOptScalar *fobj, 
                     ParOptScalar *cons):
   # Call the objective function
   x = _init_PVec(_x)
   fail, _fobj, _cons = (<object>_self).evalObjCon(x)

   # Copy over the objective value
   fobj[0] = _fobj

   # Copy the values from the numpy arrays
   for i in range(ncon):
      cons[i] = _cons[i]
   return fail

cdef int _evalobjcongradient(void *_self, int nvars, int ncon,
                             ParOptVec *_x, ParOptVec *_g, 
                             ParOptVec **A):
   # The numpy arrays that will be used for x
   x = _init_PVec(_x)
   g = _init_PVec(_g)
   Ac = []
   for i in range(ncon):
      Ac.append(_init_PVec(A[i]))

   # Call the objective function
   fail = (<object>_self).evalObjConGradient(x, g, Ac)
   return fail

cdef int _evalhvecproduct(void *_self, int nvars, int ncon, int nwcon,
                          ParOptVec *_x, ParOptScalar *_z, ParOptVec *_zw,
                          ParOptVec *_px, ParOptVec *_hvec):
   x = _init_PVec(_x)
   zw = None
   if _zw != NULL:
      zw = _init_PVec(_zw)
   px = _init_PVec(_px)
   hvec = _init_PVec(_hvec)
   
   z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z)

   # Call the objective function
   fail = (<object>_self).evalHvecProduct(x, z, zw, px, hvec)
   return fail

cdef int _evalhessiandiag(void *_self, int nvars, int ncon, int nwcon,
                          ParOptVec *_x, ParOptScalar *_z, ParOptVec *_zw,
                          ParOptVec *_hdiag):
   x = _init_PVec(_x)
   zw = None
   if _zw != NULL:
      zw = _init_PVec(_zw)
   hdiag = _init_PVec(_hdiag)
   
   z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z)

   # Call the objective function
   fail = (<object>_self).evalHessianDiag(x, z, zw, hdiag)
   return fail

cdef void _evalsparsecon(void *_self, int nvars, int nwcon,
                         ParOptVec *_x, ParOptVec *_con):
   x = _init_PVec(_x)
   con = _init_PVec(_con)

   (<object>_self).evalSparseCon(x, con)
   return

cdef void _addsparsejacobian(void *_self, int nvars, 
                             int nwcon, ParOptScalar alpha, 
                             ParOptVec *_x, ParOptVec *_px, 
                             ParOptVec *_con):
   x = _init_PVec(_x)
   px = _init_PVec(_px)
   con = _init_PVec(_con)

   (<object>_self).addSparseJacobian(alpha, x, px, con)
   return

cdef void _addsparsejacobiantranspose(void *_self, int nvars, 
                                      int nwcon, ParOptScalar alpha, 
                                      ParOptVec *_x, ParOptVec *_pzw, 
                                      ParOptVec *_out):
   x = _init_PVec(_x)
   pzw = _init_PVec(_pzw)
   out = _init_PVec(_out)
   (<object>_self).addSparseJacobianTranspose(alpha, x, pzw, out)
   return

cdef void _addsparseinnerproduct(void *_self, int nvars,
                                 int nwcon, int nwblock, ParOptScalar alpha,
                                 ParOptVec *_x, ParOptVec *_c, 
                                 ParOptScalar *_A):
   x = _init_PVec(_x)
   c = _init_PVec(_c)
   A = inplace_array_1d(PAROPT_NPY_SCALAR, nwcon*nwblock*nwblock,
                        <void*>_A)

   (<object>_self).addSparseInnerProduct(alpha, x, c, A)
   return

cdef class pyParOptProblemBase:
   def __cinit__(self):
      self.ptr = NULL

   def createDesignVec(self):
      cdef ParOptVec *vec = NULL
      vec = self.ptr.createDesignVec()
      return _init_PVec(vec)

   def createConstraintVec(self):
      cdef ParOptVec *vec = NULL
      vec = self.ptr.createConstraintVec()
      return _init_PVec(vec)

# "Wrap" the abtract base class ParOptProblem 
cdef class pyParOptProblem(pyParOptProblemBase):
   cdef CyParOptProblem *me
   def __init__(self, MPI.Comm comm, int nvars, int ncon,
                int nwcon=0, int nwblock=0):
      # Convert the communicator
      cdef MPI_Comm c_comm = comm.ob_mpi

      # Create the pointer to the underlying C++ object
      self.me = new CyParOptProblem(c_comm, nvars, ncon, nwcon, nwblock)
      self.me.setSelfPointer(<void*>self)
      self.me.setGetVarsAndBounds(_getvarsandbounds)
      self.me.setEvalObjCon(_evalobjcon)
      self.me.setEvalObjConGradient(_evalobjcongradient)
      self.me.setEvalHvecProduct(_evalhvecproduct)
      self.me.setEvalHessianDiag(_evalhessiandiag)
      self.me.setEvalSparseCon(_evalsparsecon)
      self.me.setAddSparseJacobian(_addsparsejacobian)
      self.me.setAddSparseJacobianTranspose(_addsparsejacobiantranspose)
      self.me.setAddSparseInnerProduct(_addsparseinnerproduct)
      self.ptr = self.me
      self.ptr.incref()
      return

   def __dealloc__(self):
      if self.ptr:
         self.ptr.decref()
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
      self.me.setInequalityOptions(dense, sparse, lower, upper)
   
      return

# Constants that define what Quasi-Newton method to use
BFGS = PAROPT_BFGS
SR1 = PAROPT_SR1

cdef class PVec:
   def __cinit__(self):
      self.ptr = NULL
      return

   def __dealloc__(self):
      if self.ptr:
         self.ptr.decref()

   def __len__(self):
      cdef int size = 0
      size = self.ptr.getArray(NULL)
      return size

   def __add__(self, b):
      return self[:] + b

   def __sub__(self, b):
      return self[:] - b

   def __mul__(self, b):
      return self[:]*b

   def __div__(self, b):
      return self[:]/b

   def __iadd__(self, b):
      cdef int size = 0
      cdef int bsize = 0
      cdef ParOptScalar *array = NULL
      cdef ParOptScalar *barray = NULL
      cdef ParOptScalar value = 0.0
      cdef ParOptVec *bptr = NULL
      size = self.ptr.getArray(&array)
      if isinstance(b, PVec):
         bptr = (<PVec>b).ptr
         bsize = bptr.getArray(&barray)
         if bsize == size:
            for i in range(size):
               array[i] += barray[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)
      elif hasattr(b, '__len__'):
         bsize = len(b)
         if bsize == size:
            for i in range(size):
               array[i] += b[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)         
      else:
         value = b
         for i in range(size):
            array[i] += value
      return

   def __isub__(self, b):
      cdef int size = 0
      cdef int bsize = 0
      cdef ParOptScalar *array = NULL
      cdef ParOptScalar *barray = NULL
      cdef ParOptScalar value = 0.0
      cdef ParOptVec *bptr = NULL
      size = self.ptr.getArray(&array)
      if isinstance(b, PVec):
         bptr = (<PVec>b).ptr
         bsize = bptr.getArray(&barray)
         if bsize == size:
            for i in range(size):
               array[i] -= barray[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)
      elif hasattr(b, '__len__'):
         bsize = len(b)
         if bsize == size:
            for i in range(size):
               array[i] -= b[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)         
      else:
         value = b
         for i in range(size):
            array[i] -= value
      return

   def __imul__(self, b):
      cdef int size = 0
      cdef int bsize = 0
      cdef ParOptScalar *array = NULL
      cdef ParOptScalar *barray = NULL
      cdef ParOptScalar value = 0.0
      cdef ParOptVec *bptr = NULL
      size = self.ptr.getArray(&array)
      if isinstance(b, PVec):
         bptr = (<PVec>b).ptr
         bsize = bptr.getArray(&barray)
         if bsize == size:
            for i in range(size):
               array[i] *= barray[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)
      elif hasattr(b, '__len__'):
         bsize = len(b)
         if bsize == size:
            for i in range(size):
               array[i] *= b[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)         
      else:
         value = b
         for i in range(size):
            array[i] *= value
      return

   def __idiv__(self, b):
      cdef int size = 0
      cdef int bsize = 0
      cdef ParOptScalar *array = NULL
      cdef ParOptScalar *barray = NULL
      cdef ParOptScalar value = 0.0
      cdef ParOptVec *bptr = NULL
      size = self.ptr.getArray(&array)
      if isinstance(b, PVec):
         bptr = (<PVec>b).ptr
         bsize = bptr.getArray(&barray)
         if bsize == size:
            for i in range(size):
               array[i] /= barray[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)
      elif hasattr(b, '__len__'):
         bsize = len(b)
         if bsize == size:
            for i in range(size):
               array[i] /= b[i]
         else:
            errmsg = 'PVecs must be the same size'
            raise ValueError(errmsg)         
      else:
         value = b
         for i in range(size):
            array[i] /= value
      return

   def __getitem__(self, k):
      cdef int size = 0
      cdef ParOptScalar *array
      size = self.ptr.getArray(&array)
      if isinstance(k, int):
         if k < 0 or k >= size:
            errmsg = 'Index %d out of range [0,%d)'%(k, size)
            raise IndexError(errmsg)
         return array[k]
      elif isinstance(k, slice):
         start, stop, step = k.indices(size)
         d = (stop-1 - start)/step + 1
         arr = np.zeros(d, dtype=dtype)
         index = 0
         for i in range(start, stop, step):
            if i < 0:
               i = size+i
            if i >= 0 and i < size:
               arr[index] = array[i]
            else:
               raise IndexError('Index %d out of range [0,%d)'%(i, size))
            index += 1
         return arr
      else:
         errmsg = 'Index must be of type int or slice'
         raise ValueError(errmsg)

   def __setitem__(self, k, values):
      cdef int size = 0
      cdef ParOptScalar *array
      size = self.ptr.getArray(&array)
      if isinstance(k, int):
         if k < 0 or k >= size:
            errmsg = 'Index %d out of range [0,%d)'%(k, size)
            raise IndexError(errmsg)
         array[k] = values
      elif isinstance(k, slice):
         start, stop, step = k.indices(size)
         if hasattr(values, '__len__'):
            index = 0
            for i in range(start, stop, step):
               if i < 0:
                  i = size+i
               if i >= 0 and i < size:
                  array[i] = values[index]
               else:
                  raise IndexError('Index %d out of range [0,%d)'%(i, size))
               index += 1
         else:
            for i in range(start, stop, step):
               if i < 0:
                  i = size+i
               if i >= 0 and i < size:
                  array[i] = values
               else:
                  raise IndexError('Index %d out of range [0,%d)'%(i, size))
      else:
         errmsg = 'Index must be of type int or slice'
         raise ValueError(errmsg)
      return

   def copyValues(self, PVec vec):
      if self.ptr and vec.ptr:
         self.ptr.copyValues(vec.ptr)
      return

   def norm(self):
      return self.ptr.norm()

   def l1norm(self):
      return self.ptr.l1norm()

   def maxabs(self):
      return self.ptr.maxabs()

   def dot(self, PVec vec):
      return self.ptr.dot(vec.ptr)

# Python classes for the ParOptCompactQuasiNewton methods
cdef class CompactQuasiNewton:
   cdef ParOptCompactQuasiNewton *ptr
   def __cinit__(self):
      self.ptr = NULL

   def __dealloc__(self):
      if self.ptr:
         self.ptr.decref()

   def update(self, PVec s, PVec y):
      if self.ptr:
         self.ptr.update(s.ptr, y.ptr)

   def mult(self, PVec x, PVec y):
      if self.ptr:
         self.ptr.mult(x.ptr, y.ptr)

   def multAdd(self, ParOptScalar alpha, PVec x, PVec y):
      if self.ptr:
         self.ptr.multAdd(alpha, x.ptr, y.ptr)

cdef class LBFGS(CompactQuasiNewton):
   def __cinit__(self, pyParOptProblemBase prob, int subspace=10):
      self.ptr = new ParOptLBFGS(prob.ptr, subspace)
      self.ptr.incref()

cdef class LSR1(CompactQuasiNewton):
   def __cinit__(self, pyParOptProblemBase prob, int subspace=10):
      self.ptr = new ParOptLSR1(prob.ptr, subspace)
      self.ptr.incref()

# Python class for corresponding instance ParOpt
cdef class pyParOpt:
   cdef ParOpt *ptr
   def __cinit__(self, pyParOptProblemBase _prob, 
                 int max_qn_subspace, 
                 ParOptQuasiNewtonType qn_type):
      self.ptr = new ParOpt(_prob.ptr, max_qn_subspace, qn_type)
      self.ptr.incref()
      return
      
   def __dealloc__(self):
      if self.ptr:
         self.ptr.decref()
      
   # Perform the optimization
   def optimize(self, char *checkpoint=''):
      if checkpoint is None: 
         return self.ptr.optimize(NULL)
      else:
         return self.ptr.optimize(&checkpoint[0])
      
   def getOptimizedPoint(self):
      '''
      Get the optimized solution in PVec form for interpolation purposes
      '''
      cdef int ncon = 0
      cdef ParOptScalar *_z = NULL
      cdef ParOptVec *_x = NULL
      cdef ParOptVec *_zw = NULL
      cdef ParOptVec *_zl = NULL
      cdef ParOptVec *_zu = NULL
      cdef int self_owned = 0

      # Get the problem size/vector for the values
      self.ptr.getProblemSizes(NULL, &ncon, NULL, NULL)      
      self.ptr.getOptimizedPoint(&_x, &_z, &_zw, &_zl, &_zu);
      
      # Set the default values
      z = None
      x = None
      zw = None
      zl = None
      zu = None

      # Convert the multipliers to an in-place numpy array. This is
      # duplicated on all processors, and must have the same values
      # on all processors.
      if _z != NULL:
         z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z, self)

      # Note that these vectors are owned by the ParOpt class, we're simply
      # passing references to them back to the python layer.
      if _x != NULL:
         x = _init_PVec(_x)
      if _zw != NULL:
         zw = _init_PVec(_zw)
      if _zl != NULL:
         zl = _init_PVec(_zl)
      if _zu != NULL:
         zu = _init_PVec(_zu)

      return x, z, zw, zl, zu
   
   # Check objective and constraint gradients
   def checkGradients(self, double dh):    
      self.ptr.checkGradients(dh)
      
   # Set optimizer parameters
   def setNormType(self, ParOptNormType norm_typ):
      self.ptr.setNormType(norm_typ)

   def setBarrierStrategy(self, ParOptBarrierStrategy strategy):
      self.ptr.setBarrierStrategy(strategy)

   def setInitStartingPoint(self, int init):
      self.ptr.setInitStartingPoint(init)
      
   def setMaxMajorIterations(self, int iters):
      self.ptr.setMaxMajorIterations(iters)
      
   def setAbsOptimalityTol(self, double tol):
      self.ptr.setAbsOptimalityTol(tol)

   def setRelFunctionTol(self, double tol):
      self.ptr.setRelFunctionTol(tol)

   def setPenaltyGamma(self, double gamma):
      self.ptr.setPenaltyGamma(gamma)

   def setBarrierFraction(self, double frac):
      self.ptr.setBarrierFraction(frac)
      
   def setBarrierPower(self, double power):
      self.ptr.setBarrierPower(power)
      
   def setHessianResetFreq(self, int freq):
      self.ptr.setHessianResetFreq(freq)
   
   def setQNDiagonalFactor(self, double sigma):
      self.ptr.setQNDiagonalFactor(sigma)

   def setBFGSUpdateType(self, ParOptBFGSUpdateType update):
      self.ptr.setBFGSUpdateType(update)
      
   def setSequentialLinearMethod(self, int truth):
      self.ptr.setSequentialLinearMethod(truth)
      
   # Set/obtain the barrier parameter
   def setInitBarrierParameter(self, double mu):
      self.ptr.setInitBarrierParameter(mu)
      
   def getBarrierParameter(self):
      return self.ptr.getBarrierParameter()

   def getComplementarity(self):
      return self.ptr.getComplementarity()
  
   # Reset the quasi-Newton Hessian
   def resetQuasiNewtonHessian(self):
      self.ptr.resetQuasiNewtonHessian()

   # Reset the design variables and bounds
   def setQuasiNewton(self, CompactQuasiNewton qn):
      self.ptr.setQuasiNewton(qn.ptr)

   def setUseQuasiNewtonUpdates(self, int truth):
      self.ptr.setUseQuasiNewtonUpdates(truth)
      
   def resetDesignAndBounds(self):
      self.ptr.resetDesignAndBounds()

   # Set parameters associated with the linesearch
   def setUseLineSearch(self, int truth):
      self.ptr.setUseLineSearch(truth)
      
   def setMaxLineSearchIters(self, int iters):
      self.ptr.setMaxLineSearchIters(iters)
      
   def setBacktrackingLineSearch(self, int truth):
      self.ptr.setBacktrackingLineSearch(truth)
      
   def setArmijoParam(self, double c1):
      self.ptr.setArmijoParam(c1)
      
   def setPenaltyDescentFraction(self, double frac):
      self.ptr.setPenaltyDescentFraction(frac)
      
   # Set parameters for the interal GMRES algorithm
   def setUseHvecProduct(self, int truth):
      self.ptr.setUseHvecProduct(truth)

   # Set the use of an exact diagonal hessian
   def setUseDiagHessian(self, int truth):
      self.ptr.setUseDiagHessian(truth)
      
   def setUseQNGMRESPreCon(self, int truth):
      self.ptr.setUseQNGMRESPreCon(truth)
      
   def setNKSwitchTolerance(self, double tol):
      self.ptr.setNKSwitchTolerance(tol)
      
   def setEisenstatWalkerParameters(self, double gamma, double alpha):
      self.ptr.setEisenstatWalkerParameters(gamma, alpha)
      
   def setGMRESTolerances(self, double rtol, double atol):
      self.ptr.setGMRESTolerances(rtol, atol)
      
   def setGMRESSubspaceSize(self, int _gmres_subspace_size):
      self.ptr.setGMRESSubspaceSize(_gmres_subspace_size)
      
   # Set other parameters
   def setOutputFrequency(self, int freq):
      self.ptr.setOutputFrequency(freq)
      
   def setMajorIterStepCheck(self, int step):
      self.ptr.setMajorIterStepCheck(step)
      
   def setOutputFile(self, char *filename):
      if filename is not None:
         self.ptr.setOutputFile(filename)

   def setOutputLevel(self, int level):
      self.ptr.setOutputLevel(level)
         
   def setGradientCheckFrequency(self, int freq, double step_size):
       self.ptr.setGradientCheckFrequency(freq, step_size)
         
   # Write out the design variables to binary format (fast MPI/IO)
   def writeSolutionFile(self, char *filename):
      if filename is not None:
         return self.ptr.writeSolutionFile(filename)
  
   def readSolutionFile(self, char *filename):
      if filename is not None:
         return self.ptr.readSolutionFile(filename)

cdef class pyMMA(pyParOptProblemBase):
   cdef ParOptMMA *mma
   def __cinit__(self, pyParOptProblemBase _prob, use_mma=True):
      cdef int use_true_mma = 0
      if use_mma:
         use_true_mma = 1
      self.mma = new ParOptMMA(_prob.ptr, use_true_mma)
      self.mma.incref()
      self.ptr = self.mma
      return

   def setIteration(self, int mma_iter):
      self.mma.setIteration(mma_iter)

   def setMultipliers(self, np.ndarray[ParOptScalar, ndim=1, mode='c'] z,
                      PVec zw=None, PVec zl=None, PVec zu=None):
      cdef ParOptVec *v = NULL
      cdef ParOptVec *vl = NULL
      cdef ParOptVec *vu = NULL
      if zw is not None:
         v = zw.ptr
      if zl is not None:
         vl = zl.ptr
      if zu is not None:
         vu = zu.ptr
      self.mma.setMultipliers(<ParOptScalar*>z.data, v, vl, vu)
      return

   def initializeSubProblem(self, PVec vec=None):
      cdef ParOptVec *v = NULL
      if vec is not None:
         v = vec.ptr
      self.mma.initializeSubProblem(v)

   def computeKKTError(self):
      cdef double l1 = 0.0
      cdef double linfty = 0.0
      cdef double infeas = 0.0
      self.mma.computeKKTError(&l1, &linfty, &infeas)
      return l1, linfty, infeas

   def getOptimizedPoint(self):
      cdef ParOptVec *x
      self.mma.getOptimizedPoint(&x)
      return _init_PVec(x)

   def getAsymptotes(self):
      cdef ParOptVec *L = NULL
      cdef ParOptVec *U = NULL
      self.mma.getAsymptotes(&L, &U)
      return _init_PVec(L), _init_PVec(U)

   def getDesignHistory(self):
      cdef ParOptVec* x1 = NULL
      cdef ParOptVec *x2 = NULL
      self.mma.getDesignHistory(&x1, &x2)
      return _init_PVec(x1), _init_PVec(x2)
      
   def setPrintLevel(self, int level):
      self.mma.setPrintLevel(level)

   def setOutputFile(self, char *filename):
      self.mma.setOutputFile(filename)

   def setAsymptoteContract(self, double val):
      self.mma.setAsymptoteContract(val)

   def setAsymptoteRelax(self, double val):
      self.mma.setAsymptoteRelax(val)

   def setInitAsymptoteOffset(self, double val):
      self.mma.setInitAsymptoteOffset(val)

   def setMinAsymptoteOffset(self, double val):
      self.mma.setMinAsymptoteOffset(val)

   def setMaxAsymptoteOffset(self, double val):
      self.mma.setMaxAsymptoteOffset(val)

   def setBoundRelax(self, double val):
      self.mma.setBoundRelax(val)

   def setRegularization(self, double eps, double delta):
      self.mma.setRegularization(eps, delta)
