# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np

# Ensure that numpy is initialized
np.import_array()

# Import the definition required for const strings
from libc.string cimport const_char

# Import C methods for python
from cpython cimport PyObject, Py_INCREF

# Import the TACS module
from tacs.TACS cimport *
from tacs.constitutive cimport *

cdef extern from "mpi-compat.h":
    pass

cdef extern from "PSMultiTopo.h":
    cdef cppclass PSMultiTopoProperties(TACSObject):
        PSMultiTopoProperties(TacsScalar*, TacsScalar*,
                              TacsScalar*, int)
        void setPenalization(double)
        
    cdef cppclass PSMultiTopo(PlaneStressStiffness):
        PSMultiTopo(PSMultiTopoProperties *_mats,
                    int nodes[], double weights[], int nnodes)
        void setLinearization(const TacsScalar*, int)

    cdef cppclass LocatePoint:
        LocatePoint(const TacsScalar*, int, int)
        void locateKClosest(int, int*, TacsScalar*, const TacsScalar *)
        
    cdef void assembleResProjectDVSens(TACSAssembler*,
                                       const TacsScalar*, int, TACSBVec*)

cdef class Locator:
    cdef LocatePoint *ptr
    def __cinit__(self, np.ndarray[TacsScalar, ndim=2, mode='c'] xpts):
        assert(xpts.shape[1] == 3)
        self.ptr = new LocatePoint(<TacsScalar*>xpts.data,
                                   xpts.shape[0], 10)
        return

    def __dealloc__(self):
        del self.ptr
        return

    def closest(self, xpt, k=10):
        '''Find the closest points and return their distance'''
        cdef np.ndarray pt = np.array([xpt[0], xpt[1], xpt[2]], dtype=np.double)
        cdef np.ndarray indices = np.zeros(k, dtype=np.intc)
        cdef np.ndarray dist = np.zeros(k, dtype=np.double)
        self.ptr.locateKClosest(k, <int*>indices.data,
                                <double*>dist.data, <double*>pt.data)
        return indices, dist        

cdef class MultiTopoProperties:
    cdef PSMultiTopoProperties *ptr
    def __cinit__(self, 
                  np.ndarray[TacsScalar, ndim=1, mode='c'] rho,
                  np.ndarray[TacsScalar, ndim=1, mode='c'] E,
                  np.ndarray[TacsScalar, ndim=1, mode='c'] nu):
        assert((len(rho) == len(E)) and (len(rho) == len(nu)))
        self.ptr = new PSMultiTopoProperties(<TacsScalar*>rho.data,
                                             <TacsScalar*>E.data,
                                             <TacsScalar*>nu.data,
                                             len(rho))
        self.ptr.incref()
        return

    def __dealloc__(self):
        self.ptr.decref()
    
cdef class MultiTopo(PlaneStress):
    cdef PSMultiTopo *self_ptr
    def __cinit__(self, MultiTopoProperties props,
                  np.ndarray[int, ndim=1, mode='c'] nodes,
                  np.ndarray[double, ndim=1, mode='c'] weights):
        '''Multimaterial topology optimization'''
        assert(len(nodes) == len(weights))
        self.self_ptr = new PSMultiTopo(props.ptr, <int*>nodes.data,
                                        <double*>weights.data, len(nodes))
        self.ptr = self.self_ptr
        self.ptr.incref()
        return

    def setLinearization(self, np.ndarray[TacsScalar, ndim=1, mode='c'] dvs):
        self.self_ptr.setLinearization(<TacsScalar*>dvs.data, len(dvs))
        return
    
def assembleProjectDVSens(Assembler assembler,
                          np.ndarray[TacsScalar, ndim=1, mode='c'] px,
                          Vec residual):
    assembleResProjectDVSens(assembler.ptr,
                             <TacsScalar*>px.data, len(px),
                             residual.ptr)
    return
