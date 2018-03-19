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
    enum PSPenaltyType"PSMultiTopoProperties::PSPenaltyType":
        PS_RAMP_CONVEX"PSMultiTopoProperties::PS_RAMP_CONVEX"
        PS_RAMP_FULL"PSMultiTopoProperties::PS_RAMP_FULL"
        PS_SIMP_CONVEX"PSMultiTopoProperties::PS_SIMP_CONVEX"
        PS_SIMP_FULL"PSMultiTopoProperties::PS_SIMP_FULL"
      
    cdef cppclass PSMultiTopoProperties(TACSObject):
        PSMultiTopoProperties(TacsScalar*, TacsScalar*, int)
        void setPenalization(double)
        double getPenalization()
        int getNumMaterials()
        void setPenaltyType(PSPenaltyType)
        PSPenaltyType getPenaltyType()
        
    cdef cppclass PSMultiTopo(PlaneStressStiffness):
        PSMultiTopo(PSMultiTopoProperties *_mats,
                    int nodes[], double weights[], int nnodes)
        void setLinearization(const TacsScalar*, int)
        int getFilteredDesignVars(const TacsScalar**)

    cdef cppclass LocatePoint:
        LocatePoint(const TacsScalar*, int, int)
        void locateKClosest(int, int*, TacsScalar*, const TacsScalar *)
        
    cdef void assembleResProjectDVSens(TACSAssembler*,
                                       const TacsScalar*, int,
                                       TacsScalar*, TACSBVec*)
    cdef void addNegdefiniteHessianProduct(TACSAssembler*,
                                           const TacsScalar*, int,
                                           TacsScalar*)

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
                  np.ndarray[TacsScalar, ndim=2, mode='c'] Cmat):
        assert((len(rho) == Cmat.shape[0]) and (Cmat.shape[1] == 6))
        self.ptr = new PSMultiTopoProperties(<TacsScalar*>rho.data,
                                             <TacsScalar*>Cmat.data,
                                             len(rho))
        self.ptr.incref()
        return

    def __dealloc__(self):
        self.ptr.decref()

    def getNumMaterials(self):
        return self.ptr.getNumMaterials()

    def setPenalization(self, double q):
        self.ptr.setPenalization(q)
        return

    def getPenalization(self):
        return self.ptr.getPenalization()

    def setPenaltyType(self, penalty='convex', ptype='ramp'):
        if penalty == 'convex' and ptype == 'ramp':
            self.ptr.setPenaltyType(PS_RAMP_CONVEX)
        elif penalty == 'full' and ptype == 'ramp':
            self.ptr.setPenaltyType(PS_RAMP_FULL)
        elif penalty == 'convex' and ptype == 'simp':
            self.ptr.setPenaltyType(PS_SIMP_CONVEX)
        elif penalty == 'full' and ptype == 'simp':
            self.ptr.setPenaltyType(PS_SIMP_FULL)

    def getPenaltyType(self):
        if self.ptr.getPenaltyType() == PS_RAMP_CONVEX:
            return 'convex', 'ramp'
        elif self.ptr.getPenaltyType() == PS_RAMP_FULL:
            return 'full', 'ramp'
        elif self.ptr.getPenaltyType() == PS_SIMP_CONVEX:
            return 'convex', 'simp'
        else:
            return 'full', 'simp'
    
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
    
    def getFilteredDesignVars(self):
        '''Get the filtered values of the design variables'''
        cdef int nmats
        cdef const TacsScalar *xf
        nmats = self.self_ptr.getFilteredDesignVars(&xf)
        x = np.zeros(nmats+1)
        for i in xrange(nmats+1):
            x[i] = xf[i]
        return x
    
def assembleProjectDVSens(Assembler assembler,
                          np.ndarray[TacsScalar, ndim=1, mode='c'] px,
                          np.ndarray[TacsScalar, ndim=1, mode='c'] deriv,
                          Vec residual):
    assert(len(deriv) == len(px))
    assembleResProjectDVSens(assembler.ptr,
                             <TacsScalar*>px.data, len(px),
                             <TacsScalar*>deriv.data,
                             residual.ptr)
    return

def addNegHessianProduct(Assembler assembler,
                         np.ndarray[TacsScalar, ndim=1, mode='c'] px,
                         np.ndarray[TacsScalar, ndim=1, mode='c'] deriv):
    assert(len(deriv) == len(px))
    addNegdefiniteHessianProduct(assembler.ptr,
                                 <TacsScalar*>px.data, len(px),
                                 <TacsScalar*>deriv.data)
    return
