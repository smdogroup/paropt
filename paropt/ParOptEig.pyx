# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np

# Ensure that numpy is initialized
np.import_array()

# Import the TACS module
from paropt.ParOpt cimport *

# Import tracebacks for callbacks
import traceback

cdef extern from "ParOptCompactEigenvalueApprox.h":
    cppclass ParOptCompactEigenApprox(ParOptBase):
        ParOptCompactEigenApprox(ParOptProblem*, int)
        void getApproximation(ParOptScalar**, ParOptVec**,
                              int*, ParOptScalar**, ParOptScalar**, ParOptVec***)

    cppclass ParOptEigenQuasiNewton(ParOptCompactQuasiNewton):
        ParOptEigenQuasiNewton(ParOptCompactQuasiNewton*, ParOptCompactEigenApprox*, int)

    ctypedef void (*updateeigenmodel)(void*, ParOptVec*, ParOptCompactEigenApprox*)

    cppclass ParOptEigenSubproblem(ParOptTrustRegionSubproblem):
        ParOptEigenSubproblem(ParOptProblem*, ParOptEigenQuasiNewton*)
        void setEigenModelUpdate(void*, updateeigenmodel)

cdef class CompactEigenApprox:
    cdef ParOptCompactEigenApprox *ptr
    def __cinit__(self, ProblemBase problem=None, N=None):
        if problem is not None and N is not None:
            self.ptr = new ParOptCompactEigenApprox(problem.ptr, N)
            self.ptr.incref()
        else:
            self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            self.ptr.decref()

    def getApproximationVectors(self):
        cdef int N = 0
        cdef ParOptVec *g0 = NULL
        cdef ParOptVec **hvecs = NULL

        if self.ptr is not NULL:
            self.ptr.getApproximation(NULL, &g0, &N, NULL, NULL, &hvecs)

        hlist = []
        for i in range(N):
            hlist.append(_init_PVec(hvecs[i]))

        return _init_PVec(g0), hlist

    def setApproximationValues(self, ParOptScalar c, M, Minv):
        cdef int N = 0
        cdef ParOptScalar *c0 = NULL
        cdef ParOptScalar *M0 = NULL
        cdef ParOptScalar *M0inv = NULL

        if self.ptr is not NULL:
            self.ptr.getApproximation(&c0, NULL, &N, &M0, &M0inv, NULL)

        if c0 is not NULL:
            c0[0] = c
        for i in range(N):
            for j in range(N):
                M0[N*i + j] = M[i][j]
                M0inv[N*i + j] = Minv[i][j]

        return

cdef _init_CompactEigenApprox(ParOptCompactEigenApprox *ptr):
    obj = CompactEigenApprox()
    obj.ptr = ptr
    obj.ptr.incref()
    return obj

cdef class EigenQuasiNewton(CompactQuasiNewton):
    cdef ParOptEigenQuasiNewton *eptr
    def __cinit__(self, CompactQuasiNewton qn, CompactEigenApprox eigh, int index=0):
        cdef ParOptCompactQuasiNewton *qn_ptr = NULL
        if qn is not None:
            qn_ptr = qn.ptr
        self.eptr = new ParOptEigenQuasiNewton(qn_ptr, eigh.ptr, index)
        self.ptr = self.eptr
        self.ptr.incref()
        return

cdef void _updateeigenmodel(void *_self, ParOptVec *_x,
                            ParOptCompactEigenApprox *_approx):
    fail = 0
    try:
        obj = <object>_self
        x = _init_PVec(_x)
        approx = _init_CompactEigenApprox(_approx)
        obj(x, approx)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

cdef class EigenSubproblem(TrustRegionSubproblem):
    cdef ParOptEigenSubproblem *me
    cdef object callback
    def __init__(self, ProblemBase problem, EigenQuasiNewton eig):
        self.me = new ParOptEigenSubproblem(problem.ptr, eig.eptr)
        self.subproblem = self.me
        self.subproblem.incref()
        self.ptr = self.subproblem
        self.callback = None
        return

    def setUpdateEigenModel(self, callback):
        self.callback = callback
        self.me.setEigenModelUpdate(<void*>self.callback, _updateeigenmodel)
        return
