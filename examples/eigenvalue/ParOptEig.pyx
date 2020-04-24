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

cdef extern from "ParOptCompactEigenvalueApprox.h":
    cppclass ParOptCompactEigenApprox(ParOptBase):
        ParOptCompactEigenApprox(ParOptProblem*, int)
        void getApproximation(ParOptScalar**, ParOptVec**,
                              int*, ParOptScalar**, ParOptVec***)

    cppclass ParOptEigenQuasiNewton(ParOptCompactQuasiNewton):
        ParOptEigenQuasiNewton(ParOptCompactQuasiNewton*, ParOptCompactEigenApprox*)

    cppclass ParOptEigenSubproblem(ParOptTrustRegionSubproblem):
        ParOptEigenSubproblem(ParOptProblem*, ParOptEigenQuasiNewton*)

cdef class CompactEigenApprox:
    cdef ParOptCompactEigenApprox *ptr
    def __cinit__(self, ProblemBase problem, int N):
        self.ptr = new ParOptCompactEigenApprox(problem.ptr, N)
        self.ptr.incref()

    def __dealloc__(self):
        if self.ptr != NULL:
            self.ptr.decref()

    def getApproximationVectors(self):
        cdef int N
        cdef ParOptVec *g0
        cdef ParOptVec **hvecs

        self.ptr.getApproximation(NULL, &g0, &N, NULL, &hvecs)

        hlist = []
        for i in range(N):
            hlist.append(_init_PVec(hvecs[i]))

        return _init_PVec(g0), hlist

    def setApproximationValues(self, ParOptScalar c, M):
        cdef int N
        cdef ParOptScalar *c0
        cdef ParOptScalar *M0

        self.ptr.getApproximation(&c0, NULL, &N, &M0, NULL)

        c0[0] = c
        for i in range(N):
            for j in range(N):
                M0[N*i + j] = M[i][j]

        return

cdef class EigenQuasiNewton(CompactQuasiNewton):
    cdef ParOptEigenQuasiNewton *eptr
    def __cinit__(self, CompactQuasiNewton qn, CompactEigenApprox eigh):
        cdef ParOptCompactQuasiNewton *qn_ptr = NULL
        if qn is not None:
            qn_ptr = qn.ptr
        self.eptr = new ParOptEigenQuasiNewton(qn_ptr, eigh.ptr)
        self.ptr = self.eptr
        self.ptr.incref()
        return

cdef class EigenSubproblem(TrustRegionSubproblem):
    def __cinit__(self, ProblemBase problem, EigenQuasiNewton eig):
        self.subproblem = new ParOptEigenSubproblem(problem.ptr, eig.eptr)
        self.subproblem.incref()
        self.ptr = self.subproblem
        return
