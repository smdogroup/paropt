# distutils: language=c++

# For MPI capabilities
from mpi4py.MPI cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np

# Import ParOpt c++ headers
from paropt.cpp_headers.ParOpt cimport *

cdef class PVec:
    cdef ParOptVec *ptr

cdef inline _init_PVec(ParOptVec *ptr):
    vec = PVec()
    vec.ptr = ptr
    vec.ptr.incref()
    return vec

cdef class CompactQuasiNewton:
    cdef ParOptCompactQuasiNewton *ptr

cdef class ProblemBase:
    cdef ParOptProblem *ptr

cdef class TrustRegionSubproblem(ProblemBase):
    cdef ParOptTrustRegionSubproblem *subproblem
