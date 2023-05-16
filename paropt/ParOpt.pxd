# distutils: language=c++

# For MPI capabilities
from mpi4py.MPI cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np

# Typdefs required for either real or complex mode
include "ParOptTypedefs.pxi"

cdef extern from "ParOptVec.h":
    cppclass ParOptBase:
        void incref()
        void decref()

    cppclass ParOptVec(ParOptBase):
        ParOptVec()
        ParOptVec(MPI_Comm, int)
        void zeroEntries()
        int getArray(ParOptScalar**)
        void copyValues(ParOptVec*)
        ParOptScalar norm()
        ParOptScalar l1norm()
        ParOptScalar maxabs()
        ParOptScalar dot(ParOptVec*)

cdef class PVec:
    cdef ParOptVec *ptr

cdef inline _init_PVec(ParOptVec *ptr):
    vec = PVec()
    vec.ptr = ptr
    vec.ptr.incref()
    return vec

cdef extern from "ParOptProblem.h":
    cdef cppclass ParOptProblem(ParOptBase):
        ParOptProblem()
        ParOptProblem(MPI_Comm)
        MPI_Comm getMPIComm()
        void setProblemSizes(int, int, int)
        void setNumInequalities(int, int)
        ParOptVec *createDesignVec()
        ParOptVec *createConstraintVec()
        void getProblemSizes(int*, int*, int*)
        void checkGradients(double, ParOptVec*, int)

cdef extern from "ParOptQuasiNewton.h":
    enum ParOptBFGSUpdateType:
        PAROPT_SKIP_NEGATIVE_CURVATURE
        PAROPT_DAMPED_UPDATE

    enum ParOptQuasiNewtonDiagonalType:
        PAROPT_YTY_OVER_YTS
        PAROPT_YTS_OVER_STS
        PAROPT_INNER_PRODUCT_YTY_OVER_YTS
        PAROPT_INNER_PRODUCT_YTS_OVER_STS

    cdef cppclass ParOptCompactQuasiNewton(ParOptBase):
        ParOptCompactQuasiNewton()
        void setInitDiagonalType(ParOptQuasiNewtonDiagonalType)
        void reset()
        int update(ParOptVec*, const double*, ParOptVec*,
                   ParOptVec*, ParOptVec*)
        void mult(ParOptVec*, ParOptVec*)
        void multAdd(ParOptScalar, ParOptVec*, ParOptVec*)

    cdef cppclass ParOptLBFGS(ParOptCompactQuasiNewton):
        ParOptLBFGS(ParOptProblem*, int)
        void setBFGSUpdateType(ParOptBFGSUpdateType)

    cdef cppclass ParOptLSR1(ParOptCompactQuasiNewton):
        ParOptLSR1(ParOptProblem*, int)

cdef class CompactQuasiNewton:
    cdef ParOptCompactQuasiNewton *ptr

cdef extern from "CyParOptProblem.h":
    ctypedef void (*getvarsandbounds)(void *_self, int nvars, ParOptVec *x,
                                      ParOptVec *lb, ParOptVec *ub)
    ctypedef int (*evalobjcon)(void *_self, int nvars, int ncon,
                               ParOptVec *x, ParOptScalar *fobj,
                               ParOptScalar *cons)
    ctypedef int (*evalobjcongradient)(void *_self, int nvars, int ncon,
                                       ParOptVec *x, ParOptVec *gobj,
                                       ParOptVec **A)
    ctypedef int (*evalsparseobjcon)(void *_self, int nvars, int ncon, int nwcon,
                                     ParOptVec *x, ParOptScalar *fobj,
                                     ParOptScalar *cons, ParOptVec *sparse_con)
    ctypedef int (*evalsparseobjcongradient)(void *_self, int nvars, int ncon, int nwcon,
                                             ParOptVec *x, ParOptVec *gobj,
                                             ParOptVec **A, int nnz, ParOptScalar *data)
    ctypedef int (*evalhvecproduct)(void *_self, int nvars, int ncon, int nwcon,
                                    ParOptVec *x, ParOptScalar *z,
                                    ParOptVec *zw, ParOptVec *px,
                                    ParOptVec *hvec)
    ctypedef int (*evalhessiandiag)(void *_self, int nvars, int ncon, int nwcon,
                                    ParOptVec *x, ParOptScalar *z,
                                    ParOptVec *zw, ParOptVec *hdiag)
    ctypedef void (*computequasinewtonupdatecorrection)(void *_self, int nvars, int ncon,
                                                        ParOptVec *x, ParOptScalar *z,
                                                        ParOptVec *zw,
                                                        ParOptVec *s, ParOptVec *y)
    ctypedef void (*evalsparsecon)(void *_self, int nvars, int nwcon,
                                   ParOptVec *x, ParOptVec *out)
    ctypedef void (*addsparsejacobian)(void *_self, int nvars, int nwcon,
                                       ParOptScalar alpha,
                                       ParOptVec *x, ParOptVec *px,
                                       ParOptVec *out)
    ctypedef void (*addsparsejacobiantranspose)(void *_self,
                                                int nvars, int nwcon,
                                                ParOptScalar alpha,
                                                ParOptVec *x, ParOptVec *px,
                                                ParOptVec *out)
    ctypedef void (*addsparseinnerproduct)(void *_self, int nvars,
                                           int nwcon, int nwblock,
                                           ParOptScalar alpha,
                                           ParOptVec *x, ParOptVec *c,
                                           ParOptScalar *out)

    cdef cppclass CyParOptBlockProblem(ParOptProblem):
        CyParOptBlockProblem(MPI_Comm, int)
        void setVarBoundOptions(int, int)
        void setSelfPointer(void *_self)
        void setGetVarsAndBounds(getvarsandbounds usr_func)
        void setEvalObjCon(evalobjcon usr_func)
        void setEvalObjConGradient(evalobjcongradient usr_func)
        void setEvalHvecProduct(evalhvecproduct usr_func)
        void setEvalHessianDiag(evalhessiandiag usr_func)
        void setComputeQuasiNewtonUpdateCorrection(computequasinewtonupdatecorrection usr_func)
        void setEvalSparseCon(evalsparsecon usr_func)
        void setAddSparseJacobian(addsparsejacobian usr_func)
        void setAddSparseJacobianTranspose(addsparsejacobiantranspose usr_func)
        void setAddSparseInnerProduct(addsparseinnerproduct usr_func)

    cdef cppclass CyParOptSparseProblem(ParOptProblem):
        CyParOptSparseProblem(MPI_Comm)
        void setVarBoundOptions(int, int)
        void setSparseJacobianData(const int *, const int*)
        void setSelfPointer(void *_self)
        void setGetVarsAndBounds(getvarsandbounds usr_func)
        void setEvalObjCon(evalsparseobjcon usr_func)
        void setEvalObjConGradient(evalsparseobjcongradient usr_func)
        void setEvalHvecProduct(evalhvecproduct usr_func)
        void setEvalHessianDiag(evalhessiandiag usr_func)
        void setComputeQuasiNewtonUpdateCorrection(computequasinewtonupdatecorrection usr_func)

cdef extern from "ParOptOptions.h":
    cppclass ParOptOptions(ParOptBase):
        ParOptOptions()
        ParOptOptions(MPI_Comm)
        int isOption(const char*)
        int setOption(const char*, const char*)
        int setOption(const char* int)
        int setOption(const char*, double)

        const char* getStringOption(const char*)
        int getBoolOption(const char*)
        int getIntOption(const char*)
        double getFloatOption(const char*)
        const char* getEnumOption(const char*)
        int getOptionType(const char*)
        const char* getDescription(const char*)

        int getIntRange(const char*, int*, int*)
        int getFloatRange(const char*, double*, double*)
        int getEnumRange(const char*, int*, char***)

        void begin()
        const char* getName()
        int next()

cdef extern from "ParOptInteriorPoint.h":
    cppclass ParOptInteriorPoint(ParOptBase):
        ParOptInteriorPoint(ParOptProblem*, ParOptOptions*) except +
        int optimize(const char*)
        void getProblemSizes(int*, int*, int*)
        void getOptimizedPoint(ParOptVec**,
                               ParOptScalar**, ParOptVec**,
                               ParOptVec**, ParOptVec**)
        void getOptimizedSlacks(ParOptScalar**, ParOptScalar**, ParOptVec**, ParOptVec**)
        void checkGradients(double)
        void setPenaltyGamma(double)
        void setPenaltyGamma(double*)
        double getBarrierParameter()
        ParOptScalar getComplementarity()
        void setQuasiNewton(ParOptCompactQuasiNewton*)
        void resetQuasiNewtonHessian()
        void resetDesignAndBounds()
        int writeSolutionFile(const char*)
        int readSolutionFile(const char*)

    void ParOptInteriorPointAddDefaultOptions"ParOptInteriorPoint::addDefaultOptions"(ParOptOptions*)

cdef extern from "ParOptMMA.h":
    cdef cppclass ParOptMMA(ParOptProblem):
        ParOptMMA(ParOptProblem*, ParOptOptions*)
        void optimize(ParOptInteriorPoint*)
        void getOptimizedPoint(ParOptVec**)
        void getAsymptotes(ParOptVec**, ParOptVec**)
        void getDesignHistory(ParOptVec**, ParOptVec**)

    void ParOptMMAAddDefaultOptions"ParOptMMA::addDefaultOptions"(ParOptOptions*)

cdef extern from "ParOptTrustRegion.h":
    cdef cppclass ParOptTrustRegionSubproblem(ParOptProblem):
        pass

    cdef cppclass ParOptQuadraticSubproblem(ParOptTrustRegionSubproblem):
        ParOptQuadraticSubproblem(ParOptProblem*,
                                  ParOptCompactQuasiNewton*)

    cdef cppclass ParOptTrustRegion(ParOptBase):
        ParOptTrustRegion(ParOptTrustRegionSubproblem*,
                          ParOptOptions*)
        void setPenaltyGamma(double)
        void setPenaltyGamma(double*)
        int getPenaltyGamma(double**)
        void setPenaltyGammaMax(double)
        void setPenaltyGammaMin(double)
        void optimize(ParOptInteriorPoint*)
        void getOptimizedPoint(ParOptVec**)

    void ParOptTrustRegionAddDefaultOptions"ParOptTrustRegion::addDefaultOptions"(ParOptOptions*)

cdef extern from "ParOptOptimizer.h":
    cdef cppclass ParOptOptimizer(ParOptBase):
        ParOptOptimizer(ParOptProblem*, ParOptOptions*)
        ParOptOptions* getOptions()
        ParOptProblem* getProblem()
        void optimize()
        void getOptimizedPoint(ParOptVec**, ParOptScalar**,
                               ParOptVec**, ParOptVec**, ParOptVec**)
        void setTrustRegionSubproblem(ParOptTrustRegionSubproblem*)

    void ParOptOptimizerAddDefaultOptions"ParOptOptimizer::addDefaultOptions"(ParOptOptions*)

cdef class ProblemBase:
    cdef ParOptProblem *ptr

cdef class TrustRegionSubproblem(ProblemBase):
    cdef ParOptTrustRegionSubproblem *subproblem
