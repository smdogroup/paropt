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
        ParOptProblem(MPI_Comm, int, int, int, int)
        ParOptVec *createDesignVec()
        ParOptVec *createConstraintVec()
        void checkGradients(double, ParOptVec*, int)

cdef extern from "ParOptQuasiNewton.h":
    cdef cppclass ParOptCompactQuasiNewton(ParOptBase):
        ParOptCompactQuasiNewton()
        void reset()
        int update(ParOptVec*, const double*, ParOptVec*,
                   ParOptVec*, ParOptVec*)
        void mult(ParOptVec*, ParOptVec*)
        void multAdd(ParOptScalar, ParOptVec*, ParOptVec*)

    cdef cppclass ParOptLBFGS(ParOptCompactQuasiNewton):
        ParOptLBFGS(ParOptProblem*, int)

    cdef cppclass ParOptLSR1(ParOptCompactQuasiNewton):
        ParOptLSR1(ParOptProblem*, int)

cdef class CompactQuasiNewton:
    cdef ParOptCompactQuasiNewton *ptr

cdef extern from "CyParOptProblem.h":
    # Define the callback types
    ctypedef void (*getvarsandbounds)(void *_self, int nvars, ParOptVec *x,
                                      ParOptVec *lb, ParOptVec *ub)
    ctypedef int (*evalobjcon)(void *_self, int nvars, int ncon,
                               ParOptVec *x, ParOptScalar *fobj,
                               ParOptScalar *cons)
    ctypedef int (*evalobjcongradient)(void *_self, int nvars, int ncon,
                                       ParOptVec *x, ParOptVec *gobj,
                                       ParOptVec **A)
    ctypedef int (*evalhvecproduct)(void *_self, int nvars, int ncon, int nwcon,
                                    ParOptVec *x, ParOptScalar *z,
                                    ParOptVec *zw, ParOptVec *px,
                                    ParOptVec *hvec)
    ctypedef int (*evalhessiandiag)(void *_self, int nvars, int ncon, int nwcon,
                                    ParOptVec *x, ParOptScalar *z,
                                    ParOptVec *zw, ParOptVec *hdiag)
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

    cdef cppclass CyParOptProblem(ParOptProblem):
        CyParOptProblem(MPI_Comm _comm, int _nvars, int _ncon,
                        int _nwcon, int _nwblock)

        # Set options for the inequality constraints
        void setInequalityOptions(int _isSparseInequal,
                                  int _isDenseInequal,
                                  int _useLower, int _useUpper)
        # Set the callback functions
        void setSelfPointer(void *_self)
        void setGetVarsAndBounds(getvarsandbounds usr_func)
        void setEvalObjCon(evalobjcon usr_func)
        void setEvalObjConGradient(evalobjcongradient usr_func)
        void setEvalHvecProduct(evalhvecproduct usr_func)
        void setEvalHessianDiag(evalhessiandiag usr_func)
        void setEvalSparseCon(evalsparsecon usr_func)
        void setAddSparseJacobian(addsparsejacobian usr_func)
        void setAddSparseJacobianTranspose(addsparsejacobiantranspose usr_func)
        void setAddSparseInnerProduct(addsparseinnerproduct usr_func)

cdef extern from "ParOptQuasiNewton.h":
    enum ParOptBFGSUpdateType:
        PAROPT_SKIP_NEGATIVE_CURVATURE
        PAROPT_DAMPED_UPDATE

cdef extern from "ParOptInteriorPoint.h":
    # Set the quasi-Newton type to use
    enum ParOptQuasiNewtonType:
        PAROPT_BFGS
        PAROPT_SR1
        PAROPT_NO_HESSIAN_APPROX

    enum ParOptNormType:
        PAROPT_INFTY_NORM
        PAROPT_L1_NORM
        PAROPT_L2_NORM

    enum ParOptBarrierStrategy:
        PAROPT_MONOTONE
        PAROPT_MEHROTRA
        PAROPT_COMPLEMENTARITY_FRACTION

    enum ParOptStartingPointStrategy:
        PAROPT_NO_START_STRATEGY
        PAROPT_LEAST_SQUARES_MULTIPLIERS
        PAROPT_AFFINE_STEP

    cppclass ParOptInteriorPoint(ParOptBase):
        ParOptInteriorPoint(ParOptProblem*, int,
                            ParOptQuasiNewtonType qn_type) except +

        # Perform the optimiztion
        int optimize(const char*)

        # Get the problem dimensions
        void getProblemSizes(int*, int*, int*, int*)

        # Retrieve the optimized point
        void getOptimizedPoint(ParOptVec**,
                               ParOptScalar**, ParOptVec**,
                               ParOptVec**, ParOptVec**)
        void getOptimizedSlacks(ParOptScalar**, ParOptScalar**, ParOptVec**)

        # Check objective and constraint gradients
        void checkGradients(double)

        # Set optimizer parameters
        void setNormType(ParOptNormType)
        void setBarrierStrategy(ParOptBarrierStrategy)
        void setStartingPointStrategy(ParOptStartingPointStrategy)
        void setMaxMajorIterations(int)
        void setAbsOptimalityTol(double)
        void setRelFunctionTol(double)
        void setAbsStepTol(double)
        void setPenaltyGamma(double)
        void setPenaltyGamma(double*)
        int getPenaltyGamma(double**)
        void setBarrierFraction(double)
        void setBarrierPower(double)
        void setHessianResetFreq(int)
        void setQNDiagonalFactor(double)
        void setBFGSUpdateType(ParOptBFGSUpdateType)
        void setSequentialLinearMethod(int)
        void setStartAffineStepMultiplierMin(double)

        # Set/obtain the barrier parameter
        void setInitBarrierParameter(double)
        double getBarrierParameter()
        void setRelativeBarrier(double)
        ParOptScalar getComplementarity()

        # Advanced quasi-Newton options
        void setQuasiNewton(ParOptCompactQuasiNewton*)
        void setUseQuasiNewtonUpdates(int)
        void resetQuasiNewtonHessian()

        # Reset the design variables and the bounds
        void resetDesignAndBounds()

        # Set parameters associated with the line search
        void setUseLineSearch(int)
        void setMaxLineSearchIters(int)
        void setBacktrackingLineSearch(int)
        void setArmijoParam(double)
        void setPenaltyDescentFraction(double)
        void setMinPenaltyParameter(double)

        # Set parameters for the internal GMRES algorithm
        void setUseDiagHessian(int)
        void setUseHvecProduct(int)
        void setUseQNGMRESPreCon(int)
        void setNKSwitchTolerance(double)
        void setEisenstatWalkerParameters(double, double)
        void setGMRESTolerances(double, double)
        void setGMRESSubspaceSize(int)

        # Set other parameters
        void setOutputFrequency(int)
        void setMajorIterStepCheck(int)
        void setGradientCheckFrequency(int, double)

        # Set the output file/print level
        void setOutputFile(const char*)
        void setOutputLevel(int)

        # Write out the design variables to binary format (fast MPI/IO)
        int writeSolutionFile(const char*)
        int readSolutionFile(const char*)

cdef extern from "ParOptMMA.h":
    cdef cppclass ParOptMMA(ParOptProblem):
        ParOptMMA(ParOptProblem*, int)
        void setIteration(int)
        void setMultipliers(ParOptScalar*, ParOptVec*, ParOptVec*, ParOptVec*)
        int initializeSubProblem(ParOptVec*)
        void computeKKTError(double*, double*, double*)
        void getOptimizedPoint(ParOptVec**)
        void getAsymptotes(ParOptVec**, ParOptVec**)
        void getDesignHistory(ParOptVec**, ParOptVec**)
        void setPrintLevel(int)
        void setOutputFile(const char*)
        void setAsymptoteContract(double)
        void setAsymptoteRelax(double)
        void setInitAsymptoteOffset(double)
        void setMinAsymptoteOffset(double)
        void setMaxAsymptoteOffset(double)
        void setBoundRelax(double)
        void setRegularization(double, double)

cdef extern from "ParOptTrustRegion.h":
    cdef cppclass ParOptTrustRegionSubproblem(ParOptProblem):
        pass

    cdef cppclass ParOptQuadraticSubproblem(ParOptTrustRegionSubproblem):
        ParOptQuadraticSubproblem(ParOptProblem*,
                                  ParOptCompactQuasiNewton*)

    cdef cppclass ParOptTrustRegion(ParOptBase):
        ParOptTrustRegion(ParOptTrustRegionSubproblem*,
                          double, double, double, double, double, double)
        void initialize()
        void update(ParOptVec*, const ParOptScalar*, ParOptVec*,
                    double*, double*, double*)
        void setOutputFile(const char*)
        void setPrintLevel(int)
        void setAdaptiveGammaUpdate(int)
        void setMaxTrustRegionIterations(int)
        void setTrustRegionTolerances(double, double, double)
        void setPenaltyGamma(double)
        void setPenaltyGamma(double*)
        int getPenaltyGamma(double**)
        void setPenaltyGammaMax(double)
        void setOutputFrequency(int)
        void optimize(ParOptInteriorPoint*)

cdef class ProblemBase:
    cdef ParOptProblem *ptr

cdef class TrustRegionSubproblem(ProblemBase):
    cdef ParOptTrustRegionSubproblem *subproblem
