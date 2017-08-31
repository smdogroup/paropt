# For MPI capabilities
from mpi4py.MPI cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np

# Typdefs required for either real or complex mode
include "ParOptTypedefs.pxi"

cdef extern from "ParOptProblem.h":
   cdef cppclass ParOptProblem:
      ParOptProblem()
      ParOptProblem(MPI_Comm)
      ParOptProblem(MPI_Comm, int, int, int, int)

cdef extern from "CyParOptProblem.h":
   # Define the callback types
   ctypedef void (*getvarsandbounds)(void *_self, int nvars,
                                     ParOptScalar *x,
                                     ParOptScalar *lb, ParOptScalar *ub)
   ctypedef int (*evalobjcon)(void *_self, int nvars, int ncon,
                              ParOptScalar *x, ParOptScalar *fobj,
                              ParOptScalar *cons)
   ctypedef int (*evalobjcongradient)(void *_self, int nvars, int ncon,
                                      ParOptScalar *x, ParOptScalar *gobj,
                                      ParOptScalar *A)
   ctypedef int (*evalhvecproduct)(void *_self, int nvars, int ncon, 
                                   int nwcon, ParOptScalar *x, ParOptScalar *z,
                                   ParOptScalar *zw, ParOptScalar *px,
                                   ParOptScalar *hvec)
   ctypedef void (*evalsparsecon)(void *_self, int nvars, int nwcon,
                                  ParOptScalar *x, ParOptScalar *out)
   ctypedef void (*addsparsejacobian)(void *_self, int nvars, int nwcon,
                                      ParOptScalar alpha, ParOptScalar *x, 
                                      ParOptScalar *px, ParOptScalar *out)
   ctypedef void (*addsparsejacobiantranspose)(void *_self, 
                                               int nvars, int nwcon,
                                               ParOptScalar alpha, 
                                               ParOptScalar *x, 
                                               ParOptScalar *px, 
                                               ParOptScalar *out)
   ctypedef void (*addsparseinnerproduct)(void *_self, int nvars, 
                                          int nwcon, int nwblock,
                                          ParOptScalar alpha, ParOptScalar *x, 
                                          ParOptScalar *c, ParOptScalar *A)

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
      void setEvalSparseCon(evalsparsecon usr_func)
      void setAddSparseJacobian(addsparsejacobian usr_func)
      void setAddSparseJacobianTranspose(addsparsejacobiantranspose usr_func)
      void setAddSparseInnerProduct(addsparseinnerproduct usr_func)

cdef extern from "ParOptQuasiNewton.h":
   enum BFGSUpdateType"LBFGS::BFGSUpdateType":
      SKIP_NEGATIVE_CURVATURE"LBFGS::SKIP_NEGATIVE_CURVATURE"
      DAMPED_UPDATE"LBFGS::DAMPED_UPDATE"

cdef extern from "ParOptVec.h":
   cppclass ParOptVec:
      ParOptVec(MPI_Comm comm, int n)
      
      # Retrieve the values from the array
      int getArray(ParOptScalar **array)

cdef extern from "ParOpt.h":
   # Set the quasi-Newton type to use
   enum QuasiNewtonType"ParOpt::QuasiNewtonType": 
      PAROPT_BFGS"ParOpt::BFGS"
      PAROPT_SR1"ParOpt::SR1"

   cppclass ParOpt:
      ParOpt(ParOptProblem *_prob, int _max_lbfgs_subspace, 
             QuasiNewtonType qn_type) except +

      # Perform the optimiztion
      int optimize(const char *checkpoint)

      # Get the problem dimensions
      void getProblemSizes(int *nvars, int *ncon, 
                           int *nwcon, int *nwblock)

      # Get the initial multipliers
      void getInitMultipliers(ParOptScalar**, ParOptVec**,
                              ParOptVec**, ParOptVec**)
      
      # Retrieve the optimized point
      void getOptimizedPoint(ParOptVec **_x,
                             const ParOptScalar **_z, ParOptVec **_zw,
                             ParOptVec **_zl, ParOptVec **_zu)

      # Check objective and constraint gradients
      void checkGradients(double dh)
      
      # Set optimizer parameters
      void setInitStartingPoint(int init)
      void setMaxMajorIterations(int iters)
      void setAbsOptimalityTol(double tol)
      void setRelFunctionTol(double tol)
      void setBarrierFraction(double frac)
      void setBarrierPower(double power)
      void setHessianResetFreq(int freq)
      void setQNDiagonalFactor(double sigma)
      void setBFGSUpdateType(BFGSUpdateType)
      void setSequentialLinearMethod(int truth)

      # Set/obtain the barrier parameter
      void setInitBarrierParameter(double mu)
      double getBarrierParameter()
      ParOptScalar getComplementarity()

      # Reset the quasi-Newton approximation
      void resetQuasiNewtonHessian()

      # Reset the design variables and the bounds
      void resetDesignAndBounds()
      
      # Set parameters associated with the line search
      void setUseLineSearch(int truth)
      void setMaxLineSearchIters(int iters)
      void setBacktrackingLineSearch(int truth)
      void setArmijioParam(double c1)
      void setPenaltyDescentFraction(double frac)

      # Set parameters for the internal GMRES algorithm
      void setUseHvecProduct(int truth)
      void setUseQNGMRESPreCon(int truth)
      void setNKSwitchTolerance(double tol)
      void setEisenstatWalkerParameters(double gamma, double alpha)
      void setGMRESTolerances(double rtol, double atol)
      void setGMRESSubspaceSize(int _gmres_subspace_size)

      # Set other parameters
      void setOutputFrequency(int freq)
      void setMajorIterStepCheck(int step)
      void setOutputFile(const char *filename)
      void setGradientCheckFrequency(int freq, double step_size)

      # Write out the design variables to binary format (fast MPI/IO)
      int writeSolutionFile(const char *filename)
      int readSolutionFile(const char *filename)

cdef class pyParOptProblemBase:
   cdef ParOptProblem *ptr

