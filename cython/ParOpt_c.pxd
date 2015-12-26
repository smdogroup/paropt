# For MPI capabilities
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np
from libc.string cimport const_char

cdef extern from "CyParOptProblem.h":
   # Define the callback types
   ctypedef void (*getvarsandbounds)(void *_self, int nvars,
                                     double *x, double *lb, double *ub)
   ctypedef int (*evalobjcon)(void *_self, int nvars, int ncon,
                              double *x, double *fobj, double *cons)
   ctypedef int (*evalobjcongradient)(void *_self, int nvars, int ncon,
                                      double *x, double *gobj, double *A)

   cdef cppclass CyParOptProblem:
      CyParOptProblem(MPI_Comm _comm, int _nvars, int _ncon)

      # Set the callback functions
      void setSelfPointer(void *_self)
      void setGetVarsAndBounds(getvarsandbounds usr_func)
      void setEvalObjCon(evalobjcon usr_func)
      void setEvalObjConGradient(evalobjcongradient usr_func)

cdef extern from "ParOpt.h":
   cdef cppclass ParOpt:
      ParOpt(CyParOptProblem *_prob, 
             int _max_lbfgs_subspace) except +
             
      # Perform the optimiztion
      int optimize(const_char *checkpoint)
            
      # Check objective and constraint gradients
      void checkGradients(double dh)
      
      # Set optimizer parameters
      void setInitStartingPoint(int init)
      void setMaxMajorIterations(int iters)
      void setAbsOptimalityTol(double tol)
      void setBarrierFraction(double frac)
      void setBarrierPower(double power)
      void setHessianResetFreq(int freq)
      void setSequentialLinearMethod(int truth)

      # Set/obtain the barrier parameter
      void setInitBarrierParameter(double mu)
      double getBarrierParameter()
      
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
      void setGMRESSusbspaceSize(int _gmres_subspace_size)

      # Set other parameters
      void setOutputFrequency(int freq)
      void setMajorIterStepCheck(int step)
      void setOutputFile(const_char *filename)

      # Write out the design variables to binary format (fast MPI/IO)
      int writeSolutionFile(const_char * filename)
      int readSolutionFile(const_char *filename)
