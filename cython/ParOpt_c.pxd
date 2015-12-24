# For MPI capabilities
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Import numpy
import numpy as np
cimport numpy as np
from libc.string cimport const_char

cdef extern from "ParOptVec.h":
   cdef cppclass ParOptVec:
      MPI_Comm _comm
      ParOptVec(MPI_Comm _comm, int n) 

      # Perform standard operations required for linear algebra
      void set(double alpha)
      void zeroEntries()
      void copyValues(ParOptVec * vec)
      double norm()
      double maxabs()
      double dot(ParOptVec *vec)
      void mdot(ParOptVec ** vecs, int nvecs, double *output)
      void scale(double alpha)
      void axpy(double alpha, ParOptVec *x)
      int getArray(double **array)
      int writeToFile(const char * filename)

cdef extern from "ParOptProblem.h":
   cdef cppclass ParOptProblem:
      MPI_Comm _comm
      ParOptProblem()
      ParOptProblem(MPI_Comm _comm, int _nvars, int _ncon, 
                    int _nwcon, int _nwblock)
      
      # Get the communicator for the problem
      MPI_Comm getMPIComm()

      # Get the problem's dimensions
      void getProblemSizes(int *_nvars, int *_ncon, 
                           int *_nwcon, int *_nwblock)

      # Functions to indicate the type of sparse constraints
      int isSparseInequality()
      int isDenseInequality()
      int useLowerBounds()
      int useUpperBounds()

      # Get variables and bounds from the problem
      void getVarsAndBounds(ParOptVec *x, 
                            ParOptVec *lb,
                            ParOptVec *ub)

      # Evaluate the objective and constraints
      int evalObjCon(ParOptVec *x, double *fobj, double *cons)

      # Evaluate the objective and constraint gradients
      int evalObjConGradient(ParOptVec *x, 
                             ParOptVec *g,
                             ParOptVec **Ac)

      # Evaluate the product of the Hessian with a given vector
      int evalHvecProduct(ParOptVec *x, double *z, 
                          ParOptVec *zw,
                          ParOptVec *px, 
                          ParOptVec *hvec)

      # Evaluate the constraints
      void evalSparseCon(ParOptVec *x, 
                         ParOptVec *out)

      # Compute the Jacobian-vector product out = J(x)*px
      void addSparseJacobian(double alpha, ParOptVec *x, 
                             ParOptVec *px,
                             ParOptVec *out)

      # Compute the tranpose Jacobian-vector product out = J(x)^T*pzw
      void addSparseJacobianTranspose(double alpha, 
                                      ParOptVec *x,
                                      ParOptVec *pzw, 
                                      ParOptVec *out)

      # Add the inner product of the constraints to the matrix such
      # that A += J(x)*cvec*J(x)^T, where cvec is a diagonal matrix
      void addSparseInnerProduct(double alpha, ParOptVec *x,
                                 ParOptVec *cvec, double *A)

      # Overwrite this function if the printing frequency is desired 
      # to match that of the output files
      void writeOutput(int iter, ParOptVec *x)

cdef extern from "ParOpt.h":
   cdef cppclass ParOpt:
      ParOpt(ParOptProblem *_prob, 
             int _max_lbfgs_subspace) except +
             
      # Perform the optimiztion
      int optimize( const_char *checkpoint)
      
      # Retrieve values of design variables and Lagrange multipliers 
      void getOptimizedPoint(ParOptVec **_x, 
                             const double **_z, 
                             ParOptVec **_zw, 
                             ParOptVec **_zl,
                             ParOptVec **_zu)
      
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
