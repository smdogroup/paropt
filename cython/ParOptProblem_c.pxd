#For the usage of MPI
cimport mpi4py.MPI as MPI
from mpi4py.libmpi cimport *
#For the declaration in the input arguments
cimport ParOptVec_c

import numpy as np
cimport numpy as np

cdef extern from "mpi-compat.h":
   pass
 
cdef extern from "ParOptProblem.h":
   cdef cppclass ParOptProblem:
      
      MPI_Comm _comm
      ParOptProblem()
      ParOptProblem(MPI_Comm _comm, int _nvars, int _ncon, int _nwcon, int _nwblock)
      
      #Get the communicator for the problem
      MPI_Comm getMPIComm()
      #Get the problem's dimensions
      void getProblemSizes(int *_nvars, int *_ncon, int *_nwcon, int *_nwblock)
      #Functions to indicate the type of sparse constraints
      int isSparseInequality()
      int isDenseInequality()
      int useLowerBounds()
      int useUpperBounds()
      #Get variables and bounds from the problem
      void getVarsAndBounds(ParOptVec_c.ParOptVec *x, ParOptVec_c.ParOptVec *lb,
                            ParOptVec_c.ParOptVec *ub)
      #Evaluate the objective and constraints
      int evalObjCon(ParOptVec_c.ParOptVec *x, double *fobj, double *cons)
      #Evaluate the objective and constraint gradients
      int evalObjConGradient(ParOptVec_c.ParOptVec *x, ParOptVec_c.ParOptVec *g,
                             ParOptVec_c.ParOptVec **Ac)
      #Evaluate the product of the Hessian with a given vector
      int evalHvecProduct(ParOptVec_c.ParOptVec *x, double *z, ParOptVec_c.ParOptVec *zw,
                          ParOptVec_c.ParOptVec *px, ParOptVec_c.ParOptVec *hvec)
      #Evaluate the constraints
      void evalSparseCon(ParOptVec_c.ParOptVec *x, ParOptVec_c.ParOptVec *out)
      #Compute the Jacobian-vector product out = J(x)*px
      void addSparseJacobian(double alpha, ParOptVec_c.ParOptVec *x, ParOptVec_c.ParOptVec *px,
                             ParOptVec_c.ParOptVec *out)
      #Compute the tranpose Jacobian-vector product out = J(x)^T*pzw
      void addSparseJacobianTranspose(double alpha, ParOptVec_c.ParOptVec *x,
                                      ParOptVec_c.ParOptVec *pzw, ParOptVec_c.ParOptVec *out)
      #Add the inner product of the constraints to the matrix such
      #that A += J(x)*cvec*J(x)^T, where cvec is a diagonal matrix
      void addSparseInnerProduct(double alpha, ParOptVec_c.ParOptVec *x,
                                 ParOptVec_c.ParOptVec *cvec, double *A)
      #Overwrite this function if the printing frequency is desired 
      #to match that of the output files
      void writeOutput(int iter, ParOptVec_c.ParOptVec *x)
      
