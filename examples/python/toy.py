# Create the toy example that is used in Svanberg MMA
from __future__ import print_function

# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
import argparse
import os

# Import ParOpt
from paropt import ParOpt

class Toy(ParOpt.Problem):
    def __init__(self, comm):
        # Set the communicator pointer
        self.comm = comm
        self.nvars = 3
        self.ncon = 2

        # The design history file
        self.x_hist = []

        # Initialize the base class
        super(Toy, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Set the values of the bounds'''
        x[0] = 4.0
        x[1] = 3.0
        x[2] = 2.0
        
        lb[:] = .0
        ub[:] = 5.0
        return
    
    def evalObjCon(self, x):
        '''Evaluate the objective and constraint'''
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(self.ncon)
        fobj = x[0]**2 + x[1]**2 + x[2]**2
        print("x is ", np.array(x))
        print("Objective is ", fobj)
        con[0] = 9.0 - (x[0]-5.)**2 - (x[1]-2)**2 - (x[2]-1)**2
        con[1] = 9.0 - (x[0]-3.)**2 - (x[1]-4)**2 - (x[2]-3)**2
        print("constraint values are ", np.array(con))
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        '''Evaluate the objective and constraint gradient'''
        fail = 0
        
        # The objective gradient
        g[0] = 2.0*x[0]
        g[1] = 2.0*x[1]
        g[2] = 2.0*x[2]

        A[0][0] = -2.0*(x[0] - 5.)
        A[0][1] = -2.0*(x[1] - 2.)
        A[0][2] = -2.0*(x[2] - 1.)

        A[1][0] = -2.0*(x[0] - 3.)
        A[1][1] = -2.0*(x[1] - 4.)
        A[1][2] = -2.0*(x[2] - 3.)
        return fail
    
# The communicator
comm = MPI.COMM_WORLD

# Create an argument parser to read in arguments from the commnad line
p = argparse.ArgumentParser()
p.add_argument('--prefix', type=str, default='./')
p.add_argument('--max_opt_iters', type=int, default=250)
p.add_argument('--opt_abs_tol', type=float, default=1e-6)
p.add_argument('--opt_barrier_frac', type=float, default=0.25)
p.add_argument('--opt_barrier_power', type=float, default=1.0)
p.add_argument('--output_freq', type=int, default=1)
p.add_argument('--max_lbfgs', type=int, default=10)
p.add_argument('--hessian_reset', type=int, default=10)
args = p.parse_args()

max_mma_iters = 10
problem = Toy(comm)
problem.setInequalityOptions(dense_ineq=True, sparse_ineq=False,
                             use_lower=True, use_upper=True)

# Set the ParOpt problem into MMA
mma = ParOpt.MMA(problem, use_mma=True)
mma.setInitAsymptoteOffset(0.5)
mma.setMinAsymptoteOffset(0.01)
mma.setBoundRelax(1e-4)
mma.setOutputFile(os.path.join(args.prefix, 
                            'mma_output.out'))

# Create the ParOpt problem
opt = ParOpt.InteriorPoint(mma, args.max_lbfgs, ParOpt.BFGS)

# Set parameters
opt.setMaxMajorIterations(args.max_opt_iters)
opt.setHessianResetFreq(args.hessian_reset)
opt.setAbsOptimalityTol(args.opt_abs_tol)
opt.setBarrierFraction(args.opt_barrier_frac)
opt.setBarrierPower(args.opt_barrier_power)
opt.setOutputFrequency(args.output_freq)
opt.setAbsOptimalityTol(1e-7)
opt.setUseDiagHessian(1)

# Set the starting point using the mass fraction
x = mma.getOptimizedPoint()
print('Initial x = ', np.array(x))

# Initialize the subproblem
mma.initializeSubProblem()
opt.resetDesignAndBounds()
filename = os.path.join(args.prefix, 'paropt.out')
opt.setOutputFile(filename)

# Enter the optimization loop
for i in range (max_mma_iters):
    print('Iteration number: ', i)
    opt.setInitBarrierParameter(0.1)
    opt.optimize()
        
    # Get the optimized point
    x, z, zw, zl, zu = opt.getOptimizedPoint()
    
    mma.setMultipliers(z, zw, zl, zu)
    mma.initializeSubProblem(x)
    opt.resetDesignAndBounds()
    
    # Compute the KKT error
    l1_norm, linfty_norm, infeas = mma.computeKKTError()
    if comm.rank == 0:
        print('z = ', z)
        print('l1_norm = ', l1_norm)
        print('linfty = ', linfty_norm)
        print('infeas = ', infeas)
        
    if l1_norm < 1e-5 and infeas < 1e-6:
        break
