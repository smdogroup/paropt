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

problem = Toy(comm)

options = {
    'algorithm': 'mma',
    'mma_init_asymptote_offset': 0.5,
    'mma_min_asymptote_offset': 0.01,
    'mma_bound_relax': 1e-4,
    'mma_max_iterations': 100}

# Create the ParOpt problem
opt = ParOpt.Optimizer(problem, options)

# Optimize
opt.optimize()
x, z, zw, zl, zu = opt.getOptimizedPoint()
