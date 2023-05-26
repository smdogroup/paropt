from mpi4py import MPI
import numpy as np
from paropt import ParOpt


# Create the rosenbrock function class
class Sellar(ParOpt.Problem):
    def __init__(self):
        # Initialize the base class
        nvars = 4
        ncon = 1
        super(Sellar, self).__init__(MPI.COMM_SELF, nvars=nvars, ncon=ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""

        x[0] = 2.0
        x[1] = 1.0
        x[2] = 0.0
        x[3] = 0.0

        lb[0] = 0.0
        lb[1] = 0.0
        lb[2] = -1.0
        lb[3] = -1.0

        ub[0] = 10.0
        ub[1] = 10.0
        ub[2] = 3.16
        ub[3] = 24.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        fail = 0
        fobj = x[1] * x[1] + x[0] + x[2] + np.exp(-x[3])
        cons = np.array([x[0] + x[1] - 1.0])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        g[0] = 1.0
        g[1] = 2.0 * x[1]
        g[2] = 1.0
        g[3] = -np.exp(-x[3])

        A[0][0] = 1.0
        A[0][1] = 1.0

        return fail


# Allocate the optimization problem
problem = Sellar()

# Set up the optimization problem
options = {}
opt = ParOpt.Optimizer(problem, options)
opt.optimize()
