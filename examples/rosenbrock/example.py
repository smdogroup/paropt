# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt

# Import ParOpt
from paropt import ParOpt


# Create the rosenbrock function class
class Rosenbrock(ParOpt.Problem):
    def __init__(self):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = 2
        self.ncon = 1

        # The design history file
        self.x_hist = []

        # Initialize the base class
        super(Rosenbrock, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = -1.0
        lb[:] = -2.0
        ub[:] = 2.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1)
        fobj = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        con[0] = x[0] + x[1] + 5.0
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[0] = 200 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1 - x[0])
        g[1] = 200 * (x[1] - x[0] ** 2)

        # The constraint gradient
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail


problem = Rosenbrock()

options = {"algorithm": "ip"}
opt = ParOpt.Optimizer(problem, options)
opt.optimize()
