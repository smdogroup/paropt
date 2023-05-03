"""
This code demonstrates how to create a reduced optimization problem by fixing
a subset of design variables.

Original problem:
min  x0**4 + x1**4 + x2**4
s.t. x0 + x1 + x2 - 1 >= 0

reduced problem:
fix: x0 = 0.1
"""

import os
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
from paropt import ParOpt


class OriginalProblem(ParOpt.Problem):
    def __init__(self):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = 3
        self.ncon = 1

        # Initialize the base class
        super().__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = 1.0
        lb[:] = 0.0
        ub[:] = 10.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        fail = 0
        con = np.zeros(1)
        fobj = x[0] ** 4 + x[1] ** 4 + x[2] ** 4
        con[0] = x[0] + x[1] + x[2] - 1
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[0] = 4 * x[0] ** 3
        g[1] = 4 * x[1] ** 3
        g[2] = 4 * x[2] ** 3

        # The constraint gradient
        A[0][0] = 1.0
        A[0][1] = 1.0
        A[0][2] = 1.0
        return fail


class ReducedProblem(ParOpt.Problem):
    def __init__(self, original_prob, fixed_dv_idx, fixed_dv_vals):
        self.comm = MPI.COMM_WORLD
        self.ncon = 1
        self.prob = original_prob

        # Allocate full-size vectors for the original problem
        self._x = self.prob.createDesignVec()
        self._g = self.prob.createDesignVec()
        self._A = []
        for i in range(self.ncon):
            self._A.append(self.prob.createDesignVec())

        # Get indices of fixed design variables, these indices
        # are with respect to the original full-sized problem
        self.fixed_dv_idx = fixed_dv_idx
        self.fixed_dv_vals = fixed_dv_vals

        # Compute the indices of fixed design variables, these indices
        # are with respect to the original full-sized problem
        self.free_dv_idx = [
            i for i in range(len(self._x)) if i not in self.fixed_dv_idx
        ]
        self.nvars = len(self.free_dv_idx)

        # Get vars and bounds from the original problem
        self._x0 = self.prob.createDesignVec()
        self._lb = self.prob.createDesignVec()
        self._ub = self.prob.createDesignVec()
        self.prob.getVarsAndBounds(self._x0, self._lb, self._ub)

        super().__init__(self.comm, self.nvars, self.ncon)
        return

    def getVarsAndBounds(self, x, lb, ub):
        x[:] = self._x0[self.free_dv_idx]
        lb[:] = self._lb[self.free_dv_idx]
        ub[:] = self._ub[self.free_dv_idx]
        return

    def evalObjCon(self, x):
        self._x[self.fixed_dv_idx] = self.fixed_dv_vals
        self._x[self.free_dv_idx] = x[:]
        return self.prob.evalObjCon(self._x)

    def evalObjConGradient(self, x, g, A):
        self._x[self.fixed_dv_idx] = self.fixed_dv_vals
        self._x[self.free_dv_idx] = x[:]
        fail = self.prob.evalObjConGradient(self._x, self._g, self._A)
        g[:] = self._g[self.free_dv_idx]
        for i in range(self.ncon):
            A[i][:] = self._A[i][self.free_dv_idx]
        return fail


options = {
    "algorithm": "tr",
    "tr_init_size": 0.05,
    "tr_min_size": 1e-6,
    "tr_max_size": 10.0,
    "tr_eta": 0.1,
    "tr_adaptive_gamma_update": True,
    "tr_max_iterations": 200,
}

original = OriginalProblem()
redu = ReducedProblem(original, fixed_dv_idx=[0], fixed_dv_vals=[0.1])
opt = ParOpt.Optimizer(redu, options)
opt.optimize()
