import numpy as np
import mpi4py.MPI as MPI
from paropt import ParOpt
import argparse
import os
import matplotlib.pylab as plt


# Create the rosenbrock function class
class ConvexProblem(ParOpt.Problem):
    def __init__(self, Q, Affine, b, Acon, bcon):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = len(b)
        self.ncon = 1

        # Record the quadratic terms
        self.Q = Q
        self.Affine = Affine
        self.b = b
        self.Acon = Acon
        self.bcon = bcon

        self.obj_scale = -1.0

        # Initialize the base class
        super(ConvexProblem, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = 0.05 + 0.9 * np.random.uniform(size=len(x))
        lb[:] = 0.0
        ub[:] = 1.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1, dtype=ParOpt.dtype)

        # Compute the artificial stiffness matrix
        self.K = self.Affine + np.dot(self.Q, np.dot(np.diag(x), self.Q.T))

        # Compute the displacements
        self.u = np.linalg.solve(self.K, self.b)

        # Compute the artifical compliance
        fobj = np.dot(self.u, self.b)

        if self.obj_scale < 0.0:
            self.obj_scale = 1.0 / fobj

        # Compute the linear constraint
        con[0] = self.bcon - np.dot(self.Acon, x)

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[:] = -np.dot(self.Q.T, self.u) ** 2

        # The constraint gradient
        A[0][:] = -self.Acon[:]

        return fail


def create_random_spd(n):
    """
    Create a random positive definite matrix with the given
    eigenvalues
    """
    # Create the eigenvalues for the matrix
    eigs = np.random.uniform(size=n)

    # Create a random square (n x n) matrix
    B = np.random.uniform(size=(n, n))

    # Orthogonalize the columns of B so that Q = range(B) = R^{n}
    Q, s, v = np.linalg.svd(B)

    # Compute A = Q*diag(eigs)*Q^{T}
    A = np.dot(Q, np.dot(np.diag(eigs), Q.T))

    return A


def solve_problem(eigs, filename=None, data_type="orthogonal", use_tr=False):
    # Create a random orthogonal Q vector
    if data_type == "orthogonal":
        B = np.random.uniform(size=(n, n))
        Q, s, v = np.linalg.svd(B)

        # Create a random Affine matrix
        Affine = create_random_spd(eigs)
    else:
        Q = np.random.uniform(size=(n, n))
        Affine = np.diag(1e-3 * np.ones(n))

    # Create the random right-hand-side
    b = np.random.uniform(size=n)

    # Create the constraint data
    Acon = np.random.uniform(size=n)
    bcon = 0.25 * np.sum(Acon)

    # Create the convex problem
    problem = ConvexProblem(Q, Affine, b, Acon, bcon)

    options = {
        "algorithm": "ip",
        "abs_res_tol": 1e-8,
        "starting_point_strategy": "affine_step",
        "barrier_strategy": "monotone",
        "start_affine_multiplier_min": 0.01,
        "penalty_gamma": 1000.0,
        "qn_subspace_size": 10,
        "qn_type": "bfgs",
        "output_file": filename,
    }

    if use_tr:
        options = {
            "algorithm": "tr",
            "tr_init_size": 0.05,
            "tr_min_size": 1e-6,
            "tr_max_size": 10.0,
            "tr_eta": 0.25,
            "tr_adaptive_gamma_update": True,
            "tr_max_iterations": 200,
            "penalty_gamma": 10.0,
            "qn_subspace_size": 10,
            "qn_type": "bfgs",
            "abs_res_tol": 1e-8,
            "output_file": filename,
            "tr_output_file": os.path.splitext(filename)[0] + ".tr",
            "starting_point_strategy": "affine_step",
            "barrier_strategy": "monotone",
            "use_line_search": False,
        }

    opt = ParOpt.Optimizer(problem, options)

    # Set a new starting point
    opt.optimize()
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    return x


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=100, help="Dimension of the problem")
parser.add_argument("--optimizer", type=str, default="ip")
args = parser.parse_args()

use_tr = False
if args.optimizer != "ip":
    use_tr = True

# Set the eigenvalues for the matrix
n = args.n
print("n = ", n)

np.random.seed(0)

# Solve the problem
x = solve_problem(n, filename="paropt.out", use_tr=use_tr)
