import numpy as np
import mpi4py.MPI as MPI
from paropt import ParOpt
import argparse
import os
import matplotlib.pylab as plt


# Random quadratic problem class
class Quadratic(ParOpt.Problem):
    def __init__(self, A, b, Acon, bcon):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = len(b)
        self.ncon = 1

        # Record the quadratic terms
        self.A = A
        self.b = b
        self.Acon = Acon
        self.bcon = bcon

        # Initialize the base class
        super(Quadratic, self).__init__(self.comm, nvars=self.nvars, ncon=self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = -2.0 + np.random.uniform(size=len(x))
        lb[:] = -5.0
        ub[:] = 5.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Append the point to the solution history

        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1, dtype=ParOpt.dtype)

        fobj = 0.5 * np.dot(x, np.dot(self.A, x)) + np.dot(self.b, x)
        con[0] = np.dot(x, self.Acon) + self.bcon
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[:] = np.dot(self.A, x) + self.b

        # The constraint gradient
        A[0][:] = self.Acon[:]

        return fail


def create_random_problem(eigs):
    """
    Create a random positive definite matrix with the given
    eigenvalues
    """

    # The dimension of the matrix
    n = len(eigs)

    # Create a random square (n x n) matrix
    B = np.random.uniform(size=(n, n))

    # Orthogonalize the columns of B so that Q = range(B) = R^{n}
    Q, s, v = np.linalg.svd(B)

    # Compute A = Q*diag(eigs)*Q^{T}
    A = np.dot(Q, np.dot(np.diag(eigs), Q.T))

    return A


def solve_problem(eigs, filename=None, use_stdout=False, use_tr=False):
    # Get the A matrix
    A = create_random_problem(eigs)

    # Create the other problem data
    b = np.random.uniform(size=len(eigs))
    Acon = np.random.uniform(size=len(eigs))
    bcon = np.random.uniform()

    problem = Quadratic(A, b, Acon, bcon)

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
            "tr_max_iterations": 1000,
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

    return


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="ip")
parser.add_argument("--n", type=int, default=100, help="Dimension of the problem")
parser.add_argument("--eig_min", type=float, default=1.0, help="Minimum eigenvalue")
parser.add_argument("--eig_max", type=float, default=1e5, help="Minimum eigenvalue")
parser.add_argument("--use_stdout", dest="use_stdout", action="store_true")
parser.set_defaults(use_stdout=False)
args = parser.parse_args()

use_tr = False
if args.algorithm != "ip":
    use_tr = True

# Set the eigenvalues for the matrix
n = args.n
eig_min = args.eig_min
eig_max = args.eig_max
use_stdout = args.use_stdout

print("n = ", n)
print("eig_min = %g" % (eig_min))
print("eig_max = %g" % (eig_max))
print("cond = %g" % (eig_max / eig_min))

# Solve the problem with linear spacing of eigenvalues
eigs_linear = np.linspace(eig_min, eig_max, n)

# Solve the problem with a clustered spacing of the eigenvalues
eigs_clustered = np.zeros(n)
for i in range(1, n + 1):
    u = (1.0 * n) / (n - 1) * (1.0 / (n + 1 - i) - 1.0 / n)
    eigs_clustered[i - 1] = eig_min + (eig_max - eig_min) * u**0.9

# Solve the two problem types
solve_problem(
    eigs_linear, filename="opt_linear_eigs.out", use_stdout=use_stdout, use_tr=use_tr
)
solve_problem(
    eigs_clustered,
    filename="opt_cluster_eigs.out",
    use_stdout=use_stdout,
    use_tr=use_tr,
)

plt.plot(range(1, n + 1), eigs_linear, "-o", linewidth=2, label="linear")
plt.plot(range(1, n + 1), eigs_clustered, "-s", linewidth=2, label="clustered")
plt.xlabel("Index", fontsize=17)
plt.ylabel("Eigenvalue", fontsize=17)
plt.legend()
plt.show()
