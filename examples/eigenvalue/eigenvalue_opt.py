import os, sys
import numpy as np
import mpi4py.MPI as MPI
from paropt import ParOpt
from paropt import ParOptEig
import argparse
import matplotlib.pylab as plt


class SpectralAggregate(ParOpt.Problem):
    def __init__(self, n, ndv, rho=10.0):
        """
        This class creates a spectral aggregate for a randomly generated problem.

        The spectral aggregate is formed from a matrix A(x) that takes the form

        A(x) = A0 + sum_{i}^{ndv} x[i]*Q[:,i]*Q[:,i].T

        where x are the design variables, and Q is a randomly generated matrix. The
        spectral aggregate approximates the minimum eigenvalue (a concave function).

        c(x) = -1/rho*log(trace(exp(-rho*A))) =
             = -1/rho*log(sum_{i} (exp(-rho*eigs[i])))
        """
        # Set the communicator
        self.comm = MPI.COMM_WORLD

        # Set the problem dimensions
        self.n = n  # The dimension of the matrix
        self.ndv = ndv  # The number of design variables
        self.rho = rho  # The KS parameter value
        self.ncon = 1  # The number of constraints
        self.itr = 0

        # Generate a random set of vectors
        self.Q1 = np.random.uniform(size=(self.n, self.ndv), low=-1.0, high=1.0)
        self.Q2 = np.random.uniform(size=(self.n, self.ndv), low=-1.0, high=1.0)

        # Pick a B0 such that we know a feasible point exists
        self.x0 = np.ones(self.ndv) / self.ndv

        A = np.dot(self.Q1, np.dot(np.diag(self.x0), self.Q1.T)) + np.dot(
            self.Q2, np.dot(np.diag(self.x0), self.Q2.T)
        )

        fact = 0.1 * np.trace(A) / self.ndv

        # Create a positive definite B0 matrix
        Qb, Rb = np.linalg.qr(np.random.uniform(size=(self.n, self.n)))
        lamb = fact * np.ones(self.n)
        self.B0 = np.dot(Qb, np.dot(np.diag(lamb), Qb.T))

        # Initialize the base class
        super(SpectralAggregate, self).__init__(self.comm, self.ndv, self.ncon)

        return

    def evalModel(self, x):
        """
        Evaluate a quadratic approximation of the model.

        Args:
            x (np.ndarray) The design point

        Returns:
            The lowest eigenvalue, and three coefficients of the quadratic approximation.
        """

        # Compute the matrix A(x)
        A = (
            np.dot(self.Q1, np.dot(np.diag(x), self.Q1.T))
            + np.dot(self.Q2, np.dot(np.diag(x), self.Q2.T))
        ) - self.B0

        # Compute the full eigen decomposition of the matrix A
        self.eigs, self.vecs = np.linalg.eigh(A)

        # Store the diagonal matrix
        self.W = np.zeros((self.ndv, self.n))

        # Compute the number of off-diagonal vectors
        m = self.n * (self.n - 1) >> 1
        self.V = np.zeros((self.ndv, m))

        # Compute the eta values
        min_eig = np.min(self.eigs)
        self.eta = np.exp(-self.rho * (self.eigs - min_eig))
        self.beta = np.sum(self.eta)
        self.eta[:] = self.eta / self.beta

        # Compute the maximum eigenvalue
        ks_value = min_eig - np.log(self.beta) / rho

        # Compute the gradients - fill in the W and V entries
        self.P = np.zeros((m, m))
        index = 0
        for i in range(self.n):
            # Compute the derivative
            self.W[:, i] = (
                np.dot(self.Q1.T, self.vecs[:, i]) ** 2
                + np.dot(self.Q2.T, self.vecs[:, i]) ** 2
            )

            for j in range(i + 1, self.n):
                self.V[:, index] = np.dot(self.Q1.T, self.vecs[:, i]) * np.dot(
                    self.Q1.T, self.vecs[:, j]
                ) + np.dot(self.Q2.T, self.vecs[:, i]) * np.dot(
                    self.Q2.T, self.vecs[:, j]
                )
                self.P[index, index] = 0.0
                if self.eigs[i] != self.eigs[j]:
                    self.P[index, index] = (
                        2.0
                        * (self.eta[i] - self.eta[j])
                        / (self.eigs[i] - self.eigs[j])
                    )
                else:
                    self.P[index, index] = 2.0 * self.rho * self.eta[i]
                index += 1

        self.M = self.rho * (np.outer(self.eta, self.eta) - np.diag(self.eta))

        # Compute the gradient
        ks_gradient = np.dot(self.W, self.eta)

        # Compute the Hessian
        ks_hessian = np.dot(self.W, np.dot(self.M, self.W.T)) + np.dot(
            self.V, np.dot(self.P, self.V.T)
        )

        return min_eig, ks_value, ks_gradient, ks_hessian

    def verify_derivatives(self, x0, dh=1e-6):
        """
        Verify the derivatives of the model are accurate using a finite different approximation.

        Args:
            x0 (np.ndarray) The design point
            dh (float) Finite-difference step size
        """
        pert = np.random.uniform(size=self.ndv)

        # Evaluate the model at x0
        lam0, ks0, grad0, H0 = self.evalModel(x0)

        # Evaluate the model at x0 + dh*pert
        x = x0 + dh * pert
        lam1, ks1, grad1, H1 = self.evalModel(x)

        fd = (ks1 - ks0) / dh
        exact = np.dot(grad0, pert)
        print(
            "FD approx grad: %25.15e  Exact: %25.15e  Rel err: %25.15e"
            % (fd, exact, (fd - exact) / fd)
        )

        fd = (grad1 - grad0) / dh
        exact = np.dot(H0, pert)
        for i, val in enumerate(fd):
            print(
                "FD approx Hess[%2d]: %25.15e  Exact: %25.15e  Rel err: %25.15e"
                % (i, val, exact[i], (val - exact[i]) / val)
            )

        return

    def updateModel(self, x, approx):
        """Update the eigenvalue model"""

        # Get the vectors from the approximation object
        g0, hvecs = approx.getApproximationVectors()

        # Set the components of the gradient
        g0[:] = self.grad[:]

        # Create the M-matrix in the approximation = V*M*V^{T}
        nhv = len(hvecs)
        M = np.zeros((nhv, nhv))

        # Find the number of diagonal entries in M exceeding the tolerance
        # but not more than nhv/2. These will be included
        nmv = 0

        # Include terms that exceed a specified tolerance. In practice,
        # these are rarely included.
        tol = 0.01
        for i in range(nhv // 2):
            if self.M[i, i] >= tol:
                nmv += 1

        # Fill in the values of the approximation matrix from M
        npv = nhv - nmv
        for i in range(nmv):
            hvecs[i][:] = self.W[:, i]
            M[i, :nmv] = self.M[i, :nmv]

        # Calculate the vectors with the largest contributions
        indices = range(npv)

        # Extract the values from the P matrix to fill in the remainder
        # of the matrix approximation
        for i in range(npv):
            hvecs[i + nmv][:] = self.V[:, indices[i]]
            M[i + nmv, i + nmv] = self.P[indices[i], indices[i]]

        # Compute the Moore-Penrose inverse of the matrix
        Minv = np.linalg.pinv(M)

        approx.setApproximationValues(self.ks, M, Minv)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = self.x0[:]
        lb[:] = 0.0
        ub[:] = 1.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints
        fail = 0

        # Evaluate the objective - the approximate compliance
        fobj = 0.5 * np.sum(x[:] ** 2)

        # Evaluate the model using the eigenvalue constraint
        self.lam, self.ks, self.grad, self.H = self.evalModel(x[:])

        # Print out the minimum eigenvalue
        print(
            "[%3d] min(eigs) = %15.6e" % (self.itr, np.min(self.eigs))
            + " ks = %15.6e" % (self.ks)
            + " fobj = %15.6e" % (fobj)
            + " sum(x) = %15.6e" % (np.sum(x[:]))
        )
        self.itr += 1

        con = [self.ks]

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[:] = x[:]

        # The constraint gradient
        A[0][:] = self.grad[:]

        return fail

    def writeOutput(self, it):
        pass


def solve_problem(
    n, ndv, N, rho, filename=None, use_quadratic_approx=True, verify=False
):
    problem = SpectralAggregate(n, ndv, rho=rho)

    if verify:
        x0 = np.random.uniform(size=ndv)
        problem.verify_derivatives(x0)

    options = {
        "algorithm": "tr",
        "tr_init_size": 0.05,
        "tr_min_size": 1e-6,
        "tr_max_size": 10.0,
        "tr_eta": 0.1,
        "tr_output_file": os.path.splitext(filename)[0] + ".tr",
        "tr_penalty_gamma_max": 1e6,
        "tr_adaptive_gamma_update": True,
        "tr_infeas_tol": 1e-6,
        "tr_l1_tol": 1e-6,
        "tr_linfty_tol": 0.0,
        "tr_max_iterations": 200,
        "output_level": 2,
        "penalty_gamma": 10.0,
        "qn_subspace_size": 10,
        "qn_type": "bfgs",
        "abs_res_tol": 1e-10,
        "output_file": filename,
        "starting_point_strategy": "affine_step",
        "barrier_strategy": "monotone",
        "start_affine_multiplier_min": 0.01,
    }

    if use_quadratic_approx:
        options["qn_subspace_size"] = 0
        options["qn_type"] = "none"

    opt = ParOpt.Optimizer(problem, options)

    if use_quadratic_approx:
        # Create the BFGS approximation
        qn = ParOpt.LBFGS(problem, subspace=10)

        # Create the quadratic eigenvalue approximation object
        approx = ParOptEig.CompactEigenApprox(problem, N)

        # Set up the corresponding quadratic approximation, specifying the index
        # of the eigenvalue constraint
        eig_qn = ParOptEig.EigenQuasiNewton(qn, approx, index=0)

        # Set up the eigenvalue optimization subproblem
        subproblem = ParOptEig.EigenSubproblem(problem, eig_qn)
        subproblem.setUpdateEigenModel(problem.updateModel)

        opt.setTrustRegionSubproblem(subproblem)

    opt.optimize()

    # Get the optimized point from the trust-region subproblem
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    print("max(x) = %15.6e" % (np.max(x[:])))
    print("avg(x) = %15.6e" % (np.average(x[:])))
    print("sum(x) = %15.6e" % (np.sum(x[:])))

    if verify:
        for h in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
            problem.checkGradients(h)
            problem.verify_derivatives(x[:], h)

    return x


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", type=int, default=100, help="Dimension of the problem matrix"
)
parser.add_argument("--ndv", type=int, default=200, help="Number of design variables")
parser.add_argument(
    "--N",
    type=int,
    default=10,
    help="Number of terms in the eigenvalue Hessian approx.",
)
parser.add_argument("--rho", type=float, default=10.0, help="KS aggregation parameter")
parser.add_argument(
    "--linearized",
    default=False,
    action="store_true",
    help="Use a linearized approximation",
)
parser.add_argument(
    "--verify",
    default=False,
    action="store_true",
    help="Perform a finite-difference verification",
)
args = parser.parse_args()

# Set the eigenvalues for the matrix
n = args.n
ndv = args.ndv
rho = args.rho
N = args.N

use_quadratic_approx = True
if args.linearized:
    use_quadratic_approx = False

# Set a consistent random seed to produce the same results.
np.random.seed(0)

# Solve the problem
x = solve_problem(
    n,
    ndv,
    N,
    rho,
    filename="output.out",
    use_quadratic_approx=use_quadratic_approx,
    verify=args.verify,
)
