# Import numpy
import os
import numpy as np
import mpi4py.MPI as MPI

# Import ParOpt
from paropt import ParOpt
import ParOptEig

# Import argparse
import argparse

# Import matplotlib
import matplotlib.pylab as plt

class SpectralAggregate(ParOpt.Problem):
    def __init__(self, n, ndv, rho=10.0, approx=None):
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
        self.n = n # The dimension of the matrix
        self.ndv = ndv # The number of design variables
        self.rho = rho # The KS parameter value
        self.ncon = 1
        self.approx = approx

        # Generate a random
        self.Q = np.random.uniform(size=(self.n, self.ndv))
        self.A0 = 1e-6*np.dot(self.Q, self.Q.T)

        # Set up a random vector
        self.f = np.random.uniform(size=self.n, low=-1.0, high=1.0)

        # Initialize the base class
        super(SpectralAggregate, self).__init__(self.comm, self.ndv, self.ncon)

        # Set the inequality options for this problem
        self.setInequalityOptions(dense_ineq=True, sparse_ineq=True,
                                  use_lower=True, use_upper=False)

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
        A = self.A0 + np.dot(self.Q, np.dot(np.diag(x), self.Q.T))

        # Compute the full eigen decomposition of the matrix A
        self.eigs, self.vecs = np.linalg.eigh(A)

        # Store the diagonal matrix
        self.W = np.zeros((self.ndv, self.n))

        # Compute the number of off-diagonal vectors
        m = self.n*(self.n-1) >> 1
        self.V = np.zeros((self.ndv, m))

        # Compute the eta values
        min_eig = np.min(self.eigs)
        self.eta = np.exp(-self.rho*(self.eigs - min_eig))
        self.beta = np.sum(self.eta)
        self.eta[:] = self.eta/self.beta

        # Compute the maximum eigenvalue
        ks_value = min_eig - np.log(self.beta)/rho

        # Compute the gradients - fill in the W and V entries
        self.P = np.zeros((m, m))
        index = 0
        for i in range(self.n):
            # Compute the inner product of
            self.W[:,i] = np.dot(self.Q.T, self.vecs[:,i])**2

            for j in range(i+1, self.n):
                self.V[:,index] = np.dot(self.Q.T, self.vecs[:,i])*np.dot(self.Q.T, self.vecs[:,j])
                self.P[index, index] = 0.0
                if self.eigs[i] != self.eigs[j]:
                    self.P[index, index] = 2.0*(self.eta[i] - self.eta[j])/(self.eigs[i] - self.eigs[j])
                else:
                    self.P[index, index] = 2.0*self.rho*self.eta[i]
                index += 1

        self.M = self.rho*(np.outer(self.eta, self.eta) - np.diag(self.eta))

        # Compute the gradient
        ks_gradient = np.dot(self.W, self.eta)

        # Compute the Hessian
        ks_hessian = np.dot(self.W, np.dot(self.M, self.W.T)) + np.dot(self.V, np.dot(self.P, self.V.T))

        if self.approx is not None:
            g0, hvecs = self.approx.getApproximationVectors()

            g0[:] = ks_gradient[:]

            nhv = len(hvecs)
            M = np.zeros((nhv, nhv))

            # Find the number of diagonal entries in M exceeding the tolerance
            # but not more than nhv/2. These will be included
            nmv = 0
            tol = 1e-3
            for i in range(nhv//2):
                if self.M[i,i] >= tol:
                    nmv += 1

            # Fill in the values of the approximation matrix from M
            npv = nhv - nmv
            for i in range(nmv):
                hvecs[i][:] = self.W[:,i]
                M[i,:nmv] = self.M[i,:nmv]

            diag = range(m)
            indices = np.argsort(self.P[diag, diag])[:npv]

            # Extract the values from the P matrix to fill in the remainder
            # of the matrix approximation
            for i in range(npv):
                hvecs[i+nmv][:] = self.V[:,indices[i]]
                M[i+nmv,i+nmv] = self.P[indices[i], indices[i]]

            Minv = np.linalg.pinv(M)

            self.approx.setApproximationValues(ks_value, M, Minv)

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
        x = x0 + dh*pert
        lam1, ks1, grad1, H1 = self.evalModel(x)

        fd = (ks1 - ks0)/dh
        exact = np.dot(grad0, pert)
        print('FD approx grad: %25.15e  Exact: %25.15e  Rel err: %25.15e'%(fd, exact, (fd - exact)/fd))

        fd = (grad1 - grad0)/dh
        exact = np.dot(H0, pert)
        for i, val in enumerate(fd):
            print('FD approx Hess[%2d]: %25.15e  Exact: %25.15e  Rel err: %25.15e'%(
                i, val, exact[i], (val - exact[i])/val))

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = 1.0
        lb[:] = 0.0
        ub[:] = 5e20
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints
        fail = 0

        # Evaluate the objective - the approximate compliance
        A = self.A0 + np.dot(self.Q, np.dot(np.diag(x[:]), self.Q.T))
        self.u = np.linalg.solve(A, self.f)
        fobj = np.dot(self.u, self.f)

        # Evaluate the model using the eigenvalue constraint
        self.lam, self.ks, self.grad, self.H = self.evalModel(x[:])

        con = [self.ks]

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[:] = - np.dot(self.u, self.Q)**2

        # The constraint gradient
        A[0][:] = self.grad[:]

        return fail

    def writeOutput(self, it):
        pass

def solve_problem(n, ndv, rho, filename=None,
                  use_quadratic_approx=True, verify=False):
    problem = SpectralAggregate(n, ndv, rho=rho, approx=None)

    if verify:
        x0 = np.random.uniform(size=ndv)
        problem.verify_derivatives(x0)

    # Create the trust region problem
    max_lbfgs = 10
    tr_init_size = 0.05
    tr_min_size = 1e-6
    tr_max_size = 10.0
    tr_eta = 0.1
    tr_penalty_gamma = 10.0

    qn = ParOpt.LBFGS(problem, subspace=max_lbfgs)
    if use_quadratic_approx:
        # Number of approximation vectors
        napprox = 10
        approx = ParOptEig.CompactEigenApprox(problem, napprox)
        problem.approx = approx

        eig_qn = ParOptEig.EigenQuasiNewton(qn, approx)
        subproblem = ParOptEig.EigenSubproblem(problem, eig_qn)
    else:
        subproblem = ParOpt.QuadraticSubproblem(problem, qn)

    tr = ParOpt.TrustRegion(subproblem, tr_init_size,
                            tr_min_size, tr_max_size,
                            tr_eta, tr_penalty_gamma)
    tr.setMaxTrustRegionIterations(500)

    infeas_tol = 1e-6
    l1_tol = 1e-4
    linfty_tol = 1e-4
    tr.setTrustRegionTolerances(infeas_tol, l1_tol, linfty_tol)

    # Set up the optimization problem
    tr_opt = ParOpt.InteriorPoint(subproblem, max_lbfgs, ParOpt.BFGS)
    if filename is not None:
        tr_opt.setOutputFile(filename)

    # Set the tolerances
    tr_opt.setAbsOptimalityTol(1e-8)
    tr_opt.setStartingPointStrategy(ParOpt.AFFINE_STEP)
    tr_opt.setStartAffineStepMultiplierMin(0.01)
    tr_opt.setBarrierStrategy(ParOpt.MONOTONE)

    # Set optimization parameters
    tr_opt.setArmijoParam(1e-5)
    tr_opt.setMaxMajorIterations(5000)
    tr_opt.setBarrierPower(2.0)
    tr_opt.setBarrierFraction(0.1)

    # optimize
    tr.setOutputFile(os.path.splitext(filename)[0] + '.tr')
    # tr.setPrintLevel(1)
    tr.optimize(tr_opt)

    # Get the optimized point from the trust-region subproblem
    x, z, zw, zl, zu = tr_opt.getOptimizedPoint()

    return x

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100,
                    help='Dimension of the proble matrix')
parser.add_argument('--ndv', type=int, default=200,
                    help='Number of design variables')
parser.add_argument('--rho', type=float, default=10.0,
                    help='KS aggregation parameter')
parser.add_argument('--linearized', default=False, action='store_true',
                    help='Use a linearized approximation')
args = parser.parse_args()

# Set the eigenvalues for the matrix
n = args.n
ndv = args.ndv
rho = args.rho

use_quadratic_approx = True
if args.linearized:
    use_quadratic_approx = False

# Solve the problem
x = solve_problem(n, ndv, rho, filename='eig_problem.out',
                  use_quadratic_approx=use_quadratic_approx)
