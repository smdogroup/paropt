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
        self.n = n # The dimension of the matrix
        self.ndv = ndv # The number of design variables
        self.rho = rho # The KS parameter value
        self.ncon = 1

        # Create a random objective array
        self.obj_array = np.random.uniform(size=self.ndv, low=1.0, high=10.0)

        # Generate a random set of vectors
        self.Q = np.random.uniform(size=(self.n, self.ndv))

        # Create a positive definite B0 matrix
        Qb, Rb = np.linalg.qr(np.random.uniform(size=(self.n, self.n)))
        lamb = np.linspace(1, 5, self.n)**2
        self.B0 = np.dot(Qb, np.dot(np.diag(lamb), Qb.T))

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
        A = self.B0 - np.dot(self.Q, np.dot(np.diag(x), self.Q.T))

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
            # Compute the derivative
            self.W[:,i] = -np.dot(self.Q.T, self.vecs[:,i])**2

            for j in range(i+1, self.n):
                self.V[:,index] = -np.dot(self.Q.T, self.vecs[:,i])*np.dot(self.Q.T, self.vecs[:,j])
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
        for i in range(nhv//2):
            if self.M[i,i] >= tol:
                nmv += 1

        # Fill in the values of the approximation matrix from M
        npv = nhv - nmv
        for i in range(nmv):
            hvecs[i][:] = self.W[:,i]
            M[i,:nmv] = self.M[i,:nmv]

        # Calculate the vectors with the largest contributions
        indices = range(npv)

        # Extract the values from the P matrix to fill in the remainder
        # of the matrix approximation
        for i in range(npv):
            hvecs[i+nmv][:] = self.V[:,indices[i]]
            M[i+nmv,i+nmv] = self.P[indices[i], indices[i]]

        # Compute the Moore-Penrose inverse of the matrix
        Minv = np.linalg.pinv(M)

        approx.setApproximationValues(self.ks, M, Minv)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = 1.0
        lb[:] = 0.0
        ub[:] = 10.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints
        fail = 0

        # Evaluate the objective - the approximate compliance
        fobj = np.sum(self.obj_array/(1.0 + x[:]))

        # Evaluate the model using the eigenvalue constraint
        self.lam, self.ks, self.grad, self.H = self.evalModel(x[:])

        # Print out the minimum eigenvalue
        print('min(eigs) = %15.6e'%(np.min(self.eigs)) + ' fobj = %15.6e'%(fobj))

        con = [self.ks]

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[:] = -self.obj_array/(1.0 + x[:])**2

        # The constraint gradient
        A[0][:] = self.grad[:]

        return fail

    def writeOutput(self, it):
        pass

def solve_problem(n, ndv, N, rho, filename=None,
                  use_quadratic_approx=True, verify=False):
    problem = SpectralAggregate(n, ndv, rho=rho)

    if verify:
        x0 = np.random.uniform(size=ndv)
        problem.verify_derivatives(x0)

    # Create the trust region problem
    max_lbfgs = 0
    tr_init_size = 0.05
    tr_min_size = 1e-6
    tr_max_size = 10.0
    tr_eta = 0.1
    tr_penalty_gamma = 10.0

    qn = ParOpt.LBFGS(problem, subspace=max_lbfgs)
    if use_quadratic_approx:
        # Create the quadratic eigenvalue approximation object
        approx = ParOptEig.CompactEigenApprox(problem, N)

        # Set up the corresponding quadratic approximation, specifying the index
        # of the eigenvalue constraint
        eig_qn = ParOptEig.EigenQuasiNewton(qn, approx, index=0)

        # Set up the eigenvalue optimization subproblem
        subproblem = ParOptEig.EigenSubproblem(problem, eig_qn, index=0)
        subproblem.setUpdateEigenModel(problem.updateModel)
    else:
        subproblem = ParOpt.QuadraticSubproblem(problem, qn)

    tr = ParOpt.TrustRegion(subproblem, tr_init_size,
                            tr_min_size, tr_max_size,
                            tr_eta, tr_penalty_gamma)
    tr.setMaxTrustRegionIterations(25)

    infeas_tol = 1e-6
    l1_tol = 5e-4
    linfty_tol = 5e-4
    tr.setTrustRegionTolerances(infeas_tol, l1_tol, linfty_tol)

    # Set up the optimization problem
    opt = ParOpt.InteriorPoint(subproblem, max_lbfgs, ParOpt.BFGS)
    if filename is not None:
        opt.setOutputFile(filename)
        tr.setOutputFile(os.path.splitext(filename)[0] + '.tr')

    # Set the tolerances
    opt.setAbsOptimalityTol(1e-7)
    opt.setStartingPointStrategy(ParOpt.AFFINE_STEP)
    opt.setStartAffineStepMultiplierMin(0.01)
    opt.setBarrierStrategy(ParOpt.MONOTONE)
    opt.setOutputLevel(2)

    # Set optimization parameters
    opt.setArmijoParam(1e-5)
    opt.setMaxMajorIterations(5000)
    opt.setBarrierPower(2.0)
    opt.setBarrierFraction(0.1)

    # optimize
    tr.setAdaptiveGammaUpdate(1)
    tr.setPrintLevel(1)
    tr.optimize(opt)

    # Get the optimized point from the trust-region subproblem
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    print('max(x) = %15.6e'%(np.max(x[:])))
    print('avg(x) = %15.6e'%(np.average(x[:])))

    if verify:
        for h in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
            problem.verify_derivatives(x[:], h)

    return x

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100,
                    help='Dimension of the proble matrix')
parser.add_argument('--ndv', type=int, default=200,
                    help='Number of design variables')
parser.add_argument('--N', type=int, default=10,
                    help='Number of terms in the eigenvalue Hessian approx.')
parser.add_argument('--rho', type=float, default=10.0,
                    help='KS aggregation parameter')
parser.add_argument('--linearized', default=False, action='store_true',
                    help='Use a linearized approximation')
parser.add_argument('--verify', default=False, action='store_true',
                    help='Perform a finite-difference verification')
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
x = solve_problem(n, ndv, N, rho, filename='output.out',
                  use_quadratic_approx=use_quadratic_approx,
                  verify=args.verify)
