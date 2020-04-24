from __future__ import print_function

# Import numpy
import numpy as np
import mpi4py.MPI as MPI

# Import ParOpt
from paropt import ParOpt

# Import argparse
import argparse

# Import matplotlib
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

        # Set the inequality options for this problem
        self.setInequalityOptions(dense_ineq=True, sparse_ineq=True,
                                  use_lower=True, use_upper=True)
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Set the values of the bounds'''
        x[:] = 0.05 + 0.9*np.random.uniform(size=len(x))
        lb[:] = 0.0
        ub[:] = 1.0
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint'''
        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1)

        # Compute the artificial stiffness matrix
        self.K = self.Affine + np.dot(self.Q, np.dot(np.diag(x), self.Q.T))

        # Compute the displacements
        self.u = np.linalg.solve(self.K, self.b)

        # Compute the artifical compliance
        fobj = np.dot(self.u, self.b)

        if self.obj_scale < 0.0:
            self.obj_scale = 1.0/fobj

        # Compute the linear constraint
        con[0] = self.bcon - np.dot(self.Acon, x)

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        '''Evaluate the objective and constraint gradient'''
        fail = 0

        # The objective gradient
        g[:] = -np.dot(self.Q.T, self.u)**2

        # The constraint gradient
        A[0][:] = -self.Acon[:]

        return fail

def create_random_spd(n):
    '''
    Create a random positive definite matrix with the given
    eigenvalues
    '''
    # Create the eigenvalues for the matrix
    eigs = np.random.uniform(size=n)

    # Create a random square (n x n) matrix
    B = np.random.uniform(size=(n, n))

    # Orthogonalize the columns of B so that Q = range(B) = R^{n}
    Q, s, v = np.linalg.svd(B)

    # Compute A = Q*diag(eigs)*Q^{T}
    A = np.dot(Q, np.dot(np.diag(eigs), Q.T))

    return A

def solve_problem(eigs, filename=None, data_type='orthogonal',
                use_tr=False):
    # Create a random orthogonal Q vector
    if data_type == 'orthogonal':
        B = np.random.uniform(size=(n, n))
        Q, s, v = np.linalg.svd(B)

        # Create a random Affine matrix
        Affine = create_random_spd(eigs)
    else:
        Q = np.random.uniform(size=(n, n))
        Affine = np.diag(1e-3*np.ones(n))

    # Create the random right-hand-side
    b = np.random.uniform(size=n)

    # Create the constraint data
    Acon = np.random.uniform(size=n)
    bcon = 0.25*np.sum(Acon)

    # Create the convex problem
    problem = ConvexProblem(Q, Affine, b, Acon, bcon)

    if use_tr:
        # Create the trust region problem
        max_lbfgs = 10
        tr_init_size = 0.05
        tr_min_size = 1e-6
        tr_max_size = 10.0
        tr_eta = 0.1
        tr_penalty_gamma = 10.0

        qn = ParOpt.LBFGS(problem, subspace=max_lbfgs)
        subproblem = ParOpt.QuadraticSubproblem(problem, qn)
        tr = ParOpt.TrustRegion(subproblem, tr_init_size,
                                tr_min_size, tr_max_size,
                                tr_eta, tr_penalty_gamma)
        tr.setMaxTrustRegionIterations(500)

        infeas_tol = 1e-6
        l1_tol = 1e-5
        linfty_tol = 1e-5
        tr.setTrustRegionTolerances(infeas_tol, l1_tol, linfty_tol)

        # Set up the optimization problem
        tr_opt = ParOpt.InteriorPoint(subproblem, 10, ParOpt.BFGS)
        if filename is not None:
            tr_opt.setOutputFile(filename)

        # Set the tolerances
        tr_opt.setAbsOptimalityTol(1e-8)
        tr_opt.setStartingPointStrategy(ParOpt.AFFINE_STEP)
        tr_opt.setStartAffineStepMultiplierMin(0.01)

        # Set optimization parameters
        tr_opt.setArmijoParam(1e-5)
        tr_opt.setMaxMajorIterations(5000)
        tr_opt.setBarrierPower(2.0)
        tr_opt.setBarrierFraction(0.1)

        # optimize
        tr.setOutputFile(filename + '_tr')
        tr.setPrintLevel(1)
        tr.optimize(tr_opt)

        # Get the optimized point from the trust-region subproblem
        x, z, zw, zl, zu = tr_opt.getOptimizedPoint()
    else:
        # Set up the optimization problem
        max_lbfgs = 50
        opt = ParOpt.InteriorPoint(problem, max_lbfgs, ParOpt.BFGS)
        if filename is not None:
            opt.setOutputFile(filename)

        # Set optimization parameters
        opt.setArmijoParam(1e-5)
        opt.setMaxMajorIterations(5000)
        opt.setBarrierPower(2.0)
        opt.setBarrierFraction(0.1)
        opt.optimize()

        # Get the optimized point
        x, z, zw, zl, zu = opt.getOptimizedPoint()

    return x

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100,
                    help='Dimension of the problem')
parser.add_argument('--optimizer', type=str, default='ip')
args = parser.parse_args()

use_tr = False
if args.optimizer != 'ip':
    use_tr = True

# Set the eigenvalues for the matrix
n = args.n
print('n = ', n)

# Solve the problem
x = solve_problem(n, filename='opt_convex.out', use_tr=use_tr)
