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
class ConvexProblem(ParOpt.pyParOptProblem):
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

        self.A = np.zeros(Affine.shape)

        # Initialize the base class
        super(ConvexProblem, self).__init__(self.comm, self.nvars, self.ncon)

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

        # Compute the matrix A = Affine + Q*diag(x)*Q^{T}
        self.A = self.Affine + np.dot(self.Q, np.dot(np.diag(x)**3, self.Q.T))

        # Solve u = A^{-1}*b
        self.u = np.linalg.solve(self.A, self.b)
        fobj = np.dot(self.u, self.b)
        
        # Compute the constraint
        con[0] = self.bcon - np.dot(x, self.Acon)
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        '''Evaluate the objective and constraint gradient'''
        fail = 0
        
        # The objective gradient
        g[:] = -3*x**2*(np.dot(self.Q.T, self.u))**2
        
        # The constraint gradient
        A[0,:] = -self.Acon[:]

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

def create_random_data(n):
    '''
    Create a random orthogonal basis with the given eigenvalues
    '''

    # Create a random square (n x n) matrix
    B = np.random.uniform(size=(n, n))

    # Orthogonalize the columns of B so that Q = range(B) = R^{n}
    Q, s, v = np.linalg.svd(B)

    return Q

def solve_problem(n, filename=None):
    # Create the objective data
    print 'Generating random data ...'
    Q = create_random_data(n)
    print 'Generating random SPD matrix ...'
    Affine = create_random_spd(n)
    b = np.random.uniform(size=n)
    
    # Create the constraint data
    Acon = np.random.uniform(size=n)
    bcon = 0.5*n*np.random.uniform()

    # Set up/allocate the problem
    problem = ConvexProblem(Q, Affine, b,
                            Acon, bcon)

    print 'Solving the optimization problem ...'

    # Set up the optimization problem
    max_lbfgs = 20
    opt = ParOpt.pyParOpt(problem, max_lbfgs, ParOpt.BFGS)
    if filename is not None:
        opt.setOutputFile(filename)

    # Set optimization parameters
    opt.checkGradients(1e-6)
    opt.setArmijioParam(1e-5)
    opt.setMaxMajorIterations(1000)
    opt.setBarrierPower(1.5)
    opt.setBarrierFraction(0.1)
    opt.optimize()

    x = opt.getOptimizedPoint()
    
    return x

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100,
                    help='Dimension of the problem')
args = parser.parse_args()

# Set the eigenvalues for the matrix
n = args.n

print 'n = ', n

# Solve the two problem types
x = solve_problem(n, filename='opt_convex.out')

print 'Discrete infeasibility: ', np.sqrt(np.dot(x*(1.0 - x), x*(1.0 - x)))
