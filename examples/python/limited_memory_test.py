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

# Random quadratic problem class
class Quadratic(ParOpt.Problem):
    def __init__(self, eigs):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = len(eigs)
        self.ncon = 1

        # Record the quadratic terms
        self.A = self.createRandomProblem(eigs)
        self.b = np.random.uniform(self.nvars)
        self.Acon = np.ones(self.nvars)
        self.bcon = 0.0

        # Initialize the base class
        super(Quadratic, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Set the values of the bounds'''
        x[:] = -2.0 + np.random.uniform(size=len(x))
        lb[:] = -5.0
        ub[:] = 5.0
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint'''
        # Append the point to the solution history

        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1)

        fobj = 0.5*np.dot(x, np.dot(self.A, x)) + np.dot(self.b, x)
        con[0] = np.dot(x, self.Acon) + self.bcon
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        '''Evaluate the objective and constraint gradient'''
        fail = 0

        # The objective gradient
        g[:] = np.dot(self.A, x) + self.b

        # The constraint gradient
        A[0][:] = self.Acon[:]

        return fail

    def createRandomProblem(self, eigs):
        '''
        Create a random matrix with the given eigenvalues
        '''

        # The dimension of the matrix
        n = len(eigs)

        # Create a random square (n x n) matrix
        B = np.random.uniform(size=(n, n))

        # Orthogonalize the columns of B so that Q = range(B) = R^{n}
        Q, s, v = np.linalg.svd(B)

        # Compute A = Q*diag(eigs)*Q^{T}
        A = np.dot(Q, np.dot(np.diag(eigs), Q.T))

        return A

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--qn_type', type=str, default='sr1')
args = parser.parse_args()
qn_type = args.qn_type

# This test compares a limited-memory Hessian with their dense counterparts
n = 50
eigs = np.linspace(1, 1+n, n)
problem = Quadratic(eigs)

if qn_type == 'sr1':
    # Create the LSR1 object
    qn = ParOpt.LSR1(problem, subspace=n)
else:
    # Create the limited-memory BFGS object
    qn = ParOpt.LBFGS(problem, subspace=n)

# Create a random set of steps and their corresponding vectors
S = np.random.uniform(size=(n, n))
Y = np.dot(problem.A, S)

# Create paropt vectors
ps = problem.createDesignVec()
py = problem.createDesignVec()

# Compute the update to the
y0 = Y[:,-1]
s0 = S[:,-1]
if qn_type == 'sr1':
    B = np.eye(n)
else:
    B = (np.dot(s0, y0)/np.dot(s0, s0))*np.eye(n)

for i in range(n):
    s = S[:,i]
    y = Y[:,i]

    # Update the dense variant
    if qn_type == 'sr1':
        r = y - np.dot(B, s)
        B += np.outer(r, r)/np.dot(r, s)
    else:
        r = np.dot(B, s)
        rho = 1.0/np.dot(y, s)
        beta = 1.0/np.dot(s, r)
        B += - beta*np.outer(r, r) + rho*np.outer(y, y)

    # Update the paropt problem
    ps[:] = s[:]
    py[:] = y[:]
    qn.update(ps, py)

# Now, check that the update works
for i in range(n):
    s = np.random.uniform(size=n)
    ps[:] = s[:]
    qn.mult(ps, py)

    # Compute the residual
    r = py[:] - np.dot(B, s)

    # Compute the relative error
    print('relative err = ', np.sqrt(np.dot(r, r)/np.dot(s, np.dot(B, s))))
