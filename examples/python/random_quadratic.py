# Import numpy
import numpy as np
import mpi4py.MPI as MPI

# Import ParOpt
from paropt import ParOpt

# Create the rosenbrock function class
class Quadratic(ParOpt.pyParOptProblem):
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

        # The design history file
        self.x_hist = []

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
        self.x_hist.append(np.array(x))

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
        A[0,:] = self.Acon[:]

        return fail

def create_random_problem(eigs):
    '''
    Create a random positive definite matrix with the given
    eigenvalues
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

def solve_problem(eigs):
    # Get the A matrix
    A = create_random_problem(eigs)

    # Create the other problem data
    b = np.random.uniform(size=len(eigs))
    Acon = np.random.uniform(size=len(eigs))
    bcon = np.random.uniform()

    problem = Quadratic(A, b, Acon, bcon)

    # Set up the optimization problem
    max_lbfgs = 20
    opt = ParOpt.pyParOpt(problem, max_lbfgs, ParOpt.BFGS)
    opt.optimize()

    return

# Set the eigenvalues for the matrix
n = 150
eig_min = 1.0
eig_max = 1e5

# Solve the problem with linear spacing of eigenvalues
eigs_linear = np.linspace(eig_min, eig_max, n)

# Solve the problem with a clustered spacing of the eigenvalues
eigs_clustered = np.zeros(n)
for i in xrange(1,n+1):
    u = (1.0*n)/(n-1)*(1.0/(n + 1 - i) - 1.0/n)
    eigs_clustered[i-1] = eig_min + (eig_max - eig_min)*u

# Solve the two problem types
print 'Linear spectrum of eigenvalue\n\n'
solve_problem(eigs_linear)

print '\n\nClustered eigenvalue spectrum\n\n'
solve_problem(eigs_clustered)
