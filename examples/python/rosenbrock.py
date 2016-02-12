import numpy as np
import mpi4py.MPI as MPI

# Import ParOpt
from paropt import ParOpt

# Create the rosenbrock function class
class Rosenbrock(ParOpt.pyParOptProblem):
    def __init__(self):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = 2
        self.ncon = 1

        # Initialize the base class
        super(Rosenbrock, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Set the values of the bounds'''
        x[:] = 0.25
        lb[:] = -2.0
        ub[:] = 2.0
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint'''
        fail = 0
        con = np.zeros(1)
        fobj = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
        con[0] = x[0] + x[1]
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        '''Evaluate the objective and constraint gradient'''
        fail = 0
        
        # The objective gradient
        g[0] = 200*(x[1]-x[0]**2)*(-2*x[0]) - 2*(1-x[0])
        g[1] = 200*(x[1]-x[0]**2)

        # The constraint gradient
        A[0,0] = 1.0
        A[0,1] = 1.0
        return fail

# Create the Rosenbrock problem class
rosen = Rosenbrock()

# Create the optimizer
max_lbfgs = 20
opt = ParOpt.pyParOpt(rosen, max_lbfgs, ParOpt.BFGS)

# Set optimization parameters
opt.setUseHvecProduct(0)
opt.setMajorIterStepCheck(45)
opt.setMaxMajorIterations(1500)

#opt.checkGradients(1e-6)
opt.setGradientCheckFrequency(5, 1e-6)

opt.optimize()
