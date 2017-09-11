# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
import sys, traceback

# Import ParOpt
from paropt import ParOpt

# Import TACS for analysis
from tacs import TACS, elements, constitutive, functions

# Set DV bounds
min_thickness  = 0.01    # minimum thickness of elements in m
max_thickness  = 1.00    # maximum thickness of elemetns in m
init_thickness = 0.10    # initial value of thickness in m

# Create the rosenbrock function class
class PlateOpt(ParOpt.pyParOptProblem):
    def __init__(self, comm, physics):
        # Set the communicator pointer
        self.comm = comm
        self.nvars = 1
        self.ncon = 1
        self.physics = physics
        
        # The design history file
        self.x_hist = []

        # Control redundant evalution of cost and constraint
        # gradients. Evaluate the gradient only if X has changed from
        # previous X.
        self.currentX = None

        # Space for current function and gradient values
        self.funcVals = np.zeros((self.ncon+1))
        self.gradVals = np.zeros((self.ncon+1)*self.nvars)
        
        # Initialize the base class
        super(PlateOpt, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        '''
        Set the values of the bounds for design variables
        '''
        
         # Set the bounds
        lb[:] = min_thickness
        ub[:] = max_thickness

        # Initial design vector
        x[:] = init_thickness

        return

    def evalObjCon(self, x):
        '''
        Evaluate the objective and constraint
        '''
        # Set the fail flag
        fail = 0
        
        # Append the point to the solution history
        self.x_hist.append(np.array(x[:]))

        # Call the solver
        try:
            self.currentX = x
            self.physics.getFuncGrad(self.nvars, x[:], self.funcVals,
                                     self.gradVals)            
        except:
            traceback.print_exc(file=sys.stdout)
            fail = 1
            
        # Store the objective function value
        fobj = self.funcVals[0]
            
        # Store the constraint values
        con = self.funcVals[1:]

        # Return the values
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        '''
        Evaluate the objective and constraint gradient
        '''
        # Set the fail flag
        fail = 0

        # Evaluate gradients if this is a new design point
        if np.array_equal(self.currentX,x) is False:
            try:
                print "Info: evaluating gradients at new x:", x
                self.currentX = x
                self.physics.getFuncGrad(self.nvars, x[:], self.funcVals,
                                         self.gradVals)            
            except:
                traceback.print_exc(file=sys.stdout)
                fail = 1
            
        # Set the objective gradient
        g[:] = self.gradVals[0:self.nvars]

        # Set the constraint gradient
        for c in xrange(self.ncon):
            A[c][:] = self.gradVals[(c+1)*(self.nvars):(c+2)*(self.nvars)]
        
        return fail

######################################################################
# Create the Plate analysis using TACS
######################################################################

bdfFileName = "plate.bdf" # Specify the name of the file to load which
                          # contains the mesh

#---------------------------------------------------------------------!
# Properties for dynamics (Initial Conditions)
#---------------------------------------------------------------------!

gravity = elements.GibbsVector(0.0, 0.0, -9.81)
v0      = elements.GibbsVector(0.0, 0.0, 0.0)
w0      = elements.GibbsVector(0., 0., 0.)

#---------------------------------------------------------------------!
# Properties for the structure
#---------------------------------------------------------------------!

rho           = 2500.0  # density, kg/m^3
E             = 70.0e9  # elastic modulus, Pa
nu            = 0.30    # poisson's ratio
kcorr         = 5.0/6.0 # shear correction factor
ys            = 350.0e6 # yield stress, Pa

#---------------------------------------------------------------------!
# Load input BDF, set properties and create TACS
#---------------------------------------------------------------------!

mesh = TACS.MeshLoader(MPI.COMM_WORLD)
mesh.scanBDFFile(bdfFileName)

num_components  = mesh.getNumComponents()
for i in xrange(num_components):
    descriptor  = mesh.getElementDescript(i)
    stiff       = constitutive.isoFSDT(rho, E, nu, kcorr, ys,
                                       init_thickness, i,
                                       min_thickness, max_thickness)
    element     = None
    if descriptor in ["CQUAD"]:
        element = elements.MITC(stiff, gravity, v0, w0)        
        mesh.setElement(i, element)

tacs = mesh.createTACS(8)

######################################################################
# Create the Integrator (solver) and configure it with objective and
# functions to evaluate
#######################################################################

tinit             = 0.00
tfinal            = 0.01
num_steps_per_sec = 1000
order             = 3

solver = TACS.DIRKIntegrator(tacs, tinit, tfinal, num_steps_per_sec, order)
solver.setPrintLevel(0)

# Create the objective and constaint functions
funcs = []
funcs.append(functions.StructuralMass(tacs))
funcs.append(functions.InducedFailure(tacs, 1000.0))
solver.setFunction(funcs)

#######################################################################
# Create the optimization problem for Paropt
#######################################################################

problem = PlateOpt(MPI.COMM_WORLD, solver)

# Set up the optimization problem using Paropt
max_lbfgs = 20
opt = ParOpt.pyParOpt(problem, max_lbfgs, ParOpt.BFGS)
opt.resetQuasiNewtonHessian()
opt.setInitBarrierParameter(0.1)
opt.setUseLineSearch(1)
opt.setMaxMajorIterations(100)
opt.optimize()

# Get the final design point
x, z, zw, zu, zl = opt.getOptimizedPoint()

# Print the design variables
for i in range(len(x)):
    print 'x[%2d] %10.5e'%(i, x[i])

# Do any post processing and plot making
