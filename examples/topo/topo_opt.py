
# Import numpy
import numpy as np

# Import MPI
from mpi4py import MPI

# Import TACS and assorted repositories
from tacs import TACS, elements, constitutive, functions

# Import ParOpt
from paropt import ParOpt

class MatProps:
    '''Multimaterial properties'''
    def __init__(self, rho, E, nu, q, eps):
        self.eps = eps
        self.q = q

        # Set the raw material properties
        self.rho = np.array(rho)
        self.E = np.array(E)
        self.nu = nu

        # Compute the derived properties
        self.D = self.E/(1.0 - self.nu**2)
        self.G = 0.5*self.E/(1.0 + self.nu)

        # Set the number of materials
        self.nmats = len(self.rho)
        
        return
    
class PS(constitutive.pyPlaneStress):
    '''
    Constitutive class used to perform plane-stress analysis within
    TACS. Note call-backs are made through python.
    '''
    def __init__(self, props, num):
        '''
        Set the values of the design 
        '''
        self.props = props
        self.num = num

        # Set the linearization point
        self.x = 0.5*np.ones(props.nmats)
        self.x0 = 0.5*np.ones(props.nmats)
        self.xcon = np.zeros(props.nmats)
        self.xlin = np.ones(props.nmats)
        
        return

    def setLinearization(self, x):
        '''Set the linearization point'''
        props = self.props
        self.x0[:] = x[props.nmats*self.num:props.nmats*(self.nmat+1)]
        self.xcon[:] = self.x0/(1.0 + props.q*(1.0 - self.x0))
        self.xlin[:] = (props.q+1.0)/(1.0 + props.q*(1.0 - self.x0))**2
        return
    
    def setDesignVars(self, x):
        '''Set the design variable values'''
        nmats = self.props.nmats
        self.x[:] = x[nmats*self.num:nmats*(self.num+1)]
        return

    def getDesignVars(self, x):
        '''Get the design variable values'''
        nmats = self.props.nmats
        x[nmats*self.num:nmats*(self.nmats+1)] = self.x        
        return

    def getDesignVarRange(self, lb, ub):
        '''Get the range of design variable values'''
        props = self.props
        nmats = self.props.nmats
        lb[nmats*self.num:nmats*(self.nmats+1)] = props.q*self.x0**2/(1.0 + props.q)
        ub[nmats*self.num:nmats*(self.nmats+1)] = 1e30
        return

    def calculateStress(self, pt, e):
        '''Compute the stress at the point'''

        # Compute the stresses
        s = np.zeros(e.shape)
        D = np.dot(self.props.D, self.xcon + self.xlin*(self.x - self.x0)) + self.props.eps
        G = np.dot(self.props.G, self.xcon + self.xlin*(self.x - self.x0)) + self.props.eps
        
        s[0] = D*(e[0] + self.props.nu*e[1])
        s[1] = D*(e[1] + self.props.nu*e[0])
        s[2] = G*e[2]
        return s

    def addStressDVSens(self, pt, e, alpha, psi, fdvs):
        '''Add the derivative of stress w.r.t.'''

        for i in xrange(self.props.nmats):
            D = self.props.D[i]
            G = self.props.G[i]
            nu = self.props.nu
            result = self.xlin[i]*(psi[0]*(D*(e[0] + nu*e[1])) +
                                   psi[1]*(D*(e[1] + nu*e[0])) +
                                   psi[2]*G*e[2])            
            fdvs[self.props.nmats*self.num+i] += alpha*result

        return

    def getPointwiseMass(self, pt):
        '''Return the pointwise mass'''
        return np.dot(self.props.rho, self.x)

    def addPointwiseMassDVSens(p, ascale, fdvs):
        '''Add the derivative of the mass'''
        props = self.props
        fdvs[props.nmats*self.num:props.nmats*(self.num+1)] += ascale*props.rho
        return

    def failure(self, pt, e):
        '''Evaluate the stress'''

        # Compute the stresses
        s = np.zeros(e.shape)
        s[0] = self.D*(e[0] + self.nu*e[1])
        s[1] = self.D*(e[1] + self.nu*e[0])
        s[2] = self.G*e[2]

        # Compute the valure of the failure function
        fval = (s[0]**2 + s[1]**2 - s[0]*s[1] + 3*s[2]**2)/self.ys**2 

        return fval

# Set the number of elements in the x/y directions
nx = 8
ny = 8

# Set the material properties
E = [70e3, 95e3, 120e3]
rho = [1.0, 1.3, 1.9]
nu = 0.3

# Create the material properties object
q = 1.0
eps = 1.0
mat_props = MatProps(rho, E, nu, q, eps)

# Set the number of design variables
num_design_vars = 3*nx*ny

# Create the TACS creator object
comm = MPI.COMM_WORLD
creator = TACS.Creator(comm, 2)

# Set up the mesh on the root processor
if comm.rank == 0:
    # Set the nodes
    nnodes = (2*nx+1)*(2*ny+1)
    nelems = nx*ny
    nodes = np.arange(nnodes).reshape((2*nx+1, 2*ny+1))

    # Set the connectivity and create the corresponding elements
    conn = np.zeros((nelems, 9), dtype=np.intc)
    for j in xrange(ny):
        for i in xrange(nx):
            # Append the first set of nodes
            n = i + nx*j
            conn[n,0] = nodes[2*i, 2*j]
            conn[n,1] = nodes[2*i+1, 2*j]
            conn[n,2] = nodes[2*i+2, 2*j]
            conn[n,3] = nodes[2*i, 2*j+1]
            conn[n,4] = nodes[2*i+1, 2*j+1]
            conn[n,5] = nodes[2*i+2, 2*j+1]
            conn[n,6] = nodes[2*i, 2*j+2]
            conn[n,7] = nodes[2*i+1, 2*j+2]
            conn[n,8] = nodes[2*i+2, 2*j+2]            

    # Set the node pointers
    conn = conn.flatten()
    ptr = np.arange(0, 9*nelems+1, 9, dtype=np.intc)
    elem_ids = np.arange(nelems, dtype=np.intc)
    creator.setGlobalConnectivity(nnodes, ptr, conn, elem_ids)

    # Set up the boundary conditions
    bcnodes = np.array(nodes[0,:], dtype=np.intc)

    # Set the boundary condition variables
    nbcs = 2*bcnodes.shape[0]
    bcvars = np.zeros(nbcs, dtype=np.intc)
    bcvars[:nbcs:2] = 0
    bcvars[1:nbcs:2] = 1

    # Set the boundary condition pointers
    bcptr = np.arange(0, nbcs+1, 2, dtype=np.intc)
    creator.setBoundaryConditions(bcnodes, bcvars, bcptr)

    # Set the node locations
    Xpts = np.zeros(3*nnodes)
    x = np.linspace(0, 10, 2*nx+1)
    y = np.linspace(0, 10, 2*nx+1)
    for j in xrange(2*ny+1):
        for i in xrange(2*nx+1):
            Xpts[3*nodes[i,j]] = x[i]
            Xpts[3*nodes[i,j]+1] = y[j]
            
    # Set the node locations
    creator.setNodes(Xpts)  
    
# Create the elements
elems = [] 
for j in xrange(ny):
    for i in xrange(nx):
        n = i + nx*j
        ps = PS(mat_props, n)
        elems.append(elements.PlaneQuad(3, ps))

# Set the elements
creator.setElements(elems)

# Create the tacs assembler object
tacs = creator.createTACS()



# Set/get the design variable values
xvals = np.zeros(num_design_vars)
tacs.getDesignVars(xvals)
tacs.setDesignVars(xvals)

res = tacs.createVec()
ans = tacs.createVec()
mat = tacs.createFEMat()

# Create the preconditioner for the corresponding matrix
pc = TACS.Pc(mat)

# Assemble the Jacobian
alpha = 1.0
beta = 0.0
gamma = 0.0
tacs.assembleJacobian(res, mat, alpha, beta, gamma)
pc.factor()

res.setRand(1.0, 1.0)
res.applyBCs()
pc.applyFactor(res, ans)
ans.scale(-1.0)

tacs.setVariables(ans)

# Create the function list
funcs = []

# Create the KS function
ksweight = 100.0
for i in xrange(1):
    funcs.append(functions.ksfailure(tacs, ksweight))

func_vals = tacs.evalFunctions(funcs)
print func_vals

# fdvsens = tacs.evalDVSens(funcs, nnodes)

# Set the element flag
flag = (TACS.ToFH5.NODES |
        TACS.ToFH5.DISPLACEMENTS |
        TACS.ToFH5.STRAINS)
f5 = TACS.ToFH5(tacs, TACS.PY_PLANE_STRESS, flag)
f5.writeToFile('triangle_test.f5')
