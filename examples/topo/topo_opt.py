import argparse
import os
import numpy as np
from mpi4py import MPI

# Import TACS and assorted repositories
from tacs import TACS, elements, constitutive, functions

# Import ParOpt
from paropt import ParOpt

# Include the extension module
import multitopo

class TACSAnalysis(ParOpt.pyParOptProblem):
    def __init__(self, comm, tacs, const, num_materials,
                 xpts=None, conn=None, m_fixed=1.0):
        '''
        Analysis problem
        '''

        # Keep the communicator
        self.comm = comm

        # Set the TACS object and the constitutive list
        self.tacs = tacs
        self.const = const

        # Set the material information
        self.num_materials = num_materials
        self.num_elements = self.tacs.getNumElements()

        # Keep the pointer to the connectivity/positions
        self.xpts = xpts
        self.conn = conn

        # Set the number of design variables
        ncon = 1
        nwblock = 1
        self.num_design_vars = (self.num_materials+1)*self.num_elements

        # Initialize the super class
        super(TACSAnalysis, self).__init__(comm,
                                           self.num_design_vars, ncon,
                                           self.num_elements, nwblock)

        # Set the size of the design variable 'blocks'
        self.nblock = self.num_materials+1

        # Create the state variable vectors
        self.res = tacs.createVec()
        self.u = tacs.createVec()
        self.psi = tacs.createVec()
        self.mat = tacs.createFEMat()

        # Create the preconditioner for the corresponding matrix
        self.pc = TACS.Pc(self.mat)

        # Create the KSM object
        subspace_size = 20
        nrestart = 0
        self.ksm = TACS.KSM(self.mat, self.pc,
                            subspace_size, nrestart)
        self.ksm.setTolerances(1e-12, 1e-30)

        # Set the block size
        self.nwblock = self.num_materials+1

        # Set the linearization point
        self.xinit = np.zeros(self.num_design_vars)
        self.xcurr = np.zeros(self.num_design_vars)

        # Allocate a vector that stores the gradient of the mass
        self.gmass = np.zeros(self.num_design_vars)

        # Create the mass function and evaluate the gradient of the
        # mass. This is assumed to remain constatnt throughout the
        # optimization.
        self.mass_func = functions.StructuralMass(self.tacs)
        self.tacs.evalDVSens(self.mass_func, self.gmass)

        # Set the initial variable values
        self.xinit[:] = 1.0
        self.xinit[::self.nblock] = 1.0

        # Set the target fixed mass
        self.m_fixed = m_fixed

        # Set the initial design variable values
        xi = self.m_fixed/np.dot(self.gmass, self.xinit)
        self.xinit[:] = xi/self.num_materials
        self.xinit[::self.nblock] = xi

        # Set the initial linearization
        self.RAMP_penalty = 0.0

        # Create the FH5 file object
        flag = (TACS.ToFH5.NODES |
                TACS.ToFH5.DISPLACEMENTS |
                TACS.ToFH5.STRAINS)
        self.f5 = TACS.ToFH5(self.tacs, TACS.PY_PLANE_STRESS, flag)

        # Set the scaling for the objective value
        self.obj_scale = None

        # Set the number of function/gradient/hessian-vector
        # evaluations to zero
        self.fevals = 0
        self.gevals = 0
        self.hevals = 0

        return
        
    def setNewInitPointPenalty(self, x):
        '''
        Set the linearized penalty function, given the design variable
        values from the previous iteration and the penalty parameters
        '''

        # Set the linearization point
        self.xinit[:] = x[:]      

        # For each constitutive point, set the new linearization point
        for con in self.const:
            con.setLinearization(self.xinit)
            
        return

    def getDiscreteInfeas(self, x):
        '''
        Compute the discrete infeasibility measure at a given design point
        '''
        d = np.zeros(self.num_elements)
        for i in xrange(self.num_elements):
            tnum = self.nblock*i
            d[i] = 1.0 - (x[tnum] - 1.0)**2 - sum(x[tnum+1:tnum+self.nblock]**2)
            
        return d

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the design variable values and the bounds'''
        q = 5.0
        
        self.tacs.getDesignVars(x)
        self.tacs.getDesignVarRange(lb, ub)
        ub[:] = 1e30
        lb[:] = q/(1.0 + q)*self.xinit[:]
        ub[::self.nblock] = 1.0
        lb[::self.nblock] = -1e30
        return
        
    def evalSparseCon(self, x, con):
        '''Evaluate the sparse constraints'''
        n = self.nblock*self.num_elements
        con[:] = (2.0*x[:n:self.nblock] - 
                  np.sum(x[:n].reshape(-1, self.nblock), axis=1))
        return

    def addSparseJacobian(self, alpha, x, px, con):
        '''Compute the Jacobian-vector product con = alpha*J(x)*px'''
        n = self.nblock*self.num_elements
        con[:] += alpha*(2.0*px[:n:self.nblock] - 
                         np.sum(px[:n].reshape(-1, self.nblock), axis=1))
        return

    def addSparseJacobianTranspose(self, alpha, x, pz, out):
        '''Compute the transpose Jacobian-vector product alpha*J^{T}*pz'''
        n = self.nblock*self.num_elements
        out[:n:self.nblock] += alpha*pz
        for k in xrange(1,self.nblock):
            out[k:n:self.nblock] -= alpha*pz
        return

    def addSparseInnerProduct(self, alpha, x, c, A):
        '''Add the results from the product J(x)*C*J(x)^{T} to A'''
        n = self.nblock*self.num_elements
        A[:] += alpha*np.sum(c[:n].reshape(-1, self.nblock), axis=1)        
        return

    def getCompliance(self, x):
        '''Compute the full objective'''

        # Set the design variable values
        self.tacs.setDesignVars(x)
        self.setNewInitPointPenalty(x)

        # Assemble the Jacobian
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, self.res, self.mat)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        # Compute the compliance objective
        compliance = self.u.dot(self.res)

        return compliance

    def getMass(self, x):
        '''Return the mass of the truss'''
        return np.dot(self.gmass, x)

    def evalObjCon(self, x):
        '''
        Evaluate the objective (compliance) and constraint (mass)
        '''
        # Copy the design variable values
        self.xcurr[:] = x[:]    

        # Add the number of function evaluations
        self.fevals += 1

        # Set the design variable values
        self.tacs.setDesignVars(x)

        # Assemble the Jacobian
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, self.res, self.mat)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        # Compute the compliance objective
        obj = self.u.dot(self.res)
        if self.obj_scale is None:
            self.obj_scale = 1.0*obj

        # Scale the variables and set the state variables
        self.u.scale(-1.0)
        self.tacs.setVariables(self.u)
        
        # Scale the compliance objective
        obj = obj/self.obj_scale
                    
        # Compute the mass of the entire truss
        mass = np.dot(self.gmass, x)

        # Create the constraint c(x) >= 0.0 for the mass
        con = np.array([mass/self.m_fixed - 1.0])

        fail = 0
        return fail, obj, con

    def evalObjConGradient(self, x, gobj, Acon):
        '''
        Evaluate the derivative of the compliance and mass
        '''
        
        # Add the number of gradient evaluations
        self.gevals += 1

        # Evaluate the derivative
        self.tacs.evalAdjointResProduct(self.u, gobj)

        # Scale the objective gradient
        gobj[:] /= -self.obj_scale

        # Add the contribution to the constraint
        Acon[0,:] = self.gmass/self.m_fixed

        # Return the 
        fail = 0
        return fail

    def evalHvecProduct(self, x, z, zw, px, hvec):
        '''
        Evaluate the product of the input vector px with the Hessian
        of the Lagrangian.
        '''

        # Add the number of function evaluations
        self.hevals += 1
        
        # Zero the hessian-vector product
        hvec[:] = 0.0

        # Assemble the residual
        multitopo.assembleProjectDVSens(self.tacs, px, self.res)

        # Solve K(x)*psi = res
        self.ksm.solve(self.res, self.psi)

        # Evaluate the adjoint-residual product
        self.tacs.evalAdjointResProduct(self.psi, hvec)      

        # Scale the result to the correct range
        hvec /= 0.5*self.obj_scale

        fail = 0
        return fail

    def writeOutput(self, filename):
        # Set the element flag
        self.f5.writeToFile(filename)
        return
    
    def writeTikzFile(self, x, filename):
        '''Write a tikz file'''

        if (self.comm.rank == 0 and
            (self.xpts is not None) and (self.conn is not None)):
            # Create the initial part of the tikz string
            tikz = '\\documentclass{article}\n'
            tikz += '\\usepackage[usenames,dvipsnames]{xcolor}\n'
            tikz += '\\usepackage{tikz}\n'
            tikz += '\\usepackage[active,tightpage]{preview}\n'
            tikz += '\\PreviewEnvironment{tikzpicture}\n'
            tikz += '\\setlength\PreviewBorder{5pt}%\n\\begin{document}\n'
            tikz += '\\begin{figure}\n\\begin{tikzpicture}[x=0.25cm, y=0.25cm]\n'
            tikz += '\\sffamily\n'
               
            # Write out a tikz file
            grey = [225, 225, 225]
            rgbvals = [(44, 160, 44),
                       (255, 127, 14),
                       (31, 119, 180)]

            for i in xrange(self.conn.shape[0]):
                # Determine the color to use
                kmax = np.argmax(x[self.nblock*i+1:self.nblock*(i+1)])
            
                u = x[self.nblock*i]
                r = (1.0 - u)*grey[0] + u*rgbvals[kmax][0]
                g = (1.0 - u)*grey[1] + u*rgbvals[kmax][1]
                b = (1.0 - u)*grey[2] + u*rgbvals[kmax][2]

                tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}\n'%(
                    int(r), int(g), int(b))
                tikz += '\\fill[mycolor] (%f,%f) -- (%f,%f)'%(
                    self.xpts[self.conn[i,0],0], self.xpts[self.conn[i,0],1],
                    self.xpts[self.conn[i,1],0], self.xpts[self.conn[i,1],1])
                tikz += '-- (%f,%f) -- (%f,%f) -- cycle;\n'%(
                    self.xpts[self.conn[i,3],0], self.xpts[self.conn[i,3],1],
                    self.xpts[self.conn[i,2],0], self.xpts[self.conn[i,2],1])
                        
            tikz += '\\end{tikzpicture}\\end{figure}\\end{document}\n'
    
            # Write the solution file
            fp = open(filename, 'w')
            if fp:
                fp.write(tikz)
                fp.close()

        return

def rectangular_domain(nx, ny, Lx=100.0):

    # Set the y-dimension based on a unit aspect ratio
    Ly = (ny*Lx)/nx

    # Compute the total area
    area = Lx*Ly
    
    # Set the number of elements/nodes in the problem
    nnodes = (nx+1)*(ny+1)
    nelems = nx*ny
    nodes = np.arange(nnodes).reshape((nx+1, ny+1))          

    # Set the node locations
    xpts = np.zeros((nnodes, 3))
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    for j in xrange(ny+1):
        for i in xrange(nx+1):
            xpts[nodes[i,j],0] = x[i]
            xpts[nodes[i,j],1] = y[j]

    # Set the connectivity and create the corresponding elements
    conn = np.zeros((nelems, 4), dtype=np.intc)
    for j in xrange(ny):
        for i in xrange(nx):
            # Append the first set of nodes
            n = i + nx*j
            conn[n,0] = nodes[i, j]
            conn[n,1] = nodes[i+1, j]
            conn[n,2] = nodes[i, j+1]
            conn[n,3] = nodes[i+1, j+1]

    bcs = np.array(nodes[0,:], dtype=np.intc)
            
    # Create the tractions and add them to the surface
    surf = 1 # The u=1 positive surface
    tx = np.zeros(2)
    ty = -100*np.ones(2)
    trac = elements.PSQuadTraction(surf, tx, ty)

    # Create the auxiliary element class    
    aux = TACS.AuxElements()
    for j in xrange(ny/8):
        num = nx-1 + nx*j
        aux.addElement(num, trac)
        
    return xpts, conn, bcs, aux, area

def create_structure(comm, props, xpts, conn, bcs, aux, m_fixed, r0=4):
    '''
    Create a structure with the speicified number of nodes along the
    x/y directions, respectively.
    '''
    
    # Set the number of design variables
    nnodes = len(xpts)
    nelems = len(conn)
    num_design_vars = (len(E) + 1)*nelems

    # Create the TACS creator object
    creator = TACS.Creator(comm, 2)

    # Set up the mesh on the root processor
    if comm.rank == 0:        
        # Set the node pointers
        ptr = np.arange(0, 4*nelems+1, 4, dtype=np.intc)
        elem_ids = np.arange(nelems, dtype=np.intc)
        creator.setGlobalConnectivity(nnodes, ptr, conn.flatten(), elem_ids)

        # Set up the boundary conditions
        bcnodes = np.array(bcs, dtype=np.intc)
        
        # Set the boundary condition variables
        nbcs = 2*bcnodes.shape[0]
        bcvars = np.zeros(nbcs, dtype=np.intc)
        bcvars[:nbcs:2] = 0
        bcvars[1:nbcs:2] = 1
        
        # Set the boundary condition pointers
        bcptr = np.arange(0, nbcs+1, 2, dtype=np.intc)
        creator.setBoundaryConditions(bcnodes, bcptr, bcvars)
        
        # Set the node locations
        creator.setNodes(xpts.flatten())

    # Create the filter and the elements for each point
    elems = []
    const = []
    xcentroid = np.zeros((nelems, 3))
    for i in xrange(nelems):
        xcentroid[i] = 0.25*(xpts[conn[i,0]] + xpts[conn[i,1]] +
                             xpts[conn[i,2]] + xpts[conn[i,3]])    

    # Find the closest points
    locator = multitopo.Locator(xcentroid)
    max_points = 15
    for i in xrange(nelems):
        # Get the closest points to the centroid of this element
        nodes, dist = locator.closest(xcentroid[i], max_points)
        index = 0
        while index < max_points and dist[index] < r0:
            index += 1

        # Create the filter weights
        nodes = nodes[:index]
        weights = ((r0 - dist[:index])/r0)**2
        weights = weights/np.sum(weights)

        # Create the plane stress object
        ps = multitopo.MultiTopo(props, nodes, weights)
        const.append(ps)
        elems.append(elements.PlaneQuad(2, ps))

    # Set the elements
    creator.setElements(elems)

    # Create the tacs assembler object
    tacs = creator.createTACS()

    # Retrieve the element partition
    if comm.rank == 0:
        partition = creator.getElementPartition()
        
        # Broadcast the partition
        comm.bcast(partition, root=0)
    else:
        partition = comm.bcast(root=0)
    
    # Add the auxiliary element class to TACS
    tacs.setAuxElements(aux)

    # Create the analysis object
    analysis = TACSAnalysis(comm, tacs, const, len(E),
                            conn=conn, xpts=xpts,
                            m_fixed=m_fixed)

    return analysis

def create_paropt(analysis, use_hessian=False,
                  max_qn_subspace=50, qn_type=ParOpt.BFGS):
    '''
    Optimize the given structure using ParOpt
    '''

    # Set the inequality options
    analysis.setInequalityOptions(dense_ineq=False, 
                                  sparse_ineq=False,
                                  use_lower=True,
                                  use_upper=True)
    
    # Create the optimizer
    opt = ParOpt.pyParOpt(analysis, max_qn_subspace, qn_type)

    # Set the optimality tolerance
    opt.setAbsOptimalityTol(1e-6)

    # Set the Hessian-vector product iterations
    if use_hessian:
        opt.setUseLineSearch(0)
        opt.setUseHvecProduct(1)
        opt.setGMRESSubspaceSize(50)
        opt.setNKSwitchTolerance(1.0)
        opt.setEisenstatWalkerParameters(0.5, 0.0)
        opt.setGMRESTolerances(1.0, 1e-30)
    else:
        opt.setUseHvecProduct(0)

    # Set optimization parameters
    opt.setArmijioParam(1e-5)
    opt.setMaxMajorIterations(2500)

    # Perform a quick check of the gradient (and Hessian)
    opt.checkGradients(1e-8)

    return opt

def create_pyopt(analysis, optimizer='snopt', options={}):
    '''
    Take the given problem and optimize it with the given optimizer
    from the pyOptSparse library of optimizers.
    '''
    # Import the optimization problem
    from pyoptsparse import Optimization, OPT
    from scipy import sparse

    class pyOptWrapper:
        optimizer = None
        options = {}
        opt = None
        prob = None
        def __init__(self, analysis):
            self.analysis = analysis
        def objcon(self, x):
            fail, obj, con = self.analysis.evalObjCon(x['x'])
            funcs = {'objective': obj, 'con': con}
            return funcs, fail
        def gobjcon(self, x, funcs):
            g = np.zeros(x['x'].shape)
            A = np.zeros((1, x['x'].shape[0]))
            fail = self.analysis.evalObjConGradient(x['x'], g, A)
            sens = {'objective': {'x': g}, 'con': {'x': A}}
            return sens, fail

        # Thin wrapper methods to make this look somewhat like ParOpt
        def optimize(self):
            self.opt = OPT(self.optimizer, options=self.options)
            self.sol = self.opt(self.prob, sens=self.gobjcon)
            return

        def setOutputFile(self, fname):
            if self.optimizer == 'snopt':
                self.options['Print file'] = fname
                self.options['Summary file'] = fname + '_summary'
            elif self.optimizer == 'ipopt':
                self.options['bound_relax_factor'] = 0.0
                self.options['linear_solver'] = 'ma27'
                self.options['output_file'] = fname
                self.options['max_iter'] = 2500
            return

        def setInitBarrierParameter(self, *args):
            return

        def getOptimizedPoint(self):
            x = np.array(self.analysis.xcurr)                         
            return x
        
    # Set the design variables
    wrap = pyOptWrapper(analysis)
    prob = Optimization('topo', wrap.objcon)

    # Add the linear constraint
    n = analysis.num_design_vars

    # Create the sparse matrix for the design variable weights
    rowp = [0]
    cols = []
    data = []
    nrows = analysis.num_elements
    ncols = analysis.num_design_vars

    nblock = analysis.num_materials+1
    for i in xrange(analysis.num_elements):
        data.append(1.0)
        cols.append(i*nblock)
        for j in xrange(i*nblock+1, (i+1)*nblock):
            data.append(-1.0)
            cols.append(j)
        rowp.append(len(cols))

    Asparse = {'csr':[rowp, cols, data], 'shape':[nrows, ncols]}

    lower = np.zeros(analysis.num_elements)
    upper = np.zeros(analysis.num_elements)
    prob.addConGroup('lincon', analysis.num_elements,
                     lower=lower, upper=upper,
                     linear=True, wrt=['x'], jac={'x': Asparse})

    # Determine the initial variable values and their lower/upper
    # bounds in the design problem
    x0 = np.zeros(n)
    lb = np.zeros(n)
    ub = np.zeros(n)
    analysis.getVarsAndBounds(x0, lb, ub)

    # Set the variable bounds and initial values
    prob.addVarGroup('x', n, value=x0, lower=lb, upper=ub)

    # Set the constraints
    prob.addConGroup('con', 1, lower=0.0, upper=0.0)
        
    # Add the objective
    prob.addObj('objective')

    # Set the values into the wrapper
    wrap.optimizer = optimizer
    wrap.options = options
    wrap.prob = prob

    return wrap

def optimize_plane_stress(comm, analysis, root_dir='results',
                          parameter=5.0, max_iters=50, optimizer='paropt'):
    # Optimize the structure
    penalization = 'RAMP'
    heuristic = '%s%.0f_%s'%(penalization, parameter, optimizer)
    prefix = os.path.join(root_dir, '%dx%d'%(nx, ny), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
       
    # Set up the optimization problem in ParOpt
    if optimizer == 'paropt':
        opt = create_paropt(analysis,
                            use_hessian=use_hessian,
                            qn_type=ParOpt.BFGS)
        
    # Log the optimization file
    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, "compliance" '
    s += '"min d", "max d", "tau", '
    s += 'feval, geval, hvec, time\n'
    s += 'Zone T = %s\n'%(heuristic)
    fp.write(s)

    # Set the penalty parameter
    analysis.RAMP_penalty = parameter
    analysis.setNewInitPointPenalty(analysis.xinit)
    
    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Set the initial compliance value
    comp_prev = 0.0

    # Keep track of the number of iterations
    niters = 0
    for k in xrange(max_iters):
        if optimizer != 'paropt':
            opt = create_pyopt(analysis, optimizer=optimizer)
        
        # Set the output file to use
        fname = os.path.join(prefix, 'history_iter%d.out'%(k)) 
        opt.setOutputFile(fname)

        # Optimize the truss
        if k > 0:
            opt.setInitStartingPoint(0)
            opt.setInitBarrierParameter(1e-4)
        opt.optimize()

        # Get the optimized point
        x = opt.getOptimizedPoint()

        # Get the discrete infeasibility measure
        d = analysis.getDiscreteInfeas(x)

        # Compute the discrete infeasibility measure
        tau = np.sum(d)

        # Get the compliance and objective values
        comp = analysis.getCompliance(x)

        # Print out the iteration information to the screen
        print 'Iteration %d'%(k)
        print 'Min/max d:     %15.5e %15.5e  Total: %15.5e'%(
            np.min(d), np.max(d), np.sum(d))

        s = '%d %e %e %e %e '%(k, comp, np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e\n'%(
            analysis.fevals, analysis.gevals, analysis.hevals, 
            MPI.Wtime() - init_time)
        fp.write(s)
        fp.flush()

        # Print the output
        filename = 'opt_struct_iter%d.f5'%(k)
        output = os.path.join(prefix, filename)
        analysis.writeOutput(output)

        # Print out the design variables
        filename = 'opt_struct_iter%d.tex'%(k)
        output = os.path.join(prefix, filename)
        analysis.writeTikzFile(x, output)

        if k > 0 and (np.fabs((comp - comp_prev)/comp) < 1e-3):
            break

        # Set the new penalty
        analysis.setNewInitPointPenalty(x)

        # Store the previous value of the objective function
        comp_prev = 1.0*comp

        # Increase the iteration counter
        niters += 1

    # Close the log file
    fp.close()

    # Print out the design variables
    filename = 'final_opt_struct.tex'
    output = os.path.join(prefix, filename)
    analysis.writeTikzFile(x, output)

    # Save the final optimized point
    fname = os.path.join(prefix, 'x_opt.dat')
    x = opt.getOptimizedPoint()
    np.savetxt(fname, x)

    return

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=48,
                    help='Nodes in x-direction')
parser.add_argument('--ny', type=int, default=48, 
                    help='Nodes in y-direction')
parser.add_argument('--parameter', type=float, default=5.0,
                    help='Penalization parameter')
parser.add_argument('--optimizer', type=str, default='paropt',
                    help='Optimizer name')
parser.add_argument('--use_hessian', default=False,
                    action='store_true',
                    help='Use hessian-vector products')
args = parser.parse_args()

# Get the arguments
nx = args.nx
ny = args.ny
parameter = args.parameter
optimizer = args.optimizer
use_hessian = args.use_hessian

# Set the root results directory
root_dir = 'results'

comm = MPI.COMM_WORLD

# Create the connectivity data
xpts, conn, bcs, aux, area = rectangular_domain(nx, ny)

# Set the material properties
rho =    np.array([0.7,   1.0, 1.15])
E = 70e3*np.array([0.725, 1.0, 1.125])
nu =     np.array([0.3,  0.3, 0.3])
props = multitopo.MultiTopoProperties(rho, E, nu)

# Compute the fixed mass fraction
m_fixed = 0.3*area*rho[1]

# Create the analysis object
r0 = 2*np.sqrt(area/len(conn))
analysis = create_structure(comm, props, xpts, conn,
                            bcs, aux, m_fixed, r0=r0)


# Optimize the plane stress problem
optimize_plane_stress(comm, analysis, root_dir=root_dir,
                      parameter=parameter,
                      max_iters=50, optimizer=optimizer)
