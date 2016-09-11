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
    def __init__(self, tacs, const, num_materials,
                 sigma=10.0, m_fixed=1.0, eps=1e-4):
        '''
        Analysis problem
        '''

        # Set the TACS object and the constitutive list
        self.tacs = tacs
        self.const = const

        # Set the material information
        self.num_materials = num_materials
        self.num_elements = self.tacs.getNumElements()

        # Set the number of design variables
        ncon = 1
        nwblock = 1
        self.num_design_vars = (self.num_materials+1)*self.num_elements + 2

        # Initialize the super class
        super(TACSAnalysis, self).__init__(MPI.COMM_SELF,
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

        # Set the l1 penalty function
        self.penalty = np.zeros(self.num_design_vars)
        self.xinit = np.zeros(self.num_design_vars)
        self.xcurr = np.zeros(self.num_design_vars)

        # Allocate a vector that stores the gradient of the mass
        self.gmass = np.zeros(self.num_design_vars)

        # Create the mass function and evaluate the gradient of the
        # mass. This is assumed to remain constatnt throughout the
        # optimization.
        self.mass_func = functions.mass(self.tacs)
        self.tacs.evalDVSens(self.mass_func, self.gmass)

        # Set the initial variable values
        self.xinit[:] = 1.0/self.num_materials
        self.xinit[::self.nblock] = 1.0

        # Set the initial linearization
        self.RAMP_penalty = 0.0

        # Create the FH5 file object
        flag = (TACS.ToFH5.NODES |
                TACS.ToFH5.DISPLACEMENTS |
                TACS.ToFH5.STRAINS)
        self.f5 = TACS.ToFH5(self.tacs, TACS.PY_PLANE_STRESS, flag)

        # Set the scaling for the objective value
        self.obj_scale = None

        # Set the sigma value for the mass constraint
        self.sigma = sigma

        # Set the target fixed mass
        self.m_fixed = m_fixed

        # Set the number of function/gradient/hessian-vector
        # evaluations to zero
        self.fevals = 0
        self.gevals = 0
        self.hevals = 0

        return
        
    def setNewInitPointPenalty(self, x, gamma):
        '''
        Set the linearized penalty function, given the design variable
        values from the previous iteration and the penalty parameters
        '''

        # Set the linearization point
        self.xinit[:] = x[:]      

        # For each constitutive point, set the new linearization point
        for con in self.const:
            con.setLinearization(self.RAMP_penalty, self.xinit)

        # Set the penalty parameters
        for i in xrange(self.num_elements):
            self.penalty[self.nblock*i] = gamma[i]*(1.0 - x[self.nblock*i])
            for j in xrange(1, self.nblock):
                self.penalty[self.nblock*i+j] = -gamma[i]*x[self.nblock*i+j]

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

    def computeLimitDesign(self, x):
        '''
        Compute the solution as gamma -> infty
        '''
        xinfty = np.zeros(x.shape)

        for i in xrange(self.num_elements):
            jmax = np.argmax(x[i*self.nblock+1:(i+1)*self.nblock])+1
            if 1.0 - x[i*self.nblock] > x[i*self.nblock+jmax]:
                jmax = 0

            if jmax != 0:
                xinfty[i*self.nblock] = 1.0
                xinfty[i*self.nblock+jmax] = 1.0

        return xinfty                

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the design variable values and the bounds'''
        self.tacs.getDesignVars(x)
        self.tacs.getDesignVarRange(lb, ub)

        # Set bounds for the mass penalty function
        lb[-2:] = 0.0
        ub[-2:] = 1e20
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

    def getL1Objective(self, x, gamma):
        '''Compute the full objective'''

        # Set the design variable values
        self.tacs.setDesignVars(x)
        self.setNewInitPointPenalty(x, gamma)

        # Assemble the Jacobian
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(self.res, self.mat,
                                   1.0, 0.0, 0.0)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        # Compute the compliance objective
        compliance = self.u.dot(self.res)
        fobj = compliance/self.obj_scale
        
        # Set the variable values
        self.u.scale(-1.0)
        self.tacs.setVariables(self.u)

        # Set the penalty parameters
        fpenalty = 0.0
        for i in xrange(self.num_elements):
            fpenalty += 0.5*gamma[i]*((2.0 - x[self.nblock*i])*x[self.nblock*i])
            for j in xrange(1, self.nblock):
                fpenalty -= 0.5*gamma[i]*x[self.nblock*i+j]**2

        # Add the mass constraint term
        mass = np.dot(self.gmass, x)
        fobj += self.sigma*x[-2]

        # Add the full penalty from the objective
        fpenalty += fobj

        return compliance, fobj, fpenalty

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
        self.tacs.assembleJacobian(self.res, self.mat,
                                   1.0, 0.0, 0.0)
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
        obj = obj/self.obj_scale + np.dot(self.penalty, x)
                    
        # Compute the mass of the entire truss
        mass = np.dot(self.gmass, x)

        # Create the constraint c(x) >= 0.0 for the mass
        obj += self.sigma*x[-2]
        con = np.array([mass/self.m_fixed - 1.0 - x[-2] + x[-1]])

        fail = 0
        return fail, obj, con

    def evalObjConGradient(self, x, gobj, Acon):
        '''
        Evaluate the derivative of the compliance and mass
        '''
        
        # Add the number of gradient evaluations
        self.gevals += 1

        # Zero the objecive and constraint gradients
        gobj[:] = 0.0

        # Evaluate the derivative
        self.tacs.evalAdjointResProduct(self.u, gobj)

        # Scale the objective gradient
        gobj[:] /= -self.obj_scale
        gobj[:] += self.penalty

        # Add the contribution to the constraint
        Acon[0,:] = self.gmass/self.m_fixed
        Acon[0,-2] = -1.0
        Acon[0,-1] = 1.0
        
        # Add the contribution to the objective gradient
        gobj[-2] += self.sigma

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
    
def create_structure(comm, nx=8, ny=8, Lx=100.0, Ly=100.0,
                     sigma=20, eps=1e-3):
    '''
    Create a structure with the speicified number of nodes along the
    x/y directions, respectively.
    '''

    # Set the material properties
    rho =    np.array([0.7,   1.0, 1.15])
    E = 70e3*np.array([0.725, 1.0, 1.125])
    nu =     np.array([0.3,  0.3, 0.3])

    # Compute the fixed mass fraction
    m_fixed = 0.3*Lx*Ly*rho[1]
    
    # Set the number of design variables
    num_design_vars = (len(E) + 1)*nx*ny

    # Create the TACS creator object
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
        x = np.linspace(0, Lx, 2*nx+1)
        y = np.linspace(0, Ly, 2*ny+1)
        for j in xrange(2*ny+1):
            for i in xrange(2*nx+1):
                Xpts[3*nodes[i,j]] = x[i]
                Xpts[3*nodes[i,j]+1] = y[j]
                
        # Set the node locations
        creator.setNodes(Xpts)  
    
    # Create the elements in the 
    elems = []
    const = []
    for j in xrange(ny):
        for i in xrange(nx):
            # Create the plane stress stiffness
            n = i + nx*j
            dv_offset = (1 + len(E))*n
            ps = multitopo.MultiTopo(rho, E, nu, dv_offset, eps)
            const.append(ps)
            elems.append(elements.PlaneQuad(3, ps))

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
    
    # Create the tractions and add them to the surface
    surf = 1 # The u=1 positive surface
    tx = np.zeros(3)
    ty = -100*np.ones(3)
    trac = elements.PSQuadTraction(surf, tx, ty)

    # Create the auxiliary element class    
    aux = TACS.AuxElements()
    for j in xrange(ny/8):
        num = nx-1 + nx*j
        aux.addElement(num, trac)

    # Add the auxiliary element class to TACS
    tacs.setAuxElements(aux)

    # Create the analysis object
    analysis = TACSAnalysis(tacs, const, len(E),
                            sigma=sigma, eps=eps, m_fixed=m_fixed)

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

def write_tikz_file(x, nx, ny, nmats, filename):
    '''Write a tikz file'''
    
    # Create the initial part of the tikz string
    tikz = '\\documentclass{article}\n'
    tikz += '\\usepackage[usenames,dvipsnames]{xcolor}\n'
    tikz += '\\usepackage{tikz}\n'
    tikz += '\\usepackage[active,tightpage]{preview}\n'
    tikz += '\\PreviewEnvironment{tikzpicture}\n'
    tikz += '\\setlength\PreviewBorder{5pt}%\n\\begin{document}\n'
    tikz += '\\begin{figure}\n\\begin{tikzpicture}[x=0.25cm, y=0.25cm]\n'
    tikz += '\\sffamily\n'

    x = x.reshape((ny, nx, nmats+1))
    
    # Write out the file so that it looks like a multi-material problem
    grey = [225, 225, 225]
    rgbvals = [(44, 160, 44),
               (255, 127, 14),
               (31, 119, 180)]

    X = np.linspace(0, 50, nx+1)
    Y = np.linspace(0, 50, ny+1)

    for j in xrange(ny):
        for i in xrange(nx):
            # Determine the color to use
            kmax = np.argmax(x[j, i, 1:])
            
            u = x[j,i,0]
            r = (1.0 - u)*grey[0] + u*rgbvals[kmax][0]
            g = (1.0 - u)*grey[1] + u*rgbvals[kmax][1]
            b = (1.0 - u)*grey[2] + u*rgbvals[kmax][2]

            tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}\n'%(
                int(r), int(g), int(b))
            tikz += '\\fill[mycolor] (%f, %f) rectangle (%f, %f);\n'%(
                X[i], Y[j], X[i+1], Y[j+1])
                        
    tikz += '\\end{tikzpicture}\\end{figure}\\end{document}\n'
    # Write the solution file
    fp = open(filename, 'w')
    if fp:
        fp.write(tikz)
        fp.close()

    return

def optimize_plane_stress(comm, nx, ny, root_dir='results',
                          sigma=100.0, max_d=1e-4, theta=1e-3,
                          parameter=5.0, max_iters=50, optimizer='paropt'):
    # Optimize the structure
    penalization='RAMP'
    heuristic = '%s%.0f_%s'%(penalization, parameter, optimizer)
    prefix = os.path.join(root_dir, '%dx%d'%(nx, ny), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
   
    # Create the ground structure and optimization
    analysis = create_structure(comm, nx, ny, sigma=sigma)
    
    # Set up the optimization problem in ParOpt
    if optimizer == 'paropt':
        opt = create_paropt(analysis,
                            use_hessian=use_hessian,
                            qn_type=ParOpt.BFGS)
        
    # Log the optimization file
    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, "compliance", "fobj", "fpenalty", '
    s += '"min gamma", "max gamma", "gamma", '
    s += '"min d", "max d", "tau", "ninfeas", "mass infeas", '
    s += 'feval, geval, hvec, time\n'
    s += 'Zone T = %s\n'%(heuristic)
    fp.write(s)

    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Initialize the gamma values
    gamma = np.zeros(analysis.num_elements)

    # Previous value of the objective function
    fobj_prev = 0.0

    # Set the first time
    first_time = True

    # Set the initial compliance value
    comp_prev = 0.0

    # Set the tolerances for increasing/decreasing tau
    delta_tau_target = 1.0

    # Set the target rate of increase in gamma
    delta_max = 10.0
    delta_min = 1e-3

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
            opt.setInitBarrierParameter(1e-4)
        opt.optimize()

        # Get the optimized point
        x = opt.getOptimizedPoint()

        # Get the discrete infeasibility measure
        d = analysis.getDiscreteInfeas(x)

        # Compute the infeasibility of the mass constraint
        m_infeas = analysis.getMass(x)/analysis.m_fixed - 1.0

        # Compute the discrete infeasibility measure
        tau = np.sum(d)

        # Get the compliance and objective values
        comp, fobj, fpenalty = analysis.getL1Objective(x, gamma)

        # Keep track of how many bars are discrete infeasible
        draw_list = []
        for i in xrange(len(d)):
            if d[i] > max_d:
                draw_list.append(i)

        # Print out the iteration information to the screen
        print 'Iteration %d'%(k)
        print 'Min/max gamma: %15.5e %15.5e  Total: %15.5e'%(
            np.min(gamma), np.max(gamma), np.sum(gamma))
        print 'Min/max d:     %15.5e %15.5e  Total: %15.5e'%(
            np.min(d), np.max(d), np.sum(d))
        print 'Mass infeas:   %15.5e'%(m_infeas)

        s = '%d %e %e %e %e %e %e %e %e %e %2d %e '%(
            k, comp, fobj, fpenalty,
            np.min(gamma), np.max(gamma), np.sum(gamma),
            np.min(d), np.max(d), np.sum(d), len(draw_list), m_infeas)
        s += '%d %d %d %e\n'%(
            analysis.fevals, analysis.gevals, analysis.hevals, 
            MPI.Wtime() - init_time)
        fp.write(s)
        fp.flush()

        # Terminate if the maximum discrete infeasibility measure is
        # sufficiently low
        if np.max(d) <= max_d:
            break

        # Print the output
        filename = 'opt_struct_iter%d.f5'%(k)
        output = os.path.join(prefix, filename)
        analysis.writeOutput(output)

        # Print out the design variables
        filename = 'opt_struct_iter%d.tex'%(k)
        output = os.path.join(prefix, filename)
        write_tikz_file(x[:-2], nx, ny, analysis.num_materials, output)

        if (np.fabs((comp - comp_prev)/comp) < 1e-3):
            if first_time:
                # Set the new value of delta
                gamma[:] = delta_min

                # Keep track of the previous value of the discrete
                # infeasibility measure
                tau_iter = 1.0*tau
                delta_iter = 1.0*delta_min

                # Set the first time flag to false
                first_time = False
            else:
                # Set the maximum delta initially
                delta = 1.0*delta_max

                # Limit the rate of discrete infeasibility increase
                tau_rate = (tau_iter - tau)/delta_iter 
                delta = max(min(delta, delta_tau_target/tau_rate), delta_min)
                gamma[:] = gamma + delta

                # Print out the chosen scaling for the design variables
                print 'Delta:         %15.5e'%(delta)

                # Keep track of the discrete infeasibility measure
                tau_iter = 1.0*tau
                delta_iter = 1.0*delta

        xinfty = analysis.computeLimitDesign(x)
        
        # Set the new penalty
        analysis.RAMP_penalty = parameter
        analysis.setNewInitPointPenalty(x, gamma)

        # Store the previous value of the objective function
        fobj_prev = 1.0*fobj
        comp_prev = 1.0*comp

        # Increase the iteration counter
        niters += 1

    # Close the log file
    fp.close()

    # Print out the design variables
    filename = 'final_opt_struct.tex'
    output = os.path.join(prefix, filename)
    write_tikz_file(x[:-2], nx, ny, analysis.num_materials, output)

    # Save the final optimized point
    fname = os.path.join(prefix, 'x_opt.dat')
    x = opt.getOptimizedPoint()
    np.savetxt(fname, x)

    # Get the rounded design
    xinfty = analysis.computeLimitDesign(x)
    fname = os.path.join(prefix, 'x_opt_infty.dat')
    np.savetxt(fname, xinfty)

    return

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=24, 
                    help='Nodes in x-direction')
parser.add_argument('--ny', type=int, default=24, 
                    help='Nodes in y-direction')
parser.add_argument('--parameter', type=float, default=5.0,
                    help='Penalization parameter')
parser.add_argument('--sigma', type=float, default=100.0,
                    help='Mass penalty parameter value')
parser.add_argument('--optimizer', type=str, default='paropt',
                    help='Optimizer name')
parser.add_argument('--x_infty', type=str, default=None,
                    help='Infinity solution for analysis')
parser.add_argument('--use_hessian', default=False,
                    action='store_true',
                    help='Use hessian-vector products')
args = parser.parse_args()

# Get the arguments
nx = args.nx
ny = args.ny
parameter = args.parameter
sigma = args.sigma
optimizer = args.optimizer
use_hessian = args.use_hessian
x_infty = args.x_infty

# Set the root results directory
root_dir = 'results'

comm = MPI.COMM_WORLD
if x_infty is not None:
    # Load in the input file
    x = np.loadtxt(x_infty)

    # Create the analysis object
    analysis = create_structure(comm, nx, ny, sigma=sigma)
    analysis.evalObjCon(x)

    # Perform the analysis about the infinity design point
    gamma = np.zeros(analysis.num_elements)
    comp, fobj, fpenalty = analysis.getL1Objective(x, gamma)

    print 'Compliance = %15.8e'%(comp)
else:
    optimize_plane_stress(comm, nx, ny, root_dir=root_dir,
                          sigma=sigma, parameter=parameter,
                          max_iters=50, optimizer=optimizer)
