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
        self.obj_scale = 1.0

        # Set the sigma value for the mass constraint
        self.sigma = sigma

        # Set the target fixed mass
        self.m_fixed = m_fixed

        # Set the number of function/gradient/hessian-vector evaluations to zero
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
        for i in xrange(self.nelems):
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

        for i in xrange(self.nelems):
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
        self.tacs.setNewInitPointPenalty(x, gamma)

        # Assemble the Jacobian
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(self.res, self.mat,
                                   alpha, beta, gamma)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        # Compute the compliance objective
        compliance = self.u.dot(self.res)
        fobj = compliance/self.obj_scale
        
        # Set the penalty parameters
        fpenalty = 0.0
        for i in xrange(self.nelems):
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
        
        # Add the number of function evaluations
        self.fevals += 1

        # Set the design variable values
        self.tacs.setDesignVars(x)

        # Assemble the Jacobian
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(self.res, self.mat,
                                   alpha, beta, gamma)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        self.tacs.testElement(0, 2)

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

        self.ksm.solve(self.res, self.psi)

        # Evaluate the adjoint-residual product
        self.tacs.evalAdjointResProduct(self.psi, hvec)      

        # Scale the result to the correct range
        hvec /= self.obj_scale

        fail = 0
        return fail

    def writeOutput(self):
        # Set the element flag
        self.f5.writeToFile('plane_stress%04d.f5'%(self.fevals))
        return
    
def create_structure(comm, nx=8, ny=8, sigma=20, eps=1e-4):
    '''
    Create a structure with the speicified number of nodes along the
    x/y directions, respectively.
    '''

    # Set the material properties
    E = np.array([70e3, 95e3, 120e3])
    rho = np.array([1.0, 1.3, 1.9])
    nu = np.array([0.3, 0.3, 0.3])
    
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
        x = np.linspace(0, 10, 2*nx+1)
        y = np.linspace(0, 10, 2*nx+1)
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
                            sigma=sigma, eps=eps)

    return analysis

def create_paropt(analysis, use_hessian=False,
                  max_qn_subspace=50, qn_type=ParOpt.BFGS):
    '''
    Optimize the given structure using ParOpt
    '''

    # Create the optimizer
    opt = ParOpt.pyParOpt(analysis, max_qn_subspace, qn_type)

    # Set the optimality tolerance
    opt.setAbsOptimalityTol(1e-5)

    # Set the Hessian-vector product iterations
    if use_hessian:
        opt.setUseLineSearch(0)
        opt.setUseHvecProduct(1)
        opt.setGMRESSubspaceSize(100)
        opt.setNKSwitchTolerance(1.0)
        opt.setEisenstatWalkerParameters(0.5, 0.0)
        opt.setGMRESTolerances(1.0, 1e-30)
    else:
        opt.setUseHvecProduct(0)

    # Set optimization parameters
    opt.setArmijioParam(1e-5)
    opt.setMaxMajorIterations(2500)

    # Perform a quick check of the gradient (and Hessian)
    opt.checkGradients(1e-6)

    return opt

def optimize_plane_stress(comm, nx, ny, root_dir='results',
                          sigma=100.0, max_d=1e-4, theta=1e-3,
                          parameter=5.0, max_iters=50):
    # Optimize the structure
    penalization='RAMP'
    heuristic = '%s%.0f'%(penalization, parameter)
    prefix = os.path.join(root_dir, '%dx%d'%(nx, ny), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
   
    # Create the ground structure and optimization
    analysis = create_structure(comm, nx, ny, sigma=sigma)
    
    # Set up the optimization problem in ParOpt
    opt = create_paropt(analysis, use_hessian=use_hessian,
                        qn_type=ParOpt.BFGS)

    # # Create a vector of all ones
    # m_add = 0.0
    # if use_mass_constraint:
    #     xones = np.ones(truss.gmass.shape)
    #     m_add = truss.getMass(xones)/truss.nmats
    
    # # Keep track of the fixed mass
    # m_fixed_init = 1.0*truss.m_fixed
    # truss.m_fixed = m_fixed_init + truss.x_lb*m_add

    # # Log the optimization file
    # log_filename = os.path.join(prefix, 'log_file.dat')
    # fp = open(log_filename, 'w')

    # # Write the header out to the file
    # s = 'Variables = iteration, "compliance", "fobj", "fpenalty", '
    # s += '"min gamma", "max gamma", "gamma", '
    # s += '"min d", "max d", "tau", "ninfeas", "mass infeas", '
    # s += 'feval, geval, hvec, time\n'
    # s += 'Zone T = %s\n'%(heuristic)
    # fp.write(s)

    # # Keep track of the ellapsed CPU time
    # init_time = MPI.Wtime()

    # # Initialize the gamma values
    # gamma = np.zeros(truss.nelems)

    # # Previous value of the objective function
    # fobj_prev = 0.0

    # # Set the first time
    # first_time = True

    # # Set the initial compliance value
    # comp_prev = 0.0

    # # Set the tolerances for increasing/decreasing tau
    # delta_tau_target = 1.0

    # # Set the target rate of increase in gamma
    # delta_max = 10.0
    # delta_min = 1e-3

    # # Keep track of the number of iterations
    # niters = 0
    # for k in xrange(max_iters):
    #     # Set the output file to use
    #     fname = os.path.join(prefix, 'truss_paropt_iter%d.out'%(k)) 
    #     opt.setOutputFile(fname)

    #     # Optimize the truss
    #     if k > 0:
    #         opt.setInitBarrierParameter(1e-4)
    #     opt.optimize()

    #     # Get the optimized point
    #     x = opt.getOptimizedPoint()

    #     # Get the discrete infeasibility measure
    #     d = truss.getDiscreteInfeas(x)

    #     # Compute the infeasibility of the mass constraint
    #     m_infeas = truss.getMass(x)/truss.m_fixed - 1.0

    #     # Compute the discrete infeasibility measure
    #     tau = np.sum(d)

    #     # Get the compliance and objective values
    #     comp, fobj, fpenalty = truss.getL1Objective(x, gamma)

    #     # Keep track of how many bars are discrete infeasible
    #     draw_list = []
    #     for i in xrange(len(d)):
    #         if d[i] > max_d:
    #             draw_list.append(i)

    #     # Print out the iteration information to the screen
    #     print 'Iteration %d'%(k)
    #     print 'Min/max gamma: %15.5e %15.5e  Total: %15.5e'%(
    #         np.min(gamma), np.max(gamma), np.sum(gamma))
    #     print 'Min/max d:     %15.5e %15.5e  Total: %15.5e'%(
    #         np.min(d), np.max(d), np.sum(d))
    #     print 'Mass infeas:   %15.5e'%(m_infeas)

    #     s = '%d %e %e %e %e %e %e %e %e %e %2d %e '%(
    #         k, comp, fobj, fpenalty,
    #         np.min(gamma), np.max(gamma), np.sum(gamma),
    #         np.min(d), np.max(d), np.sum(d), len(draw_list), m_infeas)
    #     s += '%d %d %d %e\n'%(
    #         truss.fevals, truss.gevals, truss.hevals, 
    #         MPI.Wtime() - init_time)
    #     fp.write(s)
    #     fp.flush()

    #     # Terminate if the maximum discrete infeasibility measure is
    #     # sufficiently low
    #     if np.max(d) <= max_d:
    #         break

    #     # Print the output
    #     filename = 'opt_truss_iter%d.tex'%(k)
    #     output = os.path.join(prefix, filename)
    #     truss.printTruss(x, filename=output, draw_list=draw_list)

    #     if (np.fabs((comp - comp_prev)/comp) < 1e-3):
    #         if first_time:
    #             # Set the new value of delta
    #             gamma[:] = delta_min

    #             # Keep track of the previous value of the discrete
    #             # infeasibility measure
    #             tau_iter = 1.0*tau
    #             delta_iter = 1.0*delta_min

    #             # Set the first time flag to false
    #             first_time = False
    #         else:
    #             # Set the maximum delta initially
    #             delta = 1.0*delta_max

    #             # Limit the rate of discrete infeasibility increase
    #             tau_rate = (tau_iter - tau)/delta_iter 
    #             delta = max(min(delta, delta_tau_target/tau_rate), delta_min)
    #             gamma[:] = gamma + delta

    #             # Print out the chosen scaling for the design variables
    #             print 'Delta:         %15.5e'%(delta)

    #             # Keep track of the discrete infeasibility measure
    #             tau_iter = 1.0*tau
    #             delta_iter = 1.0*delta

    #     xinfty = truss.computeLimitDesign(x)

    #     # Print the output
    #     filename = 'opt_limit_truss_iter%d.tex'%(k)
    #     output = os.path.join(prefix, filename)
    #     truss.printTruss(xinfty, filename=output)
        
    #     # Set the new penalty
    #     truss.SIMP = parameter
    #     truss.RAMP = parameter
    #     truss.penalization = penalization
    #     truss.setNewInitPointPenalty(x, gamma)

    #     # Store the previous value of the objective function
    #     fobj_prev = 1.0*fobj
    #     comp_prev = 1.0*comp

    #     # Increase the iteration counter
    #     niters += 1

    # # Close the log file
    # fp.close()
    
    # # Print out the last optimized truss
    # filename = 'opt_truss.tex'
    # output = os.path.join(prefix, filename)
    # truss.printTruss(x, filename=output)
    # os.system('cd %s; pdflatex %s > /dev/null ; cd ..;'%(prefix, filename))

    # # Save the final optimized point
    # fname = os.path.join(prefix, 'x_opt.dat')
    # x = opt.getOptimizedPoint()
    # np.savetxt(fname, x)

    # # Get the rounded design
    # xinfty = truss.computeLimitDesign(x)
    # fname = os.path.join(prefix, 'x_opt_infty.dat')
    # np.savetxt(fname, xinfty)

    return

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=24, 
                    help='Nodes in x-direction')
parser.add_argument('--ny', type=int, default=24, 
                    help='Nodes in y-direction')
parser.add_argument('--parameter', type=float, default=5.0,
                    help='Penalization parameter')
parser.add_argument('--sigma', type=float, default=20.0,
                    help='Penalty parameter value')
args = parser.parse_args()

# Get the arguments
nx = args.nx
ny = args.ny
parameter = args.parameter
sigma = args.sigma

# Set the root results directory
root_dir = 'results'

# Always use the Hessian-vector product implementation
use_hessian = True

comm = MPI.COMM_WORLD
optimize_plane_stress(comm, nx, ny, root_dir=root_dir,
                      sigma=sigma, parameter=parameter, max_iters=80)
