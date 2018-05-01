from __future__ import print_function
import argparse
import os
import sys
import numpy as np
from mpi4py import MPI

# Import TACS and assorted repositories
from tacs import TACS, elements, constitutive, functions

# Import ParOpt
from paropt import ParOpt

# Include the extension module
import multitopo

def get_transform(theta):
    '''
    Get the inverse of the stress transformation matrix
    '''
    c = np.cos(theta)
    s = np.sin(theta)

    Tinv = np.zeros((3,3))
    Tinv[0,0] = c**2
    Tinv[0,1] = s**2
    Tinv[0,2] = -2*s*c
    
    Tinv[1,0] = s**2
    Tinv[1,1] = c**2
    Tinv[1,2] = 2*s*c
    
    Tinv[2,0] = s*c
    Tinv[2,1] = -s*c
    Tinv[2,2] = c**2 - s**2

    return Tinv

def get_stiffness(E1, E2, nu12, G12):
    '''
    Given the engineernig constants E1, E2, nu12 and G12, compute the
    stiffness in the material reference frame.
    '''

    # Compute the stiffness matrix in the material coordinate frame
    Q = np.zeros((3,3))

    nu21 = nu12*E2/E1
    fact = 1.0/(1.0 - nu12*nu21)

    # Assign the values to Q
    Q[0,0] = fact*E1
    Q[0,1] = fact*E2*nu12
    
    Q[1,0] = fact*E1*nu21
    Q[1,1] = fact*E2

    Q[2,2] = G12

    return Q

def get_global_stiffness(E1, E2, nu12, G12, thetas):
    '''
    Compute the stiffness matrices for each of the given angles in the
    global coordinate frame.
    '''

    # Get the stiffness in the material frame
    Q = get_stiffness(E1, E2, nu12, G12)

    # Allocate the Cmat array of matrices
    Cmats = np.zeros((len(thetas), 3, 3))

    # Compute the transformed stiffness for each angle
    for i in range(len(thetas)):
        Tinv = get_transform(thetas[i])
        
        # Compute the Qbar matrix
        Cmats[i,:,:] = np.dot(Tinv, np.dot(Q, Tinv.T))

    return Cmats

class TACSAnalysis(ParOpt.pyParOptProblem):
    def __init__(self, comm, props, tacs, const, num_materials,
                 xpts=None, conn=None, m_fixed=1.0, qt=5.0,
                 min_mat_fraction=-1.0):
        '''
        Analysis problem
        '''

        # Keep the communicator
        self.comm = comm

        # Set the TACS object and the constitutive list
        self.props = props
        self.tacs = tacs
        self.const = const

        # Set the material information
        self.num_materials = num_materials
        self.num_elements = self.tacs.getNumElements()

        # Keep the pointer to the connectivity/positions
        self.xpts = xpts
        self.conn = conn

        # Set the target fixed mass
        self.m_fixed = m_fixed

        # Set a fixed material fraction
        self.min_mat_fraction = min_mat_fraction

        # Set the number of constraints
        self.ncon = 1
        if self.min_mat_fraction > 0.0:
            self.ncon += self.num_materials

        # Set the number of design variables
        nwblock = 1
        self.num_design_vars = (self.num_materials+1)*self.num_elements

        # Initialize the super class
        super(TACSAnalysis, self).__init__(comm,
                                           self.num_design_vars, self.ncon,
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

        # Allocate a vector that stores the gradient of the mass
        self.gmass = np.zeros(self.num_design_vars)

        # Create the mass function and evaluate the gradient of the
        # mass. This is assumed to remain constatnt throughout the
        # optimization.
        self.mass_func = functions.StructuralMass(self.tacs)
        self.tacs.evalDVSens(self.mass_func, self.gmass)

        # Set the initial variable values
        self.xinit = np.ones(self.num_design_vars)

        # Set the initial design variable values
        xi = self.m_fixed/np.dot(self.gmass, self.xinit)
        self.xinit[:] = xi

        # Set the penalization
        self.qt = qt
        tval = xi*self.num_materials
        # self.xinit[::self.nblock] = tval/(1.0 + self.qt*(1 - tval))
        self.xinit[::self.nblock] = (self.qt + 1)*tval/(1.0 + self.qt*tval)

        # Create a temporary vector for the hessian-vector products
        self.hvec_tmp = np.zeros(self.xinit.shape)

        # Set the initial linearization/point
        self.RAMP_penalty = 0.0
        self.setNewInitPointPenalty(self.xinit)

        # Set the number of function/gradient/hessian-vector
        # evaluations to zero
        self.fevals = 0
        self.gevals = 0
        self.hevals = 0

        # Evaluate the objective at the initial point
        self.obj_scale = 1.0
        fail, obj, con = self.evalObjCon(self.xinit)
        self.obj_scale = 10.0*obj

        print('objective scaling = ', self.obj_scale)

        # Create the FH5 file object
        flag = (TACS.ToFH5.NODES |
                TACS.ToFH5.DISPLACEMENTS |
                TACS.ToFH5.STRAINS)
        self.f5 = TACS.ToFH5(self.tacs, TACS.PY_PLANE_STRESS, flag)

        return
        
    def setNewInitPointPenalty(self, x):
        '''
        Set the linearized penalty function, given the design variable
        values from the previous iteration and the penalty parameters
        '''

        # Set the linearization point
        self.xinit[:] = x[:]      

        # For each constitutive point, set the new linearization point
        self.props.setPenalization(self.RAMP_penalty)
        for con in self.const:
            con.setLinearization(self.xinit)
        return

    def getDiscreteInfeas(self, x):
        '''
        Compute the discrete infeasibility measure at a given design point
        '''
        d = np.zeros(self.num_elements)
        for i in range(self.num_elements):
            tnum = self.nblock*i
            d[i] = 1.0 - (x[tnum] - 1.0)**2 - np.sum(x[tnum+1:tnum+self.nblock]**2)
        return d

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the design variable values and the bounds'''
        q = self.props.getPenalization()
        x[:] = self.xinit[:]

        # Set the upper bound for the material/thickness variables
        ub[:] = 1.0

        penalty, ptype = self.props.getPenaltyType()
        if penalty == 'convex':
            if ptype == 'ramp':
                # Set the lower bound for the design variables
                lb[:] = (q/(q + 1.0))*self.xinit[:]**2

                # Set the lower bound for the thickness variables
                t0 = self.xinit[::self.nblock]
                lb[::self.nblock] = (self.qt/(self.qt + 1.0))*t0**2
            else:
                lb[:] = (2.0/3.0)*self.xinit[:]

        return
        
    def evalSparseCon(self, x, con):
        '''Evaluate the sparse constraints'''
        n = self.nblock*self.num_elements
        t = x[:n:self.nblock]
        xmat = x[:n].reshape(-1, self.nblock)[:,1:]

        # Get the penalty type
        penalty, ptype = self.props.getPenaltyType()

        if penalty == 'convex':
            t0 = self.xinit[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t0))
            con[:] = (a*t0 + (self.qt+1)*a*a*(t - t0) - 
                      np.sum(xmat, axis=1))
        else:
            a = 1.0/(1.0 + self.qt*(1.0 - t))
            con[:] = (self.qt+1)*a*t - np.sum(xmat, axis=1)

        return

    def addSparseJacobian(self, alpha, x, px, con):
        '''Compute the Jacobian-vector product con = alpha*J(x)*px'''
        n = self.nblock*self.num_elements
        pt = px[:n:self.nblock]
        pmat = px[:n].reshape(-1, self.nblock)[:,1:]

        # Get the penalty type
        penalty, ptype = self.props.getPenaltyType()

        if penalty == 'convex':
            t0 = self.xinit[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t0))
            con[:] += alpha*((self.qt + 1.0)*a*a*pt - np.sum(pmat, axis=1))
        else:
            t = x[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t))
            con[:] += alpha*((self.qt + 1.0)*a*a*pt - np.sum(pmat, axis=1))

        return

    def addSparseJacobianTranspose(self, alpha, x, pz, out):
        '''Compute the transpose Jacobian-vector product alpha*J^{T}*pz'''
        n = self.nblock*self.num_elements
        for k in range(1, self.nblock):
            out[k:n:self.nblock] -= alpha*pz[:]

        # Get the penalty type
        penalty, ptype = self.props.getPenaltyType()

        if penalty == 'convex':
            t0 = self.xinit[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t0))
            out[:n:self.nblock] += alpha*(self.qt + 1.0)*a*a*pz[:]
        else:
            t = x[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t))
            out[:n:self.nblock] += alpha*(self.qt + 1.0)*a*a*pz[:]

        return

    def addSparseInnerProduct(self, alpha, x, c, A):
        '''Add the results from the product J(x)*C*J(x)^{T} to A'''
        n = self.nblock*self.num_elements
        cmat = c[:n].reshape(-1, self.nblock)[:,1:]
        A[:] += alpha*np.sum(cmat, axis=1)

        # Get the penalty type
        penalty, ptype = self.props.getPenaltyType()

        if penalty == 'convex':
            t0 = self.xinit[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t0))
            A[:] += alpha*c[:n:self.nblock]*((self.qt + 1.0)*a*a)**2
        else:
            t = x[:n:self.nblock]
            a = 1.0/(1.0 + self.qt*(1.0 - t))
            A[:] += alpha*c[:n:self.nblock]*((self.qt + 1.0)*a*a)**2

        return

    def getCompliance(self, x):
        '''Compute the full objective'''

        # Set the design variable values
        self.tacs.setDesignVars(x[:])
        self.setNewInitPointPenalty(x[:])

        # Assemble the Jacobian
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, self.res, self.mat)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        # Compute the compliance objective
        compliance = self.u.dot(self.res)

        # Set the variables
        self.u.scale(-1.0)
        self.tacs.setVariables(self.u)

        return compliance

    def getMass(self, x):
        '''Return the mass'''
        return np.dot(self.gmass, x[:])

    def evalObjCon(self, x):
        '''
        Evaluate the objective (compliance) and constraint (mass)
        '''
        # Add the number of function evaluations
        self.fevals += 1

        # Set the design variable values
        self.tacs.setDesignVars(x[:])

        # Assemble the Jacobian
        self.tacs.zeroVariables()
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, self.res, self.mat)
        self.pc.factor()
        self.ksm.solve(self.res, self.u)

        # Compute the compliance objective
        obj = self.u.dot(self.res)

        # Scale the variables and set the state variables
        self.u.scale(-1.0)
        self.tacs.setVariables(self.u)
        
        # Scale the compliance objective
        obj = obj/self.obj_scale
                    
        # Compute the mass
        mass = np.dot(self.gmass, x)

        # Create the constraint c(x) >= 0.0 for the mass
        if self.min_mat_fraction > 0.0:
            con = np.zeros(self.ncon)
            con[0] = 1.0 - mass/self.m_fixed

            # Compute the mass fraction constraints
            n = self.nblock*self.num_elements
            for i in range(self.num_materials):
                con[i+1] = np.sum(x[1+i:n:self.nwblock])/self.num_elements 
                con[i+1] -= self.min_mat_fraction
        else:
            con = np.array([1.0 - mass/self.m_fixed])

        fail = 0
        return fail, obj, con

    def evalObjConGradient(self, x, gobj, Acon):
        '''
        Evaluate the derivative of the compliance and mass
        '''
        
        # Add the number of gradient evaluations
        self.gevals += 1

        # Evaluate the derivative
        g = np.zeros(len(gobj))
        self.tacs.evalAdjointResProduct(self.u, g)

        # Scale the objective gradient
        gobj[:] = -(1.0/self.obj_scale)*g

        # Add the contribution to the constraint
        Acon[0][:] = -self.gmass/self.m_fixed

        if self.min_mat_fraction > 0.0:
            n = self.nblock*self.num_elements
            for i in range(self.num_materials):
                Acon[i+1][:] = 0.0
                Acon[i+1][1+i:n:self.nwblock] += 1.0/self.num_elements 

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
        
        # Assemble the residual
        self.hvec_tmp[:] = 0.0
        multitopo.assembleProjectDVSens(self.tacs, px[:],
                                        self.hvec_tmp, self.res)
        hvec[:] = self.hvec_tmp

        # Solve K(x)*psi = res
        self.ksm.solve(self.res, self.psi)

        # Evaluate the adjoint-residual product
        self.tacs.evalAdjointResProduct(self.psi, self.hvec_tmp)
        hvec[:] += 2.0*self.hvec_tmp

        # Scale the result to the correct range
        hvec[:] /= self.obj_scale

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
            tikz += '\\begin{figure}\n\\begin{tikzpicture}'
            tikz += '[x=0.25cm, y=0.25cm]\n'
            tikz += '\\sffamily\n'

            grey = [225, 225, 225]
            
            if self.num_materials <= 3:
                # Write out a tikz file
                rgbvals = [(44, 160, 44),
                           (255, 127, 14),
                           (31, 119, 180)]
                
                for i in range(self.conn.shape[0]):
                    # Determine the color to use
                    xf = self.const[i].getFilteredDesignVars()
                    kmax = np.argmax(xf[1:])

                    u = xf[0]/(1.0 + self.qt*(1.0 - xf[0]))
                    r = (1.0 - u)*grey[0] + u*rgbvals[kmax][0]
                    g = (1.0 - u)*grey[1] + u*rgbvals[kmax][1]
                    b = (1.0 - u)*grey[2] + u*rgbvals[kmax][2]

                    tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}\n'%(
                        int(r), int(g), int(b))
                    tikz += '\\fill[mycolor] (%f,%f) -- (%f,%f)'%(
                        self.xpts[self.conn[i,0],0], 
                        self.xpts[self.conn[i,0],1],
                        self.xpts[self.conn[i,1],0], 
                        self.xpts[self.conn[i,1],1])
                    tikz += '-- (%f,%f) -- (%f,%f) -- cycle;\n'%(
                        self.xpts[self.conn[i,3],0], 
                        self.xpts[self.conn[i,3],1],
                        self.xpts[self.conn[i,2],0], 
                        self.xpts[self.conn[i,2],1])
            else:
                # Compute the angles
                thetas = np.linspace(-90, 90, self.num_materials+1)[1:]
                thetas *= (np.pi/180.0)

                for i in range(self.conn.shape[0]):
                    xf = self.const[i].getFilteredDesignVars()

                    rgb = (44, 160, 44)
                    u = xf[0]/(1.0 + self.qt*(1.0 - xf[0]))
                    r = (1.0 - u)*grey[0] + u*rgb[0]
                    g = (1.0 - u)*grey[1] + u*rgb[1]
                    b = (1.0 - u)*grey[2] + u*rgb[2]

                    tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}\n'%(
                        int(r), int(g), int(b))
                    tikz += '\\fill[mycolor] (%f,%f) -- (%f,%f)'%(
                        self.xpts[self.conn[i,0],0], 
                        self.xpts[self.conn[i,0],1],
                        self.xpts[self.conn[i,1],0], 
                        self.xpts[self.conn[i,1],1])
                    tikz += '-- (%f,%f) -- (%f,%f) -- cycle;\n'%(
                        self.xpts[self.conn[i,3],0], 
                        self.xpts[self.conn[i,3],1],
                        self.xpts[self.conn[i,2],0], 
                        self.xpts[self.conn[i,2],1])

                    dx = (self.xpts[self.conn[i,1],0] - 
                          self.xpts[self.conn[i,0],0])
                    dy = (self.xpts[self.conn[i,1],1] - 
                          self.xpts[self.conn[i,0],1])
                    l = np.sqrt(dx*dx + dy*dy)

                    # Compute the line width and scale the dimension of
                    width = 0.3*l
                    l *= 0.85

                    xav = 0.0
                    yav = 0.0
                    for j in range(4):
                        xav += 0.25*self.xpts[self.conn[i,j],0]
                        yav += 0.25*self.xpts[self.conn[i,j],1]

                    # Now, draw the angle to use
                    for j in range(len(xf)-1):
                        if xf[1+j] > 0.5:
                            c = np.cos(thetas[j])
                            s = np.sin(thetas[j])
                            tikz += '\\draw[line width=%.2fmm, color=black!%d]'%(
                                width, int(100*xf[0]))
                            tikz += ' (%f, %f) -- (%f, %f);\n'%(
                                xav - 0.5*l*c, yav - 0.5*l*s,
                                xav + 0.5*l*c, yav + 0.5*l*s)
                                                   
            tikz += '\\end{tikzpicture}\\end{figure}\\end{document}\n'
    
            # Write the solution file
            fp = open(filename, 'w')
            if fp:
                fp.write(tikz)
                fp.close()

        return

def rectangular_domain(nx, ny, Ly=100.0):

    # Set the y-dimension based on a unit aspect ratio
    Lx = (nx*Ly)/ny

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
    for j in range(ny+1):
        for i in range(nx+1):
            xpts[nodes[i,j],0] = x[i]
            xpts[nodes[i,j],1] = y[j]

    # Set the connectivity and create the corresponding elements
    conn = np.zeros((nelems, 4), dtype=np.intc)
    for j in range(ny):
        for i in range(nx):
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
    for j in range(int(ny/8)):
        num = nx-1 + nx*j
        aux.addElement(num, trac)
        
    return xpts, conn, bcs, aux, area

def create_structure(comm, props, xpts, conn, bcs, aux, m_fixed, 
                     r0=4, min_mat_fraction=-1.0):
    '''
    Create a structure with the speicified number of nodes along the
    x/y directions, respectively.
    '''
    
    # Set the number of design variables
    nnodes = len(xpts)
    nelems = len(conn)
    nmats = props.getNumMaterials()
    num_design_vars = (nmats + 1)*nelems

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
    for i in range(nelems):
        xcentroid[i] = 0.25*(xpts[conn[i,0]] + xpts[conn[i,1]] +
                             xpts[conn[i,2]] + xpts[conn[i,3]])    

    # Find the closest points
    locator = multitopo.Locator(xcentroid)
    max_points = 15
    for i in range(nelems):
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
    analysis = TACSAnalysis(comm, props, tacs, const, nmats,
                            conn=conn, xpts=xpts,
                            m_fixed=m_fixed, 
                            min_mat_fraction=min_mat_fraction)

    return analysis

def create_paropt(analysis, use_hessian=False,
                  max_qn_subspace=50, qn_type=ParOpt.BFGS):
    '''
    Optimize the given structure using ParOpt
    '''

    # Set the inequality options
    analysis.setInequalityOptions(dense_ineq=True, 
                                  sparse_ineq=False,
                                  use_lower=True,
                                  use_upper=True)
    
    # Create the optimizer
    opt = ParOpt.pyParOpt(analysis, max_qn_subspace, qn_type)

    # Set the optimality tolerance
    opt.setAbsOptimalityTol(1e-6)

    # Set the Hessian-vector product iterations
    if use_hessian:
        opt.setUseLineSearch(1)
        opt.setUseHvecProduct(1)
        opt.setGMRESSubspaceSize(100)
        opt.setNKSwitchTolerance(1.0)
        opt.setEisenstatWalkerParameters(0.5, 0.0)
        opt.setGMRESTolerances(0.1, 1e-30)
    else:
        opt.setUseHvecProduct(0)

    # Set the barrier strategy to use
    opt.setBarrierStrategy(ParOpt.MONOTONE)

    # Set the norm to use
    opt.setNormType(ParOpt.L1_NORM)

    # Set optimization parameters
    opt.setArmijoParam(1e-5)
    opt.setMaxMajorIterations(5000)

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
            self.xcurr = np.array(x['x'])
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
                self.options['Major optimality tolerance'] = 1e-5
                
            elif self.optimizer == 'ipopt':                
                self.options['print_user_options'] = 'yes'
                self.options['tol'] = 1e-5
                self.options['nlp_scaling_method'] = 'none'
                self.options['limited_memory_max_history'] = 25
                self.options['bound_relax_factor'] = 0.0
                self.options['linear_solver'] = 'ma27'
                self.options['output_file'] = fname
                self.options['max_iter'] = 10000
            return

        def setInitBarrierParameter(self, *args):
            return

        def getOptimizedPoint(self):
            return self.xcurr
        
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
    for i in range(analysis.num_elements):
        data.append(1.0)
        cols.append(i*nblock)
        for j in range(i*nblock+1, (i+1)*nblock):
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
                          parameter=5.0, max_iters=1000,
                          optimizer='paropt', case='isotropic',
                          use_hessian=False, start_strategy='point',
                          ptype='ramp', final_full_opt=False):
    # Optimize the structure
    optimizer = optimizer.lower()
    penalization = ptype.upper()
    heuristic = '%s%.0f_%s_%s'%(penalization, parameter,
                                case, start_strategy)
    prefix = os.path.join(root_dir, optimizer, '%dx%d'%(nx, ny), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Write out the stdout output to a file
    sys.stdout = open(os.path.join(prefix, 'stdout.out'), 'w')
       
    # Set up the optimization problem in ParOpt
    opt = create_paropt(analysis,
                        use_hessian=use_hessian,
                        qn_type=ParOpt.BFGS)

    # Log the optimization file
    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, "compliance" '
    s += '"min d", "max d", "tau", '
    s += 'feval, geval, hvec, time, "kkt error infty", "kkt error l2"\n'
    s += 'Zone T = %s\n'%(heuristic)
    fp.write(s)

    # Set the penalty parameter
    analysis.RAMP_penalty = parameter

    # Set the penalty parameter
    if start_strategy == 'convex':
        # Set the optimality tolerance
        analysis.RAMP_penalty = 0.0
        analysis.props.setPenaltyType(penalty='convex', ptype='ramp')
        analysis.setNewInitPointPenalty(analysis.xinit)        
    elif start_strategy == 'uniform':
        analysis.xinit[:] = 1.0/analysis.num_materials
        analysis.xinit[::(analysis.num_materials+1)] = 1.0
        analysis.props.setPenaltyType(penalty='convex', ptype=ptype)
        analysis.setNewInitPointPenalty(analysis.xinit)        
    else:
        # Set the initial starting point strategy
        analysis.props.setPenaltyType(penalty='convex', ptype=ptype)
        analysis.setNewInitPointPenalty(analysis.xinit)

    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Keep track of the number of iterations
    niters = 0

    for k in range(max_iters):
        # Set the output file to use
        fname = os.path.join(prefix, 'history_iter%d.out'%(niters)) 
        opt.setOutputFile(fname)

        # Optimize
        if k > 0 and optimizer == 'paropt':
            opt.setInitStartingPoint(0)

            # Reset the design variable and bounds
            opt.resetDesignAndBounds()

            # Get the new complementarity
            mu = opt.getComplementarity()
            opt.setInitBarrierParameter(mu)

        # Optimize the new point
        opt.optimize()

        # Get the optimized point
        x, z, zw, zl, zu = opt.getOptimizedPoint()

        # Get the discrete infeasibility measure
        d = analysis.getDiscreteInfeas(x)

        # Compute the discrete infeasibility measure
        tau = np.sum(d)

        # Print the output to the file
        if k % 10 == 0:
            # Print out the design
            filename = 'opt_struct_iter%d.tex'%(niters)
            output = os.path.join(prefix, filename)
            analysis.writeTikzFile(x, output)

        # Get the compliance and objective values
        analysis.RAMP_penalty = parameter
        analysis.setNewInitPointPenalty(x)       
        comp = analysis.getCompliance(x)

        # Compute the KKT error based on the original problem
        x, z, zw, zl, zu = opt.getOptimizedPoint()

        # Evaluate the objective and constraints
        gobj1 = analysis.createDesignVec()
        Acon1 = []
        for i in range(len(z)):
            Acon1.append(analysis.createDesignVec())
        product = analysis.createDesignVec()
        fail, obj1, con1 = analysis.evalObjCon(x)
        analysis.evalObjConGradient(x, gobj1, Acon1)
        analysis.addSparseJacobianTranspose(-1.0, x, zw, product)

        # Compute the KKT error
        kkt = gobj1[:] + product[:] - zl[:] + zu[:]
        for i in range(len(z)):
            kkt[:] -= Acon1[i][:]*z[i]

        # Compute the maximum error contribution
        kkt_max_err = np.amax(np.fabs(kkt))
        kkt_l1_err = np.sum(np.fabs(kkt))
        kkt_l2_err = np.sqrt(np.dot(kkt, kkt))

        g_max = np.amax(np.fabs(gobj1))
        g_l1 = np.sum(np.fabs(gobj1))
        g_l2 = np.sqrt(np.dot(gobj1, gobj1))

        # Print out the iteration information to the screen
        if k % 10 == 0:
            print('%4s %10s %10s %10s %10s %10s %10s %10s'%(
                'Iter', 'tau', 'KKT infty', 'KKT l1', 'KKT l2', 
                'Rel infty', 'Rel l1', 'Rel l2'))
        print('%4d %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e'%(
            niters, np.sum(d), kkt_max_err, kkt_l1_err, kkt_l2_err, 
            kkt_max_err/g_max, kkt_l1_err/g_l1, kkt_l2_err/g_l2))

        # Write out the data to the log file
        s = '%d %e %e %e %e '%(niters, comp, np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e %e %e\n'%(
            analysis.fevals, analysis.gevals, analysis.hevals,
            MPI.Wtime() - init_time, kkt_max_err, kkt_l2_err)
        fp.write(s)

        # Flush the file/stdout
        fp.flush()
        sys.stdout.flush()
        
        # Increase the iteration counter
        niters += 1        

        # Quit when the relative KKT error is less than 10^{-4}
        if kkt_l1_err/g_l1 < 1e-3:
            break        

    # Print out the design
    filename = 'opt_struct_iter%d.tex'%(niters)
    output = os.path.join(prefix, filename)
    analysis.writeTikzFile(x, output)

    if final_full_opt:
        # Set the penalty parameter
        analysis.RAMP_penalty = parameter
        analysis.props.setPenaltyType('full', ptype=ptype)

        # Get the new complementarity
        opt.resetDesignAndBounds()
        mu = opt.getComplementarity()
        opt.setInitBarrierParameter(mu)
        opt.setUseHvecProduct(0)
        opt.setUseLineSearch(1)

        # Set the output file to use
        fname = os.path.join(prefix, 'history_iter%d.out'%(niters)) 
        opt.setOutputFile(fname)
    
        # Optimize the new point
        opt.optimize()
    
        # Compute the KKT error based on the original problem
        x, z, zw, zl, zu = opt.getOptimizedPoint()
        
        # Get the discrete infeasibility measure
        d = analysis.getDiscreteInfeas(x)
        
        # Get the compliance and objective values
        analysis.RAMP_penalty = parameter
        analysis.setNewInitPointPenalty(x)       
        comp = analysis.getCompliance(x)

        # Evaluate the objective and constraints
        gobj1 = analysis.createDesignVec()
        Acon1 = []
        for i in range(len(z)):
            Acon1.append(analysis.createDesignVec())
        product = analysis.createDesignVec()
        fail, obj1, con1 = analysis.evalObjCon(x)
        analysis.evalObjConGradient(x, gobj1, Acon1)
        analysis.addSparseJacobianTranspose(-1.0, x, zw, product)

        # Compute the KKT error
        kkt = gobj1[:] + product[:] - zl[:] + zu[:]
        for i in range(len(z)):
            kkt[:] -= Acon1[i][:]*z[i]

        # Compute the maximum error contribution
        kkt_max_err = np.amax(np.fabs(kkt))
        kkt_l2_err = np.sqrt(np.dot(kkt, kkt))

        g_max = np.amax(np.fabs(gobj1))
        g_l2 = np.sqrt(np.dot(gobj1, gobj1))
        
        # Print out the iteration information to the screen
        print('%4s %10s %10s %10s %10s %10s'%(
            'Iter', 'tau', 'KKT infty', 'KKT l2', 'Rel infty', 'Rel l2'))
        print('%4d %10.4e %10.4e %10.4e %10.4e %10.4e'%(
            niters, np.sum(d), kkt_max_err, kkt_l2_err, 
            kkt_max_err/g_max, kkt_l2_err/g_l2))

        s = '%d %e %e %e %e '%(niters, comp, np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e %e %e\n'%(
            analysis.fevals, analysis.gevals, analysis.hevals,
            MPI.Wtime() - init_time, kkt_max_err, kkt_l2_err)
        fp.write(s)
    
    # Close the log file
    fp.close()

    # Get the final, optimzied point
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    # Print out the design variables
    filename = 'final_opt_struct.tex'
    output = os.path.join(prefix, filename)
    analysis.writeTikzFile(x, output)

    # Save the final optimized point
    fname = os.path.join(prefix, 'x_opt.dat')
    np.savetxt(fname, x)

    # Write out the optimization    
    filename = 'final_opt_struct.f5'
    output = os.path.join(prefix, filename)
    analysis.writeOutput(output)

    return

def optimize_plane_stress_full(comm, analysis, root_dir='results',
                               parameter=5.0, optimizer='paropt', 
                               case='isotropic',
                               use_hessian=False, start_strategy='point',
                               ptype='ramp'):
    # Optimize the structure
    optimizer = optimizer.lower()
    penalization = ptype.upper()
    heuristic = '%s%.0f_%s_%s'%(penalization, parameter,
                                case, start_strategy)
    prefix = os.path.join(root_dir, optimizer + '_full',
                          '%dx%d'%(nx, ny), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
       
    # Set up the optimization problem in ParOpt
    if optimizer == 'paropt':
        opt = create_paropt(analysis, use_hessian=use_hessian,
                            qn_type=ParOpt.BFGS)
    opt.setBarrierStrategy(ParOpt.MONOTONE)
    opt.setInitBarrierParameter(10.0)

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
    analysis.props.setPenaltyType('full', ptype=ptype)

    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Set the penalty parameter
    iteration = 0
    if start_strategy == 'convex':
        # Set the optimality tolerance
        analysis.RAMP_penalty = 0.0
        analysis.props.setPenaltyType('convex', ptype=ptype)

        if optimizer != 'paropt':
            opt = create_pyopt(analysis, optimizer=optimizer)
        
        # Set the output file to use
        fname = os.path.join(prefix, 'history_iter%d.out'%(iteration)) 
        opt.setOutputFile(fname)
        
        # Optimize
        opt.optimize()

        # Get the optimized point
        x, z, zw, zl, zu = opt.getOptimizedPoint()

        # Make sure that all of the variables are strictly positive
        for i in range(len(x)):
            x[i] = max(0.0, x[i])
        
        # Get the discrete infeasibility measure
        d = analysis.getDiscreteInfeas(x)
        
        # Compute the discrete infeasibility measure
        tau = np.sum(d)
        
        # Get the compliance and objective values
        comp = analysis.getCompliance(x)
        
        s = '%d %e %e %e %e '%(iteration, comp, np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e\n'%(
            analysis.fevals, analysis.gevals, analysis.hevals, 
            MPI.Wtime() - init_time)
        fp.write(s)

        iteration += 1
        analysis.RAMP_penalty = parameter
        analysis.props.setPenaltyType('full', ptype=ptype)
        analysis.setNewInitPointPenalty(x)

        # Print out the design variables
        filename = 'opt_struct_iter%d.tex'%(iteration)
        output = os.path.join(prefix, filename)
        analysis.writeTikzFile(x, output)
    elif start_strategy == 'uniform':
        analysis.xinit[:] = 1.0/analysis.num_materials
        analysis.xinit[::(analysis.num_materials+1)] = 1.0

    analysis.setNewInitPointPenalty(analysis.xinit)

    if optimizer != 'paropt':
        opt = create_pyopt(analysis, optimizer=optimizer)
    else:
        opt = create_paropt(analysis,
                            use_hessian=use_hessian,
                            qn_type=ParOpt.BFGS,
                            max_qn_subspace=2)
        opt.setUseHvecProduct(0)
        opt.setUseLineSearch(1)
        opt.setBarrierStrategy(ParOpt.MEHROTRA)
        opt.setHessianResetFreq(1000)
        opt.setMaxMajorIterations(500)

    # Set the output file to use
    fname = os.path.join(prefix, 'history_iter%d.out'%(iteration)) 
    opt.setOutputFile(fname)

    # Optimize
    opt.optimize()

    # Get the optimized point
    x, z, zw, zl, zu = opt.getOptimizedPoint()

    # Get the discrete infeasibility measure
    d = analysis.getDiscreteInfeas(x)
    
    # Compute the discrete infeasibility measure
    tau = np.sum(d)

    # Get the compliance and objective values
    comp = analysis.getCompliance(x)

    s = '%d %e %e %e %e '%(iteration, comp, np.min(d), np.max(d), np.sum(d))
    s += '%d %d %d %e\n'%(
        analysis.fevals, analysis.gevals, analysis.hevals, 
        MPI.Wtime() - init_time)
    fp.write(s)
    fp.close()

    # Print out the design variables
    filename = 'final_opt_struct.tex'
    output = os.path.join(prefix, filename)
    analysis.writeTikzFile(x, output)

    # Save the final optimized point
    fname = os.path.join(prefix, 'x_opt.dat')
    np.savetxt(fname, x[:])

    return

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--start_strategy', type=str, default='point',
                    help='start up strategy to use')
parser.add_argument('--nx', type=int, default=48,
                    help='Nodes in x-direction')
parser.add_argument('--ny', type=int, default=48, 
                    help='Nodes in y-direction')
parser.add_argument('--parameter', type=float, default=5.0,
                    help='Penalization parameter')
parser.add_argument('--ptype', type=str, default='ramp',
                    help='Penalty type')
parser.add_argument('--optimizer', type=str, default='paropt',
                    help='Optimizer name')
parser.add_argument('--full_penalty', default=False, action='store_true',
                    help='Use the full RAMP penalization')
parser.add_argument('--use_hessian', default=False, action='store_true',
                    help='Use hessian-vector products')
parser.add_argument('--case', type=str, default='isotropic',
                    help='Case name')
parser.add_argument('--root_dir', type=str, default='results',
                    help='Root directory')
args = parser.parse_args()

# Get the arguments
nx = args.nx
ny = args.ny
parameter = args.parameter
ptype = args.ptype
optimizer = args.optimizer
use_hessian = args.use_hessian
start_strategy = args.start_strategy
root_dir = args.root_dir

# Make sure everything is in lower case
ptype = ptype.lower()

# The MPI communicator
comm = MPI.COMM_WORLD

# Create the connectivity data
xpts, conn, bcs, aux, area = rectangular_domain(nx, ny)

# Set the material properties for the isotropic case
if args.case == 'isotropic':
    rho = np.array([0.85, 1.0, 1.2])
    E = 70e3*np.array([0.85, 1.0, 1.15])
    nu = np.array([0.2,  0.3, 0.3])

    print('mat # %15s %15s %15s'%('E/rho', 'G/rho', 'E/G'))
    for i in range(len(rho)):
        G = 0.5*E[i]/(1.0 + nu[i])
        print('mat %d %15g %15g %15g'%(i+1, E[i]/rho[i], G/rho[i], E[i]/G))
            
    C = np.zeros((len(rho), 6))
    
    for i in range(len(rho)):
        D = E[i]/(1.0 - nu[i]**2)
        G = 0.5*E[i]/(1.0 + nu[i])
        C[i,0] = D
        C[i,1] = nu[i]*D
        C[i,3] = D
        C[i,5] = G

    # Compute the fixed mass fraction
    m_fixed = 0.4*area*rho[2]
else:
    # These properties are taken from Jones, pg. 101 for a
    # graphite-epoxy material. Note that units are in MPa.
    E1 = 207e3
    E2 = 5e3
    nu12 = 0.25
    G12 = 2.6e3

    # The density of the material
    rho_mat = 1.265

    # Set the angles
    nmats = 12
    thetas = (np.pi/180.0)*np.linspace(-90, 90, nmats+1)[1:]

    # Create the Cmat matrices
    Cmats = get_global_stiffness(E1, E2, nu12, G12, thetas)

    # Copy out the stiffness properties
    C = np.zeros((nmats, 6))
    rho = rho_mat*np.ones(nmats)
    for i in range(nmats):
        C[i,0] = Cmats[i,0,0]
        C[i,1] = Cmats[i,0,1]
        C[i,2] = Cmats[i,0,2]
        C[i,3] = Cmats[i,1,1]
        C[i,4] = Cmats[i,1,2]
        C[i,5] = Cmats[i,2,2]

    m_fixed = 0.4*area*rho_mat
        
# Create the material properties
props = multitopo.MultiTopoProperties(rho, C)
props.setPenalization(parameter)

# Create the analysis object
r0 = np.sqrt(4.99)*np.sqrt(area/len(conn))
print('r0 = ', r0)

min_mat_fraction = -1.0
if args.case is 'isotropic':
    min_mat_fraction = 0.1

analysis = create_structure(comm, props, xpts, conn,
                            bcs, aux, m_fixed, r0=r0,
                            min_mat_fraction=min_mat_fraction)

if args.full_penalty:
    # Optimize the plane stress problem
    optimize_plane_stress_full(comm, analysis, root_dir=root_dir,
                               parameter=parameter, optimizer=optimizer,
                               start_strategy=start_strategy,
                               use_hessian=use_hessian,
                               case=args.case, ptype=ptype)
else:
    # Optimize the plane stress problem
    optimize_plane_stress(comm, analysis, root_dir=root_dir,
                          parameter=parameter, optimizer=optimizer,
                          start_strategy=start_strategy,
                          use_hessian=use_hessian,
                          case=args.case, ptype=ptype)
