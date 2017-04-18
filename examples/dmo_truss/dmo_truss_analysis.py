# Import numpy 
import numpy as np
import scipy.linalg as linalg

# Import parts of matplotlib for plotting
import matplotlib.pyplot as plt

# Import MPI
from mpi4py import MPI

# Import ParOpt
from paropt import ParOpt

class TrussAnalysis(ParOpt.pyParOptProblem):
    def __init__(self, conn, xpos, loads, bcs, 
                 E, rho, Avals, m_fixed,
                 x_lb=0.0, epsilon=1e-12, no_bound=1e30):
        '''
        Analysis problem for mass-constrained compliance minimization
        '''

        # Store pointer to the data
        self.conn = conn
        self.xpos = xpos
        self.loads = loads
        self.bcs = bcs

        # Set the value of the Young's modulus
        self.E = E
        
        # Set the values of the areas -- all must be non-zero
        self.Avals = np.array(Avals)
        self.rho = rho

        # Fixed mass value
        self.m_fixed = m_fixed

        # Set the factor on the lowest value of the thickness
        # This avoids
        self.epsilon = epsilon

        # Set the bound for no variable bound
        self.no_bound = no_bound

        # Keep a vector that stores the element areas
        self.A = np.zeros(len(self.conn))

        # Set the sizes for the problem
        self.nmats = len(self.Avals)
        self.nblock = self.nmats+1
        self.nelems = len(self.conn)
        self.nvars = len(self.xpos)

        # Initialize the super class
        ncon = 1
        ndv = self.nblock*self.nelems
        nwcon = self.nelems
        nwblock = 1
        super(TrussAnalysis, self).__init__(MPI.COMM_SELF,
                                            ndv, ncon, nwcon, nwblock)

        # Allocate a vector that stores the gradient of the mass
        self.gmass = np.zeros(ndv)

        # Allocate the matrices required
        self.K = np.zeros((self.nvars, self.nvars))
        self.Kp = np.zeros((self.nvars, self.nvars))
        self.f = np.zeros(self.nvars)
        self.u = np.zeros(self.nvars)
        self.phi = np.zeros(self.nvars)
        
        # Set the scaling of the objective
        self.obj_scale = None

        # Keep track of the different counts
        self.fevals = 0
        self.gevals = 0
        self.hevals = 0

        # Allocate the matrices
        self.Ke = np.zeros((len(self.conn), 4, 4))

        # Compute the gradient of the mass for each bar in the mesh
        index = 0
        for bar in self.conn:
            # Get the first and second node numbers from the bar
            n1 = bar[0]
            n2 = bar[1]

            # Compute the nodal locations
            xd = self.xpos[2*n2] - self.xpos[2*n1]
            yd = self.xpos[2*n2+1] - self.xpos[2*n1+1]
            Le = np.sqrt(xd**2 + yd**2)
            C = xd/Le
            S = yd/Le

            # Compute the element stiffness matrix
            self.Ke[index,:,:] = (self.E/Le)*np.array(
                [[C**2, C*S, -C**2, -C*S],
                 [C*S, S**2, -C*S, -S**2],
                 [-C**2, -C*S, C**2, C*S],
                 [-C*S, -S**2, C*S, S**2]])

            # Compute the gradient of the mass
            for j in xrange(self.nmats):
                self.gmass[self.nblock*index+1+j] += self.rho[j]*Le

            index += 1

        # Set the fixed mass
        self.m_fixed = m_fixed
        max_mass = np.sum(self.gmass)
        xi = self.m_fixed/max_mass

        # Set the initial design variable values
        self.xinit = np.zeros(ndv)
        self.xinit[:] = xi/self.nmats
        self.xinit[::self.nblock] = xi

        # Set the initial linearization
        self.penalization = None
        self.SIMP = 1.0
        self.RAMP = 0.0
        self.xconst = np.array(self.xinit)
        self.xlinear = np.ones(ndv)

        # Set the lower bounds on the variables
        self.x_lb = max(x_lb, 0.0)

        return

    def setNewInitPointPenalty(self, x):
        '''
        Set the linearized penalty function, given the design variable
        values from the previous iteration and the penalty parameters
        '''

        # Set the new initial design variable values
        self.xinit[:] = x[:]

        if self.penalization == 'RAMP':
            # Compute the RAMP linearization terms
            for i in xrange(len(self.xconst)):
                self.xconst[i] = x[i]/(1.0 + self.RAMP*(1.0 - x[i]))
                self.xlinear[i] = (self.RAMP+1.0)/(1.0 + self.RAMP*(1.0 - x[i]))**2
        elif self.penalization == 'SIMP':
            # Compute the SIMP linearization terms
            self.xconst[:] = self.xinit**(self.SIMP)
            self.xlinear[:] = self.SIMP*self.xinit**(self.SIMP-1.0)
        else:
            self.xconst[:] = x[:]
            self.xlinear[:] = 1.0

        return

    def getDiscreteInfeas(self, x):
        '''
        Compute the discrete infeasibility measure at a given design point
        '''
        
        d = np.zeros(self.nelems)
        for i in xrange(self.nelems):
            tnum = self.nblock*i
            d[i] = 1.0 - (x[tnum] - 1.0)**2 - sum(x[tnum+1:tnum+self.nblock]**2)
            
        return d

    def computeLimitDesign(self, x):
        '''
        Compute the solution as the penalty approaches infinity
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

    def getCompliance(self, x, limit=False):
        '''Compute the strain energy in each bar'''

        if limit:
            self.u[:] = self.getLimitDisplacements(x)
        else:
            # Set the cross-sectional areas from the design variable
            # values
            self.setAreas(x, lb_factor=self.epsilon)

            # Evaluate compliance objective
            self.assembleMat(self.A, self.K)
            self.assembleLoadVec(self.f)
            self.applyBCs(self.K, self.f)
            
            # Copy the values
            self.u[:] = self.f[:]
            
            # Perform the Cholesky factorization
            self.L = linalg.cholesky(self.K, lower=True)
            
            # Solve the resulting linear system of equations
            linalg.solve_triangular(self.L, self.u, lower=True,
                                    trans='N', overwrite_b=True)
            linalg.solve_triangular(self.L, self.u, lower=True, 
                                    trans='T', overwrite_b=True)

        return np.dot(self.u, self.f)

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        
        # Set the variable values
        x[:] = self.xinit[:]

        # Set the bounds on the material selection variables
        if self.penalization == 'SIMP':
            lb[:] = self.xinit*(self.SIMP - 1.0)/self.SIMP
        elif self.penalization == 'RAMP':
            lb[:] = self.RAMP*self.xinit**2/(self.RAMP+1.0)
        else:
            lb[:] = 0.0
        ub[:] = self.no_bound

        # Set the bounds on the thickness variables
        lb[::self.nblock] = -self.no_bound
        ub[::self.nblock] = 1.0

        return

    def setAreas(self, x, lb_factor=0.0):
        '''Set the areas from the design variable values'''
        
        # Zero all the areas
        self.A[:] = self.Avals[0]*lb_factor

        # Add up the contributions to the areas from each 
        # discrete variable
        for i in xrange(len(self.conn)):
            for j in xrange(1, self.nblock):
                # Compute the value of the area variable
                val = (self.xconst[i*self.nblock+j] + 
                       self.xlinear[i*self.nblock+j]*(x[i*self.nblock+j] - 
                                                      self.xinit[i*self.nblock+j]))
                self.A[i] += self.Avals[j-1]*val

        return

    def setAreasLinear(self, px):
        '''Set the area as a linearization of the area'''
        
        self.A[:] = 0.0
                      
        # Add up the contributions to the areas from each 
        # discrete variable
        for i in xrange(len(self.conn)):
            for j in xrange(1, self.nblock):
                # Compute the value of the bar area
                val = self.xlinear[i*self.nblock+j]*px[i*self.nblock+j]
                self.A[i] += self.Avals[j-1]*val

        return

    def getMass(self, x):
        '''Return the mass of the truss'''
        return np.dot(self.gmass, x)

    def evalObjCon(self, x):
        '''
        Evaluate the objective (compliance) and constraint (mass)
        '''
        
        # Add the number of function evaluations
        self.fevals += 1

        # Set the cross-sectional areas from the design variable
        # values
        self.setAreas(x, lb_factor=self.epsilon)

        # Evaluate compliance objective
        self.assembleMat(self.A, self.K)
        self.assembleLoadVec(self.f)
        self.applyBCs(self.K, self.f)

        # Copy the values
        self.u[:] = self.f[:]

        # Perform the Cholesky factorization
        self.L = linalg.cholesky(self.K, lower=True)
            
        # Solve the resulting linear system of equations
        linalg.solve_triangular(self.L, self.u, lower=True,
                                trans='N', overwrite_b=True)
        linalg.solve_triangular(self.L, self.u, lower=True, 
                                trans='T', overwrite_b=True)

        # Compute the compliance objective
        obj = np.dot(self.u, self.f)
        if self.obj_scale is None:
            self.obj_scale = 1.0*obj

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

        # Set the areas from the design variable values
        self.setAreas(x, lb_factor=self.epsilon)

        # Zero the objecive and constraint gradients
        gobj[:] = 0.0

        # Set the number of materials
        nmats = len(self.Avals)+1
        
        # Add up the contribution to the gradient
        for i in xrange(len(self.conn)):
            # Get the first and second node numbers from the bar
            n1 = self.conn[i][0]
            n2 = self.conn[i][1]

            # Find the element variables
            ue = np.array([self.u[2*n1], self.u[2*n1+1],
                           self.u[2*n2], self.u[2*n2+1]])
            
            # Compute the inner product with the element stiffness matrix
            g = -np.dot(ue, np.dot(self.Ke[i,:,:], ue))  
            
            # Add the contribution to each derivative
            gobj[i*self.nblock+1:(i+1)*self.nblock] += g*(
                self.Avals*self.xlinear[i*self.nblock+1:(i+1)*self.nblock])

        # Scale the objective gradient
        gobj /= self.obj_scale

        # Add the contribution to the constraint
        Acon[0,:] = self.gmass[:]/self.m_fixed

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

        # Assemble the stiffness matrix along the px direction
        self.setAreasLinear(px)
        self.assembleMat(self.A, self.Kp)
        np.dot(self.Kp, self.u, out=self.phi)
        self.applyBCs(self.Kp, self.phi)

        # Solve the resulting linear system of equations
        linalg.solve_triangular(self.L, self.phi, lower=True,
                                trans='N', overwrite_b=True)
        linalg.solve_triangular(self.L, self.phi, lower=True, 
                                trans='T', overwrite_b=True)
        
        # Add up the contribution to the gradient
        for i in xrange(len(self.conn)):
            # Get the first and second node numbers from the bar
            n1 = self.conn[i][0]
            n2 = self.conn[i][1]

            # Find the element variables
            ue = np.array([self.u[2*n1], self.u[2*n1+1],
                           self.u[2*n2], self.u[2*n2+1]])
            phie = np.array([self.phi[2*n1], self.phi[2*n1+1],
                             self.phi[2*n2], self.phi[2*n2+1]])
            
            # Add the product to the derivative of the compliance
            h = 2.0*np.dot(phie, np.dot(self.Ke[i,:,:], ue))

            # Add the contribution to each derivative
            hvec[i*self.nblock+1:(i+1)*self.nblock] += h*(
                self.Avals*self.xlinear[i*self.nblock+1:(i+1)*self.nblock])

        # Evaluate the derivative
        hvec /= self.obj_scale

        fail = 0
        return fail

    def assembleMat(self, A, K):
        '''
        Given the connectivity, nodal locations and material properties,
        assemble the stiffness matrix
        
        input:
        A:   the bar areas

        output:
        K:   the stiffness matrix
        '''

        # Zero the stiffness matrix
        K[:,:] = 0.0

        # Loop over each element in the mesh
        index = 0
        for bar, A_bar in zip(self.conn, A):
            # Get the first and second node numbers from the bar
            n1 = bar[0]
            n2 = bar[1]

            # Create a list of the element variables for convenience
            elem_vars = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        
            # Add the element stiffness matrix to the global stiffness
            # matrix
            K[np.ix_(elem_vars, elem_vars)] += A_bar*self.Ke[index,:,:]

            index += 1
                    
        return

    def getLimitDisplacements(self, xinfty):
        '''
        Given the connectivity, nodal locations and material properties,
        assemble the stiffness matrix
        
        input:
        xinfty:  the limit design variables

        output:
        uinfty:  the displacements
        '''

        # Set the bar areas
        self.setAreas(xinfty, lb_factor=self.epsilon)

        # Zero the stiffness matrix
        K = np.zeros((self.nvars, self.nvars))
        uinfty = np.zeros(self.nvars)
        f = np.zeros(self.nvars)

        mark = np.zeros(self.nvars, dtype=np.int)

        # Loop over each element in the mesh
        index = 0
        for bar, A_bar in zip(self.conn, self.A):
            # Get the first and second node numbers from the bar
            n1 = bar[0]
            n2 = bar[1]
            
            # Create a list of the element variables for convenience
            elem_vars = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
                
            # Add the element stiffness matrix to the global stiffness
            # matrix
            K[np.ix_(elem_vars, elem_vars)] += A_bar*self.Ke[index,:,:]
            
            # Mark variables that are non-zero
            mark[elem_vars] = 1
            
            index += 1
                
        # Reorder for the non-zero variable
        var = []
        for i in xrange(self.nvars):
            if mark[i] == 1:
                var.append(i)
                    
        # Assemble the right-hand-side
        self.assembleLoadVec(f)

        # Apply the boundary conditions
        self.applyBCs(self.K, self.f)

        # Reduce the DOF to elements/nodes
        K = K[np.ix_(var, var)]
        f = f[var]

        # Solve the linear system
        uinfty[var] = np.linalg.solve(K, f)

        return uinfty

    def assembleLoadVec(self, f):
        '''
        Create the load vector and populate the vector with entries
        '''
        
        f[:] = 0.0
        for node in self.loads:
            # Add the values to the nodal locations
            f[2*node] += self.loads[node][0]
            f[2*node+1] += self.loads[node][1]

        return

    def applyBCs(self, K, f):
        ''' 
        Apply the boundary conditions to the stiffness matrix and load
        vector
        '''

        # For each node that is in the boundary condition dictionary
        for node in self.bcs:
            uv_list = self.bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in uv_list:
                var = 2*node + index

                # Apply the boundary condition for the variable
                K[var, :] = 0.0
                K[:, var] = 0.0
                K[var, var] = 1.0
                f[var] = 0.0

        return

    def evalSparseCon(self, x, con):
        '''Evaluate the sparse constraints'''
        n = self.nblock*self.nelems
        con[:] = (2.0*x[:n:self.nblock] - 
                  np.sum(x[:n].reshape(-1, self.nblock), axis=1))
        return

    def addSparseJacobian(self, alpha, x, px, con):
        '''Compute the Jacobian-vector product con = alpha*J(x)*px'''
        n = self.nblock*self.nelems
        con[:] += alpha*(2.0*px[:n:self.nblock] - 
                         np.sum(px[:n].reshape(-1, self.nblock), axis=1))
        return

    def addSparseJacobianTranspose(self, alpha, x, pz, out):
        '''Compute the transpose Jacobian-vector product alpha*J^{T}*pz'''
        n = self.nblock*self.nelems
        out[:n:self.nblock] += alpha*pz
        for k in xrange(1,self.nblock):
            out[k:n:self.nblock] -= alpha*pz
        return

    def addSparseInnerProduct(self, alpha, x, c, A):
        '''Add the results from the product J(x)*C*J(x)^{T} to A'''
        n = self.nblock*self.nelems
        A[:] += alpha*np.sum(c[:n].reshape(-1, self.nblock), axis=1)        
        return

    def getTikzPrefix(self):
        '''Return the file prefix'''

        s = '\\documentclass{article}\n'
        s += '\\usepackage[usenames,dvipsnames]{xcolor}\n'
        s += '\\usepackage{tikz}\n'
        s += '\\usepackage[active,tightpage]{preview}\n'
        s += '\\usepackage{amsmath}\n'
        s += '\\usepackage{helvet}\n'
        s += '\\usepackage{sansmath}\n'
        s += '\\PreviewEnvironment{tikzpicture}\n'
        s += '\\setlength\PreviewBorder{5pt}\n'
        s += '\\begin{document}\n'
        s += '\\begin{figure}\n'
        s += '\\begin{tikzpicture}[x=1cm, y=1cm]\n'
        s += '\\sffamily\n'

        return s

    def printTruss(self, x, filename='file.tex', draw_list=[]):
        '''Print the truss to an output file'''

        s = self.getTikzPrefix()
        bar_colors = ['Black', 'ForestGreen', 'Blue']
        
        # Get the minimum value
        Amin = 1.0*min(self.Avals)

        if draw_list is None:
            draw_list = range(self.nelems)

        for i in range(self.nelems):
            # Get the node numbers for this element
            n1 = self.conn[i][0]
            n2 = self.conn[i][1]

            t = x[self.nblock*i]
            if t >= self.epsilon:
                for j in xrange(self.nmats):
                    xj = x[self.nblock*i+1+j]
                    if xj > self.epsilon:
                        s += '\\draw[line width=%f, color=%s, opacity=%f]'%(
                            2.0*self.Avals[j]/Amin, bar_colors[j], xj)
                        s += '(%f,%f) -- (%f,%f);\n'%(
                            self.xpos[2*n1], self.xpos[2*n1+1], 
                            self.xpos[2*n2], self.xpos[2*n2+1])

        for i in draw_list:
            # Get the node numbers for this element
            n1 = self.conn[i][0]
            n2 = self.conn[i][1]

            j = np.argmax(x[self.nblock*i+1:self.nblock*(i+1)])
            xj = x[self.nblock*i+1+j]
            if xj > self.epsilon:
                s += '\\draw[line width=%f, color=Red]'%(
                    2.0*self.Avals[j]/Amin)
                s += '(%f,%f) -- (%f,%f);\n'%(
                    self.xpos[2*n1], self.xpos[2*n1+1], 
                    self.xpos[2*n2], self.xpos[2*n2+1])

        s += '\\end{tikzpicture}'
        s += '\\end{figure}'
        s += '\\end{document}'

        # Write the file
        fp = open(filename, 'w')
        fp.write(s)
        fp.close()

        return
