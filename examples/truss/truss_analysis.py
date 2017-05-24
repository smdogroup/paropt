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
                 E, rho, m_fixed, A_min, A_max, A_init=None,
                 Area_scale=1e-3, mass_scale=None):
        '''
        Analysis problem for mass-constrained compliance minimization
        '''

        # Initialize the super class
        nvars = len(conn)
        ncon = 1
        super(TrussAnalysis, self).__init__(MPI.COMM_SELF, 
                                            nvars, ncon)

        # Store pointer to the data
        self.conn = conn
        self.xpos = xpos
        self.loads = loads
        self.bcs = bcs

        # Set the material properties
        self.E = E
        self.rho = rho

        # Fixed mass value
        self.m_fixed = m_fixed

        # Set the values for the scaling
        self.A_min = A_min
        self.A_max = A_max
        self.Area_scale = Area_scale

        if A_init is None:
            self.A_init = 0.5*(A_min + A_max)
        else:
            self.A_init = A_init

        # Scaling for the objective
        self.obj_scale = None

        # Scaling for the mass constraint
        if mass_scale is None:
            self.mass_scale = self.m_fixed/nvars
        else:
            self.mass_scale = mass_scale

        # Allocate the matrices required
        nvars = len(self.xpos)
        self.K = np.zeros((nvars, nvars))
        self.Kp = np.zeros((nvars, nvars))
        self.f = np.zeros(nvars)
        self.u = np.zeros(nvars)
        self.phi = np.zeros(nvars)
        
        # Keep track of the different counts
        self.fevals = 0
        self.gevals = 0
        self.hevals = 0
            
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        lb[:] = self.A_min/self.Area_scale
        ub[:] = self.A_max/self.Area_scale
        x[:] = self.A_init/self.Area_scale
        return

    def evalSparseCon(self, x, con):
        '''Evaluate the sparse constraints'''
        return
    
    def addSparseJacobian(self, alpha, x, px, con):
        '''Compute the Jacobian-vector product con = alpha*J(x)*px'''
        return

    def addSparseJacobianTranspose(self, alpha, x, pz, out):
        '''Compute the transpose Jacobian-vector product alpha*J^{T}*pz'''
        return

    def addSparseInnerProduct(self, alpha, x, c, A):
        '''Add the results from the product J(x)*C*J(x)^{T} to A'''
        return
    
    def evalObjCon(self, x):
        '''
        Evaluate the objective (compliance) and constraint (mass)
        '''
        
        # Add the number of function evaluations
        self.fevals += 1

        # Convert the design variables with the scaling
        A = self.Area_scale*x

        # Evaluate compliance objective
        self.assembleMat(A, self.K)
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
            self.obj_scale = obj/10.0

        # Scale the compliance objective
        obj = obj/self.obj_scale

        # Compute the mass of the entire truss
        mass = 0.0
        index = 0

        for bar in self.conn:
            # Get the first and second node numbers from the bar
            n1 = bar[0]
            n2 = bar[1]

            # Compute the nodal locations
            xd = self.xpos[2*n2] - self.xpos[2*n1]
            yd = self.xpos[2*n2+1] - self.xpos[2*n1+1]
            Le = np.sqrt(xd**2 + yd**2)
            mass += self.rho*Le*A[index]

            index += 1

        # Create the array of constraints >= 0.0
        con = np.array([self.m_fixed - mass])/self.mass_scale

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
        Acon[:] = 0.0

        # Retrieve the area variables
        A = self.Area_scale*x
        
        # Add up the contribution to the gradient
        index = 0
        for bar, A_bar in zip(self.conn, A):
            # Get the first and second node numbers from the bar
            n1 = bar[0]
            n2 = bar[1]

            # Compute the nodal locations
            xd = self.xpos[2*n2] - self.xpos[2*n1]
            yd = self.xpos[2*n2+1] - self.xpos[2*n1+1]
            Le = np.sqrt(xd**2 + yd**2)
            C = xd/Le
            S = yd/Le
            
            # Add the contribution to the gradient of the mass
            Acon[0, index] += self.rho*Le

            # Compute the element stiffness matrix
            Ke = (self.E/Le)*np.array(
                [[C**2, C*S, -C**2, -C*S],
                 [C*S, S**2, -C*S, -S**2],
                 [-C**2, -C*S, C**2, C*S],
                 [-C*S, -S**2, C*S, S**2]])
            
            # Create a list of the element variables for convenience
            elem_vars = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            
            # Add the product to the derivative of the compliance
            for i in xrange(4):
                for j in xrange(4):
                    gobj[index] -= \
                        self.u[elem_vars[i]]*self.u[elem_vars[j]]*Ke[i, j]
            
            index += 1

        # Create the array of constraints >= 0.0
        Acon[0, :] *= -self.Area_scale/self.mass_scale

        # Scale the objective gradient
        gobj *= self.Area_scale/self.obj_scale

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

        # Retrieve the area variables
        A = self.Area_scale*x

        # Assemble the stiffness matrix along the px direction
        self.assembleMat(self.Area_scale*px, self.Kp)
        np.dot(self.Kp, self.u, out=self.phi)
        self.applyBCs(self.Kp, self.phi)

        # Solve the resulting linear system of equations
        linalg.solve_triangular(self.L, self.phi, lower=True,
                                trans='N', overwrite_b=True)
        linalg.solve_triangular(self.L, self.phi, lower=True, 
                                trans='T', overwrite_b=True)
        
        # Add up the contribution to the gradient
        index = 0
        for bar, A_bar in zip(self.conn, A):
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
            Ke = (self.E/Le)*np.array(
                [[C**2, C*S, -C**2, -C*S],
                 [C*S, S**2, -C*S, -S**2],
                 [-C**2, -C*S, C**2, C*S],
                 [-C*S, -S**2, C*S, S**2]])
            
            # Create a list of the element variables for convenience
            elem_vars = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
            
            # Add the product to the derivative of the compliance
            for i in xrange(4):
                for j in xrange(4):
                    hvec[index] += \
                        2.0*self.phi[elem_vars[i]]*self.u[elem_vars[j]]*Ke[i, j]
            
            index += 1

        hvec *= self.Area_scale/self.obj_scale

        fail = 0
        return fail

    def assembleMat(self, A, K):
        '''
        Given the connectivity, nodal locations and material properties,
        assemble the stiffness matrix
        
        input:
        A:   the bar area

        output:
        K:   the stiffness matrix
        '''

        # Zero the stiffness matrix
        K[:,:] = 0.0

        # Loop over each element in the mesh
        for bar, A_bar in zip(self.conn, A):
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
            Ke = self.E*A_bar/Le*np.array(
                [[C**2, C*S, -C**2, -C*S],
                 [C*S, S**2, -C*S, -S**2],
                 [-C**2, -C*S, C**2, C*S],
                 [-C*S, -S**2, C*S, S**2]])
        
            # Create a list of the element variables for convenience
            elem_vars = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        
            # Add the element stiffness matrix to the global stiffness
            # matrix
            for i in xrange(4):
                for j in xrange(4):
                    K[elem_vars[i], elem_vars[j]] += Ke[i, j]
                    
        return

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

    def computeForces(self, A, u):
        '''
        Compute the forces in each of the truss members
        '''

        # Create the global stiffness matrix
        bar_forces = np.zeros(len(self.conn))

        # Loop over each element in the mesh
        index = 0
        for bar, A_bar in zip(self.conn, A):
            # Get the first and second node numbers from the bar
            n1 = bar[0]
            n2 = bar[1]

            # Compute the nodal locations
            xd = self.xpos[2*n2] - self.xpos[2*n1]
            yd = self.xpos[2*n2+1] - self.xpos[2*n1+1]
            Le = np.sqrt(xd**2 + yd**2)
            C = xd/Le
            S = yd/Le

            # Compute the hat displacements
            u1_hat = C*u[2*n1] + S*u[2*n1+1]
            u2_hat = C*u[2*n2] + S*u[2*n2+1]

            # Compute the strain
            epsilon = (u2_hat - u1_hat)/Le

            bar_forces[index] = self.E*A_bar*epsilon
            index += 1

        return bar_forces

    def printResult(self, x):
        '''
        Evaluate the derivative of the compliance and mass
        '''

        A = self.Area_scale*x

        # Evaluate compliance objective
        self.assembleMat(A, self.K)
        self.assembleLoadVec(self.f)
        self.applyBCs(self.K, self.f)

        # Solve the resulting linear system of equations
        self.u = np.linalg.solve(self.K, self.f)
        
        forces = self.computeForces(A, self.u)

        print 'Compliance:     %15.10f'%(0.5*np.dot(self.u, self.f))
        print 'Max strain:     %15.10f'%(max(forces/(self.E*A)))
        print 'Max abs strain: %15.10f'%(max(np.fabs(forces/(self.E*A))))
        print 'Min strain:     %15.10f'%(min(forces/(self.E*A)))

        return

    def fullyStressed(self, A, sigma_max, A_min):
        '''
        Perform the fully stress design procedure
        '''

        for i in xrange(100):
            # Evaluate compliance objective
            self.assembleMat(A, self.K)
            self.assembleLoadVec(self.f)
            self.applyBCs(self.K, self.f)
            
            # Solve the resulting linear system of equations
            self.u = np.linalg.solve(self.K, self.f)

            # Evaluate the forces
            forces = self.compute_forces(A, u)

            # Compute the mass
            mass = 0.0
            for bar, A_bar in zip(self.conn, A):
                # Get the first and second node numbers from the bar
                n1 = bar[0]
                n2 = bar[1]

                # Compute the nodal locations
                xd = self.xpos[2*n2] - self.xpos[2*n1]
                yd = self.xpos[2*n2+1] - self.xpos[2*n1+1]
                Le = np.sqrt(xd**2 + yd**2)

                mass += self.rho*A_bar*Le

            print mass

            for k in xrange(len(A)):
                A[k] = np.max([np.fabs(forces[k])/sigma_max, A_min])

        return A

    def plotTruss(self, x, tol=None, filename='opt_truss.pdf'):
        '''
        Plot the deformed and undeformed truss structure
        '''

        # Scale the values of the design variables
        A = self.Area_scale*x

        # Find out if the tolerance is set
        if tol is None:
            tol = 0.0

        # Set the background colour
        fig = plt.figure(facecolor='w')

        # Evaluate compliance objective
        self.assembleMat(A, self.K)
        self.assembleLoadVec(self.f)
        self.applyBCs(self.K, self.f)
            
        # Solve the resulting linear system of equations
        self.u = np.linalg.solve(self.K, self.f)
        
        index = 0
        for bar in self.conn:
            n1 = bar[0]
            n2 = bar[1]

            if A[index] >= tol:
                plt.plot([self.xpos[2*n1], self.xpos[2*n2]], 
                         [self.xpos[2*n1+1], self.xpos[2*n2+1]], '-ko', 
                         linewidth=5*(A[index]/max(A)))
            index += 1

        plt.axis('equal')
        plt.savefig(filename)
        plt.close()

        return

    def writeOutputFiles(self, A, show=False):
        '''
        Write out something to the screen
        '''

        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            plt.draw()

            # Draw a visualization of the truss
            index = 0
            max_A = max(A)
            self.lines = []

            for bar in self.conn:
                n1 = bar[0]
                n2 = bar[1]
                xv = [self.xpos[2*n1], self.xpos[2*n2]]
                yv = [self.xpos[2*n1+1], self.xpos[2*n2+1]]

                line, = self.ax.plot(xv, yv, '-ko', 
                                     linewidth=A[index])
                self.lines.append(line)
                index += 1
 
        else:
            # Set the value of the lines
            index = 0
            max_A = max(A)
            
            for bar in self.conn:
                plt.setp(self.lines[index], 
                         linewidth=5*(A[index]/max(A)))
                index += 1
 
        plt.axis('equal')
        plt.draw()

        return
