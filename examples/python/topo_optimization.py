'''
Perform a 2D plane stress analysis for topology optimization
'''

import numpy as np
from mpi4py import MPI
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from paropt import ParOpt

class TopoAnalysis(ParOpt.Problem):
    def __init__(self, nxelems, nyelems, Lx, Ly, r0=1.5, p=3.0, 
                 E0=1.0, nu=0.3, default_bcs=True):
        '''
        The constructor for the topology optimization class. 

        This function sets up the data that is requried to perform a
        plane stress analysis of a square, plane stress structure.
        This is probably only useful for topology optimization.
        '''
        super().__init__(MPI.COMM_SELF, nxelems*nyelems, 1)

        self.nxelems = nxelems
        self.nyelems = nyelems
        self.Lx = Lx
        self.Ly = Ly
        self.r0 = r0
        self.p = p
        self.E0 = E0
        self.nu = 0.3

        self.uvars = np.zeros((self.nxelems+1, self.nyelems+1), dtype=int)
        self.vvars = np.zeros((self.nxelems+1, self.nyelems+1), dtype=int)

        if default_bcs:
            # Set the boundary conditions - these could be modified
            for j in range(self.nyelems+1):
                self.uvars[0, j] = -1
                self.vvars[0, j] = -1

            self.initialize()
        
        return

    def initialize(self):
        '''
        Initialize the finite-element problem set up for analys. 

        This counts up the number of variables and applies the forces
        to the problem. 
        '''
        
        # Count up the number of variables in the problem
        nvars = 0
        for j in range(self.nyelems+1):
            for i in range(self.nxelems+1):
                # Set the state variable for x-displacement
                if self.uvars[i, j] >= 0:
                    self.uvars[i, j] = nvars
                    nvars += 1

                # Set the state variable for y-displacement 
                if self.vvars[i, j] >= 0:
                    self.vvars[i, j] = nvars
                    nvars += 1

        self.nvars = nvars

        # Now, compute the filter weights and store them as a sparse
        # matrix
        F = sparse.lil_matrix((self.nxelems*self.nyelems,
                               self.nxelems*self.nyelems))

        # Compute the inter corresponding to the filter radius
        ri = int(np.ceil(self.r0))

        for j in range(self.nyelems):
            for i in range(self.nxelems):
                w = []
                wvars = []

                # Compute the filtered design variable: xfilter
                for jj in range(max(0, j-ri), min(self.nyelems, j+ri+1)):
                    for ii in range(max(0, i-ri), min(self.nxelems, i+ri+1)):
                        r = np.sqrt((i - ii)**2 + (j - jj)**2)
                        if r < self.r0:
                            w.append((self.r0 - r)/self.r0)
                            wvars.append(ii + jj*self.nxelems)

                # Normalize the weights
                w = np.array(w)
                w /= np.sum(w)

                # Set the weights into the filter matrix W
                F[i + j*self.nxelems, wvars] = w
                
        # Covert the matrix to a CSR data format
        self.F = F.tocsr()

        # Set the force vector
        self.f = np.zeros(self.nvars)
        self.f[self.vvars[self.nxelems, 0]] = -1e3

        return

    def mass(self, x):
        '''
        Compute the mass of the structure
        '''

        area = (self.Lx/self.nxelems)*(self.Ly/self.nyelems)

        return area*np.sum(x)

    def mass_grad(self, x):
        '''
        Compute the derivative of the mass
        '''

        area = (self.Lx/self.nxelems)*(self.Ly/self.nyelems)
        dmdx = area*np.ones(x.shape)

        return dmdx

    def compliance(self, x):
        '''
        Compute the structural compliance
        '''

        # Compute the filtered compliance. Note that 'dot' is scipy
        # matrix-vector multiplicataion
        xfilter = self.F.dot(x)

        # Compute the Young's modulus in each element
        E = self.E0*xfilter**self.p
            
        # Compute the stiffness
        self.analyze(E)

        # Return the compliance
        return 0.5*np.dot(self.f, self.u)

    def compliance_grad(self, x):
        '''
        Compute the gradient of the compliance using the adjoint
        method. 

        Since the governing equations are self-adjoint, and the
        function itself takes a special form: 

        K*psi = 0.5*f => psi = 0.5*u
        
        So we can skip the adjoint computation itself since we have
        the displacement vector u from the solution.

        d(compliance)/dx = - 0.5*u^{T}*d(K*u - f)/dx = - 0.5*u^{T}*dK/dx*u
        '''

        # Compute the filtered variables
        xfilter = self.F.dot(x)

        # First compute the derivative with respect to the filtered
        # variables
        dcdxf = np.zeros(x.shape)

        # Sum up the contributions from each
        kelem = self.compute_kelem()

        # Compute df/dE
        for j in range(self.nyelems):
            for i in range(self.nxelems):
                # Retrieve the element variables from the
                # finite-element solution vector
                evars = np.zeros(8)

                # Set up the element variables that are not on a
                # Dirichlet boundary condition
                gvars = [self.uvars[i, j], self.vvars[i, j],
                         self.uvars[i+1, j], self.vvars[i+1, j],
                         self.uvars[i, j+1], self.vvars[i, j+1],
                         self.uvars[i+1, j+1], self.vvars[i+1, j+1]]
                
                # Add the values to the stiffness matrix
                for ii in range(8):
                    if gvars[ii] >= 0:
                        evars[ii] = self.u[gvars[ii]]
                        
                dxfdE = self.E0*self.p*xfilter[i + j*self.nxelems]**(self.p - 1.0)
                dcdxf[i + j*self.nxelems] = -0.5*np.dot(evars, np.dot(kelem, evars))*dxfdE

        # Now evaluate the effect of the filter
        dcdx = (self.F.transpose()).dot(dcdxf)

        return dcdx

    def compute_kelem(self):
        '''
        Compute the element stiffness matrix using a Gauss quadrature
        scheme.

        Note that this code assumes that all elements are uniformly
        rectangular and so the same element stiffness matrix can be
        used for every element.
        '''

        # Compute the element stiffness matrix
        gauss_pts = [-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)]

        # Create the 8 x 8 element stiffness matrix
        kelem = np.zeros((8, 8))
        B = np.zeros((3, 8))

        # Compute the constitutivve matrix
        C = np.array([[1.0, self.nu, 0.0],
                      [self.nu, 1.0, 0.0],
                      [0.0, 0.0, 0.5*(1.0 - self.nu)]])
        C = 1.0/(1.0 - self.nu**2)*C

        # Set the terms for the area-dependences
        xi = 2.0*self.nxelems/self.Lx
        eta = 2.0*self.nyelems/self.Ly
        area = 1.0/(xi*eta)

        for x in gauss_pts:
            for y in gauss_pts:
                # Evaluate the derivative of the shape functions with
                # respect to the x/y directions
                Nx = 0.25*xi*np.array([y - 1.0, 1.0 - y, -1.0 - y, 1.0 + y])
                Ny = 0.25*eta*np.array([x - 1.0, -1.0 - x, 1.0 - x, 1.0 + x])

                # Evaluate the B matrix
                B = np.array(
                    [[ Nx[0], 0.0, Nx[1], 0.0, Nx[2], 0.0, Nx[3], 0.0 ],
                     [ 0.0, Ny[0], 0.0, Ny[1], 0.0, Ny[2], 0.0, Ny[3] ],
                     [ Ny[0], Nx[0], Ny[1], Nx[1], Ny[2], Nx[2], Ny[3], Nx[3] ]])

                # Add the contribution to the stiffness matrix
                kelem += area*np.dot(B.transpose(), np.dot(C, B))

        return kelem

    def analyze(self, E):
        '''
        Given the elastic modulus variable values, perform the
        analysis and update the state variables.

        This function sets up and solves the linear finite-element
        problem with the given set of elastic moduli. Note that E > 0
        (component wise). 

        Args:
           E: An array of the elastic modulus for every element in the
              plane stress domain
        '''

        # Compute the finite-element stiffness matrix
        kelem = self.compute_kelem()

        # Now, go through all the elements in the domain, add add the
        # product of E times the element stiffness matrix to the
        # global stiffness matrix        
        K = sparse.lil_matrix((self.nvars, self.nvars))

        for j in range(self.nyelems):
            for i in range(self.nxelems):
                # Set up the element variables that are not on a
                # Dirichlet boundary condition
                gvars = [self.uvars[i, j], self.vvars[i, j],
                         self.uvars[i+1, j], self.vvars[i+1, j],
                         self.uvars[i, j+1], self.vvars[i, j+1],
                         self.uvars[i+1, j+1], self.vvars[i+1, j+1]]
                
                # Add the values to the stiffness matrix
                for ii in range(8):
                    if gvars[ii] >= 0:
                        for jj in range(8):
                            if gvars[jj] >= 0:
                                K[gvars[ii], gvars[jj]] +=\
                                             E[i + j*self.nxelems]*kelem[ii, jj]

        # Convert to csc format
        self.K = K.tocsc()

        # Solve the sparse linear system for the load vector
        self.LU = linalg.dsolve.factorized(self.K)
        
        # Compute the solution to the linear system K*u = f
        self.u = self.LU(self.f)

        return 

    def eval_vm_stress(self, i, j):
        '''
        Evaluate the stress for the given element number
        '''
        
        # Set up the element variables that are not on a
        # Dirichlet boundary condition
        gvars = [self.uvars[i, j], self.vvars[i, j],
                 self.uvars[i+1, j], self.vvars[i+1, j],
                 self.uvars[i, j+1], self.vvars[i, j+1],
                 self.uvars[i+1, j+1], self.vvars[i+1, j+1]]
                
        # Add the values to the stiffness matrix
        evars = np.zeros(8)
        for ii in range(8):
            if gvars[ii] >= 0:
                evars[ii] = self.u[gvars[ii]]

        # Compute the constitutivve matrix
        C = np.array([[1.0, self.nu, 0.0],
                      [self.nu, 1.0, 0.0],
                      [0.0, 0.0, 0.5*(1.0 - self.nu)]])
        C = 1.0/(1.0 - self.nu**2)*C

        # Evaluate the derivative of the shape functions with
        # respect to the x/y directions
        Nx = 0.25*xi*np.array([-1.0, 1.0, -1.0, 1.0])
        Ny = 0.25*eta*np.array([-1.0, -1.0, 1.0, 1.0])

        # Evaluate the B matrix
        xi = 2.0*self.nxelems/self.Lx
        eta = 2.0*self.nyelems/self.Ly

        B = np.array(
            [[ Nx[0], 0.0, Nx[1], 0.0, Nx[2], 0.0, Nx[3], 0.0 ],
             [ 0.0, Ny[0], 0.0, Ny[1], 0.0, Ny[2], 0.0, Ny[3] ],
             [ Ny[0], Nx[0], Ny[1], Nx[1], Ny[2], Nx[2], Ny[3], Nx[3] ]])

        s = self.E0*np.dot(C, np.dot(B, evars))

        return np.sqrt(s[0]**2 + s[1]**2 - s[0]*s[1] + 3.0*s[2]**2)

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        lb[:] = 1e-3
        ub[:] = 1.0
        x[:] = 0.95
        return

    def evalObjCon(self, x):
        '''
        Return the objective, constraint and fail flag
        '''

        fail = 0
        obj = self.compliance(x[:])
        con = np.array([40.0 - self.mass(x[:])])
        
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        '''
        Return the objective, constraint and fail flag
        '''

        fail = 0
        g[:] = self.compliance_grad(x[:])
        A[0][:] = -self.mass_grad(x[:])

        self.write_output(x[:])

        return fail

    def write_output(self, x):
        '''
        Write out something to the screen
        '''

        if not hasattr(self, 'fig'):
            plt.ion()
            self.fig, self.ax = plt.subplots()
            plt.draw()

        # Prepare a pixel visualization of the design vars
        image = np.zeros((self.nyelems, self.nxelems))
        for j in range(self.nyelems):
            for i in range(self.nxelems):
                image[j, i] = x[i + j*self.nxelems]

        x = np.linspace(0, self.Lx, self.nxelems)
        y = np.linspace(0, self.Ly, self.nyelems)

        self.ax.contourf(x, y, image)
        plt.axis('equal')
        plt.savefig('topology.pdf')

        return

if __name__ == '__main__':
    nxelems = 24
    nyelems = 24
    Lx = 5.0
    Ly = 5.0
    problem = TopoAnalysis(nxelems, nyelems,
                           Lx, Ly, E0=70e3)
    problem.checkGradients()

    # Create the quasi-Newton Hessian approximation
    qn = ParOpt.LBFGS(problem, subspace=10)

    # Create the trust region problem
    tr_init_size = 0.05
    tr_min_size = 1e-6
    tr_max_size = 10.0
    tr_eta = 0.25
    tr_penalty_gamma = 10.0
    tr = ParOpt.TrustRegion(problem, qn, tr_init_size,
                            tr_min_size, tr_max_size,
                            tr_eta, tr_penalty_gamma)
    tr.setTrustRegionTolerances(0.0, 0.0, 0.0)

    filename = 'topo_optimization.out'

    # Set up the optimization problem
    tr_opt = ParOpt.InteriorPoint(tr, 2, ParOpt.BFGS)

    # Set up the optimization problem
    if filename is not None:
        tr_opt.setOutputFile(filename)

    # Set the tolerances
    tr_opt.setAbsOptimalityTol(1e-8)
    tr_opt.setStartingPointStrategy(ParOpt.AFFINE_STEP)
    tr_opt.setStartAffineStepMultiplierMin(0.01)

    # Set optimization parameters
    tr_opt.setArmijoParam(1e-5)
    tr_opt.setMaxMajorIterations(5000)
    tr_opt.setBarrierPower(2.0)
    tr_opt.setBarrierFraction(0.1)

    # optimize
    tr.setOutputFile(filename + '_tr')
    tr.setPrintLevel(1)
    tr.optimize(tr_opt)
