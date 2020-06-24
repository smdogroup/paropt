"""
Cart-pole trajectory optimization problem from "An Introduction to
Trajectory Optimization" by Matthew Kelly from SIAM Review, Vol 59, No 4.

The initial conditions are that the pole be oriented in the downward
configuration and at rest, and at the final configuration, the pole be
vertical and at rest.
"""

import os
import numpy as np
import mpi4py.MPI as MPI
from paropt import ParOpt


def generate_jacobian():
    """
    Generate the expressions for the Jacobian using sympy
    """
    import sympy as sym

    # Create the symbols
    q = []
    qdot = []
    res = []
    for i in range(4):
        q.append(sym.Symbol('q[%d]'%(i)))
        qdot.append(sym.Symbol('qdot[%d]'%(i)))
        res.append(sym.Symbol('res[%d]'%(i)))
    u = sym.Symbol('u')

    # Create the constants
    m1 = sym.Symbol('self.m1')
    m2 = sym.Symbol('self.m2')
    L = sym.Symbol('self.L')
    g = sym.Symbol('self.g')
    alpha = sym.Symbol('alpha')
    beta = sym.Symbol('beta')

    res[0] = q[2] - qdot[0]
    res[1] = q[3] - qdot[1]
    res[2] = ((m1 + m2*(1.0 - sym.cos(q[1])**2))*qdot[2] -
            (L*m2*sym.sin(q[1])*q[3]**2 + u +
                m2*g*sym.cos(q[1])*sym.sin(q[1])))
    res[3] = (L*(m1 + m2*(1.0 - sym.cos(q[1])**2))*qdot[3] +
            (L*m2*sym.cos(q[1])*sym.sin(q[1])*q[3]**2 +
                u*sym.cos(q[1]) +
                (m1 + m2)*g*sym.sin(q[1])))

    for i in range(4):
        for j in range(4):
            expr = (beta*sym.simplify(sym.diff(res[i], qdot[j])) +
                    alpha*sym.simplify(sym.diff(res[i], q[j])))
            print('J[%d,%d] = '%(i, j) + str(expr))

    return

class CartPole(ParOpt.Problem):
    def __init__(self, t, m1=1.0, m2=1.0, L=1.0):
        self.m1 = m1
        self.m2 = m2
        self.L = L
        self.g = 9.81
        self.t = t

        # Compute the weights for the objective function
        self.tw = np.zeros(t.shape)
        intervals = self.t[1:] - self.t[:-1]
        self.tw[:-1] += 0.5*intervals
        self.tw[1:] += 0.5*intervals
        self.tw[:] /= 100.0

        self.max_newton_iters = 10
        self.newton_tol = 1e-8
        self.con_scale = 10.0

        # Set the number of design variables and constraints
        nvars = len(t)-1
        ncon = 4

        # Initialize the base class
        super(CartPole, self).__init__(MPI.COMM_WORLD, nvars, ncon)

        # Set options for the inequality constraints
        self.setInequalityOptions(dense_ineq=False,
                                  use_lower=True, use_upper=True)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = np.linspace(0, 1, len(x))
        lb[:] = -10.0
        ub[:] = 10.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Evaluate the objective and constraints

        # Set the control input: There is no control for the initial condition
        u = np.zeros(len(x)+1, dtype=ParOpt.dtype)
        u[1:] = x[:]

        fail = 0
        con = np.zeros(4, dtype=ParOpt.dtype)

        # Compute the sum square of the weights
        fobj = np.dot(self.tw[1:], x[:]**2)

        # Compute the full trajectory based on the input forces
        self.q = self.computeTrajectory(self.t, u)

        # Compute the constraints
        con[0] = self.q[-1, 0] # q1 = 0
        con[1] = self.q[-1, 1] - np.pi # q2 = np.pi
        con[2] = self.q[-1, 2] # qdot(q1) = 0
        con[3] = self.q[-1, 3] # qdot(q2) = 0
        con[:] *= self.con_scale

        print('obj: %15g  infeas: %15g'%(fobj, np.sqrt(np.dot(con, con))))

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # Set the control input: There is no control for the initial condition
        u = np.zeros(len(x)+1, dtype=ParOpt.dtype)
        u[1:] = x[:]

        # The objective gradient
        g[:] = 2.0*self.tw[1:]*x[:]

        # Create a vector to store the derivative of the constraints
        dfdx = np.zeros(self.t.shape)
        for state in range(4):
            self.computeAdjointDeriv(self.t, self.q, u, state, dfdx)
            A[state][:] = self.con_scale*dfdx[1:]

        return fail

    def computeResidual(self, q, qdot, u, res):
        """
        Compute the residual of the system dynamics.
        """
        # q = [q1, q2, q1dot, q2dot]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]

        # Compute the residual for the first equation of motion
        res[2] = ((self.m1 + self.m2*(1.0 - np.cos(q[1])**2))*qdot[2] -
            (self.L*self.m2*np.sin(q[1])*q[3]**2 + u +
            self.m2*self.g*np.cos(q[1])*np.sin(q[1])))

        # Compute the residual for the second equation of motion
        res[3] = (self.L*(self.m1 + self.m2*(1.0 - np.cos(q[1])**2))*qdot[3] +
            (self.L*self.m2*np.cos(q[1])*np.sin(q[1])*q[3]**2 +
            u*np.cos(q[1]) +
            (self.m1 + self.m2)*self.g*np.sin(q[1])))
        return

    def computeJacobian(self, alpha, beta, q, qdot, u, J):
        """
        Compute the Jacobian of the system dynamics.
        """

        J[:,:] = 0.0
        J[0,0] = -beta
        J[0,2] = alpha
        J[1,1] = -beta
        J[1,3] = alpha
        J[2,1] = (alpha*self.m2*(-q[3]**2*self.L*np.cos(q[1]) +
            qdot[2]*np.sin(2*q[1]) - self.g*np.cos(2*q[1])))
        J[2,2] = beta*(1.0*self.m1 + 1.0*self.m2*np.sin(q[1])**2)
        J[2,3] = -2*alpha*q[3]*self.L*self.m2*np.sin(q[1])
        J[3,1] = (alpha*(q[3]**2*self.L*self.m2*np.cos(2*q[1]) +
            qdot[3]*self.L*self.m2*np.sin(2*q[1]) + self.g*self.m1*np.cos(q[1]) +
            self.g*self.m2*np.cos(q[1]) - u*np.sin(q[1])))
        J[3,3] = (alpha*q[3]*self.L*self.m2*np.sin(2*q[1]) +
            beta*self.L*(self.m1 + self.m2*np.sin(q[1])**2))

        return

    def computeTrajectory(self, t, u):
        """
        Given the input control force u[i] for t[i] = 0, to t final,
        compute the trajectory.
        """

        # Allocate space for the state variables
        q = np.zeros((len(t), 4), dtype=ParOpt.dtype)

        # Set the initial conditions.
        q[0,:] = 0.0

        # Compute the residual and Jacobian
        res = np.zeros(4, dtype=ParOpt.dtype)
        J = np.zeros((4, 4), dtype=ParOpt.dtype)

        # Integrate forward in time
        for i in range(1, len(t)):
            # Copy the starting point for the first iteration
            q[i,:] = q[i-1,:]

            # Solve the nonlinear equations for q[i]
            for j in range(self.max_newton_iters):
                # Compute the approximate value of the velocities
                alpha = 1.0
                beta = 1.0/(t[i] - t[i-1])
                qdot = beta*(q[i,:] - q[i-1,:])

                self.computeResidual(q[i,:], qdot, u[i], res)
                self.computeJacobian(alpha, beta, q[i,:], qdot, u[i], J)
                update = np.linalg.solve(J, res)
                q[i,:] -= update

                rnorm = np.sqrt(np.dot(res, res))
                if rnorm < self.newton_tol:
                    break

        return q

    def computeAdjointDeriv(self, t, q, u, state, dfdx):
        """
        Compute the derivative of the final state index with respect to
        the control.
        """
        # Zero-out the contributions to the state variables
        dfdx[:] = 0.0

        # Compute the residual and Jacobian
        res = np.zeros(4, dtype=ParOpt.dtype)
        res[state] = 1.0 # df/du
        J = np.zeros((4, 4), dtype=ParOpt.dtype)

        # Integrate the adjoint in reverse
        for i in range(len(t)-1, 0, -1):
            # Set alpha and the qdot values
            alpha = 1.0
            beta = 1.0/(t[i] - t[i-1])
            qdot = beta*(q[i,:] - q[i-1,:])

            # Compute the Jacobian matrix
            self.computeJacobian(alpha, beta, q[i,:], qdot, u[i], J)

            # Compute the adjoint variables
            adjoint = -np.linalg.solve(J.T, res)

            # Compute the total derivative
            dfdx[i] += -adjoint[2] + adjoint[3]*np.cos(q[i,1])

            # Compute the right-hand-side for the next adjoint
            self.computeJacobian(0.0, -beta, q[i,:], qdot, u[i], J)
            res = np.dot(J.T, adjoint)

        return

    def visualize(self, q):
        """
        Visualize the output from a simulation
        """

        import matplotlib.pylab as plt
        from matplotlib.collections import LineCollection
        import matplotlib.cm as cm

        fig = plt.figure()
        plt.axis('equal')
        plt.axis('off')

        values = np.linspace(0, 1.0, q.shape[0])
        cmap = cm.get_cmap('viridis')

        x = []
        y = []
        for i in range(q.shape[0]):
            color = cmap(values[i])

            x1 = q[i,0]
            y1 = 0.0
            x2 = q[i,0] + self.L*np.sin(q[i,1])
            y2 = -self.L*np.cos(q[i,1])

            x.append(x2)
            y.append(y2)

            plt.plot([x1], [y1], color=color, marker='o')
            plt.plot([x1, x2], [y1, y2], linewidth=2, color=color)

        # Create the line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, cmap=cmap)

        # Set the values used for colormapping
        lc.set_array(values)
        fig.axes[0].add_collection(lc)

        plt.show()

# Create the
n = 251
t = np.linspace(0, 3.0, n)
problem = CartPole(t)

problem.checkGradients(1e-5)

filename = 'paropt.out'
options = {
    'algorithm': 'ip',
    'norm_type': 'infinity',
    'max_major_iters': 1000,
    'barrier_strategy': 'monotone',
    'starting_point_strategy': 'least_squares_multipliers',
    'qn_type': 'bfgs',
    'output_file': filename}

opt = ParOpt.Optimizer(problem, options)

# Set a new starting point
opt.optimize()
x, z, zw, zl, zu = opt.getOptimizedPoint()

# Visualize the final design
problem.visualize(problem.q[::10,:])

