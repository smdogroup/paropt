from __future__ import print_function

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import argparse

# Import ParOpt
from paropt import ParOpt

class Problem1(ParOpt.pyParOptProblem):
    def __init__(self):
        self.x_hist = []
        super(Problem1, self).__init__(MPI.COMM_SELF, 2, 1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        lb[:] = -2.0
        ub[:] = 2.0
        x[:] = -2.0 + 4.0*np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint values'''
        
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        fobj = 2*x[0]**2 + 2*x[1]**2 + x[0]*x[1]
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 4*x[0] + x[1]
        g[1] = 4*x[1] + x[0]
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail

class Problem2(ParOpt.pyParOptProblem):
    def __init__(self):
        self.x_hist = []
        super(Problem2, self).__init__(MPI.COMM_SELF, 2, 1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        lb[:] = -2.0
        ub[:] = 2.0
        x[:] = -2.0 + 4.0*np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint values'''

        # Append the point to the history
        self.x_hist.append(np.array(x))
        
        # Evaluate the objective and constraints
        fail = 0
        fobj = x[0]**4 + x[1]**2 + 2*x[0]*x[1] - x[0] - x[1]
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 4*x[0]**3 + 2*x[1] - 1.0 
        g[1] = 2*x[1] + 2*x[0] - 1.0
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail

class Problem3(ParOpt.pyParOptProblem):
    def __init__(self):
        self.x_hist = []
        super(Problem3, self).__init__(MPI.COMM_SELF, 2, 1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        lb[:] = -2.0
        ub[:] = 2.0
        x[:] = -2.0 + 4.0*np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint values'''

        # Append the point to the solution history
        self.x_hist.append(np.array(x))
        
        # Evaluate the objective and constraints
        fail = 0
        fobj = x[0]**4 + x[1]**4 + 1 - x[0]**2 - x[1]**2
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 4*x[0]**3 - 2*x[0]
        g[1] = 4*x[1]**3 - 2*x[1]
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail

class Problem4(ParOpt.pyParOptProblem):
    def __init__(self):
        self.x_hist = []
        super(Problem4, self).__init__(MPI.COMM_SELF, 2, 1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        '''Get the variable values and bounds'''
        lb[:] = -2.0
        ub[:] = 2.0
        x[:] = -2.0 + 4.0*np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        '''Evaluate the objective and constraint values'''

        # Append the point to the solution history
        self.x_hist.append(np.array(x))
        
        # Evaluate the objective and constraints
        fail = 0
        fobj = -10*x[0]**2 + 10*x[1]**2 + 4*np.sin(x[0]*x[1]) - 2*x[0] + x[0]**4
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = -20*x[0] + 4*np.cos(x[0]*x[1])*x[1] - 2.0 + 4*x[0]**3
        g[1] =  20*x[1] + 4*np.cos(x[0]*x[1])*x[0]
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail

def plot_it_all(problem, use_tr=False):
    '''
    Plot a carpet plot with the search histories for steepest descent,
    conjugate gradient and BFGS from the same starting point.
    '''

    # Set up the optimization problem
    max_lbfgs = 20
    opt = ParOpt.pyParOpt(problem, max_lbfgs, ParOpt.BFGS)
    opt.checkGradients(1e-6)

    # Create the data for the carpet plot
    n = 150
    xlow = -4.0
    xhigh = 4.0
    x1 = np.linspace(xlow, xhigh, n)
    r = np.zeros((n, n))

    for j in range(n):
        for i in range(n):
            fail, fobj, con = problem.evalObjCon([x1[i], x1[j]])
            r[j, i] = fobj

    # Assign the contour levels
    levels = np.min(r) + np.linspace(0, 1.0, 75)**2*(np.max(r) - np.min(r))

    # Create the plot
    fig = plt.figure(facecolor='w')
    plt.contour(x1, x1, r, levels)

    colours = ['-bo', '-ko', '-co', '-mo', '-yo',
               '-bx', '-kx', '-cx', '-mx', '-yx' ]

    for k in range(len(colours)):
        # Optimize the problem
        problem.x_hist = []

        if use_tr:
            # Create the quasi-Newton Hessian approximation
            qn = ParOpt.LBFGS(problem, subspace=2)

            # Create the trust region problem
            tr_init_size = 0.05
            tr_min_size = 1e-6
            tr_max_size = 10.0
            tr_eta = 0.25
            tr_penalty_gamma = 10.0
            tr = ParOpt.pyTrustRegion(problem, qn, tr_init_size,
                                      tr_min_size, tr_max_size,
                                      tr_eta, tr_penalty_gamma)

            # Set up the optimization problem
            tr_opt = ParOpt.pyParOpt(tr, 2, ParOpt.BFGS)
    
            # Optimize
            tr.optimize(tr_opt)
        else:
            opt.resetQuasiNewtonHessian()
            opt.setInitBarrierParameter(0.1)
            opt.setUseLineSearch(1)
            opt.optimize()

        # Copy out the steepest descent points
        sd = np.zeros((2, len(problem.x_hist)))
        for i in range(len(problem.x_hist)):
            sd[0, i] = problem.x_hist[i][0]
            sd[1, i] = problem.x_hist[i][1]

        plt.plot(sd[0, :], sd[1, :], colours[k],
                 label='SD %d'%(sd.shape[1]))
        plt.plot(sd[0, -1], sd[1, -1], '-ro')

    plt.legend()
    plt.axis([xlow, xhigh, xlow, xhigh])
    plt.show()

problems = [Problem1(), Problem2(), Problem3(), Problem4()]

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='ip',
                    help='optimizer type')
args = parser.parse_args()

use_tr = False
if args.optimizer != 'ip':
    use_tr = True

for problem in problems:
    plot_it_all(problem, use_tr)
