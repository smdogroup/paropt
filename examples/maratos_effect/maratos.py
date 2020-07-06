'''
This is example 15.4 in Numerical Optimization by Nocedal Et al.
which might be able to show the Maratos effect that prevents
the optimizer to converge rapidly.

Problem:

min f(x1, x2) = 2(x1^2 + x2^2 - 1) - x1
s.t. 1 - (x1^2 + x2^2)  = 0
'''

import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt
from paropt import ParOpt
import argparse

class Maratos(ParOpt.Problem):
    def __init__(self, plot_label=False):
        self.comm = MPI.COMM_WORLD
        self.nvars = 2
        self.ncon = 1
        self.nineq = 0
        self.design_counter = 0
        super(Maratos, self).__init__(self.comm, self.nvars,
                                      self.ncon, self.nineq)
        self.plot_label = plot_label

    def fun(self, x):
        return 2.0*(x[0]**2 + x[1]**2 - 1.0) - x[0]

    def fun_grad(self, x):
        return np.array([4.0*x[0]-1.0, 4.0*x[1]])

    def con(self, x):
        return 1.0 - (x[0]**2 + x[1]**2)

    def con_grad(self, x):
        return np.array([-2.0*x[0], -2.0*x[1]])

    def plot_contour(self):
        n = 200
        x = np.linspace(-2.0, 2.0, n)
        y = np.linspace(-2.0, 2.0, n)
        X,Y = np.meshgrid(x, y)

        f = np.zeros((n,n))
        c = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                f[i, j] = self.fun([X[i, j], Y[i, j]])
                c[i, j] = self.con([X[i, j], Y[i, j]])

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.contour(X, Y, f, levels=100)
        ax.contour(X, Y, c, levels=[0.0], colors=['red'])
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_aspect('equal', 'box')
        fig.tight_layout()

        self.ax = ax
        self.fig = fig

        return

    def plot_design(self, x):
        plt.plot(x[0], x[1], 'b.')
        label = str(self.design_counter)

        if self.plot_label:
            self.ax.annotate(label,(x[0], x[1]))
            self.design_counter += 1

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[0] = -1.5
        x[1] = 1.5

        lb[:] = -10.0
        ub[:] =  10.0

        '''
        We actually don't need lb and ub in this case
        because we only have equality constraint,
        However, this doesn't work for ip solver (tr works fine)
        '''
        # lb = None
        # ub = None
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        con = np.zeros(self.ncon, dtype=ParOpt.dtype)
        fobj = self.fun(x)
        con[0] = self.con(x)
        fail = 0
        self.plot_design(x)
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        g[0] = self.fun_grad(x)[0]
        g[1] = self.fun_grad(x)[1]
        A[0][0] = self.con_grad(x)[0]
        A[0][1] = self.con_grad(x)[1]
        fail = 0
        return fail


parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='tr')
parser.add_argument('--no_label', action='store_false', default=True)
args = parser.parse_args()
optimizer = args.optimizer
plot_label = args.no_label

problem = Maratos(plot_label=plot_label)
problem.plot_contour()

options = {
    'algorithm': 'tr',
    'qn_type': 'bfgs',
    'abs_res_tol': 1e-8,
    'output_level': 0,
    'use_backtracking_alpha': True,
    'max_major_iters': 100,
    'tr_init_size': 0.1,
    'tr_min_size': 1e-6,
    'tr_max_size': 1.0,
    'tr_eta': 0.25,
    'penalty_gamma': 1.0,
    'tr_adaptive_gamma_update': True,
    'tr_penalty_gamma_max': 1e5,
    'tr_penalty_gamma_min': 1e-5,
    'tr_max_iterations': 200,
    'use_line_search': False}

if optimizer == 'ip':
    options = {
            'algorithm': 'ip',
            'qn_subspace_size': 10,
            'abs_res_tol': 1e-6,
            'barrier_strategy': 'monotone',
            'output_level': 1,
            'armijo_constant': 1e-5,
            'max_major_iters': 500}

opt = ParOpt.Optimizer(problem, options)
opt.optimize()
plt.show()