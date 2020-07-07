
"""
ref: Benchmarking Optimization Software with COPS 3.0
problem 1. Largest Small Polygon
nv: number of vertices
dv: r_i, theta_i, i=0,1,...,nv-1
max   f(r,theta) = 1/2 * sum(r_{i+1} * r{i} * sin(theta_{i+1} - theta_{i}))  i=0,1,...,nv-2
s.t.  c1 = theta_{i+1} - theta_i >= 0, i=0,1,...,nv-2
      c2 = 1 - r_i^2 - r_j^2 + 2 * r_i * r_j * cos(theta_i - theta_j) >= 0        i=0,1,..,nv-2; j=i+1,...,nv-1
      0 <= r_i     <= 10
      0 <= theta_i <= pi
"""

from paropt import ParOpt
import mpi4py.MPI as MPI
import numpy as np
import argparse

class Polygon(ParOpt.Problem):
    def __init__(self, nv):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nv = nv
        self.nr = nv
        self.ntheta = nv
        self.nvars = 2*nv
        self.nc1 = nv - 1
        self.nc2 = int(nv * (nv - 1) / 2)
        self.ncon = self.nc1 + self.nc2

        # Initialize the base class
        super(Polygon, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        nv = self.nv

        x[:nv] = np.random.uniform(low=0.1, high=9.9, size=nv)
        lb[:nv] = 0.0
        ub[:nv] = 10.0

        x[nv:] = np.linspace(start=0.1*np.pi, stop=0.9*np.pi, num=nv)
        lb[nv:] = 0.0
        ub[nv:] = np.pi

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        fail = 0
        nv = self.nv
        con = np.zeros(self.ncon, dtype=ParOpt.dtype)
        c1 = con[:nv-1]
        c2 = con[nv-1:]
        r =     x[:nv]
        theta = x[nv:]

        index = 0
        # evaluate obj and c1
        fobj = 0
        for i in range(nv - 1):
            fobj -= 0.5 * r[i] * r[i+1] * np.sin(theta[i+1] - theta[i])
            c1[i] = theta[i+1] - theta[i]

            # evaluate
            for j in range(i + 1, nv):
                c2[index] = 1.0 - r[i]**2 - r[j]**2 + 2 * r[i] * r[j] * np.cos(theta[i] - theta[j])
                index += 1

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0
        nv = self.nv
        r = x[:nv]
        theta = x[nv:]

        # Compute g
        g[:] = 0.0
        for i in range(nv - 1):
            g[i] -= 0.5 * r[i+1] * np.sin(theta[i+1] - theta[i])
            g[i+1] -= 0.5 * r[i] * np.sin(theta[i+1] - theta[i])
            g[nv+i] -=  -0.5 * r[i] * r[i+1] * np.cos(theta[i+1] - theta[i])
            g[nv+i+1] -= 0.5 * r[i] * r[i+1] * np.cos(theta[i+1] - theta[i])

        # Compute A1t
        for i in range(nv - 1):
            A[i][nv+i] = - 1.0
            A[i][nv+i+1] = 1.0

        # Compute A2r and A2t
        index = nv-1
        for i in range(nv - 1):
            for j in range(i + 1, nv):
                # Derivatives w.r.t. r
                A[index][i] = -2.0 * r[i] + 2.0 * r[j] * np.cos(theta[i] - theta[j])
                A[index][j] = -2.0 * r[j] + 2.0 * r[i] * np.cos(theta[i] - theta[j])

                # Derivatives w.r.t. theta
                A[index][nv + i] = - 2.0 * r[i] * r[j] * np.sin(theta[i] - theta[j])
                A[index][nv + j] =   2.0 * r[i] * r[j] * np.sin(theta[i] - theta[j])
                index += 1

        return fail


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ip')
    parser.add_argument('--n', type=int, default=6, help='number of vertices')
    args = parser.parse_args()

    use_tr = False
    if args.algorithm != 'ip':
        use_tr = True

    # use interior point algorithm
    options = {
        'algorithm': 'ip',
        'qn_type': 'bfgs',
        'qn_subspace_size': 10,
        'abs_res_tol': 1e-6,
        'barrier_strategy': 'monotone',
        'starting_point_strategy': 'affine_step',
        'use_backtracking_alpha': True,
        'output_level': 1,
        'max_major_iters': 1000}

    # use trust region algorithm
    if use_tr:
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
            'tr_max_iterations': 500,
            'use_line_search': False}

    polygon = Polygon(args.n)
    polygon.checkGradients()
    opt = ParOpt.Optimizer(polygon, options)
    opt.optimize()

