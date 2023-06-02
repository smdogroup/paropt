"""
ref: Benchmarking Optimization Software with COPS 3.0
problem 2. Distribution of Electrons on a Sphere
n: number of electrons
dv: xi, yi, zi, i=0,1,...,n-1
max    sum of ((x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2)^{-1/2}
where  i=0,1, ...,n-2; j=i+1,...,n-1
s.t.   c = x_i^2 + y_i^2 + z_i^2 - 1 = 0, i = 0,1,...,n-1
"""

from paropt import ParOpt
import mpi4py.MPI as MPI
import numpy as np
import argparse


class Electron(ParOpt.Problem):
    def __init__(self, n, epsilon):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.n = n
        self.nvars = 3 * n
        self.num_sparse_constraints = n
        self.epsilon = epsilon

        rowp = [0]
        cols = []
        for i in range(self.n):
            cols.extend([i, n + i, 2 * n + i])
            rowp.append(len(cols))

        # Initialize the base class
        super(Electron, self).__init__(
            self.comm,
            nvars=self.nvars,
            num_sparse_constraints=self.num_sparse_constraints,
            num_sparse_inequalities=0,
            rowp=rowp,
            cols=cols,
        )

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        n = self.n

        # x = [x_1, ..., x_n, y_1, ..., y_n, z_1, ..., z_n]
        np.random.seed(0)
        alpha = np.random.uniform(low=0.0, high=2 * np.pi, size=n)
        beta = np.random.uniform(low=-np.pi, high=np.pi, size=n)
        for i in range(n):
            x[i] = np.cos(beta[i]) * np.cos(alpha[i])
            x[n + i] = np.cos(beta[i]) * np.sin(alpha[i])
            x[2 * n + i] = np.sin(beta[i])

        lb[:] = -10.0
        ub[:] = 10.0

        return

    def evalSparseObjCon(self, x, sparse_cons):
        """Evaluate the objective and constraint"""
        n = self.n
        epsilon = self.epsilon
        _x = x[:n]
        _y = x[n : 2 * n]
        _z = x[2 * n :]

        fobj = 0.0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                dsq = (_x[i] - _x[j]) ** 2 + (_y[i] - _y[j]) ** 2 + (_z[i] - _z[j]) ** 2
                if dsq < epsilon:
                    dsq = epsilon
                fobj += dsq ** (-1 / 2)

        for i in range(n):
            sparse_cons[i] = 1.0 - (_x[i] ** 2 + _y[i] ** 2 + _z[i] ** 2)

        con = []
        fail = 0

        return fail, fobj, con

    def evalSparseObjConGradient(self, x, g, A, data):
        """Evaluate the objective and constraint gradient"""
        n = self.n
        epsilon = self.epsilon
        _x = x[:n]
        _y = x[n : 2 * n]
        _z = x[2 * n :]

        g[:] = 0.0
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                dsq = (_x[i] - _x[j]) ** 2 + (_y[i] - _y[j]) ** 2 + (_z[i] - _z[j]) ** 2
                if dsq < epsilon:
                    dsq = epsilon
                else:
                    fact = dsq ** (-3 / 2)
                    g[i] += -(_x[i] - _x[j]) * fact
                    g[j] += (_x[i] - _x[j]) * fact
                    g[n + i] += -(_y[i] - _y[j]) * fact
                    g[n + j] += (_y[i] - _y[j]) * fact
                    g[2 * n + i] += -(_z[i] - _z[j]) * fact
                    g[2 * n + j] += (_z[i] - _z[j]) * fact

        for i in range(n):
            data[3 * i] = -2.0 * _x[i]
            data[3 * i + 1] = -2.0 * _y[i]
            data[3 * i + 2] = -2.0 * _z[i]

        fail = 0

        return fail


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="tr")
    parser.add_argument("--n", type=int, default=10, help="number of electron")
    args = parser.parse_args()

    use_tr = False
    if args.algorithm != "ip":
        use_tr = True

    # use interior point algorithm
    options = {
        "algorithm": "ip",
        "norm_type": "infinity",
        "qn_type": "bfgs",
        "qn_subspace_size": 10,
        "starting_point_strategy": "least_squares_multipliers",
        "qn_update_type": "damped_update",
        "abs_res_tol": 1e-6,
        "barrier_strategy": "monotone",
        "armijo_constant": 1e-5,
        "penalty_gamma": 100.0,
        "max_major_iters": 500,
    }

    # use trust region algorithm
    if use_tr:
        options = {
            "algorithm": "tr",
            "output_level": 0,
            "tr_l1_tol": 1e-30,
            "tr_linfty_tol": 1e-30,
            "tr_init_size": 0.05,
            "tr_min_size": 1e-6,
            "tr_max_size": 1e3,
            "tr_eta": 0.25,
            "tr_adaptive_gamma_update": False,
            "tr_accept_step_strategy": "penalty_method",
            "tr_use_soc": False,
            "tr_max_iterations": 200,
            "max_major_iters": 100,
        }

    problem = Electron(args.n, 1e-15)
    problem.checkGradients()
    opt = ParOpt.Optimizer(problem, options)
    opt.optimize()
