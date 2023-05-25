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


class DensePolygon(ParOpt.Problem):
    def __init__(self, nv):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nv = nv
        self.nr = nv
        self.ntheta = nv
        self.nvars = 2 * nv
        self.nc1 = nv - 1
        self.nc2 = int(nv * (nv - 1) / 2)
        self.num_dense_constraints = self.nc1 + self.nc2

        # Initialize the base class
        super(DensePolygon, self).__init__(
            self.comm,
            nvars=self.nvars,
            num_dense_constraints=self.num_dense_constraints,
        )

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        nv = self.nv

        np.random.seed(0)
        x[:nv] = np.random.uniform(low=0.1, high=9.9, size=nv)
        lb[:nv] = 0.0
        ub[:nv] = 10.0

        x[nv:] = np.linspace(start=0.1 * np.pi, stop=0.9 * np.pi, num=nv)
        lb[nv:] = 0.0
        ub[nv:] = np.pi

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        fail = 0
        nv = self.nv
        r = x[:nv]
        theta = x[nv:]

        # evaluate obj and c1
        fobj = 0
        for i in range(nv - 1):
            fobj -= 0.5 * r[i] * r[i + 1] * np.sin(theta[i + 1] - theta[i])

        cons = np.zeros(self.num_dense_constraints)
        index = 0
        for i in range(nv - 1):
            cons[index] = theta[i + 1] - theta[i]
            index += 1

        for i in range(nv - 1):
            for j in range(i + 1, nv):
                coef = 2.0 * r[i] * r[j] * np.cos(theta[i] - theta[j])
                cons[index] = 1.0 - r[i] ** 2 - r[j] ** 2 + coef
                index += 1

        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0
        nv = self.nv
        r = x[:nv]
        theta = x[nv:]

        # Compute g
        g[:] = 0.0
        for i in range(nv - 1):
            g[i] -= 0.5 * r[i + 1] * np.sin(theta[i + 1] - theta[i])
            g[i + 1] -= 0.5 * r[i] * np.sin(theta[i + 1] - theta[i])
            g[nv + i] -= -0.5 * r[i] * r[i + 1] * np.cos(theta[i + 1] - theta[i])
            g[nv + i + 1] -= 0.5 * r[i] * r[i + 1] * np.cos(theta[i + 1] - theta[i])

        # Compute A1t
        index = 0
        col_index = 0
        for i in range(nv - 1):
            A[index][nv + i] = -1.0
            A[index][nv + i + 1] = 1.0
            index += 1

        # Compute A2r and A2t
        for i in range(nv - 1):
            for j in range(i + 1, nv):
                # Derivatives w.r.t. r
                ccoef = np.cos(theta[i] - theta[j])
                A[index][i] = -2.0 * r[i] + 2.0 * r[j] * ccoef
                A[index][j] = -2.0 * r[j] + 2.0 * r[i] * ccoef

                # Derivatives w.r.t. theta
                scoef = np.sin(theta[i] - theta[j])
                A[index][self.nv + i] = -2.0 * r[i] * r[j] * scoef
                A[index][self.nv + j] = 2.0 * r[i] * r[j] * scoef
                index += 1

        return fail


class Polygon(ParOpt.Problem):
    def __init__(self, nv):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nv = nv
        self.nr = nv
        self.ntheta = nv
        self.nvars = 2 * nv
        self.nc1 = nv - 1
        self.nc2 = int(nv * (nv - 1) / 2)
        self.num_sparse_constraints = self.nc1 + self.nc2

        rowp = [0]
        cols = []

        for i in range(self.nv - 1):
            cols.extend([self.nv + i, self.nv + i + 1])
            rowp.append(len(cols))

        for i in range(self.nv - 1):
            for j in range(i + 1, self.nv):
                cols.extend([i, j, self.nv + i, self.nv + j])
                rowp.append(len(cols))

        # Initialize the base class
        super(Polygon, self).__init__(
            self.comm,
            nvars=self.nvars,
            num_sparse_constraints=self.num_sparse_constraints,
            rowp=rowp,
            cols=cols,
        )

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        nv = self.nv

        np.random.seed(0)
        x[:nv] = np.random.uniform(low=0.1, high=9.9, size=nv)
        lb[:nv] = 0.0
        ub[:nv] = 10.0

        x[nv:] = np.linspace(start=0.1 * np.pi, stop=0.9 * np.pi, num=nv)
        lb[nv:] = 0.0
        ub[nv:] = np.pi

        return

    def evalSparseObjCon(self, x, sparse_cons):
        """Evaluate the objective and constraint"""
        fail = 0
        nv = self.nv
        con = []
        r = x[:nv]
        theta = x[nv:]

        # evaluate obj and c1
        fobj = 0
        for i in range(nv - 1):
            fobj -= 0.5 * r[i] * r[i + 1] * np.sin(theta[i + 1] - theta[i])

        index = 0
        for i in range(nv - 1):
            sparse_cons[index] = theta[i + 1] - theta[i]
            index += 1

        for i in range(nv - 1):
            for j in range(i + 1, nv):
                coef = 2.0 * r[i] * r[j] * np.cos(theta[i] - theta[j])
                sparse_cons[index] = 1.0 - r[i] ** 2 - r[j] ** 2 + coef
                index += 1

        return fail, fobj, con

    def evalSparseObjConGradient(self, x, g, A, data):
        """Evaluate the objective and constraint gradient"""
        fail = 0
        nv = self.nv
        r = x[:nv]
        theta = x[nv:]

        # Compute g
        g[:] = 0.0
        for i in range(nv - 1):
            g[i] -= 0.5 * r[i + 1] * np.sin(theta[i + 1] - theta[i])
            g[i + 1] -= 0.5 * r[i] * np.sin(theta[i + 1] - theta[i])
            g[nv + i] -= -0.5 * r[i] * r[i + 1] * np.cos(theta[i + 1] - theta[i])
            g[nv + i + 1] -= 0.5 * r[i] * r[i + 1] * np.cos(theta[i + 1] - theta[i])

        # Compute A1t
        col_index = 0
        for i in range(nv - 1):
            data[col_index] = -1.0
            data[col_index + 1] = 1.0
            col_index += 2

        # Compute A2r and A2t
        for i in range(nv - 1):
            for j in range(i + 1, nv):
                # Derivatives w.r.t. r
                ccoef = np.cos(theta[i] - theta[j])
                data[col_index] = -2.0 * r[i] + 2.0 * r[j] * ccoef
                data[col_index + 1] = -2.0 * r[j] + 2.0 * r[i] * ccoef

                # Derivatives w.r.t. theta
                scoef = np.sin(theta[i] - theta[j])
                data[col_index + 2] = -2.0 * r[i] * r[j] * scoef
                data[col_index + 3] = 2.0 * r[i] * r[j] * scoef
                col_index += 4

        return fail


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="tr")
    parser.add_argument("--n", type=int, default=10, help="number of vertices")
    args = parser.parse_args()

    use_tr = False
    if args.algorithm != "ip":
        use_tr = True

    # use interior point algorithm
    options = {
        "algorithm": "ip",
        "qn_type": "bfgs",
        "qn_update_type": "damped_update",
        "qn_subspace_size": 10,
        "abs_res_tol": 1e-6,
        "barrier_strategy": "monotone",
        "starting_point_strategy": "affine_step",
        "output_level": 1,
        "max_major_iters": 1000,
    }

    # use trust region algorithm
    if use_tr:
        options = {
            "algorithm": "tr",
            "qn_type": "bfgs",
            "tr_l1_tol": 1e-30,
            "tr_linfty_tol": 1e-30,
            "output_level": 0,
            "max_major_iters": 100,
            "tr_init_size": 0.1,
            "tr_min_size": 1e-6,
            "tr_max_size": 1e2,
            "tr_eta": 0.25,
            "penalty_gamma": 1e2,
            "tr_adaptive_gamma_update": False,
            "tr_accept_step_strategy": "penalty_method",
            "tr_use_soc": False,
            "tr_penalty_gamma_max": 1e5,
            "tr_penalty_gamma_min": 1e-5,
            "tr_max_iterations": 200,
            # 'use_backtracking_alpha': True,
            # 'use_line_search': False
        }

    polygon = Polygon(args.n)
    polygon.checkGradients()
    opt = ParOpt.Optimizer(polygon, options)
    opt.optimize()
