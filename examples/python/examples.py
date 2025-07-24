import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import argparse
import os

# Import ParOpt
from paropt import ParOpt


class Problem1(ParOpt.Problem):
    def __init__(self):
        self.x_hist = []
        super(Problem1, self).__init__(MPI.COMM_SELF, nvars=2, ncon=1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = -3.0
        ub[:] = 3.0
        x[:] = -2.0 + 4.0 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint values"""

        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        fobj = 2 * x[0] ** 2 + 2 * x[1] ** 2 + x[0] * x[1]
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 4 * x[0] + x[1]
        g[1] = 4 * x[1] + x[0]
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail


class Problem2(ParOpt.Problem):
    def __init__(self):
        self.x_hist = []
        super(Problem2, self).__init__(MPI.COMM_SELF, nvars=2, ncon=1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = -3.0
        ub[:] = 3.0
        x[:] = -2.0 + 4.0 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint values"""

        # Append the point to the history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        fobj = x[0] ** 4 + x[1] ** 2 + 2 * x[0] * x[1] - x[0] - x[1]
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 4 * x[0] ** 3 + 2 * x[1] - 1.0
        g[1] = 2 * x[1] + 2 * x[0] - 1.0
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail


class Problem3(ParOpt.Problem):
    def __init__(self):
        self.x_hist = []
        super(Problem3, self).__init__(MPI.COMM_SELF, nvars=2, ncon=1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = -3.0
        ub[:] = 3.0
        x[:] = -2.0 + 4.0 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint values"""

        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        fobj = x[0] ** 4 + x[1] ** 4 + 1 - x[0] ** 2 - x[1] ** 2
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 4 * x[0] ** 3 - 2 * x[0]
        g[1] = 4 * x[1] ** 3 - 2 * x[1]
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail


class Problem4(ParOpt.Problem):
    def __init__(self):
        self.x_hist = []
        super(Problem4, self).__init__(MPI.COMM_SELF, nvars=2, ncon=1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = -3.0
        ub[:] = 3.0
        x[:] = -2.0 + 4.0 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint values"""

        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        fobj = (
            -10 * x[0] ** 2
            + 10 * x[1] ** 2
            + 4 * np.sin(x[0] * x[1])
            - 2 * x[0]
            + x[0] ** 4
        )
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = -20 * x[0] + 4 * np.cos(x[0] * x[1]) * x[1] - 2.0 + 4 * x[0] ** 3
        g[1] = 20 * x[1] + 4 * np.cos(x[0] * x[1]) * x[0]
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail


class Problem5(ParOpt.Problem):
    def __init__(self):
        self.x_hist = []
        super(Problem5, self).__init__(MPI.COMM_SELF, nvars=2, ncon=1)
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = -3.0
        ub[:] = 3.0
        x[:] = -2.0 + 4.0 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint values"""

        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        fobj = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        cons = np.array([x[0] + x[1] - 0.5])
        return fail, fobj, cons

    def evalObjConGradient(self, x, g, A):
        fail = 0
        g[0] = 200 * (x[1] - x[0] ** 2) * (-2 * x[0]) - 2 * (1 - x[0])
        g[1] = 200 * (x[1] - x[0] ** 2)
        A[0][0] = 1.0
        A[0][1] = 1.0
        return fail


def plot_it_all(problem, use_tr=False):
    """
    Plot a carpet plot with the search histories for steepest descent,
    conjugate gradient and BFGS from the same starting point.
    """

    # Check the problem gradients
    problem.checkGradients(1e-6)

    # Create the data for the carpet plot
    n = 150
    xlow = -4.0
    xhigh = 4.0
    x1 = np.linspace(xlow, xhigh, n)

    ylow = -3.0
    yhigh = 3.0
    x2 = np.linspace(ylow, yhigh, n)
    r = np.zeros((n, n))

    for j in range(n):
        for i in range(n):
            fail, fobj, con = problem.evalObjCon([x1[i], x2[j]])
            r[j, i] = fobj

    # Assign the contour levels
    levels = np.min(r) + np.linspace(0, 1.0, 75) ** 2 * (np.max(r) - np.min(r))

    # Create the plot
    fig = plt.figure(facecolor="w")
    plt.contour(x1, x2, r, levels)
    plt.plot([0.5 - yhigh, 0.5 - ylow], [yhigh, ylow], "-k")

    colours = ["-bo", "-ko", "-co", "-mo", "-yo", "-bx", "-kx", "-cx", "-mx", "-yx"]

    for k in range(len(colours)):
        # Optimize the problem
        problem.x_hist = []

        # Set the trust region parameters
        filename = "paropt.out"

        options = {
            "algorithm": "ip",
            "output_file": filename,
            "abs_res_tol": 1e-8,
            "starting_point_strategy": "affine_step",
            "barrier_strategy": "monotone",
            "start_affine_multiplier_min": 0.01,
            "penalty_gamma": 100.0,
            "qn_subspace_size": 10,
            "qn_type": "bfgs",
        }

        if use_tr:
            options = {
                "algorithm": "tr",
                "tr_init_size": 0.05,
                "tr_min_size": 1e-6,
                "tr_max_size": 10.0,
                "tr_eta": 0.25,
                "penalty_gamma": 10.0,
                "qn_subspace_size": 10,
                "qn_type": "bfgs",
                "abs_res_tol": 1e-8,
                "output_file": filename,
                "tr_output_file": os.path.splitext(filename)[0] + ".tr",
                "starting_point_strategy": "affine_step",
                "barrier_strategy": "monotone",
                "use_line_search": False,
            }

        opt = ParOpt.Optimizer(problem, options)

        # Set a new starting point
        opt.optimize()
        x, z, zw, zl, zu = opt.getOptimizedPoint()

        # Copy out the steepest descent points
        popt = np.zeros((2, len(problem.x_hist)))
        for i in range(len(problem.x_hist)):
            popt[0, i] = problem.x_hist[i][0]
            popt[1, i] = problem.x_hist[i][1]

        plt.plot(
            popt[0, :], popt[1, :], colours[k], label="ParOpt %d" % (popt.shape[1])
        )
        plt.plot(popt[0, -1], popt[1, -1], "-ro")

        # Print the data to the screen
        g = np.zeros(2)
        A = np.zeros((1, 2))
        problem.evalObjConGradient(x, g, A)

        print("The design variables:    ", x[:])
        print("The multipliers:         ", z[:])
        print("The objective gradient:  ", g[:])
        print("The constraint gradient: ", A[:])

    ax = fig.axes[0]
    ax.set_aspect("equal", "box")
    plt.legend()


# Allocate the problems
problems = [Problem1(), Problem2(), Problem3(), Problem4(), Problem5()]

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default="ip", help="optimizer type")
args = parser.parse_args()

# Use a consistent seed for consistent results
np.random.seed(0)

use_tr = False
if args.algorithm != "ip":
    use_tr = True

for problem in problems:
    plot_it_all(problem, use_tr)

# Show the results at the end
plt.show()
