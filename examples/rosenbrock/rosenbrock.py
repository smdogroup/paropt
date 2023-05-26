# Import some utilities
import numpy as np
import mpi4py.MPI as MPI
import matplotlib.pyplot as plt

# Import ParOpt
from paropt import ParOpt


# Create the rosenbrock function class
class Rosenbrock(ParOpt.Problem):
    def __init__(self):
        # Set the communicator pointer
        self.comm = MPI.COMM_WORLD
        self.nvars = 2
        self.ncon = 1

        # The design history file
        self.x_hist = []

        # Initialize the base class
        super(Rosenbrock, self).__init__(self.comm, nvars=self.nvars, ncon=self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""
        x[:] = -2.0 + np.random.uniform(size=len(x))
        lb[:] = -2.0
        ub[:] = 2.0
        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        con = np.zeros(1)
        fobj = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
        con[0] = x[0] + x[1] + 5.0
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        fail = 0

        # The objective gradient
        g[0] = -400.0 * (x[1] - x[0] ** 2) * x[0] - 2.0 * (1.0 - x[0])
        g[1] = 200.0 * (x[1] - x[0] ** 2)

        # The constraint gradient
        A[0][0] = -1.0
        A[0][1] = -1.0
        return fail


def plot_it_all(problem):
    """
    Plot a carpet plot with the search histories for steepest descent,
    conjugate gradient and BFGS from the same starting point.
    """

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
    levels = np.min(r) + np.linspace(0, 1.0, 75) ** 2 * (np.max(r) - np.min(r))

    # Create the plot
    fig = plt.figure(facecolor="w")
    plt.contour(x1, x1, r, levels)

    colours = ["-bo", "-ko", "-co", "-mo", "-yo", "-bx", "-kx", "-cx", "-mx", "-yx"]

    options = {
        "algorithm": "tr",
        "qn_type": "bfgs",
        "qn_update_type": "damped_update",
        "tr_init_size": 0.5,
        "tr_min_size": 1e-6,
        "tr_max_size": 10.0,
        "tr_eta": 0.1,
        "tr_adaptive_gamma_update": True,
        "tr_max_iterations": 200,
    }

    for k in range(len(colours)):
        # Optimize the problem
        problem.x_hist = []
        rosen.checkGradients(1e-6)

        opt = ParOpt.Optimizer(rosen, options)
        opt.optimize()

        # Copy out the steepest descent points
        sd = np.zeros((2, len(problem.x_hist)))
        for i in range(len(problem.x_hist)):
            sd[0, i] = problem.x_hist[i][0]
            sd[1, i] = problem.x_hist[i][1]

        plt.plot(sd[0, :], sd[1, :], colours[k], label="IP %d" % (sd.shape[1]))
        plt.plot(sd[0, -1], sd[1, -1], "-ro")

    plt.legend()
    plt.axis([xlow, xhigh, xlow, xhigh])
    plt.show()


# Create the Rosenbrock problem class
rosen = Rosenbrock()

plot_it_all(rosen)
