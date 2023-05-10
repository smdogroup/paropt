from __future__ import print_function

# Import this for directory creation
import os

# Import the greatest common divisor code
from math import gcd

# Import MPI
import mpi4py.MPI as MPI

# Import ParOpt
from paropt import ParOpt

# Import argparse
import argparse

# Import parts of matplotlib for plotting
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the truss analysis problem
from truss_analysis import TrussAnalysis


def get_ground_structure(N=4, M=4, L=2.5, P=10.0, n=5):
    """
    Set up the connectivity for a ground structure consisting of a 2D
    mesh of (N x M) nodes of completely connected elements.

    A single point load is applied at the lower right-hand-side of the
    mesh with a value of P. All nodes are evenly spaced at a distance
    L.

    input:
    N:  number of nodes in the x-direction
    M:  number of nodes in the y-direction
    L:  distance between nodes

    returns
    conn:   the connectivity
    xpos:   the nodal locations
    loads:  the loading conditions applied to the problem
    P:      the load applied to the mesh
    """

    # First, generate a co-prime grid that will be used to
    grid = []
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            if gcd(x, y) == 1:
                grid.append((x, y))

    # Reflect the ge
    reflect = []
    for d in grid:
        reflect.append((-d[0], d[1]))

    grid.extend(reflect)
    grid.extend([(0, 1), (1, 0)])

    # Set up the connectivity array
    conn = []
    for i in range(N):
        for j in range(M):
            n1 = i + N * j
            for d in grid:
                if (
                    (i + d[0] < N)
                    and (j + d[1] < M)
                    and (i + d[0] >= 0)
                    and (j + d[1] >= 0)
                ):
                    n2 = i + d[0] + (j + d[1]) * N
                    conn.append([n1, n2])

    # Set the positions of all the nodes
    xpos = []
    for j in range(M):
        for i in range(N):
            xpos.extend([i * L, j * L])

    # Set the node locations
    loads = {}
    loads[N - 1] = [0, -P]

    bcs = {}
    for j in range(M):
        bcs[j * N] = [0, 1]

    return conn, xpos, loads, bcs


def setup_ground_struct(N, M, L=2.5, E=70e9, rho=2700.0, A_min=5e-4, A_max=10.0):
    """
    Create a ground structure with a given number of nodes and
    material properties.
    """

    # Create the ground structure
    conn, xpos, loads, bcs = get_ground_structure(N=N, M=M, L=L, P=10e3)

    # Set the scaling for the material variables
    Area_scale = 1.0

    # Set the fixed mass constraint
    mass_fixed = 5.0 * N * M * L * rho

    # Create the truss topology optimization object
    truss = TrussAnalysis(
        conn, xpos, loads, bcs, E, rho, mass_fixed, A_min, A_max, Area_scale=Area_scale
    )

    return truss


def paropt_truss(truss, use_hessian=False, use_tr=False, prefix="results"):
    """
    Optimize the given truss structure using ParOpt
    """

    fname = os.path.join(prefix, "truss_paropt%dx%d.out" % (N, M))
    options = {
        "algorithm": "ip",
        "qn_subspace_size": 10,
        "abs_res_tol": 1e-5,
        "norm_type": "l1",
        "init_barrier_param": 10.0,
        "monotone_barrier_fraction": 0.25,
        "barrier_strategy": "monotone",  # "complementarity_fraction",
        "starting_point_strategy": "least_squares_multipliers",
        "use_hvec_product": True,
        "gmres_subspace_size": 50,
        "nk_switch_tol": 1e3,
        "eisenstat_walker_gamma": 0.01,
        "eisenstat_walker_alpha": 0.0,
        "max_gmres_rtol": 1.0,
        "output_level": 1,
        "armijo_constant": 1e-5,
        "gradient_verification_frequency": 2,
        "output_file": fname,
    }

    if use_tr:
        options["algorithm"] = "tr"
        options["abs_res_tol"] = 1e-8
        options["barrier_strategy"] = "monotone"
        options["tr_max_size"] = 100.0
        options["tr_linfty_tol"] = 1e-5
        options["tr_l1_tol"] = 0.0
        options["tr_max_iterations"] = 2000
        options["tr_penalty_gamma_max"] = 1e6
        options["tr_adaptive_gamma_update"] = True
        options["tr_output_file"] = fname.split(".")[0] + ".tr"

    if use_hessian is False:
        options["use_hvec_product"] = False

    opt = ParOpt.Optimizer(truss, options)
    opt.optimize()

    return opt


def pyopt_truss(truss, optimizer="snopt", options={}):
    """
    Take the given problem and optimize it with the given optimizer
    from the pyOptSparse library of optimizers.
    """
    # Import the optimization problem
    from pyoptsparse import Optimization, OPT

    class pyOptWrapper:
        def __init__(self, truss):
            self.truss = truss

        def objcon(self, x):
            fail, obj, con = self.truss.evalObjCon(x["x"])
            funcs = {"objective": obj, "con": con}
            return funcs, fail

        def gobjcon(self, x, funcs):
            g = np.zeros(x["x"].shape)
            A = np.zeros((1, x["x"].shape[0]))
            fail = self.truss.evalObjConGradient(x["x"], g, A)
            sens = {"objective": {"x": g}, "con": {"x": A}}
            return sens, fail

    # Set the design variables
    wrap = pyOptWrapper(truss)
    prob = Optimization("Truss", wrap.objcon)

    # Determine the initial variable values and their lower/upper
    # bounds in the design problem
    n = len(truss.conn)
    x0 = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    truss.getVarsAndBounds(x0, lower, upper)

    # Set the variable bounds and initial values
    prob.addVarGroup("x", n, value=x0, lower=lower, upper=upper)

    # Set the constraints
    prob.addConGroup("con", 1, lower=0.0, upper=0.0)

    # Add the objective
    prob.addObj("objective")

    # Optimize the problem
    try:
        opt = OPT(optimizer, options=options)
        sol = opt(prob, sens=wrap.gobjcon)
    except:
        opt = None
        sol = None

    return opt, prob, sol


def get_performance_profile(r, tau_max):
    """
    Get the performance profile for the given ratio
    """

    # Sort the ratios in increasing order
    r = sorted(r)

    # Find the first break-point at which tau >= 1.0
    n = 0
    while n < len(r) and r[n] <= 1.0:
        n += 1

    # Add the first part of the profile to the plot
    rho = [0.0, 1.0 * n / len(r)]
    tau = [1.0, 1.0]

    # Add all subsequent break-points to the plot
    while n < len(r) and r[n] < tau_max:
        rho.extend([1.0 * n / len(r), 1.0 * (n + 1) / len(r)])
        tau.extend([r[n], r[n]])
        n += 1

    # Finish off the profile to the max value
    rho.append(1.0 * n / len(r))
    tau.append(tau_max)

    return tau, rho


# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=4, help="Nodes in x-direction")
parser.add_argument("--M", type=int, default=3, help="Nodes in y-direction")
parser.add_argument(
    "--profile", action="store_true", default=False, help="Performance profile"
)
parser.add_argument(
    "--use_hessian",
    action="store_true",
    default=False,
    help="Use the exact Hessian-vector products",
)
parser.add_argument(
    "--use_tr",
    action="store_true",
    default=False,
    help="Use the trust region variant of the optimization algorithm",
)
parser.add_argument(
    "--optimizer", default="None", help="Optimizer name from pyOptSparse"
)
args = parser.parse_args()

# Get the arguments
N = args.N
M = args.M
profile = args.profile
use_hessian = args.use_hessian
use_tr = args.use_tr
optimizer = args.optimizer

# Set the options for the pyOptSparse optimizers for comparisons
all_options = {
    "slsqp": {"MAXIT": 5000},
    "snopt": {"Major iterations limit": 10000000, "Minor iterations limit": 10000000},
    "ipopt": {},
    "nsga2": {"PrintOut": 1},
    "alpso": {
        "fileout": 3,
        "maxOuterIter": 250,
        "stopCriteria": 1,
        "atol": 1e-6,
        "Scaling": 1,
    },
}

# Set the output file name
outfile_names = {
    "None": None,
    "slsqp": "IFILE",
    "snopt": "Print file",
    "ipopt": "output_file",
    "alpso": "filename",
    "nsga2": None,
}
outfile_name = outfile_names[optimizer]

if profile:
    # Set the trusses that will be optimized
    trusses = [
        [3, 3],
        [4, 3],
        [5, 3],
        [6, 3],
        [4, 4],
        [5, 4],
        [6, 4],
        [7, 4],
        [5, 5],
        [6, 5],
        [7, 5],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [9, 5],
        [9, 6],
        [9, 7],
        [9, 8],
        [9, 9],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
    ]

    # Perform the optimization with and without the Hessian
    if optimizer == "None":
        # Set the prefix to use
        if use_hessian:
            prefix = "hessian"
        elif use_tr:
            prefix = "tr"
        else:
            prefix = "bfgs"
    else:
        prefix = optimizer

    # Get to the full path
    prefix = os.path.join(os.getcwd(), prefix)

    # Create the directory if it does not yet exist
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    fname = os.path.join(prefix, "performance_profile.dat")
    fp = open(fname, "w")

    # Iterate over all the trusses
    index = 0
    for vals in trusses:
        # Set the values of N/M
        N = vals[0]
        M = vals[1]

        print("Optimizing truss (%d x %d) ..." % (N, M))

        # Optimize each of the trusses
        truss = setup_ground_struct(N, M)
        t0 = MPI.Wtime()
        if optimizer == "None":
            opt = paropt_truss(
                truss, prefix=prefix, use_tr=use_tr, use_hessian=use_hessian
            )

            # Get the optimized point
            x, z, zw, zl, zu = opt.getOptimizedPoint()
        else:
            # Read out the options from the dictionary of options
            options = all_options[optimizer]

            # Set the output file
            filename = os.path.join(prefix, "output_%dx%d.out" % (N, M))
            options[outfile_name] = filename

            # Optimize the truss with the specified optimizer
            opt, prob, sol = pyopt_truss(truss, optimizer=optimizer, options=options)

            # Extract the design variable values
            if sol is not None:
                x = []
                for var in sol.variables["x"]:
                    x.append(var.value)
                x = np.array(x)
            else:
                x = None

        # Keep track of the optimization time
        t0 = MPI.Wtime() - t0

        # Plot the truss
        filename = os.path.join(prefix, "opt_truss%dx%d.pdf" % (N, M))
        if x is not None:
            truss.plotTruss(x, tol=1e-1, filename=filename)

        # Record the performance of the algorithm
        fp.write(
            "%d %d %d %d %d %d %e\n"
            % (
                index,
                len(truss.conn),
                truss.fevals,
                truss.gevals,
                truss.hevals,
                truss.fevals + truss.hevals,
                t0,
            )
        )
        fp.flush()
        index += 1

    # Close the file
    fp.close()

    profiles = ["bfgs", "hessian", "snopt", "slsqp", "alpso"]
    colours = ["g", "b", "r", "k", "c"]

    # Read the performance values from each file
    perform = 1e3 * np.ones((len(trusses), len(profiles)))
    index = 0
    for index, prefix in enumerate(profiles):
        fname = os.path.join(prefix, "performance_profile.dat")
        if os.path.isfile(fname):
            perf = np.loadtxt(fname)

            # Set the performance metric
            for i in range(perf.shape[0]):
                perform[i, index] = perf[i, -1]

    # Create the performance profiles
    nprob = len(trusses)
    r = np.zeros(perform.shape)

    # Compute the ratios for the best performance
    for i in range(nprob):
        best = 1.0 * min(perform[i, :])
        r[i, :] = perform[i, :] / best

    # Plot the data
    fig = plt.figure(facecolor="w")

    tau_max = 10.0
    for k in range(len(profiles)):
        tau, rho = get_performance_profile(r[:, k], tau_max)
        plt.plot(tau, rho, colours[k], linewidth=2, label=profiles[k])

    # Finish off the plot and print it
    plt.legend(loc="lower right")
    plt.axis([0.95, tau_max, 0.0, 1.1])
    filename = "performance_profile.pdf"
    plt.savefig(filename)
    plt.close()
else:
    # Optimize the structure
    prefix = "results"
    truss = setup_ground_struct(N, M)

    if optimizer == "None":
        opt = paropt_truss(truss, prefix=prefix, use_tr=use_tr, use_hessian=use_hessian)

        # Retrieve the optimized multipliers
        x, z, zw, zl, zu = opt.getOptimizedPoint()
        print("z =  ", z)
    else:
        # Read out the options from the dictionary of options
        options = all_options[optimizer]

        # Set the output file
        if outfile_name is not None:
            options[outfile_name] = os.path.join(prefix, "output_%dx%d.out" % (N, M))
        # Optimize the truss with the specified optimizer
        opt, prob, sol = pyopt_truss(truss, optimizer=optimizer, options=options)

        # Extract the design variable values
        x = []
        for var in sol.variables["x"]:
            x.append(var.value)
        x = np.array(x)

    # Plot the truss
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    truss.plotTruss(x, tol=0.1, filename=prefix + "/opt_truss%dx%d.pdf" % (N, M))
