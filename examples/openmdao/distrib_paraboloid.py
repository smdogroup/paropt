import argparse
import sys

import numpy as np
from mpi4py import MPI

import openmdao.api as om
from paropt.paropt_driver import ParOptDriver

"""
Example to demonstrate parallel optimization with OpenMDAO
using distributed components

Minimize: y = Sum((x - 5)^2) + (w - 10)^2
w.r.t. x
subject to: a = Sum(x^3) <= 10.0

The size of x depends on the number of procs used:
size(x) = 2*num_procs + 1
"""


class DistribParaboloid(om.ExplicitComponent):
    def setup(self):
        self.options["distributed"] = True

        if self.comm.rank == 0:
            ndvs = 3
        else:
            ndvs = 2

        self.add_input("w", val=1.0)  # this will connect to a non-distributed IVC
        self.add_input("x", shape=ndvs)  # this will connect to a distributed IVC

        self.add_output("y", shape=1)  # all-gathered output, duplicated on all procs
        self.add_output("z", shape=ndvs)  # distributed output
        self.add_output("a", shape=1)  # all-gathered output, duplicated on all procs
        self.declare_partials("y", "x")
        self.declare_partials("y", "w")
        self.declare_partials("z", "x")
        self.declare_partials("a", "x")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        local_y = np.sum((x - 5) ** 2)
        y_g = np.zeros(self.comm.size)
        self.comm.Allgather(local_y, y_g)
        outputs["y"] = np.sum(y_g) + (inputs["w"] - 10) ** 2

        z = x**3
        outputs["z"] = z

        local_a = np.sum(z)
        a_g = np.zeros(self.comm.size)
        self.comm.Allgather(local_a, a_g)
        outputs["a"] = np.sum(a_g)

    def compute_partials(self, inputs, J):
        x = inputs["x"]
        J["y", "x"] = 2.0 * (x - 5.0)
        J["y", "w"] = 2.0 * (inputs["w"] - 10.0)
        J["z", "x"] = np.diag(2.0 * x)
        J["a", "x"] = 3.0 * x * x


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--driver",
        default="paropt",
        choices=["paropt", "scipy", "pyoptsparse"],
        help="driver",
    )
    parser.add_argument(
        "--algorithm", default="ip", choices=["ip", "tr"], help="optimizer type"
    )
    args = parser.parse_args()
    driver = args.driver
    algorithm = args.algorithm

    comm = MPI.COMM_WORLD

    # Build the model
    p = om.Problem()

    # Set the number of design variables on each processor
    if comm.rank == 0:
        ndvs = 3
    else:
        ndvs = 2

    # Define the independent variables that are distributed
    d_ivc = p.model.add_subsystem(
        "distrib_ivc", om.IndepVarComp(distributed=True), promotes=["*"]
    )
    d_ivc.add_output("x", 2 * np.ones(ndvs))

    # Define the independent variables that are non-distributed
    # These non-distributed variables will be duplicated on each processor
    ivc = p.model.add_subsystem(
        "ivc", om.IndepVarComp(distributed=False), promotes=["*"]
    )
    ivc.add_output("w", 2.0)

    # Add the paraboloid model
    p.model.add_subsystem("dp", DistribParaboloid(), promotes=["*"])

    # Define the optimization problem
    p.model.add_design_var("x", upper=10.0)
    p.model.add_objective("y")
    p.model.add_constraint("a", upper=10.0)

    # Create and set the driver
    if driver == "paropt":
        p.driver = ParOptDriver()
        p.driver.options["algorithm"] = algorithm
    elif driver == "scipy":
        p.driver = ScipyOptimizeDriver()
    elif driver == "pyoptsparse":
        p.driver = pyOptSparseDriver()
        p.driver.options["optimizer"] = "ParOpt"

    p.setup()
    p.run_driver()

    # Print the objective and constraint values at the optimized point
    if comm.rank == 0:
        print("f = {0:.2f}".format(p.get_val("dp.y")[0]))
        print("c = {0:.2f}".format(p.get_val("dp.a")[0] - 10.0))

    # Print the x location of the minimum
    print("Rank = {0}; x = {1}".format(comm.rank, p.get_val("dp.x")))
