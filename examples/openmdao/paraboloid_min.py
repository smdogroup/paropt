from openmdao.api import (
    Problem,
    ScipyOptimizeDriver,
    pyOptSparseDriver,
    ExecComp,
    IndepVarComp,
)
from paropt.paropt_driver import ParOptDriver
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--driver",
    default="paropt",
    choices=["paropt", "scipy", "pyoptsparse"],
    help="driver",
)
parser.add_argument(
    "--algorithm", default="ip", choices=["ip", "tr", "mma"], help="optimizer type"
)
args = parser.parse_args()
driver = args.driver
algorithm = args.algorithm

# Build the model
prob = Problem()

# Define the independent variables
indeps = prob.model.add_subsystem("indeps", IndepVarComp())
indeps.add_output("x", 3.0)
indeps.add_output("y", -4.0)

# Define the objective and the constraint functions
prob.model.add_subsystem("paraboloid", ExecComp("f = (x-3)**2 + x*y + (y+4)**2 - 3"))
prob.model.add_subsystem("con", ExecComp("c = x**2 + y**2"))

# Connect the model
prob.model.connect("indeps.x", "paraboloid.x")
prob.model.connect("indeps.y", "paraboloid.y")
prob.model.connect("indeps.x", "con.x")
prob.model.connect("indeps.y", "con.y")

# Define the optimization problem
prob.model.add_design_var("indeps.x", lower=-50, upper=50)
prob.model.add_design_var("indeps.y", lower=-50, upper=50)
prob.model.add_objective("paraboloid.f")
prob.model.add_constraint("con.c", equals=27.0)

# Create and set the ParOpt driver
if driver == "paropt":
    prob.driver = ParOptDriver()
    prob.driver.options["algorithm"] = algorithm
elif driver == "scipy":
    prob.driver = ScipyOptimizeDriver()
elif driver == "pyoptsparse":
    prob.driver = pyOptSparseDriver()
    prob.driver.options["optimizer"] = "ParOpt"

# Run the problem
prob.setup()
prob.run_driver()

# Print the minimum value
print("Minimum value = {fmin:.2f}".format(fmin=prob["paraboloid.f"][0]))

# Print the x/y location of the minimum
print(
    "(x, y) = ({x:.2f}, {y:.2f})".format(x=prob["indeps.x"][0], y=prob["indeps.y"][0])
)
print("x**2 + y**2 = ", prob["indeps.x"][0] ** 2 + prob["indeps.y"][0] ** 2)
