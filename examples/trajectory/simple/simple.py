import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import argparse

import sys

sys.path.append("../cart_pole_dymos")
from test_driver import ParOptTestDriver


class SimpleODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # Inputs
        self.add_input("m", val=1.0, desc="mass", units="kg")
        self.add_input("v", val=np.zeros(nn), desc="velocity", units="m/s")
        self.add_input("u", val=np.zeros(nn), desc="control force", units="N")
        self.add_output(
            "xdot", val=np.zeros(nn), desc="horizontal velocity", units="m/s"
        )
        self.add_output(
            "vdot", val=np.zeros(nn), desc="acceleration mag.", units="m/s**2"
        )
        self.add_output(
            "Jdot",
            val=np.zeros(nn),
            desc="time derivative of total control",
            units="N**2",
        )

        # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)
        self.declare_partials(of="xdot", wrt="v", rows=arange, cols=arange)
        self.declare_partials(of="vdot", wrt="u", rows=arange, cols=arange)
        self.declare_partials(of="Jdot", wrt="u", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        u = inputs["u"]
        v = inputs["v"]
        m = inputs["m"]

        outputs["xdot"] = v
        outputs["vdot"] = u / m
        outputs["Jdot"] = u**2

    def compute_partials(self, inputs, jacobian):
        u = inputs["u"]
        m = inputs["m"]

        jacobian["xdot", "v"] = 1.0
        jacobian["vdot", "u"] = 1.0 / m
        jacobian["Jdot", "u"] = 2 * u


# Add options
parser = argparse.ArgumentParser()
parser.add_argument(
    "--optimizer", default="ParOpt", help="Optimizer name from pyOptSparse"
)
parser.add_argument("--algorithm", default="tr", help="algorithm used in ParOpt")
parser.add_argument("--less_force", action="store_true", default=False)

args = parser.parse_args()

optimizer = args.optimizer
algorithm = args.algorithm
less_force = args.less_force

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()

p.model.add_subsystem("traj", subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
transcript = dm.GaussLobatto(num_segments=10, order=3)
phase = dm.Phase(ode_class=SimpleODE, transcription=transcript)

traj.add_phase(name="phase0", phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=5.0, units="s")

# Define state variables
phase.add_state("x", units="m", fix_initial=True, fix_final=True, rate_source="xdot")
phase.add_state(
    "v",
    units="m/s",
    fix_initial=True,
    fix_final=True,
    rate_source="vdot",
    targets=["v"],
)
phase.add_state(
    "J", units="N*N*s", fix_initial=True, fix_final=False, rate_source="Jdot"
)

# Define control variable
if less_force:
    max_force = 0.5
else:
    max_force = 10.0

phase.add_control(
    name="u",
    units="N",
    lower=-max_force,
    upper=max_force,
    targets=["u"],
    fix_initial=False,
    fix_final=False,
)

# Minimize final time.
phase.add_objective("J", loc="final")

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states.
p.set_val(
    "traj.phase0.states:x", phase.interpolate(ys=[0, 5], nodes="state_input"), units="m"
)

p.set_val(
    "traj.phase0.states:v",
    phase.interpolate(ys=[0, 0], nodes="state_input"),
    units="m/s",
)

p.set_val(
    "traj.phase0.controls:u",
    phase.interpolate(ys=[0, 0], nodes="control_input"),
    units="N",
)

# Create the driver
p.driver = ParOptTestDriver()

options = {
    "algorithm": "ip",
    "norm_type": "l1",
    "qn_subspace_size": 10,
    "qn_update_type": "damped_update",
    "abs_res_tol": 1e-6,
    "barrier_strategy": "monotone",
    "output_level": 0,
    "armijo_constant": 1e-5,
    "max_major_iters": 500,
    "penalty_gamma": 2.0e2,
}

for key in options:
    p.driver.options[key] = options[key]

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results by using scipy.integrate.solve_ivp to
# integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

axes[0].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.states:x"),
    "ro",
    label="solution",
)

axes[0].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.states:x"),
    "b-",
    label="simulation",
)

axes[0].set_xlabel("time")
axes[0].set_ylabel("x")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.controls:u"),
    "ro",
    label="solution",
)

axes[1].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.controls:u"),
    "b-",
    label="simulation",
)

axes[1].set_xlabel("time")
axes[1].set_ylabel("u")
axes[1].legend()
axes[1].grid()

plt.show()
