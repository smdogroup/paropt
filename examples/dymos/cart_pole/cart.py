import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import argparse
import numpy
from paropt.paropt_sparse_driver import ParOptSparseDriver

numpy.seterr(all="warn")


class CartODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # Constants
        self.g = 9.81
        self.l = 0.5
        self.m1 = 1.0
        self.m2 = 0.3

        # Inputs
        self.add_input("q2", val=np.zeros(nn))
        self.add_input("q3", val=np.zeros(nn))
        self.add_input("q4", val=np.zeros(nn))
        self.add_input("u", val=np.zeros(nn))

        # Outputs
        self.add_output("q1dot", val=np.zeros(nn))
        self.add_output("q2dot", val=np.zeros(nn))
        self.add_output("q3dot", val=np.zeros(nn))
        self.add_output("q4dot", val=np.zeros(nn))
        self.add_output("Jdot", val=np.zeros(nn))

        # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)

        self.declare_partials(of="q1dot", wrt="q2", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q1dot", wrt="q3", rows=arange, cols=arange, val=1.0)
        self.declare_partials(of="q1dot", wrt="q4", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q1dot", wrt="u", rows=arange, cols=arange, val=0.0)

        self.declare_partials(of="q2dot", wrt="q2", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q2dot", wrt="q3", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q2dot", wrt="q4", rows=arange, cols=arange, val=1.0)
        self.declare_partials(of="q2dot", wrt="u", rows=arange, cols=arange, val=0.0)

        self.declare_partials(of="q3dot", wrt="q2", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q3dot", wrt="q3", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q3dot", wrt="q4", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q3dot", wrt="u", rows=arange, cols=arange, val=0.0)

        self.declare_partials(of="q4dot", wrt="q2", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q4dot", wrt="q3", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q4dot", wrt="q4", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="q4dot", wrt="u", rows=arange, cols=arange, val=0.0)

        self.declare_partials(of="Jdot", wrt="q2", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="Jdot", wrt="q3", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="Jdot", wrt="q4", rows=arange, cols=arange, val=0.0)
        self.declare_partials(of="Jdot", wrt="u", rows=arange, cols=arange, val=0.0)

    def compute(self, inputs, outputs):
        l = self.l
        m1 = self.m1
        m2 = self.m2
        g = self.g
        q2 = inputs["q2"]
        q3 = inputs["q3"]
        q4 = inputs["q4"]
        u = inputs["u"]
        sin_q2 = np.sin(q2)
        cos_q2 = np.cos(q2)

        outputs["q1dot"] = q3
        outputs["q2dot"] = q4
        outputs["q3dot"] = (
            l * m2 * sin_q2 * q4**2 + u + m2 * g * cos_q2 * sin_q2
        ) / (m1 + m2 * (1 - cos_q2**2))
        outputs["q4dot"] = -(
            l * m2 * cos_q2 * sin_q2 * q4**2 + u * cos_q2 + (m1 + m2) * g * sin_q2
        ) / (l * m1 + l * m2 * (1 - cos_q2**2))
        outputs["Jdot"] = u**2

    def compute_partials(self, inputs, jacobian):
        l = self.l
        m1 = self.m1
        m2 = self.m2
        g = self.g
        q2 = inputs["q2"]
        q3 = inputs["q3"]
        q4 = inputs["q4"]
        u = inputs["u"]
        sin_q2 = np.sin(q2)
        cos_q2 = np.cos(q2)

        y1 = l * m2 * sin_q2 * q4**2 + u + m2 * g * cos_q2 * sin_q2
        y2 = m1 + m2 * (1 - cos_q2**2)
        dy1dq2 = (
            l * m2 * cos_q2 * q4**2
            - m2 * g * sin_q2 * sin_q2
            + m2 * g * cos_q2 * cos_q2
        )
        dy2dq2 = m2 * 2 * cos_q2 * sin_q2
        dy1dq4 = 2 * l * m2 * sin_q2 * q4

        jacobian["q3dot", "q2"] = (dy1dq2 * y2 - dy2dq2 * y1) / y2**2
        jacobian["q3dot", "q4"] = dy1dq4 / y2
        jacobian["q3dot", "u"] = 1 / y2

        y3 = -(l * m2 * cos_q2 * sin_q2 * q4**2 + u * cos_q2 + (m1 + m2) * g * sin_q2)
        y4 = l * m1 + l * m2 * (1 - cos_q2**2)
        dy3dq2 = -(
            -l * m2 * sin_q2 * sin_q2 * q4**2
            + l * m2 * cos_q2 * cos_q2 * q4**2
            - u * sin_q2
            + (m1 + m2) * g * cos_q2
        )
        dy4dq2 = l * m2 * 2 * cos_q2 * sin_q2
        dy3dq4 = -2 * l * m2 * cos_q2 * sin_q2 * q4

        jacobian["q4dot", "q2"] = (dy3dq2 * y4 - dy4dq2 * y3) / y4**2
        jacobian["q4dot", "q4"] = dy3dq4 / y4
        jacobian["q4dot", "u"] = -cos_q2 / y4

        jacobian["Jdot", "u"] = 2 * u


# Add options
parser = argparse.ArgumentParser()
parser.add_argument("--nn", type=int, default=25, help="number of nodes")
parser.add_argument(
    "--order", type=int, default=3, help="order of Gauss-Lobatto collocation"
)
parser.add_argument("--algorithm", default="ip", help="Algorithm used in ParOpt")
parser.add_argument(
    "--show_sparsity",
    default=False,
    action="store_true",
    help="Show the sparsity pattern",
)
args = parser.parse_args()

nn = args.nn
order = args.order
show_sparsity = args.show_sparsity

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object and add to problem
traj = dm.Trajectory()
p.model.add_subsystem("traj", subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
transcript = dm.GaussLobatto(num_segments=nn, order=order)
phase = dm.Phase(ode_class=CartODE, transcription=transcript)

traj.add_phase(name="phase0", phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=2.0)

# Define state variable
phase.add_state(
    "q1", fix_initial=True, fix_final=True, rate_source="q1dot", lower=0.0, upper=2.0
)
phase.add_state(
    "q2",
    fix_initial=True,
    fix_final=True,
    rate_source="q2dot",
    lower=-100.0,
    upper=100.0,
)
phase.add_state("q3", fix_initial=True, rate_source="q3dot", lower=-100.0, upper=100.0)
phase.add_state("q4", fix_initial=True, rate_source="q4dot", lower=-100.0, upper=100.0)
phase.add_state("J", fix_initial=True, rate_source="Jdot", lower=0.0, upper=100.0)

# Define control variable
phase.add_control(
    name="u", lower=-20.0, upper=20.0, continuity=True, rate_continuity=True
)

# Minimize J
phase.add_objective("J", loc="final", ref=10.0)

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states.
p.set_val("traj.phase0.states:q1", phase.interp("q1", ys=[0, 1.0]))
p.set_val("traj.phase0.states:q2", phase.interp("q2", ys=[0, np.pi]))
p.set_val("traj.phase0.states:q3", phase.interp("q3", ys=[0, 0]))
p.set_val("traj.phase0.states:q4", phase.interp("q4", ys=[0, 0]))
p.set_val(
    "traj.phase0.controls:u", phase.interp("u", ys=[0, 1.0], nodes="control_input")
)
p.set_val("traj.phase0.states:J", phase.interp("J", ys=[0, 1]))

# Create the driver
p.driver = ParOptSparseDriver()

if args.algorithm == "ip":
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
elif args.algorithm == "mma":
    options = {
        "algorithm": "mma",
        "qn_type": "none",
        "max_major_iters": 100,
        "abs_res_tol": 1e-8,
        "starting_point_strategy": "affine_step",
        "barrier_strategy": "mehrotra_predictor_corrector",
        "use_line_search": False,
        "mma_use_constraint_linearization": True,
        "mma_max_iterations": 1000,
    }
elif args.algorithm == "tr":
    options = {
        "algorithm": "tr",
        "tr_init_size": 0.05,
        "tr_min_size": 1e-6,
        "tr_max_size": 10.0,
        "tr_eta": 0.25,
        "tr_infeas_tol": 1e-6,
        "tr_l1_tol": 1e-3,
        "tr_linfty_tol": 0.0,
        "tr_adaptive_gamma_update": True,
        "tr_max_iterations": 1000,
        "max_major_iters": 100,
        "penalty_gamma": 1e3,
        "qn_subspace_size": 10,
        "qn_type": "bfgs",
        "abs_res_tol": 1e-8,
        "starting_point_strategy": "affine_step",
        "barrier_strategy": "mehrotra",
        "use_line_search": False,
    }

for key in options:
    p.driver.options[key] = options[key]

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring(show_summary=True, show_sparsity=show_sparsity)

# Run the driver to solve the problem
dm.run_problem(p, make_plots=True, simulate=True)

x, z, zw, zl, zu = p.driver.opt.getOptimizedPoint()
print(np.max(np.abs(zw[:])))

# Check the validity of our results by using scipy.integrate.solve_ivp to
# integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8))

axes[0].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.states:q1"),
    "ro",
    label="solution",
)

axes[0].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.states:q1"),
    "b-",
    label="simulation",
)

axes[0].set_xlabel("time")
axes[0].set_ylabel("distance")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.states:q2"),
    "ro",
    label="solution",
)

axes[1].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.states:q2"),
    "b-",
    label="simulation",
)

axes[1].set_xlabel("time")
axes[1].set_ylabel("angle")
axes[1].legend()
axes[1].grid()

axes[2].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.controls:u"),
    "ro",
    label="solution",
)

axes[2].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.controls:u"),
    "b-",
    label="simulation",
)

axes[2].set_xlabel("time")
axes[2].set_ylabel("control")
axes[2].legend()
axes[2].grid()

axes[3].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.states:J"),
    "ro",
    label="solution",
)

axes[3].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.states:J"),
    "b-",
    label="simulation",
)

axes[3].set_xlabel("time")
axes[3].set_ylabel("J")
axes[3].legend()
axes[3].grid()

plt.show()
