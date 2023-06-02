# Import paths
import openmdao.api as om
import numpy as np
import dymos as dm
import matplotlib.pyplot as plt
from paropt.paropt_sparse_driver import ParOptSparseDriver


# Define penalty function
def pen_f(r0, r1, r2):
    p = 10
    return (
        1.0
        / (1.0 + np.exp(-p * (r0 - 1.0)))
        * 1.0
        / (1.0 + np.exp(-p * (r1 - 1.0)))
        * 1.0
        / (1.0 + np.exp(-p * (r2 - 1.0)))
    )


def position0_f(x0, y0, t, xdot=0, ydot=0):
    return x0 + xdot * t, y0 + ydot * t


# Define the problem class (which sets up the problem)
class ODESystem1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        # ode_options.declare_time(units='s', targets = ['comp.time'])

        # Inputs
        self.add_input("x", val=np.zeros(nn), desc="Horizontal Position", units="m")
        self.add_input("y", val=np.zeros(nn), desc="Vertical Position", units="m")
        self.add_input("t", val=np.zeros(nn), desc="time", units="s")
        self.add_input("theta", val=np.zeros(nn), desc="angle of wire", units="rad")
        self.add_output(
            "xdot", val=np.zeros(nn), desc="horizontal velocity", units="m/s"
        )
        self.add_output("ydot", val=np.zeros(nn), desc="vertical velocity", units="m/s")
        self.add_output(
            "p1", val=np.zeros(nn), desc="collision detection", units="m"
        )  # for plotting purposes
        self.add_output(
            "p2", val=np.zeros(nn), desc="collision detection", units="m"
        )  # for plotting purposes
        self.add_output(
            "p3", val=np.zeros(nn), desc="collision detection", units="m"
        )  # for plotting purposes
        self.add_output(
            "x1", val=np.zeros(nn), desc="obstacle x position", units="m"
        )  # for plotting purposes
        self.add_output(
            "y1", val=np.zeros(nn), desc="obstacle y position", units="m"
        )  # for plotting purposes
        self.add_output(
            "x2", val=np.zeros(nn), desc="obstacle x position", units="m"
        )  # for plotting purposes
        self.add_output(
            "y2", val=np.zeros(nn), desc="obstacle y position", units="m"
        )  # for plotting purposes
        self.add_output(
            "x3", val=np.zeros(nn), desc="obstacle x position", units="m"
        )  # for plotting purposes
        self.add_output(
            "y3", val=np.zeros(nn), desc="obstacle y position", units="m"
        )  # for plotting purposes

        # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)

        self.declare_partials("xdot", wrt="x", rows=arange, cols=arange)
        self.declare_partials("xdot", wrt="y", rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="theta", rows=arange, cols=arange)

        self.declare_partials("ydot", wrt="x", rows=arange, cols=arange)
        self.declare_partials("ydot", wrt="y", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="theta", rows=arange, cols=arange)

        self.declare_partials("p1", "x", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p1", "y", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p1", "t", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p2", "x", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p2", "y", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p2", "t", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p3", "x", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p3", "y", method="fd", step_calc="rel", step=1e-10)
        self.declare_partials("p3", "t", method="fd", step_calc="rel", step=1e-10)

    def compute(self, inputs, outputs):
        theta = inputs["theta"]
        x = inputs["x"]
        y = inputs["y"]
        t = inputs["t"]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x0n, y0n = position0_f(x0, y0, t, xdot=0, ydot=-1)
        x1n, y1n = position0_f(x1, y1, t, xdot=0, ydot=1)
        x2n, y2n = position0_f(x2, y2, t, xdot=-1, ydot=-1)
        r0 = np.sqrt((x - x0n) ** 2 + (y - y0n) ** 2)
        r1 = np.sqrt((x - x1n) ** 2 + (y - y1n) ** 2)
        r2 = np.sqrt((x - x2n) ** 2 + (y - y2n) ** 2)
        outputs["xdot"] = 5 * sin_theta
        outputs["ydot"] = -5 * cos_theta
        outputs["p1"] = r0**2
        outputs["p2"] = r1**2
        outputs["p3"] = r2**2
        outputs["x1"] = x0n
        outputs["x2"] = x1n
        outputs["x3"] = x2n
        outputs["y1"] = y0n
        outputs["y2"] = y1n
        outputs["y3"] = y2n

    def compute_partials(self, inputs, jacobian):
        theta = inputs["theta"]
        x = inputs["x"]
        y = inputs["y"]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        jacobian["xdot", "x"] = 0
        jacobian["xdot", "y"] = 0
        jacobian["xdot", "theta"] = 5 * cos_theta
        jacobian["ydot", "x"] = 0
        jacobian["ydot", "y"] = 0
        jacobian["ydot", "theta"] = 5 * sin_theta


# Define obstacles
x0, y0 = 3, 3
x1, y1 = 5, 5
x2, y2 = 8, 8

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()

# Add model system
p.model.add_subsystem("traj", subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(
    ode_class=ODESystem1, transcription=dm.GaussLobatto(num_segments=25, order=3)
)
traj.add_phase(name="phase0", phase=phase)

# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
phase.set_time_options(
    fix_initial=True, duration_bounds=(0.5, 10.0), units="s", targets=["t"]
)

# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The rate source points to the output in the ODE which provides the time derivative of the given state.
phase.add_state(
    "x", fix_initial=True, fix_final=True, units="m", rate_source="xdot", targets=["x"]
)
phase.add_state(
    "y", fix_initial=True, fix_final=True, units="m", rate_source="ydot", targets=["y"]
)

# Define theta as a control.
phase.add_control(name="theta", units="rad", lower=0, upper=np.pi, targets=["theta"])

# Define constraints
phase.add_timeseries_output(name="p1", units="m")
phase.add_timeseries_output(name="p2", units="m")
phase.add_timeseries_output(name="p3", units="m")
phase.add_timeseries_output(name="x1", units="m")
phase.add_timeseries_output(name="x2", units="m")
phase.add_timeseries_output(name="x3", units="m")
phase.add_timeseries_output(name="y1", units="m")
phase.add_timeseries_output(name="y2", units="m")
phase.add_timeseries_output(name="y3", units="m")
phase.add_path_constraint(name="p1", lower=1, upper=None, units="m")
phase.add_path_constraint(name="p2", lower=1, upper=None, units="m")
phase.add_path_constraint(name="p3", lower=1, upper=None, units="m")

# Minimize final time.
phase.add_objective("time", loc="final")

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states
p.set_val(
    "traj.phase0.states:x",
    phase.interpolate(ys=[0, 10], nodes="state_input"),
    units="m",
)
p.set_val(
    "traj.phase0.states:y",
    phase.interpolate(ys=[0, 10], nodes="state_input"),
    units="m",
)
p.set_val(
    "traj.phase0.controls:theta",
    phase.interpolate(ys=[90, 90], nodes="control_input"),
    units="deg",
)

p.driver = ParOptSparseDriver()

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

for key in options:
    p.driver.options[key] = options[key]

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
axes[0].plot(
    p.get_val("traj.phase0.timeseries.states:x"),
    p.get_val("traj.phase0.timeseries.states:y"),
    "ro",
    label="Solution",
)
axes[0].plot(
    sim_out.get_val("traj.phase0.timeseries.states:x"),
    sim_out.get_val("traj.phase0.timeseries.states:y"),
    "b-",
    label="Simulation",
)
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
axes[0].legend()
axes[0].grid()

axes[1].plot(
    p.get_val("traj.phase0.timeseries.x1"),
    p.get_val("traj.phase0.timeseries.y1"),
    "bo",
    label="Solution",
)
axes[1].plot(
    p.get_val("traj.phase0.timeseries.x2"),
    p.get_val("traj.phase0.timeseries.y2"),
    "g^",
    label="Solution",
)
axes[1].plot(
    p.get_val("traj.phase0.timeseries.x3"),
    p.get_val("traj.phase0.timeseries.y3"),
    "rs",
    label="Solution",
)
axes[1].plot(
    sim_out.get_val("traj.phase0.timeseries.x1"),
    sim_out.get_val("traj.phase0.timeseries.y1"),
    "b--",
    label="Simulation",
)
axes[1].plot(
    sim_out.get_val("traj.phase0.timeseries.x2"),
    sim_out.get_val("traj.phase0.timeseries.y2"),
    "g--",
    label="Simulation",
)
axes[1].plot(
    sim_out.get_val("traj.phase0.timeseries.x3"),
    sim_out.get_val("traj.phase0.timeseries.y3"),
    "r--",
    label="Simulation",
)
axes[1].set_xlabel("px (m)")
axes[1].set_ylabel("py (m)")
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].legend()
axes[1].grid()

axes[2].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.p1", units="m"),
    "ro",
    label="Solution",
)
axes[2].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.p2", units="m"),
    "g^",
    label="Solution",
)
axes[2].plot(
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.p3", units="m"),
    "bs",
    label="Solution",
)
axes[2].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.p1", units="m"),
    "r--",
    label="Simulation",
)
axes[2].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.p2", units="m"),
    "g--",
    label="Simulation",
)
axes[2].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.p3", units="m"),
    "b--",
    label="Simulation",
)
axes[2].set_xlabel("time (s)")
axes[2].set_ylabel("$d$ (m)")
axes[2].set_ylim(0, 10)
axes[2].legend()
axes[2].grid()

# Add circles
circle0 = plt.Circle((x0, y0), 1, color="k", fill=False, hatch="///")
circle1 = plt.Circle((x1, y1), 1, color="k", fill=False, hatch="///")
circle2 = plt.Circle((x2, y2), 1, color="k", fill=False, hatch="///")
axes[0].add_artist(circle0)
axes[0].add_artist(circle1)
axes[0].add_artist(circle2)


# Additional plots
x_v = np.linspace(0, 10, 50)
y_v = np.linspace(0, 10, 50)
x_m, y_m = np.meshgrid(x_v, y_v)
r0_m = np.sqrt((x_m - x0) ** 2 + (y_m - y0) ** 2)
r1_m = np.sqrt((x_m - x1) ** 2 + (y_m - y1) ** 2)
r2_m = np.sqrt((x_m - x2) ** 2 + (y_m - y2) ** 2)
eta_m = pen_f(r0_m, r1_m, r2_m)
z_m = 5 * eta_m
fig, ax = plt.subplots(1, 1)
cplot = ax.contourf(x_m, y_m, z_m)
cbar = fig.colorbar(cplot)  # Add a colorbar to a plot
cbar.set_label("velocity (m/s)")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.grid()

# Add circles
circle0 = plt.Circle((x0, y0), 1, color="k", fill=False)
circle1 = plt.Circle((x1, y1), 1, color="k", fill=False)
circle2 = plt.Circle((x2, y2), 1, color="k", fill=False)
ax.add_artist(circle0)
ax.add_artist(circle1)
ax.add_artist(circle2)
plt.show()
