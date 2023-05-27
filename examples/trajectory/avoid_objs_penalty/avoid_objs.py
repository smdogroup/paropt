# Import libraries
import sys
import openmdao.api as om
import numpy as np
import dymos as dm
import matplotlib.pyplot as plt

sys.path.append("../cart_pole_dymos")
from test_driver import ParOptTestDriver


class ODESystem1(om.ExplicitComponent):
    def initialize(self):
        # Define obstacles
        self.x0 = 8
        self.y0 = 6
        self.x1 = 5
        self.y1 = 5
        self.x2 = 8
        self.y2 = 8

        # Define penalty
        self.p = 45.0

        # Declare options
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        # Inputs
        self.add_input("x", val=np.zeros(nn), desc="Horizontal Position", units="m")
        self.add_input("y", val=np.zeros(nn), desc="Vertical Position", units="m")
        self.add_input("theta", val=np.zeros(nn), desc="angle of wire", units="rad")

        # outputs
        self.add_output(
            "xdot", val=np.zeros(nn), desc="horizontal velocity", units="m/s"
        )
        self.add_output("ydot", val=np.zeros(nn), desc="vertical velocity", units="m/s")
        self.add_output(
            "cflag", val=np.zeros(nn), desc="collision detection", units="m"
        )

        # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)
        self.declare_partials(of="xdot", wrt="theta", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="theta", rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="y", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="y", rows=arange, cols=arange)
        self.declare_partials(of="cflag", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="cflag", wrt="y", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs["theta"]
        x = inputs["x"]
        y = inputs["y"]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        r0 = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2) + 0.01
        r1 = np.sqrt((x - self.x1) ** 2 + (y - self.y1) ** 2) + 0.01
        r2 = np.sqrt((x - self.x2) ** 2 + (y - self.y2) ** 2) + 0.01

        cflag = (
            1.0
            / (1.0 + np.exp(-self.p * (r0 - 1.0)))
            * 1.0
            / (1.0 + np.exp(-self.p * (r1 - 1.0)))
            * 1.0
            / (1.0 + np.exp(-self.p * (r2 - 1.0)))
        )
        outputs["xdot"] = 5.0 * cflag * cos_theta
        outputs["ydot"] = 5.0 * cflag * sin_theta
        outputs["cflag"] = cflag

    def compute_partials(self, inputs, jacobian):
        theta = inputs["theta"]
        x = inputs["x"]
        y = inputs["y"]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        r0 = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2) + 0.01
        r1 = np.sqrt((x - self.x1) ** 2 + (y - self.y1) ** 2) + 0.01
        r2 = np.sqrt((x - self.x2) ** 2 + (y - self.y2) ** 2) + 0.01

        cflag = (
            1.0
            / (1.0 + np.exp(-self.p * (r0 - 1.0)))
            * 1.0
            / (1.0 + np.exp(-self.p * (r1 - 1.0)))
            * 1.0
            / (1.0 + np.exp(-self.p * (r2 - 1.0)))
        )

        jacobian["xdot", "theta"] = -5 * cflag * sin_theta
        jacobian["ydot", "theta"] = 5 * cflag * cos_theta

        cflag__r0 = (
            cflag
            / (1.0 + np.exp(-self.p * (r0 - 1.0)))
            * (self.p)
            * np.exp(-self.p * (r0 - 1.0))
        )
        cflag__r1 = (
            cflag
            / (1.0 + np.exp(-self.p * (r1 - 1.0)))
            * (self.p)
            * np.exp(-self.p * (r1 - 1.0))
        )
        cflag__r2 = (
            cflag
            / (1.0 + np.exp(-self.p * (r2 - 1.0)))
            * (self.p)
            * np.exp(-self.p * (r2 - 1.0))
        )
        r0__x = (x - self.x0) / r0
        r0__y = (y - self.y0) / r0
        r1__x = (x - self.x1) / r1
        r1__y = (y - self.y1) / r1
        r2__x = (x - self.x2) / r2
        r2__y = (y - self.y2) / r2

        jacobian["cflag", "x"] = (
            cflag__r0 * r0__x + +cflag__r1 * r1__x + +cflag__r2 * r2__x
        )
        jacobian["cflag", "y"] = (
            cflag__r0 * r0__y + +cflag__r1 * r1__y + +cflag__r2 * r2__y
        )

        xdot__cflag = 5 * cos_theta
        ydot__cflag = 5 * sin_theta

        jacobian["xdot", "x"] = xdot__cflag * jacobian["cflag", "x"]
        jacobian["xdot", "y"] = xdot__cflag * jacobian["cflag", "y"]

        jacobian["ydot", "x"] = ydot__cflag * jacobian["cflag", "x"]
        jacobian["ydot", "y"] = ydot__cflag * jacobian["cflag", "y"]


def pen_f(r0, r1, r2):
    p = 45
    return (
        1
        / (1 + np.exp(-p * (r0 - 1)))
        * 1
        / (1 + np.exp(-p * (r1 - 1)))
        * 1
        / (1 + np.exp(-p * (r2 - 1)))
    )


# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

p.driver = ParOptTestDriver()

options = {
    "algorithm": "ip",
    "tr_linfty_tol": 1e-30,
    "tr_l1_tol": 1e-30,
    "output_level": 0,
    "qn_type": "bfgs",
    "max_major_iters": 500,
    "tr_max_iterations": 200,
    "qn_update_type": "damped_update",
    "penalty_gamma": 1e3,
    "tr_min_size": 1e-2,
    "tr_adaptive_gamma_update": False,
    "tr_accept_step_strategy": "penalty_method",
    "tr_use_soc": False,
}

for key in options:
    p.driver.options[key] = options[key]

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significantly speed up the execution of Dymos.
p.driver.declare_coloring()

# Define a Trajectory object
traj = p.model.add_subsystem("traj", subsys=dm.Trajectory())

# Define a Dymos Phase object with GaussLobatto Transcription
n_segments = 20
transcript = dm.GaussLobatto(num_segments=n_segments, order=3)
phase = dm.Phase(ode_class=ODESystem1, transcription=transcript)
traj.add_phase(name="phase0", phase=phase)

# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
phase.set_time_options(
    initial_bounds=(0, 0), duration_bounds=(0.5, 10.0), units="s"  # fix_initial = True,
)

# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The rate source points to the output in the ODE which provides the time derivative of the given state.
phase.add_state(
    "x", rate_source="xdot", units="m", fix_initial=True, fix_final=True, targets=["x"]
)

phase.add_state(
    "y", rate_source="ydot", units="m", fix_initial=True, fix_final=True, targets=["y"]
)

# Define theta as a control.
phase.add_control(
    name="theta", targets=["theta"], units="rad", lower=-np.pi / 2, upper=np.pi / 2
)

# Define constraints
phase.add_timeseries_output("cflag", output_name="cflag", units="m")

# Minimize final time.
phase.add_objective("time", loc="final")

# Setup the problem
p.setup(check=True)

p.set_val(
    "traj.phase0.states:x",
    phase.interpolate(xs=[0, 1, 2], ys=[0, 5, 10], kind="linear", nodes="state_input"),
    units="m",
)

# initialize 0: left, 1: middle, 2: right
init = 1
if init == 0:
    p.set_val(
        "traj.phase0.states:y",
        phase.interpolate(
            xs=[0, 1, 2], ys=[0, 8, 10], kind="quadratic", nodes="state_input"
        ),
        units="m",
    )
elif init == 1:
    p.set_val(
        "traj.phase0.states:y",
        phase.interpolate(ys=[0, 10], nodes="state_input"),
        units="m",
    )
elif init == 2:
    p.set_val(
        "traj.phase0.states:y",
        phase.interpolate(
            xs=[0, 1, 2], ys=[0, 2, 10], kind="quadratic", nodes="state_input"
        ),
        units="m",
    )


p.set_val(
    "traj.phase0.controls:theta",
    phase.interpolate(ys=[-5, 5], nodes="control_input"),
    units="deg",
)

p["traj.phase0.t_duration"] = 3.0
p["traj.phase0.t_initial"] = 0.0

# Run the driver to solve the problem
p.run_driver()
print(p.get_val("traj.phase0.timeseries.time")[-1])

# Check the validity of our results by using scipy.integrate.solve_ivp to integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
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
    p.get_val("traj.phase0.timeseries.time"),
    p.get_val("traj.phase0.timeseries.cflag", units="m"),
    "ro",
    label="Solution",
)
axes[1].plot(
    sim_out.get_val("traj.phase0.timeseries.time"),
    sim_out.get_val("traj.phase0.timeseries.cflag", units="m"),
    "b-",
    label="Simulation",
)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("$v$ (m/s)")
axes[1].legend()
axes[1].grid()

ode = ODESystem1()
ode.initialize()

# Add circles
circle0 = plt.Circle((ode.x0, ode.y0), 1, color="k", fill=False, hatch="///")
circle1 = plt.Circle((ode.x1, ode.y1), 1, color="k", fill=False, hatch="///")
circle2 = plt.Circle((ode.x2, ode.y2), 1, color="k", fill=False, hatch="///")
axes[0].add_artist(circle0)
axes[0].add_artist(circle1)
axes[0].add_artist(circle2)

# Additional plots
x_v = np.linspace(0, 10, 50)
y_v = np.linspace(0, 10, 50)
x_m, y_m = np.meshgrid(x_v, y_v)
r0_m = np.sqrt((x_m - ode.x0) ** 2 + (y_m - ode.y0) ** 2)
r1_m = np.sqrt((x_m - ode.x1) ** 2 + (y_m - ode.y1) ** 2)
r2_m = np.sqrt((x_m - ode.x2) ** 2 + (y_m - ode.y2) ** 2)
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
circle0 = plt.Circle((ode.x0, ode.y0), 1, color="k", fill=False)
circle1 = plt.Circle((ode.x1, ode.y1), 1, color="k", fill=False)
circle2 = plt.Circle((ode.x2, ode.y2), 1, color="k", fill=False)
ax.add_artist(circle0)
ax.add_artist(circle1)
ax.add_artist(circle2)

plt.show()
