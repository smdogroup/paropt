import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
import matplotlib as mpl

from dymos.examples.racecar.combinedODE import CombinedODE
from dymos.examples.racecar.spline import get_spline, get_track_points
from dymos.examples.racecar.tracks import ovaltrack

from paropt.paropt_sparse_driver import ParOptSparseDriver

# change track here and in curvature.py. Tracks are defined in tracks.py
track = ovaltrack

# generate nodes along the centerline for curvature calculation (different
# than collocation nodes)
points = get_track_points(track)

# fit the centerline spline.
finespline, gates, gatesd, curv, slope = get_spline(points, s=0.0)

# by default 10000 points
s_final = track.get_total_length()

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem("traj", subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(
    ode_class=CombinedODE,
    transcription=dm.GaussLobatto(num_segments=80, order=3, compressed=True),
)

traj.add_phase(name="phase0", phase=phase)

# Set the time options, in this problem we perform a change of variables. So 'time' is
# actually 's' (distance along the track centerline)
# This is done to fix the collocation nodes in space, which saves us the calculation of
# the rate of change of curvature.
# The state equations are written with respect to time, the variable change occurs in
# timeODE.py
phase.set_time_options(
    fix_initial=True,
    fix_duration=True,
    duration_val=s_final,
    name="s",
    targets=["curv.s"],
    units="m",
    duration_ref=s_final,
    duration_ref0=10,
)

# Set the reference values
t_ref = 100.0
n_ref = 4.0
V_ref = 40.0
lambda_ref = 0.01
alpha_ref = 0.15
omega_ref = 0.3
ax_ref = 8.0
ay_ref = 8.0
delta_ref = 0.04
thrust_ref = 10.0

# Define states
phase.add_state(
    "t",
    ref=t_ref,
    units="s",
    fix_initial=True,
    fix_final=False,
    lower=0.0,
    upper=10000.0,
    rate_source="dt_ds",
)

# Normal distance to centerline. The bounds on n define the width of the track
phase.add_state(
    "n",
    ref=n_ref,
    units="m",
    fix_initial=False,
    fix_final=False,
    upper=4.0,
    lower=-4.0,
    rate_source="dn_ds",
    targets=["n"],
)

# velocity
phase.add_state(
    "V",
    ref=V_ref,
    ref0=5,
    units="m/s",
    fix_initial=False,
    fix_final=False,
    lower=-500.0,
    upper=500.0,
    rate_source="dV_ds",
    targets=["V"],
)

# vehicle heading angle with respect to centerline
phase.add_state(
    "alpha",
    ref=alpha_ref,
    units="rad",
    fix_initial=False,
    fix_final=False,
    lower=-0.5 * np.pi,
    upper=0.5 * np.pi,
    rate_source="dalpha_ds",
    targets=["alpha"],
)

# vehicle slip angle, or angle between the axis of the vehicle
# and velocity vector (all cars drift a little)
phase.add_state(
    "lambda",
    ref=lambda_ref,
    units="rad",
    fix_initial=False,
    fix_final=False,
    lower=-0.5 * np.pi,
    upper=0.5 * np.pi,
    rate_source="dlambda_ds",
    targets=["lambda"],
)

# yaw rate
phase.add_state(
    "omega",
    ref=omega_ref,
    units="rad/s",
    fix_initial=False,
    fix_final=False,
    lower=-30.0,
    upper=30.0,
    rate_source="domega_ds",
    targets=["omega"],
)

# longitudinal acceleration
phase.add_state(
    "ax",
    ref=ax_ref,
    units="m/s**2",
    fix_initial=False,
    fix_final=False,
    lower=-100.0,
    upper=100.0,
    rate_source="dax_ds",
    targets=["ax"],
)

# Lateral acceleration
phase.add_state(
    "ay",
    ref=ay_ref,
    units="m/s**2",
    fix_initial=False,
    fix_final=False,
    lower=-100.0,
    upper=100.0,
    rate_source="day_ds",
    targets=["ay"],
)

# Define Controls

# steering angle
phase.add_control(
    name="delta",
    ref=delta_ref,
    units="rad",
    fix_initial=False,
    fix_final=False,
    lower=-0.5 * np.pi,
    upper=0.5 * np.pi,
    rate_continuity=True,
)

# the thrust controls the longitudinal force of the rear tires and is positive
# while accelerating, negative while braking
phase.add_control(
    name="thrust",
    ref=thrust_ref,
    units=None,
    lower=-1000.0,
    upper=1000.0,
    fix_initial=False,
    fix_final=False,
    rate_continuity=True,
)

# Performance Constraints
pmax = 960000.0  # W
phase.add_path_constraint("power", upper=pmax, ref=100000.0)  # engine power limit

# The following four constraints are the tire friction limits, with 'rr' designating the
# rear right wheel etc. This limit is computed in tireConstraintODE.py
phase.add_path_constraint("c_rr", upper=1.0)
phase.add_path_constraint("c_rl", upper=1.0)
phase.add_path_constraint("c_fr", upper=1.0)
phase.add_path_constraint("c_fl", upper=1.0)

# Some of the vehicle design parameters are available to set here. Other parameters can
# be found in their respective ODE files.
# vehicle mass
phase.add_parameter(
    "M",
    val=800.0,
    units="kg",
    opt=False,
    targets=["car.M", "tire.M", "tireconstraint.M", "normal.M"],
    static_target=True,
)

# brake bias
phase.add_parameter(
    "beta", val=0.62, units=None, opt=False, targets=["tire.beta"], static_target=True
)

# center of pressure location
phase.add_parameter(
    "CoP", val=1.6, units="m", opt=False, targets=["normal.CoP"], static_target=True
)

# center of gravity height
phase.add_parameter(
    "h", val=0.3, units="m", opt=False, targets=["normal.h"], static_target=True
)

# roll stiffness
phase.add_parameter(
    "chi", val=0.5, units=None, opt=False, targets=["normal.chi"], static_target=True
)

# downforce coefficient*area
phase.add_parameter(
    "ClA", val=4.0, units="m**2", opt=False, targets=["normal.ClA"], static_target=True
)

# drag coefficient*area
phase.add_parameter(
    "CdA", val=2.0, units="m**2", opt=False, targets=["car.CdA"], static_target=True
)

# Minimize final time.
# note that we use the 'state' time instead of Dymos 'time'
phase.add_objective("t", loc="final")

# Add output timeseries
phase.add_timeseries_output("*")
phase.add_timeseries_output("t", output_name="time")

# Link the states at the start and end of the phase in order to ensure a continous lap
traj.link_phases(
    phases=["phase0", "phase0"],
    vars=["V", "n", "alpha", "omega", "lambda", "ax", "ay"],
    locs=["final", "initial"],
    connected=True,
)

# Set up the optimization driver
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
p.driver.declare_coloring(show_summary=True, show_sparsity=False)

# Setup the problem
p.setup(check=True)

# States
# Nonzero velocity to avoid division by zero errors
p.set_val("traj.phase0.states:V", phase.interp("V", [20, 20]), units="m/s")

# All other states start at 0
p.set_val(
    "traj.phase0.states:lambda", phase.interp("lambda", [0.01, 0.01]), units="rad"
)
p.set_val("traj.phase0.states:omega", phase.interp("omega", [0.0, 0.0]), units="rad/s")
p.set_val("traj.phase0.states:alpha", phase.interp("alpha", [0.0, 0.0]), units="rad")
p.set_val("traj.phase0.states:ax", phase.interp("ax", [0.0, 0.0]), units="m/s**2")
p.set_val("traj.phase0.states:ay", phase.interp("ay", [0.0, 0.0]), units="m/s**2")
p.set_val("traj.phase0.states:n", phase.interp("n", [0.0, 0.0]), units="m")

# initial guess for what the final time should be
p.set_val("traj.phase0.states:t", phase.interp("t", [0.0, 100.0]), units="s")

# Controls
# A small amount of thrust can speed up convergence
p.set_val("traj.phase0.controls:delta", phase.interp("delta", [0.0, 0.0]), units="rad")
p.set_val("traj.phase0.controls:thrust", phase.interp("thrust", [0.1, 0.1]), units=None)

p.run_driver()
print("Optimization finished")

# Get optimized time series
n = p.get_val("traj.phase0.timeseries.states:n")
s = p.get_val("traj.phase0.timeseries.s")
V = p.get_val("traj.phase0.timeseries.states:V")
thrust = p.get_val("traj.phase0.timeseries.controls:thrust")
delta = p.get_val("traj.phase0.timeseries.controls:delta")
power = p.get_val("traj.phase0.timeseries.power", units="W")

print("Plotting")

# Plot the main vehicle telemetry
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))

# Velocity vs s
axes[0].plot(s, p.get_val("traj.phase0.timeseries.states:V"), label="solution")

axes[0].set_xlabel("s (m)")
axes[0].set_ylabel("V (m/s)")
axes[0].grid()
axes[0].set_xlim(0, s_final)

# n vs s
axes[1].plot(
    s, p.get_val("traj.phase0.timeseries.states:n", units="m"), label="solution"
)

axes[1].set_xlabel("s (m)")
axes[1].set_ylabel("n (m)")
axes[1].grid()
axes[1].set_xlim(0, s_final)

# throttle vs s
axes[2].plot(s, thrust)

axes[2].set_xlabel("s (m)")
axes[2].set_ylabel("thrust")
axes[2].grid()
axes[2].set_xlim(0, s_final)

# delta vs s
axes[3].plot(
    s, p.get_val("traj.phase0.timeseries.controls:delta", units=None), label="solution"
)

axes[3].set_xlabel("s (m)")
axes[3].set_ylabel("delta")
axes[3].grid()
axes[3].set_xlim(0, s_final)

plt.tight_layout()

# Performance constraint plot. Tire friction and power constraints
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
plt.subplots_adjust(right=0.82, bottom=0.14, top=0.97, left=0.07)

axes.plot(s, p.get_val("traj.phase0.timeseries.c_fl", units=None), label="c_fl")
axes.plot(s, p.get_val("traj.phase0.timeseries.c_fr", units=None), label="c_fr")
axes.plot(s, p.get_val("traj.phase0.timeseries.c_rl", units=None), label="c_rl")
axes.plot(s, p.get_val("traj.phase0.timeseries.c_rr", units=None), label="c_rr")

axes.plot(s, power / pmax, label="Power")

axes.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
axes.set_xlabel("s (m)")
axes.set_ylabel("Performance constraints")
axes.grid()
axes.set_xlim(0, s_final)

plt.show()
