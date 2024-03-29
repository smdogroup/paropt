import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.plotting import plot_results
from dymos.examples.brachistochrone import BrachistochroneODE
import matplotlib.pyplot as plt
from paropt.paropt_sparse_driver import ParOptSparseDriver

# Initialize the Problem and the optimization driver
p = om.Problem(model=om.Group())

# Create a trajectory and add a phase to it
traj = p.model.add_subsystem("traj", dm.Trajectory())

transcript = dm.GaussLobatto(num_segments=10)
phase = traj.add_phase(
    "phase0",
    dm.Phase(ode_class=BrachistochroneODE, transcription=transcript),
)

# Set the variables
phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 10))

phase.add_state(
    "x",
    rate_source="xdot",
    units="m",
    fix_initial=True,
    fix_final=True,
    solve_segments=False,
)

phase.add_state(
    "y",
    rate_source="ydot",
    units="m",
    fix_initial=True,
    fix_final=True,
    solve_segments=False,
)

phase.add_state(
    "v",
    rate_source="vdot",
    units="m/s",
    fix_initial=True,
    fix_final=False,
    solve_segments=False,
)

phase.add_control(
    "theta",
    continuity=True,
    rate_continuity=True,
    units="deg",
    lower=0.01,
    upper=179.9,
)

phase.add_parameter(
    "g",
    units="m/s**2",
    val=9.80665,
)

# Minimize time at the end of the phase
phase.add_objective("time", loc="final", scaler=10)
p.model.linear_solver = om.DirectSolver()

# Setup the Problem
p.setup()

# Set the initial values
p["traj.phase0.t_initial"] = 0.0
p["traj.phase0.t_duration"] = 2.0

p["traj.phase0.states:x"] = phase.interp(ys=[0, 10], nodes="state_input")
p["traj.phase0.states:y"] = phase.interp(ys=[10, 5], nodes="state_input")
p["traj.phase0.states:v"] = phase.interp(ys=[0, 9.9], nodes="state_input")
p["traj.phase0.controls:theta"] = phase.interp(ys=[5, 100.5], nodes="control_input")

# Create the driver
p.driver = ParOptSparseDriver()

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring(show_summary=True, show_sparsity=True)

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

# Run the driver to solve the problem
p.run_driver()

# Test the results
assert_near_equal(
    p.get_val("traj.phase0.timeseries.time")[-1], 1.8016, tolerance=1.0e-3
)

# Generate the explicitly simulated trajectory
exp_out = traj.simulate()

plot_results(
    [
        (
            "traj.phase0.timeseries.states:x",
            "traj.phase0.timeseries.states:y",
            "x (m)",
            "y (m)",
        ),
        (
            "traj.phase0.timeseries.time",
            "traj.phase0.timeseries.controls:theta",
            "time (s)",
            "theta (deg)",
        ),
    ],
    title="Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method",
    p_sol=p,
    p_sim=exp_out,
)

plt.show()
