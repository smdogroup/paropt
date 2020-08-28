import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.plotting import plot_results
from dymos.examples.brachistochrone import BrachistochroneODE
import matplotlib.pyplot as plt
import argparse

# Define options
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', default='paropt', choices=['paropt', 'scipy'],
                    help='Optimizer name from pyOptSparse')
args = parser.parse_args()
optimizer = args.optimizer

# Initialize the Problem and the optimization driver
p = om.Problem(model=om.Group())
if optimizer == 'scipy':
    p.driver = om.ScipyOptimizeDriver()
elif optimizer == 'paropt':
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'ParOpt'
    p.driver.opt_settings['algorithm'] = 'tr'
p.driver.declare_coloring()

# Create a trajectory and add a phase to it
traj = p.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase0',
                       dm.Phase(ode_class=BrachistochroneODE,
                                transcription=dm.GaussLobatto(num_segments=10)))

# Set the variables
phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                units=BrachistochroneODE.states['x']['units'],
                fix_initial=True, fix_final=True, solve_segments=False)

phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                units=BrachistochroneODE.states['y']['units'],
                fix_initial=True, fix_final=True, solve_segments=False)

phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                targets=BrachistochroneODE.states['v']['targets'],
                units=BrachistochroneODE.states['v']['units'],
                fix_initial=True, fix_final=False, solve_segments=False)

phase.add_control('theta', targets=BrachistochroneODE.parameters['theta']['targets'],
                  continuity=True, rate_continuity=True,
                  units='deg', lower=0.01, upper=179.9)

phase.add_input_parameter('g', targets=BrachistochroneODE.parameters['g']['targets'],
                    units='m/s**2', val=9.80665)

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', scaler=10)
p.model.linear_solver = om.DirectSolver()

# Setup the Problem
p.setup()

# Set the initial values
p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = 2.0

p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
p['traj.phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

# Solve for the optimal trajectory
dm.run_problem(p)

# Test the results
assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

# Generate the explicitly simulated trajectory
exp_out = traj.simulate()

plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
               'x (m)', 'y (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:theta',
               'time (s)', 'theta (deg)')],
             title='Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method',
             p_sol=p, p_sim=exp_out)

plt.show()