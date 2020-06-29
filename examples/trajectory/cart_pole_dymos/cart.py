import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import math
import argparse

import numpy
numpy.seterr(all='warn')

class CartODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Constants
        self.g = 9.81
        self.l = 0.5
        self.m1 = 1.0
        self.m2 = 0.3

        # Inputs
        self.add_input('q2', val=np.zeros(nn))
        self.add_input('q3', val=np.zeros(nn))
        self.add_input('q4', val=np.zeros(nn))
        self.add_input('u', val=np.zeros(nn))

        # Outputs
        self.add_output('q1dot', val=np.zeros(nn))
        self.add_output('q2dot', val=np.zeros(nn))
        self.add_output('q3dot', val=np.zeros(nn))
        self.add_output('q4dot', val=np.zeros(nn))
        self.add_output('Jdot', val=np.zeros(nn))

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials(of='q1dot', wrt='q2', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='q1dot', wrt='q3', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='q1dot', wrt='q4', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='q1dot', wrt='u', rows=arange, cols=arange, val=0.0)

        self.declare_partials(of='q2dot', wrt='q2', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='q2dot', wrt='q3', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='q2dot', wrt='q4', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='q2dot', wrt='u', rows=arange, cols=arange, val=0.0)

        self.declare_partials(of='q3dot', wrt='q2', rows=arange, cols=arange)
        self.declare_partials(of='q3dot', wrt='q3', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='q3dot', wrt='q4', rows=arange, cols=arange)
        self.declare_partials(of='q3dot', wrt='u', rows=arange, cols=arange)

        self.declare_partials(of='q4dot', wrt='q2', rows=arange, cols=arange)
        self.declare_partials(of='q4dot', wrt='q3', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='q4dot', wrt='q4', rows=arange, cols=arange)
        self.declare_partials(of='q4dot', wrt='u', rows=arange, cols=arange)

        self.declare_partials(of='Jdot', wrt='q2', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='Jdot', wrt='q3', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='Jdot', wrt='q4', rows=arange, cols=arange, val=0.0)
        self.declare_partials(of='Jdot', wrt='u', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        l = self.l
        m1 = self.m1
        m2 = self.m2
        g = self.g
        q2 = inputs['q2']
        q3 = inputs['q3']
        q4 = inputs['q4']
        u = inputs['u']
        sin_q2 = np.sin(q2)
        cos_q2 = np.cos(q2)

        outputs['q1dot'] = q3
        outputs['q2dot'] = q4
        outputs['q3dot'] =   (l * m2 * sin_q2 * q4 ** 2 + u + m2 * g * cos_q2 * sin_q2) / (m1 + m2 * (1 - cos_q2 ** 2))
        outputs['q4dot'] = - (l * m2 * cos_q2 * sin_q2 * q4 ** 2 + u * cos_q2 + (m1 + m2) * g * sin_q2) / (l * m1 + l * m2 * (1 - cos_q2 ** 2))
        outputs['Jdot'] = u ** 2

    def compute_partials(self, inputs, jacobian):
        l = self.l
        m1 = self.m1
        m2 = self.m2
        g = self.g
        q2 = inputs['q2']
        q3 = inputs['q3']
        q4 = inputs['q4']
        u = inputs['u']
        sin_q2 = np.sin(q2)
        cos_q2 = np.cos(q2)

        y1 = l * m2 * sin_q2 * q4 ** 2 + u + m2 * g * cos_q2 * sin_q2
        y2 = m1 + m2 * (1 - cos_q2 ** 2)
        dy1dq2 = l * m2 * cos_q2 * q4 ** 2 - m2 * g * sin_q2 * sin_q2 + m2 * g * cos_q2 * cos_q2
        dy2dq2 = m2 * 2 * cos_q2 * sin_q2
        dy1dq4 = 2 * l * m2 * sin_q2 * q4

        jacobian['q3dot', 'q2'] = (dy1dq2 * y2 - dy2dq2 * y1) / y2 ** 2
        jacobian['q3dot', 'q4'] = dy1dq4 / y2
        jacobian['q3dot', 'u'] = 1 / y2

        y3 = - (l * m2 * cos_q2 * sin_q2 * q4 ** 2 + u * cos_q2 + (m1 + m2) * g * sin_q2)
        y4 = l * m1 + l * m2 * (1 - cos_q2 ** 2)
        dy3dq2 = - ( - l * m2 * sin_q2 * sin_q2 * q4 ** 2 + l * m2 * cos_q2 * cos_q2 * q4 ** 2 - u * sin_q2 + (m1 + m2) * g * cos_q2)
        dy4dq2 =l * m2 * 2 * cos_q2 * sin_q2
        dy3dq4 = - 2 * l * m2 * cos_q2 * sin_q2 * q4

        jacobian['q4dot', 'q2'] = (dy3dq2 * y4 - dy4dq2 * y3) / y4 ** 2
        jacobian['q4dot', 'q4'] = dy3dq4 / y4
        jacobian['q4dot', 'u'] = - cos_q2 / y4

        jacobian['Jdot', 'u'] = 2 * u

# Add options
parser = argparse.ArgumentParser()
parser.add_argument('--nn', type=int, default=25,
                    help='number of nodes')
parser.add_argument('--order', type=int, default=3,
                    help='order of Gauss-Lobatto collocation')
parser.add_argument('--optimizer', default='SLSQP',
                    help='Optimizer name from pyOptSparse')
args = parser.parse_args()

nn = args.nn
order = args.order
optimizer = args.optimizer

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object and add to problem
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=CartODE,
                 transcription=dm.GaussLobatto(num_segments=nn, order=order))

traj.add_phase(name='phase0', phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=2.0)

# Define state variable
phase.add_state('q1', fix_initial=True, fix_final=True, rate_source='q1dot', lower=-2.0, upper=2.0)
phase.add_state('q2', fix_initial=True, fix_final=True, rate_source='q2dot', targets='q2', lower=-100.0, upper=100.0)
phase.add_state('q3', fix_initial=True, fix_final=True, rate_source='q3dot', targets='q3', lower=-100.0, upper=100.0)
phase.add_state('q4', fix_initial=True, fix_final=True, rate_source='q4dot', targets='q4', lower=-100.0, upper=100.0)
phase.add_state('J', fix_initial=True, fix_final=False, rate_source='Jdot', lower=-100.0, upper=100.0)

# Define control variable
phase.add_control(name='u', lower=-20.0, upper=20.0, continuity=True, rate_continuity=True, targets='u')

# Add constraints
# phase.add_boundary_constraint('q1', loc='initial', equals=0.0)
# phase.add_boundary_constraint('q2', loc='initial', equals=0.0)
# phase.add_boundary_constraint('q3', loc='initial', equals=0.0)
# phase.add_boundary_constraint('q4', loc='initial', equals=0.0)
# phase.add_boundary_constraint('J', loc='initial', equals=0.0)
# phase.add_boundary_constraint('q1', loc='final', equals=1.0)
# phase.add_boundary_constraint('q2', loc='final', equals=np.pi)
# phase.add_boundary_constraint('q3', loc='final', equals=0.0)
# phase.add_boundary_constraint('q4', loc='final', equals=0.0)

# Minimize J
phase.add_objective('J', loc='final')

# Set the driver.
p.driver = om.pyOptSparseDriver()

if optimizer == "SLSQP":
    p.driver.options['optimizer'] = 'SLSQP'

else:
    p.driver.options['optimizer'] = "ParOpt"
    p.driver.opt_settings['algorithm'] = 'tr'
    p.driver.opt_settings['output_level'] = 2
    p.driver.opt_settings['tr_max_size'] = 1e2
    p.driver.opt_settings['penalty_gamma'] = 1e2
    p.driver.opt_settings['tr_penalty_gamma_min'] = 100.0
    p.driver.opt_settings['tr_adaptive_gamma_update'] = False
    p.driver.opt_settings['tr_max_iterations'] = 500
    p.driver.opt_settings['norm_type'] = 'infinity'
    p.driver.opt_settings['abs_res_tol'] = 1e-8
    p.driver.opt_settings['max_major_iters'] = 1000
    # p.driver.opt_settings['barrier_strategy'] = 'mehrotra'
    p.driver.opt_settings['qn_type'] = 'none'
    p.driver.opt_settings['sequential_linear_method'] = True
#    p.driver.opt_settings['gradient_verification_frequency'] = 1


# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
# p.driver.declare_coloring()

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states.
p.set_val('traj.phase0.states:q1',
          phase.interpolate(ys=[0, 1], nodes='state_input'))

p.set_val('traj.phase0.states:q2',
          phase.interpolate(ys=[0, np.pi], nodes='state_input'))

p.set_val('traj.phase0.states:q3',
          phase.interpolate(ys=[0, 0], nodes='state_input'))

p.set_val('traj.phase0.states:q4',
          phase.interpolate(ys=[0, 0], nodes='state_input'))

p.set_val('traj.phase0.controls:u',
          phase.interpolate(ys=[20, -20], nodes='control_input'))


# Check gradients
# fd_checks = p.check_partials(out_stream=None, compact_print=True)
# for k1, v1 in fd_checks.items():
#         for k2, v2 in v1.items():
#             for k3, v3 in v2.items():
#                 if 'err' in k3:
#                     for values in v3:
#                         # print(type(values), values)
#                         if not math.isnan(values):
#                             if abs(values) > 1e-20:
#                                 print(k2, k3, v3)

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results by using scipy.integrate.solve_ivp to
# integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))

axes[0].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.states:q1'),
             'ro', label='solution')

axes[0].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.states:q1'),
             'b-', label='simulation')

axes[0].set_xlabel('time')
axes[0].set_ylabel('distance')
axes[0].legend()
axes[0].grid()

axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.states:q2'),
             'ro', label='solution')

axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.states:q2'),
             'b-', label='simulation')

axes[1].set_xlabel('time')
axes[1].set_ylabel('angle')
axes[1].legend()
axes[1].grid()

axes[2].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.controls:u'),
             'ro', label='solution')

axes[2].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.controls:u'),
             'b-', label='simulation')

axes[2].set_xlabel('time')
axes[2].set_ylabel('control')
axes[2].legend()
axes[2].grid()

plt.show()