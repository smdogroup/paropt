#Import libraries
import sys
#sys.path.insert(0, 'C:/msys64/home/Charizard/tools_traj_opt/pyoptsparse')
# sys.path.insert(0, 'C:/msys64/home/Charizard/pyoptsparse-build/lib/python3.8/site-packages')
import openmdao.api as om
import numpy as np
import dymos as dm
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from odesystem1 import ODESystem1

# ###############################################
n_segments = 10

#Define obstacles
x0, y0 = 8, 6
x1, y1 = 5, 5
x2, y2 = 8, 8

# ###############################
def pen_f(r0, r1, r2):
   p = 45
   return 1 / (1 + np.exp(-p * (r0 - 1))) *\
          1 / (1 + np.exp(-p * (r1 - 1))) *\
          1 / (1 + np.exp(-p * (r2 - 1)))

# #####################################################
# Define the OpenMDAO problem
p = om.Problem(model = om.Group())

# Set the driver: om.SimpleGADriver(), om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver()
# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.add_recorder(om.SqliteRecorder('twodim%d.sql' % n_segments))
# p.driver.recording_options['includes'] = ['*']
# p.driver.recording_options['record_constraints'] = True
# p.driver.recording_options['record_inputs'] = True
# p.driver.recording_options['record_objectives'] = True
# p.driver.recording_options['record_desvars'] = True
# p.driver.recording_options['record_responses'] = True
# p.driver.recording_options['record_derivatives'] = True

p.driver.options['optimizer'] = 'ParOpt'
p.driver.opt_settings['algorithm'] = 'tr'
p.driver.opt_settings['tr_linfty_tol'] = 1e-30
p.driver.opt_settings['tr_l1_tol'] = 1e-30
p.driver.opt_settings['output_level'] = 0
p.driver.opt_settings['qn_type'] = 'bfgs'
p.driver.opt_settings['max_major_iters'] = 100
p.driver.opt_settings['tr_max_iterations'] = 200
p.driver.opt_settings['qn_update_type'] = 'damped_update'
# p.driver.opt_settings['norm_type'] = 'infinity'
# p.driver.opt_settings['barrier_strategy'] = 'monotone'
# p.driver.opt_settings['starting_point_strategy'] = 'affine_step'
# p.driver.opt_settings['tr_steering_barrier_strategy'] = 'mehrotra_predictor_corrector'
# p.driver.opt_settings['tr_steering_starting_point_strategy'] = 'affine_step'
# p.driver.opt_settings['use_line_search'] = True
p.driver.opt_settings['penalty_gamma'] = 1e3
p.driver.opt_settings['tr_min_size'] = 1e-2
p.driver.opt_settings['tr_adaptive_gamma_update'] = False
p.driver.opt_settings['tr_accept_step_strategy'] = 'penalty_method'
p.driver.opt_settings['tr_use_soc'] = False
p.driver.opt_settings['tr_soc_use_quad_model'] = False


# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significantly speed up the execution of Dymos.
#p.driver.declare_coloring()

# Define a Trajectory object
#traj = dm.Trajectory()
#p.model.add_subsystem('traj', subsys = traj)
traj = p.model.add_subsystem('traj', subsys = dm.Trajectory())

# Define a Dymos Phase object with GaussLobatto Transcription
# phase = dm.Phase(ode_class = ODESystem1,
#                 transcription = dm.GaussLobatto(num_segments = 10, order = 3))
#traj.add_phase(name = 'phase0', phase = phase)

phase = traj.add_phase('phase0', dm.Phase(ode_class=ODESystem1,
            transcription=dm.GaussLobatto(num_segments=n_segments)))

# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
phase.set_time_options(initial_bounds=(0, 0), # fix_initial = True,
                       duration_bounds = (0.5, 10.0),
                       units = 's')

# ##########################
# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The rate source points to the output in the ODE which provides the time derivative of the given state.
phase.add_state('x',
                rate_source = 'xdot',
                units = 'm',
                fix_initial = True, fix_final = True,
                targets = ['x'])

phase.add_state('y',
                rate_source = 'ydot',
                units = 'm',
                fix_initial = True, fix_final = True,
                targets = ['y'])

#phase.add_state('t', fix_initial = True, fix_final = False, units = 's', rate_source = 'xdot', targets = ['t']) #XXX

# ##########################
# Define theta as a control.
phase.add_control(name = 'theta', targets = ['theta'],
                  units = 'rad',
                  lower = -np.pi/2,
                  upper =  np.pi/2)

# phase.add_control('v', targets='v',
#                   continuity=True, rate_continuity=True,
#                   units='m/s', lower=19., upper=20.)

# ##########################
# ### Define constraints
#phase.add_timeseries_output(name = 'cflag', units = 'm')
# phase.add_path_constraint('cflag',
#                           lower = 0.1, # upper = None,
#                           units = 'm')

# ###################################################
phase.add_timeseries_output('cflag',
                             output_name = 'cflag',
                             units='m')

# Minimize final time.
phase.add_objective('time', loc = 'final')

# Setup the problem
p.setup(check = True)

# Now that the OpenMDAO problem is setup, we can set the values of the states
#p['traj.phase0.states:y'][:] = 0
#p['traj.phase0.states:y'][-1] = 10


p.set_val('traj.phase0.states:x',
           phase.interpolate(xs = [0, 1, 2], ys = [0, 5, 10],
                             kind = 'linear', nodes = 'state_input'),
           units = 'm')

############################################################
# initialize 0: left, 1: middle, 2: right
init = 1

# #######################################################
if init == 0:
    p.set_val('traj.phase0.states:y',
              phase.interpolate(xs = [0, 1, 2], ys = [0, 8, 10],
                                kind = 'quadratic', nodes = 'state_input'),
              units = 'm')
elif init == 1:
    p.set_val('traj.phase0.states:y',
               phase.interpolate(ys = [0, 10], nodes = 'state_input'),
               units = 'm')
elif init == 2:
    p.set_val('traj.phase0.states:y',
               phase.interpolate(xs = [0, 1, 2], ys = [0, 2, 10],
                                 kind = 'quadratic', nodes = 'state_input'),
               units = 'm')


p.set_val('traj.phase0.controls:theta',
           phase.interpolate(ys = [-5, 5], nodes = 'control_input'),
           units = 'deg')

p['traj.phase0.t_duration'] = 3.0
p['traj.phase0.t_initial'] = 0.0

# Run the driver to solve the problem
p.run_driver()
print(p.get_val('traj.phase0.timeseries.time')[-1])

# Check the validity of our results by using scipy.integrate.solve_ivp to integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 4.5))
axes[0].plot(p.get_val('traj.phase0.timeseries.states:x'),
      p.get_val('traj.phase0.timeseries.states:y'), 'ro', label = 'Solution')
axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:x'),
             sim_out.get_val('traj.phase0.timeseries.states:y'),
             'b-', label = 'Simulation')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].legend()
axes[0].grid()

#axes[1].plot(p.get_val('traj.phase0.timeseries.time'), p.get_val('traj.phase0.timeseries.controls:theta', units = 'deg'), 'ro', label = 'solution')
#axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'), sim_out.get_val('traj.phase0.timeseries.controls:theta', units = 'deg'), 'b-', label = 'simulation')
#axes[1].set_xlabel('time (s)')
#axes[1].set_ylabel(r'$\theta$ (deg)')
#axes[1].legend()
#axes[1].grid()

axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.cflag',
             units = 'm'),
             'ro', label = 'Solution')
axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.cflag',
             units = 'm'), 'b-', label = 'Simulation')
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('$v$ (m/s)')
axes[1].legend()
axes[1].grid()

#Add circles
circle0 = plt.Circle((x0, y0), 1, color = 'k', fill = False, hatch = '///')
circle1 = plt.Circle((x1, y1), 1, color = 'k', fill = False, hatch = '///')
circle2 = plt.Circle((x2, y2), 1, color = 'k', fill = False, hatch = '///')
axes[0].add_artist(circle0)
axes[0].add_artist(circle1)
axes[0].add_artist(circle2)
# plt.savefig('fig1_midinit.pdf')



#Additional plots
x_v = np.linspace(0, 10, 50)
y_v = np.linspace(0, 10, 50)
x_m, y_m = np.meshgrid(x_v, y_v)
r0_m = np.sqrt((x_m-x0)**2 + (y_m-y0)**2)
r1_m = np.sqrt((x_m-x1)**2 + (y_m-y1)**2)
r2_m = np.sqrt((x_m-x2)**2 + (y_m-y2)**2)
eta_m = pen_f(r0_m, r1_m, r2_m)
z_m = 5 * eta_m
fig, ax = plt.subplots(1,1)
cplot = ax.contourf(x_m, y_m, z_m)
cbar = fig.colorbar(cplot) # Add a colorbar to a plot
cbar.set_label('velocity (m/s)')
ax.set_xlabel('x position (m)')
ax.set_ylabel('y position (m)')
ax.grid()

#Add circles
circle0 = plt.Circle((x0, y0), 1, color = 'k', fill = False)
circle1 = plt.Circle((x1, y1), 1, color = 'k', fill = False)
circle2 = plt.Circle((x2, y2), 1, color = 'k', fill = False)
ax.add_artist(circle0)
ax.add_artist(circle1)
ax.add_artist(circle2)
# plt.savefig('velpen_midinit.pdf')
plt.show()