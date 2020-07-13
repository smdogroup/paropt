#Import paths
import sys
# sys.path.insert(0, 'C:\msys64\home\Mark\pyoptsparse')
# sys.path.insert(0, 'C:\msys64\home\Mark\pyoptsparse-build\lib\python3.8\site-packages')
import matplotlib, scipy, openmdao.api as om, numpy as np, dymos as dm, matplotlib.pyplot as plt
from matplotlib import animation, rc
matplotlib.use('TkAgg')
rc('animation', html = 'html5')

#Define penalty function
def pen_f(r0, r1, r2):
    p = 10
    return 1 / (1 + np.exp(-p * (r0 - 1))) *\
           1 / (1 + np.exp(-p * (r1 - 1))) *\
           1 / (1 + np.exp(-p * (r2 - 1)))

def position0_f(x0, y0, t, xdot = 0, ydot = 0):
    return x0 + xdot * t, y0 + ydot * t

#Define the problem class (which sets up the problem)
class ODESystem1(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        #ode_options.declare_time(units='s', targets = ['comp.time'])

        # Inputs
        self.add_input('x', val=np.zeros(nn), desc='Horizontal Position', units='m')
        self.add_input('y', val=np.zeros(nn), desc='Vertical Position', units='m')
        self.add_input('t', val=np.zeros(nn), desc='time', units='s')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')
        self.add_output('xdot', val=np.zeros(nn), desc='horizontal velocity', units='m/s')
        self.add_output('ydot', val=np.zeros(nn), desc='vertical velocity', units='m/s')
        self.add_output('p1', val=np.zeros(nn), desc='collision detection', units='m') #for plotting purposes
        self.add_output('p2', val=np.zeros(nn), desc='collision detection', units='m') #for plotting purposes
        self.add_output('p3', val=np.zeros(nn), desc='collision detection', units='m') #for plotting purposes
        self.add_output('x1', val=np.zeros(nn), desc='obstacle x position', units='m') #for plotting purposes
        self.add_output('y1', val=np.zeros(nn), desc='obstacle y position', units='m') #for plotting purposes
        self.add_output('x2', val=np.zeros(nn), desc='obstacle x position', units='m') #for plotting purposes
        self.add_output('y2', val=np.zeros(nn), desc='obstacle y position', units='m') #for plotting purposes
        self.add_output('x3', val=np.zeros(nn), desc='obstacle x position', units='m') #for plotting purposes
        self.add_output('y3', val=np.zeros(nn), desc='obstacle y position', units='m') #for plotting purposes
        #self.add_output('vdot', val=np.zeros(nn), desc='acceleration mag.', units='m/s**2')
        #self.add_output('v_out', val=np.zeros(nn), desc='veclity', units='m/s') #for plotting purposes

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)
        #self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)


        self.declare_partials('xdot', wrt='x', rows=arange, cols=arange)
        self.declare_partials('xdot', wrt='y', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials('ydot', wrt='x', rows=arange, cols=arange)
        self.declare_partials('ydot', wrt='y', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials('p1', 'x', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p1', 'y', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p1', 't', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p2', 'x', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p2', 'y', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p2', 't', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p3', 'x', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p3', 'y', method='fd', step_calc='rel', step=1e-10)
        self.declare_partials('p3', 't', method='fd', step_calc='rel', step=1e-10)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        x = inputs['x']
        y = inputs['y']
        t = inputs['t']

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x0n,y0n = position0_f(x0,y0,t, xdot = 0, ydot = -1)
        x1n,y1n = position0_f(x1,y1,t, xdot = 0, ydot = 1)
        x2n,y2n = position0_f(x2,y2,t, xdot = -1, ydot = -1)
        r0 = np.sqrt((x-x0n)**2 + (y-y0n)**2)
        r1 = np.sqrt((x-x1n)**2 + (y-y1n)**2)
        r2 = np.sqrt((x-x2n)**2 + (y-y2n)**2)
        outputs['xdot'] = 5 * sin_theta
        outputs['ydot'] = -5 * cos_theta
        outputs['p1'] = r0**2
        outputs['p2'] = r1**2
        outputs['p3'] = r2**2
        outputs['x1'] = x0n
        outputs['x2'] = x1n
        outputs['x3'] = x2n
        outputs['y1'] = y0n
        outputs['y2'] = y1n
        outputs['y3'] = y2n
        #outputs['p2'] = r1**2
        #outputs['p3'] = r2**2
        #outputs['v_out'] = 5*eta

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        x = inputs['x']
        y = inputs['y']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        jacobian['xdot', 'x'] = 0
        jacobian['xdot', 'y'] = 0
        jacobian['xdot', 'theta'] = 5*cos_theta
        jacobian['ydot', 'x'] = 0
        jacobian['ydot', 'y'] = 0
        jacobian['ydot', 'theta'] = 5*sin_theta

#Define obstacles
x0, y0 = 3, 3
x1, y1 = 5, 5
x2, y2 = 8, 8

# Define the OpenMDAO problem
p = om.Problem(model = om.Group())

# Define a Trajectory object
traj = dm.Trajectory()

#Add model system
p.model.add_subsystem('traj', subsys = traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class = ODESystem1, transcription = dm.GaussLobatto(num_segments = 10, order = 3))
traj.add_phase(name = 'phase0', phase = phase)

# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
phase.set_time_options(fix_initial = True, duration_bounds = (0.5, 10.0), units = 's', targets = ['t'])

# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The rate source points to the output in the ODE which provides the time derivative of the given state.
phase.add_state('x', fix_initial = True, fix_final = True, units = 'm', rate_source = 'xdot', targets = ['x'])
phase.add_state('y', fix_initial = True, fix_final = True, units = 'm', rate_source = 'ydot', targets = ['y'])

# Define theta as a control.
phase.add_control(name = 'theta', units = 'rad', lower = 0, upper = np.pi, targets = ['theta'])

#Define constraints
phase.add_timeseries_output(name = 'p1', units = 'm')
phase.add_timeseries_output(name = 'p2', units = 'm')
phase.add_timeseries_output(name = 'p3', units = 'm')
phase.add_timeseries_output(name = 'x1', units = 'm')
phase.add_timeseries_output(name = 'x2', units = 'm')
phase.add_timeseries_output(name = 'x3', units = 'm')
phase.add_timeseries_output(name = 'y1', units = 'm')
phase.add_timeseries_output(name = 'y2', units = 'm')
phase.add_timeseries_output(name = 'y3', units = 'm')
phase.add_path_constraint(name = 'p1', lower = 1, upper = None, units = 'm')
phase.add_path_constraint(name = 'p2', lower = 1, upper = None, units = 'm')
phase.add_path_constraint(name = 'p3', lower = 1, upper = None, units = 'm')

# Minimize final time.
phase.add_objective('time', loc = 'final')

# Set the driver.
#p.driver = om.SimpleGADriver()
p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver()
# p.driver.options['optimizer'] = 'SNOPT'
# p.driver.options['optimizer'] = 'IPOPT'

p.driver.options['optimizer'] = 'ParOpt'
p.driver.opt_settings['algorithm'] = 'tr'
p.driver.opt_settings['tr_linfty_tol'] = 1e-30
p.driver.opt_settings['tr_l1_tol'] = 1e-30
p.driver.opt_settings['output_level'] = 0
p.driver.opt_settings['qn_type'] = 'bfgs'
p.driver.opt_settings['max_major_iters'] = 500
p.driver.opt_settings['tr_max_iterations'] = 50
p.driver.opt_settings['qn_update_type'] = 'damped_update'
# p.driver.opt_settings['norm_type'] = 'infinity'
# p.driver.opt_settings['barrier_strategy'] = 'monotone'
# p.driver.opt_settings['starting_point_strategy'] = 'affine_step'
# p.driver.opt_settings['tr_steering_barrier_strategy'] = 'mehrotra_predictor_corrector'
# p.driver.opt_settings['tr_steering_starting_point_strategy'] = 'affine_step'
# p.driver.opt_settings['use_line_search'] = True
p.driver.opt_settings['penalty_gamma'] = 1e2
p.driver.opt_settings['tr_min_size'] = 1e-2
p.driver.opt_settings['tr_adaptive_gamma_update'] = False
p.driver.opt_settings['tr_use_filter'] = True
p.driver.opt_settings['tr_use_soc'] = True
p.driver.opt_settings['tr_soc_use_quad_model'] = True


# Allow OpenMDAO to automatically determine our sparsity pattern. Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check = True)

# Now that the OpenMDAO problem is setup, we can set the values of the states
#p['traj.phase0.states:y'][:] = 0
#p['traj.phase0.states:y'][-1] = 10
p.set_val('traj.phase0.states:x', phase.interpolate(ys = [0, 10], nodes = 'state_input'), units = 'm')
p.set_val('traj.phase0.states:y', phase.interpolate(ys = [0, 10], nodes = 'state_input'), units = 'm')
p.set_val('traj.phase0.controls:theta', phase.interpolate(ys = [90, 90], nodes = 'control_input'), units = 'deg')

# Run the driver to solve the problem
p.run_driver()

# Check the validity of our results by using scipy.integrate.solve_ivp to integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 4.5))
axes[0].plot(p.get_val('traj.phase0.timeseries.states:x'), p.get_val('traj.phase0.timeseries.states:y'), 'ro', label = 'Solution')
axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:x'), sim_out.get_val('traj.phase0.timeseries.states:y'), 'b-', label = 'Simulation')
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('y (m)')
axes[0].legend()
axes[0].grid()

axes[1].plot(p.get_val('traj.phase0.timeseries.x1'), p.get_val('traj.phase0.timeseries.y1'), 'bo', label = 'Solution')
axes[1].plot(p.get_val('traj.phase0.timeseries.x2'), p.get_val('traj.phase0.timeseries.y2'), 'g^', label = 'Solution')
axes[1].plot(p.get_val('traj.phase0.timeseries.x3'), p.get_val('traj.phase0.timeseries.y3'), 'rs', label = 'Solution')
axes[1].plot(sim_out.get_val('traj.phase0.timeseries.x1'), sim_out.get_val('traj.phase0.timeseries.y1'), 'b--', label = 'Simulation')
axes[1].plot(sim_out.get_val('traj.phase0.timeseries.x2'), sim_out.get_val('traj.phase0.timeseries.y2'), 'g--', label = 'Simulation')
axes[1].plot(sim_out.get_val('traj.phase0.timeseries.x3'), sim_out.get_val('traj.phase0.timeseries.y3'), 'r--', label = 'Simulation')
axes[1].set_xlabel('px (m)')
axes[1].set_ylabel('py (m)')
axes[1].set_xlim(0,10)
axes[1].set_ylim(0,10)
axes[1].legend()
axes[1].grid()

#axes[1].plot(p.get_val('traj.phase0.timeseries.time'), p.get_val('traj.phase0.timeseries.controls:theta', units = 'deg'), 'ro', label = 'solution')
#axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'), sim_out.get_val('traj.phase0.timeseries.controls:theta', units = 'deg'), 'b-', label = 'simulation')
#axes[1].set_xlabel('time (s)')
#axes[1].set_ylabel(r'$\theta$ (deg)')
#axes[1].legend()
#axes[1].grid()

axes[2].plot(p.get_val('traj.phase0.timeseries.time'), p.get_val('traj.phase0.timeseries.p1', units = 'm'), 'ro', label = 'Solution')
axes[2].plot(p.get_val('traj.phase0.timeseries.time'), p.get_val('traj.phase0.timeseries.p2', units = 'm'), 'g^', label = 'Solution')
axes[2].plot(p.get_val('traj.phase0.timeseries.time'), p.get_val('traj.phase0.timeseries.p3', units = 'm'), 'bs', label = 'Solution')
axes[2].plot(sim_out.get_val('traj.phase0.timeseries.time'), sim_out.get_val('traj.phase0.timeseries.p1', units = 'm'), 'r--', label = 'Simulation')
axes[2].plot(sim_out.get_val('traj.phase0.timeseries.time'), sim_out.get_val('traj.phase0.timeseries.p2', units = 'm'), 'g--', label = 'Simulation')
axes[2].plot(sim_out.get_val('traj.phase0.timeseries.time'), sim_out.get_val('traj.phase0.timeseries.p3', units = 'm'), 'b--', label = 'Simulation')
axes[2].set_xlabel('time (s)')
axes[2].set_ylabel('$d$ (m)')
axes[2].set_ylim(0,10)
axes[2].legend()
axes[2].grid()

#Add circles
circle0 = plt.Circle((x0, y0), 1, color = 'k', fill = False, hatch = '///')
circle1 = plt.Circle((x1, y1), 1, color = 'k', fill = False, hatch = '///')
circle2 = plt.Circle((x2, y2), 1, color = 'k', fill = False, hatch = '///')
axes[0].add_artist(circle0)
axes[0].add_artist(circle1)
axes[0].add_artist(circle2)


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
#plt.show()



#ANIMATION ##################################################################
x_v, y_v = p.get_val('traj.phase0.timeseries.states:x'), p.get_val('traj.phase0.timeseries.states:y')
t_v = p.get_val('traj.phase0.timeseries.time')
d1_v = p.get_val('traj.phase0.timeseries.p1')
d2_v = p.get_val('traj.phase0.timeseries.p2')
d3_v = p.get_val('traj.phase0.timeseries.p3')
x1_v, y1_v = p.get_val('traj.phase0.timeseries.x1'), p.get_val('traj.phase0.timeseries.y1')
x2_v, y2_v = p.get_val('traj.phase0.timeseries.x2'), p.get_val('traj.phase0.timeseries.y2')
x3_v, y3_v = p.get_val('traj.phase0.timeseries.x3'), p.get_val('traj.phase0.timeseries.y3')

#Animation wrapper function
def run_animation():

    #Allow pausing on click
    anim_running = True
    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    #Animation function (called sequentially)
    def animate(i):

        #Update obstacles
        patch0.center = (x1_v[i], y1_v[i])
        patch1.center = (x2_v[i], y2_v[i])
        patch2.center = (x3_v[i], y3_v[i])

        #Update lines
        traj1.set_data(x_v[:i+1], y_v[:i+1])
        dist1.set_data(t_v[:i+1], d1_v[:i+1])
        dist2.set_data(t_v[:i+1], d2_v[:i+1])
        dist3.set_data(t_v[:i+1], d3_v[:i+1])


        return patch0, patch1, patch2, traj1, dist1, dist2, dist3

    #Initialization function: plot the background of each frame
    def init():

        #Initialize obstacles
        axs[0].add_patch(patch0)
        axs[0].add_patch(patch1)
        axs[0].add_patch(patch2)

        #Initialize trajectory and distance lines
        traj1.set_data([], [])
        dist1.set_data([], [])
        dist2.set_data([], [])
        dist3.set_data([], [])

        axs[0].set_xlabel('x position (m)')
        axs[0].set_ylabel('y position (m)')
        axs[1].set_xlabel('iteration')
        axs[1].set_ylabel('objective')

        return patch0, patch1, patch2, traj1, dist1, dist2, dist2

    fig, axs = plt.subplots(1,2, figsize = (12, 4.5))
    patch0 = plt.Circle((x0, y0), 1, color = 'k', fill = False, hatch = '///')
    patch1 = plt.Circle((x1, y1), 1, color = 'k', fill = False, hatch = '///')
    patch2 = plt.Circle((x2, y2), 1, color = 'k', fill = False, hatch = '///')
    traj1, = axs[0].plot([], [], lw = 2, color = 'k', marker = 'o', ls = '--')
    dist1, = axs[1].plot([], [], lw = 2, color = 'b', marker = '', ls = '-')
    dist2, = axs[1].plot([], [], lw = 2, color = 'g', marker = '', ls = '-')
    dist3, = axs[1].plot([], [], lw = 2, color = 'r', marker = '', ls = '-')
    fig.canvas.mpl_connect('button_press_event', onClick)
    anim = animation.FuncAnimation(fig, animate, init_func = init, fargs = (), frames = x_v.shape[0], interval = 100, blit = True)
    axs[0].set_xlim((0, 10))
    axs[0].set_ylim((0, 10))
    axs[1].set_xlim((0, 10))
    axs[1].set_ylim((0, 10))
    plt.show()
    anim.save('movingObstacle.gif', writer = 'imagemagick', fps = 5)
# run_animation()