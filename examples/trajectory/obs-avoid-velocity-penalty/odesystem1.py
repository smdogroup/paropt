# Define the problem class (which sets up the problem)
import sys

# sys.path.insert(0, 'C:/msys64/home/Charizard/tools_traj_opt/pyoptsparse')
# sys.path.insert(0, 'C:/msys64/home/Charizard/pyoptsparse-build/lib/python3.8/site-packages')

import openmdao.api as om
import numpy as np
import dymos as dm

# Define obstacles
x0, y0 = 8, 6
x1, y1 = 5, 5
x2, y2 = 8, 8
p = 45

# Define penalty function
# def pen_f(r0, r1, r2):
#    p = 10
#    return 1 / (1 + np.exp(-p * (r0 - 1))) *\
#           1 / (1 + np.exp(-p * (r1 - 1))) *\
#           1 / (1 + np.exp(-p * (r2 - 1)))


class ODESystem1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        # ###############
        # Inputs
        # ###############
        self.add_input("x", val=np.zeros(nn), desc="Horizontal Position", units="m")
        self.add_input("y", val=np.zeros(nn), desc="Vertical Position", units="m")
        # self.add_input('t', val=np.zeros(nn),
        #               desc='Time', units='s')
        self.add_input("theta", val=np.zeros(nn), desc="angle of wire", units="rad")
        # ###############
        # outputs
        # ###############
        self.add_output(
            "xdot", val=np.zeros(nn), desc="horizontal velocity", units="m/s"
        )
        self.add_output("ydot", val=np.zeros(nn), desc="vertical velocity", units="m/s")
        self.add_output(
            "cflag", val=np.zeros(nn), desc="collision detection", units="m"
        )  # for plotting
        # self.add_output('vdot', val=np.zeros(nn), desc='acceleration mag.', units='m/s**2')
        # self.add_output('v_out', val=np.zeros(nn), desc='veclity', units='m/s') #for plotting purposes
        # #################
        # Setup partials
        # #################
        arange = np.arange(self.options["num_nodes"], dtype=int)
        # self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="theta", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="theta", rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="xdot", wrt="y", rows=arange, cols=arange)
        self.declare_partials(of="ydot", wrt="y", rows=arange, cols=arange)
        # self.declare_partials('xdot', 'x', method='fd',
        #                      step_calc='rel', step=1e-10)
        # self.declare_partials('xdot', 'y', method='fd',
        #                      step_calc='rel', step=1e-10)
        # self.declare_partials('ydot', 'x', method='fd', step_calc='rel',
        #                     step=1e-10)
        # self.declare_partials('ydot', 'y', method='fd', step_calc='rel',
        #                      step=1e-10)
        self.declare_partials(of="cflag", wrt="x", rows=arange, cols=arange)
        self.declare_partials(of="cflag", wrt="y", rows=arange, cols=arange)

    # ###########################################
    def compute(self, inputs, outputs):
        theta = inputs["theta"]
        x = inputs["x"]
        y = inputs["y"]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        r0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 0.01
        r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) + 0.01
        r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2) + 0.01
        # eta = pen_f(r0, r1, r2)
        cflag = (
            1
            / (1 + np.exp(-p * (r0 - 1)))
            * 1
            / (1 + np.exp(-p * (r1 - 1)))
            * 1
            / (1 + np.exp(-p * (r2 - 1)))
        )
        outputs["xdot"] = 5 * cflag * cos_theta
        outputs["ydot"] = 5 * cflag * sin_theta
        outputs["cflag"] = cflag
        # outputs['v_out'] = 5*eta

    # ###########################################
    def compute_partials(self, inputs, jacobian):
        theta = inputs["theta"]
        x = inputs["x"]
        y = inputs["y"]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        r0 = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 0.01
        r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2) + 0.01
        r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2) + 0.01
        cflag = (
            1
            / (1 + np.exp(-p * (r0 - 1)))
            * 1
            / (1 + np.exp(-p * (r1 - 1)))
            * 1
            / (1 + np.exp(-p * (r2 - 1)))
        )
        # eta = pen_f(r0, r1, r2)
        # jacobian['vdot', 'theta'] = 0
        jacobian["xdot", "theta"] = -5 * cflag * sin_theta
        jacobian["ydot", "theta"] = 5 * cflag * cos_theta
        #
        cflag__r0 = cflag / (1 + np.exp(-p * (r0 - 1))) * (p) * np.exp(-p * (r0 - 1))
        cflag__r1 = cflag / (1 + np.exp(-p * (r1 - 1))) * (p) * np.exp(-p * (r1 - 1))
        cflag__r2 = cflag / (1 + np.exp(-p * (r2 - 1))) * (p) * np.exp(-p * (r2 - 1))
        r0__x = (x - x0) / r0
        r0__y = (y - y0) / r0
        r1__x = (x - x1) / r1
        r1__y = (y - y1) / r1
        r2__x = (x - x2) / r2
        r2__y = (y - y2) / r2
        #
        jacobian["cflag", "x"] = (
            cflag__r0 * r0__x + +cflag__r1 * r1__x + +cflag__r2 * r2__x
        )
        #
        jacobian["cflag", "y"] = (
            cflag__r0 * r0__y + +cflag__r1 * r1__y + +cflag__r2 * r2__y
        )

        xdot__cflag = 5 * cos_theta
        ydot__cflag = 5 * sin_theta

        jacobian["xdot", "x"] = xdot__cflag * jacobian["cflag", "x"]
        jacobian["xdot", "y"] = xdot__cflag * jacobian["cflag", "y"]

        jacobian["ydot", "x"] = ydot__cflag * jacobian["cflag", "x"]
        jacobian["ydot", "y"] = ydot__cflag * jacobian["cflag", "y"]
