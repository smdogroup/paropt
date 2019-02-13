"""
OpenMDAO Wrapper for ParOpt
"""

from __future__ import print_function
import sys
import numpy as np
import mpi4py.MPI as MPI
from paropt import ParOpt

import openmdao
import openmdao.utils.coloring as coloring_mod
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.general_utils import warn_deprecation

_optimizers = ['Interior Point']#, 'Trust Region', 'MMA']

class ParOptDriver(Driver):
    """
    Driver wrapper for ParOpt

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from scipy.optimize call.
    opt_settings : dict
        Dictionary of solver-specific options. See the ParOpt documentation.


    """

    def __init__(self, **kwargs):
        """
        Initialize the ParOptDriver

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Driver options.
        """
        super(ParOptDriver, self).__init__(**kwargs)

        self.result = None
        self._dvlist = None
        self.fail = False
        self.iter_count = 0

        return

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """

        # Many more options to add
        
        self.options.declare('optimizer', 'Interior Point', values=_optimizers,
        desc = 'Optimization algorithm to use')
        

        return

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        paropt_problem : <Problem>
            Pointer
        """
         # TODO:
         # - logic for different opt algorithms
         # - treat equality constraints
        
        super(ParOptDriver, self)._setup_driver(problem)
        opt_type = self.options['optimizer'] # Not currently used

        # Create the ParOptProblem from the OpenMDAO problem
        opt = pyParOpt.ParOptProblem(problem)

        # Set the options into ParOpt (TODO)


        # This opt object will be used again when 'run' is executed
        self.opt = opt

        return 

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        # Note: failure flag is always False

        # Run the optimization, everything else has been setup
        self.opt.optimize()

        return opt.fail


def class ParOptProblem(ParOpt.pyParOptProblem):

    def __init__(self, problem):
        """
        ParOptProblem class to pass to the ParOptDriver. Takes
        in an instance of the OpenMDAO problem class and creates
        a ParOpt problem to be passed into ParOpt through the
        ParOptDriver.
        """

        self.problem = problem
        
        self.comm = self.problem.comm
        self.nvars = None
        self.ncon = None

        self.x_hist = []

        # Get the number of design vars from the openmdao problem
        self.nvars = len(self.problem.get_design_vars())

        # Get the number of constraints from the openmdao problem
        self.ncon = len(self.problem.model.get_constraints())

        # Initialize the base class
        super(ParOptProblem, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBound(self, x, lb, ub):
        """ Set the values of the bounds """
        # Todo:
        # - add check that num dvs are consistent
        # - make sure lb/ub are handled for the case where
        # they aren't set


        # Get design vars from openmdao as a dictionary
        xdict = self.problem.get_design_vars()
        
        # Set the dv and bound values
        for key, value in xdict.items():
            x[xdict.keys().index(key)] = self.problem[key]
            lb[xdict.keys().index(key)] = value['lower']
            ub[xdict.keys().index(key)] = value['upper']

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Todo:
        # - add check that # of constraints are consisten
        # - add check that there is only one objective

        
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Update the design variables in OpenMDAO
        xdict = self.problem.get_design_vars()
        for key, value in xdict.items():
            self.driver.set_design_var(key, np.array(x[xdict.keys().index(key)]))
        # Q: best way to update f, c values?

        # Evaluate the objective and constraints
        condict = self.problem.model.get_constraints()
        objdict = self.problem.model.get_objectives()

        for key, value in condict.items():
            con[condict.keys().index(key)] = self.problem[key]

        fobj = prob[objdict.keys()[0]][0]

        fail = 0
        
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        # Todo:
        # - 

        # Get the gradients from OpenMDAO
        condict = self.problem.model.get_constraints()
        objdict = self.problem.model.get_objectives()

        # The objective gradient 
        g[:] = self.problem.compute_totals(of=objdict.keys()[0], return_format='array')
        
        # The constraint gradient
        for key, value in condict.items():
            A[condict.keys().index(key)][:] = self.problem.compute_totals(of=key, return_format='array')

        fail = 0

        return fail
