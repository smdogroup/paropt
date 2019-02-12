"""
OpenMDAO Wrapper for the ParOpt optimizer
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

optimizers = ['Interior Point']#, 'Trust Region', 'MMA']

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
        

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare()

    def _setup_driver(self, paropt_problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        paropt_problem : <Problem>
            Pointer
        """
        super(ParOptDriver, self)._setup_driver(paropt_problem)
        opt = self.options['optimizer']
        

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem
        opt = self.options['optimizer']
        model = problem.model
        self.iter_count = 0
        self._total_jac = None


def class ParOptProblem():

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
        self.ncon = None # *

        self.nwcon = None # *
        self.nwblock = None # *
        self.max_qn_subspace = 0 #*
        self.qn_type = None #*

        # Get the design vars from the openmdao problem
        self.x_hist = []
        self.nvars = len(self.x_hist)

        # Get the constraint info from the openmdao problem
        self.ncon = len(self.problem.get_constraints())

        # Initialize the base class
        super(ParOptProblem, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBound(self, x, lb, ub):
        """ Set the values of the bounds """
        
        x[:] = self.problem.get_design_vars()
        lb[:] = -1e10 # *
        ub[:] = 1e10 # *

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        
        # Append the point to the solution history
        self.x_hist.append(np.array(x))

        # Evaluate the objective and constraints
        fail = 0
        con = self.problem.model.get_constraints()
        fobj = self.problem.model.get_objectives()
        
        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""
        
        fail = 0
        
        # The objective gradient
        g[0] = 200*(x[1]-x[0]**2)*(-2*x[0]) - 2*(1-x[0])
        g[1] = 200*(x[1]-x[0]**2)

        # The constraint gradient
        A[0][0] = 1.0
        A[0][1] = 1.0
        
        return fail
        
        
