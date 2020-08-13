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

from six import iteritems

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

        info = ParOpt.getOptionsInfo()

        for name in info:
            default = info[name].default
            descript = info[name].descript
            values = info[name].values
            if info[name].option_type == 'bool':
                self.options.declare(name, default, types=bool,
                                     desc=descript)
            elif info[name].option_type == 'int':
                self.options.declare(name, default, types=int,
                                     lower=values[0], upper=values[1],
                                     desc=descript)
            elif info[name].option_type == 'float':
                self.options.declare(name, default, types=float,
                                     lower=values[0], upper=values[1],
                                     desc=descript)
            elif info[name].option_type == 'str':
                if default is None:
                    self.options.declare(name, default, types=str,
                                         allow_none=True, desc=descript)
                else:
                    self.options.declare(name, default, types=str,
                                         desc=descript)
            elif info[name].option_type == 'enum':
                self.options.declare(name, default,
                                     values=values, desc=descript)

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
         # - treat equality constraints

        super(ParOptDriver, self)._setup_driver(problem)

        # Raise error if multiple objectives are provided
        if len(self._objs) > 1:
            msg = 'ParOpt currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.__class__.__name__))

        # Create the ParOptProblem from the OpenMDAO problem
        self.paropt_problem = ParOptProblem(problem)

        # Take only the options declared from ParOpt
        info = ParOpt.getOptionsInfo()
        paropt_options = {}
        for key in self.options:
            if key in info.keys():
                paropt_options[key] = self.options[key]

        self.opt = ParOpt.Optimizer(self.paropt_problem, paropt_options)

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
        self.opt.optimize()

        return False


class ParOptProblem(ParOpt.Problem):

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

        # Get the design variable objects from OpenMDAO
        self.om_dvs = self.problem.driver._designvars

        # Get the objective and constraint objects from OpenMDAO
        self.om_con = self.problem.driver._cons
        self.om_obj = self.problem.driver._objs

        # Get the number of design vars from the openmdao problem
        self.nvars = 0
        for name, meta in self.om_dvs.items():
            size = len(self.problem[name])
            self.nvars += size

        # Get the number of constraints from the openmdao problem
        self.ncon = 0
        for name, meta in iteritems(self.problem.model.get_constraints()):
            self.ncon += meta['size']

        # Initialize the base class
        super(ParOptProblem, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """ Set the values of the bounds """
        # Todo:
        # - add check that num dvs are consistent
        # - make sure lb/ub are handled for the case where
        # they aren't set

        i = 0
        for name, meta in self.om_dvs.items():
            size = len(self.problem[name])
            x[i:i + size] = self.problem[name]
            lb[i:i + size] = meta['lower']
            ub[i:i + size] = meta['upper']
            i += size

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Todo:
        # - add check that # of constraints are consistent

        # Pass the updated design variables back to OpenMDAO
        i = 0
        for name, meta in self.om_dvs.items():
            size = meta['size']
            self.problem[name] = x[i:i + size]
            i += size

        # Solve the problem
        self.problem.model._solve_nonlinear()

        # Extract the values of the objectives and constraints
        con = np.zeros(self.ncon)

        i = 0
        for name, meta in self.om_con.items():
            size = meta['size']
            con[i:i + size] = self.problem[name]
            i += size

        # We only accept the first gradient
        for name, meta in self.om_obj.items():
            fobj = self.problem[name][0]
            break

        fail = 0

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""

        # The objective gradient
        for name, meta in self.om_obj.items():
            grad = self.problem.compute_totals(of=[name], wrt=self.om_dvs,
                                               return_format='array')
            g[:] = grad[0, :]
            break

        # Extract the constraint gradients
        i = 0
        for name, meta in self.om_con.items():
            size = meta['size']
            cgrad = self.problem.compute_totals(of=[name], wrt=self.om_dvs,
                                                return_format='array')
            for j in range(size):
                A[i + j][:] = cgrad[j, :]
            i += size

        fail = 0

        return fail
