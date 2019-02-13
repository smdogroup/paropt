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
_norm_types = ['Infinity', 'L1', 'L2']
_barrier_types = ['Monotone', 'Mehrotra', 'Complementarity fraction']
_start_types = ['None', 'Least squares multipliers', 'Affine step']
_bfgs_updates = ['Skip negative', 'Damped']

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

        # Options currently incomplete
        
        self.options.declare('optimizer', 'Interior Point', values=_optimizers,
                             desc='Type of optimization algorithm')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination. For detailed '
                             'control, use solver-specific options.')
        self.options.declare('maxiter', 200, lower=0,
                             desc='Maximum number of iterations')
        
        desc = 'Finite difference step size. If None, no gradient check will be performed.'
        self.options.declare('dh', None, lower=1e-30,
                             desc=desc, allow_none=True)
        self.options.declare('norm_type', None, values=_norm_types,
                             desc='Norm type', allow_none=True)
        self.options.declare('barrier_strategy', None, values=_barrier_types,
                             desc='Barrier strategy', allow_none=True)
        self.options.declare('start_strategy', None, values=_start_types,
                             desc='Starting point strategy', allow_none=True)
        self.options.declare('penalty_gamma', None, desc='Value of penalty parameter gamma',
                             allow_none=True)
        self.options.declare('barrier_fraction', None,
                             desc='Barrier fraction', allow_none=True)
        self.options.declare('barrier_power', None,
                             desc='Barrier power', allow_none=True)
        self.options.declare('hessian_reset_freq', None,
                             desc='Hessian reset frequency', allow_none=True)
        self.options.declare('qn_diag_factor', None,
                             desc='QN diagonal factor', allow_none=True)
        self.options.declare('BFGS_update_type', None, values=_bfgs_updates,
                             desc='Barrier fraction', allow_none=True)
        self.options.declare('use_sequential_linear', None,
                             desc='Boolean to indicate if a sequential linear method should be used',
                             allow_none=True)
        # Where I left off copying options from ParOpt.pyx

        self.options.declare('output_freq', None,
                             desc='Output frequency', allow_none=True)
        self.options.declare('output_file', None,
                             desc='Output file name', allow_none=True)

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
        self.paropt_problem = ParOptProblem(problem)

        # Set the limited-memory options
        max_qn_subspace = 10
        qn_type = ParOpt.BFGS
        opt = ParOpt.pyParOpt(self.paropt_problem, max_qn_subspace, qn_type)

        # Apply the options to ParOpt
        # Currently incomplete
        opt.setAbsOptimalityTol(self.options['tol'])
        opt.setMaxMajorIterations(self.options['maxiter'])
        if self.options['dh']:
            opt.checkGradients(self.options['dh'])
        if self.options['norm_type']:
            opt.setNormType(self.options['norm_type'])
        if self.options['barrier_strategy']:
            opt.setBarrierStrategy(self.options['barrier_strategy'])

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

        return False


class ParOptProblem(ParOpt.pyParOptProblem):

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
        self.nvars = len(self.problem.model.get_design_vars())

        # Get the number of constraints from the openmdao problem
        self.ncon = len(self.problem.model.get_constraints())

        # Initialize the base class
        super(ParOptProblem, self).__init__(self.comm, self.nvars, self.ncon)

        return

    def getVarsAndBounds(self, x, lb, ub):
        """ Set the values of the bounds """
        # Todo:
        # - add check that num dvs are consistent
        # - make sure lb/ub are handled for the case where
        # they aren't set

        # Get design vars from openmdao as a dictionary
        xdict = self.problem.model.get_design_vars()
        
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
        xdict = self.problem.model.get_design_vars()
        for key, value in xdict.items():
            self.problem[key] = x[xdict.keys().index(key)]
        # Q: best way to update f, c values?

        # Evaluate the objective and constraints
        condict = self.problem.model.get_constraints()
        objdict = self.problem.model.get_objectives()

        con = np.zeros(self.ncon)
        for key, value in condict.items():
            con[condict.keys().index(key)] = self.problem[key]

        fobj = self.problem[objdict.keys()[0]][0]

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
        # print(objdict.keys()[0])
        try:
            g[:] = self.problem.compute_totals(of=objdict.keys()[0],
                                               return_format='array')
        except KeyError:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            
        # return_format='array')
        
        # The constraint gradient
        # for key, value in condict.items():
        #     print('key = ', key)
        #     A[condict.keys().index(key)][:] = self.problem.compute_totals(of=key,
        #                                                                  return_format='array')

        fail = 0

        return fail
