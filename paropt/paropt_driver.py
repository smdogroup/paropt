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

_optimizers = ['Interior Point', 'Trust Region']
_qn_types = ['BFGS', 'SR1', 'No Hessian approx']
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

        self.options.declare('optimizer', 'Interior Point', values=_optimizers,
                             desc='Type of optimization algorithm')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination')
        self.options.declare('maxiter', 200, lower=0, types=int,
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
        self.options.declare('penalty_gamma', None,
                             desc='Value of penalty parameter gamma',
                             allow_none=True)
        self.options.declare('barrier_fraction', None,
                             desc='Barrier fraction', allow_none=True)
        self.options.declare('barrier_power', None,
                             desc='Barrier power', allow_none=True)
        self.options.declare('hessian_reset_freq', None, types=int,
                             desc='Hessian reset frequency', allow_none=True)
        self.options.declare('qn_type', 'BFGS', values=_qn_types,
                             desc='Type of Hessian approximation to use')
        self.options.declare('max_qn_subspace', 10, types=int,
                             desc='Size of the QN subspace')
        self.options.declare('qn_diag_factor', None,
                             desc='QN diagonal factor', allow_none=True)
        self.options.declare('bfgs_update_type', None, values=_bfgs_updates,
                             desc='Barrier fraction', allow_none=True)

        desc = 'Boolean to indicate if a sequential linear method should be used'
        self.options.declare('use_sequential_linear', None, types=bool,
                             desc=desc, allow_none=True)
        self.options.declare('affine_step_multiplier_min', None, allow_none=True,
                             desc='Minimum multiplier for affine step')
        self.options.declare('init_barrier_parameter', None, allow_none=True,
                             desc='Initial barrier parameter')
        self.options.declare('relative_barrier', None, allow_none=True,
                             desc='Relative barrier parameter')
        self.options.declare('set_qn', None, allow_none=True,
                             desc='Quasi-Newton')
        self.options.declare('qn_updates', None, allow_none=True, types=bool,
                             desc='Update the Quasi-Newton')

        # Line-search parameters
        self.options.declare('use_line_search', None, allow_none=True, types=bool,
                             desc='Use line search')
        self.options.declare('max_ls_iters', None, allow_none=True, types=int,
                             desc='Max number of line search iterations')
        self.options.declare('backtrack_ls', None, allow_none=True, types=bool,
                             desc='Use backtracking line search')
        self.options.declare('armijo_param', None, allow_none=True,
                             desc='Armijo parameter for line search')
        self.options.declare('penalty_descent_frac', None, allow_none=True,
                             desc='Descent fraction penalty')
        self.options.declare('min_penalty_param', None, allow_none=True,
                             desc='Minimum line search penalty')

        # GMRES parameters
        self.options.declare('use_hvec_prod', None, allow_none=True, types=bool,
                             desc='Use Hvec product with GMRES')
        self.options.declare('use_diag_hessian', None, allow_none=True, types=bool,
                             desc='Use a diagonal Hessian')
        self.options.declare('use_qn_gmres_precon', None, allow_none=True, types=bool,
                             desc='Use QN GMRES preconditioner')
        self.options.declare('set_nk_switch_tol', None, allow_none=True,
                             desc='NK switch tolerance')
        self.options.declare('eisenstat_walker_param', None, allow_none=True,
                             desc='Eisenstat Walker parameters: array([gamma, alpha])')
        self.options.declare('gmres_tol', None, allow_none=True,
                             desc='GMRES tolerances: array([rtol, atol])')
        self.options.declare('gmres_subspace_size', None, allow_none=True, types=int,
                             desc='GMRES subspace size')

        # Output options
        self.options.declare('output_freq', None, allow_none=True, types=int,
                             desc='Output frequency')
        self.options.declare('output_file', None, allow_none=True,
                             desc='Output file name')
        self.options.declare('major_iter_step_check', None, allow_none=True, types=int,
                             desc='Major iter step check')
        self.options.declare('output_level', None, allow_none=True, types=int,
                             desc='Output level')
        self.options.declare('grad_check_freq', None, allow_none=True,
                             desc='Gradient check frequency: array([freq, step_size])')

        # Set options for the trust region method
        self.options.declare('tr_adaptive_gamma_update', default=True, types=bool,
                             desc='Use the adaptive penalty algorithm')
        self.options.declare('tr_min_size', default=1e-6, lower=0.0,
                             desc='Minimum trust region radius size')
        self.options.declare('tr_max_size', default=10.0, lower=0.0,
                             desc='Maximum trust region radius size')
        self.options.declare('tr_init_size', default=1.0, lower=0.0,
                             desc='Initial trust region radius size')
        self.options.declare('tr_eta', default=0.25, lower=0.0, upper=1.0,
                             desc='Trust region radius acceptance ratio')
        self.options.declare('tr_penalty_gamma', default=10.0, lower=0.0,
                             desc='Trust region penalty parameter value')
        self.options.declare('tr_penalty_gamma_max', default=1e4, lower=0.0,
                             desc='Trust region maximum penalty parameter value')
        self.options.declare('tr_max_iterations', default=200, types=int,
                             desc='Maximum trust region iterations')

        # Trust region convergence tolerances
        self.options.declare('tr_infeas_tol', default=1e-5, lower=0.0,
                             desc='Trust region infeasibility tolerance (l1 norm)')
        self.options.declare('tr_l1_tol', default=1e-5, lower=0.0,
                             desc='Trust region optimality tolerance (l1 norm)')
        self.options.declare('tr_linfty_tol', default=1e-5, lower=0.0,
                             desc='Trust region optimality tolerance (l-infinity norm)')

        # Trust region output file name
        self.options.declare('tr_output_file', None, allow_none=True,
                             desc='Trust region output file name')
        self.options.declare('tr_write_output_freq', default=10, types=int,
                             desc='Trust region output frequency')

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
        opt_type = self.options['optimizer']

        # Raise error if multiple objectives are provided
        if len(self._objs) > 1:
            msg = 'ParOpt currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.__class__.__name__))

        # Set the limited-memory options
        max_qn_subspace = self.options['max_qn_subspace']
        if self.options['qn_type'] == 'BFGS':
            qn_type = ParOpt.BFGS
        elif self.options['qn_type'] == 'SR1':
            qn_type = ParOpt.SR1
        elif self.options['qn_type'] == 'No Hessian approx':
            qn_type = ParOpt.NO_HESSIAN_APPROX
        else:
            qn_type = ParOpt.BFGS

        # Create the ParOptProblem from the OpenMDAO problem
        self.paropt_problem = ParOptProblem(problem)

        # Create the problem
        if opt_type == 'Trust Region':
            # For the trust region method, you have to use a Hessian
            # approximation
            if qn_type == ParOpt.NO_HESSIAN_APPROX:
                qn = ParOpt.BFGS
            if max_qn_subspace < 1:
                max_qn_subspace = 1

            # Create the quasi-Newton method
            qn = ParOpt.LBFGS(self.paropt_problem, subspace=max_qn_subspace)

            # Retrieve the options for the trust region problem
            tr_min_size = self.options['tr_min_size']
            tr_max_size = self.options['tr_max_size']
            tr_eta = self.options['tr_eta']
            tr_penalty_gamma = self.options['tr_penalty_gamma']
            tr_init_size = self.options['tr_init_size']

            # Create the trust region sub-problem
            tr_init_size = min(tr_max_size, max(tr_init_size, tr_min_size))
            tr = ParOpt.pyTrustRegion(self.paropt_problem, qn, tr_init_size,
                                      tr_min_size, tr_max_size,
                                      tr_eta, tr_penalty_gamma)

            # Set the penalty parameter
            tr.setPenaltyGammaMax(self.options['tr_penalty_gamma_max'])
            tr.setMaxTrustRegionIterations(self.options['tr_max_iterations'])

            # Trust region convergence tolerances
            infeas_tol = self.options['tr_infeas_tol']
            l1_tol = self.options['tr_l1_tol']
            linfty_tol = self.options['tr_linfty_tol']
            tr.setTrustRegionTolerances(infeas_tol, l1_tol, linfty_tol)

            # Trust region output file name
            if self.options['tr_output_file'] is not None:
                tr.setOutputFile(self.options['tr_output_file'])
                tr.setOutputFrequency(self.options['tr_write_output_freq'])

            # Create the interior-point optimizer for the trust region sub-problem
            opt = ParOpt.pyParOpt(tr, 0, ParOpt.NO_HESSIAN_APPROX)
            self.tr = tr
        else:
            # Create the ParOpt object with the interior point method
            opt = ParOpt.pyParOpt(self.paropt_problem, max_qn_subspace,
                                  qn_type)

        # Apply the options to ParOpt
        # Currently incomplete
        opt.setAbsOptimalityTol(self.options['tol'])
        opt.setMaxMajorIterations(self.options['maxiter'])
        if self.options['dh']:
            opt.checkGradients(self.options['dh'])
        if self.options['norm_type']:
            opt.setNormType(self.options['norm_type'])

        # Set barrier strategy
        if self.options['barrier_strategy']:
            if self.options['barrier_strategy'] == 'Monotone':
                barrier_strategy = ParOpt.MONOTONE
            elif self.options['barrier_strategy'] == 'Mehrotra':
                barrier_strategy = ParOpt.MEHROTRA
            elif self.options['barrier_strategy'] == 'Complementarity fraction':
                barrier_strategy = ParOpt.COMPLEMENTARITY_FRACTION
            opt.setBarrierStrategy(barrier_strategy)

        # Set starting point strategy
        if self.options['start_strategy']:
            if self.options['barrier_strategy'] == 'None':
                start_strategy = ParOpt.NO_START_STRATEGY
            elif self.options['barrier_strategy'] == 'Least squares multipliers':
                start_strategy = ParOpt.LEAST_SQUARES_MULTIPLIERS
            elif self.options['barrier_strategy'] == 'Affine step':
                start_strategy = ParOpt.AFFINE_STEP
            opt.setStartingPointStrategy(start_strategy)

        # Set norm type
        if self.options['norm_type']:
            if self.options['norm_type'] == 'Infinity':
                norm_type = ParOpt.INFTY_NORM
            elif self.options['norm_type'] == 'L1':
                norm_type = ParOpt.L1_NORM
            elif self.options['norm_type'] == 'L2':
                norm_type = ParOpt.L2_NORM
            opt.setBarrierStrategy(norm_type)

        # Set BFGS update strategy
        if self.options['bfgs_update_type']:
            if self.options['bfgs_update_type'] == 'Skip negative':
                bfgs_update_type = ParOpt.SKIP_NEGATIVE_CURVATURE
            elif self.options['bfgs_update_type'] == 'Damped':
                bfgs_update_type = ParOpt.DAMPED_UPDATE
            opt.setBFGSUpdateType(bfgs_update_type)

        if self.options['penalty_gamma']:
            opt.setPenaltyGamma(self.options['penalty_gamma'])

        if self.options['barrier_fraction']:
            opt.setBarrierFraction(self.options['barrier_fraction'])

        if self.options['barrier_power']:
            opt.setBarrierPower(self.options['barrier_power'])

        if self.options['hessian_reset_freq']:
            opt.setHessianResetFrequency(self.options['hessian_reset_freq'])

        if self.options['qn_diag_factor']:
            opt.setQNDiagonalFactor(self.options['qn_diag_factor'])

        if self.options['use_sequential_linear']:
            opt.setSequentialLinearMethod(self.options['use_sequential_linear'])

        if self.options['affine_step_multiplier_min']:
            opt.setStartAffineStepMultiplierMin(self.options['affine_step_multiplier_min'])

        if self.options['init_barrier_parameter']:
            opt.setInitBarrierParameter(self.options['init_barrier_parameter'])

        if self.options['relative_barrier']:
            opt.setRelativeBarrier(self.options['relative_barrier'])

        if self.options['set_qn']:
            opt.setQuasiNewton(self.options['set_qn'])

        if self.options['qn_updates']:
            opt.setUseQuasiNewtonUpdates(self.options['qn_updates'])

        if self.options['use_line_search']:
            opt.setUseLineSearch(self.options['use_line_search'])

        if self.options['max_ls_iters']:
            opt.setMaxLineSearchIters(self.options['max_ls_iters'])

        if self.options['backtrack_ls']:
            opt.setBacktrackingLineSearch(self.options['backtrack_ls'])

        if self.options['armijo_param']:
            opt.setArmijoParam(self.options['armijo_param'])

        if self.options['penalty_descent_frac']:
            opt.setPenaltyDescentFraction(self.options['penalty_descent_frac'])

        if self.options['min_penalty_param']:
            opt.setMinPenaltyParameter(self.options['min_penalty_param'])

        if self.options['use_hvec_prod']:
            opt.setUseHvecProduct(self.options['use_hvec_prod'])

        if self.options['use_diag_hessian']:
            opt.setUseDiagHessian(self.options['use_diag_hessian'])

        if self.options['use_qn_gmres_precon']:
            opt.setUseQNGMRESPreCon(self.options['use_qn_gmres_precon'])

        if self.options['set_nk_switch_tol']:
            opt.setNKSwitchTolerance(self.options['set_nk_switch_tol'])

        if self.options['eisenstat_walker_param']:
            opt.setEisenstatWalkerParameters(self.options['eisenstat_walker_param'][0],
                                             self.options['eisenstat_walker_param'][1])

        if self.options['gmres_tol']:
            opt.setGMRESTolerances(self.options['gmres_tol'][0],
                                   self.options['gmres_tol'][1])

        if self.options['gmres_subspace_size']:
            opt.setGMRESSubspaceSize(self.options['gmres_subspace_size'])

        if self.options['output_freq']:
            opt.setOutputFrequency(self.options['output_freq'])

        if self.options['output_file']:
            opt.setOutputFile(self.options['output_file'])

        if self.options['major_iter_step_check']:
            opt.setMajorIterStepCheck(self.options['major_iter_step_check'])

        if self.options['output_level']:
            opt.setOutputLevel(self.options['output_level'])

        if self.options['grad_check_freq']:
            opt.setGradCheckFrequency(self.options['grad_check_freq'])

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
        if self.options['optimizer'] == 'Trust Region':
            self.tr.optimize(self.opt)
        else:
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

        # Get the design variable names
        self.dvs = [name for name, meta in iteritems(self.problem.model.get_design_vars())]

        # Get the number of design vars from the openmdao problem
        self.nvars = 0
        for name, meta in iteritems(self.problem.model.get_design_vars()):
            self.nvars += meta['size']

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

        # Get design vars from openmdao as a dictionary
        desvars = self.problem.model.get_design_vars()

        i = 0
        for name, meta in iteritems(desvars):
            size = meta['size']
            x[i:i + size] = self.problem[name]
            lb[i:i + size] = meta['lower']
            ub[i:i + size] = meta['upper']
            i += size

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""
        # Todo:
        # - add check that # of constraints are consistent

        # Set the design variable values
        i = 0
        for name, meta in iteritems(self.problem.model.get_design_vars()):
            size = meta['size']
            self.problem[name] = x[i:i + size]
            i += size

        # Solve the problem
        self.problem.model._solve_nonlinear()

        # Extract the values of the objectives and constraints
        con = np.zeros(self.ncon)

        i = 0
        for name, meta in iteritems(self.problem.model.get_constraints()):
            size = meta['size']
            con[i:i + size] = self.problem[name]
            i += size

        # We only accept the first gradient
        for name, meta in iteritems(self.problem.model.get_objectives()):
            fobj = self.problem[name]
            break

        fail = 0

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""

        # The objective gradient
        for name, meta in iteritems(self.problem.model.get_objectives()):
            grad = self.problem.compute_totals(of=[name], wrt=self.dvs,
                                               return_format='array')
            g[:] = grad[0,:]
            break

        # Extract the constraint gradients
        i = 0
        for name, meta in iteritems(self.problem.model.get_constraints()):
            cgrad = self.problem.compute_totals(of=[name], wrt=self.dvs,
                                                return_format='array')
            for j in range(meta['size']):
                A[i + j][:] = cgrad[j,:]
            i += meta['size']

        fail = 0

        return fail
