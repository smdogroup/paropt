"""
OpenMDAO Wrapper for ParOpt
"""

from __future__ import print_function
import numpy as np
import mpi4py.MPI as MPI

from paropt import ParOpt
import openmdao
from openmdao.core.driver import Driver


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
        Result returned from optimize call.
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
        self.paropt_use_qn_correction = False

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
            if info[name].option_type == "bool":
                self.options.declare(name, default, types=bool, desc=descript)
            elif info[name].option_type == "int":
                self.options.declare(
                    name,
                    default,
                    types=int,
                    lower=values[0],
                    upper=values[1],
                    desc=descript,
                )
            elif info[name].option_type == "float":
                self.options.declare(
                    name,
                    default,
                    types=float,
                    lower=values[0],
                    upper=values[1],
                    desc=descript,
                )
            elif info[name].option_type == "str":
                if default is None:
                    self.options.declare(
                        name, default, types=str, allow_none=True, desc=descript
                    )
                else:
                    self.options.declare(name, default, types=str, desc=descript)
            elif info[name].option_type == "enum":
                self.options.declare(name, default, values=values, desc=descript)

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

        super(ParOptDriver, self)._setup_driver(problem)

        # Raise error if multiple objectives are provided
        if len(self._objs) > 1:
            msg = "ParOpt currently does not support multiple objectives."
            raise RuntimeError(msg.format(self.__class__.__name__))

        # Create the ParOptProblem from the OpenMDAO problem
        self.paropt_problem = ParOptProblem(problem)

        # We may bind the external method for quasi-newton update correction
        # if specified
        if self.paropt_use_qn_correction:
            self.paropt_problem.computeQuasiNewtonUpdateCorrection = (
                self.computeQuasiNewtonUpdateCorrection.__get__(self.paropt_problem)
            )

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

    def use_qn_correction(self, method):
        """
        Bind an external function which handles the quasi-newton update
        correction to the paropt problem instance
        """

        self.paropt_use_qn_correction = True
        self.computeQuasiNewtonUpdateCorrection = method
        return


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
        self.nineq = None
        self.constr_upper_limit = (
            1e20  # Discard constraints with upper bound larger than this
        )
        self.constr_lower_limit = (
            -1e20
        )  # Discard constraints with lower bound smaller than this

        # Get the design variable, objective and constraint objects from OpenMDAO
        self.om_dvs = self.problem.model.get_design_vars()
        self.om_con = self.problem.model.get_constraints()
        self.om_obj = self.problem.model.get_objectives()

        # Get the number of design vars from the openmdao problem
        self.nvars = 0
        for name, meta in self.om_dvs.items():
            size = meta["size"]
            self.nvars += size

        # Get the number of constraints from the openmdao problem
        self.ncon = 0
        self.nineq = 0
        for name, meta in self.om_con.items():
            # If current constraint is equality constraint
            if meta["equals"] is not None:
                self.ncon += meta["size"]
            # Else, current constraint is inequality constraint
            else:
                if meta["lower"] > self.constr_lower_limit:
                    self.ncon += meta["size"]
                    self.nineq += meta["size"]
                if meta["upper"] < self.constr_upper_limit:
                    self.ncon += meta["size"]
                    self.nineq += meta["size"]

        # Initialize the base class
        super(ParOptProblem, self).__init__(
            self.comm, self.nvars, self.ncon, self.nineq
        )

        return

    def getVarsAndBounds(self, x, lb, ub):
        """Set the values of the bounds"""

        # Get design vars from openmdao as a dictionary
        desvar_vals = self.problem.driver.get_design_var_values()

        i = 0
        for name, meta in self.om_dvs.items():
            size = meta["size"]
            x[i : i + size] = desvar_vals[name]
            lb[i : i + size] = meta["lower"]
            ub[i : i + size] = meta["upper"]
            i += size

        # Check if number of design variables consistent
        if i != self.nvars:
            raise ValueError(
                "Number of design variables get (%d) is not equal to the"
                "number of design variables during initialzation (%d)" % (i, self.nvars)
            )

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint"""

        # Pass the updated design variables back to OpenMDAO
        i = 0
        for name, meta in self.om_dvs.items():
            size = meta["size"]
            self.problem.driver.set_design_var(name, x[i : i + size], set_remote=False)
            i += size

        # Solve the problem
        self.problem.model._solve_nonlinear()

        # Extract the values of the objectives and constraints
        con = np.zeros(self.ncon)
        constr_vals = (
            self.problem.driver.get_constraint_values()
        )  # Returns the global array

        i = 0
        # First we extract all inequality constraints
        for name, meta in self.om_con.items():
            if meta["equals"] is None:
                size = meta["size"]
                om_con_vals = np.zeros(size)
                om_con_vals[:] = constr_vals[name][0:size]  # Get the local values
                if meta["lower"] > self.constr_lower_limit:
                    con[i : i + size] = om_con_vals - meta["lower"]
                    i += size
                if meta["upper"] < self.constr_upper_limit:
                    con[i : i + size] = meta["upper"] - om_con_vals
                    i += size

        # Then, extract rest of the equality constraints:
        for name, meta in self.om_con.items():
            if meta["equals"] is not None:
                size = meta["size"]
                om_con_vals = np.zeros(size)
                om_con_vals[:] = constr_vals[name][0:size]  # Get the local values
                con[i : i + size] = om_con_vals - meta["equals"]
                i += size

        # We only accept the first objective
        obj_vals = self.problem.driver.get_objective_values()
        for name, meta in self.om_obj.items():
            if isinstance(obj_vals[name], float):
                fobj = obj_vals[name]
            else:
                # In parallel, obj_vals will be array w/ len=nprocs,
                # so just take the first value
                fobj = obj_vals[name][0]
            break

        fail = 0

        return fail, fobj, con

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradient"""

        # Extract gradients of objective and constraints w.r.t. all design variables
        objcon_grads = self.problem.compute_totals(get_remote=False)

        # Extract the objective gradient
        for name, meta in self.om_obj.items():
            i_dv = 0
            for dv_name in self.om_dvs:
                dv_subsize = self.om_dvs[dv_name]["size"]
                g[i_dv : i_dv + dv_subsize] = objcon_grads[(name, dv_name)][0]
                i_dv += dv_subsize
            break

        # Extract the constraint gradients
        # We first extract gradients of inequality constraints
        i = 0
        for name, meta in self.om_con.items():
            if meta["equals"] is None:
                if meta["lower"] > self.constr_lower_limit:
                    for j in range(meta["size"]):
                        i_dv = 0
                        for dv_name in self.om_dvs:
                            dv_subsize = self.om_dvs[dv_name]["size"]
                            A[i + j][i_dv : i_dv + dv_subsize] = objcon_grads[
                                (name, dv_name)
                            ][j]
                            i_dv += dv_subsize
                    i += meta["size"]
                if meta["upper"] < self.constr_upper_limit:
                    for j in range(meta["size"]):
                        i_dv = 0
                        for dv_name in self.om_dvs:
                            dv_subsize = self.om_dvs[dv_name]["size"]
                            A[i + j][i_dv : i_dv + dv_subsize] = -objcon_grads[
                                (name, dv_name)
                            ][j]
                            i_dv += dv_subsize
                    i += meta["size"]

        # Then, extract equality constraint gradients
        for name, meta in self.om_con.items():
            if meta["equals"] is not None:
                for j in range(meta["size"]):
                    i_dv = 0
                    for dv_name in self.om_dvs:
                        dv_subsize = self.om_dvs[dv_name]["size"]
                        A[i + j][i_dv : i_dv + dv_subsize] = objcon_grads[
                            (name, dv_name)
                        ][j]
                        i_dv += dv_subsize
                i += meta["size"]

        fail = 0

        return fail
