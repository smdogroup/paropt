# Standard Python modules
import datetime
import os
import time

# External modules
import numpy as np

from paropt import ParOpt
from mpi4py import MPI

# Local modules
from pyoptsparse.pyOpt_optimizer import Optimizer
from pyoptsparse.pyOpt_utils import extractRows, INFINITY, IROW, ICOL


class ParOptSparseProblem(ParOpt.Problem):
    def __init__(
        self,
        comm,
        ptr,
        nvars,
        num_sparse_constraints,
        num_sparse_inequalities,
        rowp,
        cols,
        xs,
        blx,
        bux,
    ):
        self.ptr = ptr

        super().__init__(
            comm,
            nvars=nvars,
            num_sparse_constraints=num_sparse_constraints,
            num_sparse_inequalities=num_sparse_inequalities,
            rowp=rowp,
            cols=cols,
        )

        self.xs = xs
        self.blx = blx
        self.bux = bux
        self.fobj = 0.0
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        # Find the average distance between lower and upper bound
        bound_sum = 0.0
        for i in range(len(x)):
            if self.blx[i] <= -INFINITY or self.bux[i] >= INFINITY:
                bound_sum += 1.0
            else:
                bound_sum += self.bux[i] - self.blx[i]
        bound_sum = bound_sum / len(x)

        for i in range(len(x)):
            x[i] = self.xs[i]
            lb[i] = self.blx[i]
            ub[i] = self.bux[i]
            if self.xs[i] <= self.blx[i]:
                x[i] = self.blx[i] + 0.5 * np.min(
                    (bound_sum, self.bux[i] - self.blx[i])
                )
            elif self.xs[i] >= self.bux[i]:
                x[i] = self.bux[i] - 0.5 * np.min(
                    (bound_sum, self.bux[i] - self.blx[i])
                )

        return

    def evalSparseObjCon(self, x, con_sparse):
        """Evaluate the objective and constraint values"""
        fobj, fcon, fail = self.ptr._masterFunc(x[:], ["fobj", "fcon"])
        self.fobj = fobj
        con_sparse[:] = -fcon
        return fail, fobj, []

    def evalSparseObjConGradient(self, x, g, A, data):
        """Evaluate the objective and constraint gradients"""
        gobj, gcon, fail = self.ptr._masterFunc(x[:], ["gobj", "gcon"])
        g[:] = gobj[:]
        data[:] = -gcon[:]
        return fail


class ParOptDenseProblem(ParOpt.Problem):
    def __init__(self, comm, ptr, nvars, ncon, num_dense_inequalities, xs, blx, bux):
        super().__init__(
            comm,
            nvars=nvars,
            ncon=ncon,
            num_dense_inequalities=num_dense_inequalities,
        )
        self.ptr = ptr
        self.nvars = nvars
        self.ncon = ncon
        self.num_dense_inequalities = num_dense_inequalities
        self.xs = xs
        self.blx = blx
        self.bux = bux
        self.fobj = 0.0
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        # Find the average distance between lower and upper bound
        bound_sum = 0.0
        for i in range(len(x)):
            if self.blx[i] <= -INFINITY or self.bux[i] >= INFINITY:
                bound_sum += 1.0
            else:
                bound_sum += self.bux[i] - self.blx[i]
        bound_sum = bound_sum / len(x)

        for i in range(len(x)):
            x[i] = self.xs[i]
            lb[i] = self.blx[i]
            ub[i] = self.bux[i]
            if self.xs[i] <= self.blx[i]:
                x[i] = self.blx[i] + 0.5 * np.min(
                    (bound_sum, self.bux[i] - self.blx[i])
                )
            elif self.xs[i] >= self.bux[i]:
                x[i] = self.bux[i] - 0.5 * np.min(
                    (bound_sum, self.bux[i] - self.blx[i])
                )

        return

    def evalObjCon(self, x):
        """Evaluate the objective and constraint values"""
        fobj, fcon, fail = self.ptr._masterFunc(x[:], ["fobj", "fcon"])
        self.fobj = fobj
        return fail, fobj, -fcon

    def evalObjConGradient(self, x, g, A):
        """Evaluate the objective and constraint gradients"""
        gobj, gcon, fail = self.ptr._masterFunc(x[:], ["gobj", "gcon"])
        g[:] = gobj[:]
        gcon = np.atleast_2d(gcon)
        for i in range(self.ncon):
            A[i][:] = -gcon[i][:]
        return fail


class ParOptSparse(Optimizer):
    """
    ParOpt optimizer class

    ParOpt has the capability to handle distributed design vectors.
    This is not replicated here since pyOptSparse does not have the
    capability to handle this type of design problem.
    """

    def __init__(self, raiseError=True, options={}, sparse=True):
        name = "ParOpt"
        category = "Local Optimizer"
        self.sparse = sparse

        # Create and fill-in the dictionary of default option values
        self.defOpts = {}
        paropt_default_options = ParOpt.getOptionsInfo()

        # Manually override the options with missing default values
        paropt_default_options["ip_checkpoint_file"].default = "default.out"
        paropt_default_options["problem_name"].default = "problem"

        # Change the default algorithm to interior point if sparse since trust-region doesn't support sparse constraints
        if self.sparse:
            paropt_default_options["algorithm"].default = "ip"

        for option_name in paropt_default_options:
            # Get the type and default value of the named argument
            _type = None
            if paropt_default_options[option_name].option_type == "bool":
                _type = bool
            elif paropt_default_options[option_name].option_type == "int":
                _type = int
            elif paropt_default_options[option_name].option_type == "float":
                _type = float
            else:
                _type = str
            default_value = paropt_default_options[option_name].default

            # Set the entry into the dictionary
            self.defOpts[option_name] = [_type, default_value]

        self.set_options = {}
        self.informs = {}
        super().__init__(
            name,
            category,
            defaultOptions=self.defOpts,
            informs=self.informs,
            options=options,
        )

        # ParOpt can use a CSR or dense Jacobian format
        if self.sparse:
            self.jacType = "csr"
        else:
            self.jacType = "dense2d"

        return

    def __call__(
        self,
        optProb,
        sens=None,
        sensStep=None,
        sensMode=None,
        storeHistory=None,
        hotStart=None,
        storeSens=True,
    ):
        """
        This is the main routine used to solve the optimization
        problem.

        Parameters
        ----------
        optProb : Optimization or Solution class instance
            This is the complete description of the optimization problem
            to be solved by the optimizer

        sens : str or python Function.
            Specifiy method to compute sensitivities. To
            explictly use pyOptSparse gradient class to do the
            derivatives with finite differenes use \'FD\'. \'sens\'
            may also be \'CS\' which will cause pyOptSpare to compute
            the derivatives using the complex step method. Finally,
            \'sens\' may be a python function handle which is expected
            to compute the sensitivities directly. For expensive
            function evaluations and/or problems with large numbers of
            design variables this is the preferred method.

        sensStep : float
            Set the step size to use for design variables. Defaults to
            1e-6 when sens is \'FD\' and 1e-40j when sens is \'CS\'.

        sensMode : str
            Use \'pgc\' for parallel gradient computations. Only
            available with mpi4py and each objective evaluation is
            otherwise serial

        storeHistory : str
            File name of the history file into which the history of
            this optimization will be stored

        hotStart : str
            File name of the history file to "replay" for the
            optimziation.  The optimization problem used to generate
            the history file specified in \'hotStart\' must be
            **IDENTICAL** to the currently supplied \'optProb\'. By
            identical we mean, **EVERY SINGLE PARAMETER MUST BE
            IDENTICAL**. As soon as he requested evaluation point
            from ParOpt does not match the history, function and
            gradient evaluations revert back to normal evaluations.

        storeSens : bool
            Flag sepcifying if sensitivities are to be stored in hist.
            This is necessay for hot-starting only.
        """
        # Raise an error if the user is trying to solve a sparse problem with the trust region algorithm
        if self.sparse and self.set_options["algorithm"].lower() == "tr":
            raise ValueError(
                "Trust region algorithm does not support sparse constraints, please use the interior point or MMA algorithms instead."
            )

        self.startTime = time.time()
        self.callCounter = 0
        self.storeSens = storeSens

        if len(optProb.constraints) == 0:
            # If the problem is unconstrained, add a dummy constraint.
            self.unconstrained = True
            optProb.dummyConstraint = True

        # Save the optimization problem and finalize constraint
        # Jacobian, in general can only do on root proc
        self.optProb = optProb
        self.optProb.finalize()

        # Set history/hotstart
        self._setHistory(storeHistory, hotStart)
        self._setInitialCacheValues()
        self._setSens(sens, sensStep, sensMode)
        blx, bux, xs = self._assembleContinuousVariables()
        xs = np.maximum(xs, blx)
        xs = np.minimum(xs, bux)

        # The number of design variables
        nvars = len(xs)

        oneSided = True

        if self.unconstrained:
            # Data for the single dummy constraint
            ncon = 1
            indices = [0]
            ninequalities = 1
        else:
            # Count the number of inequalities
            indices, _, _, _ = self.optProb.getOrdering(["ni", "li"], oneSided=oneSided)
            ninequalities = len(indices)

            # Now order the constraints first with the inequalities
            indices, blc, buc, fact = self.optProb.getOrdering(
                ["ni", "li", "ne", "le"], oneSided=oneSided
            )

            # Set the properties of the constraint Jacobian
            ncon = len(indices)
            self.optProb.jacIndices = indices
            self.optProb.fact = fact
            self.optProb.offset = buc

        if self.optProb.comm.rank == 0:
            optTime = MPI.Wtime()

            if self.sparse:
                # Build the sparsity pattern for the Jacobian
                gcon = {}
                for iCon in self.optProb.constraints:
                    gcon[iCon] = self.optProb.constraints[iCon].jac
                jac = self.optProb.processConstraintJacobian(gcon)
                jac = extractRows(jac, indices)

                # Extract the non-zero CSR pattern
                rowp = jac["csr"][IROW]
                cols = jac["csr"][ICOL]

                # Optimize the problem
                problem = ParOptSparseProblem(
                    MPI.COMM_SELF,
                    self,
                    nvars,
                    ncon,
                    ninequalities,
                    rowp,
                    cols,
                    xs,
                    blx,
                    bux,
                )
            else:
                problem = ParOptDenseProblem(
                    MPI.COMM_SELF,
                    self,
                    nvars,
                    ncon,
                    ninequalities,
                    xs,
                    blx,
                    bux,
                )

            if self.set_options["gradient_verification_frequency"] > 0:
                problem.checkGradients(1e-6)

            optimizer = ParOpt.Optimizer(problem, self.set_options)
            optimizer.optimize()
            x, z, zw, zl, zu = optimizer.getOptimizedPoint()

            # Set the total opt time
            optTime = MPI.Wtime() - optTime

            # Get the obective function value
            fobj = problem.fobj

            if self.storeHistory:
                self.metadata["endTime"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                self.metadata["optTime"] = optTime
                self.hist.writeData("metadata", self.metadata)
                self.hist.close()

            # Create the optimization solution. Note that the signs on the multipliers
            # are switch since ParOpt uses a formulation with c(x) >= 0, while pyOpt
            # uses g(x) = -c(x) <= 0. Therefore the multipliers are reversed.
            sol_inform = {}
            sol_inform["value"] = ""
            sol_inform["text"] = ""

            # If number of constraints is zero, ParOpt returns z as None.
            # Thus if there is no constraints, should pass an empty list
            # to multipliers instead of z.
            if self.sparse:
                if zw is not None:
                    sol = self._createSolution(
                        optTime, sol_inform, fobj, x[:], multipliers=-zw[:]
                    )
                else:
                    sol = self._createSolution(
                        optTime, sol_inform, fobj, x[:], multipliers=[]
                    )
            else:
                if z is not None:
                    sol = self._createSolution(
                        optTime, sol_inform, fobj, x[:], multipliers=-z[:]
                    )
                else:
                    sol = self._createSolution(
                        optTime, sol_inform, fobj, x[:], multipliers=[]
                    )

            # Indicate solution finished
            self.optProb.comm.bcast(-1, root=0)
        else:  # We are not on the root process so go into waiting loop:
            self._waitLoop()
            sol = None

        # Communication solution and return
        sol = self._communicateSolution(sol)

        return sol

    def _on_setOption(self, name, value):
        """
        Add the value to the set_options dictionary.
        """
        self.set_options[name] = value
