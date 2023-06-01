import numpy as np
import json
import openmdao.api as om
from openmdao.core.driver import Driver
from paropt import ParOpt


class ParOptSparseProblem(ParOpt.Problem):
    def __init__(
        self,
        comm,
        nvars,
        num_sparse_constraints,
        num_sparse_inequalities,
        rowp,
        cols,
        driver,
    ):
        self.driver = driver

        super().__init__(
            comm,
            nvars=nvars,
            num_sparse_constraints=num_sparse_constraints,
            num_sparse_inequalities=num_sparse_inequalities,
            rowp=rowp,
            cols=cols,
        )

    def getVarsAndBounds(self, x, lb, ub):
        x0, lb0, ub0 = self.driver.get_paropt_vars_and_bounds()
        x[:] = x0[:]
        lb[:] = lb0[:]
        ub[:] = ub0[:]

        return

    def evalSparseObjCon(self, x, con_sparse):
        fail = self.driver.set_paropt_vars(x)
        fobj, con = self.driver.get_paropt_objcon()
        con_sparse[:] = con[:]
        return fail, fobj, []

    def evalSparseObjConGradient(self, x, g, A, data0):
        gobj, rowp, cols, data = self.driver.get_paropt_objcon_gradient()
        g[:] = gobj[:]
        data0[:] = data[:]

        fail = 0
        return fail


class ParOptTestDriver(Driver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # What we support
        self.supports["optimization"] = True
        self.supports["inequality_constraints"] = True
        self.supports["equality_constraints"] = True
        self.supports["simultaneous_derivatives"] = True
        self.supports["two_sided_constraints"] = True
        self.supports["total_jac_sparsity"] = True

        # What we don't support yet
        self.supports["active_set"] = False
        self.supports["integer_design_vars"] = False
        self.supports["distributed_design_vars"] = False
        self.supports["multiple_objectives"] = False
        self.supports["linear_constraints"] = True
        self.supports._read_only = True

        self._indep_list = []
        self._nonlinear_quantities = []
        self._linear_quantities = []
        self._linear_sens_dict = None

        # Set the Jacobian data to None
        self._jac_dict = None

        self._total_jac_format = "dict"

        # Discard constraints with upper bound larger than this
        self.constr_upper_limit = 1e20

        # Discard constraints with lower bound smaller than this
        self.constr_lower_limit = -1e20

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """

        self.options.declare(
            "check_sparse_jacobian",
            False,
            types=bool,
            desc="Check the partial derivatives",
        )

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
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        # Set up the sub-Jacobian sparsity pattern. This will be used to extract the non-zero entries.
        self._setup_tot_jac_sparsity()

    def _convert_jacobian_to_csr(self, sens_dict):
        """
        Convert the constraint Jacobian to a CSR data structure
        """

        # Store a consistent design variable list for each constraint. This
        # way the ordering will always be the same for subsequent calls.
        if self._jac_dict is None:
            self._jac_dict = {}
            for cname in self._con_list:
                dvs = []
                for name in sens_dict[cname].keys():
                    dvs.append(name)
                self._jac_dict[cname] = dvs

        # Convert to global COO ordering
        rows = []
        cols = []
        data = []

        for icon, cname in enumerate(self._con_list):
            coffset = self._con_list_ptr[icon]
            csize = self._con_list_ptr[icon + 1] - coffset
            ctype = self._con_type[icon]

            for dvname in self._jac_dict[cname]:
                dvoffset = self._indep_offset_dict[dvname]
                dvsize = self._indep_size_dict[dvname]

                # Do a conversion to COO
                if cname in self._res_subjacs and dvname in self._res_subjacs[cname]:
                    coo = self._res_subjacs[cname][dvname]
                    row, col, _ = coo["coo"]

                    arr = sens_dict[cname][dvname]
                    if ctype == "equals" or ctype == "lower":
                        data.extend(arr[row, col])
                    elif ctype == "upper":
                        data.extend(-arr[row, col])

                    # Add the offsets into the row and column
                    cols.extend(col + dvoffset)
                    rows.extend(row + coffset)
                else:
                    # This is a dense format
                    col = np.mod(np.arange(csize * dvsize), dvsize)
                    row = np.arange(csize * dvsize) // dvsize

                    arr = sens_dict[cname][dvname]
                    if ctype == "equals" or ctype == "lower":
                        data.extend(arr.flatten())
                    elif ctype == "upper":
                        data.extend(-arr.flatten())

                    # Add the offsets into the row and column
                    cols.extend(col + dvoffset)
                    rows.extend(row + coffset)

        # Number of constraints x number of variables
        ncon = self._con_list_ptr[-1]
        nvars = self._indep_list_ptr[-1]

        return self._convert_coo_to_csr(ncon, nvars, rows, cols, data)

    def _convert_coo_to_csr(self, nrows, ncols, row, col, coo):
        """
        Convert the COO format to CSR
        """
        nnz = len(row)
        rowp = np.zeros(nrows + 1, dtype=np.intc)
        cols = np.zeros(nnz, dtype=np.intc)
        data = np.zeros(nnz, dtype=ParOpt.dtype)
        for i in range(nnz):
            rowp[row[i] + 1] += 1
        for i in range(nrows):
            rowp[i + 1] += rowp[i]

        count = np.zeros(nrows, dtype=int)
        for i in range(nnz):
            index = rowp[row[i]] + count[row[i]]
            cols[index] = col[i]
            data[index] = coo[i]
            count[row[i]] += 1

        return rowp, cols, data

    def get_paropt_vars_and_bounds(self):
        size = self._indep_list_ptr[-1]
        x = np.zeros(size, dtype=ParOpt.dtype)
        lb = np.zeros(size, dtype=ParOpt.dtype)
        ub = np.zeros(size, dtype=ParOpt.dtype)

        values = self.get_design_var_values()

        for i, name in enumerate(self._indep_list):
            first = self._indep_list_ptr[i]
            last = self._indep_list_ptr[i + 1]
            x[first:last] = values[name]
            lb[first:last] = self._designvars[name]["lower"]
            ub[first:last] = self._designvars[name]["upper"]

        return x, lb, ub

    def set_paropt_vars(self, x):
        # Pass the updated design variables back to OpenMDAO
        for i, name in enumerate(self._indep_list):
            first = self._indep_list_ptr[i]
            last = self._indep_list_ptr[i + 1]
            self.set_design_var(name, np.array(x[first:last]), set_remote=False)

        model = self._problem().model
        model.run_solve_nonlinear()

        fail = 0
        return fail

    def get_paropt_objcon(self):
        fobj_dict = self.get_objective_values()
        con_dict = self.get_constraint_values()

        fobj = fobj_dict[self._obj_name]

        con = np.zeros(self._con_list_ptr[-1], dtype=ParOpt.dtype)
        for icon, name in enumerate(self._con_list):
            first = self._con_list_ptr[icon]
            last = self._con_list_ptr[icon + 1]
            if self._con_type[icon] == "equals":
                con[first:last] = con_dict[name] - self._cons[name]["equals"]
            elif self._con_type[icon] == "lower":
                con[first:last] = con_dict[name] - self._cons[name]["lower"]
            elif self._con_type[icon] == "upper":
                con[first:last] = self._cons[name]["upper"] - con_dict[name]

        return fobj, con

    def get_paropt_objcon_gradient(self):
        # Extract a dictionary of total derivatives for the nonlinear parts
        total_dict = self._compute_totals(
            of=self._nonlinear_quantities,
            wrt=self._indep_list,
            return_format=self._total_jac_format,
        )

        # Create a new dictionary - OpenMDAO saves "total_dict" and so modifying
        # it will cause an error.
        sens_dict = {}
        sens_dict.update(self._linear_sens_dict)
        sens_dict.update(total_dict)

        # Extract the
        gobj = np.zeros(self._indep_list_ptr[-1])
        if self._obj_name in sens_dict:
            for i, name in enumerate(self._indep_list):
                if name in sens_dict[self._obj_name]:
                    first = self._indep_list_ptr[i]
                    last = self._indep_list_ptr[i + 1]
                    gobj[first:last] = sens_dict[self._obj_name][name]

        # Extract the constraint Jacobian
        rowp, cols, data = self._convert_jacobian_to_csr(sens_dict)

        return gobj, rowp, cols, data

    def check_sparse_jacobian(self, x=None, dh=1e-6):
        if x is None:
            x, lb, ub = self.get_paropt_vars_and_bounds()

        self.set_paropt_vars(x)
        px = np.random.uniform(size=x.shape)

        fobj1, con1 = self.get_paropt_objcon()
        gobj, rowp, cols, data = self.get_paropt_objcon_gradient()

        self.set_paropt_vars(x + dh * px)
        fobj2, con2 = self.get_paropt_objcon()

        fd = (con2 - con1) / dh

        ans = np.zeros(len(rowp) - 1, dtype=ParOpt.dtype)
        for i in range(len(ans)):
            for jp in range(rowp[i], rowp[i + 1]):
                ans[i] += data[jp] * px[cols[jp]]

        for i in range(len(ans)):
            print(
                "ans[%3d] = %25.15e  fd[%3d] = %25.15e  err = %25.15e"
                % (i, ans[i], i, fd[i], (ans[i] - fd[i]) / ans[i])
            )
        return

    def run(self):
        problem = self._problem()

        # Add all objectives
        objs = self.get_objective_values()
        for name in objs:
            self._obj_name = name

            # The objective is always added to the nonlinear quantities because a
            # constraint can be linear, but an objective can't be???
            self._nonlinear_quantities.append(name)

        # Make a list of design variables. Capture the offset into the design variable list that
        # we'll use to inject the variables into the ParOpt design vector
        self._indep_list = list(self._designvars)
        self._indep_list_ptr = [0]
        self._indep_offset_dict = {}
        self._indep_size_dict = {}
        for i, name in enumerate(self._indep_list):
            index = self._indep_list_ptr[i] + self._designvars[name]["size"]
            self._indep_list_ptr.append(index)
            self._indep_offset_dict[name] = self._indep_list_ptr[i]
            self._indep_size_dict[name] = self._designvars[name]["size"]

        # Number of variables in the problem
        nvars = self._indep_list_ptr[-1]

        # Capture the constraints - look first for the inequality constraints that must be ordered first.
        # If they are two-sided constraints, we have to repeat them since ParOpt enforces c(x) >= 0.
        equalities = []
        inequalities = []

        num_equalities = 0
        num_inequalities = 0
        for name, meta in self._cons.items():
            # If current constraint is equality constraint
            if meta["equals"] is not None:
                equalities.append(name)
                num_equalities += meta["size"]
            # Else, current constraint is inequality constraint
            else:
                if (
                    meta["lower"] > self.constr_lower_limit
                    or meta["upper"] < self.constr_upper_limit
                ):
                    inequalities.append(name)
                if meta["lower"] > self.constr_lower_limit:
                    num_inequalities += meta["size"]
                if meta["upper"] < self.constr_upper_limit:
                    num_inequalities += meta["size"]

            if meta["linear"]:
                self._linear_quantities.append(name)
            else:
                self._nonlinear_quantities.append(name)

        self.num_equalities = num_equalities
        self.num_inequalities = num_inequalities
        self.num_constraints = num_inequalities + num_equalities

        self._con_list_ptr = [0]
        self._con_list = []
        self._con_type = []
        for name in inequalities:
            meta = self._cons[name]
            if meta["lower"] > self.constr_lower_limit:
                self._con_list.append(name)
                self._con_type.append("lower")
                self._con_list_ptr.append(self._con_list_ptr[-1] + meta["size"])
            if meta["upper"] < self.constr_upper_limit:
                self._con_list.append(name)
                self._con_type.append("upper")
                self._con_list_ptr.append(self._con_list_ptr[-1] + meta["size"])

        for name in equalities:
            meta = self._cons[name]
            self._con_list.append(name)
            self._con_type.append("equals")
            self._con_list_ptr.append(self._con_list_ptr[-1] + meta["size"])

        # Set the design variables - invoking run_solve_nonlinear
        model = self._problem().model
        model.run_solve_nonlinear()

        # Compute the total derivatives for the linear constraints
        if self._linear_sens_dict is None:
            if len(self._linear_quantities) > 0:
                self._linear_sens_dict = self._compute_totals(
                    of=self._linear_quantities,
                    wrt=self._indep_list,
                    return_format=self._total_jac_format,
                )
            else:
                self._linear_sens_dict = {}

        # Extract a dictionary of total derivatives for the nonlinear parts
        total_dict = self._compute_totals(
            of=self._nonlinear_quantities,
            wrt=self._indep_list,
            return_format=self._total_jac_format,
        )

        # Create a new dictionary - OpenMDAO saves "total_dict" and so modifying
        # it will cause an error.
        sens_dict = {}
        sens_dict.update(self._linear_sens_dict)
        sens_dict.update(total_dict)

        # Extract the constraint Jacobian
        rowp, cols, _ = self._convert_jacobian_to_csr(sens_dict)

        comm = problem.comm
        self.paropt_problem = ParOptSparseProblem(
            comm, nvars, self.num_constraints, self.num_inequalities, rowp, cols, self
        )

        if self.options["check_sparse_jacobian"]:
            self.check_sparse_jacobian()
            exit(0)

        # Take only the options declared from ParOpt
        info = ParOpt.getOptionsInfo()
        paropt_options = {}
        for key in self.options:
            if key in info.keys():
                paropt_options[key] = self.options[key]

        self.opt = ParOpt.Optimizer(self.paropt_problem, paropt_options)
        self.opt.optimize()

        return False

    def _setup_tot_jac_sparsity(self, coloring=None):
        """
        Set up total jacobian subjac sparsity.

        Parameters
        ----------
        coloring : Coloring or None
            Current coloring.
        """
        total_sparsity = None
        self._res_subjacs = {}
        coloring = coloring if coloring is not None else self._get_static_coloring()
        if coloring is not None:
            total_sparsity = coloring.get_subjac_sparsity()
            if self._total_jac_sparsity is not None:
                raise RuntimeError(
                    "Total jac sparsity was set in both _total_coloring"
                    " and _total_jac_sparsity."
                )
        elif self._total_jac_sparsity is not None:
            if isinstance(self._total_jac_sparsity, str):
                with open(self._total_jac_sparsity, "r") as f:
                    self._total_jac_sparsity = json.load(f)
            total_sparsity = self._total_jac_sparsity

        if total_sparsity is None:
            return

        for (
            res,
            dvdict,
        ) in total_sparsity.items():  # res are 'driver' names (prom name or alias)
            if res in self._objs:  # skip objectives
                continue
            self._res_subjacs[res] = {}
            for dv, (rows, cols, shape) in dvdict.items():  # dvs are src names
                rows = np.array(rows, dtype=np.intc)
                cols = np.array(cols, dtype=np.intc)

                self._res_subjacs[res][dv] = {
                    "coo": [rows, cols, np.zeros(rows.size)],
                    "shape": shape,
                }

        return
