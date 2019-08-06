Rosenbrock example
==================

In this example we consider a python implementation of the following optimization problem:

.. math::

    \begin{align}
        \text{min} \qquad & 100(x_2 - x_1^2)^2 + (1 - x_1)^2 \\
        \text{with respect to} \qquad & -2 \le x_{i} \le 2 \qquad i = 1, 2\\
        \text{subject to} \qquad & x_{1} + x_{2} + 5 \ge 0 \\
    \end{align}

Python implementation
---------------------

The python implementation of this problem is as follows

.. code-block:: python

  # Import some utilities
  import numpy as np
  import mpi4py.MPI as MPI
  import matplotlib.pyplot as plt

  # Import ParOpt
  from paropt import ParOpt

  # Create the rosenbrock function class
  class Rosenbrock(ParOpt.Problem):
      def __init__(self):
          # Set the communicator pointer
          self.comm = MPI.COMM_WORLD
          self.nvars = 2
          self.ncon = 1

          # The design history file
          self.x_hist = []

          # Initialize the base class
          super(Rosenbrock, self).__init__(self.comm, self.nvars, self.ncon)

          return

      def getVarsAndBounds(self, x, lb, ub):
          '''Set the values of the bounds'''
          x[:] = -1.0
          lb[:] = -2.0
          ub[:] = 2.0
          return

      def evalObjCon(self, x):
          '''Evaluate the objective and constraint'''
          # Append the point to the solution history
          self.x_hist.append(np.array(x))

          # Evaluate the objective and constraints
          fail = 0
          con = np.zeros(1)
          fobj = 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
          con[0] = x[0] + x[1] + 5.0
          return fail, fobj, con

      def evalObjConGradient(self, x, g, A):
          '''Evaluate the objective and constraint gradient'''
          fail = 0
          
          # The objective gradient
          g[0] = 200*(x[1]-x[0]**2)*(-2*x[0]) - 2*(1-x[0])
          g[1] = 200*(x[1]-x[0]**2)

          # The constraint gradient
          A[0][0] = 1.0
          A[0][1] = 1.0
          return fail

  problem = Rosenbrock()
  max_lbfgs = 20
  opt = ParOpt.InteriorPoint(problem, max_lbfgs, ParOpt.BFGS)
  opt.optimize()

This code produces the output:

::

  ParOpt: Parameter values
  total variables                              2
  constraints                                  1
  max_qn_size                                 20
  norm_type                           INFTY_NORM
  penalty_gamma                             1000
  max_major_iters                           1000
  starting_point_strategy          LEAST_SQUARES
  barrier_param                              0.1
  abs_res_tol                              1e-05
  rel_func_tol                                 0
  use_line_search                              1
  use_backtracking_alpha                       0
  max_line_iters                              10
  penalty_descent_fraction                   0.3
  min_rho_penalty_search                       0
  armijo_constant                          1e-05
  monotone_barrier_fraction                 0.25
  monotone_barrier_power                     1.1
  rel_bound_barrier                            1
  min_fraction_to_boundary                  0.95
  major_iter_step_check                       -1
  write_output_frequency                      10
  gradient_check_frequency                    -1
  gradient_check_step                      1e-06
  sequential_linear_method                     0
  hessian_reset_freq                   100000000
  use_quasi_newton_update                      1
  qn_sigma                                     0
  use_hvec_product                             0
  use_diag_hessian                             0
  use_qn_gmres_precon                          1
  nk_switch_tol                            0.001
  eisenstat_walker_alpha                     1.5
  eisenstat_walker_gamma                       1
  gmres_subspace_size                          0
  max_gmres_rtol                             0.1
  gmres_atol                               1e-30

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    0    1    1    0      --      --      --  4.04000e+02 1.0e+03 3.0e+00 2.9e+00 1.0e-01 1.7e+00       --      -- 
    1    2    2    0 1.0e+00 1.1e-03 2.3e-03  1.01561e+02 1.0e+03 3.0e+00 3.2e+00 1.0e-01 1.4e+00 -1.4e+03 0.0e+00 
    2    3    3    0 1.0e+00 2.8e-03 2.8e-01  1.01292e+02 7.2e+02 3.0e+00 5.7e+00 1.0e-01 2.0e+00 -4.8e+01 0.0e+00 
    3    4    4    0 1.0e+00 2.6e-01 1.7e-01  7.03966e+01 6.0e+02 2.2e+00 2.0e+00 1.0e-01 8.7e-01 -3.7e+01 0.0e+00 
    4    5    5    0 1.0e+00 1.0e+00 3.8e-01  3.81744e+00 3.7e+02 4.9e-17 1.2e+00 1.0e-01 4.9e-01 -1.2e+02 0.0e+00 
    5    6    6    0 1.0e+00 1.0e+00 1.0e+00  1.66498e+00 3.9e+00 4.2e-16 5.7e-02 1.0e-01 1.1e-01 -4.0e+00 0.0e+00 
    6    7    7    0 1.0e+00 1.0e+00 1.0e+00  1.63609e+00 2.4e+00 2.0e-16 3.0e-04 1.0e-01 1.0e-01 -5.2e-02 0.0e+00 
    7    8    8    0 1.0e+00 1.0e+00 1.0e+00  1.62224e+00 2.0e+00 4.1e-17 1.5e-06 1.0e-01 1.0e-01 -1.6e-02 0.0e+00 
    8    9    9    0 1.0e+00 1.0e+00 1.0e+00  1.49793e+00 3.5e+00 5.6e-16 1.2e-04 1.0e-01 1.0e-01 -1.4e-01 0.0e+00 
    9   11   10    0 1.0e-02 1.0e+00 1.0e+00  1.46101e+00 3.6e+00 5.0e-16 1.3e-04 1.0e-01 1.0e-01 -3.7e+00 0.0e+00 skipH 

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    10   13   11    0 2.6e-02 1.0e+00 1.0e+00  1.41460e+00 2.5e+00 3.8e-16 1.3e-04 1.0e-01 1.0e-01 -3.1e+00 0.0e+00 
    11   14   12    0 1.0e+00 1.0e+00 1.0e+00  1.36112e+00 2.7e+00 7.2e-17 1.5e-05 1.0e-01 1.0e-01 -5.6e-02 0.0e+00 
    12   16   13    0 5.1e-02 1.0e+00 1.0e+00  1.29641e+00 2.7e+00 5.7e-17 4.0e-05 1.0e-01 1.0e-01 -1.3e+00 0.0e+00 
    13   18   14    0 7.9e-02 1.0e+00 1.0e+00  1.21616e+00 2.6e+00 3.9e-16 7.8e-05 1.0e-01 1.0e-01 -1.1e+00 0.0e+00 
    14   20   15    0 1.5e-01 1.0e+00 1.0e+00  1.11513e+00 3.1e+00 9.8e-17 1.5e-04 1.0e-01 1.0e-01 -7.6e-01 0.0e+00 
    15   22   16    0 3.5e-01 1.0e+00 1.0e+00  1.00562e+00 5.3e+00 6.9e-16 2.5e-04 1.0e-01 1.0e-01 -4.3e-01 0.0e+00 
    16   23   17    0 1.0e+00 1.0e+00 1.0e+00  8.97353e-01 7.3e+00 3.4e-16 2.3e-04 1.0e-01 1.0e-01 -1.7e-01 0.0e+00 
    17   24   18    0 1.0e+00 1.0e+00 1.0e+00  7.14752e-01 4.2e+00 4.1e-17 9.6e-05 1.0e-01 1.0e-01 -2.3e-01 0.0e+00 
    18   25   19    0 1.0e+00 1.0e+00 1.0e+00  5.43310e-01 4.3e+00 6.0e-17 4.0e-04 1.0e-01 1.0e-01 -2.4e-01 0.0e+00 
    19   26   20    0 1.0e+00 1.0e+00 1.0e+00  3.85438e-01 2.5e+00 4.0e-16 3.0e-04 1.0e-01 1.0e-01 -2.9e-01 0.0e+00 

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    20   28   21    0 7.7e-02 1.0e+00 1.0e+00  3.44610e-01 1.8e+00 1.4e-16 3.1e-04 1.0e-01 1.0e-01 -5.5e-01 0.0e+00 
    21   30   22    0 2.1e-02 1.0e+00 1.0e+00  3.29234e-01 1.6e+00 4.0e-16 3.1e-04 1.0e-01 1.0e-01 -7.6e-01 0.0e+00 
    22   32   23    0 5.3e-02 1.0e+00 1.0e+00  3.03766e-01 1.1e+00 3.4e-16 3.1e-04 1.0e-01 1.0e-01 -5.0e-01 0.0e+00 
    23   34   24    0 1.0e-01 1.0e+00 1.0e+00  2.72543e-01 8.9e-01 3.8e-16 7.5e-02 2.5e-02 1.0e-01 -3.3e-01 0.0e+00 
    24   36   25    0 3.9e-01 1.0e+00 1.0e+00  2.36299e-01 3.2e+00 2.9e-16 4.7e-02 2.5e-02 7.1e-02 -2.0e-01 0.0e+00 
    25   37   26    0 1.0e+00 1.0e+00 1.0e+00  2.21830e-01 3.5e+00 2.9e-18 7.1e-04 2.5e-02 2.5e-02 -4.7e-02 0.0e+00 
    26   38   27    0 1.0e+00 1.0e+00 1.0e+00  1.60158e-01 3.6e+00 8.6e-17 3.9e-05 2.5e-02 2.5e-02 -8.6e-02 0.0e+00 
    27   39   28    0 1.0e+00 1.0e+00 1.0e+00  1.00878e-01 1.4e+00 8.0e-16 6.4e-05 2.5e-02 2.5e-02 -8.1e-02 0.0e+00 
    28   40   29    0 1.0e+00 1.0e+00 1.0e+00  7.23458e-02 4.5e+00 1.1e-16 1.8e-04 2.5e-02 2.5e-02 -5.6e-02 0.0e+00 
    29   41   30    0 1.0e+00 1.0e+00 1.0e+00  3.66023e-02 1.5e-01 5.6e-16 1.9e-02 6.3e-03 2.5e-02 -6.1e-02 0.0e+00 

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    30   42   31    0 1.0e+00 1.0e+00 1.0e+00  2.07562e-02 3.1e+00 3.3e-16 1.6e-03 6.3e-03 6.3e-03 -4.4e-02 0.0e+00 
    31   43   32    0 1.0e+00 1.0e+00 1.0e+00  1.21674e-02 7.5e-01 3.9e-16 1.5e-05 6.3e-03 6.3e-03 -1.3e-02 0.0e+00 
    32   44   33    0 1.0e+00 1.0e+00 1.0e+00  4.01074e-03 8.8e-01 2.8e-16 3.4e-05 6.3e-03 6.2e-03 -1.2e-02 0.0e+00 
    33   45   34    0 1.0e+00 1.0e+00 1.0e+00  1.24559e-03 5.8e-01 1.2e-15 1.5e-05 6.3e-03 6.2e-03 -3.7e-03 0.0e+00 
    34   46   35    0 1.0e+00 1.0e+00 1.0e+00  3.12305e-04 1.2e-01 5.0e-16 4.9e-06 6.3e-03 6.2e-03 -1.4e-03 0.0e+00 
    35   47   36    0 1.0e+00 1.0e+00 1.0e+00  1.25404e-04 4.5e-01 4.2e-16 6.9e-06 6.3e-03 6.2e-03 -5.1e-04 0.0e+00 
    36   48   37    0 1.0e+00 1.0e+00 1.0e+00  4.48419e-05 4.1e-03 2.8e-16 4.7e-03 1.6e-03 6.2e-03 -3.3e-04 0.0e+00 
    37   49   38    0 1.0e+00 1.0e+00 1.0e+00  7.00269e-06 1.5e-02 1.8e-15 1.2e-03 3.9e-04 1.6e-03 -3.6e-03 0.0e+00 
    38   50   39    0 1.0e+00 1.0e+00 1.0e+00  1.69068e-07 8.0e-03 6.4e-16 5.4e-06 3.9e-04 3.9e-04 -8.9e-04 0.0e+00 
    39   51   40    0 1.0e+00 1.0e+00 1.0e+00  9.65441e-08 5.6e-04 1.2e-15 2.9e-04 9.8e-05 3.9e-04 -8.0e-08 0.0e+00 

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    40   52   41    0 1.0e+00 1.0e+00 1.0e+00  6.07846e-09 1.9e-05 1.3e-15 7.3e-05 2.4e-05 9.8e-05 -2.2e-04 0.0e+00 
    41   53   42    0 1.0e+00 1.0e+00 1.0e+00  3.69844e-10 6.8e-06 6.4e-17 1.8e-05 6.1e-06 2.4e-05 -5.5e-05 0.0e+00 
    42   54   43    0 1.0e+00 1.0e+00 1.0e+00  2.30420e-11 4.2e-07 1.2e-15 4.6e-06 1.5e-06 6.1e-06 -1.4e-05 0.0e+00 
    43   55   44    0 1.0e+00 1.0e+00 1.0e+00  1.43924e-12 2.7e-08 7.5e-17 5.3e-07 1.0e-06 1.5e-06 -3.4e-06 0.0e+00 

The output from ParOptInteriorPoint consists of a summary of the parameter values.
A brief description of each parameter is provided if the file is not stdout.
Next, the output consists of a summary of the iteration history with the following columns:

* ``iter`` the current iteration
* ``nobj`` the number of objective and constraint evaluations
* ``ngrd`` the number of objective and constraint gradient evaluations
* ``nhvc`` the number of Hessian-vector products
* ``alpha`` the step size
* ``alphax`` the scaling factor applied to the primal components of the step
* ``alphaz`` the scaling factor applied to the dual components of the step
* ``fobj`` the objective value
* ``|opt|`` the optimality error in the specified norm
* ``|infeas|`` the infeasibility in the specified norm
* ``|dual|`` the error in the complementarity equation
* ``mu`` the barrier parameter
* ``comp`` the complementarity
* ``dmerit`` the derivative of the merit function
* ``rho`` the value of the penalty parameter in the line search
* ``info`` additional information, usually about the quasi-Newton Hessian update or line search

Note that this output can be directed to a file by adding the following line of code:

.. code-block:: python

  opt.setOutputFile('paropt.out')

The file can be visualized using the example in ``examples/plot_history/plot_history.py``.
This output can be unpacked from the file using:

.. code-block:: python
  
  values = paropt.unpack_output('paropt.out')