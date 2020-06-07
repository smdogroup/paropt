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

  options = {'algorithm': 'ip'}
  opt = ParOpt.Optimizer(problem, options)
  opt.optimize()

This code produces the following output in the default file "paropt.out":

::

  ParOpt Parameter Summary:
  abs_res_tol                              1e-06
  abs_step_tol                             0
  algorithm                                ip
  armijo_constant                          1e-05
  barrier_strategy                         monotone
  design_precision                         1e-14
  eisenstat_walker_alpha                   1.5
  eisenstat_walker_gamma                   1
  function_precision                       1e-10
  gmres_atol                               1e-30
  gmres_subspace_size                      0
  gradient_check_step_length               1e-06
  gradient_verification_frequency          -1
  hessian_reset_freq                       1000000
  init_barrier_param                       0.1
  init_rho_penalty_search                  0
  ip_checkpoint_file                       (null)
  max_bound_value                          1e+20
  max_gmres_rtol                           0.1
  max_line_iters                           10
  max_major_iters                          5000
  min_fraction_to_boundary                 0.95
  min_rho_penalty_search                   0
  mma_asymptote_contract                   0.7
  mma_asymptote_relax                      1.2
  mma_bound_relax                          0
  mma_delta_regularization                 1e-05
  mma_eps_regularization                   0.001
  mma_infeas_tol                           1e-05
  mma_init_asymptote_offset                0.25
  mma_l1_tol                               1e-06
  mma_linfty_tol                           1e-06
  mma_max_asymptote_offset                 10
  mma_max_iterations                       200
  mma_min_asymptote_offset                 0.01
  mma_output_file                          paropt.mma
  mma_use_constraint_linearization         1
  monotone_barrier_fraction                0.25
  monotone_barrier_power                   1.1
  nk_switch_tol                            0.001
  norm_type                                infinity
  output_file                              paropt.out
  output_level                             0
  penalty_descent_fraction                 0.3
  penalty_gamma                            1000
  problem_name                             (null)
  qn_sigma                                 0
  qn_subspace_size                         10
  qn_type                                  bfgs
  rel_bound_barrier                        1
  rel_func_tol                             0
  sequential_linear_method                 0
  start_affine_multiplier_min              0.001
  starting_point_strategy                  affine_step
  tr_adaptive_constraint                   linear_constraint
  tr_adaptive_gamma_update                 1
  tr_adaptive_objective                    linear_objective
  tr_bound_relax                           0.0001
  tr_eta                                   0.25
  tr_infeas_tol                            1e-05
  tr_init_size                             0.1
  tr_l1_tol                                1e-06
  tr_linfty_tol                            1e-06
  tr_max_iterations                        200
  tr_max_size                              1
  tr_min_size                              0.001
  tr_output_file                           paropt.tr
  tr_penalty_gamma_max                     10000
  tr_penalty_gamma_min                     0
  tr_write_output_frequency                10
  use_backtracking_alpha                   0
  use_diag_hessian                         0
  use_hvec_product                         0
  use_line_search                          1
  use_qn_gmres_precon                      1
  use_quasi_newton_update                  1
  write_output_frequency                   10

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    0    1    1    0      --      --      --  4.04000e+02 1.3e+03 6.7e+02 5.7e+05 1.2e+05 1.2e+05       --      --
    1    3    2    0 3.4e-01 9.6e-03 9.6e-01  5.93594e+00 2.3e+02 6.6e+02 6.5e+05 3.0e+04 1.4e+05 -3.7e+05 0.0e+00
    2    4    3    0 1.0e+00 1.0e+00 1.0e+00  1.10539e+00 5.0e+02 1.1e-13 6.8e+04 7.5e+03 3.7e+04 -6.9e+05 0.0e+00
    3    5    4    0 1.0e+00 1.0e+00 1.0e+00  1.11595e+00 3.5e+00 2.8e-14 8.0e+03 1.9e+03 7.5e+03 -7.4e+04 0.0e+00
    4    6    5    0 1.0e+00 1.0e+00 1.0e+00  1.58212e+00 3.6e+00 8.9e-16 1.6e+03 4.7e+02 1.9e+03 -8.6e+03 0.0e+00
    5    7    6    0 1.0e+00 1.0e+00 1.0e+00  2.46206e+00 5.5e+00 8.9e-16 6.1e+02 1.2e+02 4.7e+02 -1.6e+03 0.0e+00 skipH
    6    8    7    0 1.0e+00 1.0e+00 1.0e+00  7.63309e+00 2.0e+02 8.0e-16 1.7e+02 2.9e+01 1.1e+02 -5.1e+01 0.0e+00
    7    9    8    0 1.0e+00 1.0e+00 9.2e-01  4.56887e-01 1.9e+01 1.4e-16 3.1e+01 7.3e+00 3.0e+01 -7.8e+01 0.0e+00
    8   10    9    0 1.0e+00 1.0e+00 1.0e+00  3.03942e-01 2.6e+00 7.6e-16 5.7e+00 1.8e+00 7.3e+00 -1.6e+01 0.0e+00
    9   11   10    0 1.0e+00 1.0e+00 1.0e+00  2.76404e-01 6.2e-01 1.6e-15 1.4e+00 4.6e-01 1.8e+00 -4.1e+00 0.0e+00

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    10   12   11    0 1.0e+00 1.0e+00 1.0e+00  2.65533e-01 1.2e+00 2.7e-16 3.6e-01 1.1e-01 4.6e-01 -1.0e+00 0.0e+00
    11   13   12    0 1.0e+00 1.0e+00 1.0e+00  2.32897e-01 4.0e+00 3.9e-16 1.6e-02 1.1e-01 1.1e-01 -3.2e-01 0.0e+00
    12   14   13    0 1.0e+00 1.0e+00 1.0e+00  2.06177e-01 2.8e+00 5.8e-16 8.5e-05 1.1e-01 1.1e-01 -3.1e-02 0.0e+00
    13   15   14    0 1.0e+00 1.0e+00 1.0e+00  1.23235e-01 2.0e+00 1.9e-16 5.8e-04 1.1e-01 1.1e-01 -1.2e-01 0.0e+00
    14   16   15    0 1.0e+00 1.0e+00 1.0e+00  8.05909e-02 1.5e+00 5.6e-16 4.0e-04 1.1e-01 1.1e-01 -5.3e-02 0.0e+00
    15   17   16    0 1.0e+00 1.0e+00 1.0e+00  4.49292e-02 5.5e-01 5.7e-16 8.6e-02 2.9e-02 1.1e-01 -4.4e-02 0.0e+00
    16   18   17    0 1.0e+00 1.0e+00 1.0e+00  7.36520e-02 8.8e+00 4.8e-16 8.2e-03 2.9e-02 2.9e-02 -1.0e-01 0.0e+00
    17   19   18    0 1.0e+00 1.0e+00 1.0e+00  2.03940e-02 2.1e+00 1.1e-15 2.0e-04 2.9e-02 2.9e-02 -1.5e-01 0.0e+00
    18   20   19    0 1.0e+00 1.0e+00 1.0e+00  1.01586e-02 1.1e-01 3.4e-16 2.1e-02 7.2e-03 2.9e-02 -1.2e-02 0.0e+00
    19   21   20    0 1.0e+00 1.0e+00 1.0e+00  1.13286e-02 3.8e+00 5.3e-16 1.8e-03 7.2e-03 7.3e-03 -2.9e-02 0.0e+00

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    20   22   21    0 1.0e+00 1.0e+00 1.0e+00  4.92903e-03 3.1e-01 7.9e-17 1.3e-04 7.2e-03 7.1e-03 -1.8e-02 0.0e+00
    21   23   22    0 1.0e+00 1.0e+00 1.0e+00  3.28287e-03 4.0e-01 1.2e-15 6.2e-06 7.2e-03 7.2e-03 -1.8e-03 0.0e+00
    22   24   23    0 1.0e+00 1.0e+00 1.0e+00  9.13214e-04 9.5e-01 5.6e-16 3.2e-05 7.2e-03 7.1e-03 -4.1e-03 0.0e+00
    23   25   24    0 1.0e+00 1.0e+00 1.0e+00  3.15152e-04 2.1e-01 3.7e-17 1.3e-07 7.2e-03 7.2e-03 -9.4e-04 0.0e+00
    24   26   25    0 1.0e+00 1.0e+00 1.0e+00  7.32130e-05 2.1e-02 5.8e-16 5.4e-03 1.8e-03 7.2e-03 -2.6e-04 0.0e+00
    25   27   26    0 1.0e+00 1.0e+00 1.0e+00  5.02289e-06 3.7e-02 1.5e-15 6.7e-05 1.8e-03 1.8e-03 -4.1e-03 0.0e+00
    26   28   27    0 1.0e+00 1.0e+00 1.0e+00  2.15778e-06 3.5e-04 4.3e-16 1.3e-03 4.5e-04 1.8e-03 -2.5e-06 0.0e+00
    27   29   28    0 1.0e+00 1.0e+00 1.0e+00  1.05783e-07 1.9e-03 9.6e-16 3.4e-04 1.1e-04 4.5e-04 -1.0e-03 0.0e+00
    28   30   29    0 1.0e+00 1.0e+00 1.0e+00  7.93318e-09 7.6e-05 1.5e-15 8.4e-05 2.8e-05 1.1e-04 -2.5e-04 0.0e+00
    29   31   30    0 1.0e+00 1.0e+00 1.0e+00  4.83990e-10 1.8e-05 6.8e-17 2.1e-05 7.0e-06 2.8e-05 -6.3e-05 0.0e+00

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    30   32   31    0 1.0e+00 1.0e+00 1.0e+00  3.02135e-11 1.3e-07 7.2e-17 5.2e-06 1.7e-06 7.0e-06 -1.6e-05 0.0e+00
    31   33   32    0 1.0e+00 1.0e+00 1.0e+00  1.88688e-12 3.1e-06 2.1e-16 1.3e-06 4.4e-07 1.7e-06 -3.9e-06 0.0e+00
    32   34   33    0 1.0e+00 1.0e+00 1.0e+00  1.20749e-13 2.4e-06 3.9e-16 3.4e-07 1.0e-07 4.4e-07 -9.8e-07 0.0e+00
    33   35   34    0 1.0e+00 1.0e+00 1.0e+00  6.28070e-15 4.3e-08 4.3e-16 9.9e-10 1.0e-07 1.0e-07 -2.6e-07 0.0e+00

  ParOpt: Successfully converged to requested tolerance

The output from ParOptInteriorPoint consists of a summary of the parameter values.
Note that the "ParOptOptimizer" interface is used to run this example.
This class provides a common interface for all optimizers in ParOpt and therefore the file lists all options for all the optimizers, including the trust region method and the method of moving asymptotes.
Following the option settings, the output consists of a summary of the iteration history with the following columns:

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

The file can be visualized using the example in ``examples/plot_history/plot_history.py``.
This output can be unpacked from the file using:

.. code-block:: python

  values = paropt.unpack_output('paropt.out')

The trust region variant of the algorithm, that is used as the default setting, can be run by modifying the options:

.. code-block:: python

  options = {'algorithm': tr}

This produces the following result in the output in "paropt.tr":

::

  ParOptTrustRegion Parameter Summary:
  abs_res_tol                              1e-06
  abs_step_tol                             0
  algorithm                                tr
  armijo_constant                          1e-05
  barrier_strategy                         monotone
  design_precision                         1e-14
  eisenstat_walker_alpha                   1.5
  eisenstat_walker_gamma                   1
  function_precision                       1e-10
  gmres_atol                               1e-30
  gmres_subspace_size                      0
  gradient_check_step_length               1e-06
  gradient_verification_frequency          -1
  hessian_reset_freq                       1000000
  init_barrier_param                       0.1
  init_rho_penalty_search                  0
  ip_checkpoint_file                       (null)
  max_bound_value                          1e+20
  max_gmres_rtol                           0.1
  max_line_iters                           10
  max_major_iters                          5000
  min_fraction_to_boundary                 0.95
  min_rho_penalty_search                   0
  mma_asymptote_contract                   0.7
  mma_asymptote_relax                      1.2
  mma_bound_relax                          0
  mma_delta_regularization                 1e-05
  mma_eps_regularization                   0.001
  mma_infeas_tol                           1e-05
  mma_init_asymptote_offset                0.25
  mma_l1_tol                               1e-06
  mma_linfty_tol                           1e-06
  mma_max_asymptote_offset                 10
  mma_max_iterations                       200
  mma_min_asymptote_offset                 0.01
  mma_output_file                          paropt.mma
  mma_use_constraint_linearization         1
  monotone_barrier_fraction                0.25
  monotone_barrier_power                   1.1
  nk_switch_tol                            0.001
  norm_type                                infinity
  output_file                              paropt.out
  output_level                             0
  penalty_descent_fraction                 0.3
  penalty_gamma                            1000
  problem_name                             (null)
  qn_sigma                                 0
  qn_subspace_size                         10
  qn_type                                  bfgs
  rel_bound_barrier                        1
  rel_func_tol                             0
  sequential_linear_method                 0
  start_affine_multiplier_min              0.001
  starting_point_strategy                  affine_step
  tr_adaptive_constraint                   linear_constraint
  tr_adaptive_gamma_update                 1
  tr_adaptive_objective                    linear_objective
  tr_bound_relax                           0.0001
  tr_eta                                   0.25
  tr_infeas_tol                            1e-05
  tr_init_size                             0.1
  tr_l1_tol                                1e-06
  tr_linfty_tol                            1e-06
  tr_max_iterations                        200
  tr_max_size                              1
  tr_min_size                              0.001
  tr_output_file                           paropt.tr
  tr_penalty_gamma_max                     10000
  tr_penalty_gamma_min                     0
  tr_write_output_frequency                10
  use_backtracking_alpha                   0
  use_diag_hessian                         0
  use_hvec_product                         0
  use_line_search                          1
  use_qn_gmres_precon                      1
  use_quasi_newton_update                  0
  write_output_frequency                   0

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
      0  4.04000e+02  0.00e+00  9.61e+02  6.19e+02  1.00e-01  1.50e-01  8.97e-01  1.20e+02  9.56e-08  9.56e-08  1.00e+03  1.00e+03 20/14
      1  2.96020e+02  0.00e+00  6.60e+02  3.97e+02  1.50e-01  2.25e-01  1.03e+00  1.17e+02  8.64e-08  8.64e-08  1.00e+03  1.00e+03 20/14
      2  1.75328e+02  0.00e+00  3.31e+02  1.71e+02  2.25e-01  3.38e-01  1.12e+00  9.75e+01  7.66e-08  7.66e-08  1.00e+03  1.00e+03 20/14
      3  6.64257e+01  0.00e+00  1.10e+02  6.40e+01  3.21e-01  5.06e-01  1.31e+00  4.15e+01  6.79e-08  6.79e-08  1.00e+03  1.00e+03 20/14
      4  1.20441e+01  0.00e+00  2.32e+01  1.34e+01  2.09e-01  7.59e-01  1.19e+00  8.36e+00  6.39e-08  6.39e-08  1.00e+03  1.00e+03 20/14
      5  2.05970e+00  0.00e+00  3.00e+00  2.67e+00  5.55e-02  1.00e+00  1.07e+00  4.66e-01  6.29e-08  6.29e-08  1.00e+03  1.00e+03 20/14
      6  1.56179e+00  0.00e+00  2.83e+00  2.16e+00  6.60e-03  1.00e+00  1.72e+00  9.09e-03  6.28e-08  6.28e-08  1.00e+03  1.00e+03 20/14
      7  1.54617e+00  0.00e+00  4.12e+00  2.91e+00  3.53e-02  1.00e+00  1.67e+00  3.97e-02  6.24e-08  6.24e-08  1.00e+03  1.00e+03 20/15
      8  1.47972e+00  0.00e+00  4.75e+00  2.97e+00  1.52e-01  1.00e+00  2.30e+00  1.50e-01  6.58e-08  6.58e-08  1.00e+03  1.00e+03 skipH 20/15
      9  1.13633e+00  0.00e+00  4.75e+00  2.97e+00  0.00e+00  2.50e-01 -4.30e-01  1.28e+02  8.76e-08  8.76e-08  1.00e+03  1.00e+03 21/15

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    10  1.13633e+00  0.00e+00  3.31e+00  2.15e+00  2.25e-02  3.75e-01  1.19e+00  5.13e-02  7.66e-08  7.66e-08  1.00e+03  1.00e+03 20/15
    11  1.07542e+00  0.00e+00  4.22e+00  2.15e+00  1.04e-02  5.62e-01  1.72e+00  8.20e-03  6.09e-08  6.09e-08  1.00e+03  1.00e+03 20/14
    12  1.06130e+00  0.00e+00  6.38e+00  4.62e+00  4.64e-02  8.44e-01  1.37e+00  3.66e-02  6.05e-08  6.05e-08  1.00e+03  1.00e+03 20/14
    13  1.01113e+00  0.00e+00  6.86e+00  5.73e+00  4.37e-02  1.00e+00  1.60e+00  3.43e-02  6.01e-08  6.01e-08  1.00e+03  1.00e+03 20/14
    14  9.56353e-01  0.00e+00  6.72e+00  6.43e+00  8.93e-02  1.00e+00  1.43e+00  9.64e-02  5.89e-08  5.89e-08  1.00e+03  1.00e+03 20/14
    15  8.18524e-01  0.00e+00  3.78e+00  3.64e+00  7.57e-02  1.00e+00  1.51e+00  1.27e-01  5.76e-08  5.76e-08  1.00e+03  1.00e+03 20/15
    16  6.26030e-01  0.00e+00  5.28e+00  3.95e+00  1.06e-01  1.00e+00  1.47e+00  9.94e-02  5.61e-08  5.61e-08  1.00e+03  1.00e+03 20/15
    17  4.79703e-01  0.00e+00  4.30e+00  2.61e+00  1.09e-01  1.00e+00  1.09e+00  1.46e-01  5.42e-08  5.42e-08  1.00e+03  1.00e+03 20/15
    18  3.20830e-01  0.00e+00  4.30e+00  2.61e+00  0.00e+00  2.50e-01 -2.04e+01  3.05e-01  4.92e-08  4.92e-08  1.00e+03  1.00e+03 20/15
    19  3.20830e-01  0.00e+00  3.00e+00  2.00e+00  9.31e-03  3.75e-01  1.86e+00  8.06e-03  5.35e-08  5.35e-08  1.00e+03  1.00e+03 20/15

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    20  3.05835e-01  0.00e+00  3.00e+00  2.00e+00  0.00e+00  9.38e-02  1.65e-01  8.16e-02  5.19e-08  5.19e-08  1.00e+03  1.00e+03 20/15
    21  3.05835e-01  0.00e+00  9.55e-01  7.77e-01  6.34e-02  1.41e-01  1.79e+00  3.74e-02  5.25e-08  5.25e-08  1.00e+03  1.00e+03 20/15
    22  2.39018e-01  0.00e+00  9.55e-01  7.77e-01  0.00e+00  3.52e-02 -9.85e-01  7.75e-02  5.03e-08  5.03e-08  1.00e+03  1.00e+03 20/14
    23  2.39018e-01  0.00e+00  1.70e+00  1.24e+00  3.52e-02  5.27e-02  1.65e+00  1.86e-02  5.18e-08  5.18e-08  1.00e+03  1.00e+03 21/14
    24  2.08346e-01  0.00e+00  4.51e+00  2.42e+00  5.27e-02  7.91e-02  9.19e-01  3.58e-02  5.09e-08  5.09e-08  1.00e+03  1.00e+03 21/15
    25  1.75478e-01  0.00e+00  9.18e+00  4.94e+00  7.02e-02  1.19e-01  8.47e-01  1.90e-02  4.99e-08  4.99e-08  1.00e+03  1.00e+03 20/15
    26  1.59405e-01  0.00e+00  6.02e+00  3.13e+00  4.94e-03  1.78e-01  1.67e+00  1.39e-02  1.71e-08  1.71e-08  1.00e+03  1.00e+03 21/15
    27  1.36136e-01  0.00e+00  9.21e-01  6.14e-01  8.61e-02  2.67e-01  1.38e+00  3.91e-02  2.05e-08  2.05e-08  1.00e+03  1.00e+03 21/15
    28  8.23159e-02  0.00e+00  5.61e+00  3.31e+00  1.26e-01  4.00e-01  1.21e+00  2.51e-02  4.75e-08  4.75e-08  1.00e+03  1.00e+03 20/15
    29  5.18909e-02  0.00e+00  2.98e+00  1.73e+00  4.68e-02  6.01e-01  1.58e+00  1.18e-02  6.22e-08  6.22e-08  1.00e+03  1.00e+03 20/15

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    30  3.31700e-02  0.00e+00  4.98e+00  3.13e+00  1.22e-01  9.01e-01  1.08e+00  1.37e-02  1.71e-08  1.71e-08  1.00e+03  1.00e+03 21/15
    31  1.82999e-02  0.00e+00  3.01e+00  2.01e+00  5.94e-02  9.01e-01  6.36e-01  1.62e-02  5.55e-08  5.55e-08  1.00e+03  1.00e+03 20/15
    32  8.02349e-03  0.00e+00  1.61e+00  1.04e+00  7.87e-02  1.00e+00  1.02e+00  6.28e-03  4.63e-08  4.63e-08  1.00e+03  1.00e+03 20/15
    33  1.61748e-03  0.00e+00  4.80e-01  2.98e-01  3.87e-03  1.00e+00  1.35e+00  5.74e-04  1.66e-08  1.66e-08  1.00e+03  1.00e+03 21/15
    34  8.41319e-04  0.00e+00  1.93e-01  1.22e-01  3.70e-02  1.00e+00  1.26e+00  5.93e-04  1.95e-08  1.95e-08  1.00e+03  1.00e+03 21/15
    35  9.29508e-05  0.00e+00  1.05e-01  6.92e-02  1.48e-02  1.00e+00  1.18e+00  7.40e-05  2.01e-08  2.01e-08  1.00e+03  1.00e+03 21/15
    36  5.89284e-06  0.00e+00  1.41e-02  9.59e-03  2.86e-03  1.00e+00  9.81e-01  5.88e-06  2.03e-08  2.03e-08  1.00e+03  1.00e+03 21/15
    37  1.31736e-07  0.00e+00  2.61e-03  1.78e-03  6.48e-04  1.00e+00  8.15e-01  1.56e-07  4.67e-08  4.67e-08  1.00e+03  1.00e+03 20/15
    38  4.56083e-09  0.00e+00  9.93e-06  6.77e-06  1.02e-04  1.00e+00  9.96e-01  4.58e-09  4.66e-08  4.66e-08  1.00e+03  1.00e+03 20/15
    39  5.24274e-14  0.00e+00  1.42e-08  9.14e-09  4.51e-07  1.00e+00  1.00e+00  4.75e-14  4.66e-08  4.66e-08  1.00e+03  1.00e+03 20/14

The trust region output again outputs the parameter values.
Next, the file contains the iteration history with the following columns

* ``iter`` the current iteration
* ``fobj`` the objective value
* ``infeas`` the l1 norm of the constraint violation
* ``l1`` the l1 norm of the optimality conditions
* ``linfty`` the l-infinity norm of the optimality conditions
* ``|x - xk|`` the step lenth
* ``tr`` the trust region radius
* ``rho`` the ratio of the actual improvement to the expected improvement used in the acceptance criteria
* ``mod red.`` the predicted model reduction
* ``avg z`` the average multiplier
* ``max z`` the maximum multiplier
* ``avg pen.`` the average penalty value
* ``max pen.`` the maximum penalty value
* ``info`` information about the interior-point subproblem solution (iteration for the step/iteration for the penalty problem)
