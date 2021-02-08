Options and generic interface for ParOpt optimziers
===================================================

ParOpt consists of three different optimizers: an interior point method, a trust-region method and the method of moving asymptotes.

These optimizers can be accessed through the common python interface "ParOpt.Optimizers".
This python object is allocated with a problem class which inherits from "ParOpt.Problem", and a dictionary of options.

The optimizer interface is generally executed as follows:

.. code-block:: python

  # Create the optimizer with the specified options. Here we specify a
  # trust-region optimizer, with an initial trust region size of 0.1 and
  # a maximum size of 10.0. All other options are set to default.
  options = {
      'algorithm': 'tr',
      'tr_init_size': 0.1,
      'tr_max_size': 10.0}
  opt = ParOpt.Optimizer(problem, options)

  # Execute the optimization
  opt.optimize()

  # Extract the optimized values and multipliers
  x, z, zw, zl, zu = opt.getOptimizedPoint()

Switching the above optimization problem to use the interior-point method or the method of moving asymptotes will be as simple as specifying ``'ip'`` or ``'mma'`` as the argument associated with ``'algorithm'``.

Options
-------

The option data is populated directly from the C++ code.
The options are pulled from all optimizers, so not all options are applicable.
In general the options specific to the trust region method have ``tr_`` as a prefix while options associated with the method of moving asymptotes have ``mma_`` as a prefix.
Options without the ``tr_`` or ``mma_`` prefix apply to the interior point method.

The full set of options can displayed as follows:

.. code-block:: python

  from paropt import ParOpt
  ParOpt.printOptionSummary()

This produces the following output:

::

  Absolute stopping criterion
  abs_res_tol                              1e-06
  Range of values: lower limit 0  upper limit 1e+20

  Absolute stopping norm on the step size
  abs_step_tol                             0
  Range of values: lower limit 0  upper limit 1e+20

  The type of optimization algorithm
  algorithm                                tr
  Range of values:                         ip
                                           tr
                                           mma

  The Armijo constant for the line search
  armijo_constant                          1e-05
  Range of values: lower limit 0  upper limit 1

  The type of barrier update strategy to use
  barrier_strategy                         monotone
  Range of values:                         monotone
                                           mehrotra
                                           mehrotra_predictor_corrector
                                           complementarity_fraction

  The absolute precision of the design variables
  design_precision                         1e-14
  Range of values: lower limit 0  upper limit 1

  Exponent in the Eisenstat-Walker INK forcing equation
  eisenstat_walker_alpha                   1.5
  Range of values: lower limit 0  upper limit 2

  Multiplier in the Eisenstat-Walker INK forcing equation
  eisenstat_walker_gamma                   1
  Range of values: lower limit 0  upper limit 1

  The absolute precision of the function and constraints
  function_precision                       1e-10
  Range of values: lower limit 0  upper limit 1

  The absolute GMRES tolerance (almost never relevant)
  gmres_atol                               1e-30
  Range of values: lower limit 0  upper limit 1

  The subspace size for GMRES
  gmres_subspace_size                      0
  Range of values: lower limit 0  upper limit 1000

  Step length used to check the gradient
  gradient_check_step_length               1e-06
  Range of values: lower limit 0  upper limit 1

  Print to screen the output of the gradient check at this frequency during an optimization
  gradient_verification_frequency          -1
  Range of values: lower limit -1000000  upper limit 1000000

  Do a hard reset of the Hessian at this specified major iteration frequency
  hessian_reset_freq                       1000000
  Range of values: lower limit 1000000  upper limit 1000000

  The initial value of the barrier parameter
  init_barrier_param                       0.1
  Range of values: lower limit 0  upper limit 1e+20

  Initial value of the line search penalty parameter
  init_rho_penalty_search                  0
  Range of values: lower limit 0  upper limit 1e+20

  Checkpoint file for the interior point method
  ip_checkpoint_file                       None

  Maximum bound value at which bound constraints are omitted
  max_bound_value                          1e+20
  Range of values: lower limit 0  upper limit 1e+300

  The maximum relative tolerance used for GMRES, above this the quasi-Newton approximation is used
  max_gmres_rtol                           0.1
  Range of values: lower limit 0  upper limit 1

  Maximum number of line search iterations
  max_line_iters                           10
  Range of values: lower limit 1  upper limit 100

  The maximum number of major iterations before quiting
  max_major_iters                          5000
  Range of values: lower limit 0  upper limit 1000000

  Minimum fraction to the boundary rule < 1
  min_fraction_to_boundary                 0.95
  Range of values: lower limit 0  upper limit 1

  Minimum value of the line search penalty parameter
  min_rho_penalty_search                   0
  Range of values: lower limit 0  upper limit 1e+20

  Contraction factor applied to the asymptotes
  mma_asymptote_contract                   0.7
  Range of values: lower limit 0  upper limit 1

  Expansion factor applied to the asymptotes
  mma_asymptote_relax                      1.2
  Range of values: lower limit 1  upper limit 1e+20

  Relaxation bound for computing the error in the KKT conditions
  mma_bound_relax                          0
  Range of values: lower limit 0  upper limit 1e+20

  Regularization term applied in the MMA approximation
  mma_delta_regularization                 1e-05
  Range of values: lower limit 0  upper limit 1e+20

  Regularization term applied in the MMA approximation
  mma_eps_regularization                   0.001
  Range of values: lower limit 0  upper limit 1e+20

  Infeasibility tolerance
  mma_infeas_tol                           1e-05
  Range of values: lower limit 0  upper limit 1e+20

  Initial aymptote offset from the variable bounds
  mma_init_asymptote_offset                0.25
  Range of values: lower limit 0  upper limit 1

  l1 tolerance for the optimality tolerance
  mma_l1_tol                               1e-06
  Range of values: lower limit 0  upper limit 1e+20

  l-infinity tolerance for the optimality tolerance
  mma_linfty_tol                           1e-06
  Range of values: lower limit 0  upper limit 1e+20

  Maximum asymptote offset from the variable bounds
  mma_max_asymptote_offset                 10
  Range of values: lower limit 0  upper limit 1e+20

  Maximum number of iterations
  mma_max_iterations                       200
  Range of values: lower limit 0  upper limit 1000000

  Minimum asymptote offset from the variable bounds
  mma_min_asymptote_offset                 0.01
  Range of values: lower limit 0  upper limit 1e+20

  Ouput file name for MMA
  mma_output_file                          paropt.mma

  If false, linearized the constraints
  mma_use_constraint_linearization         True

  Factor applied to the barrier update < 1
  monotone_barrier_fraction                0.25
  Range of values: lower limit 0  upper limit 1

  Exponent for barrier parameter update > 1
  monotone_barrier_power                   1.1
  Range of values: lower limit 1  upper limit 10

  Switch to the Newton-Krylov method at this residual tolerance
  nk_switch_tol                            0.001
  Range of values: lower limit 0  upper limit 1e+20

  The type of norm to use in all computations
  norm_type                                infinity
  Range of values:                         infinity
                                           l1
                                           l2

  Output file name
  output_file                              paropt.out

  Output level indicating how verbose the output should be
  output_level                             0
  Range of values: lower limit 0  upper limit 1000000

  Fraction of infeasibility used to enforce a descent direction
  penalty_descent_fraction                 0.3
  Range of values: lower limit 1e-06  upper limit 1

  l1 penalty parameter applied to slack variables
  penalty_gamma                            1000
  Range of values: lower limit 0  upper limit 1e+20

  The problem name
  problem_name                             None

  Scalar added to the diagonal of the quasi-Newton approximation > 0
  qn_sigma                                 0
  Range of values: lower limit 0  upper limit 1e+20

  The maximum dimension of the quasi-Newton approximation
  qn_subspace_size                         10
  Range of values: lower limit 0  upper limit 1000

  The the of quasi-Newton approximation to use
  qn_type                                  bfgs
  Range of values:                         bfgs
                                           sr1
                                           none

  The type of BFGS update to apply when the curvature condition fails
  qn_update_type                           skip_negative_curvature
  Range of values:                         skip_negative_curvature
                                           damped_update

  Relative factor applied to barrier parameter for bound constraints
  rel_bound_barrier                        1
  Range of values: lower limit 0  upper limit 1e+20

  Relative function value stopping criterion
  rel_func_tol                             0
  Range of values: lower limit 0  upper limit 1e+20

  Discard the quasi-Newton approximation (but not necessarily the exact Hessian)
  sequential_linear_method                 False

  Minimum multiplier for the affine step initialization strategy
  start_affine_multiplier_min              1
  Range of values: lower limit 0  upper limit 1e+20

  Initialize the Lagrange multiplier estimates and slack variables
  starting_point_strategy                  affine_step
  Range of values:                         least_squares_multipliers
                                           affine_step
                                           no_start_strategy

  The type of constraint to use for the adaptive penalty subproblem
  tr_adaptive_constraint                   linear_constraint
  Range of values:                         linear_constraint
                                           subproblem_constraint

  Adaptive penalty parameter update
  tr_adaptive_gamma_update                 True

  The type of objective to use for the adaptive penalty subproblem
  tr_adaptive_objective                    linear_objective
  Range of values:                         constant_objective
                                           linear_objective
                                           subproblem_objective

  Upper and lower bound relaxing parameter
  tr_bound_relax                           0.0001
  Range of values: lower limit 0  upper limit 1e+20

  Trust region trial step acceptance ratio
  tr_eta                                   0.25
  Range of values: lower limit 0  upper limit 1

  Infeasibility tolerance
  tr_infeas_tol                            1e-05
  Range of values: lower limit 0  upper limit 1e+20

  The initial trust region radius
  tr_init_size                             0.1
  Range of values: lower limit 0  upper limit 1e+20

  l1 tolerance for the optimality tolerance
  tr_l1_tol                                1e-06
  Range of values: lower limit 0  upper limit 1e+20

  l-infinity tolerance for the optimality tolerance
  tr_linfty_tol                            1e-06
  Range of values: lower limit 0  upper limit 1e+20

  Maximum number of trust region iterations
  tr_max_iterations                        200
  Range of values: lower limit 0  upper limit 1000000

  The maximum trust region radius
  tr_max_size                              1
  Range of values: lower limit 0  upper limit 1e+20

  The minimum trust region radius
  tr_min_size                              0.001
  Range of values: lower limit 0  upper limit 1e+20

  Trust region output file
  tr_output_file                           paropt.tr

  Maximum value for the penalty parameter
  tr_penalty_gamma_max                     10000
  Range of values: lower limit 0  upper limit 1e+20

  Minimum value for the penalty parameter
  tr_penalty_gamma_min                     0
  Range of values: lower limit 0  upper limit 1e+20

  The barrier update strategy to use for the steering method subproblem
  tr_steering_barrier_strategy             mehrotra_predictor_corrector
  Range of values:                         monotone
                                           mehrotra
                                           mehrotra_predictor_corrector
                                           complementarity_fraction
                                           default

  The barrier update strategy to use for the steering method subproblem
  tr_steering_starting_point_strategy      affine_step
  Range of values:                         least_squares_multipliers
                                           affine_step
                                           no_start_strategy
                                           default

  Write output frequency
  tr_write_output_frequency                10
  Range of values: lower limit 0  upper limit 1000000

  Perform a back-tracking line search
  use_backtracking_alpha                   False

  Use or do not use the diagonal Hessian computation
  use_diag_hessian                         False

  Use or do not use Hessian-vector products
  use_hvec_product                         False

  Perform or skip the line search
  use_line_search                          True

  Use or do not use the quasi-Newton method as a preconditioner
  use_qn_gmres_precon                      True

  Update the quasi-Newton approximation at each iteration
  use_quasi_newton_update                  True

  Write out the solution file and checkpoint file at this frequency
  write_output_frequency                   10
  Range of values: lower limit 0  upper limit 1000000
