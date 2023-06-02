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

This code produces the following output in the default file ``paropt.out``:

::

  ParOptInteriorPoint Parameter Summary:
  algorithm                                ip

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
     0    1    1    0      --      --      --  4.04000e+02 8.5e+02 9.4e+02 8.4e+05 1.7e+05 1.7e+05       --      --
     1    3    2    0 3.5e-01 1.0e-03 1.0e-01  9.02442e+00 1.6e+02 9.4e+02 8.7e+05 4.2e+04 1.6e+05 -5.0e+05 0.0e+00
     2    4    3    0 1.0e+00 8.3e-01 1.0e+00  1.11146e+03 1.8e+03 1.6e+02 2.8e+05 1.0e+04 9.1e+04 -7.2e+05 0.0e+00
     3    5    4    0 1.0e+00 1.0e+00 1.0e+00  2.84010e+02 1.0e+03 3.7e-01 1.7e+04 2.6e+03 1.1e+04 -4.2e+05 0.0e+00
     4    6    5    0 1.0e+00 1.0e+00 1.0e+00  6.92825e+01 5.1e+02 5.3e-15 3.5e+03 6.5e+02 2.7e+03 -1.9e+04 0.0e+00
     5    7    6    0 1.0e+00 1.0e+00 1.0e+00  2.07229e+01 1.0e+02 8.9e-16 9.2e+02 1.6e+02 6.5e+02 -2.4e+03 0.0e+00
     6    8    7    0 1.0e+00 1.0e+00 1.0e+00  2.68808e+00 4.8e+01 2.8e-17 1.5e+02 4.1e+01 1.6e+02 -8.5e+01 0.0e+00
     7    9    8    0 1.0e+00 1.0e+00 1.0e+00  2.34024e-01 1.4e+01 2.8e-16 3.4e+01 1.0e+01 4.1e+01 -8.6e+01 0.0e+00
     8   10    9    0 1.0e+00 1.0e+00 1.0e+00  1.12205e-01 1.9e+00 1.4e-15 7.9e+00 2.6e+00 1.0e+01 -2.1e+01 0.0e+00
     9   11   10    0 1.0e+00 1.0e+00 1.0e+00  1.12286e-01 4.5e-02 2.5e-16 1.9e+00 6.4e-01 2.6e+00 -5.6e+00 0.0e+00

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    10   12   11    0 1.0e+00 1.0e+00 1.0e+00  1.11649e-01 2.8e-01 1.2e-15 4.8e-01 1.6e-01 6.4e-01 -1.4e+00 0.0e+00
    11   13   12    0 1.0e+00 1.0e+00 1.0e+00  1.09062e-01 8.3e-01 7.2e-16 1.2e-01 4.0e-02 1.6e-01 -3.6e-01 0.0e+00
    12   14   13    0 1.0e+00 1.0e+00 1.0e+00  1.04593e-01 1.5e+00 1.4e-15 1.2e-03 4.0e-02 4.0e-02 -9.5e-02 0.0e+00
    13   15   14    0 1.0e+00 1.0e+00 1.0e+00  9.69024e-02 2.5e+00 2.8e-17 7.7e-06 4.0e-02 4.0e-02 -9.8e-03 0.0e+00
    14   16   15    0 1.0e+00 1.0e+00 1.0e+00  8.39219e-02 3.4e+00 1.7e-16 3.7e-05 4.0e-02 4.0e-02 -1.7e-02 0.0e+00
    15   17   16    0 1.0e+00 1.0e+00 1.0e+00  6.08092e-02 3.3e+00 3.2e-16 8.3e-05 4.0e-02 4.0e-02 -2.9e-02 0.0e+00
    16   18   17    0 1.0e+00 1.0e+00 1.0e+00  3.10043e-02 7.5e-01 2.3e-16 1.3e-04 4.0e-02 4.0e-02 -4.2e-02 0.0e+00
    17   19   18    0 1.0e+00 1.0e+00 1.0e+00  1.78352e-02 2.5e+00 2.6e-16 2.5e-04 4.0e-02 4.0e-02 -2.0e-02 0.0e+00
    18   20   19    0 1.0e+00 1.0e+00 1.0e+00  8.31785e-03 4.6e-01 4.6e-16 5.9e-05 4.0e-02 4.0e-02 -1.3e-02 0.0e+00
    19   21   20    0 1.0e+00 1.0e+00 1.0e+00  3.40797e-03 1.2e+00 8.2e-16 1.7e-04 4.0e-02 4.0e-02 -5.7e-03 0.0e+00

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    20   22   21    0 1.0e+00 1.0e+00 1.0e+00  1.73702e-03 1.3e-02 6.3e-16 3.0e-02 1.0e-02 4.0e-02 -2.4e-03 0.0e+00
    21   23   22    0 1.0e+00 1.0e+00 1.0e+00  3.31850e-04 3.3e-01 1.4e-15 1.3e-03 1.0e-02 1.0e-02 -2.4e-02 0.0e+00
    22   24   23    0 1.0e+00 1.0e+00 1.0e+00  1.17623e-04 8.6e-02 9.6e-17 7.5e-03 2.5e-03 1.0e-02 -2.1e-04 0.0e+00
    23   25   24    0 1.0e+00 1.0e+00 1.0e+00  8.89746e-06 8.3e-02 1.4e-16 1.2e-04 2.5e-03 2.5e-03 -5.8e-03 0.0e+00
    24   26   25    0 1.0e+00 1.0e+00 1.0e+00  4.74120e-06 1.9e-02 2.7e-16 1.9e-03 6.2e-04 2.5e-03 -1.1e-05 0.0e+00
    25   27   26    0 1.0e+00 1.0e+00 1.0e+00  3.36109e-07 8.3e-04 3.0e-16 4.7e-04 1.6e-04 6.2e-04 -1.4e-03 0.0e+00
    26   28   27    0 1.0e+00 1.0e+00 1.0e+00  1.52109e-08 4.4e-04 6.8e-17 1.2e-04 3.9e-05 1.6e-04 -3.5e-04 0.0e+00
    27   29   28    0 1.0e+00 1.0e+00 1.0e+00  9.47900e-10 2.3e-05 1.2e-15 2.9e-05 9.7e-06 3.9e-05 -8.8e-05 0.0e+00
    28   30   29    0 1.0e+00 1.0e+00 1.0e+00  5.87613e-11 5.4e-07 4.8e-17 7.3e-06 2.4e-06 9.7e-06 -2.2e-05 0.0e+00
    29   31   30    0 1.0e+00 1.0e+00 1.0e+00  3.66518e-12 3.6e-08 4.5e-16 1.8e-06 6.1e-07 2.4e-06 -5.5e-06 0.0e+00

  iter nobj ngrd nhvc   alpha   alphx   alphz         fobj   |opt| |infes|  |dual|      mu    comp   dmerit     rho info
    30   32   31    0 1.0e+00 1.0e+00 1.0e+00  2.29012e-13 3.6e-09 5.6e-16 4.6e-07 1.5e-07 6.1e-07 -1.4e-06 0.0e+00
    31   33   32    0 1.0e+00 1.0e+00 1.0e+00  1.30833e-14 2.4e-10 9.5e-16 4.6e-08 1.0e-07 1.5e-07 -3.5e-07 0.0e+00

ParOpt: Successfully converged to requested tolerance
The output from ``ParOptInteriorPoint`` consists of a summary of the non-default parameter values.
Note that the ``ParOptOptimizer`` interface is used to run this example.
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

This produces the following result in the output in ``paropt.tr``:

::

  ParOptTrustRegion Parameter Summary:
  algorithm                                tr
  use_quasi_newton_update                  False
  write_output_frequency                   0

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
     0  4.04000e+02  0.00e+00  9.61e+02  6.19e+02  1.00e-01  1.50e-01  8.97e-01  1.20e+02  4.40e-08  4.40e-08  1.00e+03  1.00e+03 21/14
     1  2.96020e+02  0.00e+00  6.60e+02  3.97e+02  1.50e-01  2.25e-01  1.03e+00  1.17e+02  4.00e-08  4.00e-08  1.00e+03  1.00e+03 21/14
     2  1.75328e+02  0.00e+00  3.31e+02  1.71e+02  2.25e-01  3.38e-01  1.12e+00  9.75e+01  3.56e-08  3.56e-08  1.00e+03  1.00e+03 21/14
     3  6.64257e+01  0.00e+00  1.10e+02  6.40e+01  3.21e-01  5.06e-01  1.31e+00  4.15e+01  3.16e-08  3.16e-08  1.00e+03  1.00e+03 21/14
     4  1.20441e+01  0.00e+00  2.32e+01  1.34e+01  2.09e-01  7.59e-01  1.19e+00  8.36e+00  2.99e-08  2.99e-08  1.00e+03  1.00e+03 21/14
     5  2.05970e+00  0.00e+00  3.00e+00  2.67e+00  5.55e-02  1.00e+00  1.07e+00  4.66e-01  2.94e-08  2.94e-08  1.00e+03  1.00e+03 21/14
     6  1.56179e+00  0.00e+00  2.83e+00  2.16e+00  6.60e-03  1.00e+00  1.72e+00  9.09e-03  2.94e-08  2.94e-08  1.00e+03  1.00e+03 21/14
     7  1.54617e+00  0.00e+00  4.12e+00  2.91e+00  3.53e-02  1.00e+00  1.67e+00  3.97e-02  2.92e-08  2.92e-08  1.00e+03  1.00e+03 21/14
     8  1.47972e+00  0.00e+00  4.75e+00  2.97e+00  1.52e-01  1.00e+00  2.30e+00  1.50e-01  2.45e-08  2.45e-08  1.00e+03  1.00e+03 skipH 21/14
     9  1.13633e+00  0.00e+00  1.21e+01  1.07e+01  2.00e-01  1.00e+00  4.55e-01  2.53e-01  2.51e-08  2.51e-08  1.00e+03  1.00e+03 21/14

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    10  1.02106e+00  0.00e+00  2.03e+00  1.94e+00  1.16e-01  1.00e+00  2.96e-01  2.59e-01  2.82e-08  2.82e-08  1.00e+03  1.00e+03 21/14
    11  9.44283e-01  0.00e+00  3.86e+00  2.35e+00  4.52e-02  1.00e+00  1.66e+00  4.35e-02  2.80e-08  2.80e-08  1.00e+03  1.00e+03 21/14
    12  8.72328e-01  0.00e+00  8.91e+00  7.93e+00  9.37e-02  1.00e+00  3.37e-01  6.44e-02  2.75e-08  2.75e-08  1.00e+03  1.00e+03 21/14
    13  8.50592e-01  0.00e+00  4.77e+00  4.13e+00  3.48e-02  1.00e+00  1.08e+00  5.10e-02  2.77e-08  2.77e-08  1.00e+03  1.00e+03 21/14
    14  7.95434e-01  0.00e+00  3.84e+00  3.02e+00  1.42e-02  1.00e+00  1.83e+00  2.42e-02  2.75e-08  2.75e-08  1.00e+03  1.00e+03 21/14
    15  7.51180e-01  0.00e+00  2.77e+00  2.70e+00  1.32e-01  1.00e+00  1.50e+00  1.42e-01  2.66e-08  2.66e-08  1.00e+03  1.00e+03 21/14
    16  5.38497e-01  0.00e+00  1.29e+01  7.66e+00  1.40e-01  1.00e+00  5.75e-01  9.45e-02  2.55e-08  2.55e-08  1.00e+03  1.00e+03 21/15
    17  4.84154e-01  0.00e+00  1.33e+00  8.31e-01  2.75e-02  1.00e+00  1.07e+00  1.26e-01  2.55e-08  2.55e-08  1.00e+03  1.00e+03 21/14
    18  3.48437e-01  0.00e+00  1.59e+00  1.31e+00  8.28e-02  1.00e+00  1.74e+00  5.06e-02  2.48e-08  2.48e-08  1.00e+03  1.00e+03 21/14
    19  2.60394e-01  0.00e+00  1.59e+00  1.31e+00  0.00e+00  2.50e-01 -2.47e+00  1.03e-01  2.31e-08  2.31e-08  1.00e+03  1.00e+03 21/14

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    20  2.60394e-01  0.00e+00  1.95e+00  1.45e+00  6.85e-02  3.75e-01  1.57e+00  3.69e-02  2.43e-08  2.43e-08  1.00e+03  1.00e+03 21/14
    21  2.02340e-01  0.00e+00  1.08e+01  6.16e+00  1.87e-01  5.62e-01  9.16e-01  7.48e-02  2.29e-08  2.29e-08  1.00e+03  1.00e+03 21/14
    22  1.33863e-01  0.00e+00  1.08e+01  6.16e+00  0.00e+00  1.41e-01  2.00e-01  4.59e-02  2.11e-08  2.11e-08  1.00e+03  1.00e+03 21/14
    23  1.33863e-01  0.00e+00  1.95e+00  1.08e+00  3.01e-02  2.11e-01  9.51e-01  3.57e-02  2.08e-08  2.08e-08  1.00e+03  1.00e+03 21/14
    24  9.99659e-02  0.00e+00  1.35e+00  7.99e-01  2.92e-02  3.16e-01  1.86e+00  7.19e-03  2.29e-08  2.29e-08  1.00e+03  1.00e+03 21/14
    25  8.66158e-02  0.00e+00  1.20e+01  7.47e+00  2.05e-01  3.16e-01  2.99e-01  4.26e-02  2.17e-08  2.17e-08  1.00e+03  1.00e+03 21/14
    26  7.38950e-02  0.00e+00  9.08e-01  7.20e-01  8.04e-02  3.16e-01  5.39e-01  5.39e-02  1.82e-08  1.82e-08  1.00e+03  1.00e+03 21/14
    27  4.48191e-02  0.00e+00  4.32e-01  2.96e-01  5.48e-02  4.75e-01  1.79e+00  7.61e-03  2.19e-08  2.19e-08  1.00e+03  1.00e+03 21/14
    28  3.12023e-02  0.00e+00  4.32e-01  2.96e-01  0.00e+00  1.19e-01 -2.00e+00  2.31e-02  2.08e-08  2.08e-08  1.00e+03  1.00e+03 21/14
    29  3.12023e-02  0.00e+00  2.26e-01  1.39e-01  5.46e-02  1.78e-01  1.76e+00  5.91e-03  2.16e-08  2.16e-08  1.00e+03  1.00e+03 21/14

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    30  2.08028e-02  0.00e+00  2.26e-01  1.39e-01  0.00e+00  4.45e-02  1.73e-01  1.64e-02  2.07e-08  2.07e-08  1.00e+03  1.00e+03 21/14
    31  2.08028e-02  0.00e+00  2.12e-01  1.79e-01  4.45e-02  6.67e-02  1.25e+00  5.38e-03  2.14e-08  2.14e-08  1.00e+03  1.00e+03 21/14
    32  1.40946e-02  0.00e+00  1.25e+00  7.53e-01  6.67e-02  1.00e-01  9.35e-01  7.50e-03  2.11e-08  2.11e-08  1.00e+03  1.00e+03 21/14
    33  7.07496e-03  0.00e+00  3.65e+00  2.37e+00  7.35e-02  1.00e-01  5.32e-01  2.76e-03  2.06e-08  2.06e-08  1.00e+03  1.00e+03 21/14
    34  5.60569e-03  0.00e+00  1.67e+00  1.06e+00  1.58e-02  1.50e-01  1.26e+00  1.79e-03  1.94e-08  1.94e-08  1.00e+03  1.00e+03 21/14
    35  3.35667e-03  0.00e+00  5.38e-01  3.26e-01  1.84e-02  2.25e-01  1.51e+00  1.05e-03  2.03e-08  2.03e-08  1.00e+03  1.00e+03 21/14
    36  1.76414e-03  0.00e+00  6.54e-02  3.54e-02  4.14e-02  3.38e-01  1.39e+00  9.90e-04  2.07e-08  2.07e-08  1.00e+03  1.00e+03 21/14
    37  3.92289e-04  0.00e+00  3.63e-01  2.40e-01  3.34e-02  5.07e-01  1.04e+00  3.35e-04  2.04e-08  2.04e-08  1.00e+03  1.00e+03 21/14
    38  4.46965e-05  0.00e+00  1.70e-02  1.05e-02  3.49e-03  7.60e-01  1.09e+00  3.97e-05  2.05e-08  2.05e-08  1.00e+03  1.00e+03 21/14
    39  1.49333e-06  0.00e+00  3.00e-03  2.00e-03  2.38e-03  1.00e+00  1.01e+00  1.48e-06  2.04e-08  2.04e-08  1.00e+03  1.00e+03 21/14

  iter         fobj    infeas        l1    linfty  |x - xk|        tr       rho  mod red.     avg z     max z  avg pen.  max pen. info
    40  2.52590e-09  0.00e+00  7.23e-05  4.88e-05  1.12e-05  1.00e+00  9.78e-01  2.58e-09  2.04e-08  2.04e-08  1.00e+03  1.00e+03 21/14
    41  2.10651e-12  0.00e+00  2.52e-09  1.56e-09  1.64e-06  1.00e+00  1.00e+00  2.11e-12  2.04e-08  2.04e-08  1.00e+03  1.00e+03 21/14

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
