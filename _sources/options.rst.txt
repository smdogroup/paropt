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

.. _options-label:

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

.. program-output:: python -c "from paropt import ParOpt; ParOpt.printOptionSummary()"
