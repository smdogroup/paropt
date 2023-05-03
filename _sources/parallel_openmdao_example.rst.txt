Parallel OpenMDAO example
=========================

This examples uses the ParOpt driver for OpenMDAO to solve the
following optimization problem in parallel:

.. math::

   \begin{align}
        \text{min} \qquad & (w - 10)^2 + \sum_i{\left( x_i - 5 \right) ^2} \\
        \text{with respect to} \qquad & x_i \le 10 \\
        \text{subject to} \qquad & \sum_i{x_i ^3} \le 10 \\
    \end{align}


There are two inputs to the distributed paraboloid component:
`x` is an array input connected to a distributed IndependentVarComp,
and `w` is a scalar input connected to a non-distributed IndependentVarComp.

Because `w` is a non-distributed variable connected as an input to a
distributed component, it will be duplicated on each processor (it has
a local size of 1, but a global size equal to the number of
processors). There are two options for connecting variables from a
non-distributed component to a distributed component. This behavior is
governed by defining `src_indices` for the component. This example
uses the default for `w` , where rank0 would get `src_indices = 0`,
rank1 would get `src_indices = 1`, and so on. When the default
behavior is used, there will be a warning issued by OpenMDAO to
clarify what the default behavior is doing, but this warning doesn't
imply that anything is wrong.

For parallel optimization with ParOpt, the objective and constraint
values are expected to be duplicated on all processors, while the design
variables are distributed across processors. Therefore in this example,
the objective (`y`) and constraint (`a`) values are computed as the sum of
an `Allgather` operation.

Python implementation
---------------------

The python implementation of this problem is as follows

.. literalinclude:: ../../examples/openmdao/distrib_paraboloid.py
    :language: python

This code can be run with any number of processors (for example, using `mpirun -np <# of processors> python distrib_paraboloid.py`). Using two processors, this code results in the following output:

::

   /usr/local/lib/python3.9/site-packages/openmdao/core/component.py:905: UserWarning:'dp' <class DistribParaboloid>: Component is distributed but input 'dp.w' was added without src_indices. Setting src_indices to np.arange(0, 1, dtype=int).reshape((1,)).
   /usr/local/lib/python3.9/site-packages/openmdao/core/component.py:905: UserWarning:'dp' <class DistribParaboloid>: Component is distributed but input 'dp.w' was added without src_indices. Setting src_indices to np.arange(1, 2, dtype=int).reshape((1,)).
   f = 133.94
   c = -0.00
   Rank = 0; x = [1.25992104 1.25992104 1.25992104]
   Rank = 1; x = [1.25992104 1.25992104]

