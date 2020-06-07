OpenMDAO example
================

This examples uses the paropt_driver for OpenMDAO to solve the following optimization problem:

.. math::

   \begin{align}
        \text{min} \qquad & (x-3)^2 + xy + (y+4)^2 - 3 \\
        \text{with respect to} \qquad & -50 \le x, y \le 50 \\
        \text{subject to} \qquad & x + y \ge 0 \\
    \end{align}

Python implementation
---------------------

The python implementation of this problem is as follows

.. literalinclude:: ../../examples/openmdao/paraboloid_min.py
    :language: python

This code results in the output:

::

   Minimum value = -27.00
   (x, y) = (7.00, -7.00)

