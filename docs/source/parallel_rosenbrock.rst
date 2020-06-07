Parallel Rosenbrock
===================

This problem deals with a generalized Rosenbrock function in parallel. In this case the objective function is

.. math::

    f(x) = \sum_{i=1}^{n-1} (1 - x_{i})^{2} + 100(x_{i+1} - x_{i}^2)^2

Two constraints are imposed in this problem. The first is that the point remain within a ball of radius 1/2 centered on the origin:

.. math::

    c_{1}(x) = \frac{1}{4} - \sum_{i=1}^{n} x_{i}^{2} \ge 0

The second is a linear constraint that the sum of all variables be greater than -10:

.. math::

    c_{2}(x) = \sum_{i=1}^{n} x_{i} + 10 \ge 0

C++ implementation
------------------

This example is a further demonstration of the ParOptProblem interface class making use of a distributed design vector, separable sparse constraints, and Hessian-vector products.

The Hessian-vector products can be used to accelerate the convergence of the optimizer.
They are generally used once the optimization problem has converged to a point that is closer to the optimum.
These products are implemented by the user in the ParOptProblem class, using the following prototype:

.. code-block:: c++

      int evalHvecProduct( ParOptVec *xvec,
                           ParOptScalar *z, ParOptVec *zwvec,
                           ParOptVec *pxvec, ParOptVec *hvec );

This function provides access to Hessian-vector products of the Lagrangian.
In ParOpt the Lagrangian function is defined as

.. math::

    \mathcal{L} \triangleq f(x) - z^{T} c(x) - z_{w}^{T} c_{w}(x)

where :math:`\mathcal{L}` is the Lagrangian, :math:`f(x), c(x), c_{w}(x)` are the objective, dense constraints and sparse separable constraints, respectively, and :math:`z, z_{w}` are multipliers associated with the dense and sparse constraints.
The Hessian-vector product is then computed as

.. math::

    h = \nabla^{2} \mathcal{L}(x, z, z_{w}) p_{x}

The interface to the sparse separable constraint code consists of four functions.
These functions consist of the following:

.. code-block:: c++

      void evalSparseCon( ParOptVec *x, ParOptVec *out );
      void addSparseJacobian( ParOptScalar alpha, ParOptVec *x,
                              ParOptVec *px, ParOptVec *out );
      void addSparseJacobianTranspose( ParOptScalar alpha, ParOptVec *x,
                                       ParOptVec *pzw, ParOptVec *out );
      void addSparseInnerProduct( ParOptScalar alpha, ParOptVec *x,
                                  ParOptVec *cvec, ParOptScalar *A );

These member functions provide the following mathematical operations:

.. math::

    \begin{align}
        \mathrm{out} & \leftarrow c_{w} \leftarrow c_{w}(x) \\
        \mathrm{out} & \leftarrow \alpha A_{w}(x) p_{x} \\
        \mathrm{out} & \leftarrow \alpha A_{w}(x)^{T} p_{z_{w}} \\
        \mathrm{out} & \leftarrow \alpha A_{w} C A_{w}(x)^{T}  \\
    \end{align}

Here :math:`A_{w}(x) = \nabla c_{w}(x)$` and :math:`C` is a diagonal matrix.

.. literalinclude:: ../../examples/rosenbrock/rosenbrock.cpp
    :language: c++