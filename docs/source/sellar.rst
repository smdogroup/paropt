Sellar problem
==============

To illustrate the application of ParOpt, consider the following optimization problem with the Sellar objective function:

.. math::

    \begin{align}
        \text{min} \qquad & x_1 + x_2^2 + x_3 + e^{-x_4} \\
        \text{with respect to} \qquad & 0 \le x_{1} \le 10 \\
        &  0 \le x_{2} \le 10 \\
        & -1 \le x_{3} \le 3.16 \\
        & -1 \le x_{4} \le 24 \\
        \text{subject to} \qquad & x_{1} + x_{2} - 1 \ge 0 \\
    \end{align}

C++ implementation
------------------

The first step to use the ParOpt optimization library is to create a problem class which inherits from ParOptProblem.
This class is used by ParOpt's interior-point or trust-region algorithms to get the function and gradient values from the problem.

Key functions required for the implementation of a ParOptProblem class are described below.

.. code-block:: c++

      void getVarsAndBounds( ParOptVec *xvec,
                             ParOptVec *lbvec, ParOptVec *ubvec );

To begin the optimization problem, the optimizer must know the starting point and the variable bounds for the problem
The member function getVarsAndBounds retrieves this information.
On return, the initial design variables are written to the design vector x, and the lower and upper bounds are written to the vectors lb and ub, respectively.

.. code-block:: c++

      int evalObjCon( ParOptVec *xvec,
                      ParOptScalar *fobj, ParOptScalar *cons );
      int evalObjConGradient( ParOptVec *xvec,
                              ParOptVec *gvec, ParOptVec **Ac );

The class inheriting from ParOptProblem must also implement member functions to evaluate the objective and constraints and their gradients.
The function evalObjCon takes in the design vector x, and returns a scalar value in fobj, and an array of the dense constraint values in cons.
When the code is run in parallel, the same objective value and constraint values must be returned on all processors.
The function evalObjConGradient sets the values of the objective and constraint gradients into the vector gvec, and the array of vectors Ac, respectively.
If an error is encountered during the evaluation of either the functions or gradients, a non-zero error code should be returned to terminate the optimization.

When implemented in C++, the complete Sellar problem is:

.. literalinclude:: ../../examples/sellar/sellar.cpp
    :language: c++

The local components of the design vector can be accessed by making a call to getArray.

.. code-block:: c++

    ParOptScalar *x;
    xvec->getArray(&x);

In this case, the code can only be run in serial, so the design vector is not distributed.

All objects in ParOpt are reference counted.
Use incref() to increase the reference count after an object is allocated.
When the object is no longer needed, call decref() to decrease the reference count and possibly delete the object.
Direct calls to delete the object should not be used.

Python implementation
---------------------

The python implementation of this problem is also straightforward.
In an analogous manner, the python implemenation uses a class inherited from ParOpt.Problem, a python wrapper for the CyParOptProblem class.
This inherited class must implement a getVarsAndBounds, evalObjCon and evalObjConGradient member functions.
Note that in python, the function signature is slightly different for evalObjCon.
Please note, the vectors returned to python access the underlying memory in ParOpt directly, therefore sometimes care must be taken to avoid expressions that do not assign values to the references returned from ParOpt.
These vectors are of type ParOpt.PVec, but act in many ways like a numpy array.

.. literalinclude:: ../../examples/sellar/sellar.py
    :language: python
