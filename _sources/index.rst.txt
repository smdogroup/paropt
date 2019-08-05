ParOpt Overview
===============

ParOpt is a parallel optimization library for large-scale optimization using distributed design variables.
ParOpt is implemented in C++ with a Python wrapper generated using Cython.
ParOpt contains both an interior-point optimization algorithm with a line search globalization strategy and an :math:`l_{\infty}` trust region algorithm with an :math:`l_{1}` penalty function.

Please cite us
--------------

The key contributions and development of ParOpt are described in the following paper: 
Ting Wei Chin, Mark K. Leader, Graeme J. Kennedy, A scalable framework for large-scale 3D multimaterial topology optimization with octree-based mesh adaptation, Advances in Engineering Software, Volume 135, 2019.

.. code-block:: none

    @article{Chin:2019,
             title = {A scalable framework for large-scale 3D multimaterial topology optimization with octree-based mesh adaptation},
             journal = {Advances in Engineering Software},
             volume = {135},
             year = {2019},
             doi = {10.1016/j.advengsoft.2019.05.004},
             author = {Ting Wei Chin and Mark K. Leader and Graeme J. Kennedy}}

Installation
------------

The installation process for ParOpt consists of first compiling the C++ source and then installing the python interface.
The C++ source requires a BLAS/LAPACK installation and a MPI compiler for C++.
Using the Python interface additional requires numpy, mpi4py and Cython.

Currently ParOpt does not use an build system so all options are set in the file "Makefile.in".
Default settings for these options are contained within the file "Makefile.in.info".
The first step in compling ParOpt is to first copy the file "Makefile.in.info" to "Makefile.in".

Key parameters that should be set within the "Makefile.in" file are:

1) "PAROPT_DIR" the root ParOpt directory
2) "CXX" the MPI-enabled C++ compiler
3) "LAPACK_LIBS" the link command for the BLAS/LAPACK libraries

By default, the shared and static library are complied to the directory "PAROPT_DIR/lib".

Installation of the python interface is performed using the "setup.py" script.
The recommended python installation command "python setup.py build_ext --inplace" can be executed by typing "make interface" in the root directory.

Introduction and examples
=========================

.. toctree::
   :maxdepth: 2
   :titlesonly:
   
   sellar
   rosenbrock
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
