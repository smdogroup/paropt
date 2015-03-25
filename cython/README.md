#Wrapping of C++ code using Cython#

This directory illustrates the usage of Cython to wrap C++ code by 
wrapping ParOpt and the Rosenbrock code in the /src and /example directory
respectively. Through this thorough example, we showed how Cython handles 
inheritance and parallelism    

Note:
If copying newer versions of ParOpt from /src directory, this might not work 
if newer functions were created and utilized

#.pxd files#
These are files that handles the declarations that are shared amongst other 
Cython source files (.pyx). By writing the declarations in these files, one can
cimport (c-level import) these declarations to any Cython source files that
requires them

#.pyx files#
These are the Cython source files that handles the wrapping.

#setup.py#
This specifies the files that are wrapped and uses distutils.

To start, install the Cython package

There are 2 ways to compile this example.

#Using Distutils#
To compile using distutils, a  standard Python packaging tool, type the following commands into the directory:
   	    CC=mpicxx python setup.py build_ext --inplace

#Using Makefile#
To compile using Makefile, simply type make in the current directory.
