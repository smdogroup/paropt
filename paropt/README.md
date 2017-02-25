#Python interface to ParOpt

This directory contains the python interface to ParOpt. The interface uses Cython with callbacks to python. 

Data passed back to python is done in place through numpy arrays. This makes the code efficient and avoids copying. However, be careful not to overwrite design variable values as this will directly modify the design variable arrays in ParOpt itself.