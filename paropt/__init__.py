'''
ParOpt is an interior point optimizer
'''

import os

def get_cython_include():
    '''
    Get the include directory for the Cython .pxd files in ParOpt
    '''
    return [os.path.abspath(os.path.dirname(__file__))]

def get_include():
    '''
    Get the include directory for the Cython .pxd files in ParOpt
    '''
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_inc_dirs = ['src']

    inc_dirs = []
    for path in rel_inc_dirs:
    	inc_dirs.append(os.path.join(root_path, path))

    return inc_dirs

def get_libraries():
    '''
    Get the library directories
    '''
    root_path, tail = os.path.split(os.path.abspath(os.path.dirname(__file__)))

    rel_lib_dirs = ['lib']
    libs = ['paropt']
    lib_dirs = []
    for path in rel_lib_dirs:
    	lib_dirs.append(os.path.join(root_path, path))

    return lib_dirs, libs

try:
    from paropt.plot_history import plot_history
except:
    pass
