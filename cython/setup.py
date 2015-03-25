
import mpi4py
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sources1 = ['ParOptVec_c.pyx', 'ParOptVec.c']
sources=['ParOpt_c.pyx', 'ParOptVec.c', 'ParOpt.c', 'Rosenbrock.c']
    
setup(
    ext_modules=[Extension("ParOptVec_c",
                           sources=sources1,
                           language="c++",
                           libraries=["mpi_cxx", "lapack", "blas"],
                           extra_compile_args=["-O3"],
                           extra_link_args=["-L/usr/lib/libmpi_cxx","-L/usr/lib/lapack/liblapack",
                                            "-L/usr/lib/libblas/libblas", "-O3"]),
                Extension("ParOpt_c",
                           sources=sources,
                           language="c++",
                           libraries=["mpi_cxx", "lapack", "blas"],
                           extra_compile_args=["-O3"],
                           extra_link_args=["-L/usr/lib/libmpi_cxx","-L/usr/lib/lapack/liblapack",
                                            "-L/usr/lib/libblas/libblas", "-O3"])],
    cmdclass = {'build_ext':build_ext},
    include_dirs=[mpi4py.get_include(),numpy.get_include()],
    )
