import os
from subprocess import check_output

# Numpy/mpi4py must be installed prior to installing TACS
import numpy
import mpi4py

# Import distutils
from setuptools import setup
from distutils.core import Extension as Ext
from Cython.Build import cythonize


# Convert from local to absolute directories
def get_global_dir(files):
    tmr_root = os.path.abspath(os.path.dirname(__file__))
    new = []
    for f in files:
        new.append(os.path.join(tmr_root, f))
    return new


def get_mpi_flags():
    # Split the output from the mpicxx command
    args = check_output(["mpicxx", "-show"]).decode("utf-8").split()

    # Determine whether the output is an include/link/lib command
    inc_dirs, lib_dirs, libs = [], [], []
    for flag_ in args:
        try:
            flag = flag_.decode("utf-8")
        except:
            flag = flag_

        if flag[:2] == "-I":
            inc_dirs.append(flag[2:])
        elif flag[:2] == "-L":
            lib_dirs.append(flag[2:])
        elif flag[:2] == "-l":
            libs.append(flag[2:])

    return inc_dirs, lib_dirs, libs


inc_dirs, lib_dirs, libs = get_mpi_flags()

# Relative paths for the include/library directories
rel_inc_dirs = ["src"]
rel_lib_dirs = ["lib"]
libs.extend(["paropt"])

# Convert from relative to absolute directories
inc_dirs.extend(get_global_dir(rel_inc_dirs))
lib_dirs.extend(get_global_dir(rel_lib_dirs))

# Add the numpy/mpi4py directories
inc_dirs.extend([numpy.get_include(), mpi4py.get_include()])

# Add tacs-dev/lib as a runtime directory
runtime_lib_dirs = get_global_dir(["lib"])

exts = []
mod = "ParOpt"
exts.append(
    Ext(
        "paropt.ParOpt",
        sources=["paropt/ParOpt.pyx"],
        language="c++",
        include_dirs=inc_dirs,
        libraries=libs,
        library_dirs=lib_dirs,
        runtime_library_dirs=runtime_lib_dirs,
    )
)
exts.append(
    Ext(
        "paropt.ParOptEig",
        sources=["paropt/ParOptEig.pyx"],
        language="c++",
        include_dirs=inc_dirs,
        libraries=libs,
        library_dirs=lib_dirs,
        runtime_library_dirs=runtime_lib_dirs,
    )
)

for e in exts:
    e.cython_directives = {
        "language_level": "3",
        "embedsignature": True,
        "binding": True,
    }

optional_dependencies = {
    "testing": ["testflo>=1.4.7"],
    "docs": [
        "sphinx",
        "breathe",
        "sphinxcontrib-programoutput",
        "sphinxcontrib-bibtex",
    ],
}

# Add an optional dependency that concatenates all others
optional_dependencies["all"] = sorted(
    [
        dependency
        for dependencies in optional_dependencies.values()
        for dependency in dependencies
    ]
)

setup(
    name="paropt",
    version="2.1.2",
    description="Parallel interior-point optimizer",
    author="Graeme J. Kennedy",
    author_email="graeme.kennedy@ae.gatech.edu",
    install_requires=["numpy", "mpi4py>=3.1.1"],
    extras_require=optional_dependencies,
    ext_modules=cythonize(exts, language="c++", include_path=inc_dirs),
)
