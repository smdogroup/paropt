PAROPT_DIR=${HOME}/git/paropt
include ${PAROPT_DIR}/Makefile.in
include ${PAROPT_DIR}/ParOpt_Common.mk

# If using the python interface you'll need to include python and numpy 
PYTHON_INCLUDE = ${shell python-config --includes}
NUMPY_DIR=${shell python -c "from __future__ import print_function; import numpy; print(numpy.get_include())"}

# If you're going to use the python interface you'll also have to have
# a working version of mpi4py
MPI4PY_DIR=${shell python -c "from __future__ import print_function; import mpi4py; print(mpi4py.get_include())"}

#Flags for python compilation
PY_INCLUDE = ${PYTHON_INCLUDE} -I${NUMPY_DIR} -I${MPI4PY_DIR}
CYTHON_INCLUDE = -I${NUMPY_DIR} -I${MPI4PY_DIR}

# ParOpt shared objects
CYTHON_SO = ParOptEig.so

default: ParOptCompactEigenvalueApprox.o ${CYTHON_SO}

debug: PAROPT_CC_FLAGS=${PAROPT_DEBUG_CC_FLAGS}
debug: default

%.cpp:%.pyx
	cython -I${PAROPT_DIR}/paropt/ --cplus $*.pyx --include-dir ${PAROPT_DIR}/paropt/

%.o:%.cpp
	${CXX} -I${PAROPT_DIR}/paropt/ ${PAROPT_CC_FLAGS} ${PY_INCLUDE} -c $< -o $@

ParOptEig.so: ParOptEig.o ParOptCompactEigenvalueApprox.o
	${CXX} ${SO_LINK_FLAGS} ParOptEig.o ParOptCompactEigenvalueApprox.o ${PAROPT_LD_FLAGS} -o $@

clean:
	${RM} *.so *.o 