#Self contained makefile

CXX = mpicxx
RM = rm -f

PAROPT_DIR = ${HOME}/hg/paropt

PAROPT_INCLUDE = -I${PAROPT_DIR}/src

#Set up for parallel make
MAKE = make -j 8

#Flags for debugging and regular compilation versions
EXTRA_DEBUG_CC_FLAGS = -fPIC -g
EXTRA_CC_FLAGS = -fPIC -O3

SO_EXT = so
SO_LINK_FLAGS = -fPIC -shared

CYTHON = cython

#Adding default lapack libraries
#LAPACK_LIBS = -L/usr/lib/ -llapack -lblas -lpthread -lmpi_cxx
LAPACK_LIBS = -L/usr/lib -llapack -lblas -lmpi_cxx

# If using the python interface you'll need to include python and numpy 

PYTHON_INCLUDE = ${shell python-config --includes}
NUMPY_DIR=${shell python -c "import numpy; print numpy.get_include()"}

# If you're going to use the python interface you'll also have to have
# a working version of mpi4py

MPI4PY_DIR=${shell python -c "import mpi4py; print mpi4py.get_include()"}

#Flags for python compilation
PY_INCLUDE = ${PYTHON_INCLUDE} -I${NUMPY_DIR} -I${MPI4PY_DIR}
CYTHON_INCLUDE = -I${NUMPY_DIR} -I${MPI4PY_DIR}

#Set command line flags for use in compilation
PAROPT_OPT_CC_FLAGS = ${EXTRA_CC_FLAGS} ${PAROPT_INCLUDE}
PAROPT_DEBUG_CC_FLAGS = ${EXTRA_DEBUG_CC_FLAGS} ${PAROPT_INCLUDE}

#Set linking flags to use
PAROPT_CC_FLAGS = ${PAROPT_OPT_CC_FLAGS} 
ROSEN_LD_FLAGS = ${LAPACK_LIBS}

CYTHON_SO = ParOpt.o \
	ParOptVec.o \
	Rosenbrock.o \
	ParOptVec_c.${SO_EXT} \
	ParOpt_c.${SO_EXT} 

OBJS = ParOptVec.o

POBJS = ParOpt.o\
	ParOptVec.o\
	Rosenbrock.o

DIR = ${PAROPT_DIR}/src

real: ${CYTHON_SO}

default:
	${MAKE} real

%.cpp:%.pyx
	${CYTHON} --cplus $*.pyx

%.o: %.c
	${CXX} ${PAROPT_CC_FLAGS} ${PY_INCLUDE} -c $< -o $*.o

%.o: %.cpp
	${CXX} ${PAROPT_CC_FLAGS} ${PY_INCLUDE} -c $< -o $@

ParOptVec_c.${SO_EXT}: ParOptVec_c.o
	${CXX} ${SO_LINK_FLAGS} -o $@ ${OBJS} ParOptVec_c.o ${ROSEN_LD_FLAGS}

ParOpt_c.${SO_EXT}: ParOpt_c.o
	${CXX} ${SO_LINK_FLAGS} -o $@ ${POBJS} ParOpt_c.o ${ROSEN_LD_FLAGS}

clean:
	${RM} *.cpp
	${RM} *.${SO_EXT}
	${RM} *.o
