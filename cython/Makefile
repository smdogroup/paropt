include ../Makefile.in
include ../ParOpt_Common.mk

#Flags for python compilation
PY_INCLUDE = ${PYTHON_INCLUDE} -I${NUMPY_DIR} -I${MPI4PY_DIR}
CYTHON_INCLUDE = -I${NUMPY_DIR} -I${MPI4PY_DIR}

# ParOpt shared objects
CYTHON_SO = ParOpt_c.${SO_EXT} 

default: ${CYTHON_SO}

debug: PAROPT_CC_FLAGS=${PAROPT_DEBUG_CC_FLAGS}
debug: default

%.cpp:%.pyx
	${CYTHON} --cplus $*.pyx

%.o:%.cpp
	${CXX} ${PAROPT_CC_FLAGS} ${PY_INCLUDE} -c $< -o $@

%.${SO_EXT}:%.o
	${CXX} ${SO_LINK_FLAGS} $< ${PAROPT_LD_FLAGS} -o $@

clean:
	${RM} *.cpp
	${RM} *.${SO_EXT}
	${RM} *.o
