TACS_DIR=${HOME}/hg/tacs
include ${TACS_DIR}/Makefile.in
include ${TACS_DIR}/TACS_Common.mk

#Flags for python compilation
PY_INCLUDE = ${PYTHON_INCLUDE} -I${NUMPY_DIR} -I${MPI4PY_DIR}
CYTHON_INCLUDE = -I${NUMPY_DIR} -I${MPI4PY_DIR}

# ParOpt shared objects
CYTHON_SO = multitopo.so

default: PSMultiTopo.o ${CYTHON_SO}

debug: TACS_CC_FLAGS=${TACS_DEBUG_CC_FLAGS}
debug: default

%.cpp:%.pyx
	cython -I${TACS_DIR}/tacs/ --cplus $*.pyx --include-dir ${TACS_DIR}/tacs/

%.o:%.cpp
	${CXX} -I${TACS_DIR}/tacs/ ${TACS_CC_FLAGS} ${PY_INCLUDE} -c $< -o $@

multitopo.so: multitopo.o PSMultiTopo.o
	${CXX} ${SO_LINK_FLAGS} multitopo.o PSMultiTopo.o ${TACS_LD_FLAGS} -o $@

clean:
	${RM} *.so *.o 
