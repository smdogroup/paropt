include ../Makefile.in
include ../ParOpt_Common.mk

#Flags for python compilation
PY_INCLUDE = ${PYTHON_INCLUDE} -I${NUMPY_DIR} -I${MPI4PY_DIR}
CYTHON_INCLUDE = -I${NUMPY_DIR} -I${MPI4PY_DIR}

# ParOpt shared objects
CYTHON_SO = ParOpt.${SO_EXT} 

default: real_defs ${CYTHON_SO}

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: complex_defs ${CYTHON_SO}

debug: PAROPT_CC_FLAGS=${PAROPT_DEBUG_CC_FLAGS}
debug: default

complex_defs:
	@echo "ctypedef complex ParOptScalar" > ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_CDOUBLE" > ParOptDefs.pxi;
	@echo "dtype = np.complex" >> ParOptDefs.pxi;

real_defs:
	@echo "ctypedef double ParOptScalar" > ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_DOUBLE" > ParOptDefs.pxi;
	@echo "dtype = np.double" >> ParOptDefs.pxi;

%.cpp:%.pyx
	${CYTHON} --cplus $*.pyx

%.o:%.cpp
	${CXX} ${PAROPT_CC_FLAGS} ${PY_INCLUDE} -c $< -o $@

ParOpt.${SO_EXT}:ParOpt.o CyParOptProblem.o
	${CXX} ${SO_LINK_FLAGS} $< CyParOptProblem.o ${PAROPT_LD_FLAGS} -o $@

clean:
	${RM} *.cpp
	${RM} *.${SO_EXT}
	${RM} *.o
