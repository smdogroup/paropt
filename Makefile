# ========================
# Makefile for PAROPT_DIR/
# ========================

include Makefile.in
include ParOpt_Common.mk

PAROPT_SUBDIRS = src

SEARCH_PATTERN=$(addsuffix /*.cpp, ${PAROPT_SUBDIRS})
PAROPT_OBJS := $(patsubst %.cpp,%.o,$(wildcard ${SEARCH_PATTERN}))

default:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE}) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef double ParOptScalar" > paropt/cpp_headers/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_DOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = np.double" >> paropt/ParOptDefs.pxi;

debug:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} debug) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef double ParOptScalar" > paropt/cpp_headers/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_DOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = np.double" >> paropt/ParOptDefs.pxi;

complex:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} complex) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef complex ParOptScalar" > paropt/cpp_headers/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_CDOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = complex" >> paropt/ParOptDefs.pxi;

complex_debug:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} complex_debug) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef complex ParOptScalar" > paropt/cpp_headers/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_CDOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = complex" >> paropt/ParOptDefs.pxi;

interface:
	${PIP} install -e .\[all\]; \

complex_interface:
	CXXFLAGS=-DPAROPT_USE_COMPLEX ${PIP} install -e .\[all\]; \

clean:
	${RM} lib/libparopt.a lib/*.so
	${RM} paropt/*.so paropt/*.cpp
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} $@ ) || exit 1; \
	done
