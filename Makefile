# ========================
# Makefile for PAROPT_DIR/
# ========================

include Makefile.in
include ParOpt_Common.mk

PAROPT_SUBDIRS = src

SEARCH_PATTERN=$(addsuffix /*.c, ${PAROPT_SUBDIRS})
PAROPT_OBJS := $(patsubst %.c,%.o,$(wildcard ${SEARCH_PATTERN}))

default:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE}) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef double ParOptScalar" > paropt/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_DOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = np.double" >> paropt/ParOptDefs.pxi;

debug:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} debug) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef double ParOptScalar" > paropt/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_DOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = np.double" >> paropt/ParOptDefs.pxi;

complex:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} complex) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} ${PAROPT_OBJS} ${PAROPT_EXTERN_LIBS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT}
	@echo "ctypedef complex ParOptScalar" > paropt/ParOptTypedefs.pxi;
	@echo "PAROPT_NPY_SCALAR = np.NPY_CDOUBLE" > paropt/ParOptDefs.pxi;
	@echo "dtype = np.complex" >> paropt/ParOptDefs.pxi;

interface:
	python setup.py build_ext --inplace

clean:
	${RM} lib/libparopt.a lib/*.so
	${RM} paropt/*.so paropt/*.cpp
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} $@ ) || exit 1; \
	done
