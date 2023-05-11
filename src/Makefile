include ../Makefile.in
include ../ParOpt_Common.mk

OBJS = ParOptOptions.o \
	ParOptInteriorPoint.o \
	ParOptVec.o \
	ParOptQuasiNewton.o \
	ParOptMMA.o \
	ParOptTrustRegion.o \
	ParOptProblem.o \
	ParOptOptimizer.o \
	ParOptSparseMat.o \
	ParOptCompactEigenvalueApprox.o \
	CyParOptProblem.o

default: ${OBJS}
	${AR} ${AR_FLAGS} ${PAROPT_LIB} ${OBJS}

%.o: %.c
	${CXX} ${CCFLAGS} -c $< -o $@

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: default

complex_debug: CCFLAGS=${CCFLAGS_DEBUG} -DPAROPT_USE_COMPLEX
complex_debug: default

clean:
	rm -rf *.o
