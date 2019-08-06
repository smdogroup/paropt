include ../Makefile.in
include ../ParOpt_Common.mk

OBJS = ParOptInteriorPoint.o \
	ParOptVec.o \
	ParOptQuasiNewton.o \
	ParOptMMA.o \
	ParOptTrustRegion.o \
	CyParOptProblem.o

default: ${OBJS}
	${AR} ${AR_FLAGS} ${PAROPT_LIB} ${OBJS}

%.o: %.c
	${CXX} ${CCFLAGS} -c $< -o $@

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: default

clean:
	rm -rf *.o
