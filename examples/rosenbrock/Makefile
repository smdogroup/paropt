include ../../Makefile.in
include ../../ParOpt_Common.mk

default: Rosenbrock.o SCA.o
	${CXX} ${CCFLAGS} -o rosenbrock Rosenbrock.o ${PAROPT_LD_FLAGS}
	${CXX} ${CCFLAGS} -o sca SCA.o ${PAROPT_LD_FLAGS}

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: default

clean:
	${RM} rosenbrock *.o
