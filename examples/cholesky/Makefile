include ../../Makefile.in
include ../../ParOpt_Common.mk

default: cholesky.o
	${CXX} ${CCFLAGS} -o cholesky cholesky.o ${PAROPT_LD_FLAGS}

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS=${CCFLAGS_DEBUG} -DPAROPT_USE_COMPLEX
complex: default

clean:
	${RM} cholesky cholesky *.o
