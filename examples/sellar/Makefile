include ../../Makefile.in
include ../../ParOpt_Common.mk

default: Sellar.o
	${CXX} ${CCFLAGS} -o sellar Sellar.o ${PAROPT_LD_FLAGS}

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: default

clean:
	${RM} sellar *.o
