include ../Makefile.in

default:
	${CXX} ${CCFLAGS} -I../src/ -c Rosenbrock.c -o Rosenbrock.o
	${CXX} ${CCFLAGS} -o rosenbrock Rosenbrock.o ../src/ParOpt.o ../src/ParOptVec.o ${LAPACK_LIBS}

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: default
