include ../Makefile.in

default:
	${CXX} ${CCFLAGS} -I../src/ -c Rosenbrock.c -o Rosenbrock.o
	${CXX} ${CCFLAGS} -o rosenbrock Rosenbrock.o ../src/ParOpt.o ../src/ParOptVec.o ${LAPACK_LIBS}
