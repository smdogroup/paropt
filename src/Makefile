include ../Makefile.in

default:
	${CXX} ${CCFLAGS} -c ParOptVec.c -o ParOptVec.o
	${CXX} ${CCFLAGS} -c ParOpt.c -o ParOpt.o

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default
