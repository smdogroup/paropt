include ../Makefile.in

OBJS = ParOpt.o \
	ParOptVec.o

default: ${OBJS}

%.o: %.c
	${CXX} ${CCFLAGS} -c $< -o $@

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

clean:
	rm -rf *.o