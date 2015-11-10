include ../Makefile.in

OBJS = ParOpt.o \
	ParOptVec.o

default: ${OBJS}

%.o: %.c
	${CXX} ${CCFLAGS} -c $< -o $@

debug: CCFLAGS=${CCFLAGS_DEBUG}
debug: default

complex: CCFLAGS+=-DPAROPT_USE_COMPLEX
complex: default

clean:
	rm -rf *.o
