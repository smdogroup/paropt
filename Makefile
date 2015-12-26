# ========================
# Makefile for PAROPT_DIR/
# ========================

include Makefile.in
include ParOpt_Common.mk

PAROPT_SUBDIRS = src

SEARCH_PATTERN=$(addsuffix /*.c, ${PAROPT_SUBDIRS})
PAROPT_OBJS := $(patsubst %.c,%.o,$(wildcard ${SEARCH_PATTERN}))

default:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE}) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT} ${PAROPT_OBJS}

debug:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} debug) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT} ${PAROPT_OBJS}

complex:
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} complex) || exit 1; \
	done
	${CXX} ${SO_LINK_FLAGS} -o ${PAROPT_DIR}/lib/libparopt.${SO_EXT} ${PAROPT_OBJS}

clean:
	${RM} lib/libparopt.a lib/*.so
	@for subdir in ${PAROPT_SUBDIRS}; do \
	   echo; (cd $$subdir && ${MAKE} $@ ) || exit 1; \
	done
