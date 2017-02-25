
# Set the linking files
PAROPT_INCLUDE = -I${PAROPT_DIR}/src
PAROPT_LIB = ${PAROPT_DIR}/lib/libparopt.a

# Set the optimized/debug compile flags
PAROPT_OPT_CC_FLAGS = ${CCFLAGS} ${PAROPT_INCLUDE}
PAROPT_DEBUG_CC_FLAGS = ${CCFLAGS_DEBUG} ${PAROPT_INCLUDE}

# Set the optimized flags to the default
PAROPT_CC_FLAGS = ${PAROPT_OPT_CC_FLAGS}

# Set the linking flags
PAROPT_EXTERN_LIBS = ${LAPACK_LIBS}
PAROPT_LD_FLAGS = ${PAROPT_LD_CMD} ${PAROPT_EXTERN_LIBS}

# This is the one rule that is used to compile all the
# source code in TACS
%.o: %.c
	${CXX} ${PAROPT_CC_FLAGS} -c $< -o $*.o
	@echo
	@echo "        --- Compiled $*.c successfully ---"
	@echo
