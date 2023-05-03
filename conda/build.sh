export PAROPT_DIR=${SRC_DIR}

if [[ $(uname) == Darwin ]]; then
  export SO_EXT="dylib"
  export SO_LINK_FLAGS="-fPIC -dynamiclib"
  export LIB_SLF="${SO_LINK_FLAGS} -install_name @rpath/libparopt.dylib"
  export LAPACK_LIBS="-framework accelerate"
elif [[ "$target_platform" == linux-* ]]; then
  export SO_EXT="so"
  export SO_LINK_FLAGS="-fPIC -shared"
  export LIB_SLF="${SO_LINK_FLAGS}"
  export LAPACK_LIBS="-L${PREFIX}/lib/ -llapack -lpthread -lblas"
fi

if [[ $scalar == "complex" ]]; then
  export OPTIONAL="complex"
  export PIP_FLAGS="-DPAROPT_USE_COMPLEX"
elif [[ $scalar == "real" ]]; then
  export OPTIONAL="default"
fi

cp Makefile.in.info Makefile.in;
make ${OPTIONAL} PAROPT_DIR=${PAROPT_DIR} \
     LAPACK_LIBS="${LAPACK_LIBS}" \
     SO_LINK_FLAGS="${LIB_SLF}" SO_EXT=${SO_EXT};
mv ${PAROPT_DIR}/lib/libparopt.${SO_EXT} ${PREFIX}/lib;

# Recursively copy all header files
mkdir ${PREFIX}/include/paropt;
find ${PAROPT_DIR}/src/ -name '*.h' -exec cp -prv '{}' ${PREFIX}/include/paropt ';'

CFLAGS=${PIP_FLAGS} ${PYTHON} -m pip install --no-deps --prefix=${PREFIX} . -vv;
