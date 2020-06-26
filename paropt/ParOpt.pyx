#distuils: language = c++
#distuils: sources = ParOpt.c
from __future__ import print_function, division

# For the use of MPI
from mpi4py.MPI cimport *
cimport mpi4py.MPI as MPI

# Import the python version information
from cpython.version cimport PY_MAJOR_VERSION

# Import the string library
from libcpp.string cimport string

# Import the declarations required from the pxd file
from ParOpt cimport *

# Import tracebacks for callbacks
import traceback

# Import numpy
import numpy as np
cimport numpy as np

# Ensure that numpy is initialized
np.import_array()

# Import C methods for python
from cpython cimport PyObject, Py_INCREF
from libc.stdlib cimport malloc, free

# Include the definitions
include "ParOptDefs.pxi"

# Include the mpi4py header
cdef extern from "mpi-compat.h":
    pass

cdef char* convert_to_chars(s):
    if isinstance(s, unicode):
        s = (<unicode>s).encode('utf8')
    return s

cdef str convert_char_to_str(const char* s):
    if s == NULL:
        return None
    elif PY_MAJOR_VERSION >= 3:
        return s.decode('utf-8')
    else:
        return str(s)

# Set the update type
SKIP_NEGATIVE_CURVATURE = PAROPT_SKIP_NEGATIVE_CURVATURE
DAMPED_UPDATE = PAROPT_DAMPED_UPDATE

def unpack_output(filename):
    """
    Unpack the parameters from the paropt output file and return them
    in a list of numpy arrays. This also returns a small string
    description from the file itself. This code relies ont he
    fixed-width nature of the file, which is guaranteed.
    """

    # The arguments that we're looking for
    args = ['iter', 'nobj', 'ngrd', 'nhvc', 'alpha', 'alphx', 'alphz',
            'fobj', '|opt|', '|infes|', '|dual|', 'mu', 'comp', 'dmerit',
            'rho']
    fmt = '4d 4d 4d 4d 7e 7e 7e 12e 7e 7e 7e 7e 7e 8e 7e'.split()

    # Loop over the file until the end
    content = []
    for f in fmt:
        content.append([])

    # Read the entire
    with open(filename, 'r') as fp:
        lines = fp.readlines()

        index = 0
        while index < len(lines):
            fargs = lines[index].split()
            if (len(fargs) > 2 and
                 (fargs[0] == args[0] and fargs[1] == args[1])):
                index += 1

                # Read at most 10 lines before searching for the next
                # header
                counter = 0
                while counter < 10 and index < len(lines):
                    line = lines[index]
                    index += 1
                    counter += 1
                    if len(line.split()) < len(args):
                        break

                    # Scan through the format list and determine how to
                    # convert the object based on the format string
                    off = 0
                    idx = 0
                    for f in fmt:
                        next = int(f[:-1])
                        s = line[off:off+next]
                        off += next+1

                        if f[-1] == 'd':
                            try:
                                content[idx].append(int(s))
                            except:
                                content[idx].append(0)
                        elif f[-1] == 'e':
                            try:
                                content[idx].append(float(s))
                            except:
                                content[idx].append(0.0)
                        idx += 1

            # Increase the index by one
            index += 1

    # Convert the lists to numpy arrays
    objs = []
    for idx in range(len(args)):
        if fmt[idx][1] == 'd':
            objs.append(np.array(content[idx], dtype=np.int))
        else:
            objs.append(np.array(content[idx]))

    return args, objs

def unpack_tr_output(filename):
    """
    Unpack the parameters from the paropt output file and return them
    in a list of numpy arrays. This also returns a small string
    description from the file itself. This code relies ont he
    fixed-width nature of the file, which is guaranteed.
    """

    # The arguments that we're looking for
    args = ['iter', 'fobj', 'infes', 'l1', 'linfty', '|x - xk|', 'tr',
            'rho', 'mod red.', 'avg z', 'max z', 'avg pen.', 'max pen.']
    fmt = '5d 12e 9e 9e 9e 9e 9e 9e 9e 9e 9e 9e 9e'.split()

    # Loop over the file until the end
    content = []
    for f in fmt:
        content.append([])

    # Read the entire
    with open(filename, 'r') as fp:
        lines = fp.readlines()

        index = 0
        while index < len(lines):
            fargs = lines[index].split()
            if (len(fargs) > 2 and
                 (fargs[0] == args[0] and fargs[1] == args[1])):
                index += 1

                # Read at most 10 lines before searching for the next
                # header
                counter = 0
                while counter < 10 and index < len(lines):
                    line = lines[index]
                    index += 1
                    counter += 1
                    if len(line.split()) < len(args):
                        break

                    # Scan through the format list and determine how to
                    # convert the object based on the format string
                    off = 0
                    idx = 0
                    for f in fmt:
                        next = int(f[:-1])
                        s = line[off:off+next]
                        off += next+1

                        if f[-1] == 'd':
                            try:
                                content[idx].append(int(s))
                            except:
                                content[idx].append(0)
                        elif f[-1] == 'e':
                            try:
                                content[idx].append(float(s))
                            except:
                                content[idx].append(0.0)
                        idx += 1

            # Increase the index by one
            index += 1

    # Convert the lists to numpy arrays
    objs = []
    for idx in range(len(args)):
        if fmt[idx][1] == 'd':
            objs.append(np.array(content[idx], dtype=np.int))
        else:
            objs.append(np.array(content[idx]))

    return args, objs

def unpack_mma_output(filename):
    """
    Unpack the parameters from a file output from MMA
    """

    args = ['MMA', 'sub-iter', 'fobj', 'l1-opt',
              'linft-opt', 'l1-lambd', 'infeas']
    fmt = ['5d', '8d', '15e', '9e', '9e', '9e', '9e']

    # Loop over the file until the end
    content = []
    for f in fmt:
        content.append([])

    # Read the entire
    with open(filename, 'r') as fp:
        lines = fp.readlines()

        index = 0
        while index < len(lines):
            fargs = lines[index].split()
            if (len(fargs) > 2 and
                 (fargs[0] == args[0] and fargs[1] == args[1])):
                index += 1

                # Read at most 10 lines before searching for the next
                # header
                counter = 0
                while counter < 10 and index < len(lines):
                    line = lines[index]
                    index += 1
                    counter += 1
                    if len(line.split()) < len(args)-2:
                        break

                    # Scan through the format list and determine how to
                    # convert the object based on the format string
                    off = 0
                    idx = 0
                    for f in fmt:
                        next = int(f[:-1])
                        s = line[off:off+next]
                        off += next+1

                        if f[-1] == 'd':
                            try:
                                content[idx].append(int(s))
                            except:
                                content[idx].append(0)
                        elif f[-1] == 'e':
                            try:
                                content[idx].append(float(s))
                            except:
                                content[idx].append(0.0)
                        idx += 1

            # Increase the index by one
            index += 1

    # Convert the lists to numpy arrays
    objs = []
    for idx in range(len(args)):
        if fmt[idx][1] == 'd':
            objs.append(np.array(content[idx], dtype=np.int))
        else:
            objs.append(np.array(content[idx]))

    return args, objs

# Read in a ParOpt checkpoint file and produce python variables
def unpack_checkpoint(filename):
    """Convert the checkpoint file to usable python objects"""

    # Open the file in read-only binary mode
    fp = open(filename, 'rb')
    sfp = fp.read()

    # Get the sizes of c integers and doubles
    ib = np.dtype(np.intc).itemsize
    fb = np.dtype(np.double).itemsize

    # Convert the sizes stored in the checkpoint file
    sizes = np.fromstring(sfp[:3*ib], dtype=np.intc)
    nvars = sizes[0]
    nwcon = sizes[1]
    ncon = sizes[2]

    # Skip first three integers and the barrier parameter value
    offset = 3*ib
    barrier = np.fromstring(sfp[offset:offset+fb], dtype=np.double)[0]
    offset += fb

    # Convert the slack variables and multipliers
    s = np.fromstring(sfp[offset:offset+fb*ncon], dtype=np.double)
    offset += fb*ncon
    z = np.fromstring(sfp[offset:offset+fb*ncon], dtype=np.double)
    offset += fb*ncon

    # Convert the variables and multipliers
    x = np.fromstring(sfp[offset:offset+fb*nvars], dtype=np.double)
    offset += fb*nvars
    zl = np.fromstring(sfp[offset:offset+fb*nvars], dtype=np.double)
    offset += fb*nvars
    zu = np.fromstring(sfp[offset:offset+fb*nvars], dtype=np.double)
    offset += fb*nvars

    return barrier, s, z, x, zl, zu

# This wraps a C++ array with a numpy array for later useage
cdef inplace_array_1d(int nptype, int dim1, void *data_ptr,
                             object base=None):
    """Return a numpy version of the array"""
    # Set the shape of the array
    cdef int size = 1
    cdef np.npy_intp shape[1]
    cdef np.ndarray ndarray

    # Set the first entry of the shape array
    shape[0] = <np.npy_intp>dim1

    # Create the array itself - Note that this function will not
    # delete the data once the ndarray goes out of scope
    ndarray = np.PyArray_SimpleNewFromData(size, shape,
                                           nptype, data_ptr)

    if base is not None:
        Py_INCREF(base)
        ndarray.base = <PyObject*>base

    return ndarray

cdef int addDictionaryToOptions(options,
                                ParOptOptions *opts) except -1:
    cdef int int_value = 0
    cdef double float_value = 0
    cdef string str_value
    cdef string key_value

    # Set the options from the dictionary
    for key in options:
        key_value = convert_to_chars(key)
        if opts.isOption(key_value.c_str()):
            value = options[key]
            if value is None:
                continue
            elif isinstance(value, bool):
                int_value = 0
                if value:
                    int_value = 1
                opts.setOption(key_value.c_str(), int_value)
            elif isinstance(value, int):
                int_value = value
                opts.setOption(key_value.c_str(), int_value)
            elif isinstance(value, float):
                float_value = value
                opts.setOption(key_value.c_str(), float_value)
            elif isinstance(value, str):
                str_value = convert_to_chars(value)
                opts.setOption(key_value.c_str(), str_value.c_str())
            else:
                errmsg = 'Cannot convert option %s value '%(str(key)) + str(value)
                errmsg += ' to ParOptOptions type'
                raise ValueError(errmsg)
        else:
            errmsg = 'Unknown ParOpt option %s'%(str(key))
            raise ValueError(errmsg)

    return 0

def printOptionSummary():
    """
    Print a summary of all the options available within all ParOpt optimizers.
    """

    info = getOptionsInfo()

    for name in info:
        print(info[name].descript)
        if info[name].option_type == 'str':
            print('%-40s %-15s'%(name, info[name].default))
        elif info[name].option_type == 'bool':
            print('%-40s %-15s'%(name, str(info[name].default)))
        elif info[name].option_type == 'int':
            print('%-40s %-15d'%(name, info[name].default))
            print('Range of values: lower limit %d  upper limit %d'%(
                info[name].values[0], info[name].values[1]))
        elif info[name].option_type == 'float':
            print('%-40s %-15g'%(name, info[name].default))
            print('Range of values: lower limit %g  upper limit %g'%(
                info[name].values[0], info[name].values[1]))
        elif info[name].option_type == 'enum':
            print('%-40s %-15s'%(name, info[name].default))
            print('%-40s %-15s'%('Range of values:', info[name].values[0]))
            for opt in info[name].values[1:]:
                print('%-40s %-15s'%(' ', opt))
        print('')

    return

def getOptionsInfo():
    """
    Get a dictionary that contains all of the option values and information
    that are used in any of the ParOpt optimizers.
    """
    cdef int index
    cdef int size
    cdef const char *name = NULL
    cdef const char *const *enum_values
    cdef int int_low, int_high
    cdef double float_low, float_high
    cdef ParOptOptions *options = NULL

    # Create the options object and populate it with information
    options = new ParOptOptions()
    options.incref()
    ParOptOptimizerAddDefaultOptions(options)

    # Set the type integer to string representation
    type_names = {1:'str', 2:'bool', 3:'int', 4:'float', 5:'enum'}

    class OptionInfo:
        def __init__(self, option_type='', default=None,
                     values=None, descript=''):
            self.option_type = option_type
            self.default = default
            self.values = values
            self.descript = descript
            return

    # Set
    opts = {}
    options.begin()
    while True:
        name = options.getName()
        index = options.getOptionType(name)
        descript = convert_char_to_str(options.getDescription(name))
        option_type = type_names[index]
        default = None
        values = None
        if index == 1: # str
            default= convert_char_to_str(options.getStringOption(name))
        elif index == 2: # bool
            default = options.getBoolOption(name)
            if default:
                default = True
            else:
                default = False
        elif index == 3: # int
            default = options.getIntOption(name)
            options.getIntRange(name, &int_low, &int_high)
            values = [int_low, int_high]
        elif index == 4: # float
            default = options.getFloatOption(name)
            options.getFloatRange(name, &float_low, &float_high)
            values = [float_low, float_high]
        elif index == 5: # enum
            default= convert_char_to_str(options.getEnumOption(name))
            options.getEnumRange(name, &size, &enum_values)
            values = []
            for i in range(size):
                val = convert_char_to_str(enum_values[i])
                values.append(val)

        str_name = convert_char_to_str(name)
        opts[str_name] = OptionInfo(option_type=option_type, default=default,
                                    values=values, descript=descript)

        if not options.next():
            break

    return opts

cdef void _getvarsandbounds(void *_self, int nvars,
                            ParOptVec *_x, ParOptVec *_lb,
                            ParOptVec *_ub):
    try:
        x = _init_PVec(_x)
        lb = _init_PVec(_lb)
        ub = _init_PVec(_ub)
        (<object>_self).getVarsAndBounds(x, lb, ub)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return

cdef int _evalobjcon(void *_self, int nvars, int ncon,
                     ParOptVec *_x, ParOptScalar *fobj,
                     ParOptScalar *cons):
    fail = 0

    try:
        # Call the objective function
        x = _init_PVec(_x)
        fail, _fobj, _cons = (<object>_self).evalObjCon(x)

        # Copy over the objective value
        fobj[0] = _fobj

        # Copy the values from the numpy arrays
        for i in range(ncon):
            cons[i] = _cons[i]
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return fail

cdef int _evalobjcongradient(void *_self, int nvars, int ncon,
                             ParOptVec *_x, ParOptVec *_g,
                             ParOptVec **A):
    fail = 0
    try:
        # The numpy arrays that will be used for x
        x = _init_PVec(_x)
        g = _init_PVec(_g)
        Ac = []
        for i in range(ncon):
            Ac.append(_init_PVec(A[i]))

        # Call the objective function
        fail = (<object>_self).evalObjConGradient(x, g, Ac)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return fail

cdef int _evalhvecproduct(void *_self, int nvars, int ncon, int nwcon,
                          ParOptVec *_x, ParOptScalar *_z, ParOptVec *_zw,
                          ParOptVec *_px, ParOptVec *_hvec):
    fail = 0
    try:
        x = _init_PVec(_x)
        zw = None
        if _zw != NULL:
            zw = _init_PVec(_zw)
        px = _init_PVec(_px)
        hvec = _init_PVec(_hvec)

        z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z)

        # Call the objective function
        fail = (<object>_self).evalHvecProduct(x, z, zw, px, hvec)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return fail

cdef int _evalhessiandiag(void *_self, int nvars, int ncon, int nwcon,
                          ParOptVec *_x, ParOptScalar *_z, ParOptVec *_zw,
                          ParOptVec *_hdiag):
    fail = 0
    try:
        x = _init_PVec(_x)
        zw = None
        if _zw != NULL:
            zw = _init_PVec(_zw)
        hdiag = _init_PVec(_hdiag)

        z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z)

        # Call the objective function
        fail = (<object>_self).evalHessianDiag(x, z, zw, hdiag)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return fail

cdef void _computequasinewtonupdatecorrection(void *_self, int nvars,
                                              ParOptVec *_s, ParOptVec *_y):
    try:
        # Call the objective function
        if hasattr(<object>_self, 'computeQuasiNewtonUpdateCorrection'):
            s = _init_PVec(_s)
            y = _init_PVec(_y)
            (<object>_self).computeQuasiNewtonUpdateCorrection(s, y)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return

cdef void _evalsparsecon(void *_self, int nvars, int nwcon,
                         ParOptVec *_x, ParOptVec *_con):
    try:
        x = _init_PVec(_x)
        con = _init_PVec(_con)

        (<object>_self).evalSparseCon(x, con)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return

cdef void _addsparsejacobian(void *_self, int nvars,
                             int nwcon, ParOptScalar alpha,
                             ParOptVec *_x, ParOptVec *_px,
                             ParOptVec *_con):
    try:
        x = _init_PVec(_x)
        px = _init_PVec(_px)
        con = _init_PVec(_con)

        (<object>_self).addSparseJacobian(alpha, x, px, con)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return

cdef void _addsparsejacobiantranspose(void *_self, int nvars,
                                      int nwcon, ParOptScalar alpha,
                                      ParOptVec *_x, ParOptVec *_pzw,
                                      ParOptVec *_out):
    try:
        x = _init_PVec(_x)
        pzw = _init_PVec(_pzw)
        out = _init_PVec(_out)
        (<object>_self).addSparseJacobianTranspose(alpha, x, pzw, out)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return

cdef void _addsparseinnerproduct(void *_self, int nvars,
                                 int nwcon, int nwblock, ParOptScalar alpha,
                                 ParOptVec *_x, ParOptVec *_c,
                                 ParOptScalar *_A):
    try:
        x = _init_PVec(_x)
        c = _init_PVec(_c)
        A = inplace_array_1d(PAROPT_NPY_SCALAR, nwcon*nwblock*nwblock,
                             <void*>_A)

        (<object>_self).addSparseInnerProduct(alpha, x, c, A)
    except:
        tb = traceback.format_exc()
        print(tb)
        exit(0)

    return

cdef class ProblemBase:
    def __cinit__(self):
        self.ptr = NULL

    def createDesignVec(self):
        cdef ParOptVec *vec = NULL
        vec = self.ptr.createDesignVec()
        return _init_PVec(vec)

    def createConstraintVec(self):
        cdef ParOptVec *vec = NULL
        vec = self.ptr.createConstraintVec()
        return _init_PVec(vec)

    def checkGradients(self, double dh=1e-6, PVec x=None,
                       check_hvec_product=False):
        cdef ParOptVec *vec = NULL
        cdef int check_hvec = 0
        if x is not None:
            vec = x.ptr
        if check_hvec_product:
            check_hvec = 1
        if self.ptr != NULL:
            self.ptr.checkGradients(dh, vec, check_hvec)
        return

cdef class Problem(ProblemBase):
    cdef CyParOptProblem *me
    def __init__(self, MPI.Comm comm, int nvars, int ncon, int nineq=-1,
                 int nwcon=0, int nwblock=0):
        cdef MPI_Comm c_comm = comm.ob_mpi
        if nineq < 0:
            nineq = ncon
        self.me = new CyParOptProblem(c_comm, nvars, ncon, nineq, nwcon, nwblock)
        self.me.setSelfPointer(<void*>self)
        self.me.setGetVarsAndBounds(_getvarsandbounds)
        self.me.setEvalObjCon(_evalobjcon)
        self.me.setEvalObjConGradient(_evalobjcongradient)
        self.me.setEvalHvecProduct(_evalhvecproduct)
        self.me.setEvalHessianDiag(_evalhessiandiag)
        self.me.setComputeQuasiNewtonUpdateCorrection(_computequasinewtonupdatecorrection)
        self.me.setEvalSparseCon(_evalsparsecon)
        self.me.setAddSparseJacobian(_addsparsejacobian)
        self.me.setAddSparseJacobianTranspose(_addsparsejacobiantranspose)
        self.me.setAddSparseInnerProduct(_addsparseinnerproduct)
        self.ptr = self.me
        self.ptr.incref()
        return

    def __dealloc__(self):
        if self.ptr:
            self.ptr.decref()
        return

    def setInequalityOptions(self, sparse_ineq=True,
                             use_lower=True, use_upper=True):
        # Assume that everything is false
        cdef int sparse = 0
        cdef int lower = 0
        cdef int upper = 0

        # Swap the integer values if the flags are set
        if sparse_ineq: sparse = 1
        if use_lower: lower = 1
        if use_upper: upper = 1

        # Set the options
        self.me.setInequalityOptions(sparse, lower, upper)

        return

cdef class PVec:
    def __cinit__(self):
        self.ptr = NULL
        return

    def __dealloc__(self):
        if self.ptr:
            self.ptr.decref()

    def __len__(self):
        cdef int size = 0
        size = self.ptr.getArray(NULL)
        return size

    def __add__(self, b):
        return self[:] + b

    def __sub__(self, b):
        return self[:] - b

    def __mul__(self, b):
        return self[:]*b

    def __radd__(self, b):
        return b + self[:]

    def __rsub__(self, b):
        return b - self[:]

    def __rmul__(self, b):
        return b * self[:]

    def __truediv__(self, b):
        return self[:]/b

    def __iadd__(self, b):
        cdef int size = 0
        cdef int bsize = 0
        cdef ParOptScalar *array = NULL
        cdef ParOptScalar *barray = NULL
        cdef ParOptScalar value = 0.0
        cdef ParOptVec *bptr = NULL
        size = self.ptr.getArray(&array)
        if isinstance(b, PVec):
            bptr = (<PVec>b).ptr
            bsize = bptr.getArray(&barray)
            if bsize == size:
                for i in range(size):
                    array[i] += barray[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        elif hasattr(b, '__len__'):
            bsize = len(b)
            if bsize == size:
                for i in range(size):
                    array[i] += b[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        else:
            value = b
            for i in range(size):
                array[i] += value
        return self

    def __isub__(self, b):
        cdef int size = 0
        cdef int bsize = 0
        cdef ParOptScalar *array = NULL
        cdef ParOptScalar *barray = NULL
        cdef ParOptScalar value = 0.0
        cdef ParOptVec *bptr = NULL
        size = self.ptr.getArray(&array)
        if isinstance(b, PVec):
            bptr = (<PVec>b).ptr
            bsize = bptr.getArray(&barray)
            if bsize == size:
                for i in range(size):
                    array[i] -= barray[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        elif hasattr(b, '__len__'):
            bsize = len(b)
            if bsize == size:
                for i in range(size):
                    array[i] -= b[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        else:
            value = b
            for i in range(size):
                array[i] -= value
        return self

    def __imul__(self, b):
        cdef int size = 0
        cdef int bsize = 0
        cdef ParOptScalar *array = NULL
        cdef ParOptScalar *barray = NULL
        cdef ParOptScalar value = 0.0
        cdef ParOptVec *bptr = NULL
        size = self.ptr.getArray(&array)
        if isinstance(b, PVec):
            bptr = (<PVec>b).ptr
            bsize = bptr.getArray(&barray)
            if bsize == size:
                for i in range(size):
                    array[i] *= barray[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        elif hasattr(b, '__len__'):
            bsize = len(b)
            if bsize == size:
                for i in range(size):
                    array[i] *= b[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        else:
            value = b
            for i in range(size):
                array[i] *= value
        return self

    def __itruediv__(self, b):
        cdef int size = 0
        cdef int bsize = 0
        cdef ParOptScalar *array = NULL
        cdef ParOptScalar *barray = NULL
        cdef ParOptScalar value = 0.0
        cdef ParOptVec *bptr = NULL
        size = self.ptr.getArray(&array)
        if isinstance(b, PVec):
            bptr = (<PVec>b).ptr
            bsize = bptr.getArray(&barray)
            if bsize == size:
                for i in range(size):
                    array[i] /= barray[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        elif hasattr(b, '__len__'):
            bsize = len(b)
            if bsize == size:
                for i in range(size):
                    array[i] /= b[i]
            else:
                errmsg = 'PVecs must be the same size'
                raise ValueError(errmsg)
        else:
            value = b
            for i in range(size):
                array[i] /= value
        return self

    def __getitem__(self, k):
        cdef int size = 0
        cdef ParOptScalar *array
        size = self.ptr.getArray(&array)
        if isinstance(k, int):
            if k < 0 or k >= size:
                errmsg = 'Index %d out of range [0,%d)'%(k, size)
                raise IndexError(errmsg)
            return array[k]
        elif isinstance(k, slice):
            start, stop, step = k.indices(size)
            d = (stop-1 - start)//step + 1
            arr = np.zeros(d, dtype=dtype)
            index = 0
            for i in range(start, stop, step):
                if i < 0:
                    i = size+i
                if i >= 0 and i < size:
                    arr[index] = array[i]
                else:
                    raise IndexError('Index %d out of range [0,%d)'%(i, size))
                index += 1
            return arr
        else:
            errmsg = 'Index must be of type int or slice'
            raise ValueError(errmsg)

    def __setitem__(self, k, values):
        cdef int size = 0
        cdef ParOptScalar *array
        size = self.ptr.getArray(&array)
        if isinstance(k, int):
            if k < 0 or k >= size:
                errmsg = 'Index %d out of range [0,%d)'%(k, size)
                raise IndexError(errmsg)
            array[k] = values
        elif isinstance(k, slice):
            start, stop, step = k.indices(size)
            if hasattr(values, '__len__'):
                index = 0
                for i in range(start, stop, step):
                    if i < 0:
                        i = size+i
                    if i >= 0 and i < size:
                        array[i] = values[index]
                    else:
                        raise IndexError('Index %d out of range [0,%d)'%(i, size))
                    index += 1
            else:
                for i in range(start, stop, step):
                    if i < 0:
                        i = size+i
                    if i >= 0 and i < size:
                        array[i] = values
                    else:
                        raise IndexError('Index %d out of range [0,%d)'%(i, size))
        else:
            errmsg = 'Index must be of type int or slice'
            raise ValueError(errmsg)
        return

    def zeroEntries(self):
        """Zero the values"""
        if self.ptr:
            self.ptr.zeroEntries()
        return

    def copyValues(self, PVec vec):
        """Copy values from the provided PVec"""
        if self.ptr and vec.ptr:
            self.ptr.copyValues(vec.ptr)
        return

    def norm(self):
        """Compute the l2 norm of the vector"""
        return self.ptr.norm()

    def l1norm(self):
        """Compute the l1 norm of the vector"""
        return self.ptr.l1norm()

    def maxabs(self):
        """Compute the linfty norm of the vector"""
        return self.ptr.maxabs()

    def dot(self, PVec vec):
        """Compute the dot product with the provided PVec"""
        return self.ptr.dot(vec.ptr)

# Python classes for the ParOptCompactQuasiNewton methods
cdef class CompactQuasiNewton:
    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr:
            self.ptr.decref()

    def update(self, PVec s, PVec y):
        if self.ptr:
            self.ptr.update(NULL, NULL, NULL, s.ptr, y.ptr)

    def mult(self, PVec x, PVec y):
        if self.ptr:
            self.ptr.mult(x.ptr, y.ptr)

    def multAdd(self, ParOptScalar alpha, PVec x, PVec y):
        if self.ptr:
            self.ptr.multAdd(alpha, x.ptr, y.ptr)

cdef class LBFGS(CompactQuasiNewton):
    def __cinit__(self, ProblemBase prob, int subspace=10,
                  ParOptBFGSUpdateType update_type=SKIP_NEGATIVE_CURVATURE):
        cdef ParOptLBFGS *lbfgs = NULL
        lbfgs = new ParOptLBFGS(prob.ptr, subspace)
        lbfgs.setBFGSUpdateType(update_type)
        self.ptr = lbfgs
        self.ptr.incref()

cdef class LSR1(CompactQuasiNewton):
    def __cinit__(self, ProblemBase prob, int subspace=10):
        self.ptr = new ParOptLSR1(prob.ptr, subspace)
        self.ptr.incref()

# Python class for corresponding instance ParOpt
cdef class InteriorPoint:
    cdef ParOptInteriorPoint *ptr
    def __cinit__(self, ProblemBase _prob, options):
        cdef ParOptOptions *opts = new ParOptOptions(_prob.ptr.getMPIComm())
        ParOptInteriorPointAddDefaultOptions(opts)
        addDictionaryToOptions(options, opts)
        self.ptr = new ParOptInteriorPoint(_prob.ptr, opts)
        self.ptr.incref()
        return

    def __dealloc__(self):
        if self.ptr:
            self.ptr.decref()

    # Perform the optimization
    def optimize(self, bytes checkpoint=None):
        if checkpoint is None:
            return self.ptr.optimize(NULL)
        else:
            return self.ptr.optimize(checkpoint)

    def getOptimizedPoint(self):
        """
        Get the optimized solution in PVec form for interpolation purposes
        """
        cdef int ncon = 0
        cdef ParOptScalar *_z = NULL
        cdef ParOptVec *_x = NULL
        cdef ParOptVec *_zw = NULL
        cdef ParOptVec *_zl = NULL
        cdef ParOptVec *_zu = NULL

        # Get the problem size/vector for the values
        self.ptr.getProblemSizes(NULL, &ncon, NULL, NULL, NULL)
        self.ptr.getOptimizedPoint(&_x, &_z, &_zw, &_zl, &_zu)

        # Set the default values
        z = None
        x = None
        zw = None
        zl = None
        zu = None

        # Convert the multipliers to an in-place numpy array. This is
        # duplicated on all processors, and must have the same values
        # on all processors.
        if _z != NULL:
            z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z, self)

        # Note that these vectors are owned by the ParOpt class, we're simply
        # passing references to them back to the python layer.
        if _x != NULL:
            x = _init_PVec(_x)
        if _zw != NULL:
            zw = _init_PVec(_zw)
        if _zl != NULL:
            zl = _init_PVec(_zl)
        if _zu != NULL:
            zu = _init_PVec(_zu)

        return x, z, zw, zl, zu

    def getOptimizedSlacks(self):
        """
        Get the optimized slack variables from the problem
        """
        cdef int ncon = 0
        cdef ParOptScalar *_s = NULL
        cdef ParOptScalar *_t = NULL
        cdef ParOptVec *_sw = NULL

        # Get the problem size/vector for the values
        self.ptr.getProblemSizes(NULL, &ncon, NULL, NULL, NULL)
        self.ptr.getOptimizedSlacks(&_s, &_t, &_sw)

        s = None
        t = None
        sw = None

        if _s != NULL:
            s = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_s, self)
        if _t != NULL:
            t = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_t, self)

        # Convert to a vector
        if _sw != NULL:
            sw = _init_PVec(_sw)

        return s, t, sw

    # Check objective and constraint gradients
    def checkGradients(self, double dh):
        self.ptr.checkGradients(dh)

    def setPenaltyGamma(self, double gamma):
        self.ptr.setPenaltyGamma(gamma)

    def setMultiplePenaltyGamma(self, list gamma):
        cdef double *g = NULL
        cdef int num_gam = 0
        num_gam = len(gamma)
        g = <double*>malloc(num_gam*sizeof(double));
        for i in range(num_gam):
            g[i] = <double>gamma[i];

        self.ptr.setPenaltyGamma(g)
        free(g)

    def getPenaltyGamma(self):
        cdef const double *penalty_gamma
        cdef int ncon
        ncon = self.ptr.getPenaltyGamma(&penalty_gamma)
        gamma = np.zeros(ncon, dtype=np.double)
        for i in range(ncon):
            gamma[i] = penalty_gamma[i]
        return gamma

    def getComplementarity(self):
        return self.ptr.getComplementarity()

    def resetQuasiNewtonHessian(self):
        self.ptr.resetQuasiNewtonHessian()

    def setQuasiNewton(self, CompactQuasiNewton qn):
        if qn is not None:
            self.ptr.setQuasiNewton(qn.ptr)
        else:
            self.ptr.setQuasiNewton(NULL)

    def resetDesignAndBounds(self):
        self.ptr.resetDesignAndBounds()

    # Write out the design variables to binary format (fast MPI/IO)
    def writeSolutionFile(self, fname):
        cdef char *filename = convert_to_chars(fname)
        if filename is not None:
            return self.ptr.writeSolutionFile(filename)

    def readSolutionFile(self, fname):
        cdef char *filename = convert_to_chars(fname)
        if filename is not None:
            return self.ptr.readSolutionFile(filename)

cdef class MMA(ProblemBase):
    cdef ParOptMMA *mma
    def __cinit__(self, ProblemBase _prob, options):
        cdef ParOptOptions *opts = new ParOptOptions(_prob.ptr.getMPIComm())
        ParOptMMAAddDefaultOptions(opts)
        addDictionaryToOptions(options, opts)
        self.mma = new ParOptMMA(_prob.ptr, opts)
        self.mma.incref()
        self.ptr = self.mma
        return

    def getOptimizedPoint(self):
        cdef ParOptVec *x
        self.mma.getOptimizedPoint(&x)
        return _init_PVec(x)

    def getAsymptotes(self):
        cdef ParOptVec *L = NULL
        cdef ParOptVec *U = NULL
        self.mma.getAsymptotes(&L, &U)
        return _init_PVec(L), _init_PVec(U)

    def getDesignHistory(self):
        cdef ParOptVec* x1 = NULL
        cdef ParOptVec *x2 = NULL
        self.mma.getDesignHistory(&x1, &x2)
        return _init_PVec(x1), _init_PVec(x2)

cdef class TrustRegionSubproblem:
    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            self.ptr.decref()

    def checkGradients(self, double dh=1e-6, PVec x=None,
                       check_hvec_product=False):
        cdef ParOptVec *vec = NULL
        cdef int check_hvec = 0
        if x is not None:
            vec = x.ptr
        if check_hvec_product:
            check_hvec = 1
        if self.ptr != NULL:
            self.ptr.checkGradients(dh, vec, check_hvec)
        return

cdef class QuadraticSubproblem(TrustRegionSubproblem):
    def __cinit__(self, ProblemBase problem, CompactQuasiNewton qn=None):
        cdef ParOptCompactQuasiNewton* qn_ptr = NULL
        if qn is not None:
            qn_ptr = qn.ptr
        self.subproblem = new ParOptQuadraticSubproblem(problem.ptr, qn_ptr)
        self.subproblem.incref()
        self.ptr = self.subproblem

cdef class TrustRegion:
    cdef ParOptTrustRegion *tr
    def __cinit__(self, TrustRegionSubproblem prob, options):
        """
        Create a trust region optimization object

        Args:
            prob: Subproblem object for the trust region problem
            options: Optimization options
        """
        cdef ParOptOptions *opts = new ParOptOptions(prob.ptr.getMPIComm())
        ParOptTrustRegionAddDefaultOptions(opts)
        addDictionaryToOptions(options, opts)
        self.tr = new ParOptTrustRegion(prob.subproblem, opts)
        self.tr.incref()
        return

    def __dealloc__(self):
        if self.tr:
            self.tr.decref()

    def optimize(self, InteriorPoint optimizer):
        self.tr.optimize(optimizer.ptr)

    def getOptimizedPoint(self):
        """
        Get the optimized solution in PVec form for interpolation purposes
        """
        cdef ParOptVec *_x = NULL
        self.tr.getOptimizedPoint(&_x)

        x = None
        if _x != NULL:
            x = _init_PVec(_x)

        return x

cdef class Optimizer:
    cdef ParOptOptimizer *ptr
    def __cinit__(self, ProblemBase problem, options):
        cdef ParOptOptions *opts = new ParOptOptions(problem.ptr.getMPIComm())
        ParOptOptimizerAddDefaultOptions(opts)
        addDictionaryToOptions(options, opts)
        self.ptr = new ParOptOptimizer(problem.ptr, opts)
        self.ptr.incref()

    def __dealloc__(self):
        if self.ptr != NULL:
            self.ptr.decref()

    def optimize(self):
        if self.ptr != NULL:
            self.ptr.optimize()

    def getOptimizedPoint(self):
        """
        Get the optimized solution in PVec form for interpolation purposes
        """
        cdef int ncon = 0
        cdef ParOptScalar *_z = NULL
        cdef ParOptVec *_x = NULL
        cdef ParOptVec *_zw = NULL
        cdef ParOptVec *_zl = NULL
        cdef ParOptVec *_zu = NULL
        cdef ParOptProblem *problem = NULL

        # Get the problem size/vector for the values
        problem = self.ptr.getProblem()
        problem.getProblemSizes(NULL, &ncon, NULL, NULL, NULL)
        self.ptr.getOptimizedPoint(&_x, &_z, &_zw, &_zl, &_zu)

        # Set the default values
        z = None
        x = None
        zw = None
        zl = None
        zu = None

        # Convert the multipliers to an in-place numpy array. This is
        # duplicated on all processors, and must have the same values
        # on all processors.
        if _z != NULL:
            z = inplace_array_1d(PAROPT_NPY_SCALAR, ncon, <void*>_z, self)

        # Note that these vectors are owned by the ParOpt class, we're simply
        # passing references to them back to the python layer.
        if _x != NULL:
            x = _init_PVec(_x)
        if _zw != NULL:
            zw = _init_PVec(_zw)
        if _zl != NULL:
            zl = _init_PVec(_zl)
        if _zu != NULL:
            zu = _init_PVec(_zu)

        return x, z, zw, zl, zu

    def setTrustRegionSubproblem(self, TrustRegionSubproblem prob):
        self.ptr.setTrustRegionSubproblem(prob.subproblem)
