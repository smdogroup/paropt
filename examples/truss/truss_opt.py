# Import MPI 
import mpi4py.MPI as MPI

# Import the truss analysis problem
from truss_analysis import TrussAnalysis

# Import ParOpt
from paropt import ParOpt

def get_ground_structure(N=4, M=4, L=2.5, P=10.0):
    # Set up the connectivity array
    conn = []
    for ii in xrange(N*M):
        for i in xrange(ii+1, N*M):
            if (i/N == ii/N or
                i%N == ii%N):
                continue
            conn.append([ii, i])

    for i in xrange(N):
        for j in xrange(M-1):
            conn.append([i + j*N, i + (j+1)*N])

    for i in xrange(N-1):
        for j in xrange(M):
            conn.append([i + j*N, i+1 + j*N])

    # Set the positions of all the nodes
    xpos = []
    for j in xrange(M):
        for i in xrange(N):
            xpos.extend([i*L, j*L])

    # Set the node locations
    loads = {}
    loads[N-1] = [0, -P]

    bcs = {}
    for j in xrange(M):
        bcs[j*N] = [0, 1]

    return conn, xpos, loads, bcs

# Set the dimensions
N = 4
M = 3
L = 2.5

# Create the ground structure
conn, xpos, loads, bcs = get_ground_structure(N=N, M=M, 
                                              L=L, P=10e3)

# Set the material properties
E = 70e9
rho = 2700.0

# Set the lower bounds for the material variables
A_min = 1e-3
A_max = 10.0
Area_scale = 1.0

# Set the fixed mass constraint
mass_fixed = 10*N*M*L*rho

# Set the mass scaling
mass_scale = 0.5*(N+M)*L*rho

# Create the truss topology optimization object
truss = TrussAnalysis(conn, xpos, loads, bcs, 
                      E, rho, mass_fixed, A_min, A_max,
                      Area_scale=Area_scale, mass_scale=mass_scale)

# Set the options
truss.setInequalityOptions(dense_ineq=False, 
                           use_lower=True,
                           use_upper=False)

# Create the optimizer
max_lbfgs = 75
opt = ParOpt.pyParOpt(truss, max_lbfgs)

# Set the optimality tolerance
opt.setAbsOptimalityTol(1e-5)

# Set the Hessian-vector product iterations
opt.setUseHvecProduct(1)
opt.setGMRESSubspaceSize(15)

# Set optimization parameters
opt.setArmijioParam(1e-5)
opt.setMaxMajorIterations(750)
opt.setHessianResetFreq(max_lbfgs)

# Set the output file to use
opt.setOutputFile('truss_paropt.out')

# Check the derivatives
opt.checkGradients(1e-6)

# Optimize the truss
opt.optimize()

# Get the optimized point
x = opt.getOptimizedPoint()

truss.plotTruss(x*Area_scale, tol=1e-2, filename='opt_truss.pdf')
print Area_scale*x

# Retrieve the optimized multipliers
z, zw, zl, zu = opt.getOptimizedMultipliers()

print 'z =  ', z
print 'zw = ', zw
print 'zl = ', zl
print 'zu = ', zu

# Check the gradients again after optimization
opt.checkGradients(1e-6)
