# Import this for directory creation
import os

# Import MPI 
import mpi4py.MPI as MPI

# Import the truss analysis problem
from truss_analysis import TrussAnalysis

# Import ParOpt
from paropt import ParOpt

# Import argparse
import argparse

# Import parts of matplotlib for plotting
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

def get_ground_structure(N=4, M=4, L=2.5, P=10.0):
    '''
    Set up the connectivity for a ground structure consisting of a 2D
    mesh of (N x M) nodes of completely connected elements.

    A single point load is applied at the lower right-hand-side of the
    mesh with a value of P. All nodes are evenly spaced at a distance
    L.

    input:
    N:  number of nodes in the x-direction
    M:  number of nodes in the y-direction
    L:  distance between nodes

    returns
    conn:   the connectivity
    xpos:   the nodal locations
    loads:  the loading conditions applied to the problem
    P:      the load applied to the mesh
    '''

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

def setup_ground_struct(N, M, L=2.5, E=70e9, rho=2700.0,
                        A_min=1e-3, A_max=10.0):
    '''
    Create a ground structure with a given number of nodes and
    material properties.
    '''

    # Create the ground structure
    conn, xpos, loads, bcs = get_ground_structure(N=N, M=M, 
                                                  L=L, P=10e3)

    # Set the scaling for the material variables
    Area_scale = 1.0

    # Set the fixed mass constraint
    mass_fixed = 5*N*M*L*rho

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

    return truss

def paropt_truss(truss, use_hessian=False, prefix='results'):
    '''
    Optimize the given truss structure using ParOpt
    '''

    # Create the optimizer
    max_lbfgs = 50
    opt = ParOpt.pyParOpt(truss, max_lbfgs)

    # Set the optimality tolerance
    opt.setAbsOptimalityTol(1e-5)

    # Set the Hessian-vector product iterations
    if use_hessian:
        opt.setUseLineSearch(0)
        opt.setUseHvecProduct(1)
        opt.setGMRESSubspaceSize(30)
        opt.setNKSwitchTolerance(1.0)
        opt.setEisenstatWalkerParameters(0.5, 1.5)
        opt.setGMRESTolerances(1.0, 1e-30)
    else:
        opt.setUseHvecProduct(0)

    # Set optimization parameters
    opt.setArmijioParam(1e-5)
    opt.setMaxMajorIterations(2500)
    opt.setHessianResetFreq(max_lbfgs)
    
    # Set the output file to use
    fname = os.path.join(prefix, 'truss_paropt%dx%d.out'%(N, M)) 
    opt.setOutputFile(fname)
    
    # Optimize the truss
    opt.optimize()

    return opt

def pyopt_truss(truss, optimizer='snopt', options={}):
    '''
    Take the given problem and optimize it with the given optimizer
    from the pyOptSparse library of optimizers.
    '''
    # Import the optimization problem
    from pyoptsparse import Optimization, OPT

    class pyOptWrapper:
        def __init__(self, truss):
            self.truss = truss
        def objcon(self, x):
            fail, obj, con = self.truss.evalObjCon(x['x'])
            funcs = {'objective': obj, 'con': con}
            return funcs, fail
        def gobjcon(self, x, funcs):
            g = np.zeros(x['x'].shape)
            A = np.zeros((1, x['x'].shape[0]))
            fail = self.truss.evalObjConGradient(x['x'], g, A)
            sens = {'objective': {'x': g}, 'con': {'x': A}}
            return sens, fail

    # Set the design variables
    wrap = pyOptWrapper(truss)
    prob = Optimization('Truss', wrap.objcon)

    # Determine the initial variable values and their lower/upper
    # bounds in the design problem
    n = len(truss.conn)
    x0 = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)
    truss.getVarsAndBounds(x0, lower, upper)
    
    # Set the variable bounds and initial values
    prob.addVarGroup('x', n, value=x0, lower=lower,
                     upper=[None]*n)
    
    # Set the constraints
    prob.addConGroup('con', 1, lower=0.0, upper=0.0)

    # Add the objective
    prob.addObj('objective')

    # Optimize the problem
    try:
        opt = OPT(optimizer, options=options)
        sol = opt(prob, sens=wrap.gobjcon)
    except:
        opt = None
        sol = None

    return opt, prob, sol

def get_performance_profile(r, tau_max):
    '''
    Get the performance profile for the given ratio
    '''

    # Sort the ratios in increasing order
    r = sorted(r)

    # Find the first break-point at which tau >= 1.0
    n = 0
    while n < len(r) and r[n] <= 1.0: 
        n += 1

    # Add the first part of the profile to the plot
    rho = [0.0, 1.0*n/len(r)]
    tau = [1.0, 1.0]

    # Add all subsequent break-points to the plot
    while n < len(r) and r[n] < tau_max:
        rho.extend([1.0*n/len(r), 1.0*(n+1)/len(r)])
        tau.extend([r[n], r[n]])
        n += 1

    # Finish off the profile to the max value
    rho.append(1.0*n/len(r))
    tau.append(tau_max)

    return tau, rho

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=4, 
                    help='Nodes in x-direction')
parser.add_argument('--M', type=int, default=3, 
                    help='Nodes in y-direction')
parser.add_argument('--profile', action='store_true', 
                    default=False, help='Performance profile')
parser.add_argument('--use_hessian', 
                    action='store_true', default=False, 
                    help='Use the exact Hessian-vector products')
parser.add_argument('--optimizer', default='None',
                    help='Optimizer name from pyOptSparse')
args = parser.parse_args()

# Get the arguments
N = args.N
M = args.M
profile = args.profile
use_hessian = args.use_hessian
optimizer = args.optimizer

# Set the options for the pyOptSparse optimizers for comparisons
all_options = {
    'slsqp': {'MAXIT': 5000},
    'snopt': {},
    'ipopt': {}}

# Set the output file name
outfile_names = {'None': None,
                 'slsqp': 'IFILE',
                 'snopt': 'Print file',
                 'ipopt': 'output_file'}
outfile_name = outfile_names[optimizer]

if profile:
    # Set the trusses that will be optimized
    trusses = [[3, 3], [4, 3], [5, 3], [6, 3],
               [4, 4], [5, 4], [6, 4], [7, 4],
               [5, 5], [6, 5], [7, 5], [8, 5]]

    # Perform the optimization with and without the Hessian
    if optimizer is 'None':
        # Set the prefix to use
        if use_hessian:
            prefix = 'hessian'
        else:
            prefix = 'bfgs'
    else:
        prefix = optimizer

    # Get to the full path
    prefix = os.path.join(os.getcwd(), prefix)

    # Create the directory if it does not yet exist
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    fname = os.path.join(prefix, 'performance_profile.dat')
    fp = open(fname, 'w')

    # Iterate over all the trusses
    index = 0
    for vals in trusses:
        # Set the values of N/M
        N = vals[0]
        M = vals[1]
            
        print 'Optimizing truss (%d x %d) ...'%(N, M)
            # Optimize each of the trusses
        truss = setup_ground_struct(N, M)
        t0 = MPI.Wtime()
        if optimizer is 'None':
            opt = paropt_truss(truss, prefix=prefix,
                               use_hessian=use_hessian)

            # Get the optimized point
            x = opt.getOptimizedPoint()
        else:
            # Read out the options from the dictionary of options
            options = all_options[optimizer]
            
            # Set the output file
            options[outfile_name] = os.path.join(prefix, 
                                                 'output_%dx%d.out'%(N, M))
            # Optimize the truss with the specified optimizer
            opt, prob, sol = pyopt_truss(truss, optimizer=optimizer,
                                         options=options)

            # Extract the design variable values
            if sol is not None:
                x = []
                for var in sol.variables['x']:
                    x.append(var.value)
                x = np.array(x)
            else:
                x = None

        # Keep track of the optimization time
        t0 = MPI.Wtime() - t0

        # Plot the truss
        filename = os.path.join(prefix, 'opt_truss%dx%d.pdf'%(N, M))
        if x is not None:
            truss.plotTruss(x, tol=1e-2, filename=filename) 
            
        # Record the performance of the algorithm
        fp.write('%d %d %d %d %d %e\n'%(
                index,
                truss.fevals, truss.gevals, truss.hevals,
                truss.fevals + truss.hevals, t0))
        fp.flush()
        index += 1
    
    # Close the file
    fp.close()

    profiles = ['bfgs', 'hessian', 'snopt', 'slsqp']
    colours = ['g', 'b', 'r', 'k']

    # Read the performance values from each file
    perform = np.zeros((len(trusses), len(profiles)))
    index = 0
    for prefix in profiles:
        fname = os.path.join(prefix, 'performance_profile.dat')
        perf = np.loadtxt(fname)

        # Set the performance metric
        for i in xrange(len(trusses)):
            perform[i, index] = perf[i, -1]
        index += 1

    # Create the performance profiles
    nprob = len(trusses)
    r = np.zeros(perform.shape)
        
    # Compute the ratios for the best performance
    for i in xrange(nprob):
        best = 1.0*min(perform[i, :])
        r[i, :] = perform[i, :]/best
                    
    # Plot the data
    fig = plt.figure(facecolor='w')

    tau_max = 10.0
    for k in xrange(len(profiles)):
        tau, rho = get_performance_profile(r[:, k], tau_max)
        plt.plot(tau, rho, colours[k], linewidth=2, label=profiles[k])

    # Finish off the plot and print it
    plt.legend(loc='lower right')
    plt.axis([0.95, tau_max, 0.0, 1.1])
    filename = 'performance_profile.pdf'
    plt.savefig(filename)
    plt.close()
else:
    # Optimize the structure
    prefix = 'results'
    truss = setup_ground_struct(N, M)
    opt = paropt_truss(truss, prefix=prefix,
                       use_hessian=use_hessian)

    # Get the optimized point
    x = opt.getOptimizedPoint()

    # Plot the truss
    truss.plotTruss(x, tol=1e-2, 
                    filename=prefix+'/opt_truss%dx%d.pdf'%(N, M))
    print truss.Area_scale*x
    
    # Retrieve the optimized multipliers
    z, zw, zl, zu = opt.getOptimizedMultipliers()
    print 'z =  ', z
    print 'zw = ', zw
    print 'zl = ', zl
    print 'zu = ', zu
