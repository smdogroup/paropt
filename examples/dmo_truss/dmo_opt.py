# Import this for directory creation
import os

# Import the greatest common divisor code
from fractions import gcd

# Import MPI 
import mpi4py.MPI as MPI

# Import ParOpt
from paropt import ParOpt

# Import argparse
import argparse

# Import parts of matplotlib for plotting
import matplotlib.pyplot as plt

# Import numpy
import numpy as np

# Import the truss analysis problem
from dmo_truss_analysis import TrussAnalysis

def get_ground_structure(N=4, M=4, L=2.5, P=10.0, n=5):
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

    # First, generate a co-prime grid that will be used to 
    grid = []
    for x in range(1,n+1):
        for y in range(1,n+1):
            if gcd(x, y) == 1:
                grid.append((x,y))

    # Reflect the ge
    reflect = []
    for d in grid:
        reflect.append((-d[0], d[1]))

    grid.extend(reflect)
    grid.extend([(0,1), (1,0)])

    # Set up the connectivity array
    conn = []
    for i in xrange(N):
        for j in xrange(M):
            n1 = i + N*j
            for d in grid:
                if ((i + d[0] < N) and (j + d[1] < M) and
                    (i + d[0] >= 0) and (j + d[1] >= 0)):
                    n2 = i + d[0] + (j + d[1])*N
                    conn.append([n1, n2])

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

def setup_ground_struct(N, M, L=2.5, E=70e9):
    '''
    Create a ground structure with a given number of nodes and
    material properties.
    '''

    # Set the values for the
    Avals = [ 0.01, 0.02, 0.05 ]
    rho = [ 25.0, 54.0, 150.0 ]

    # Create the ground structure
    conn, xpos, loads, bcs = get_ground_structure(N=N, M=M, 
                                                  L=L, P=10e3)

    # Set the scaling for the material variables
    Area_scale = 1.0

    # Set the fixed mass constraint
    m_fixed = 5.0*N*M*L*rho[0]

    # Create the truss topology optimization object
    truss = TrussAnalysis(conn, xpos, loads, bcs, 
                          E, rho, Avals, m_fixed)

    # Set the options
    truss.setInequalityOptions(dense_ineq=True, 
                               sparse_ineq=False,
                               use_lower=True,
                               use_upper=True)

    return truss

def paropt_truss(truss, use_hessian=False,
                 max_qn_subspace=50, qn_type=ParOpt.BFGS,
                 prefix='results'):
    '''
    Optimize the given truss structure using ParOpt
    '''

    # Create the optimizer
    opt = ParOpt.pyParOpt(truss, max_qn_subspace, qn_type)

    # Set the optimality tolerance
    opt.setAbsOptimalityTol(1e-5)

    # Set the Hessian-vector product iterations
    if use_hessian:
        opt.setUseLineSearch(0)
        opt.setUseHvecProduct(1)
        opt.setGMRESSubspaceSize(100)
        opt.setNKSwitchTolerance(1.0)
        opt.setEisenstatWalkerParameters(0.5, 0.0)
        opt.setGMRESTolerances(1.0, 1e-30)
    else:
        opt.setUseHvecProduct(0)

    # Set optimization parameters
    opt.setArmijioParam(1e-5)
    opt.setMaxMajorIterations(2500)

    # Perform a quick check of the gradient (and Hessian)
    opt.checkGradients(1e-6)

    return opt

def optimize_truss(N, M, heuristic):
    # Optimize the structure
    prefix = os.path.join('results', '%dx%d'%(N, M), heuristic)

    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Create the ground structure and optimization
    truss = setup_ground_struct(N, M)
    opt = paropt_truss(truss, prefix=prefix,
                       use_hessian=use_hessian,
                       qn_type=ParOpt.BFGS)

    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, "min SE", "max SE", "fobj", '
    s += '"min gamma", "max gamma", "gamma", '
    s += '"min d", "max d", "tau", '
    s += 'feval, geval, hvec, time\n'
    fp.write(s)

    init_time = MPI.Wtime()

    # Initialize the gamma values
    gamma_init = 1e-4
    gamma = gamma_init*np.ones(truss.nelems)

    # Set the lower limits for the bounds
    truss.x_lb = 0.0
    truss.t_lb = 0.0

    alpha = 1.25
    beta = 0.25
    print 'Heuristic: %s  alpha = %8.4f beta = %8.4f'%(
        heuristic, alpha, beta)

    fobj_prev = 0.0

    niters = 0
    for k in xrange(100):
        # Set the output file to use
        fname = os.path.join(prefix, 'truss_paropt_iter%d.out'%(k)) 
        opt.setOutputFile(fname)

        # Optimize the truss
        if k > 0:
            opt.setInitBarrierParameter(1e-4)
        opt.optimize()

        # Get the optimized point
        x = opt.getOptimizedPoint()

        # Get the discrete infeasibility measure
        d = truss.getDiscreteInfeas(x)

        # Get the strain energy associated with each element
        Ue = truss.getStrainEnergy(x)

        # Compute the objective function
        fobj = np.sum(Ue)

        # Print out the iteration information to the screen
        print 'Iteration %d'%(k)
        print 'Min/max SE:    %15.5e %15.5e  Total: %15.5e'%(
            np.min(Ue), np.max(Ue), np.sum(Ue))
        print 'Min/max gamma: %15.5e %15.5e  Total: %15.5e'%(
            np.min(gamma), np.max(gamma), np.sum(gamma))
        print 'Min/max d:     %15.5e %15.5e  Total: %15.5e'%(
            np.min(d), np.max(d), np.sum(d))

        # Terminate if the maximum discrete infeasibility measure is
        # sufficiently low
        if np.max(d) < 3.0*(truss.t_min - truss.t_min**2):
            break

        s = '%d %e %e %e %e %e %e %e %e %e '%(
            k, np.min(Ue), np.max(Ue), np.sum(Ue),
            np.min(gamma), np.max(gamma), np.sum(gamma),
            np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e\n'%(
            truss.fevals, truss.gevals, truss.hevals, 
            MPI.Wtime() - init_time)
        fp.write(s)
        fp.flush()

        # Print the output
        filename = 'opt_truss_iter%d.tex'%(k)
        output = os.path.join(prefix, filename)
        truss.printTruss(x, filename=output)

        # Decrease the lower bound on the thickness by up to 0.5
        truss.t_lb *= 0.5

        # Set the new penalty
        truss.setNewInitPointPenalty(x, gamma)

        if abs((fobj - fobj_prev)/fobj < 0.001):
            y = Ue*d/np.max(Ue*d)
            if heuristic == 'scalar':
                gamma *= (alpha + beta)
            elif heuristic == 'linear':
                gamma *= alpha + beta*y
            elif heuristic == 'discrete':
                gamma *= alpha + beta*d

        # Store the previous value of the objective function
        fobj_prev = 1.0*fobj

        # Increase the iteration counter
        niters += 1

    # PDFLatex all the output files
    for k in xrange(niters):
        filename = 'opt_truss_iter%d.tex'%(k)
        os.system('cd %s; pdflatex %s > /dev/null ; cd ..;'%(prefix, filename))
    
    # Print out the last optimized truss
    filename = 'opt_truss.tex'
    output = os.path.join(prefix, filename)
    truss.printTruss(x, filename=output)
    os.system('cd %s; pdflatex %s > /dev/null ; cd ..;'%(prefix, filename))

    return

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=4, 
                    help='Nodes in x-direction')
parser.add_argument('--M', type=int, default=3, 
                    help='Nodes in y-direction')
parser.add_argument('--profile', action='store_true', 
                    default=False, help='Performance profile')
parser.add_argument('--heuristic', type=str, default='scalar',
                    help='Heuristic type')

args = parser.parse_args()

# Get the arguments
N = args.N
M = args.M
profile = args.profile
heuristic = args.heuristic

# Set the use Hessian
use_hessian = True

trusses = [
    (4, 4),   (5, 5),   (6, 6), 
    (7, 7),   (8, 8),   (9, 9), 
    (10, 10), (11, 11), (12, 12), 
    (13, 13), (14, 14), (15, 15), # 1:1
    (4, 2),   (6, 3),   (8, 4), 
    (10, 5),  (12, 6),  (14, 7), # 2:1
    (9, 3),   (12, 4),  (15, 5), # 3:1
    (4, 3),   (8, 6),   (12, 4)] # 4:3

if profile:
    for N, M in trusses:
        optimize_truss(N, M, heuristic)
else:
    optimize_truss(N, M, heuristic)
