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

# Import numpy
import numpy as np

# Import the truss analysis problem
from dmo_truss_analysis import TrussAnalysis

def get_ground_structure(N=4, M=4, L=2.5, P=1e4, n=10):
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

    # Reflect the grid
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

def setup_ground_struct(N, M, L=2.5, E=70e9, 
                        x_lb=0.0, sigma=100.0,
                        use_mass_constraint=False):
    '''
    Create a ground structure with a given number of nodes and
    material properties.
    '''

    # Set the values for the
    Avals = [ 0.01, 0.02, 0.05 ]
    rho = [ 0.25,  0.55, 1.5 ]

    # Create the ground structure
    conn, xpos, loads, bcs = get_ground_structure(N=N, M=M, 
                                                  L=L, P=10e3)

    # Set the scaling for the material variables
    Area_scale = 1.0

    # Set the fixed mass constraint
    AR = 1.0*(N-1)/(M-1)
    m_fixed = (M-1)*(N-1)*L*rho[-1]

    # Create the truss topology optimization object
    truss = TrussAnalysis(conn, xpos, loads, bcs, 
                          E, rho, Avals, m_fixed, 
                          sigma=sigma, x_lb=x_lb, epsilon=1e-6,
                          use_mass_constraint=use_mass_constraint)

    # Set the options
    truss.setInequalityOptions(dense_ineq=False, 
                               sparse_ineq=False,
                               use_lower=True,
                               use_upper=True)

    return truss

def paropt_truss(truss, use_hessian=False,
                 max_qn_subspace=50, qn_type=ParOpt.BFGS):
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

def optimize_truss(N, M, root_dir='results',
                   use_mass_constraint=False, sigma=100.0,
                   max_d=1e-4, theta=1e-3, penalization='SIMP',
                   parameter=2.0, max_iters=50):
    # Optimize the structure
    heuristic = '%s%.0f'%(penalization ,parameter)
    prefix = os.path.join(root_dir, '%dx%d'%(N, M), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
   
    # Create the ground structure and optimization
    truss = setup_ground_struct(N, M, sigma=sigma,
                                use_mass_constraint=use_mass_constraint,
                                x_lb=0.0)
    
    # Set up the optimization problem in ParOpt
    opt = paropt_truss(truss, use_hessian=use_hessian,
                       qn_type=ParOpt.BFGS)

    # Create a vector of all ones
    m_add = 0.0
    if use_mass_constraint:
        xones = np.ones(truss.gmass.shape)
        m_add = truss.getMass(xones)/truss.nmats
    
    # Keep track of the fixed mass
    m_fixed_init = 1.0*truss.m_fixed
    truss.m_fixed = m_fixed_init + truss.x_lb*m_add

    # Log the optimization file
    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, "compliance", "fobj", "fpenalty", '
    s += '"min gamma", "max gamma", "gamma", '
    s += '"min d", "max d", "tau", "ninfeas", "mass infeas", '
    s += 'feval, geval, hvec, time\n'
    s += 'Zone T = %s\n'%(heuristic)
    fp.write(s)

    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Initialize the gamma values
    gamma = np.zeros(truss.nelems)

    # Previous value of the objective function
    fobj_prev = 0.0

    # Set the first time
    first_time = True

    # Set the initial compliance value
    comp_prev = 0.0

    # Set the tolerances for increasing/decreasing tau
    delta_tau_target = 1.0

    # Set the target rate of increase in gamma
    delta_max = 10.0
    delta_min = 1e-3

    # Keep track of the number of iterations
    niters = 0
    for k in xrange(max_iters):
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

        # Compute the infeasibility of the mass constraint
        m_infeas = truss.getMass(x)/truss.m_fixed - 1.0

        # Compute the discrete infeasibility measure
        tau = np.sum(d)

        # Get the compliance and objective values
        comp, fobj, fpenalty = truss.getL1Objective(x, gamma)

        # Keep track of how many bars are discrete infeasible
        draw_list = []
        for i in xrange(len(d)):
            if d[i] > max_d:
                draw_list.append(i)

        # Print out the iteration information to the screen
        print 'Iteration %d'%(k)
        print 'Min/max gamma: %15.5e %15.5e  Total: %15.5e'%(
            np.min(gamma), np.max(gamma), np.sum(gamma))
        print 'Min/max d:     %15.5e %15.5e  Total: %15.5e'%(
            np.min(d), np.max(d), np.sum(d))
        print 'Mass infeas:   %15.5e'%(m_infeas)

        s = '%d %e %e %e %e %e %e %e %e %e %2d %e '%(
            k, comp, fobj, fpenalty,
            np.min(gamma), np.max(gamma), np.sum(gamma),
            np.min(d), np.max(d), np.sum(d), len(draw_list), m_infeas)
        s += '%d %d %d %e\n'%(
            truss.fevals, truss.gevals, truss.hevals, 
            MPI.Wtime() - init_time)
        fp.write(s)
        fp.flush()

        # Terminate if the maximum discrete infeasibility measure is
        # sufficiently low
        if np.max(d) <= max_d:
            break

        # Print the output
        filename = 'opt_truss_iter%d.tex'%(k)
        output = os.path.join(prefix, filename)
        truss.printTruss(x, filename=output, draw_list=draw_list)

        if (np.fabs((comp - comp_prev)/comp) < 1e-3):
            if first_time:
                # Set the new value of delta
                gamma[:] = delta_min

                # Keep track of the previous value of the discrete
                # infeasibility measure
                tau_iter = 1.0*tau
                delta_iter = 1.0*delta_min

                # Set the first time flag to false
                first_time = False
            else:
                # Set the maximum delta initially
                delta = 1.0*delta_max

                # Limit the rate of discrete infeasibility increase
                tau_rate = (tau_iter - tau)/delta_iter 
                delta = max(min(delta, delta_tau_target/tau_rate), delta_min)
                gamma[:] = gamma + delta

                # Print out the chosen scaling for the design variables
                print 'Delta:         %15.5e'%(delta)

                # Keep track of the discrete infeasibility measure
                tau_iter = 1.0*tau
                delta_iter = 1.0*delta

        xinfty = truss.computeLimitDesign(x)

        # Print the output
        filename = 'opt_limit_truss_iter%d.tex'%(k)
        output = os.path.join(prefix, filename)
        truss.printTruss(xinfty, filename=output)
        
        # Set the new penalty
        truss.SIMP = parameter
        truss.RAMP = parameter
        truss.penalization = penalization
        truss.setNewInitPointPenalty(x, gamma)

        # Store the previous value of the objective function
        fobj_prev = 1.0*fobj
        comp_prev = 1.0*comp

        # Increase the iteration counter
        niters += 1

    # Close the log file
    fp.close()
    
    # Print out the last optimized truss
    filename = 'opt_truss.tex'
    output = os.path.join(prefix, filename)
    truss.printTruss(x, filename=output)
    os.system('cd %s; pdflatex %s > /dev/null ; cd ..;'%(prefix, filename))

    # Save the final optimized point
    fname = os.path.join(prefix, 'x_opt.dat')
    x = opt.getOptimizedPoint()
    np.savetxt(fname, x)

    # Get the rounded design
    xinfty = truss.computeLimitDesign(x)
    fname = os.path.join(prefix, 'x_opt_infty.dat')
    np.savetxt(fname, xinfty)

    return

# Parse the command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=4, 
                    help='Nodes in x-direction')
parser.add_argument('--M', type=int, default=3, 
                    help='Nodes in y-direction')
parser.add_argument('--profile', action='store_true', 
                    default=False, help='Performance profile')
parser.add_argument('--use_mass_constraint', action='store_true',
                    default=False, help='Use the mass constraint')
parser.add_argument('--parameter', type=float, default=3.0,
                    help='Penalization parameter')
parser.add_argument('--penalization', type=str, 
                    default='SIMP', help='Penalization type')
parser.add_argument('--sigma', type=float, default=20.0,
                    help='Penalty parameter value')
args = parser.parse_args()

# Get the arguments
N = args.N
M = args.M
profile = args.profile
use_mass_constraint = args.use_mass_constraint
penalization = args.penalization
parameter = args.parameter
sigma = args.sigma

# Set the root results directory
root_dir = 'results'
if use_mass_constraint:
    root_dir = 'con-results'

# Always use the Hessian-vector product implementation
use_hessian = True

# The trusses used in this instance
trusses = []
for j in range(3, 7):
   for i in range(j, 3*j+1):
       trusses.append((i, j))

# Run either the optimizations or the
if profile:
    for N, M in trusses:
        try:
            optimize_truss(N, M, root_dir=root_dir,
                           use_mass_constraint=use_mass_constraint,
                           sigma=sigma, penalization=penalization, 
                           parameter=parameter, max_iters=80)
        except:
            pass
else:
    optimize_truss(N, M, root_dir=root_dir,
                   use_mass_constraint=use_mass_constraint,
                   sigma=sigma, penalization=penalization, 
                   parameter=parameter, max_iters=80)
