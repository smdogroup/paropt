from __future__ import print_function

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
    for i in range(N):
        for j in range(M):
            n1 = i + N*j
            for d in grid:
                if ((i + d[0] < N) and (j + d[1] < M) and
                    (i + d[0] >= 0) and (j + d[1] >= 0)):
                    n2 = i + d[0] + (j + d[1])*N
                    conn.append([n1, n2])

    # Set the positions of all the nodes
    xpos = []
    for j in range(M):
        for i in range(N):
            xpos.extend([i*L, j*L])

    # Set the node locations
    loads = {}
    loads[N-1] = [0, -P]

    bcs = {}
    for j in range(M):
        bcs[j*N] = [0, 1]

    return conn, xpos, loads, bcs

def setup_ground_struct(N, M, L=2.5, E=70e9, x_lb=0.0):
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
                          x_lb=x_lb, epsilon=1e-6)

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

def optimize_truss(N, M, root_dir='results', penalization='SIMP',
                   parameter=2.0, max_iters=50,
                   optimizer='paropt', use_hessian=True,
                   start_strategy='point'):
    # Optimize the structure
    heuristic = '%s_%s%.0f'%(optimizer, penalization, parameter)
    prefix = os.path.join(root_dir, '%dx%d'%(N, M), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
   
    # Create the ground structure and optimization
    truss = setup_ground_struct(N, M, x_lb=0.0)
    
    # Set up the optimization problem in ParOpt
    opt = paropt_truss(truss, use_hessian=use_hessian,
                       qn_type=ParOpt.BFGS)
    
    # Log the optimization file
    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, compliance '
    s += '"min d", "max d", "tau",'
    s += 'feval, geval, hvec, time\n'
    s += 'Zone T = %s\n'%(heuristic)
    fp.write(s)

    if start_strategy == 'point':
        # Set the penalty parameter
        truss.SIMP = parameter
        truss.RAMP = parameter
        truss.penalization = penalization
        truss.setNewInitPointPenalty(truss.xinit)
    else:
        truss.SIMP = 1.0
        truss.RAMP = 0.0
        truss.penalization = penalization
        truss.setNewInitPointPenalty(truss.xinit)

    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Previous value of the objective function
    fobj_prev = 0.0

    # Set the first time
    first_time = True

    # Set the initial compliance value
    comp_prev = 0.0
    
    # Keep track of the number of iterations
    niters = 0
    for k in range(max_iters):
        # Set the output file to use
        fname = os.path.join(prefix, 'truss_paropt_iter%d.out'%(k)) 
        opt.setOutputFile(fname)

        # Optimize the truss
        if k > 0:
            opt.setInitStartingPoint(0)
        opt.optimize()

        # Get the optimized point
        x = opt.getOptimizedPoint()

        # Get the discrete infeasibility measure
        d = truss.getDiscreteInfeas(x)

        # Compute the discrete infeasibility measure
        tau = np.sum(d)

        # Get the compliance
        comp = truss.getCompliance(x)

        # Print out the iteration information to the screen
        print('Iteration %d'%(k))
        print('Min/max d:     %15.5e %15.5e  Total: %15.5e'%(
            np.min(d), np.max(d), np.sum(d)))

        s = '%d %e %e %e %e '%(k, comp, np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e\n'%(truss.fevals, truss.gevals, truss.hevals, 
                              MPI.Wtime() - init_time)
        fp.write(s)
        fp.flush()

        # Print the output
        filename = 'opt_truss_iter%d.tex'%(k)
        output = os.path.join(prefix, filename)
        truss.printTruss(x, filename=output)

        truss.SIMP = parameter
        truss.RAMP = parameter
        truss.penalization = penalization
        truss.setNewInitPointPenalty(x)

        if k > 0 and (np.fabs((comp - comp_prev)/comp) < 1e-4):
            break
        
        # Store the previous value of the compliance
        comp_prev = 1.0*comp

        # Increase the iteration counter
        niters += 1

    # Close the log file
    fp.close()
    
    # Print out the last optimized truss
    filename = 'opt_truss.tex'
    output = os.path.join(prefix, filename)
    truss.printTruss(x, filename=output)

    # Save the final optimized point
    fname = os.path.join(prefix, 'x_opt.dat')
    x = opt.getOptimizedPoint()
    np.savetxt(fname, x)

    # Get the rounded design
    xinfty = truss.computeLimitDesign(x)
    fname = os.path.join(prefix, 'x_opt_infty.dat')
    np.savetxt(fname, xinfty)

    return

def create_pyopt(analysis, optimizer='snopt', options={}):
    '''
    Take the given problem and optimize it with the given optimizer
    from the pyOptSparse library of optimizers.
    '''
    # Import the optimization problem
    from pyoptsparse import Optimization, OPT
    from scipy import sparse

    class pyOptWrapper:
        optimizer = None
        options = {}
        opt = None
        prob = None
        def __init__(self, analysis):
            self.xcurr = None
            self.analysis = analysis

        def objcon(self, x):
            # Copy the design variable values
            self.xcurr = np.array(x['x'])
            fail, obj, con = self.analysis.evalObjCon(x['x'])
            funcs = {'objective': obj, 'con': con}
            return funcs, fail

        def gobjcon(self, x, funcs):
            g = np.zeros(x['x'].shape)
            A = np.zeros((1, x['x'].shape[0]))
            fail = self.analysis.evalObjConGradient(x['x'], g, A)
            sens = {'objective': {'x': g}, 'con': {'x': A}}
            return sens, fail

        # Thin wrapper methods to make this look somewhat like ParOpt
        def optimize(self):
            self.opt = OPT(self.optimizer, options=self.options)
            self.sol = self.opt(self.prob, sens=self.gobjcon)
            return

        def setOutputFile(self, fname):
            if self.optimizer == 'snopt':
                self.options['Print file'] = fname
                self.options['Summary file'] = fname + '_summary'
                self.options['Minor feasibility tolerance'] = 1e-10

                # Ensure that we don't stop for iterations
                self.options['Major iterations limit'] = 5000
                self.options['Minor iterations limit'] = 100000000
                self.options['Iterations limit'] = 100000000
            elif self.optimizer == 'ipopt':
                self.options['bound_relax_factor'] = 0.0
                self.options['linear_solver'] = 'ma27'
                self.options['output_file'] = fname
                self.options['max_iter'] = 5000
            return

        def setInitBarrierParameter(self, *args):
            return

        def getOptimizedPoint(self):
            x = np.array(self.xcurr)                         
            return x
        
    # Set the design variables
    wrap = pyOptWrapper(analysis)
    prob = Optimization('topo', wrap.objcon)

    # Record the number of elements/materials/designv ars
    num_materials = analysis.nmats
    num_elements = analysis.nelems
    num_design_vars = analysis.num_design_vars

    # Add the linear constraint
    n = num_design_vars

    # Create the sparse matrix for the design variable weights
    rowp = [0]
    cols = []
    data = []
    nrows = num_elements
    ncols = num_design_vars

    nblock = num_materials+1
    for i in range(num_elements):
        data.append(1.0)
        cols.append(i*nblock)
        for j in range(i*nblock+1, (i+1)*nblock):
            data.append(-1.0)
            cols.append(j)
        rowp.append(len(cols))

    Asparse = {'csr':[rowp, cols, data], 'shape':[nrows, ncols]}

    lower = np.zeros(num_elements)
    upper = np.zeros(num_elements)
    prob.addConGroup('lincon', num_elements,
                     lower=lower, upper=upper,
                     linear=True, wrt=['x'], jac={'x': Asparse})

    # Determine the initial variable values and their lower/upper
    # bounds in the design problem
    x0 = np.zeros(n)
    lb = np.zeros(n)
    ub = np.zeros(n)
    analysis.getVarsAndBounds(x0, lb, ub)
    
    # Set the variable bounds and initial values
    prob.addVarGroup('x', n, value=x0, lower=lb, upper=ub)

    # Set the constraints
    prob.addConGroup('con', 1, lower=0.0, upper=0.0)
        
    # Add the objective
    prob.addObj('objective')

    # Set the values into the wrapper
    wrap.optimizer = optimizer
    wrap.options = options
    wrap.prob = prob

    return wrap

def optimize_truss_full(N, M, root_dir='results', penalization='SIMP',
                        parameter=2.0, optimizer='snopt',
                        start_strategy='point'):
    '''
    Optimize the truss using the full penalization method
    '''
    
    # Optimize the structure
    heuristic = 'full_%s_%s%.0f'%(optimizer, penalization, parameter)
    prefix = os.path.join(root_dir, '%dx%d'%(N, M), heuristic)
    
    # Make sure that the directory exists
    if not os.path.exists(prefix):
        os.makedirs(prefix)
   
    # Create the ground structure and optimization
    truss = setup_ground_struct(N, M, x_lb=0.0)

    # Log the optimization file
    log_filename = os.path.join(prefix, 'log_file.dat')
    fp = open(log_filename, 'w')

    # Write the header out to the file
    s = 'Variables = iteration, compliance '
    s += '"min d", "max d", "tau", '
    s += 'feval, geval, hvec, time\n'
    s += 'Zone T = %s\n'%(heuristic)
    fp.write(s)

    # Keep track of the ellapsed CPU time
    init_time = MPI.Wtime()

    # Set the iteration counter
    iteration = 0

    if start_strategy == 'convex':
        truss.opt_type = 'convex'
        truss.SIMP = 1.0
        truss.RAMP = 0.0
        truss.penalization = penalization
        truss.setNewInitPointPenalty(truss.xinit)
        
        # Create the optimizer
        opt = create_pyopt(truss, optimizer=optimizer)
    
        # Set the output file to use
        fname = os.path.join(prefix, 'truss_%s_iter%d'%(optimizer,
                                                        iteration)) 
        opt.setOutputFile(fname)
        opt.optimize()
        iteration += 1
    
        # Get the optimized point
        x = opt.getOptimizedPoint()

        # Get the discrete infeasibility measure
        d = truss.getDiscreteInfeas(x)
        
        # Compute the discrete infeasibility measure
        tau = np.sum(d)
        
        # Get the compliance
        comp = truss.getCompliance(x)
        
        s = '%d %e %e %e %e '%(0, comp, np.min(d), np.max(d), np.sum(d))
        s += '%d %d %d %e\n'%(truss.fevals, truss.gevals, truss.hevals, 
                              MPI.Wtime() - init_time)
        fp.write(s)
        fp.flush()

        # Print out the last optimized truss
        filename = 'opt_truss.tex'
        output = os.path.join(prefix, filename)
        truss.printTruss(x, filename=output)
        
        # Set the new (feasible) starting point
        truss.setNewInitPointPenalty(x)        
    elif start_strategy == 'uniform':
        truss.xinit[:] = 1.0/truss.nmats
        truss.xinit[::(truss.nmats+1)] = 1.0
        truss.setNewInitPointPenalty(truss.xinit)

    # Set the penalty parameter
    truss.opt_type = 'full'
    truss.SIMP = parameter
    truss.RAMP = parameter
    truss.penalization = penalization
    truss.setNewInitPointPenalty(truss.xinit)

    # Create the optimizer
    opt = create_pyopt(truss, optimizer=optimizer)
    
    # Set the output file to use
    fname = os.path.join(prefix, 'truss_%s_iter%d'%(optimizer,
                                                    iteration)) 
    opt.setOutputFile(fname)
    opt.optimize()

    # Get the optimized point
    x = opt.getOptimizedPoint()

    # Get the discrete infeasibility measure
    d = truss.getDiscreteInfeas(x)

    # Compute the discrete infeasibility measure
    tau = np.sum(d)

    # Get the compliance
    comp = truss.getCompliance(x)

    s = '%d %e %e %e %e '%(0, comp, np.min(d), np.max(d), np.sum(d))
    s += '%d %d %d %e\n'%(truss.fevals, truss.gevals, truss.hevals, 
                          MPI.Wtime() - init_time)
    fp.write(s)
    fp.flush()

    # Close the log file
    fp.close()

    # Print out the last optimized truss
    filename = 'opt_truss.tex'
    output = os.path.join(prefix, filename)
    truss.printTruss(x, filename=output)

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
parser.add_argument('--root_dir', type=str, default='results',
                    help='Root directory to store results')
parser.add_argument('--optimizer', type=str, default='paropt',
                    help='Optimizer to use')
parser.add_argument('--start_strategy', type=str, default='point',
                    help='start up strategy to use')
parser.add_argument('--N', type=int, default=4, 
                    help='Nodes in x-direction')
parser.add_argument('--M', type=int, default=3, 
                    help='Nodes in y-direction')
parser.add_argument('--profile', action='store_true', 
                    default=False, help='Performance profile')
parser.add_argument('--parameter', type=float, default=3.0,
                    help='Penalization parameter')
parser.add_argument('--penalization', type=str, 
                    default='SIMP', help='Penalization type')
args = parser.parse_args()

# Get the arguments
root_dir = args.root_dir
optimizer = args.optimizer
N = args.N
M = args.M
profile = args.profile
penalization = args.penalization
parameter = args.parameter
start_strategy = args.start_strategy

# The trusses used in this instance
trusses = []
for j in range(3, 7):
   for i in range(j, 3*j+1):
       trusses.append((i, j))

# Run either the optimizations or the
if profile:
    for N, M in trusses:
        try:
            if optimizer == 'paropt':
                optimize_truss(N, M, root_dir=root_dir,
                               penalization=penalization, 
                               parameter=parameter, max_iters=80,
                               start_strategy=start_strategy)
            else:
                optimize_truss_full(N, M, root_dir=root_dir,
                                    penalization=penalization, 
                                    parameter=parameter, optimizer=optimizer,
                                    start_strategy=start_strategy)
        except:
            pass
else:
    if optimizer == 'paropt':
        optimize_truss(N, M, root_dir=root_dir,
                       penalization=penalization, 
                       parameter=parameter, max_iters=80,
                       start_strategy=start_strategy)
    else:
        optimize_truss_full(N, M, root_dir=root_dir,
                            penalization=penalization, 
                            parameter=parameter, optimizer=optimizer,
                            start_strategy=start_strategy)
        
