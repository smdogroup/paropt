# Import the os for file related operations
import os

try:
    # Import tikzplots - a utility for create tikz/LaTeX plots
    import tikzplots as tikz
    plt = None
except:
    # Import parts of matplotlib for plotting
    import matplotlib.pyplot as plt
    tikz = None

# Import numpy
import numpy as np

# Import argparse
import argparse

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

# Define the performance profile objective function
parser = argparse.ArgumentParser()
parser.add_argument('--merit', type=str, default='fobj',
                    help='Nodes in x-direction')
parser.add_argument('--use_mass_constraint', action='store_true',
                    default=False, help='Use the mass constraint')

args = parser.parse_args()
merit = args.merit
use_mass_constraint = args.use_mass_constraint

root_dir = 'results'
if use_mass_constraint:
    root_dir = 'con-results'

# The heuristics to include in the plot
heuristics = ['scalar', 'linear', 'discrete']

# The trusses to include in the plot
trusses = [ (3, 3), (4, 3), (5, 3), (6, 3),
            (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4), (10, 4),
            (5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5),
            (6, 6), (7, 6), (8, 6), (9, 6), (10, 6) ]

# The variable names in the file
variables = ['iteration', 'min SE', 'max SE', 'fobj',
             'min gamma', 'max gamma', 'gamma',
             'min d', 'max d', 'tau', 'mass infeas', 
             'feval', 'geval', 'hvec', 'time']

# Get the index associated with the figure of merit
imerit = variables.index(merit)
infeas_index = variables.index('max d')
mass_index = variables.index('mass infeas')

# Check the performance 
max_badness = 1e20
perform = max_badness*np.ones((len(trusses), len(heuristics)))
discrete_infeas = np.ones((len(trusses), len(heuristics)))
mass_infeas = np.ones((len(trusses), len(heuristics)))

# Set the minimum thickness
t_min = 1e-2

# Read in all the data required for each problem
iheur = 0
for heuristic in heuristics:
    itruss = 0
    for N, M in trusses:
        prefix = os.path.join(root_dir, '%dx%d'%(N, M), heuristic)
        log_filename = os.path.join(prefix, 'log_file.dat')

        # Open the file
        if os.path.isfile(log_filename):
            fp = open(log_filename, 'r')
            for last_line in fp: pass

            # Get the list of last names
            last = last_line.split()
            
            if float(last[infeas_index]) < 3*(t_min - t_min**2):
                perform[itruss, iheur] = float(last[imerit])

            # Set the values of the discrete infeasibility
            discrete_infeas[itruss, iheur] = float(last[infeas_index])
            mass_infeas[itruss, iheur] = float(last[mass_index])            

        # Keep track of the number of trusses
        itruss += 1
    
    # Keep track of the heuristics
    iheur += 1

# Compute the ratios for the best performance
nprob = len(trusses)
r = np.zeros(perform.shape)

for i in xrange(nprob):
    best = 1.0*min(perform[i, :])
    r[i, :] = perform[i, :]/best

if tikz is None:
    # Plot the data
    fig = plt.figure(facecolor='w')
    tau_max = 10.0
    colours = ['g', 'b', 'r', 'k']
    for k in xrange(len(heuristics)):
        tau, rho = get_performance_profile(r[:, k], tau_max)
        plt.plot(tau, rho, colours[k], linewidth=2, label=heuristics[k])

    # Finish off the plot and print it
    plt.legend(loc='lower right')
    plt.axis([0.95, tau_max, 0.0, 1.1])
    filename = 'performance_profile_%s.pdf'%(variables[imerit])
    plt.savefig(filename)
    plt.close()
else:
    # Set the scale factors in the x and y directions
    xscale = 0.75
    yscale = 1.0

    # Set the bounds on the plot
    xmin = 0.95
    xmax = 2.5
    ymin = 0
    ymax = 1

    xlegend = 2.0
    
    # Set the positions of the tick locations
    yticks = [0, 0.25, 0.5, 0.75, 1.0]
    xticks = [ 1, 1.5, 2, 2.5 ]
  
    # Get the header info
    s = tikz.get_header()
    s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

    # Plot the axes
    s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                          tick_frac=0.02, ylabel_offset=0.25,
                          xscale=xscale, yscale=yscale,
                          xticks=xticks, yticks=yticks,
                          xlabel='$\\alpha$', ylabel='Fraction of problems')
    
    colors = ['Red', 'NavyBlue', 'black', 'ForestGreen']
    symbols = ['circle', 'square', 'triangle', 'delta' ]

    for k in xrange(len(heuristics)):
        tau, rho = get_performance_profile(r[:, k], 1.5*xmax)
        s += tikz.get_2d_plot(tau, rho, xscale=xscale, yscale=yscale,
                              color=colors[k], line_dim='ultra thick',
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              symbol=None)

        # Add a label to the legend
        length = 0.15
        s += tikz.get_legend_entry(xlegend, 0.4 - 0.1*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim='ultra thick',
                                   symbol=None, label=heuristics[k])
    s += tikz.get_end_tikz()

    # Create the tikz/LaTeX file
    filename = 'performance_profile_%s.pdf'%(variables[imerit])
    output = os.path.join(root_dir, filename)
    fp = open(output, 'w')
    fp.write(s)
    fp.close()

    # pdflatex the resulting file
    os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(root_dir, 
                                                       filename))

    # Create a plot of the max/min discrete infeasibility    
    xscale = 0.1
    yscale = 1.0

    # Set the bounds on the plot
    xmin = 0.5
    xmax = len(trusses)+0.5
    ymin = -2
    ymax = -1 # max(-1, np.max(np.log10(discrete_infeas)))
    
    # Set the positions of the tick locations
    yticks = [-1, np.log10(0.075), np.log10(0.05), np.log10(.025), -2]
    ytick_labels=['$10^{-1}$', '$7.5 \\times$', 
                  '$5 \\times$', '$2.5 \\times$', '$10^{-2}$']
    xticks = range(1, len(trusses)+1, 3)
  
    # Get the header info
    s = tikz.get_header()
    s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

    # Plot the axes
    s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                          xscale=xscale, yscale=yscale,
                          xticks=xticks, yticks=yticks,
                          ytick_labels=ytick_labels, tick_frac=0.1,
                          xlabel='Problem', 
                          ylabel='Discrete infeasibility')
    
    colors = ['Red', 'NavyBlue', 'black', 'ForestGreen']
    symbols = ['circle', 'square', 'triangle', 'delta' ]

    for k in xrange(len(heuristics)):
        s += tikz.get_2d_plot(range(1, len(trusses)+1), 
                              np.log10(discrete_infeas[:, k]), 
                              xscale=xscale, yscale=yscale,
                              color=colors[k], line_dim='ultra thick',
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              symbol=symbols[k], symbol_size=0.025)       

        # Add a label to the legend
        length = 1.
        s += tikz.get_legend_entry(1.5, -1.1 - 0.1*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim='ultra thick',
                                   symbol=None, label=heuristics[k])

    # Keep track of the discrete infeasibility measure
    t_min = 1e-2
    yvals = np.array([ 3*(t_min - t_min**2), 3*(t_min - t_min**2) ])
    xvals = [ 1, len(trusses) ]
    s += tikz.get_2d_plot(xvals, np.log10(yvals), 
                          xscale=xscale, yscale=yscale,
                          color=colors[-1], line_dim='ultra thick',
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    s += tikz.get_end_tikz()

    # Create the tikz/LaTeX file
    filename = 'discrete_infeasibility.tex'
    output = os.path.join(root_dir, filename)
    fp = open(output, 'w')
    fp.write(s)
    fp.close()

    # pdflatex the resulting file
    os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(root_dir, 
                                                       filename))

    # Plot the mass infeasibility 
    xscale = 0.1
    yscale = 5.0

    # Set the bounds on the plot
    xmin = 0.5
    xmax = len(trusses)+0.5
    ymin = 0
    ymax = 0.1
    
    # Set the positions of the tick locations
    yticks = [0, .025, 0.05, 0.075, 0.1 ]
    ytick_labels=['0', '2.5\%', '5\%', '7.5\%', '10\%']
    xticks = range(1, len(trusses)+1, 3)
  
    # Get the header info
    s = tikz.get_header()
    s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

    # Plot the axes
    s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                          xscale=xscale, yscale=yscale,
                          xticks=xticks, yticks=yticks,
                          ytick_labels=ytick_labels, tick_frac=1.0,
                          xlabel_offset=0.15, ylabel_offset=0.125,
                          xlabel='Problem', 
                          ylabel='Mass infeasibility')
    
    colors = ['Red', 'NavyBlue', 'black', 'ForestGreen']
    symbols = ['circle', 'square', 'triangle', 'delta' ]

    for k in xrange(len(heuristics)):
        s += tikz.get_2d_plot(range(1, len(trusses)+1), 
                              mass_infeas[:, k], 
                              xscale=xscale, yscale=yscale,
                              color=colors[k], line_dim='ultra thick',
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              symbol=symbols[k], symbol_size=0.025)

        # Add a label to the legend
        length = 1.
        s += tikz.get_legend_entry(1.5, 0.1 - 0.015*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim='ultra thick',
                                   symbol=None, label=heuristics[k])

    s += tikz.get_end_tikz()

    # Create the tikz/LaTeX file
    filename = 'mass_infeasibility.tex'
    output = os.path.join(root_dir, filename)
    fp = open(output, 'w')
    fp.write(s)
    fp.close()

    # pdflatex the resulting file
    os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(root_dir, 
                                                       filename))

