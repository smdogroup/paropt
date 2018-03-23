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
parser.add_argument('--merit', type=str, default='compliance',
                    help='Figure of merit for the performance profile')
parser.add_argument('--use_mass_constraint', action='store_true',
                    default=False, help='Use the mass constraint')

args = parser.parse_args()
merit = args.merit
use_mass_constraint = args.use_mass_constraint

root_dir = 'results'

# The heuristics to include in the plot
heuristics = ['paropt_SIMP3', 'paropt_RAMP5',
              'full_snopt_SIMP3', 'full_snopt_RAMP5',
              'full_ipopt_SIMP3', 'full_ipopt_RAMP5']
heur_labels = ['SIMP $p=3$', 'RAMP $q=5$',
               'SNOPT-SIMP $p=3$', 'SNOPT-RAMP $q=5$',
               'IPOPT-SIMP $p=3$', 'IPOPT-RAMP $q=5$']

heuristics = ['paropt_SIMP3', 'paropt_RAMP5',
              'full_snopt_RAMP5',
              'full_ipopt_SIMP3', 'full_ipopt_RAMP5']
heur_labels = ['SIMP $p=3$', 'RAMP $q=5$',
               'SNOPT-RAMP $q=5$',
               'IPOPT-SIMP $p=3$', 'IPOPT-RAMP $q=5$']

# heuristics = ['paropt_RAMP5',
#               'full_snopt_RAMP5',
#               'full_ipopt_RAMP5']
# heur_labels = ['RAMP $q=5$',
#                'SNOPT-RAMP $q=5$',
#                'IPOPT-RAMP $q=5$']

colors = ['BrickRed', 'NavyBlue', 'black', 'ForestGreen',
          'Violet', 'Magenta' ]
symbols = ['circle', 'square', 'triangle', 'delta',
           'circle', 'triangle' ]

# The trusses to include in the plot
trusses = []
for j in range(3, 7):
    for i in range(j, 3*j+1):
        trusses.append((i, j))

# The variable names in the file
variables = ['iteration', 'compliance', 'min d', 'max d', 'tau', 
             'feval', 'geval', 'hvec', 'time']

# Get the index associated with the figure of merit
imerit = variables.index(merit)
infeas_index = variables.index('max d')
fobj_index = variables.index('compliance')
feval_index = variables.index('feval')
geval_index = variables.index('geval')
hvec_index = variables.index('hvec')

# Check the performance 
max_badness = 1e20
perform = max_badness*np.ones((len(trusses), len(heuristics)))

# Count up the number of evaluations
fevals = np.ones((len(trusses), len(heuristics)))
gevals = np.ones((len(trusses), len(heuristics)))
hvecs = np.ones((len(trusses), len(heuristics)))

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
            perform[itruss, iheur] = float(last[imerit])

            # Count the number of evaluations
            fevals[itruss, iheur] = int(last[feval_index])
            gevals[itruss, iheur] = int(last[geval_index])
            hvecs[itruss, iheur] = int(last[hvec_index])

        # Keep track of the number of trusses
        itruss += 1
    
    # Keep track of the heuristics
    iheur += 1

# Compute the ratios for the best performance
nprob = len(trusses)
r = np.zeros(perform.shape)

for i in range(nprob):
    best = 1.0*min(perform[i, :])
    r[i, :] = perform[i, :]/best

if tikz is None:
    # Plot the data
    fig = plt.figure(facecolor='w')
    tau_max = 10.0
    colours = ['g', 'b', 'r', 'k']
    for k in range(len(heuristics)):
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
    xscale = 0.4
    yscale = 1.0

    # Set the bounds on the plot
    xmin = 0.9
    xmax = 5.0
    ymin = 0
    ymax = 1

    # Set the offset for the label
    ylabel_offset = 0.125
    tick_frac = 0.02

    # Set legend parameters
    length = 0.15
    xlegend = 2.0
    
    # Set the positions of the tick locations
    yticks = [0, 0.25, 0.5, 0.75, 1.0]
    xticks = [ 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5 ]

    ylabel_offset = 0.15
    length = 0.15
    ylegend = 0.025 + len(heuristics)*0.06
    xlegend = 3.5

    # Get the header info
    s = tikz.get_header()
    s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

    # Plot the axes
    s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                          tick_frac=tick_frac, ylabel_offset=ylabel_offset,
                          xscale=xscale, yscale=yscale,
                          xticks=xticks, yticks=yticks,
                          xlabel='$\\alpha$', ylabel='Fraction of problems')

    for k in range(len(heuristics)):
        tau, rho = get_performance_profile(r[:, k], 1.5*xmax)
        s += tikz.get_2d_plot(tau, rho, xscale=xscale, yscale=yscale,
                              color=colors[k], line_dim='ultra thick',
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              symbol=None)

    # Add a label to the legend
    xlegend_len = 1.5
    ylegend_len = 0.06*(len(heuristics)-1)
    s += '\\draw[fill=white] (%f,%f) rectangle (%f,%f);'%(
        (xlegend - 0.15)*xscale, (ylegend + 0.04)*yscale,
        (xlegend + xlegend_len)*xscale, (ylegend - ylegend_len - 0.04)*yscale)
    for k in range(len(heuristics)):
        s += tikz.get_legend_entry(xlegend, ylegend - 0.06*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim='ultra thick',
                                   symbol=None, font_size='footnotesize',
                                   label=heur_labels[k])
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

    # Plot the mass infeasibility 
    xscale = 0.1

    # Set the bounds on the plot
    xmin = 0
    xmax = len(trusses)+0.5

    print perform
    
    # Set the positions of the tick locations
    ymin = np.min(perform.flatten())
    ymax = np.max(perform.flatten())
    y1 = int(np.floor(ymin))
    y2 = int(np.ceil(ymax))
    if y2 - y1 <= 1:
        yticks = np.linspace(y1, y2, 11)
    elif y2 - y1 <= 5:
        yticks = np.linspace(y1, y2, 2*int(y2 - y1)+1)
    elif y2 - y1 <= 20:
        yticks = np.linspace(y1, y2, int(y2 - y1)+1)
    else:
        y1 = 5*(y1/5) # Round down to the nearst divisor by 5
        if y2 % 5 != 0:
            y2 = 5*(y2/5 + 1)
        nspan = (y2 - y1)/5
        yticks = np.linspace(y1, y2, nspan+1)                             

    # Reset the max/min y values
    ymin = y1
    ymax = y2
    yscale = 2.0/(ymax - ymin)

    # Set the tick locations along the x axis
    xticks = range(1, len(trusses)+1, 3)
  
    # Get the header info
    s = tikz.get_header()
    s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

    # Plot the axes
    s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                          xscale=xscale, yscale=yscale,
                          xticks=xticks, yticks=yticks,
                          tick_frac=0.01, xlabel_offset=0.1,
                          ylabel_offset=0.075, xlabel='Problem', 
                          ylabel=variables[imerit])
    
    for k in range(len(heuristics)):
        s += tikz.get_2d_plot(range(1, len(trusses)+1), perform[:,k],
                              xscale=xscale, yscale=yscale,
                              color=colors[k], line_dim='ultra thick',
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              symbol=symbols[k], symbol_size=0.025)

    # Add a background for the legend
    length = 1.0
    xlegend = xmax - 8
    ylegend = 0.95*ymax
    xlegend_len = 7.5
    ylegend_len = 0.6*(len(heuristics)-1)
    s += '\\draw[fill=white] (%f,%f) rectangle (%f,%f);'%(
        (xlegend - length)*xscale, (ylegend + 0.6)*yscale,
        (xlegend + xlegend_len)*xscale, (ylegend - ylegend_len - 0.6)*yscale)

    for k in range(len(heuristics)):
        # Add a label to the legend
        s += tikz.get_legend_entry(xlegend, ylegend - 0.7*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim='ultra thick',
                                   symbol=None, font_size='normalsize',
                                   label=heur_labels[k])

    s += tikz.get_end_tikz()

    # Create the tikz/LaTeX file
    filename = 'compare_%s.tex'%(variables[imerit])
    output = os.path.join(root_dir, filename)
    fp = open(output, 'w')
    fp.write(s)
    fp.close()

    # pdflatex the resulting file
    os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(root_dir, 
                                                       filename))

    for heur in heuristics:
        # Create a plot of the max/min discrete infeasibility    
        xscale = 0.1
        yscale = 0.15

        # Set the bounds on the plot
        xmin = 0.5
        xmax = len(trusses)+1.0
        ymin = -0.5
        ymax = 4
    
        # Set the positions of the tick locations
        yticks = range(0, ymax+1)
        yticks = [0, 1, 2, 
                  3, 4 ]
        ytick_labels = ['$1$', '$10$', '$10^{2}$', 
                        '$10^{3}$', '$10^{4}$']
        xticks = range(1, len(trusses)+1, 3)
        
        # Get the header info
        s = tikz.get_header()
        s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

        # Get the data
        iheur = heuristics.index(heur)
        bars = np.log10(np.vstack((fevals[:, iheur], 
                                   gevals[:, iheur], 
                                   hvecs[:, iheur])).T)

        for i in range(5):
            s += tikz.get_2d_plot([xmin, xmax], [i, i],
                                  xscale=xscale, yscale=yscale,
                                  color='gray', line_dim='thin',
                                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            
        # Plot the axes
        s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                              xscale=xscale, yscale=yscale,
                              xticks=xticks, yticks=yticks,
                              ytick_labels=ytick_labels,
                              tick_frac=0.0125,
                              xlabel_offset=0.15,
                              xlabel='Problem', 
                              ylabel_offset=0.065,
                              ylabel='Evaluations')
        
        s += tikz.get_bar_chart(bars, color_list=colors, 
                                xscale=xscale, yscale=yscale, 
                                ymin=ymin, ymax=ymax)

        # Draw the legend
        labels = ['Function evaluations', 'Gradient evaluations', 'Hessian-vector products']
        for i in range(len(labels)):
            s += r'\draw[thick, color=%s, fill=%s, fill opacity=0.3] (%f, %f) rectangle (%f, %f);'%(
                colors[i], colors[i],
                xscale*10, yscale*(5.125 - 0.5*i), xscale*10.5, yscale*(5.125 - 0.5*i) + xscale*0.5)
            
            s += r'\draw[font=\normalsize] (%f, %f) node[right] {%s};'%(
                xscale*10.5, yscale*(5.25 - 0.5*i), labels[i])

        s += tikz.get_end_tikz()

        # Create the tikz/LaTeX file
        filename = '%s_func_evals.tex'%(heur)
        output = os.path.join(root_dir, filename)
        fp = open(output, 'w')
        fp.write(s)
        fp.close()

        # pdflatex the resulting file
        os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(root_dir, 
                                                           filename))
