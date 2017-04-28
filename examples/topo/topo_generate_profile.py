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
parser.add_argument('--case', type=str, default='isotropic',
                    help='isotropic or orthotropic')
args = parser.parse_args()
merit = args.merit

output_dir = 'results'
root_dir = ['results', 'results',
            'full_penalty_results', 'full_penalty_results']

output_type = 'isotropic'
if args.case == 'isotropic':
    heuristics = ['RAMP5_paropt_isotropic_convex',  
                  'RAMP5_paropt_isotropic_point',            
                  'RAMP5_ipopt_isotropic_point',
                  'RAMP5_snopt_isotropic_point']
    heur_labels = ['ParOpt-cvx isotropic', 'ParOpt-pt isotropic',
                   'IPOPT isotropic', 'SNOPT isotropic']
else:
    output_type = 'orthotropic'
    heuristics = ['RAMP5_paropt_orthotropic_convex',  
                  'RAMP5_paropt_orthotropic_point',            
                  'RAMP5_ipopt_orthotropic_point',
                  'RAMP5_snopt_orthotropic_point']
    heur_labels = ['ParOpt-cvx orthotropic', 'ParOpt-pt orthotropic',
                   'IPOPT orthotropic', 'SNOPT orthotropic']

problems = ['32x32', '64x32', '96x32',
            '48x48', '96x48', '144x48',
            '64x64', '128x64', '192x64',
            '96x96', '192x96', '288x96',
            '128x128', '256x128', '384x128']
xtick_labels = ['$32{\\times}32$', '$64{\\times}32$', '$96{\\times}32$',
                '$48{\\times}48$', '$96{\\times}48$', '$144{\\times}48$',
                '$64{\\times}64$', '$128{\\times}64$', '$192{\\times}64$',
                '$96{\\times}96$', '$192{\\times}96$', '$288{\\times}96$',
                '$128{\\times}128$', '$256{\\times}128$', '$384{\\times}128$']

colors = ['BrickRed', 'NavyBlue', 'black', 'ForestGreen',
          'Violet', 'Magenta', 'Cyan', 'Orange', 'Gray', 'Gray' ]
symbols = ['circle', 'square', 'triangle', 'delta',
           'circle', 'square', 'triangle', 'delta',
           'circle', 'square', 'triangle', 'delta']

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
perform = max_badness*np.ones((len(problems), len(heuristics)))

# Count up the number of evaluations
fevals = np.ones((len(problems), len(heuristics)))
gevals = np.ones((len(problems), len(heuristics)))
hvecs = np.ones((len(problems), len(heuristics)))

# Read in all the data required for each problem
iheur = 0
for k in xrange(len(heuristics)):
    heuristic = heuristics[k]
    itruss = 0
    for prob in problems:
        prefix = os.path.join(root_dir[k], prob, heuristic)
        log_filename = os.path.join(prefix, 'log_file.dat')

        # Open the file
        if os.path.isfile(log_filename):
            fp = open(log_filename, 'r')

            last_line = None
            for last_line in fp: pass

            if last_line is not None:
                # Get the list of last names
                last = last_line.split()
                perform[itruss, iheur] = float(last[imerit])
                
                # Count the number of evaluations
                fevals[itruss, iheur] = int(last[feval_index])
                gevals[itruss, iheur] = int(last[geval_index])
                hvecs[itruss, iheur] = int(last[hvec_index])

        # Keep track of the number of problems
        itruss += 1
    
    # Keep track of the heuristics
    iheur += 1

# Compute the ratios for the best performance
nprob = len(problems)
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

    for k in xrange(len(heuristics)):
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
    for k in xrange(len(heuristics)):
        s += tikz.get_legend_entry(xlegend, ylegend - 0.06*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim='ultra thick',
                                   symbol=None, font_size='footnotesize',
                                   label=heur_labels[k])
    s += tikz.get_end_tikz()

    # Create the tikz/LaTeX file
    filename = 'performance_profile_%s_%s.pdf'%(output_type, variables[imerit])
    output = os.path.join(output_dir, filename)
    fp = open(output, 'w')
    fp.write(s)
    fp.close()

    # pdflatex the resulting file
    os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(output_dir, 
                                                       filename))

    # Plot the mass infeasibility 
    xscale = 0.3

    # Set the bounds on the plot
    xmin = 0.75
    xmax = len(problems)+0.25

    ylabel = variables[imerit]

    if merit == 'time':
        ylabel = 'Time [minutes]'
        perform[:] = perform[:]/60.0
    elif merit == 'compliance':
        ylabel = 'Compliance/Best'

        for i in xrange(len(problems)):
            best_iso = 1e20
            best_ortho = 1e20

            for k in xrange(len(heuristics)):
                if 'iso' in heuristics[k]:
                    if perform[i,k] < best_iso:
                        best_iso = perform[i,k]
                else:
                    if perform[i,k] < best_ortho:
                        best_ortho = perform[i,k]
            for k in xrange(len(heuristics)):
                if 'iso' in heuristics[k]:
                    perform[i,k] /= best_iso
                else:
                    perform[i,k] /= best_ortho
      
    # Set the positions of the tick locations
    ymin = 1e20
    ymax = -1e20
    p = perform.flatten()

    for val in p:
        if val < 1000.0 and val > -1000.0:
            if val < ymin:
                ymin = val
            if val > ymax:
                ymax = val  

    if ymax - ymin > 1.0:
        y1 = int(np.floor(ymin))
        y2 = int(np.ceil(ymax))
        if y2 - y1 <= 5:
            yticks = np.linspace(y1, y2, 2*int(y2 - y1)+1)
        elif y2 - y1 <= 20:
            yticks = np.linspace(y1, y2, int(y2 - y1)+1)
        elif y2 - y1 <= 50:
            y1 = 5*(y1/5) # Round down to the nearst divisor by 5
            if y2 % 5 != 0:
                y2 = 5*(y2/5 + 1)
            nspan = (y2 - y1)/5
            yticks = np.linspace(y1, y2, nspan+1)                             
        elif y2 - y1 <= 100:
            y1 = 10*(y1/10) # Round down to the nearst divisor by 5
            if y2 % 10 != 0:
                y2 = 10*(y2/10 + 1)
            nspan = (y2 - y1)/10
            yticks = np.linspace(y1, y2, nspan+1)                             
        else:
            m = 25
            y1 = m*(y1/m) # Round down to the nearst divisor by 5
            if y2 % m != 0:
                y2 = m*(y2/m + 1)
            nspan = (y2 - y1)/m
            yticks = np.linspace(y1, y2, nspan+1)
    else:
        y1 = np.floor(10*ymin)
        y2 = np.ceil(10*ymax)
        yticks = np.linspace(y1/10.0, y2/10.0, 2*int(y2 - y1)+1)
        y1 /= 10.0
        y2 /= 10.0
        
    # Reset the max/min y values
    ymin = y1
    if merit == 'compliance':
        ymin = 0.98
    
    ymax = y2
    yscale = 1.0/(ymax - ymin)

    # Set the tick locations along the x axis
    xticks = range(1, len(problems)+1, 1)
  
    # Get the header info
    s = tikz.get_header()
    s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

    # Plot the axes
    s += tikz.get_2d_axes(xmin, xmax, ymin, ymax,
                          xscale=xscale, yscale=yscale,
                          xticks=xticks, xtick_labels=xtick_labels, yticks=yticks,
                          tick_font='small',
                          tick_frac=0.015, xlabel_offset=0.1,
                          ylabel_offset=0.06, xlabel='Problem', 
                          ylabel=ylabel)
    for y in yticks:
        s += tikz.get_2d_plot([xmin, xmax], [y, y],
                              xscale=xscale, yscale=yscale,
                              color='gray', line_dim='thin',
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    for k in xrange(len(heuristics)):
        xlist = []
        ylist = []
        for i in range(len(problems)):
            if perform[i,k] < 1e10:
                xlist.append(i+1)
                ylist.append(perform[i,k])

        line_dim = 'ultra thick'
        if 'paropt' not in heuristics[k]:
            line_dim += ', densely dotted'
        s += tikz.get_2d_plot(xlist, ylist,
                              xscale=xscale, yscale=yscale,
                              color=colors[k], line_dim=line_dim,
                              xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                              symbol=symbols[k], symbol_size=0.025)

    # Add a background for the legend
    ylegend = 0.98*ymax
    xlegend = 12.5
    delta_y = 0.075*(ymax - ymin)
    if merit == 'time':
        ylegend = 0.95*ymax
        xlegend = 1.5
        delta_y = 10.0
    
    length = 0.25
    xlegend_len = 2.65
    ylegend_len = delta_y*(len(heuristics)-1)
    s += '\\draw[fill=white] (%f,%f) rectangle (%f,%f);'%(
        (xlegend - length)*xscale, (ylegend + 0.6*delta_y)*yscale,
        (xlegend + xlegend_len)*xscale, (ylegend - ylegend_len - 0.5*delta_y)*yscale)

    for k in xrange(len(heuristics)):
        # Add a label to the legend
        line_dim = 'ultra thick'
        if 'paropt' not in heuristics[k]:
            line_dim += ', densely dotted'
        s += tikz.get_legend_entry(xlegend, ylegend - delta_y*k, length,
                                   xscale=xscale, yscale=yscale,
                                   color=colors[k], line_dim=line_dim,
                                   symbol=symbols[k], font_size='normalsize',
                                   label=heur_labels[k], symbol_size=0.025)

    s += tikz.get_end_tikz()

    # Create the tikz/LaTeX file
    filename = 'compare_%s_%s.tex'%(output_type, variables[imerit])
    output = os.path.join(output_dir, filename)
    fp = open(output, 'w')
    fp.write(s)
    fp.close()

    # pdflatex the resulting file
    os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(output_dir, 
                                                       filename))

    for heur in heuristics:
        # Create a plot of the max/min discrete infeasibility    
        xscale = 0.1
        yscale = 0.15

        # Set the bounds on the plot
        xmin = 0.5
        xmax = len(problems)+1.0
        ymin = -0.5
        ymax = 4
    
        # Set the positions of the tick locations
        yticks = range(0, ymax+1)
        yticks = [0, 1, 2, 
                  3, 4 ]
        ytick_labels = ['$1$', '$10$', '$10^{2}$', 
                        '$10^{3}$', '$10^{4}$']
        xticks = range(1, len(problems)+1, 3)
        
        # Get the header info
        s = tikz.get_header()
        s += tikz.get_begin_tikz(xdim=2, ydim=2, xunit='in', yunit='in')

        # Get the data
        iheur = heuristics.index(heur)
        bars = np.log10(np.vstack((fevals[:, iheur], 
                                   gevals[:, iheur], 
                                   hvecs[:, iheur])).T)

        for i in xrange(5):
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
        for i in xrange(len(labels)):
            s += r'\draw[thick, color=%s, fill=%s, fill opacity=0.3] (%f, %f) rectangle (%f, %f);'%(
                colors[i], colors[i],
                xscale*10, yscale*(5.125 - 0.5*i), xscale*10.5, yscale*(5.125 - 0.5*i) + xscale*0.5)
            
            s += r'\draw[font=\normalsize] (%f, %f) node[right] {%s};'%(
                xscale*10.5, yscale*(5.25 - 0.5*i), labels[i])

        s += tikz.get_end_tikz()

        # Create the tikz/LaTeX file
        filename = '%s_%s_func_evals.tex'%(output_type, heur)
        output = os.path.join(output_dir, filename)
        fp = open(output, 'w')
        fp.write(s)
        fp.close()

        # pdflatex the resulting file
        os.system('cd %s; pdflatex %s > /dev/null; cd ..'%(output_dir, 
                                                           filename))
