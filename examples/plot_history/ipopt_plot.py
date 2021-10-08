#!/usr/bin/env python3

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import argparse
import re
import os

def ipopt_plot(filename, savefig):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

    # Read in all lines
    with open(filename, 'r') as f:
        lines = f.readlines()

    # metadata line text
    metadata_line = "iter    objective    inf_pr   inf_du " \
                    "lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls\n"

    # Find the line index where history data start
    line_index = 0
    for line in lines:
        if line == metadata_line:
            break
        line_index += 1
    datastart_index = line_index + 1

    # Find the total number of lines
    dataend_index = len(lines)

    # Parse data
    itern    = []  # iter count, if in restoration phase will have 'r' appended
    obj      = []  # unscaled(original) objective
    inf_pr   = []  # unscaled constraint violation, infinity norm by default
    inf_du   = []  # scaled dual infeasibility, infinity norm
    lg_mu    = []  # log_10 of barrier parameter \mu
    dnorm    = []  # infinity norm of primal step
    alpha_du = []  # stepsize for dual variables
    alpha_pr = []  # stepsize for primal variables
    for line_index in range(datastart_index, dataend_index):
        if lines[line_index] == metadata_line:
            continue
        elif lines[line_index] == '\n':
            break
        else:
            data = lines[line_index].split()
            intPattern = r'[\+-]?\d+'
            sciPattern = r'[\+-]?\d+\.\d+[eE][\+-]\d+'
            floPattern = r'[\+-]?\d+\.\d+'
            itern.append(re.findall(intPattern, data[0].replace('r',''))[0])
            obj.append(re.findall(sciPattern, data[1])[0])
            inf_pr.append(re.findall(sciPattern, data[2])[0])
            inf_du.append(re.findall(sciPattern, data[3])[0])
            lg_mu.append(re.findall(floPattern, data[4])[0])
            dnorm.append(re.findall(sciPattern, data[5])[0])
            alpha_du.append(re.findall(sciPattern, data[7])[0])
            alpha_pr.append(re.findall(sciPattern, data[8])[0])

    # Store data into numpy arrays
    itern    = np.array(itern).astype(np.int)
    obj      = np.array(obj).astype(np.float)
    inf_pr   = np.array(inf_pr).astype(np.float)
    inf_du   = np.array(inf_du).astype(np.float)
    mu       = 10**np.array(lg_mu).astype(np.float)
    dnorm    = np.array(dnorm).astype(np.float)
    alpha_du = np.array(alpha_du).astype(np.float)
    alpha_pr = np.array(alpha_pr).astype(np.float)

    # Set up axes and plot objective
    fig, ax1 = plt.subplots()
    l1 = ax1.plot(itern, obj, color=colors[0], label='objective')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function value')

    # Set up another y axis and plot others
    ax2 = ax1.twinx()
    l2 = ax2.semilogy(itern, inf_du, color=colors[1], label='scaled dual infeas')
    l3 = ax2.semilogy(itern, inf_pr, color=colors[4], label='infeas')
    l4 = ax2.semilogy(itern, mu, color=colors[3], label='barrier')
    l5 = ax2.semilogy(itern, dnorm, color=colors[2], label='linfty primal step')
    l6 = ax2.semilogy(itern, alpha_pr, color=colors[5], label='primal stepsize')
    l7 = ax2.semilogy(itern, alpha_du, color=colors[6], label='dual stepsize')
    ax2.set_ylabel('Optimality and Feasibility')

    # Set labels
    lns = l1 + l2 + l3 + l4 + l5 + l6 + l7
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper right', framealpha=0.2)

    # Plot
    plt.title(filename)
    if (savefig):
        fname = os.path.splitext(filename)[0] # Delete suffix
        fname += '_history'
        plt.savefig(fname+'.png')
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':

    # Set up parser
    p = argparse.ArgumentParser('Plot values from an IPOPT output file')
    p.add_argument('filename', metavar='IPOPT.out', type=str,
                help='IPOPT output file name')
    p.add_argument('--savefig', action='store_true')
    args = p.parse_args()

    # Call plot function
    ipopt_plot(args.filename, args.savefig)