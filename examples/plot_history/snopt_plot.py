#!/usr/bin/env python3

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import argparse
import re
import warnings
import os

# Monkeypatch warning so it doesn't print source code
def warning_on_one_line(message, category, filename,
                        lineno, file=None, line=None):
    return '[%s] %s\n' % (category.__name__, message)
warnings.formatwarning = warning_on_one_line

# Plot function
def snopt_plot(filename, savefig):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

    # Read in and parse output file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # metadata line text
    metadata_line = ' Major Minors     Step   nCon Feasible  ' \
                    'Optimal  MeritFunction     nS Penalty\n'

    # Find the line index where history data start
    line_index = 0
    for line in lines:
        if line == metadata_line:
            break
        line_index += 1
    datastart_index = line_index + 1

    # We skip the first row of data since some entries are missing
    datastart_index += 1


    # Find the total number of lines
    dataend_index = len(lines)

    # Parse data
    major    = []  # major iteration count
    step     = []  # step length alpha
    ncon     = []  # number of function calls for current major iteration
    feasible = []  # scaled nonlinear constraint violation
    optimal  = []  # scaled optimality condition
    meritfun = []  # value of augmented lagrangian merit function
    ns       = []  # current number of superbasic variables
    penalty  = []  # 2-norm of penalty parameter vector
    for line_index in range(datastart_index, dataend_index):
        if lines[line_index] == '\n':
            continue
        elif lines[line_index] == metadata_line:
            continue
        elif 'SNOPT' in lines[line_index]:
            break
        else:
            data = lines[line_index]
            intPattern = r'[\+-]?\d+'
            sciPattern = r'[\+-]?\d+\.\d+[eE][\+-]\d+'
            floPattern = r'[\+-]?\d+\.\d+'
            try:
                major.append(re.findall(intPattern, data[0:6])[0])
            except IndexError:
                try:
                    major.append(major[-1])
                except:
                    major.append(0)
                if not savefig:
                    warnings.warn('Missing major iteration number found ' \
                        'in line {:d}'.format(line_index+1), RuntimeWarning)
            try:
                step.append(re.findall(sciPattern, data[13:22])[0])
            except IndexError:
                try:
                    step.append(step[-1])
                except:
                    step.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'Step\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)
            try:
                ncon.append(re.findall(intPattern, data[22:29])[0])
            except IndexError:
                try:
                    ncon.append(ncon[-1])
                except:
                    ncon.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'nCon\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)
            try:
                feasible.append(re.findall(sciPattern, data[29:39])[0])
            except IndexError:
                try:
                    feasible.append(feasible[-1])
                except:
                    feasible.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'Feasible\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)
            try:
                optimal.append(re.findall(sciPattern, data[39:47])[0])
            except IndexError:
                try:
                    optimal.append(optimal[-1])
                except:
                    optimal.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'Optimal\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)
            try:
                meritfun.append(re.findall(sciPattern, data[47:62])[0])
            except IndexError:
                try:
                    meritfun.append(meritfun[-1])
                except:
                    meritfun.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'MeritFun\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)
            try:
                ns.append(re.findall(intPattern, data[62:69])[0])
            except IndexError:
                try:
                    ns.append(ns[-1])
                except:
                    ns.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'nS\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)
            try:
                penalty.append(re.findall(sciPattern, data[69:77])[0])
            except IndexError:
                try:
                    penalty.append(penalty[-1])
                except:
                    penalty.append(0)
                if not savefig:
                    warnings.warn('Missing entry \'Penalty\' found in major ' \
                        'iteration {:d}'.format(int(major[-1])), RuntimeWarning)

    # Store data into numpy arrays
    major    = np.array(major).astype(np.int)
    step     = np.array(step).astype(np.float)
    ncon     = np.array(ncon).astype(np.int)
    feasible = np.array(feasible).astype(np.float)
    optimal  = np.array(optimal).astype(np.float)
    meritfun = np.array(meritfun).astype(np.float)
    ns       = np.array(ns).astype(np.int)
    penalty  = np.array(penalty).astype(np.float)

    # Set up axes and plot objective
    fig, ax1 = plt.subplots()
    l1 = ax1.plot(major, meritfun, color=colors[0], label='merit fun')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Merit function value')

    # Set up another y axis and plot others
    ax2 = ax1.twinx()
    l2 = ax2.semilogy(major, optimal, color=colors[1], label='scaled opt')
    l3 = ax2.semilogy(major, feasible, color=colors[4], label='infeas')
    l4 = ax2.semilogy(major, penalty, color=colors[2], label='penalty')
    l5 = ax2.semilogy(major, step, color=colors[3], label='step len')
    l6 = ax2.semilogy(major, ncon, color=colors[5], label='fun calls per step')
    l7 = ax2.semilogy(major, ns, color=colors[6], label='superbasic vars')
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
    p = argparse.ArgumentParser('Plot values from an SNOPT output file')
    p.add_argument('filename', metavar='SNOPT.out', type=str,
                help='SNOPT summary output file name')
    p.add_argument('--savefig', action='store_true')
    args = p.parse_args()

    # Call plot
    snopt_plot(args.filename, args.savefig)