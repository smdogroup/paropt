import matplotlib
import matplotlib.pylab as plt
import numpy as np
import argparse

# Import ParOpt so that we can read the ParOpt output file
from paropt import ParOpt

p = argparse.ArgumentParser('Plot values from a paropt output file')
p.add_argument('filename', metavar='paropt.out', type=str,
               help='ParOpt output file name')
args = p.parse_args()

# Set font info
font = {'family': 'sans-serif', 'weight': 'normal', 'size': 17}
matplotlib.rc('font', **font)

# Try to unpack values for the interior point code
header, values = ParOpt.unpack_output(args.filename)

if len(values[0]) > 0:

    # You can get more stuff out of this array
    iteration = np.linspace(1, len(values[0]), len(values[0]))
    objective = values[7]
    opt = values[8]
    infeas = values[9]
    barrier = values[11]

    # Just make the iteration linear
    iteration = np.linspace(1, len(iteration), len(iteration))

    # Make the subplots
    fig, ax1 = plt.subplots()
    l1 = ax1.plot(iteration, objective, '-b', linewidth=2, label='objective')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Function value')

    ax2 = ax1.twinx()
    l2 = ax2.semilogy(iteration, opt, '-r', linewidth=2, label='opt')
    l3 = ax2.semilogy(iteration, infeas, '-m', linewidth=2, label='infeas')
    l4 = ax2.semilogy(iteration, barrier, '-g', linewidth=2, label='barrier')
    ax2.set_ylabel('Optimality and Feasibility')

    # Manually add all the lines to the legend
    lns = l1+l2+l3+l4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.title(args.filename)
else:
    # Unpack the output file
    header, values = ParOpt.unpack_tr_output(args.filename)

    if len(values[0]) > 0:
        # You can get more stuff out of this array
        iteration = np.linspace(1, len(values[0]), len(values[0]))
        objective = values[header.index('fobj')]
        # opt_l1 = values[header.index('l1')]
        opt_linfty = values[header.index('linfty')]
        infeas = values[header.index('infes')]
        tr = values[header.index('tr')]
        avg_gamma = values[header.index('avg pen.')]
        avg_z = values[header.index('avg z')]

        # Just make the iteration linear
        iteration = np.linspace(1, len(iteration), len(iteration))

        # Make the subplots
        fig, ax1 = plt.subplots()
        l1 = ax1.plot(iteration, objective, '-b', linewidth=2, label='objective')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function value')

        ax2 = ax1.twinx()
        # l2 = ax2.semilogy(iteration, opt_l1, '-r', linewidth=2, label='opt-l1')
        l2 = ax2.semilogy(iteration, opt_linfty, '-r', linewidth=2, label='opt-linfty')
        l3 = ax2.semilogy(iteration, avg_gamma, '-k', linewidth=2, label='avg. pen.')
        l4 = ax2.semilogy(iteration, avg_z, '-y', linewidth=2, label='avg z')
        l5 = ax2.semilogy(iteration, infeas, '-m', linewidth=2, label='infeas')
        l6 = ax2.semilogy(iteration, tr, '-g', linewidth=2, label='tr')
        ax2.set_ylabel('Optimality and Feasibility')

        # Manually add all the lines to the legend
        lns = l1+l2+l3+l4+l5+l6
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        plt.title(args.filename)

    else:
        # Unpack the output file
        header, values = ParOpt.unpack_mma_output(args.filename)

        # You can get more stuff out of this array
        iteration = np.linspace(1, len(values[0]), len(values[0]))
        objective = values[2]
        lone = values[3]
        linfty = values[4]
        lambd = values[5]

        # Just make the iteration linear
        iteration = np.linspace(1, len(iteration), len(iteration))

        # Make the subplots
        fig, ax1 = plt.subplots()
        l1 = ax1.plot(iteration, objective, '-b', linewidth=2, label='objective')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function value')

        ax2 = ax1.twinx()
        l2 = ax2.semilogy(iteration, lone, '-r', linewidth=2, label='l1-opt')
        l3 = ax2.semilogy(iteration, lambd, '-g', linewidth=2, label='l1-lambda')
        ax2.set_ylabel('Optimality error')

        # Manually add all the lines to the legend
        lns = l1+l2+l3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        plt.title(args.filename)

plt.show()
