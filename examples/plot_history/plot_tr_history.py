import matplotlib
import matplotlib.pylab as plt
import numpy as np
import argparse

# Import ParOpt so that we can read the ParOpt output file
from paropt import ParOpt

p = argparse.ArgumentParser('Plot values from a paropt trust region output file')
p.add_argument('filename', metavar='paropt.out', type=str,
               help='ParOpt output file name')
args = p.parse_args()

# Unpack the output file
header, values = ParOpt.unpack_tr_output(args.filename)

# Set font info
font = {'family': 'sans-serif', 'weight': 'normal', 'size': 17}
matplotlib.rc('font', **font)

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
plt.show()
