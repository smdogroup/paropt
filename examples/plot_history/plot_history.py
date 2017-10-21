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

# Unpack the output file
header, values = ParOpt.unpack_output(args.filename)

# Set font info
font = {'family': 'sans-serif', 'weight': 'normal', 'size': 17}
matplotlib.rc('font', **font)

# You can get more stuff out of this array
iteration = np.linspace(1, len(values[0]), len(values[0]))
objective = values[7]
opt = values[8]
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
l3 = ax2.semilogy(iteration, barrier, '-g', linewidth=2, label='barrier')
ax2.set_ylabel('Optimality error')

# Manually add all the lines to the legend
lns = l1+l2+l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
plt.title(args.filename)
plt.show()
