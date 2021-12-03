import matplotlib
import matplotlib.pylab as plt
import numpy as np
import argparse
import os

# This is used for multiple y axis in same plot
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_history(filename, savefig):
    # Import ParOpt so that we can read the ParOpt output file
    from paropt import ParOpt

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
            '#bcbd22', '#17becf']

    # Try to unpack values for the interior point code
    header, values = ParOpt.unpack_output(filename)

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
        l1 = ax1.plot(iteration, objective, color=colors[0], label='objective')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function value')

        ax2 = ax1.twinx()
        l2 = ax2.semilogy(iteration, opt, color=colors[1], label='opt')
        l3 = ax2.semilogy(iteration, infeas, color=colors[2], label='infeas')
        l4 = ax2.semilogy(iteration, barrier, color=colors[3], label='barrier')
        ax2.set_ylabel('Optimality and Feasibility')

        # Manually add all the lines to the legend
        lns = l1+l2+l3+l4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        plt.title(filename)
    else:
        # Unpack the output file
        header, values = ParOpt.unpack_tr_output(filename)

        # Try to unpack and plot secondary tr outputs
        header2, values2 = ParOpt.unpack_tr_2nd_output(filename)
        have_2nd_tr_data = (len(values2[0]) > 0)

        if len(values[0]) > 0:
            # You can get more stuff out of this array
            iteration = np.linspace(1, len(values[0]), len(values[0]))
            objective = values[header.index('fobj')]
            opt_linfty = values[header.index('linfty')]
            infeas = values[header.index('infes')]
            tr = values[header.index('tr')]
            avg_gamma = values[header.index('avg pen.')]
            avg_z = values[header.index('avg z')]

            # Just make the iteration linear
            iteration = np.linspace(1, len(iteration), len(iteration))

            # Make the subplots
            fig, ax1 = plt.subplots()

            # Change fig size if we need to plot the third axis
            if have_2nd_tr_data:
                fig.subplots_adjust(right=0.75)
                fig.set_size_inches(7.9, 4.8)

            l1 = ax1.plot(iteration, objective, color=colors[0], label='objective')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Function value')

            ax2 = ax1.twinx()
            l2 = ax2.semilogy(iteration, opt_linfty, color=colors[1], label='opt-linfty')
            l3 = ax2.semilogy(iteration, avg_gamma, color=colors[2], label='avg. pen.')
            l4 = ax2.semilogy(iteration, avg_z, color=colors[3], label='avg z')
            l5 = ax2.semilogy(iteration, infeas, color=colors[4], label='infeas')
            l6 = ax2.semilogy(iteration, tr, color=colors[5], label='tr')
            ax2.set_ylabel('Optimality and Feasibility')

            if have_2nd_tr_data:
                aredf = values2[header2.index('ared(f)')]
                predf = values2[header2.index('pred(f)')]
                aredc = values2[header2.index('ared(c)')]
                predc = values2[header2.index('pred(c)')]
                rho = values[header.index('rho')]

                # Compute rho for function and constraint
                rhof = aredf/predf
                rhoc = aredc/predc

                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("axes", 1.2))
                make_patch_spines_invisible(ax3)
                ax3.spines["right"].set_visible(True)
                l7 = ax3.plot(iteration, rhof, ':', color=colors[0], label='rho(f)')
                l8 = ax3.plot(iteration, rhoc, ':', color=colors[4], label='rho(c)')
                l9 = ax3.plot(iteration, rho, ':', color=colors[-1], label='rho')
                lns2 = l7 + l8 + l9
                ax3.set_ylabel('Model prediction ratios')
                ax3.set_ylim([-2.0,2.0])

            # Manually add all the lines to the legend
            lns = l1+l2+l3+l4+l5+l6
            if have_2nd_tr_data:
                lns += lns2
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc='upper right')
            plt.title(filename)

        else:
            # Unpack the output file
            header, values = ParOpt.unpack_mma_output(filename)

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
            l1 = ax1.plot(iteration, objective, color=colors[0], label='objective')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Function value')

            ax2 = ax1.twinx()
            l2 = ax2.semilogy(iteration, lone, color=colors[1], label='l1-opt')
            l3 = ax2.semilogy(iteration, lambd, color=colors[2], label='l1-lambda')
            ax2.set_ylabel('Optimality error')

            # Manually add all the lines to the legend
            lns = l1+l2+l3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc=0)
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
    p = argparse.ArgumentParser('Plot values from a paropt output file')
    p.add_argument('filename', metavar='paropt.out', type=str,
                help='ParOpt output file name')
    p.add_argument('--savefig', action='store_true')
    args = p.parse_args()

    # call plot_history
    plot_history(args.filename, args.savefig)
