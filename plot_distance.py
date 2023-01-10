import matplotlib.pylab as plt
import pickle
import numpy as np
import argparse
import matplotlib
import logging

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run algorithm',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork'])
    parser.add_argument('--stepsize', default=0.01, type=float, help="stepsize")
    parser.add_argument('--solver', type=str, help='solver type',
                        choices=['FW', 'PGA', 'MaxFair', 'MaxTP'])

    parser.add_argument('--debug_level', default='DEBUG', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    fname = 'Result_dist_{}/Result_{}_{}stepsize'.format(args.solver, args.graph_type, args.stepsize)
    result = readresult(fname)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(4, 3)
    ax.set_ylabel("Distance", fontsize=13)
    ax.set_xlabel("Iterations", fontdict=13)


    ax.plot(result)

    fig.savefig(fname + '.pdf', bbox_inches = 'tight')

