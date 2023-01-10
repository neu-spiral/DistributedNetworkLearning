import matplotlib.pylab as plt
import pickle
import numpy as np
import argparse
import matplotlib
import logging

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

topology_small = ['geant', 'abilene', 'dtelekom']
topology_small_short = ['geant', 'abilene', 'dtelekom']
topology = ['erdos_renyi', 'hypercube', 'star', 'small_world', 'grid_2d']
topology_short = ['ER', 'HC', 'star', 'SW', 'grid']
Dirs = ['FW', 'PGA', 'MaxTP', 'MaxFair']
Stepsizes = [0.01, 0.1, 0.05, 0.05]
Stepsizes_small = [0.01, 0.1, 0.1, 0.01]
algorithm = ['DFW', 'FW', 'DPGA', 'PGA', 'DMaxTP', 'MaxTP', 'DMaxFair', 'MaxFair']
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x, type):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3)
    N = len(topology) + len(topology_small)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
    ax.tick_params(labelsize=12)
    if type == 'Result':
        ylabel = 'Aggregate Utility'
    elif type == 'beta':
        ylabel = 'Avg. Norm of Est. Error'
    ax.set_ylabel(ylabel, fontsize=15)

    ax.set_xticks(ind + width*1.5)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    lgd = fig.legend(labels = algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=int(len(algorithm) / 2), fontsize=13)
    plt.show()
    fig.savefig('Figure/' + type + '.pdf', bbox_extra_artists=(lgd,), bbox_inches = 'tight')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot topology',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug_level', default='DEBUG', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--type', default='Result', type=str, help='Plot est. error or utility', choices=['beta', 'Result'])
    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj = {}
    for alg in algorithm:
        obj[alg] = {}

    if args.type == 'Result':
        for i in range(len(topology)):
            for j in range(len(Dirs)):
                fname = 'Result_{}/Result_{}_{}stepsize'.format(Dirs[j], topology[i], Stepsizes[j])
                result = readresult(fname)
                obj[algorithm[2 * j]][topology_short[i]] = result[0][1]
                obj[algorithm[2 * j + 1]][topology_short[i]] = result[1][1]
    elif args.type == 'beta':
        for i in range(len(topology)):
            for j in range(len(Dirs)):
                fname = 'Result_{}/beta_{}_{}stepsize'.format(Dirs[j], topology[i], Stepsizes[j])
                result = readresult(fname)
                obj[algorithm[2 * j]][topology_short[i]] = result[0]
                obj[algorithm[2 * j + 1]][topology_short[i]] = result[1]

    if args.type == 'Result':
        for i in range(len(topology_small)):
            for j in range(len(Dirs)):
                fname = 'Result_{}/Result_{}_{}stepsize'.format(Dirs[j], topology_small[i], Stepsizes_small[j])
                result = readresult(fname)
                obj[algorithm[2 * j]][topology_small_short[i]] = result[0][1]
                obj[algorithm[2 * j + 1]][topology_small_short[i]] = result[1][1]
    elif args.type == 'beta':
        for i in range(len(topology_small)):
            for j in range(len(Dirs)):
                fname = 'Result_{}/beta_{}_{}stepsize'.format(Dirs[j], topology_small[i], Stepsizes_small[j])
                result = readresult(fname)
                obj[algorithm[2 * j]][topology_small_short[i]] = result[0]
                obj[algorithm[2 * j + 1]][topology_small_short[i]] = result[1]

    barplot(obj, args.type)
