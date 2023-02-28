import matplotlib.pylab as plt
import pickle
import numpy as np
import argparse
import matplotlib
import logging

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

topology1 = ['geant', 'abilene', 'dtelekom']
topology2 = {'erdos_renyi': 'ER', 'hypercube': 'HC', 'small_world': 'SW', 'grid_2d': 'grid', 'balanced_tree': 'BT',
             'star': 'star'}

Stepsizes1 = [0.1, 0.01, 0.1, 0.01, 0.02, 0.01, 0.1, 0.01]
Stepsizes2 = {'erdos_renyi': [0.07, 0.01, 0.05, 0.01, 0.05, 0.01, 0.3, 0.01],
              'hypercube': [0.05, 0.01, 0.05, 0.01, 0.05, 0.01, 0.1, 0.01],
              'small_world': [0.02, 0.01, 0.01, 0.01, 0.05, 0.01, 0.2, 0.01],
              'grid_2d': [0.02, 0.01, 0.05, 0.01, 0.01, 0.01, 0.1, 0.01],
              'balanced_tree': [0.02, 0.01, 0.05, 0.01, 0.005, 0.01, 0.05, 0.01],
              'star': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01]}


algorithm = ['DFW', 'FW', 'DPGA', 'PGA', 'DMaxTP', 'MaxTP', 'DMaxFair', 'MaxFair']
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']
colors = ['r', 'sandybrown', 'gold', 'yellowgreen', 'mediumturquoise', 'dodgerblue', 'blueviolet', 'hotpink']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x, type):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 2)
    N = len(topology1) + len(topology2)
    numb_bars = len(algorithm)+1
    ind = np.arange(0, numb_bars*N, numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, label=algorithm[i], log=True,
               hatch=hatches[i], color=colors[i], edgecolor='k')
    ax.tick_params(labelsize=12)
    if type == 'Result':
        ylabel = 'Aggregate Utility'
    elif type == 'Time' or type == 'InFeasibility':
        ylabel = type
    elif type == 'beta':
        ylabel = 'Avg. Norm of Est. Error'
    ax.set_ylabel(ylabel, fontsize=13)

    ax.set_xticks(ind + width * (len(algorithm)-1) / 2)
    ax.set_xticklabels(x[algorithm[i]].keys(), fontsize=13)
    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.48, 1.11), ncol=len(algorithm), fontsize=13,
                     handletextpad=0.1, columnspacing=0.6)
    plt.show()
    fig.savefig('Figure/' + type + '.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot topology',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug_level', default='DEBUG', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])
    parser.add_argument('--type', default='Result', type=str, help='Plot est. error or utility',
                        choices=['beta', 'Result', 'Time', 'InFeasibility'])
    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj = {}
    for alg in algorithm:
        obj[alg] = {}

    for i in range(len(topology1)):
        for j in range(len(algorithm)):
            if args.type == 'beta':
                fname = 'Result_{}/beta_{}_3learners_3sources_2types_{}stepsize'.format(
                    algorithm[j], topology1[i], Stepsizes1[j])
                result = readresult(fname)
                obj[algorithm[j]][topology1[i]] = result
            else:
                fname = 'Result_{}/Result_{}_3learners_3sources_2types_{}stepsize'.format(
                    algorithm[j], topology1[i], Stepsizes1[j])
                result = readresult(fname)
                if args.type == 'Result':
                    obj[algorithm[j]][topology1[i]] = result[2]
                elif args.type == 'Time':
                    obj[algorithm[j]][topology1[i]] = result[0]
                elif args.type == 'InFeasibility':
                    obj[algorithm[j]][topology1[i]] = result[3][-1]

    for top in topology2:
        for j in range(len(algorithm)):
            if args.type == 'beta':
                fname = 'Result_{}/beta_{}_5learners_10sources_3types_{}stepsize'.format(
                    algorithm[j], top, Stepsizes2[top][j])
                result = readresult(fname)
                obj[algorithm[j]][topology2[top]] = result
            else:
                fname = 'Result_{}/Result_{}_5learners_10sources_3types_{}stepsize'.format(
                    algorithm[j], top, Stepsizes2[top][j])
                result = readresult(fname)
                if args.type == 'Result':
                    obj[algorithm[j]][topology2[top]] = result[2]
                elif args.type == 'Time':
                    obj[algorithm[j]][topology2[top]] = result[0]
                elif args.type == 'InFeasibility':
                    obj[algorithm[j]][topology2[top]] = result[3][-1]

    barplot(obj, args.type)
