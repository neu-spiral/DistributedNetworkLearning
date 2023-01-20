import matplotlib.pylab as plt
import pickle
import numpy as np
import argparse
import matplotlib
import logging

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

topology1 = ['erdos_renyi', 'geant', 'abilene', 'dtelekom']
topology1_short = ['ER', 'geant', 'abilene', 'dtelekom']
topology2 = ['hypercube', 'star', 'small_world', 'grid_2d']
topology2_short = ['HC', 'star', 'SW', 'grid']
topology3 = ['balanced_tree']
topology3_short = ['BT']
Stepsizes1 = [0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.01]
Stepsizes2 = [0.01, 0.01, 0.01, 0.01, 0.005, 0.01, 0.001, 0.01]
Stepsizes3 = [0.005, 0.01, 0.01, 0.01, 0.005, 0.01, 0.001, 0.01]

algorithm = ['DFW', 'FW', 'DPGA', 'PGA', 'DMaxTP', 'MaxTP', 'DMaxFair', 'MaxFair']
hatches = ['/', '\\\\', '|', '+', '--', '', '////',  'x', 'o', '.', '\\']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def barplot(x, type):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3)
    N = len(topology1) + len(topology2) + len(topology3)
    numb_bars = len(algorithm)+1
    ind = np.arange(0,numb_bars*N ,numb_bars)
    width = 1
    for i in range(len(algorithm)):
        y_ax = x[algorithm[i]].values()
        ax.bar(ind+i*width, y_ax, width=width, hatch=hatches[i], label=algorithm[i])
    ax.tick_params(labelsize=12)
    if type == 'Result':
        ylabel = 'Aggregate Utility'
    elif type == 'Time' or type == 'InFeasibility':
        ylabel = type
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
    parser.add_argument('--type', default='Result', type=str, help='Plot est. error or utility',
                        choices=['beta', 'Result', 'Time', 'InFeasibility'])
    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj = {}
    for alg in algorithm:
        obj[alg] = {}
    #
    # fname = 'Result_PGA/Result_geant_0.1stepsize'
    # result = readresult(fname)

    if args.type == 'Result':
        for i in range(len(topology1)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology1[i], Stepsizes1[j])
                result = readresult(fname)
                obj[algorithm[j]][topology1_short[i]] = result[2]
    elif args.type == 'Time':
        for i in range(len(topology1)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology1[i], Stepsizes1[j])
                result = readresult(fname)
                obj[algorithm[j]][topology1_short[i]] = result[0]
    elif args.type == 'InFeasibility':
        for i in range(len(topology1)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology1[i], Stepsizes1[j])
                result = readresult(fname)
                obj[algorithm[j]][topology1_short[i]] = result[3][-1]
    elif args.type == 'beta':
        for i in range(len(topology1)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/beta_{}_{}stepsize'.format(algorithm[j], topology1[i], Stepsizes1[j])
                result = readresult(fname)
                obj[algorithm[j]][topology1_short[i]] = result

    if args.type == 'Result':
        for i in range(len(topology2)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology2[i], Stepsizes2[j])
                result = readresult(fname)
                obj[algorithm[j]][topology2_short[i]] = result[2]
    elif args.type == 'Time':
        for i in range(len(topology2)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology2[i], Stepsizes2[j])
                result = readresult(fname)
                obj[algorithm[j]][topology2_short[i]] = result
    elif args.type == 'InFeasibility':
        for i in range(len(topology2)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology2[i], Stepsizes2[j])
                result = readresult(fname)
                obj[algorithm[j]][topology2_short[i]] = result[3][-1]
    elif args.type == 'beta':
        for i in range(len(topology2)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/beta_{}_{}stepsize'.format(algorithm[j], topology2[i], Stepsizes2[j])
                result = readresult(fname)
                obj[algorithm[j]][topology2_short[i]] = result

    if args.type == 'Result':
        for i in range(len(topology3)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology3[i], Stepsizes3[j])
                result = readresult(fname)
                obj[algorithm[j]][topology3_short[i]] = result[2]
    elif args.type == 'Time':
        for i in range(len(topology3)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology3[i], Stepsizes3[j])
                result = readresult(fname)
                obj[algorithm[j]][topology3_short[i]] = result
    elif args.type == 'InFeasibility':
        for i in range(len(topology3)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/Result_{}_{}stepsize'.format(algorithm[j], topology3[i], Stepsizes3[j])
                result = readresult(fname)
                obj[algorithm[j]][topology3_short[i]] = result[3][-1]
    elif args.type == 'beta':
        for i in range(len(topology3)):
            for j in range(len(algorithm)):
                fname = 'Result_15_{}/beta_{}_{}stepsize'.format(algorithm[j], topology3[i], Stepsizes3[j])
                result = readresult(fname)
                obj[algorithm[j]][topology3_short[i]] = result

    barplot(obj, args.type)
