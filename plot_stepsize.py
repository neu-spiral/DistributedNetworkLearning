import matplotlib.pyplot as plt
import logging, argparse
import pickle
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

algorithm = ['DFW', 'DPGA', 'DMaxTP', 'DMaxFair']

Stepsizes1 = {'geant': [0.001, 0.005, 0.01, 0.05, 0.1],
              'abilene': [0.001, 0.005, 0.01, 0.05, 0.1, 0.4],
              'dtelekom': [0.001, 0.005, 0.01, 0.05, 0.1, 0.4]}

Stepsizes2 = {'erdos_renyi': [0.001, 0.005, 0.01, 0.03],
              'hypercube': [0.001, 0.005, 0.01, 0.02],
              'small_world': [0.001, 0.005, 0.01, 0.05],
              'grid_2d': [0.001, 0.005, 0.01, 0.02],
              'balanced_tree': [0.001, 0.005, 0.01],
              'star': [0.001, 0.005, 0.01]}

colors = ['r', 'gold', 'mediumturquoise', 'blueviolet']
line_styles = ['s-', '*-', 'd-.', '^-.', 'v:', '.:', '+--', 'x--']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def plotSensitivity(x1, x2, x3, graph):
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(10, 3.2)
    for i in range(len(algorithm)):
        alg = algorithm[i]
        if graph in Stepsizes1:
            ax[0].plot(Stepsizes1[graph], x1[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[1].plot(Stepsizes1[graph], x2[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[2].plot(Stepsizes1[graph], x3[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
        elif graph in Stepsizes2:
            ax[0].plot(Stepsizes2[graph], x1[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[1].plot(Stepsizes2[graph], x2[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[2].plot(Stepsizes2[graph], x3[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[2].tick_params(labelsize=12)

    ax[0].set_ylabel('Aggregate Utility', fontsize=15)
    ax[1].set_ylabel('Infeasibility', fontsize=15)
    ax[2].set_ylabel('Estimation Error', fontsize=15)

    xlabel = 'Stepsize'
    ax[0].set_xlabel(xlabel, fontsize=15)
    ax[1].set_xlabel(xlabel, fontsize=15)
    ax[2].set_xlabel(xlabel, fontsize=15)

    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[2].set_xscale('log')

    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=len(algorithm), fontsize=15,
                     handletextpad=0.1, columnspacing=0.6)
    plt.tight_layout()
    plt.show()
    fig.savefig('Figure_15/sens_stepsize/{}.pdf'.format(graph),  bbox_extra_artists=(lgd,), bbox_inches='tight')
    logging.info('saved in Figure_15/sens_stepsize/{}.pdf'.format(graph))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot sensitivity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--graph_type', default="erdos_renyi", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'ToyExample'])

    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj1, obj2, obj3 = {}, {}, {}  # utility, infeasibility, beta
    if args.graph_type in Stepsizes1:
        for j in range(len(algorithm)):
            obj1[algorithm[j]] = []
            obj2[algorithm[j]] = []
            for Ss in Stepsizes1[args.graph_type]:
                fname = "Result_15_{}/Result_{}_3learners_3sources_2types_{}stepsize".format(algorithm[j],
                        args.graph_type, Ss)
                result = readresult(fname)
                obj1[algorithm[j]].append(result[2])
                obj2[algorithm[j]].append(result[3][-1])
            obj3[algorithm[j]] = []
            for Ss in Stepsizes1[args.graph_type]:
                fname = "Result_15_{}/beta_{}_3learners_3sources_2types_{}stepsize".format(algorithm[j],
                        args.graph_type, Ss)
                result = readresult(fname)
                obj3[algorithm[j]].append(result)

    elif args.graph_type in Stepsizes2:
        for j in range(len(algorithm)):
            obj1[algorithm[j]] = []
            obj2[algorithm[j]] = []
            for Ss in Stepsizes2[args.graph_type]:
                fname = "Result_15_{}/Result_{}_5learners_10sources_3types_{}stepsize".format(algorithm[j],
                        args.graph_type, Ss)
                result = readresult(fname)
                obj1[algorithm[j]].append(result[2])
                obj2[algorithm[j]].append(result[3][-1])
            obj3[algorithm[j]] = []
            for Ss in Stepsizes2[args.graph_type]:
                fname = "Result_15_{}/beta_{}_5learners_10sources_3types_{}stepsize".format(algorithm[j],
                        args.graph_type, Ss)
                result = readresult(fname)
                obj3[algorithm[j]].append(result)

    plotSensitivity(obj1, obj2, obj3, args.graph_type)