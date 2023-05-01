import matplotlib.pyplot as plt
import logging, argparse
import pickle
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

coefficients1 = {'learner': [2, 3, 4, 5, 6], 'source': [2, 3, 4, 5, 6], 'rate': [3, 5, 7, 9, 11]}
coefficients2 = {'learner': [3, 5, 7, 9, 11], 'source': [5, 8, 10, 12, 15], 'rate': [3, 5, 7, 9, 11]}

algorithm = ['DFW', 'FW', 'DPGA', 'PGA', 'DMaxTP', 'MaxTP', 'DMaxFair', 'MaxFair']

Stepsizes1 = {'geant': {'learner': [0.1, 0.01, 0.1, 0.01, 0.02, 0.01, 0.02, 0.01],
                        'source': [0.05, 0.01, 0.1, 0.01, 0.02, 0.01, 0.02, 0.01],
                        'rate': [0.05, 0.01, 0.1, 0.01, 0.02, 0.01, 0.02, 0.01]},
              'abilene': {'learner': [0.05, 0.01, 0.005, 0.01, 0.2, 0.01, 0.3, 0.01],
                          'source': [0.05, 0.01, 0.005, 0.01, 0.2, 0.01, 0.3, 0.01],
                          'rate': [0.05, 0.01, 0.005, 0.01, 0.2, 0.01, 0.4, 0.01]},
              'dtelekom': {'learner': [0.01, 0.01, 0.1, 0.01, 0.03, 0.01, 0.1, 0.01],
                           'source': [0.01, 0.01, 0.1, 0.01, 0.03, 0.01, 0.1, 0.01],
                           'rate': [0.01, 0.01, 0.1, 0.01, 0.03, 0.01, 0.1, 0.01]}}

Stepsizes2 = {'erdos_renyi': {'learner': [0.01, 0.01, 0.05, 0.01, 0.03, 0.01, 0.4, 0.01],
                              'source': [0.03, 0.01, 0.05, 0.01, 0.03, 0.01, 0.3, 0.01],
                              'rate': [0.01, 0.01, 0.05, 0.01, 0.03, 0.01, 0.3, 0.01]},
              'hypercube': {'learner': [0.01, 0.01, 0.05, 0.01, 0.05, 0.01, 0.2, 0.01],
                            'source': [0.02, 0.01, 0.05, 0.01, 0.05, 0.01, 0.2, 0.01],
                            'rate': [0.02, 0.01, 0.05, 0.01, 0.05, 0.01, 0.2, 0.01]},
              'small_world': {'learner': [0.01, 0.01, 0.05, 0.01, 0.05, 0.01, 0.2, 0.01],
                              'source': [0.05, 0.01, 0.05, 0.01, 0.05, 0.01, 0.2, 0.01],
                              'rate': [0.05, 0.01, 0.05, 0.01, 0.05, 0.01, 0.2, 0.01]},
              'grid_2d': {'learner': [0.001, 0.01, 0.05, 0.01, 0.01, 0.01, 0.1, 0.01],
                          'source': [0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.1, 0.01],
                          'rate': [0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.1, 0.01]},
              'balanced_tree': {'learner': [0.005, 0.01, 0.01, 0.01, 0.001, 0.01, 0.01, 0.011],
                                'source': [0.005, 0.01, 0.01, 0.01, 0.001, 0.01, 0.01, 0.01],
                                'rate': [0.005, 0.01, 0.01, 0.01, 0.001, 0.01, 0.01, 0.01]},
              'star': {'learner': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01],
                       'source': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01],
                       'rate': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.01]}}

colors = ['r', 'sandybrown', 'gold', 'yellowgreen', 'mediumturquoise', 'dodgerblue', 'blueviolet', 'hotpink']
line_styles = ['s-', '*-', 'd-.', '^-.', 'v:', '.:', '+--', 'x--']


def readresult(fname):
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def plotSensitivity(x1, x2, x3, change, graph):
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(10, 3.2)
    for i in range(len(algorithm)):
        alg = algorithm[i]
        if graph in Stepsizes1:
            ax[0].plot(coefficients1[change], x1[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[1].plot(coefficients1[change], x2[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[2].plot(coefficients1[change], x3[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[0].set_xticks(coefficients1[change])
            ax[1].set_xticks(coefficients1[change])
            ax[2].set_xticks(coefficients1[change])
        elif graph in Stepsizes2:
            ax[0].plot(coefficients2[change], x1[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[1].plot(coefficients2[change], x2[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[2].plot(coefficients2[change], x3[alg], line_styles[i], markersize=10, color=colors[i], label=alg, linewidth=3)
            ax[0].set_xticks(coefficients2[change])
            ax[1].set_xticks(coefficients2[change])
            ax[2].set_xticks(coefficients2[change])

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[2].tick_params(labelsize=12)

    ax[0].set_ylabel('Aggregate Utility', fontsize=15)
    ax[1].set_ylabel('Infeasibility', fontsize=15)
    ax[2].set_ylabel('Estimation Error', fontsize=15)

    if change == 'source':
        xlabel = '|S|'
    elif change == 'learner':
        xlabel = '|L|'
    elif change == 'rate':
        xlabel = '$\lambda_{s,t}$'
    ax[0].set_xlabel(xlabel, fontsize=15)
    ax[1].set_xlabel(xlabel, fontsize=15)
    ax[2].set_xlabel(xlabel, fontsize=15)

    lgd = fig.legend(labels=algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=len(algorithm), fontsize=15,
                     handletextpad=0.1, columnspacing=0.6)
    plt.tight_layout()
    plt.show()
    fig.savefig('Figure_15/sens_{}/{}.pdf'.format(change, graph),  bbox_extra_artists=(lgd,), bbox_inches='tight')
    logging.info('saved in Figure_15/sens_{}/{}.pdf'.format(change, graph))


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
    parser.add_argument('--change', type=str, help='changed variable', choices=['learner', 'source', 'rate'])

    args = parser.parse_args()

    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    obj1, obj2, obj3 = {}, {}, {}
    if args.graph_type in Stepsizes1:
        if args.change == 'learner':
            for j in range(len(algorithm)):
                obj1[algorithm[j]] = []
                obj2[algorithm[j]] = []
                for i in range(len(coefficients1[args.change])):
                    fname = "Result_15_{}/Result_{}_{}learners_3sources_2types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes1[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj1[algorithm[j]].append(result[2])
                    obj2[algorithm[j]].append(result[3][-1])
                obj3[algorithm[j]] = []
                for i in range(len(coefficients1[args.change])):
                    fname = "Result_15_{}/beta_{}_{}learners_3sources_2types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes1[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj3[algorithm[j]].append(result)
        elif args.change == 'source':
            for j in range(len(algorithm)):
                obj1[algorithm[j]] = []
                obj2[algorithm[j]] = []
                for i in range(len(coefficients1[args.change])):
                    fname = "Result_15_{}/Result_{}_3learners_{}sources_2types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes1[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj1[algorithm[j]].append(result[2])
                    obj2[algorithm[j]].append(result[3][-1])
                obj3[algorithm[j]] = []
                for i in range(len(coefficients1[args.change])):
                    fname = "Result_15_{}/beta_{}_3learners_{}sources_2types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes1[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj3[algorithm[j]].append(result)
        elif args.change == 'rate':
            for j in range(len(algorithm)):
                obj1[algorithm[j]] = []
                obj2[algorithm[j]] = []
                for i in range(len(coefficients1[args.change])):
                    fname = "Result_15_rate_{}/Result_{}_3learners_3sources_2types_{}rate_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes1[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj1[algorithm[j]].append(result[2])
                    obj2[algorithm[j]].append(result[3][-1])
                obj3[algorithm[j]] = []
                for i in range(len(coefficients1[args.change])):
                    fname = "Result_15_rate_{}/beta_{}_3learners_3sources_2types_{}rate_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes1[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj3[algorithm[j]].append(result)

    elif args.graph_type in Stepsizes2:
        if args.change == 'learner':
            for j in range(len(algorithm)):
                obj1[algorithm[j]] = []
                obj2[algorithm[j]] = []
                for i in range(len(coefficients2[args.change])):
                    fname = "Result_15_{}/Result_{}_{}learners_10sources_3types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients2[args.change][i], Stepsizes2[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj1[algorithm[j]].append(result[2])
                    obj2[algorithm[j]].append(result[3][-1])
                obj3[algorithm[j]] = []
                for i in range(len(coefficients2[args.change])):
                    fname = "Result_15_{}/beta_{}_{}learners_10sources_3types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients2[args.change][i], Stepsizes2[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj3[algorithm[j]].append(result)
        elif args.change == 'source':
            for j in range(len(algorithm)):
                obj1[algorithm[j]] = []
                obj2[algorithm[j]] = []
                for i in range(len(coefficients2[args.change])):
                    fname = "Result_15_{}/Result_{}_5learners_{}sources_3types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients2[args.change][i], Stepsizes2[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj1[algorithm[j]].append(result[2])
                    obj2[algorithm[j]].append(result[3][-1])
                obj3[algorithm[j]] = []
                for i in range(len(coefficients2[args.change])):
                    fname = "Result_15_{}/beta_{}_5learners_{}sources_3types_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients2[args.change][i], Stepsizes2[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj3[algorithm[j]].append(result)
        elif args.change == 'rate':
            for j in range(len(algorithm)):
                obj1[algorithm[j]] = []
                obj2[algorithm[j]] = []
                for i in range(len(coefficients2[args.change])):
                    fname = "Result_15_rate_{}/Result_{}_5learners_10sources_3types_{}rate_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients2[args.change][i], Stepsizes2[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj1[algorithm[j]].append(result[2])
                    obj2[algorithm[j]].append(result[3][-1])
                obj3[algorithm[j]] = []
                for i in range(len(coefficients2[args.change])):
                    fname = "Result_15_rate_{}/beta_{}_5learners_10sources_3types_{}rate_{}stepsize".format(algorithm[j],
                            args.graph_type, coefficients1[args.change][i], Stepsizes2[args.graph_type][args.change][j])
                    result = readresult(fname)
                    obj3[algorithm[j]].append(result)

    plotSensitivity(obj1, obj2, obj3, args.change, args.graph_type)