import logging, argparse
import pickle
import numpy as np
from ProbGenerate import Problem, Path
from distributedSolver import Learning

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate model through MAP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph_type', default="geant", type=str, help='Graph type',
                        choices=['erdos_renyi', 'balanced_tree', 'hypercube', "cicular_ladder", "cycle", "grid_2d",
                                 'lollipop', 'expander', 'hypercube', 'star', 'barabasi_albert', 'watts_strogatz',
                                 'regular', 'powerlaw_tree', 'small_world', 'geant', 'abilene', 'dtelekom',
                                 'servicenetwork', 'ToyExample2'])
    parser.add_argument('--types', default=3, type=int, help='Number of types')
    parser.add_argument('--learners', default=3, type=int, help='Number of learner')
    parser.add_argument('--sources', default=3, type=int, help='Number of nodes generating data')
    parser.add_argument('--solver', type=str, help='solver type',
                        choices=['DFW', 'FW', 'DPGA', 'PGA', 'DMaxFair', 'MaxFair', 'DMaxTP', 'MaxTP'])
    parser.add_argument('--stepsize', default=0.01, type=float, help="stepsize for FW")

    parser.add_argument('--random_seed', default=19930101, type=int, help='Random seed')
    parser.add_argument('--debug_level', default='INFO', type=str, help='Debug Level',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'])

    args = parser.parse_args()
    np.random.seed(args.random_seed + 2023)
    args.debug_level = eval("logging." + args.debug_level)
    logging.basicConfig(level=args.debug_level)

    fname = 'Result_{}/Result_{}_{}learners_{}sources_{}types_{}stepsize'.format(
        args.solver, args.graph_type, args.learners, args.sources, args.types, args.stepsize)
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    fname = 'Problem_10/Problem_{}_{}learners_{}sources_{}types'.format(
        args.graph_type, args.learners, args.sources, args.types)
    logging.info('Read data from ' + fname)
    with open(fname, 'rb') as f:
        P = pickle.load(f)

    learning = Learning(P)
    mean = P.prior['mean']
    covariance = P.prior['cov']
    learners = P.learners
    T = P.T

    N1 = 50
    N2 = 50
    N3 = 20
    result = results[1]
    dist = 0
    if result == 0:
        distance = 0

    # obj = learning.objU(result, 50, 50)
    for l in learners:
        norm = 0
        for k in range(N3):
            beta = np.random.multivariate_normal(mean[l].reshape(len(mean[l])), covariance[l])
            beta = beta.reshape((len(beta),1))
            noice = P.prior['noice'][l]
            for j in range(N1):
                n = learning.generate_sample1(result, l)
                for i in range(N2):
                    features = learning.generate_sample2(n)
                    a = 0
                    b = 0
                    for s in features:
                        for feature in features[s]:
                            # feature = feature * 0
                            # feature[0: int(np.floor(len(features[s][i]) / 3)), 0] = [10] * int(np.floor(len(feature) / 3))

                            a += np.dot(feature, feature.transpose()) / noice[s]
                            y = np.dot(feature.transpose(), beta) + np.random.normal(0, noice[s])
                            b += feature * y / noice[s]
                    # cov_inv = np.linalg.inv(covariance[l])
                    temp1 = a + np.linalg.inv(covariance[l])
                    temp1 = np.linalg.inv(temp1)
                    temp2 = np.dot(np.linalg.inv(covariance[l]), mean[l]) + b
                    map_l = np.dot(temp1, temp2)

                    # norm_temp = np.linalg.norm(map_l - beta)
                    # if norm_temp < 1:
                    #     print(norm_temp)
                    norm += np.linalg.norm(map_l - beta)
        norm = norm / N1 / N2 / N3
        dist += norm
    distance = dist / len(learners)
    print(distance)
    fname = 'Result_{}/beta_{}_{}learners_{}sources_{}types_{}stepsize'.format(
        args.solver, args.graph_type, args.learners, args.sources, args.types, args.stepsize)
    logging.info('Save in ' + fname)
    with open(fname, 'wb') as f:
        pickle.dump(distance, f)
